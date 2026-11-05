"""Dream-1 on the single SO-101 follower: the base model zero-shot, or a fine-tune on this arm's recordings.

  dream_run.py           dry run: the whole loop (cameras, model, chunk timing, speed caps) with the rows logged, not
                         sent. Never writes a servo register, so the arm is not moved, energized or relaxed.
  dream_run.py --live    the same loop driving the arm. Yours to run, with a hand on the arm's power switch.
  --checkpoint dream-1-ft:...   a fine-tune (#FineTuned) instead of the base model (#ZeroShot).

A fine-tune (so101/lerobot_to_dream.py made its dataset) talks in the recordings' own units: LeRobot degrees and the
gripper 0..100, rows one recorded frame (1/30 s) apart, the cameras framed as the conversion framed them (both
center-cropped to 16:9 from the sensor's own 640x480 frame, the wrist not rotated).

How one SO-101 maps onto the base model (the docs' "On another arm"):
- The arm is the right arm, through dream's own SO-101 model (`ik: so101_faas`, Dream Labs' tool center point). The
  left arm is parked at that model's rest pose, gripper open, black wrist frame, and never played.
- Cameras: front (C920, top) is the scene camera, wrist is the right wrist.
- Joints go to the model in radians from the calibrated middle pose, as dream's SO-101 driver reads them, moved by how
  far this arm's middle pose sits from the reference arm's (what `dreamcli setup` would write into the servos; here
  in software only, so LeRobot's calibration and datasets are untouched).

Live safety, on top of the docs' rules:
- Dream's own SO-101 driver runs the servos (robots.motor_arm over robots.feetech): LeRobot's gains, soft limits at the
  calibrated range, a 30 deg/s ramp onto new targets, a hold when no target comes for 0.5 s, a stop on a hot or
  blocked servo, and at the end a glide back to where it was opened (20 deg/s) before the motors switch off.
- This loop caps each joint at --max-speed (default 35 deg/s zero-shot, 90 for a fine-tune, whose demos move that
  fast) and the gripper at 0.05 a row, holds while a camera frame is older than 0.2 s, and ends after --seconds or
  on Ctrl-C (press it once: the arm then glides back). Dream's driver caps tracking at 120 deg/s whatever is asked.
- Start with the arm resting: opening it switches the servos off while they are configured.

Each model call is billed; a run asks about once a second. Logs: so101/logs/dream-<dry|live>-<time>/.
Camera placement for zero-shot: so101/camera_check.py puts your views beside the YAM station's.
"""

import argparse
import base64
import datetime
import json
import logging
import pathlib
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
from loguru import logger as jax_logger

import dream
from dream_ik import specs
from robots import adapter, config, feetech, kinematics, motor_arm, so101, so101_calibration
from robots import types as robot_types

PORT = "/dev/tty.usbmodem5B790811541"  # FOLLOWER_PORT in env.sh
CALIBRATION = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
CHECKPOINT = "dreamlabs/dream-1-beta"
IK_ARM = "so101_faas"
FRAME_SIZE = (320, 176)  # what the model sees (docs: "Each frame")
OBS_HZ = 5  # the client's default obs_fps: its window keeps 9 frames, 0.2 s apart
WARMUP_S = 9 / OBS_HZ  # fill the window before the first call
MAX_FRAME_AGE_S = 0.2  # docs: hold when a camera's newest frame is older
GRIP_STEP = 0.05  # docs: the gripper moves at most this much a row
## Commands stay this far inside each joint's calibrated range: a joint pressed into its end draws stall current, and
## both runs that lost the servo bus (Oct 21) had the arm straining or a joint at its end just before.
JOINT_MARGIN_DEG = 10.0
NAMES = [joint.name for joint in so101.JOINTS]  # base to gripper
DEG_PER_TICK = 360.0 / feetech.TICKS_PER_TURN


class Camera:
    """An OpenCV camera read on its own thread, so latest() is always the newest frame."""

    def __init__(self, index: int, rotate: int | None, size: tuple[int, int] = (640, 480)) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            raise SystemExit(f"camera {index} did not open (index shifted? check: lerobot-find-cameras opencv)")
        got = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != size:
            print(f"camera {index}: asked for {size[0]}x{size[1]}, got {got[0]}x{got[1]}")
        self._rotate = rotate
        self._lock = threading.Lock()
        self._frame, self._at = None, 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, image = self._cap.read()
            if ok:
                if self._rotate is not None:
                    image = cv2.rotate(image, self._rotate)
                with self._lock:
                    self._frame, self._at = image, time.monotonic()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._at

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class HubCamera:
    """The newest frame of one camera from dream-agent's web app (GET <url>/frame/<name>), which owns the cameras so
    its page, its planner and this loop share them. Same interface as Camera; the age comes from the server."""

    def __init__(self, url: str, name: str, rotate: int | None, size: tuple[int, int] | None = None) -> None:
        self._url, self._rotate, self._size, self._warned = f"{url.rstrip('/')}/frame/{name}", rotate, size, False
        self._lock = threading.Lock()
        self._frame, self._at = None, 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                with urllib.request.urlopen(self._url, timeout=1.0) as reply:
                    data, age = reply.read(), float(reply.headers.get("X-Frame-Age", 0))
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if self._size and not self._warned and (image.shape[1], image.shape[0]) != tuple(self._size):
                    print(f"{self._url}: asked for {self._size[0]}x{self._size[1]}, got {image.shape[1]}x{image.shape[0]}")
                    self._warned = True
                if self._rotate is not None:
                    image = cv2.rotate(image, self._rotate)
                with self._lock:
                    self._frame, self._at = image, time.monotonic() - age
            except (OSError, ValueError, cv2.error):
                time.sleep(0.2)
            time.sleep(1 / 15)

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._at

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.5)


class ReadOnlyArm:
    """The arm read straight off its bus. connect, read_calibration and sync_read only read; nothing is written."""

    def __init__(self, port: str, calibration: dict[str, feetech.Calibration]) -> None:
        self._bus = feetech.open_bus(port, so101.JOINTS, calibration)
        self._gripper = calibration["gripper"]
        self._last: tuple[np.ndarray, float] | None = None
        differ = feetech.calibration_differences(calibration, self._bus.read_calibration())
        if differ:
            self.close()
            raise SystemExit(f"the servos hold another calibration than {CALIBRATION.name}: {', '.join(differ)}")

    def read(self) -> tuple[np.ndarray, float]:
        """The five joints in radians from the calibrated middle (dream's SO-101 units), and the gripper 0..1."""
        try:
            raw = self._bus.sync_read("Present_Position", NAMES, normalize=False)
        except ConnectionError:
            if self._last is None:
                raise
            return self._last  # a dropped status packet; dream's driver holds the tick the same way
        joints = np.radians([feetech.ticks_to_deg(raw[name]) for name in NAMES[:-1]])
        grip = (raw["gripper"] - self._gripper.range_min) / (self._gripper.range_max - self._gripper.range_min)
        self._last = (joints, float(np.clip(grip, 0.0, 1.0)))
        return self._last

    def write(self, joints: np.ndarray, grip: float) -> None:
        """Dry run: nothing reaches the servos."""

    def alarm(self) -> str | None:
        return None

    def close(self) -> None:
        self._bus.disconnect(disable_torque=False)  # leave the servos as they were


class LiveArm:
    """The arm under dream's own SO-101 driver: energized at open, holding where it was opened."""

    def __init__(self, port: str, calibration_path: str) -> None:
        open_deg, closed_deg = so101.gripper_ends(calibration_path)
        cfg = config.ArmConfig(
            can_channel=port,
            gripper_open_deg=open_deg,
            gripper_closed_deg=closed_deg,
            calibration=calibration_path,
            home=(0.0,) * 6,  # dream's home move is never used here; the arm parks where it was opened
        )
        # Checks the calibration against the servos, configures them (torque off meanwhile), then holds the pose.
        self._arm = motor_arm.open_arm(so101.ARM, cfg, "right", opener=feetech.open_bus)

    def read(self) -> tuple[np.ndarray, float]:
        state = self._arm.read()
        return np.asarray(state.positions_rad[:5], dtype=np.float64), float(state.gripper)

    def write(self, joints: np.ndarray, grip: float) -> None:
        self._arm.write(robot_types.ArmCommand(positions_rad=np.append(joints, 0.0), gripper=grip))

    def alarm(self) -> str | None:
        return self._arm.alarm()

    def close(self) -> None:
        print("gliding back to the start pose, then motors off")
        self._arm.torque_off()


class Planner:
    """Calls get_action() on its own thread, one call at a time, so the loop never waits on the network (docs)."""

    def __init__(self, client: dream.DreamClient, rows_of) -> None:
        self.client = client
        self.rows_of = rows_of  # the reply as rows in arm units (#ZeroShot.rows, #FineTuned.rows)
        self.thread: threading.Thread | None = None
        self.result = None  # (t_obs, rows, wall_s) from the last reply, or the error it raised
        self.lock = threading.Lock()

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def request(self, commanded: dict[str, np.ndarray]) -> None:
        t_obs = time.monotonic()  # row 0 of the reply is the target for the observation handed in just now

        def call() -> None:
            try:
                action = self.client.get_action(commanded=commanded or None)
                result = (t_obs, self.rows_of(action), time.monotonic() - t_obs)
            except dream.DreamError as error:
                result = error
            with self.lock:
                self.result = result

        self.thread = threading.Thread(target=call, daemon=True)
        self.thread.start()

    def take(self):
        with self.lock:
            result, self.result = self.result, None
        return result

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()


def model_frame(image: np.ndarray) -> tuple[str, np.ndarray]:
    """Center-crop to 16:9, resize to 320x176, JPEG 85, base64 (docs: "Each frame"); also the small image itself."""
    h, w = image.shape[:2]
    if h * 16 > w * 9:  # taller than 16:9: keep the middle rows
        keep = w * 9 // 16
        image = image[(h - keep) // 2 : (h - keep) // 2 + keep]
    else:  # wider: keep the middle columns
        keep = h * 16 // 9
        image = image[:, (w - keep) // 2 : (w - keep) // 2 + keep]
    small = cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok, "JPEG encoding failed"
    return base64.b64encode(jpeg.tobytes()).decode("ascii"), small


def zero_offsets_deg(calibration: dict[str, feetech.Calibration], side: str = "right") -> dict[str, float]:
    """How far each joint's calibrated middle sits from the reference arm's (that side's), by dream's own rule
    (so101_calibration.ArmCalibration.finish): 0 for the wrist roll, the gripper, and a joint whose range is longer
    than the reference's (not the same stops)."""
    reference = so101_calibration.reference(side)
    offsets = {name: 0.0 for name in NAMES}
    for name, ref in reference.items():
        entry = calibration[name]
        span = (entry.range_max - entry.range_min) * DEG_PER_TICK
        off = ((entry.range_min + entry.range_max) / 2 - ref.middle) * DEG_PER_TICK
        if span > ref.span_deg + so101_calibration.LONG_RANGE_DEG:
            continue
        if abs(off) > so101_calibration.MAX_SHIFT_DEG:
            raise SystemExit(f"{name}'s calibrated middle is {off:.0f} deg from the reference arm's: recalibrate")
        offsets[name] = off
    return offsets


class ZeroShot:
    """The Dream-1 base model through dream's own SO-101 model (`ik: so101_faas`), this arm as one of the station's two
    arms (`side`, the right by default); the other is parked at that model's rest pose, gripper open, black wrist frame.

    Arm units in and out: the five joints in radians as dream's driver reads them, the gripper 0..1.
    """

    task = "Put both objects in the bowl with the {side} arm"
    row_hz = 30  # the rate the base model learned (dreamcli plays it at 20, slower)
    max_speed_deg_s = 35.0
    front_size = (1280, 720)  # the C920's own 16:9 mode: a wider view than a 4:3 frame cropped
    wrist_rotate = cv2.ROTATE_90_CLOCKWISE  # upright, fingers at the bottom, as env.sh records it
    blend_rows = 0  # the client joins each chunk onto the last itself

    def __init__(
        self,
        calibration: dict[str, feetech.Calibration],
        checkpoint: str,
        arm: str = IK_ARM,
        floor_m: float = 0.025,
        correct: bool = True,
        side: str = "right",
    ) -> None:
        self.checkpoint = checkpoint
        self.arm = arm
        # Which arm of the two-arm station this SO-101 plays. Oct 21: as the right arm it kept working on the right
        # of the table, away from the bottles on its left; --side left tests whether that is the side's habit.
        self.side, self.other = side, ("left" if side == "right" else "right")
        self.task = self.task.format(side=side)
        # dream's own workspace box (`ik` key `region`, dream_ik.specs.Region): "a table is a lo z". Its frame is the
        # arm model's root, which on so101_faas sits 3 mm above the SO-101's base plate; the solve keeps the whole
        # arm, tool point included, above it and holds the arm rather than cross it. On Oct 21 a run drove the tool
        # point to table height and scraped.
        self.floor_m = floor_m
        if arm.endswith(".json"):
            # An arm file `dreamcli tcp` wrote from the SO-101's URDF, with the tool center point set by hand: the
            # joints are the URDF's, which is what dream's SO-101 driver reads (0 at the calibrated middle pose).
            self.offsets = {name: 0.0 for name in NAMES}
            arm_file = json.loads(pathlib.Path(arm).read_text())
            self.park = np.asarray(arm_file["rest"][:5], dtype=np.float64)
            # Kept in each run's summary: editing the tool point in `dreamcli tcp` rewrites this same file.
            self.tool = {key: arm_file[key] for key in ("flange", "tcp", "approach", "jaw")}
        else:
            self.tool = {"flange": "finger_lf", "tcp": "so101_faas built-in"}
            # --no-correction: the joints as LeRobot's calibration has them.
            self.offsets = zero_offsets_deg(calibration, side) if correct else {name: 0.0 for name in NAMES}
            self.park = np.asarray(specs.FAAS_ARMS[IK_ARM][self.other].rest_pose[:5], dtype=np.float64)
        self.off = np.radians([self.offsets[name] for name in NAMES[:-1]])  # model units = arm units - off
        layout = kinematics.arm_layout(arm)
        if len(layout.joints) != 5:
            raise SystemExit(f"{arm} drives {len(layout.joints)} joints; an SO-101 arm has 5 before its gripper")
        self.embodiment = adapter.zero_shot_embodiment(layout.width, layout.client_gripper)
        self.black, _ = model_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        print("zero offsets vs reference arm (deg):", {n: round(v, 1) for n, v in self.offsets.items() if v})

    def policy(self) -> dict:
        ik = {"arm": self.arm, "head": "eef"}
        if self.floor_m is not None:
            ik["region"] = {self.side: {"lo": [None, None, self.floor_m]}}
        return {"checkpoint": self.checkpoint, "embodiment": self.embodiment, "ik": ik, "action_hz": self.row_hz}

    def observe(self, joints: np.ndarray, grip: float, scene: str, wrist: str, task: str) -> tuple[dict, dict]:
        s, o = self.side, self.other
        frames = {"video.scene": scene, f"video.wrist_{s[0]}": wrist, f"video.wrist_{o[0]}": self.black}
        observation = {
            "prompt": task,
            f"state.{s}_arm": (joints - self.off).astype(np.float32),
            f"state.{s}_grip": np.asarray([grip], dtype=np.float32),
            f"state.{o}_arm": self.park.astype(np.float32),
            f"state.{o}_grip": np.asarray([1.0], dtype=np.float32),
        }
        return frames, observation

    def rows(self, action: dict) -> np.ndarray:
        key = f"q_{self.side[0]}_chunk"
        rows = np.array(action[key], dtype=np.float64)  # [48, 5 joints + grip], model units
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise dream.DreamError(f"{key} is {rows.shape}, expected [rows, 5 joints + grip]")
        rows[:, :5] += self.off
        return rows

    def commanded(self, cmd: np.ndarray) -> dict:
        return {f"q_{self.side[0]}_chunk": cmd - self.off, f"q_{self.other[0]}_chunk": self.park}

    def describe(self) -> dict:
        return {"model": "zero-shot", "side": self.side, "arm": self.arm, "tool": self.tool,
                "offsets_deg": self.offsets, "floor_m": self.floor_m}


class FineTuned:
    """A Dream-1 fine-tune on this arm's recordings, in their units: LeRobot degrees, gripper 0..100, 30 rows a second.

    LeRobot's degrees (`MotorNormMode.DEGREES`) count from the middle of each joint's calibrated range, 4095 ticks a
    turn; dream's driver counts from tick 2048, 4096 a turn. Its gripper 0..1 is LeRobot's 0..100 over 100: both are
    linear in ticks between the calibrated ends, clipped.
    """

    task = "Pick up the medicine bottle and place it on the plate"  # the recordings' task text
    row_hz = 30  # the dataset's fps: rows are one recorded frame apart
    max_speed_deg_s = 90.0  # the demos' joints move up to about this fast (95th percentile)
    front_size = (640, 480)  # as recorded
    wrist_rotate = None  # the sensor's own frame, as so101/lerobot_to_dream.py framed the recordings
    blend_rows = 6  # docs: the client does not smooth one chunk into the next

    def __init__(self, calibration: dict[str, feetech.Calibration], checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.mid = np.array([(calibration[n].range_min + calibration[n].range_max) / 2 for n in NAMES[:-1]])

    def to_dataset(self, joints: np.ndarray) -> np.ndarray:
        ticks = np.degrees(joints) / DEG_PER_TICK + feetech.CENTER_TICK
        return (ticks - self.mid) * 360.0 / (feetech.TICKS_PER_TURN - 1)

    def from_dataset(self, degrees: np.ndarray) -> np.ndarray:
        ticks = degrees * (feetech.TICKS_PER_TURN - 1) / 360.0 + self.mid
        return np.radians((ticks - feetech.CENTER_TICK) * DEG_PER_TICK)

    def policy(self) -> dict:
        # The embodiment the fine-tune was trained on is loaded from the platform and every call checked against it.
        return {"checkpoint": self.checkpoint, "obs_fps": self.row_hz / 6, "action_hz": self.row_hz}

    def observe(self, joints: np.ndarray, grip: float, scene: str, wrist: str, task: str) -> tuple[dict, dict]:
        frames = {"video.overhead": scene, "video.wrist_r": wrist}
        observation = {
            "prompt": task,
            "state.right_arm": self.to_dataset(joints).astype(np.float32),
            "state.right_gripper": np.asarray([grip * 100.0], dtype=np.float32),
        }
        return frames, observation

    def rows(self, action: dict) -> np.ndarray:
        arm = np.asarray(action["per_key"]["action.right_arm"], dtype=np.float64)  # [48, 5], LeRobot degrees
        grip = np.asarray(action["per_key"]["action.right_gripper"], dtype=np.float64).reshape(len(arm))  # 0..100
        if arm.ndim != 2 or arm.shape[1] != 5:
            raise dream.DreamError(f"action.right_arm is {arm.shape}, expected [rows, 5]")
        return np.column_stack([self.from_dataset(arm), np.clip(grip / 100.0, 0.0, 1.0)])

    def commanded(self, cmd: np.ndarray) -> dict:
        return {}  # the client joins no chunks for a fine-tune

    def describe(self) -> dict:
        return {"model": "fine-tune"}


def wait_until_serving(client, model, cameras: list[Camera], joints: np.ndarray, grip: float, task: str) -> None:
    """One call before the run, so a model still starting (`pool_warming`: a new fine-tune can take up to an hour) is
    waited out with the motors off. The episode is reset after, so the run starts a fresh session."""

    def tick() -> None:
        images = [camera.latest()[0] for camera in cameras]
        if all(image is not None for image in images):
            scene, wrist = (model_frame(image)[0] for image in images)
            client.set_observation(*model.observe(joints, grip, scene, wrist, task))

    for _ in range(9):
        tick()
        time.sleep(1.0 / OBS_HZ)
    while True:
        try:
            client.get_action()
            break
        except dream.DreamServerError as error:
            if not error.retryable:
                raise
            wait = error.retry_after_s or 10.0
            print(f"  model not ready ({error.reason}), asking again in {wait:.0f} s")
            time.sleep(wait)
        except (dream.DreamTimeout, dream.DreamProtocolError) as error:
            print(f"  {type(error).__name__}, asking again in 1 s")
            time.sleep(1.0)
        tick()
    client.reset_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="drive the arm (default: dry run, nothing is written)")
    parser.add_argument("--task", help="the prompt (default: the model's, see ZeroShot.task and FineTuned.task)")
    parser.add_argument("--seconds", type=float, default=90.0, help="episode length")
    parser.add_argument("--no-limits", action="store_true",
                        help="zero-shot: no table floor and no margin from the joint ends, as the Oct 21 9:46 PM run "
                             "that reached the bottles ran")
    parser.add_argument("--side", choices=("right", "left"), default="left",
                        help="zero-shot: which arm of Dream's two-arm station this SO-101 plays. Left by default: "
                             "the best run (Oct 21 22:34) held an object as the left arm; as the right arm every run "
                             "headed right")
    parser.add_argument("--no-correction", action="store_true",
                        help="zero-shot: skip the joint-zero correction against dream's reference SO-101")
    parser.add_argument("--floor-cm", type=float,
                        help="zero-shot: lowest the tool point may go, cm above the arm's base plate (dream's ik "
                             "region). Default 2.5, as in the 22:34 run that held an object; with no floor a run "
                             "pressed the gripper into the table for 24 s and the servo power cut out")
    parser.add_argument("--arm-file", help="zero-shot: the arm file `dreamcli tcp` wrote (your measured tool center "
                                           "point) in place of dream's built-in so101_faas model")
    parser.add_argument("--flip-scene", action="store_true",
                        help="zero-shot: turn the top camera's image 180 deg, so an arm seen from the front shows at the "
                             "bottom as on the YAM (not needed once the camera sits behind the arm)")
    parser.add_argument("--max-speed", type=float, help="cap per joint, deg/s (default: 25 zero-shot, 90 fine-tune)")
    parser.add_argument("--front", type=int, default=0, help="OpenCV index of the top camera (env.sh)")
    parser.add_argument("--wrist", type=int, default=1, help="OpenCV index of the wrist camera (env.sh)")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--frames", help="read both cameras from dream-agent's web app at this URL "
                                         "(e.g. http://127.0.0.1:8000) instead of opening them")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("robots").setLevel(logging.INFO)  # dream's driver: open, park, torque off
    jax_logger.remove()  # the IK solver's per-solve INFO lines
    jax_logger.add(sys.stderr, level="WARNING")

    calibration = feetech.read_calibration(str(CALIBRATION), so101.JOINTS)
    if args.checkpoint.startswith("dream-1-ft:"):
        if args.flip_scene or args.arm_file:
            raise SystemExit("--flip-scene and --arm-file are for zero-shot: a fine-tune runs in its recordings' units")
        model = FineTuned(calibration, args.checkpoint)
    else:
        floor_cm = args.floor_cm if args.floor_cm is not None else 2.5
        model = ZeroShot(calibration, args.checkpoint, args.arm_file or IK_ARM,
                         None if args.no_limits else floor_cm / 100.0,
                         correct=not args.no_correction, side=args.side)
    task = args.task or model.task
    max_speed = args.max_speed or model.max_speed_deg_s
    row_s = 1.0 / model.row_hz
    step = np.radians(max_speed) * row_s
    print(f"{model.describe()['model']} {args.checkpoint}: task {task!r}, {args.seconds:.0f} s, "
          f"{model.row_hz} rows/s, joints capped at {max_speed:.0f} deg/s")

    if args.live:
        input("LIVE: arm resting, hand on its power switch, workspace clear? Enter starts, Ctrl-C quits. ")
    out = pathlib.Path(__file__).parent / "logs" / (
        f"dream-{'live' if args.live else 'dry'}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    )
    out.mkdir(parents=True)

    arm: LiveArm | ReadOnlyArm | None = None
    cameras: list[Camera] = []
    client = dream.DreamClient()
    log: dict[str, list] = {"t": [], "joints": [], "grip": [], "cmd": [], "cmd_grip": [], "row": []}
    chunks: list[dict] = []
    stale = 0
    ended = "ran to its time limit"
    try:
        if args.frames:                                  # cameras owned by dream-agent's web app
            cameras = [HubCamera(args.frames, "front", cv2.ROTATE_180 if args.flip_scene else None, model.front_size),
                       HubCamera(args.frames, "wrist", model.wrist_rotate, (640, 480))]
        else:
            cameras = [Camera(args.front, cv2.ROTATE_180 if args.flip_scene else None, model.front_size),
                       Camera(args.wrist, model.wrist_rotate)]
        client.policy(model.policy())
        time.sleep(3.0)  # let the C920 deliver and settle (env.sh warmup_s)
        probe = ReadOnlyArm(args.port, calibration)  # the pose for the warm-up call, motors untouched
        try:
            probe_joints, probe_grip = probe.read()
        finally:
            probe.close()
        print("checking the model is up (a new fine-tune's first start can take up to an hour)")
        wait_until_serving(client, model, cameras, probe_joints, probe_grip, task)

        # Last, so the motors switch on only once the cameras and the model are up.
        arm = LiveArm(args.port, str(CALIBRATION)) if args.live else ReadOnlyArm(args.port, calibration)
        joints, grip = arm.read()
        cmd, cmd_grip = joints.copy(), grip  # the last command, arm units; starts at the measured pose
        # Each joint's allowed band: its calibrated range less the margin, widened to take in where the joint starts,
        # so a joint resting near its end is never jerked inward, only kept from going further toward it.
        margin = np.radians(0.0 if args.no_limits else JOINT_MARGIN_DEG)
        joint_lo = np.minimum(np.radians([feetech.ticks_to_deg(calibration[n].range_min) for n in NAMES[:-1]]) + margin,
                              joints)
        joint_hi = np.maximum(np.radians([feetech.ticks_to_deg(calibration[n].range_max) for n in NAMES[:-1]]) - margin,
                              joints)
        commanded: dict[str, np.ndarray] = {}  # what the client's seam join takes over from (model units)
        planner = Planner(client, model.rows)
        buffer: list[np.ndarray] = []  # rows of the chunk installed last, not yet played
        played = 0  # rows of that chunk played since it took over
        since_request = 0  # rows played since the observation the pending call answers
        retry_at, errors_in_a_row = 0.0, 0
        started = next_tick = time.monotonic()
        print("running; Ctrl-C once stops" + (" and glides the arm back" if args.live else ""))
        while time.monotonic() - started < args.seconds:
            if reason := arm.alarm():
                print(f"arm alarm: {reason}")
                ended = f"arm alarm: {reason}"
                break
            joints, grip = arm.read()
            now = time.monotonic()

            images = []
            for camera in cameras:
                image, at = camera.latest()
                images.append(image if image is not None and now - at <= MAX_FRAME_AGE_S else None)
            fresh = all(image is not None for image in images)
            if fresh:
                (scene, scene_small), (wrist, wrist_small) = model_frame(images[0]), model_frame(images[1])
                client.set_observation(*model.observe(joints, grip, scene, wrist, task))
            else:
                stale += 1

            result = planner.take()
            if isinstance(result, tuple):
                t_obs, rows, wall_s = result
                errors_in_a_row = 0
                # Row k is the target for t_obs + k rows, but the arm only moved along while rows were played: skip
                # those, never more than the clock says are due (dream's ControlLoop._install_chunk). After a hold
                # the chunk resumes where the arm stopped instead of jumping ahead.
                skipped = min(since_request, round((now - t_obs) / row_s))
                if skipped < len(rows):
                    buffer, played = list(rows[skipped:]), 0
                    # Ease from the last command into the new chunk where the client does not (a fine-tune).
                    last = np.append(cmd, cmd_grip)
                    for i in range(min(model.blend_rows, len(buffer))):
                        w = (i + 1) / (model.blend_rows + 1)
                        buffer[i] = (1 - w) * last + w * buffer[i]
                timing = client.last_timing or {}
                chunks.append({"t": round(t_obs - started, 2), "wall_s": round(wall_s, 2),
                               "rtt_ms": round(timing.get("rtt_ms", float("nan"))), "mode": str(client.mode),
                               "skipped_rows": skipped})
                np.save(out / f"chunk{len(chunks) - 1}.npy", rows)
                print(f"  chunk {len(chunks) - 1}: {wall_s:.2f} s round trip, {skipped} rows skipped, {client.mode}")
            elif isinstance(result, dream.DreamSessionExpired):
                buffer = []  # a new session: hold and ask again
            elif isinstance(result, (dream.DreamServerError, dream.DreamTimeout, dream.DreamProtocolError)) and getattr(
                result, "retryable", True
            ):
                errors_in_a_row = errors_in_a_row + 1 if getattr(result, "reason", None) == "error" else 0
                if errors_in_a_row > 3:
                    raise result
                print(f"  waiting ({getattr(result, 'reason', type(result).__name__)})")
                retry_at = now + (getattr(result, "retry_after_s", None) or 1.0)
            elif result is not None:
                raise result  # not retryable: a bad key, a bad observation, out of credit

            ready = fresh and now - started >= WARMUP_S and now >= retry_at and not planner.busy()
            if ready and client.refill_due(played, len(buffer)):
                cv2.imwrite(str(out / f"chunk{len(chunks)}_scene.jpg"), scene_small)
                cv2.imwrite(str(out / f"chunk{len(chunks)}_wrist_r.jpg"), wrist_small)
                planner.request(commanded)  # the row before this tick's: the client adds the one row itself
                since_request = 0

            row = None
            if buffer and fresh:  # otherwise hold: dream's driver keeps the last target
                row = buffer.pop(0)
                cmd = np.clip(cmd + np.clip(row[:5] - cmd, -step, step), joint_lo, joint_hi)
                # Clamped too: the client's smoothing can overshoot a closed (0) or open (1) gripper slightly, and
                # dream's driver refuses a gripper command outside [0, 1].
                cmd_grip = float(np.clip(cmd_grip + np.clip(row[5] - cmd_grip, -GRIP_STEP, GRIP_STEP), 0.0, 1.0))
                arm.write(cmd, cmd_grip)
                commanded = model.commanded(cmd)
                played += 1
                since_request += 1

            for key, value in (("t", now - started), ("joints", joints), ("grip", grip), ("cmd", cmd),
                               ("cmd_grip", cmd_grip), ("row", played if row is not None else -1)):
                log[key].append(value)
            next_tick += row_s
            time.sleep(max(0.0, next_tick - time.monotonic()))
        planner.wait()
    except KeyboardInterrupt:
        ended = "stopped with Ctrl-C"
        print("stopped")
    except BaseException as error:
        ended = f"error: {type(error).__name__}: {str(error)[:200]}"
        raise
    finally:
        for camera in cameras:
            camera.close()
        client.close()
        try:
            if arm is not None:
                arm.close()
        finally:
            np.savez(out / "trajectory.npz", **{k: np.asarray(v) for k, v in log.items()})
            (out / "summary.json").write_text(json.dumps(
                {"checkpoint": args.checkpoint, "task": task, "live": args.live, "max_speed_deg_s": max_speed,
                 "row_hz": model.row_hz, **model.describe(), "seconds_asked": args.seconds,
                 "joint_margin_deg": 0.0 if args.no_limits else JOINT_MARGIN_DEG, "front": args.front,
                 "wrist": args.wrist, "ended": ended, "stale_ticks": stale, "chunks": chunks}, indent=1))
            print(f"{len(chunks)} model calls, {stale} ticks held for stale frames; saved to {out}")


if __name__ == "__main__":
    main()
