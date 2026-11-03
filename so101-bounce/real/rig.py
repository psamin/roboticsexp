"""The physical rig: joint units, the overhead camera, the ball finder, and pixels -> metres in the robot base frame.

Joint angles: LeRobot reports calibrated degrees; the simulator uses radians from the MuJoCo model's zero. LeRobot's own
kinematics feeds its degrees straight into the SO-101 URDF (deg2rad), so the default map is sign +1, offset 0 for
every joint; real/config.yaml can override it (`joint_map`), and `sysid.py set-hold` checks it: with the paddle
levelled by hand, forward kinematics of the measured angles must also say "level".
"""
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text())
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
BALL_RADIUS = 0.02
CAMERA_CALIB = HERE / "camera_calib.json"


# ------------------------------------------------------------------ joints
def _jmap():
    m = CONFIG.get("joint_map") or {}
    return np.array(m.get("sign", [1, 1, 1, 1, 1]), float), np.array(m.get("offset_deg", [0, 0, 0, 0, 0]), float)


def deg_to_sim(pos: dict) -> np.ndarray:
    """LeRobot positions (degrees, dict) -> simulator joint angles (radians, (5,))."""
    sign, off = _jmap()
    return np.deg2rad(sign * (np.array([pos[n] for n in ARM], float) - off))


def sim_to_deg(q) -> dict:
    sign, off = _jmap()
    d = np.rad2deg(np.asarray(q, float)) * sign + off
    return {n: float(v) for n, v in zip(ARM, d)}


# ------------------------------------------------------------------ camera
def camera_index(cc: dict) -> int:
    """OpenCV index of a configured camera, looked up by its name. OpenCV numbers the connected video (and muxed)
    devices sorted by uniqueID, so the numbers shift whenever something is plugged in or out (an iPhone or the
    FaceTime camera can take index 0)."""
    name = cc.get("name")
    if not name or sys.platform != "darwin":
        return cc["index"]
    import AVFoundation as AV
    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + \
        list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    devs.sort(key=lambda d: str(d.uniqueID()))                 # OpenCV's own order (cap_avfoundation_mac.mm)
    names = [str(d.localizedName()) for d in devs]
    if names.count(name) > 1:
        raise SystemExit(f"two cameras are called '{name}'; unplug one")
    if name not in names:
        raise SystemExit(f"camera '{name}' is not connected (the Mac sees: {', '.join(names) or 'no cameras'}). "
                         "Plug it in (check the USB hub) and run again.")
    return names.index(name)


class Camera:
    """Grabs frames on a thread; get() returns the newest (frame, capture time). Capture time = when the frame was
    read minus the measured pipeline latency (config `latency_s`, default 0.035), on time.perf_counter()'s clock."""

    def __init__(self, index: int, width: int, height: int, fps: int, latency_s: float = 0.035):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise SystemExit(f"camera {index} did not open (check `uv run lerobot-find-cameras opencv`)")
        self.latency = latency_s
        self.frame, self.t, self.n = None, 0.0, 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            t = time.perf_counter() - self.latency
            if ok:
                with self.lock:
                    self.frame, self.t, self.n = f, t, self.n + 1

    def get(self):
        with self.lock:
            return self.frame, self.t, self.n

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def lock_camera(cam, cc: dict):
    """Freeze auto exposure, white balance and focus at the values the camera has just settled on, so the ball's
    brightness cannot drift mid-run. OpenCV cannot set these on macOS; uvc-util can (config `uvc: {vid_pid, ...}`;
    optional `max_exposure` in 100 us units, e.g. 40 = 4 ms once the table is well lit). Settings live in camera RAM,
    so this runs every time the camera opens. Skipped with a note if uvc-util or the camera does not answer."""
    u = cc.get("uvc")
    if not u or sys.platform != "darwin":
        return None
    exe = str(Path(u.get("exe", "~/.local/bin/uvc-util")).expanduser())
    base = [exe, "-V", u["vid_pid"]]

    def get(name):
        try:
            out = subprocess.run(base + ["-g", name], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
        last = out.splitlines()[-1] if out else ""
        if last.endswith(("true", "false")):
            return int(last.endswith("true"))
        num = re.findall(r"-?\d+", last)
        return int(num[-1]) if num else None

    # Re-run auto exposure, white balance and focus on every open: the camera keeps the last manual values in its RAM,
    # so without this a later lock would just re-read stale settings (and focus from before the camera was moved).
    subprocess.run(base + ["-s", "auto-exposure-mode=8", "-s", "auto-exposure-priority=0",
                           "-s", "auto-white-balance-temp=1", "-s", "auto-focus=1"], capture_output=True, text=True, timeout=5)
    t_end = time.perf_counter() + 3.0                          # let them settle on live frames
    while time.perf_counter() < t_end:
        cam.get()
        time.sleep(0.05)
    if not u.get("lock", True):                                # config: leave the camera on its own automatic settings
        print("camera on automatic exposure, white balance and focus")
        return None
    settled = {n: get(n) for n in ("exposure-time-abs", "gain", "white-balance-temp", "focus-abs")}
    if settled["exposure-time-abs"] is None:
        print(f"note: uvc-util could not read the camera ({' '.join(base)}); exposure stays automatic")
        return None
    exp = min(settled["exposure-time-abs"], u.get("max_exposure", 10 ** 6))
    gain = settled["gain"]
    if gain is not None and exp < settled["exposure-time-abs"]:           # shorter exposure (less blur): same brightness
        gain = min(255, round(gain * settled["exposure-time-abs"] / exp))
    want = {"auto-exposure-priority": 0, "auto-exposure-mode": 1, "exposure-time-abs": exp, "gain": gain,
            "auto-white-balance-temp": 0, "white-balance-temp": settled["white-balance-temp"], "auto-focus": 0,
            "focus-abs": settled["focus-abs"]}
    if u.get("autofocus"):                                     # config: keep focusing during the run
        want["auto-focus"], want["focus-abs"] = 1, None
    args = base + sum([["-s", f"{k}={v}"] for k, v in want.items() if v is not None], [])
    subprocess.run(args, capture_output=True, text=True, timeout=5)
    back = {k: get(k) for k in ("auto-exposure-mode", "exposure-time-abs", "gain", "auto-focus")}
    print(f"camera locked: exposure {back['exposure-time-abs'] and back['exposure-time-abs'] / 10} ms, gain {back['gain']}, "
          f"autofocus {'on' if back['auto-focus'] else 'off'} "
          f"(manual = {back['auto-exposure-mode'] == 1}; auto had settled at {settled})")
    return back


# ------------------------------------------------------------------ ball finder
BALL_HSV = {"orange": ((3, 120, 120), (22, 255, 255)), "white": ((0, 0, 190), (180, 45, 255))}


class BallFinder:
    """Colour + shape: HSV mask, then the roundest blob of plausible size, searched near the last position first."""

    def __init__(self, colour: str = "orange", min_r: float = 3.0, max_r: float = 40.0):
        lo, hi = CONFIG.get("ball_hsv", {}).get(colour, BALL_HSV[colour])
        self.lo, self.hi = np.array(lo, np.uint8), np.array(hi, np.uint8)
        # a white ball is only somewhat brighter than a lit table or wall: set its brightness floor per frame from the
        # scene (90th percentile + 12), not a fixed 190 that drops the shaded or blurred part of the ball
        self.adaptive = colour == "white" and colour not in CONFIG.get("ball_hsv", {})
        self.min_r, self.max_r = min_r, max_r
        self.last = None

    def find(self, frame):
        """Returns (u, v, radius_px) in full-frame pixels, or None."""
        h, w = frame.shape[:2]
        regions = []
        if self.last is not None:
            u, v, r = self.last
            s = int(max(40, 6 * r))
            regions.append((max(0, int(u) - s), max(0, int(v) - s), min(w, int(u) + s), min(h, int(v) + s)))
        regions.append((0, 0, w, h))
        lo, hi = self.lo, self.hi
        if self.adaptive:
            v = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2HSV)[..., 2]
            lo = np.array((0, 0, int(np.clip(np.percentile(v, 90) + 12, 150, 200))), np.uint8)
            hi = np.array((180, 60, 255), np.uint8)
        for x0, y0, x1, y1 in regions:
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best, best_score = None, 0.0
            for c in cnts:
                (cu, cv_), r = cv2.minEnclosingCircle(c)
                if not (self.min_r <= r <= self.max_r):
                    continue
                fill = cv2.contourArea(c) / (math.pi * r * r + 1e-9)          # 1 for a filled disc
                if fill > 0.5 and fill * r > best_score:                  # the biggest round blob, not a glint
                    best, best_score = (cu + x0, cv_ + y0, r), fill * r
            if best is not None:
                best = self.refine(frame, *best)
                self.last = best
                return best
        self.last = None
        return None

    @staticmethod
    def refine(frame, u, v, r):
        """Centre and radius from the ball's whole area: an Otsu brightness threshold in a window around it (the
        ball is much brighter than the rubber under it), keeping the blob under the centre. The colour mask alone
        misses the shaded side, so its radius wanders by ~20 %."""
        h, w = frame.shape[:2]
        s = int(max(8, 2.5 * r))
        x0, y0, x1, y1 = max(0, int(u) - s), max(0, int(v) - s), min(w, int(u) + s + 1), min(h, int(v) + s + 1)
        g = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        k = lab[min(int(v) - y0, lab.shape[0] - 1), min(int(u) - x0, lab.shape[1] - 1)]
        if k == 0:
            return u, v, r
        rr = math.sqrt(stats[k, cv2.CC_STAT_AREA] / math.pi)
        if not 0.7 * r < rr < 1.5 * r:                      # merged with something else: keep the mask estimate
            return u, v, r
        return float(cent[k][0] + x0), float(cent[k][1] + y0), rr


# ------------------------------------------------------------------ geometry
class CameraModel:
    """Pinhole camera with its pose in the robot base frame (from calibrate_camera.py). With one camera, depth comes
    from the ball's apparent radius: z = f * R / r_px (scaled by the fitted `radius_scale`)."""

    def __init__(self, path: Path = CAMERA_CALIB):
        c = json.loads(path.read_text())
        self.K = np.array(c["K"], float)
        self.R = np.array(c["R_base_from_cam"], float)
        self.t = np.array(c["t_base_from_cam"], float)
        self.radius_scale = float(c.get("radius_scale", 1.0))
        self.ball_px = float(np.median(c["rad"])) if c.get("rad") else None   # ball radius at the calibration poses

    def ball_position(self, u, v, r_px):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        z = 0.5 * (fx + fy) * BALL_RADIUS / (r_px * self.radius_scale)
        p_cam = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
        return self.R @ p_cam + self.t


def default_K(width: int, height: int, hfov_deg: float = 70.4):
    """C920 nominal intrinsics (70.4 deg horizontal field of view at 16:9; refine with a checkerboard if needed)."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
