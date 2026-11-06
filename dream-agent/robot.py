"""The arm and the overhead camera behind the calls the planner uses: look() and run_step().

mock    a saved scene image; steps are logged and reported done. Free.
camera  the live overhead camera; steps are not run (plan only). Free.
dry     the live camera; each step runs Dream-1 through `dreamcli robot run --dry-run` on station/my_robot.yaml:
        real inference, the arm does not move. Billed.
live    `dreamcli robot run` on station/my_robot_live.yaml: Dream-1 moves the arm. Billed.
dry and live steps run only after the user confirms. A step's seconds count from dreamcli's "loop ready" (arm home);
then it gets dreamcli's Ctrl-C (home, torque off). The 2026-11-05 live run made 35 calls in 57 s. An API key cannot
read the balance, so the robots count estimated calls against a per-session cap.
"""
import base64
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SO101 = HERE.parent / "so101"
STATION = HERE / "station"
DREAM_PY = Path(os.environ.get("DREAM_PY", Path.home() / "dream-env/bin/python"))
CHECKPOINT = "dreamlabs/dream-1-beta"
READY = "robots.control: loop ready"          # dreamcli's log line once the arm is home and the loop runs
MOCK_SCENE = SO101 / "logs/dream-dry-20261020-232914/chunk1_scene.jpg"   # bottle, cup, box and plate on the table
FRONT_CAMERA = int(os.environ.get("FRONT_CAMERA", 0))      # overhead C920 (checked 2026-11-05: 1 wrist, 2 laptop)


def step_calls(seconds):
    """Estimated billed calls for one step: the warm-up call plus about one a second."""
    return 1 + int(seconds)


class Budget:
    def __init__(self, max_calls):
        self.left = max_calls

    def take(self, seconds):
        """None if the step fits (and is charged), else the reason it does not."""
        need = step_calls(seconds)
        if need > self.left:
            return f"not run: the Dream call budget for this session is used up ({self.left} left, {need} needed)"
        self.left -= need
        return None


def jpeg_b64(img, width=640):
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)))
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()


class CameraHub:
    """The overhead camera, read continuously on a thread for look() and the page's video. Released while
    dreamcli runs a step (it opens the cameras itself)."""

    def __init__(self, index=FRONT_CAMERA, size=(1280, 720)):
        self.index, self.size, self.frame, self.at, self.paused, self.cap = index, size, None, 0.0, False, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            if self.paused:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                time.sleep(0.1)
                continue
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            ok, img = self.cap.read()
            if ok:
                self.frame, self.at = img, time.time()
            else:
                time.sleep(0.05)

    def latest(self, timeout=5.0):
        t0 = time.time()
        while self.frame is None:
            if self.paused or time.time() - t0 > timeout:
                raise RuntimeError(f"overhead camera {self.index} gave no frame")
            time.sleep(0.05)
        return self.frame

    def pause(self):
        self.paused = True
        while self.cap is not None:
            time.sleep(0.05)

    def resume(self):
        self.frame, self.paused = None, False


class MockRobot:
    live = False

    def __init__(self, scene=MOCK_SCENE, max_calls=100):
        self.scene, self.steps, self.budget = Path(scene), [], Budget(max_calls)

    def look(self):
        img = cv2.imread(str(self.scene))
        if img is None:
            raise FileNotFoundError(self.scene)
        return jpeg_b64(img)

    def run_step(self, step, seconds):
        if (why := self.budget.take(seconds)):
            return why
        self.steps.append(step)
        return (f"done (mock): {step!r}, {seconds:.0f} s. Mock mode: look() keeps returning the same photo, so "
                "treat the step as having worked.")


class LiveRobot:
    """mode camera / dry / live (see the top of this file)."""
    live = True

    def __init__(self, dream_args="", confirm=input, max_calls=100, mode="live", hub=None):
        """dream_args: extra `dreamcli robot run` flags."""
        self.dream_args, self.confirm, self.steps = shlex.split(dream_args), confirm, []
        self.budget, self.mode, self.hub = Budget(max_calls), mode, hub
        self.proc, self.interrupted = None, None              # the running dreamcli, and who sent its one Ctrl-C

    def _interrupt(self, who):
        """dreamcli's Ctrl-C, once per step (a second one could cut its homing short)."""
        if self.proc is not None and self.proc.poll() is None and self.interrupted is None:
            self.interrupted = who
            self.proc.send_signal(signal.SIGINT)
            return True
        return False

    def stop(self):
        """Ctrl-C to the running step: dreamcli homes the arm and switches it off."""
        return self._interrupt("user")

    def look(self):
        if self.hub is not None:
            return jpeg_b64(self.hub.latest())
        cap = cv2.VideoCapture(FRONT_CAMERA)
        try:
            for _ in range(10):                                  # let auto exposure settle
                ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"overhead camera {FRONT_CAMERA} gave no frame")
            return jpeg_b64(img)
        finally:
            cap.release()

    def run_step(self, step, seconds):
        if self.mode == "camera":
            return (f"not run (camera-only mode: no Dream, no arm): {step!r}. The photo will not change; treat the "
                    "step as having worked.")
        task = step if " arm" in step.lower() else f"{step.rstrip('.')} with the right arm"   # docs: name the arm
        config = STATION / ("my_robot_live.yaml" if self.mode == "live" else "my_robot.yaml")
        cmd = [str(DREAM_PY), str(DREAM_PY.parent / "dreamcli"), "robot", "run", "--config", str(config),
               "--checkpoint", CHECKPOINT, "--task", task, *(["--dry-run"] if self.mode != "live" else []),
               *self.dream_args]
        if step_calls(seconds) > self.budget.left:
            return self.budget.take(seconds)
        what = ("run it on the arm (arm resting, hand on its power switch, workspace clear)" if self.mode == "live"
                else "run Dream-1 on it (the arm does not move)")
        print(f"\n  next step: {task!r} ({seconds:.0f} s, about {step_calls(seconds)} billed calls, "
              f"{self.budget.left} left this session)\n  {shlex.join(cmd)}")
        if self.confirm(f"  {what}? [y/N] ").strip().lower() != "y":
            return "skipped: the user did not confirm this step"
        self.budget.take(seconds)
        self.steps.append(task)
        t0 = time.time()
        if self.hub is not None:
            self.hub.pause()                                    # dreamcli opens both cameras itself
        log = HERE / "logs" / f"step-{time.strftime('%Y%m%d-%H%M%S')}-{self.mode}.log"
        log.parent.mkdir(exist_ok=True)
        try:
            with open(log, "w") as f:
                self.interrupted = None
                self.proc = r = subprocess.Popen(cmd, cwd=STATION, stdout=f, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, text=True)
                end = None
                while r.poll() is None:
                    if end is None and READY in log.read_text():
                        end = time.time() + seconds             # the step's time counts from the loop's start
                    if end is not None and time.time() > end:
                        self._interrupt("timer")
                    time.sleep(0.2)
                r.wait()
                self.proc = None
        finally:
            if self.hub is not None:
                self.hub.resume()
        done = {"timer": "finished", "user": "stopped by the user"}.get(
            self.interrupted, "finished" if r.returncode == 0 else f"failed (exit {r.returncode})")
        tail = " | ".join(l.strip() for l in log.read_text().splitlines()[-3:] if l.strip())
        return (f"{done} after {time.time() - t0:.0f} s" + ("" if self.mode == "live" else " (dry run: the arm did "
                "not move)") + f". dreamcli: {tail[-300:]} (log {log.name})")
