"""The arm and the overhead camera behind the calls the planner uses: look() and run_step().

mock  a saved scene image; steps are logged and reported done. No camera, no arm, no Dream credits.
live  the overhead camera; each step runs so101/dream_run.py --live with the step as Dream-1's task, only after the
      user types y. Every step is a billed Dream episode: dream_run.py makes one warm-up call, then about one call a
      second (so101/DREAM_DEBUG_LOG.md: 83 calls in 83 s). An API key cannot read the balance, so both robots
      count estimated calls against a per-session cap.
"""
import base64
import os
import shlex
import subprocess
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SO101 = HERE.parent / "so101"
DREAM_PY = Path(os.environ.get("DREAM_PY", Path.home() / "dream-env/bin/python"))
MOCK_SCENE = SO101 / "logs/dream-dry-20261020-232914/chunk1_scene.jpg"   # bottle, cup, box and plate on the table
FRONT_CAMERA = int(os.environ.get("FRONT_CAMERA", 0))                      # so101/env.sh: front (C920) is index 0


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
    """dream_run.py opens the cameras and the arm itself, so look() releases the camera before any step runs."""
    live = True

    def __init__(self, dream_args="", confirm=input, max_calls=100):
        self.dream_args, self.confirm, self.steps = shlex.split(dream_args), confirm, []
        self.budget = Budget(max_calls)
        a = self.dream_args
        self.side = a[a.index("--side") + 1] if "--side" in a else "left"      # dream_run.py's default arm

    def look(self):
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
        task = step if " arm" in step.lower() else f"{step.rstrip('.')} with the {self.side} arm"   # docs: name the arm
        cmd = [str(DREAM_PY), str(SO101 / "dream_run.py"), "--live", "--task", task, "--seconds", str(seconds),
               *self.dream_args]
        if step_calls(seconds) > self.budget.left:
            return self.budget.take(seconds)
        print(f"\n  next step: {task!r} ({seconds:.0f} s, about {step_calls(seconds)} billed calls, "
              f"{self.budget.left} left this session)\n  {shlex.join(cmd)}")
        if self.confirm("  run it on the arm? [y/N] ").strip().lower() != "y":
            return "skipped: the user did not confirm this step"
        self.budget.take(seconds)
        self.steps.append(task)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=SO101)
        return f"{'finished' if r.returncode == 0 else f'failed (exit {r.returncode})'} after {time.time() - t0:.0f} s"
