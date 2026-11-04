"""Check the overhead camera without the arm: draw where the camera calibration expects the paddle at the hold pose
(through the joint map) on a frame, and report the ball if one is in view.

  uv run python real/camera_check.py                      # live camera, same settings as bounce_run.py
  uv run python real/camera_check.py --video real/logs/run-<time>.mp4 [--frame 30]

Writes real/logs/camera-check-<time>.png. If the outline is off the blade, the camera or the paddle moved, or the
joint map changed: rerun real/calibrate_camera.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rig                                                         # noqa: E402
from bounce.env import EnvCfg                                      # noqa: E402
from bounce.robot import ArmKinematics, build_model                # noqa: E402

HERE = Path(__file__).resolve().parent


def grab(cc):
    cam = rig.Camera(rig.camera_index(cc), cc["width"], cc["height"], cc["fps"], rig.CONFIG.get("camera_latency_s", 0.035))
    try:
        rig.lock_camera(cam, cc)
        t_end = time.perf_counter() + 5.0
        while time.perf_counter() < t_end:
            frame = cam.get()[0]
            if frame is not None:
                return frame.copy()
            time.sleep(0.01)
        raise SystemExit("no frame from the camera")
    finally:
        cam.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=30)
    a = ap.parse_args()
    cc = rig.CONFIG["cameras"]["overhead"]
    if a.video:
        c = cv2.VideoCapture(a.video)
        c.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, frame = c.read()
        if not ok:
            raise SystemExit(f"could not read frame {a.frame} of {a.video}")
    else:
        frame = grab(cc)
    hold = json.loads((HERE / "hold_pose.json").read_text())
    q = torch.tensor(rig.deg_to_sim(hold), dtype=torch.float32)[None]
    cfg = EnvCfg()
    spec = cfg.paddle
    fk = rig.shift_paddle(ArmKinematics(build_model(spec), 1, "cpu")).forward(q)
    o, R = fk["paddle_o"][0].numpy(), fk["paddle_R"][0].numpy()
    model = rig.CameraModel()
    th = np.linspace(0, 2 * np.pi, 72)
    s = 2.0 / cfg.blade_shape[0]
    du = spec.blade_a * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** s
    dv = spec.blade_b * np.sign(np.sin(th)) * np.abs(np.sin(th)) ** s
    outline = np.array([model.project(o + x * R[:, 0] + y * R[:, 1]) for x, y in zip(du, dv)], np.int32)
    cv2.polylines(frame, [outline], True, (0, 255, 0), 2)
    u0, v0 = model.project(o)
    cv2.drawMarker(frame, (int(u0), int(v0)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    roi = model.paddle_roi(o, size=(cc["width"], cc["height"]))
    cv2.rectangle(frame, roi[:2], roi[2:], (255, 160, 0), 1)
    px = model.ball_px
    finder = rig.BallFinder(rig.CONFIG.get("ball_colour", "orange"), **({"min_r": 0.5 * px, "max_r": 2.5 * px} if px else {}))
    finder.roi = roi
    d = finder.find(frame)
    if d is None:
        print("no ball in the search window")
    else:
        rel = 100 * (model.ball_position(*d) - o)
        cv2.circle(frame, (int(d[0]), int(d[1])), int(d[2]), (0, 0, 255), 2)
        print(f"ball at x {rel[0]:+.1f}, y {rel[1]:+.1f}, z {rel[2]:+.1f} cm from the paddle centre "
              f"(resting on the face: z ~ +{100 * rig.BALL_RADIUS:.0f})")
    out = HERE / "logs" / f"camera-check-{time.strftime('%Y%m%d-%H%M%S')}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"paddle centre expected at pixel ({u0:.0f}, {v0:.0f}); image: {out}")


if __name__ == "__main__":
    main()
