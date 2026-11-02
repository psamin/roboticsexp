"""Your two cameras as Dream-1 sees them in a zero-shot run, beside the YAM station's views it was trained on.

  ~/dream-env/bin/python so101/camera_check.py [--flip-scene]   ->  so101/logs/camera-check.jpg

Top row: the YAM station (Dream's simulator, its top camera at the real station's calibrated pose: centered between
the arms, 17 cm behind their bases, 94 cm above the table, 60 deg down; its left arm enters at the bottom left,
its right arm at the bottom right). Bottom row: your top and wrist cameras,
framed exactly as so101/dream_run.py frames them (320x176). Move the top camera until the two top views look alike:
your arm coming in at the bottom corner of its side, the table ahead, the bottle and plate in the middle. Nothing is sent anywhere.
"""

import argparse
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dream_run import FRAME_SIZE, Camera, ZeroShot, model_frame  # noqa: E402


def yam_views(side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """The YAM station's top view and one wrist view (`side`), BGR, at the calibrated top camera pose."""
    from dream_examples.yam_mujoco import yam_env

    yam_env.TOP_CAMERA_LOWER_M = 0.0  # the simulator moves it for a closer view; the station's is the calibrated one
    yam_env.TOP_CAMERA_TILT_UP_DEG = 0.0
    env = yam_env.YamEnv()
    env.reset()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    views = []
    for name in ("top", side):
        renderer.update_scene(env.data, camera=name)
        views.append(cv2.cvtColor(renderer.render()[60:420], cv2.COLOR_RGB2BGR))  # the station's 16:9 crop
    renderer.close()
    env.close()
    return tuple(cv2.resize(v, FRAME_SIZE, interpolation=cv2.INTER_AREA) for v in views)


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flip-scene", action="store_true", help="as dream_run.py --flip-scene")
    parser.add_argument("--side", choices=("right", "left"), default="left", help="as dream_run.py --side")
    parser.add_argument("--front", type=int, default=0)
    parser.add_argument("--wrist", type=int, default=1)
    args = parser.parse_args()

    cameras = [Camera(args.front, cv2.ROTATE_180 if args.flip_scene else None, ZeroShot.front_size),
               Camera(args.wrist, ZeroShot.wrist_rotate)]
    try:
        time.sleep(3.0)  # let the C920 settle
        deadline = time.monotonic() + 10.0  # a camera another program just released can be slow to start
        frames = [camera.latest()[0] for camera in cameras]
        while any(frame is None for frame in frames) and time.monotonic() < deadline:
            time.sleep(0.2)
            frames = [camera.latest()[0] for camera in cameras]
    finally:
        for camera in cameras:
            camera.close()
    if any(frame is None for frame in frames):
        raise SystemExit("a camera gave no frame")
    print("top camera frame:", f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    mine = [model_frame(frame)[1] for frame in frames]  # the 320x176 image the model is sent
    yam_top, yam_wrist = yam_views(args.side)
    grid = np.vstack([
        np.hstack([label(yam_top, "YAM station: top"), label(yam_wrist, f"YAM station: {args.side} wrist")]),
        np.hstack([label(mine[0], "yours: top"), label(mine[1], f"yours: wrist (as the {args.side} wrist)")]),
    ])
    out = pathlib.Path(__file__).parent / "logs" / "camera-check.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
