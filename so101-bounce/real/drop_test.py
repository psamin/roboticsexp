"""How bouncy is this paddle? The arm holds the bounce pose still; drop the ball onto the paddle centre from 20-30 cm a
few times and let it bounce out. Each flight between impacts is fitted with a gravity parabola, which gives the ball's
speed at the paddle; restitution e = (speed leaving an impact) / (speed arriving at it). The training range for
e is 0.55-0.92, so this says where this paddle sits in it.

  uv run python real/drop_test.py            # about 40 s; ball in hand, camera and arm connected

Fitting whole flights uses every camera frame of a flight, so the 1-2 cm noise of height-from-ball-size averages out.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arm as A                                                    # noqa: E402
import rig                                                         # noqa: E402
from bounce.robot import ArmKinematics, build_model                # noqa: E402

G = 9.81


def restitution(track, min_pts=5):
    """track: (t, height above the paddle face) of one ball. Split into flights (above 1 cm), fit h = a + b t - g t^2 / 2
    per flight, impact speed sqrt(b^2 + 2 g a); e for each pair of consecutive flights. Returns (e list, speeds)."""
    t, h = np.asarray(track, float).T
    flights, cur = [], []
    for ti, hi in zip(t, h):
        if hi > 0.02:                                          # in flight (noise is ~1.5 cm)
            cur.append((ti, hi))
        elif cur:
            flights.append(cur)
            cur = []
    if cur:
        flights.append(cur)
    out = []                                                       # (start time, end time, impact speed)
    for f in flights:
        f = np.array(f)
        if len(f) < min_pts:
            continue
        tt = f[:, 0] - f[0, 0]
        X = np.stack([np.ones_like(tt), tt], 1)
        a, b = np.linalg.lstsq(X, f[:, 1] + 0.5 * G * tt ** 2, rcond=None)[0]
        if b * b + 2 * G * a > 0:
            out.append((f[0, 0], f[-1, 0], float(np.sqrt(b * b + 2 * G * a))))
    es = [s2 / s1 for (_, e1, s1), (b2, _, s2) in zip(out[:-1], out[1:]) if b2 - e1 < 0.2 and s2 < s1]
    return es, [s for *_, s in out]


def main(seconds=40.0):
    cc = rig.CONFIG["cameras"]["overhead"]
    cam = rig.Camera(rig.camera_index(cc), cc["width"], cc["height"], cc["fps"], rig.CONFIG.get("camera_latency_s", 0.035))
    rig.lock_camera(cam, cc)
    model = rig.CameraModel()
    px = model.ball_px
    finder = rig.BallFinder(rig.CONFIG.get("ball_colour", "orange"), **({"min_r": 0.5 * px, "max_r": 3.0 * px} if px else {}))
    kin = ArmKinematics(build_model(), 1)
    hold = json.loads(A.HOLD.read_text())
    b = A.open_bus()
    rest = A.hold_here(b)
    track = []
    try:
        print(f"to the bounce pose ({A.glide(b, hold, 3.0):.1f} deg); the arm holds still")
        fk = kin.forward(torch.tensor(rig.deg_to_sim(A.read_pos(b)), dtype=torch.float32)[None])
        o, n = fk["paddle_o"][0].numpy(), fk["paddle_R"][0, :, 2].numpy()
        print(f"drop the ball onto the paddle centre from 20-30 cm, let it bounce out, repeat; {seconds:.0f} s")
        t_end, last = time.perf_counter() + seconds, -1
        while time.perf_counter() < t_end:
            frame, tc, k = cam.get()
            if frame is None or k == last:
                time.sleep(0.003)
                continue
            last = k
            d = finder.find(frame)
            if d is None:
                continue
            p = model.ball_position(*d)
            if np.hypot(*(p - o)[:2]) < 0.10:                      # over the paddle
                track.append((tc, float((p - o) @ n) - rig.BALL_RADIUS))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        A.glide(b, {nm: rest[nm] for nm in A.ARM}, 3.0)
        A.park(b)
        cam.close()
    es, speeds = restitution(track)
    if not es:
        raise SystemExit(f"no complete bounce pairs seen ({len(track)} detections over the paddle); drop from higher, "
                         "near the centre")
    print(f"restitution e = {np.median(es):.2f} (median of {len(es)} impacts; spread {np.percentile(es, 10):.2f}-"
          f"{np.percentile(es, 90):.2f}); impact speeds {np.round(speeds[:8], 2).tolist()} m/s")
    (A.HERE / "logs").mkdir(exist_ok=True)
    np.save(A.HERE / "logs" / f"drop-{time.strftime('%Y%m%d-%H%M%S')}.npy", np.array(track))


if __name__ == "__main__":
    main()
