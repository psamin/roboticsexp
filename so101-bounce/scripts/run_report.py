"""Summarise a real-arm run log from real/bounce_run.py: per policy activation, how long it ran, the ball's flights
(apex height above the paddle face) and where the ball was over the paddle at each apex and when last seen.

  uv run python scripts/run_report.py real/logs/run-<time>.npz [more ...]

A flight is a run of detections more than 1.5 cm above the face (gaps under 0.15 s bridged) that rises above 3 cm.
x is forward (negative = toward the handle), y left, from the paddle centre at the start of the activation.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "real"))
import rig                                                          # noqa: E402
from bounce.env import EnvCfg                                       # noqa: E402
from bounce.robot import ArmKinematics, build_model                 # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def spans(state):
    """(first, last) tick index of each run of active ticks."""
    d = np.diff(np.r_[0, (np.asarray(state) == "active").astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def flights(t, h):
    """Index lists of the detections in each flight."""
    out, cur = [], []
    for m in range(len(t)):
        if h[m] > 0.015 and (not cur or t[m] - t[cur[-1]] < 0.15):
            cur.append(m)
            continue
        if cur and h[cur].max() > 0.03:
            out.append(cur)
        cur = [m] if h[m] > 0.015 else []
    if cur and h[cur].max() > 0.03:
        out.append(cur)
    return out


def report(path):
    d = np.load(path, allow_pickle=True)
    t, det = d["t"], d["detections"].reshape(-1, 4)
    q = np.stack([rig.deg_to_sim(dict(zip(ARM, r))) for r in d["q_deg"]])
    kin = rig.shift_paddle(ArmKinematics(build_model(EnvCfg().paddle), len(q), "cpu"))
    fk = kin.forward(torch.tensor(q, dtype=torch.float32))
    o, n = fk["paddle_o"].numpy(), fk["paddle_R"][:, :, 2].numpy()
    print(f"{Path(path).name}: {Path(str(d['checkpoint'])).name}, {t[-1]:.0f} s, {len(det)} detections")
    total = []
    for k, (i0, i1) in enumerate(spans(d["state"])):
        s = det[(det[:, 0] >= t[i0]) & (det[:, 0] <= t[i1])]
        j = np.clip(np.searchsorted(t, s[:, 0]), 0, len(t) - 1)            # tick at each detection
        h = np.einsum("ij,ij->i", s[:, 1:4] - o[j], n[j]) - rig.BALL_RADIUS
        xy = 100 * (s[:, 1:3] - o[i0, :2])
        f = flights(s[:, 0], h)
        top = [m[int(np.argmax(h[m]))] for m in f]
        total.append(len(f))
        line = f"  #{k + 1} at {t[i0]:5.1f} s for {t[i1] - t[i0]:5.2f} s, {len(s)} detections: {len(f)} flights"
        if f:
            ap = 100 * h[top]
            line += (f", apex {np.median(ap):.1f} cm median ({', '.join(f'{v:.0f}' for v in ap[:10])}"
                     f"{' ...' if len(ap) > 10 else ''}), x at apex {xy[top, 0].min():+.0f} to {xy[top, 0].max():+.0f} cm, "
                     f"y {xy[top, 1].min():+.0f} to {xy[top, 1].max():+.0f} cm")
        if len(s):
            line += f"; last seen x {xy[-1, 0]:+.0f}, y {xy[-1, 1]:+.0f}, {100 * h[-1]:.0f} cm up"
        print(line)
    if total:
        print(f"  {len(total)} activations, {sum(total)} flights, most in one activation {max(total)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
