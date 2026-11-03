"""Turn a real/speed_test.py log into the run-log format scripts/fit_servo.py reads (t, q_deg, goal_deg, action, state,
checkpoint). The speed test does not log its goals, so they are rebuilt from its plan: the goal at each sample is the
last 50 Hz write made by then (writes at k / 50 s from each segment's first sample, as speed_test.record does).

  uv run python scripts/speed_to_runlog.py real/logs/speed-<time>.npz out.npz [--prefix D32]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real"))
sys.path.insert(0, str(ROOT))
import arm as A                                                    # noqa: E402
import rig                                                         # noqa: E402
from bounce import launch as LAUNCH                                # noqa: E402
from bounce.env import EnvCfg                                      # noqa: E402

P_GAIN = 32                                                        # as in real/speed_test.py


def plan(label, hold, q_ref, scale):
    """The goal list speed_test.py wrote for one segment label, e.g. 'D32-launch5' or 'D0-elbow-'."""
    kind = label.split("-", 1)[1]
    if kind.startswith("launch"):
        p = LAUNCH.Launch(**LAUNCH.DEFAULTS[P_GAIN].as_dict())
        p.strike_steps = int(kind[len("launch"):])
        acts = LAUNCH.actions(p)[:, 0].numpy()
        return [rig.sim_to_deg(q_ref + scale * a) for a in acts] + [hold] * 25
    sgn = 1 if kind.endswith("+") else -1
    return [{**hold, "elbow_flex": hold["elbow_flex"] + sgn * 10}] * 30 + [hold] * 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--prefix", default="D32", help="segments to keep (D32: the gain the policies run at)")
    a = ap.parse_args()
    d = np.load(a.log, allow_pickle=True)
    labels, data = d["label"], d["data"]
    hold = json.loads(A.HOLD.read_text())
    q_ref = rig.deg_to_sim(hold)
    scale = np.array(EnvCfg().action_scale)
    keep = np.array([lb.startswith(a.prefix + "-") for lb in labels])
    t, goals = [], []
    for lb in dict.fromkeys(labels[keep]):
        m = labels == lb
        ts = data[m, 0]
        pl = plan(lb, hold, q_ref, scale)
        k = np.minimum(np.floor((ts - ts[0]) * 50 + 1e-9).astype(int), len(pl) - 1)
        goals += [[pl[i][n] for n in A.ARM] for i in k]
        t.append(ts)
    t = np.concatenate(t)
    goal_deg = np.array(goals)
    q_deg = data[keep, 1:6]
    action = np.stack([(rig.deg_to_sim(dict(zip(A.ARM, g))) - q_ref) / scale for g in goal_deg]).astype(np.float32)
    np.savez_compressed(a.out, t=t, q_deg=q_deg, goal_deg=goal_deg, action=action,
                        state=np.array(["active"] * len(t)), checkpoint=np.array(f"speed-test-p{P_GAIN}"),
                        virtual_ball=np.zeros((0, 4)), hz=np.array(50.0))
    print(f"{a.out}: {len(t)} samples over {t[-1] - t[0]:.1f} s from {sorted(set(labels[keep]))}")


if __name__ == "__main__":
    main()
