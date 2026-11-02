"""Level the paddle by hand at the bounce pose, then fit the wrist offsets (joint_map in real/config.yaml) so the
model agrees and recompute the hold pose from them. Run it after (re)mounting the paddle.

  uv run python real/level.py

Start with the arm folded at rest. It glides to the bounce pose and holds. Hold the paddle and press Enter: the two
wrist motors turn off. Tilt the paddle until the blade is level (phone level app on the face), keep it there, press
Enter. The wrist motors come back on; let go, press Enter, and the arm folds back to rest and turns off.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arm as A                                                    # noqa: E402
import rig                                                         # noqa: E402
from bounce.env import BounceEnv, EnvCfg                           # noqa: E402

WRIST = ["wrist_flex", "wrist_roll"]
CONFIG_FILE = rig.HERE / "config.yaml"
MAX_CHANGE = 20.0                                                  # deg; a bigger jump means the paddle was not level


def tilt_and_normal(kin, real_deg, sign, off):
    q = torch.tensor(np.radians(sign * (real_deg - off)), dtype=torch.float32)[None]
    n = kin.forward(q)["paddle_R"][0, :, 2].numpy().astype(float)
    return np.degrees(np.arccos(np.clip(n[2], -1, 1))), n


def fit_wrist_offsets(kin, real_deg, sign, off):
    """Wrist flex and roll offsets (deg) that make the paddle normal vertical at real_deg (5 LeRobot angles)."""
    off = np.array(off, float)
    for _ in range(20):
        f0 = tilt_and_normal(kin, real_deg, sign, off)[1][:2]
        J = np.stack([(tilt_and_normal(kin, real_deg, sign, off + 0.01 * np.eye(5)[j])[1][:2] - f0) / 0.01
                      for j in (3, 4)], 1)
        off[[3, 4]] -= np.linalg.solve(J, f0)
    return off


def main():
    sign, off0 = rig._jmap()
    env = BounceEnv(EnvCfg(num_envs=1, device="cpu"))
    kin, q_home = env.kin, env.q_nom.numpy().astype(float)
    hold = json.loads(A.HOLD.read_text())
    b = A.open_bus()
    rest = A.hold_here(b)
    pos = None
    try:
        print(f"to the bounce pose ({A.glide(b, hold, 3.0):.1f} deg)")
        input("Hold the paddle (the wrist goes limp next), press Enter...")
        b.disable_torque(WRIST)
        input("Wrist motors OFF. Tilt the paddle until the blade is level (phone level on the face), keep holding it "
              "there, press Enter...")
        pos = A.read_pos(b)
    finally:
        cur = A.read_pos(b)
        b.sync_write("Goal_Position", {n: cur[n] for n in WRIST})
        b.enable_torque(WRIST)
        if pos is not None:
            A.pause("Wrist motors on. Let go of the paddle, press Enter (the arm folds back and turns off)...")
        A.glide(b, {n: rest[n] for n in A.ARM}, 3.0)
        A.park(b)
    real = np.array([pos[n] for n in A.ARM], float)
    was = tilt_and_normal(kin, real, sign, off0)[0]
    off = fit_wrist_offsets(kin, real, sign, off0)
    change = off - off0
    if np.abs(change).max() > MAX_CHANGE or tilt_and_normal(kin, real, sign, off)[0] > 0.5:
        raise SystemExit(f"the fit wants to move the wrist offsets by {np.round(change[3:], 1)} deg; was the paddle "
                         "level when you pressed Enter? Nothing written; run it again.")
    text = CONFIG_FILE.read_text()
    text = re.sub(r"offset_deg: \[[^\]]*\]", "offset_deg: [" + ", ".join(f"{x:.1f}" for x in off) + "]", text, count=1)
    CONFIG_FILE.write_text(text)
    new_hold = np.degrees(q_home) * sign + off
    A.HOLD.write_text(json.dumps({n: round(float(v), 2) for n, v in zip(A.ARM, new_hold)}, indent=1))
    print(f"the model was {was:.1f} deg off level. Wrist offsets: flex {off0[3]:.1f} -> {off[3]:.1f}, roll "
          f"{off0[4]:.1f} -> {off[4]:.1f} deg (written to {CONFIG_FILE.name}); hold pose rewritten ({A.HOLD.name}).")


if __name__ == "__main__":
    main()
