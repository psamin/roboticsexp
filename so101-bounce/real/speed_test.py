"""Why is the real paddle slow? Run the launch stroke and elbow steps at the bounce pose while logging joint positions,
supply voltage, current and load as fast as the bus allows; once with the derivative gain D = 32 (LeRobot default),
once with D = 0. Ball off. About 30 s of motion.

  uv run python real/speed_test.py

Reading the summary: if the supply voltage dips well below ~11 V during the strokes, the 12 V supply is current-
limited (use a bigger one); if current sits at the protection limit, the servos are limiting themselves; if D = 0 is
much faster, the damping gain is the brake. Log: real/logs/speed-<time>.npz (fitted later by the training agent).
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
from bounce import launch as LAUNCH                                # noqa: E402
from bounce.env import EnvCfg                                      # noqa: E402
from bounce.robot import ArmKinematics, build_model                # noqa: E402

P_GAIN = 32


def record(b, rows, label, seconds, plan=None, hz=50):
    """Write the planned goals at `hz` while sampling the bus as fast as it allows."""
    t0, k = time.perf_counter(), 0
    while (now := time.perf_counter()) - t0 < seconds:
        if plan is not None and k < len(plan) and now - t0 >= k / hz:
            b.sync_write("Goal_Position", plan[k])
            k += 1
        pos = A.read_pos(b)
        volt = b.sync_read("Present_Voltage", A.ARM, normalize=False)
        cur = b.sync_read("Present_Current", ["shoulder_lift", "elbow_flex", "wrist_flex"], normalize=False)
        rows.append((label, now, *[pos[n] for n in A.ARM], min(volt.values()), cur["shoulder_lift"], cur["elbow_flex"],
                     cur["wrist_flex"]))
        time.sleep(max(0.0, 0.002 - (time.perf_counter() - now)))   # at most ~500 samples/s


def main():
    kin = ArmKinematics(build_model(), 1)
    scale = np.array(EnvCfg().action_scale)
    hold = json.loads(A.HOLD.read_text())
    lp = LAUNCH.DEFAULTS[P_GAIN]
    b = A.open_bus()
    rest = A.hold_here(b)
    rows = []
    try:
        A.set_registers(b, {"P_Coefficient": P_GAIN, **A.BOUNCE_REGISTERS})
        print(f"to the bounce pose ({A.glide(b, hold, 3.0):.1f} deg)")
        q_ref = rig.deg_to_sim(hold)
        for d_gain in (32, 0):
            A.set_registers(b, {"D_Coefficient": d_gain})
            for steps in (lp.strike_steps, 6):                     # tuned stroke, then a longer one
                p = LAUNCH.Launch(**lp.as_dict())
                p.strike_steps = steps
                acts = LAUNCH.actions(p)[:, 0].numpy()
                plan = [rig.sim_to_deg(q_ref + scale * a) for a in acts] + [hold] * 25
                record(b, rows, f"D{d_gain}-launch{steps}", 1.2, plan)
            for sgn in (1, -1):                                    # elbow step +-10 deg, then back
                plan = [{**hold, "elbow_flex": hold["elbow_flex"] + sgn * 10}] * 30 + [hold] * 30
                record(b, rows, f"D{d_gain}-elbow{'+' if sgn > 0 else '-'}", 1.3, plan)
        A.set_registers(b, {"D_Coefficient": 32})
    except KeyboardInterrupt:
        print("stopped")
    finally:
        A.set_registers(b, {"D_Coefficient": 32, **A.LEROBOT_DEFAULTS})
        A.glide(b, hold, 1.0)
        A.glide(b, {n: rest[n] for n in A.ARM}, 3.0)
        A.park(b)
    if not rows:
        return
    labels = np.array([r[0] for r in rows])
    a = np.array([r[1:] for r in rows], float)
    (A.HERE / "logs").mkdir(exist_ok=True)
    out = A.HERE / "logs" / f"speed-{time.strftime('%Y%m%d-%H%M%S')}.npz"
    np.savez_compressed(out, label=labels, data=a, columns=np.array(["t"] + A.ARM + ["volt_min", "cur_lift", "cur_elbow", "cur_flex"]))
    q = torch.tensor(np.stack([rig.deg_to_sim(dict(zip(A.ARM, r))) for r in a[:, 1:6]]), dtype=torch.float32)
    pz = ArmKinematics(build_model(), len(q)).forward(q)["paddle_o"][:, 2].numpy()
    print(f"{'test':16s} {'paddle up m/s':>13s} {'elbow deg/s':>11s} {'min volts':>9s} {'peak current (lift/elbow/flex)':>31s}")
    for lab in dict.fromkeys(labels):
        m = labels == lab
        t = a[m, 0]
        vz = np.gradient(pz[m], t)
        el = np.abs(np.gradient(a[m, 3], t))
        print(f"{lab:16s} {vz.max():13.2f} {el.max():11.0f} {a[m, 6].min() / 10:9.1f} "
              f"{int(a[m, 7].max()):>10d} {int(a[m, 8].max()):>9d} {int(a[m, 9].max()):>9d}")
    print(f"log: {out}  ({len(a)} samples, {len(a) / max(1e-6, a[-1, 0] - a[0, 0]):.0f} per s)")


if __name__ == "__main__":
    main()
