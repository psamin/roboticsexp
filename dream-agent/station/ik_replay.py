"""Replay a rollout's recorded hand targets through dream's own local IK (as dream.client_ik does), with and without
a region box, and count elbow-backwards solutions. Offline: no model calls, no arm.
Rollouts before the wrist-roll shift (my_robot.ROLL_SHIFT_DEG; Nov 5 00:45-00:58) were recorded with the roll in
the old units: reproduce their numbers at commit 915c00c."""
import ast
import glob
import sys

import numpy as np
from dream_ik import decode
from dream_ik.solver import ChunkIK
from dream_ik.specs import EMBODIMENTS, SERVE, Region, ServeSpec

run = sys.argv[1]
ARM = "so101_faas"
spec = EMBODIMENTS[ARM]
REST = {"right": np.array([0.14, -1.71, 1.437, 0.765, -1.5708, 0.0]),
        "left": np.array([0.3409, -1.7104, 1.5285, 1.1724, -1.4119, 0.0])}
EXEC = 20                                     # rows played before the next reply takes over (log: ~20 executed)


def build(region):
    obstacles = {s: tuple(spec.obstacles[s]) for s in spec.sides}
    if region is not None:
        box = Region(lo=tuple(region[0]), hi=tuple(region[1]))
        obstacles = {s: obstacles[s] + ((box,) if s == "right" else ()) for s in spec.sides}
    return ChunkIK(ARM, SERVE.tool, 30.0, obstacles)


GUARD_LO = np.radians([-99, -98, -26, -81, -155, 0.0])
GUARD_HI = np.radians([50, 60, 82, 92, -59, 0.0])


def replay(ik, solve="traj", guard=False):
    serve = ServeSpec(smooth_window=0, seam_blend_ticks=0, solve=solve)
    q_from = np.stack([REST[s] for s in ik.sides])
    q_before = np.stack([q_from, q_from], axis=1)
    elbows, flips, misses = [], 0, []
    for f in sorted(glob.glob(f"{run}/chunks/reply_*.npz"))[1:]:
        pk = ast.literal_eval(str(np.load(f, allow_pickle=True)["per_key"]))
        xyz = np.stack([np.asarray(pk[f"action.{s}_eef_xyz"]) for s in ik.sides])
        rot6 = np.stack([np.asarray(pk[f"action.{s}_eef_rot6"]) for s in ik.sides])
        if "left" in ik.sides:                    # the parked arm stays at rest: its targets are its rest pose's
            i = ik.sides.index("left")
            xyz[i], rot6[i] = xyz[i][:1], rot6[i][:1]
        q, res = decode.solve_chunk(ik, xyz, rot6, q_from, np.inf, serve, q_before=q_before)
        r = ik.sides.index("right")
        tgt = xyz[r, :EXEC]
        inside = np.all((tgt >= [-9, -0.12, 0.02]) & (tgt <= [0.546, 0.345, 0.25]), axis=1)
        misses += list(100 * np.asarray(res.pos_miss)[r, :EXEC][inside])
        played = q[r, :EXEC]
        elbows += list(np.degrees(played[:, 2]))
        flips += int(np.any(played[:, 2] < np.radians(-26)))
        if guard:                                 # the arm executes the clamped command (my_robot.py's guard)
            r_ = ik.sides.index("right")
            q[r_] = np.clip(q[r_], GUARD_LO, GUARD_HI)
        q_before = np.stack([q[:, EXEC - 1], q[:, EXEC - 2]], axis=1)
        q_from = q[:, EXEC - 1].copy()
        q_from[ik.sides.index("left")] = REST["left"]
    e = np.array(elbows)
    global MISS
    MISS = np.array(misses)
    return flips, len(e) // EXEC, np.percentile(e, [1, 50, 99])


BOX = ((None, -0.12, 0.02), (0.546, 0.345, 0.25))
mode = sys.argv[2] if len(sys.argv) > 2 else "patched"
if mode == "patched":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")
    import my_robot                                    # narrows dream's IK limits on import
from dream_ik import solver as _s
ik = build(BOX)
r = ik.sides.index("right")
print(mode, "IK limits (deg):", [f"{a:.0f}..{b:.0f}" for a, b in zip(np.degrees(ik.side_lo[r]), np.degrees(ik.side_hi[r]))])
flips, n, pct = replay(ik, "traj", False)
print(f"{mode}: chunks whose IK asked elbow-backwards {flips}/{n} | hand miss for targets inside the demo area: median {np.median(MISS):.1f} cm, p90 {np.percentile(MISS, 90):.1f} cm ({len(MISS)} rows)")
