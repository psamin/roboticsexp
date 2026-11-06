"""Level the paddle with the ball as the spirit level: hold the bounce pose, watch the ball roll, tilt the wrist
against the roll, repeat until the ball stays put; then fit the wrist offsets (joint_map) and the hold pose to it.

  uv run python real/level_ball.py [--start-tilt FWD LEFT]

Start with the arm folded at rest. It glides to the bounce pose and holds. Each round: set the ball near the paddle
centre and take your hand away. The camera follows the roll for up to 1.2 s; take the ball off once it has rolled. A
hollow ball rolls with acceleration 0.6 g sin(tilt) toward where the face slopes down, so the roll gives the real
tilt. One roll varies by 1-2 deg (wrist slack, the hand setting the ball), so the script takes the median of ROLLS rolls,
corrects the wrist against it, and saves once it is below GOAL_DEG.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arm as A                                                    # noqa: E402
import level as L                                                  # noqa: E402
import rig                                                         # noqa: E402
from bounce.env import BounceEnv, EnvCfg                           # noqa: E402

G = 9.81
GOAL_DEG = 0.6                      # stop when the median real tilt is below this (the wrist has ~1.5 deg of slack)
ROUNDS = 8                          # corrections
ROLLS = 3                           # rolls per correction (median): one roll varies 1-2 deg (slack, hand)
MAX_DEG = 6.0                       # a roll reading more is a pushed or dropped ball (one read 13.6 deg)
MIN_FRAMES = 12                     # an 8-frame roll (0.33 s) read 9.4 deg and aborted the 2026-11-05 levelling


def tilt_xy(kin, q):
    """Face normal's x, y components (sin of the tilt toward +x forward and +y left) for sim joint angles q."""
    return kin.forward(torch.tensor(q, dtype=torch.float32)[None])["paddle_R"][0, :2, 2].numpy().astype(float)


def on_plane(model, u, v, z):
    """Where the pixel's ray meets the level plane at height z (base frame): a ball rolling on the paddle."""
    d = model.R @ np.array([(u - model.K[0, 2]) / model.K[0, 0], (v - model.K[1, 2]) / model.K[1, 1], 1.0])
    return model.t + d * (z - model.t[2]) / d[2]


def watch_roll(cam, finder, model, home, timeout=60.0):
    """Wait for the ball on the paddle, follow it for up to 1.2 s; return the real face normal xy from its roll."""
    t_end, last, track, seen, off = time.perf_counter() + timeout, -1, [], 0, None
    while time.perf_counter() < t_end:
        frame, tc, n = cam.get()
        if frame is None or n == last:
            time.sleep(0.003)
            continue
        last = n
        d = finder.find(frame)
        if d is None:
            if track and tc - track[-1][0] > 0.2:
                break                                              # lost it (rolled off or taken away)
            continue
        seen += 1
        p = on_plane(model, d[0], d[1], home[2] + rig.BALL_RADIUS) - home
        z = (model.ball_position(*d) - home)[2]        # from the ball's size: 1 px of radius is ~4 cm, so only loosely
        on = np.hypot(p[0], p[1]) < 0.07 and abs(z) < 0.10
        if not on:
            off = (round(100 * p[0], 1), round(100 * p[1], 1), round(100 * z, 1))
            if track:
                break
            continue
        track.append((tc, p[0], p[1]))
        if track[-1][0] - track[0][0] > 1.2:
            break
    if len(track) < MIN_FRAMES:
        print(f"  ball seen in {seen} frames, {len(track)} in a row on the paddle (last off-paddle x, y, z cm: {off})")
        return None
    a = np.array(track)
    t = a[:, 0] - a[0, 0]
    X = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)            # p = p0 + v0 t + a t^2 / 2
    acc = np.linalg.lstsq(X, a[:, 1:3], rcond=None)[0][2]
    return acc / (0.6 * G), len(track), float(t[-1])


def trim_sag(b, target, cmd, n=150):
    """Nudge the command until the joints sit at `target` (gravity sag and the wrist's slack); returns the command."""
    for _ in range(n):
        pos = A.read_pos(b)
        for nme in A.ARM:
            cmd[nme] = float(np.clip(cmd[nme] + 0.06 * (target[nme] - pos[nme]), target[nme] - 12, target[nme] + 12))
        b.sync_write("Goal_Position", cmd)
        time.sleep(0.02)
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-tilt", type=float, nargs=2, metavar=("FWD", "LEFT"),
                    help="deg: a previous run's mean real tilt at the hold pose, corrected before the first roll")
    a = ap.parse_args()
    sign, off = rig._jmap()
    env = BounceEnv(EnvCfg(num_envs=1, device="cpu"))
    kin, q_home = env.kin, env.q_nom.numpy().astype(float)
    hold = json.loads(A.HOLD.read_text())
    cc = rig.CONFIG["cameras"]["overhead"]
    cam = rig.Camera(rig.camera_index(cc), cc["width"], cc["height"], cc["fps"], rig.CONFIG.get("camera_latency_s", 0.035))
    rig.lock_camera(cam, cc)
    model = rig.CameraModel()
    px = model.ball_px
    finder = rig.BallFinder(rig.CONFIG.get("ball_colour", "orange"), **({"min_r": 0.5 * px, "max_r": 2.5 * px} if px else {}))
    b = A.open_bus()
    rest = A.hold_here(b)
    saved = None
    try:
        A.set_registers(b, {"P_Coefficient": 32, **A.BOUNCE_REGISTERS})   # hold as stiffly as when bouncing
        print(f"to the bounce pose ({A.glide(b, hold, 3.0):.1f} deg)")
        target = dict(hold)
        cmd = trim_sag(b, target, dict(hold))                      # trim the sag so the joints sit at the target
        q = rig.deg_to_sim(A.read_pos(b))
        home = kin.forward(torch.tensor(q, dtype=torch.float32)[None])["paddle_o"][0].numpy()
        finder.roi = model.paddle_roi(home, size=(cc["width"], cc["height"]))  # only look around the paddle
        def correct(real, q, mdl):
            """Tilt the wrist against the slope `real` (face normal xy), measured at sim pose q where the model said mdl."""
            nonlocal target, cmd
            J = np.stack([(tilt_xy(kin, q + 0.005 * np.eye(5)[j]) - mdl) / 0.005 for j in (3, 4)], 1)
            dq = np.clip(np.linalg.solve(J, -real), -np.radians(3), np.radians(3))
            q_tgt = rig.deg_to_sim(target)
            q_tgt[3:5] += dq
            new_target = rig.sim_to_deg(q_tgt)
            new = {n: cmd[n] + new_target[n] - target[n] for n in A.ARM}
            for s in np.linspace(0, 1, 30)[1:]:                    # ramp the held command by the correction
                b.sync_write("Goal_Position", {n: cmd[n] + s * (new[n] - cmd[n]) for n in A.ARM})
                time.sleep(0.02)
            target, cmd = new_target, trim_sag(b, new_target, new, n=75)   # then make the joints reach it (slack)
            print(f"  wrist flex {np.degrees(dq[0]):+.2f} deg, roll {np.degrees(dq[1]):+.2f} deg")

        if a.start_tilt:
            q = rig.deg_to_sim(A.read_pos(b))
            print(f"starting from a tilt of {a.start_tilt} deg (forward, left) at the hold pose")
            correct(np.sin(np.radians(a.start_tilt)), q, tilt_xy(kin, q))
        fixes, rolls = 0, []
        while fixes < ROUNDS:                 # a hand over the ball ends a try early: only rolls count (Ctrl-C stops)
            print(f"fix {fixes + 1}, roll {len(rolls) + 1}/{ROLLS}: set the ball near the paddle centre, take your hand away...")
            got = watch_roll(cam, finder, model, home)
            if got is None:
                print("  did not see the ball roll on the paddle (put it near the centre); trying again")
                continue
            real, nfr, span = got
            pos = A.read_pos(b)
            q = rig.deg_to_sim(pos)
            mdl = tilt_xy(kin, q)
            deg = np.degrees(np.arcsin(np.clip(real, -1, 1)))
            print(f"  real tilt {np.round(deg, 2)} deg (toward +x forward, +y left) from {nfr} frames over {span:.2f} s; "
                  f"model says {np.round(np.degrees(mdl), 2)}; wrist flex {pos['wrist_flex']:.2f}, roll {pos['wrist_roll']:.2f}")
            if np.hypot(*deg) > MAX_DEG:
                print(f"  over {MAX_DEG} deg with the joints unmoved: a pushed or dropped ball; not counted")
                continue
            rolls.append((real, q, mdl))
            if len(rolls) < ROLLS:
                print("  take the ball off")
                time.sleep(1.0)
                continue
            real, q, mdl = np.median([r[0] for r in rolls], 0), rolls[-1][1], rolls[-1][2]
            rolls = []
            deg = np.degrees(np.arcsin(np.clip(real, -1, 1)))
            print(f"  median of {ROLLS} rolls: {np.round(deg, 2)} deg")
            if np.hypot(*deg) < GOAL_DEG:
                saved = A.read_pos(b)
                print("  level")
                break
            correct(real, q, mdl)
            fixes += 1
            print("  take the ball off")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        A.glide(b, hold, 1.0)
        A.pause("Take the ball off, press Enter (the arm folds back and turns off)...")
        A.set_registers(b, A.LEROBOT_DEFAULTS)
        A.glide(b, {n: rest[n] for n in A.ARM}, 3.0)
        A.park(b)
        cam.close()
    if saved is None:
        raise SystemExit("not levelled; nothing saved")
    real_deg = np.array([saved[n] for n in A.ARM], float)
    new = L.fit_wrist_offsets(kin, real_deg, sign, off)
    if np.abs(new - off).max() > L.MAX_CHANGE:
        raise SystemExit(f"the fit wants to move the wrist offsets by {np.round((new - off)[3:], 1)} deg; nothing written")
    import re
    text = L.CONFIG_FILE.read_text()
    L.CONFIG_FILE.write_text(re.sub(r"offset_deg: \[[^\]]*\]", "offset_deg: [" + ", ".join(f"{x:.1f}" for x in new) + "]",
                                    text, count=1))
    A.HOLD.write_text(json.dumps({n: round(float(v), 2) for n, v in zip(A.ARM, np.degrees(q_home) * sign + new)}, indent=1))
    print(f"wrist offsets: flex {off[3]:.1f} -> {new[3]:.1f}, roll {off[4]:.1f} -> {new[4]:.1f} deg; hold pose rewritten")


if __name__ == "__main__":
    main()
