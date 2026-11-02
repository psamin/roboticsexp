"""Level the paddle with the ball as the spirit level: hold the bounce pose, watch the ball roll, tilt the wrist
against the roll, repeat until the ball stays put; then fit the wrist offsets (joint_map) and the hold pose to it.

  uv run python real/level_ball.py

Start with the arm folded at rest. It glides to the bounce pose and holds. Each round: set the ball near the paddle
centre and take your hand away. The camera follows the roll for up to 1.2 s; take the ball off once it has rolled. A
hollow ball rolls with acceleration 0.6 g sin(tilt) toward where the face slopes down, so the roll gives the real
tilt. The script compares it with the tilt the model computes from the joints and corrects the wrist. If a correction
makes the roll worse, a wrist direction in the joint map is reversed: it says which and stops without saving.
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
import level as L                                                  # noqa: E402
import rig                                                         # noqa: E402
from bounce.env import BounceEnv, EnvCfg                           # noqa: E402

G = 9.81
GOAL_DEG = 0.4                      # stop when the real tilt is below this
ROUNDS = 8


def tilt_xy(kin, q):
    """Face normal's x, y components (sin of the tilt toward +x forward and +y left) for sim joint angles q."""
    return kin.forward(torch.tensor(q, dtype=torch.float32)[None])["paddle_R"][0, :2, 2].numpy().astype(float)


def watch_roll(cam, finder, model, home, timeout=60.0):
    """Wait for the ball on the paddle, follow it for up to 1.2 s; return the real face normal xy from its roll."""
    t_end, last, track = time.perf_counter() + timeout, -1, []
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
        p = model.ball_position(*d) - home
        on = np.hypot(p[0], p[1]) < 0.07 and abs(p[2]) < 0.05
        if not on:
            if track:
                break
            continue
        track.append((tc, p[0], p[1]))
        if track[-1][0] - track[0][0] > 1.2:
            break
    if len(track) < 8:
        return None
    a = np.array(track)
    t = a[:, 0] - a[0, 0]
    X = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)            # p = p0 + v0 t + a t^2 / 2
    acc = np.linalg.lstsq(X, a[:, 1:3], rcond=None)[0][2]
    return acc / (0.6 * G), len(track), float(t[-1])


def main():
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
        print(f"to the bounce pose ({A.glide(b, hold, 3.0):.1f} deg)")
        cmd = dict(hold)
        for _ in range(150):                                       # trim the sag so the joints sit at the command
            pos = A.read_pos(b)
            for nme in A.ARM:
                cmd[nme] = float(np.clip(cmd[nme] + 0.06 * (hold[nme] - pos[nme]), hold[nme] - 12, hold[nme] + 12))
            b.sync_write("Goal_Position", cmd)
            time.sleep(0.02)
        q = rig.deg_to_sim(A.read_pos(b))
        home = kin.forward(torch.tensor(q, dtype=torch.float32)[None])["paddle_o"][0].numpy()
        prev = None
        for k in range(ROUNDS):
            print(f"round {k + 1}: set the ball near the paddle centre and take your hand away...")
            got = watch_roll(cam, finder, model, home)
            if got is None:
                print("  did not see the ball roll on the paddle (put it near the centre); trying again")
                continue
            real, nfr, span = got
            q = rig.deg_to_sim(A.read_pos(b))
            mdl = tilt_xy(kin, q)
            deg = np.degrees(np.arcsin(np.clip(real, -1, 1)))
            print(f"  real tilt {np.round(deg, 2)} deg (toward +x forward, +y left) from {nfr} frames over {span:.2f} s; "
                  f"model says {np.round(np.degrees(mdl), 2)}")
            if np.hypot(*deg) < GOAL_DEG:
                saved = A.read_pos(b)
                print("  level")
                break
            if prev is not None and np.hypot(*deg) > 1.3 * np.hypot(*prev[0]):
                dmdl = mdl - prev[1]                               # how the model moved vs how reality moved
                dreal = real - np.sin(np.radians(prev[0]))
                bad = [nm for nm, i in (("wrist_flex (forward/back)", 0), ("wrist_roll (left/right)", 1))
                       if abs(dmdl[i]) > 0.005 and np.sign(dmdl[i]) != np.sign(dreal[i])]
                raise SystemExit(f"the correction made it worse: real tilt went {np.round(prev[0], 2)} -> {np.round(deg, 2)} deg "
                                 f"while the model moved {np.round(np.degrees(dmdl), 2)}. Reversed in the joint map: "
                                 f"{', '.join(bad) or 'unclear (noisy roll; run again)'}. Nothing saved.")
            J = np.stack([(tilt_xy(kin, q + 0.005 * np.eye(5)[j]) - mdl) / 0.005 for j in (3, 4)], 1)
            dq = np.linalg.solve(J, -real)                         # tilt the wrist against the measured slope
            dq = np.clip(dq, -np.radians(3), np.radians(3))
            q_cmd = rig.deg_to_sim(cmd)
            q_cmd[3:5] += dq
            new = rig.sim_to_deg(q_cmd)
            for s in np.linspace(0, 1, 30)[1:]:                    # ramp from the held (sag-trimmed) command, not
                b.sync_write("Goal_Position", {n: cmd[n] + s * (new[n] - cmd[n]) for n in A.ARM})   # from the joints
                time.sleep(0.02)
            cmd = new
            for _ in range(25):                                    # settle
                b.sync_write("Goal_Position", cmd)
                time.sleep(0.02)
            prev = (deg, mdl)
            print(f"  wrist flex {np.degrees(dq[0]):+.2f} deg, roll {np.degrees(dq[1]):+.2f} deg; take the ball off")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        A.glide(b, hold, 1.0)
        A.pause("Take the ball off, press Enter (the arm folds back and turns off)...")
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
