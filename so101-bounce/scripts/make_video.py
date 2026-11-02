"""Video of a checkpoint in four scenarios, side and overhead views, with what happened written on each frame.

  MUJOCO_GL=egl python scripts/make_video.py --checkpoint best.pt --out video.mp4

For each scenario it runs --envs episodes and shows the one with the MEDIAN bounce count (typical, not the best).
Rendering is a kinematic replay in the MuJoCo SO-101 + paddle + ball model of what the training simulator computed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from bounce import ball as B
from bounce.env import FAIL_REASONS, BounceEnv, EnvCfg
from bounce.ppo import PPO
from bounce.robot import GRIPPER_CLOSED_ON_HANDLE
from bounce.visual import sample_look
from scripts.render import scene, set_paddle

SCENARIOS = [("Launch from rest, nominal physics", 0.0, True),
             ("Ball dropped onto the paddle, nominal physics", 0.0, False),
             ("Full randomisation + random hand taps", 1.0, None),
             ("Launch from rest, full randomisation", 1.0, True)]
REASON_TEXT = {"dropped": "ball dropped", "out_of_view": "ball left the camera view",
               "paddle_limits": "paddle left its safe zone", "overload": "servo overload",
               "resting": "ball came to rest", "none": "8 s episode completed"}


def rollout(agent, env_cfg, difficulty, rest, n, seed):
    env = BounceEnv(EnvCfg(**{**env_cfg, "num_envs": n, "seed": seed}))
    env.forced = {"difficulty": difficulty, "rest_start": rest}
    obs, _ = env.reset_all()
    first_blade = env.blade.numpy().copy()
    rec = {k: [] for k in ("q", "ball", "paddle_z", "home_z", "paddle_o", "paddle_R")}
    end, reason = np.full(n, env.max_steps), np.array(["none"] * n, dtype=object)
    done = torch.zeros(n, dtype=torch.bool)
    for k in range(env.max_steps):
        obs, _, term, trunc, _ = env.step(agent.act_deterministic(obs["policy"]))
        stats = env.pop_completed()
        if stats is not None:
            for i, eid in enumerate(stats["env_id"].long().tolist()):
                if not done[eid]:
                    end[eid], reason[eid] = k, FAIL_REASONS[int(stats["reason"][i])]
        rec["q"].append(env.q.clone())
        rec["ball"].append(env.p_ball.clone())
        rec["paddle_z"].append(env.o_prev[:, 2].clone())
        rec["home_z"].append(env.home_env[:, 2].clone())
        rec["paddle_o"].append(env.o_prev.clone())
        rec["paddle_R"].append(env.R_prev.clone())
        done |= term | trunc
        if bool(done.all()):
            break
    T = {k: torch.stack(v).numpy() for k, v in rec.items()}
    T["blade"] = first_blade
    return T, end, reason, env.dt


def bounce_events(ball_z, paddle_z, end):
    """Bounce peaks for one episode: (step, height of the peak above the lowest point since the previous peak).

    Peaks, not impacts: an impact lasts one physics step and falls between the 50 Hz samples."""
    zb = ball_z[:end + 1] - B.RADIUS
    events, low, k0 = [], zb[0], 0
    for k in range(1, len(zb) - 1):
        low = min(low, zb[k])
        if zb[k] >= zb[k - 1] and zb[k] > zb[k + 1] and zb[k] - low > 0.03:
            events.append((k, zb[k] - low))
            low = zb[k]
    return events


def main():
    import imageio.v2 as iio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="video.mp4")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--set", action="append", default=[], help="env overrides, e.g. hold_jitter=0")
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, "cpu", load_optimizer=False)
    env_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()}
    env_cfg.update(device="cpu")
    for kv in a.set:
        k, v = kv.split("=", 1)
        env_cfg[k] = float(v)
    rng = np.random.default_rng(a.seed)
    cams = [dict(lookat=[0.25, 0, 0.33], distance=0.9, azimuth=-90, elevation=-6),
            dict(lookat=[0.33, 0, 0.3], distance=0.95, azimuth=180, elevation=-58)]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    W, H, BAR = 2 * a.width, a.height, 72
    writer = None
    summary = []
    for title, diff, rest in SCENARIOS:
        T, end, reason, dt = rollout(agent, env_cfg, diff, rest, a.envs, a.seed)
        first_blade = T["blade"]
        counts = [len(bounce_events(T["ball"][:, e, 2], T["paddle_z"][:, e], end[e])) for e in range(a.envs)]
        e = int(np.argsort(counts)[len(counts) // 2])
        ev = bounce_events(T["ball"][:, e, 2], T["paddle_z"][:, e], end[e])
        look = sample_look(rng, blade=first_blade[e], ball=["orange", "white"][len(summary) % 2], arm="white")
        m = scene(look)
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, a.height, a.width)
        bid = m.body("ball").id
        qadr = m.jnt_qposadr[m.body_jntadr[bid]]
        summary.append(f"{title}: bounces per episode {sorted(counts)} (median shown: {counts[e]}), "
                       f"shown episode ended: {REASON_TEXT[reason[e]]}")
        print(summary[-1], flush=True)
        if writer is None:
            writer = iio.get_writer(a.out, fps=round(1 / dt), codec="libx264", quality=8, macro_block_size=8)
        card = Image.new("RGB", (W, H + BAR), (20, 20, 22))
        dr = ImageDraw.Draw(card)
        dr.text((40, (H + BAR) // 2 - 40), title, font=font, fill=(240, 240, 240))
        dr.text((40, (H + BAR) // 2), f"episode with the median bounce count of {a.envs}", font=small,
                fill=(170, 170, 170))
        dr.text((40, (H + BAR) // 2 + 28), f"{look['top']}/{look['bottom']} rubber, {look['handle']} handle, "
                f"{look['ball']} ball, blade {2000 * look['blade'][0]:.0f} x {2000 * look['blade'][1]:.0f} mm",
                font=small, fill=(170, 170, 170))
        for _ in range(int(1.2 / dt)):
            writer.append_data(np.asarray(card))
        last = None
        n_b = 0
        for k in range(min(end[e] + 1, len(T["q"]))):
            d.qpos[:5] = T["q"][k, e]
            d.qpos[5] = GRIPPER_CLOSED_ON_HANDLE
            d.qpos[qadr:qadr + 3] = T["ball"][k, e]
            d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            set_paddle(m, d, T["paddle_o"][k, e], T["paddle_R"][k, e])
            mujoco.mj_forward(m, d)
            views = []
            for cam_kw in cams:
                cam = mujoco.MjvCamera()
                for key, val in cam_kw.items():
                    setattr(cam, key, val)
                r.update_scene(d, cam)
                views.append(r.render())
            while n_b < len(ev) and ev[n_b][0] <= k:
                last = ev[n_b][1]
                n_b += 1
            frame = Image.new("RGB", (W, H + BAR), (20, 20, 22))
            frame.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, BAR))
            dr = ImageDraw.Draw(frame)
            dr.text((16, 10), title, font=font, fill=(240, 240, 240))
            info = f"t {k * dt:4.1f} s   bounces {n_b}   last bounce {100 * last:4.1f} cm" if last is not None \
                else f"t {k * dt:4.1f} s   bounces 0"
            dr.text((16, 40), info, font=small, fill=(200, 200, 200))
            if k == end[e]:
                dr.text((W - 360, 40), REASON_TEXT[reason[e]], font=small, fill=(255, 170, 90))
            img = np.asarray(frame)
            writer.append_data(img)
            if k == end[e]:
                for _ in range(int(0.8 / dt)):
                    writer.append_data(img)
    writer.close()
    with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
