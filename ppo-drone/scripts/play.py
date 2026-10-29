"""PLAY: the trained policy flies the Crazyflie in the rendered scene; records an MP4.

Every motion comes from the policy's thrust/torque actions through the physics; nothing is scripted.
Left: a camera that follows the drone. Right: a fixed wide view.

  python scripts/play.py --checkpoint outputs/run/checkpoints/best.pt --video outputs/videos/drone_ppo_demo.mp4
  python scripts/play.py --checkpoint outputs/run/checkpoints/best.pt --gui        # watch live with a display
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="outputs/run/checkpoints/best.pt")
p.add_argument("--policy", choices=("ppo", "pd", "random"), default="ppo")
p.add_argument("--episodes", type=int, default=4)
p.add_argument("--episode_s", type=float, default=7.0)
p.add_argument("--suite", type=int, default=1, help="0 hover, 1 reach, 2 robust")
p.add_argument("--seed", type=int, default=3000)
p.add_argument("--video", default="outputs/videos/drone_ppo_demo.mp4")
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--view", choices=("close", "split"), default="close",
               help="close: one camera following the drone; split: follow + wide side by side")
p.add_argument("--no_overlay", action="store_true")
p.add_argument("--outline", action="store_true", help="draw an orange outline around the drone")
p.add_argument("--style", choices=("isaac", "white", "studio"), default="isaac")
p.add_argument("--gui", action="store_true")
p.add_argument("--render_passes", type=int, default=3, help="renders per video frame; >1 removes motion ghosting")
args, app = config.launch(p, cameras=True)


def overlay(img, lines):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(16, img.shape[0] // 32))
    except OSError:
        font = ImageFont.load_default(size=max(16, img.shape[0] // 32))
    pad, lh = 12, int(font.size * 1.35)
    w = max(d.textlength(t, font=font) for t in lines) + 2 * pad
    d.rectangle([14, 14, 14 + w, 14 + 2 * pad + lh * len(lines) - (lh - font.size)], fill=(0, 0, 0, 130))
    for i, t in enumerate(lines):
        d.text((14 + pad, 14 + pad + i * lh), t, fill=(255, 255, 255, 255), font=font)
    return np.asarray(im)


def main():
    global np
    import imageio.v2 as iio
    import numpy as np
    import omni.usd
    import torch
    from pxr import Gf, UsdGeom

    from drone_ppo import scene as S
    from drone_ppo import task as T
    from drone_ppo.evaluation import make_conditions
    from drone_ppo.ppo import PPO

    c = config.load()
    agent = None
    if args.policy == "ppo":
        agent, extra = PPO.load(args.checkpoint, "cuda:0", load_optimizer=False)
        c = extra.get("config") or c
    env = config.make_env(c, 1, stage=args.suite, render_scene=True, scene_style=args.style)
    dev = env.device
    if args.policy == "ppo":
        policy, label = agent.act_deterministic, "PPO policy (learned)"
    elif args.policy == "pd":
        policy, label = T.PDHoverController(env.mass, env.inertia_diag, env.weight), "PD controller (baseline)"
    else:
        policy, label = T.RandomPolicy(), "random actions"
    stage = omni.usd.get_context().get_stage()
    origin = env.scene.env_origins[0].tolist()
    W, H = args.width, args.height
    S.define_camera(stage, "/World/CamFollow", (1.0, -0.8, 1.2), (0, 0, 1.0), 24.0 if args.view == "close" else 32.0,
                    W, H, origin)
    wide = S.CAMERAS["wide"]
    S.define_camera(stage, "/World/CamWide", wide["eye"], wide["target"], wide["focal"], W, H, origin)
    S.set_quality("rt", auto_exposure=(args.style == "studio"))
    grab = S.Grabber({"follow": "/World/CamFollow", "wide": "/World/CamWide"}, W, H, segment=args.outline)
    follow_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/CamFollow")).GetOrderedXformOps()[0]
    conds = make_conditions(args.suite, args.episodes, args.seed, dev)
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    mk = lambda path: iio.get_writer(path, fps=round(1 / env.step_dt), codec="libx264", quality=8, macro_block_size=8,
                                     ffmpeg_params=["-pix_fmt", "yuv420p"])
    writer = mk(args.video)
    clean = mk(os.path.splitext(args.video)[0] + "_clean.mp4") if args.view == "close" else None
    o = np.asarray(origin)
    log = []
    for ep in range(args.episodes):
        env.forced = {k: v[ep:ep + 1] for k, v in conds.items()}
        obs, _ = env.reset()
        env.pop_completed()
        look = env.robot.data.root_pos_w[0].cpu().numpy()
        for _ in range(20):
            env.sim.render()
        dists = []
        for k in range(int(args.episode_s / env.step_dt)):
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
            pos = env.robot.data.root_pos_w[0].cpu().numpy()
            tgt = env.target_w[0].cpu().numpy()
            if args.view == "close":    # drone centred, the target sphere in shot, camera slightly above
                look = 0.85 * look + 0.15 * (0.75 * pos + 0.25 * tgt)
                eye, aim = look + np.array([0.45, -0.36, 0.10]), look
            else:
                look = 0.9 * look + 0.1 * (0.5 * pos + 0.5 * tgt)
                eye, aim = look + np.array([0.55, -0.45, 0.05]), look
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, aim)), Gf.Vec3d(0, 0, 1))
            follow_xf.Set(view.GetInverse())
            for _ in range(args.render_passes):
                env.sim.render()
            d = float(np.linalg.norm(pos - tgt))
            dists.append(d)
            f, wv = grab.get("follow"), grab.get("wide")
            if f is not None and args.outline:
                f = S.outline(f, grab.mask("follow", "/World/envs/env_0/Robot"))
            if f is not None and wv is not None:
                lines = [label, f"episode {ep + 1}/{args.episodes}   t = {k * env.step_dt:4.1f} s",
                         f"distance to target {100 * d:5.1f} cm"]
                frame = f if args.no_overlay else overlay(f, lines)
                writer.append_data(frame if args.view == "close" else np.concatenate([frame, wv], axis=1))
                if clean is not None:
                    clean.append_data(f)
            if bool(term[0]) or bool(trunc[0]):
                break
        log.append({"episode": ep, "final_dist_m": dists[-1], "min_dist_m": min(dists),
                    "start": conds["start"][ep].tolist(), "target": conds["target"][ep].tolist(),
                    "failed": bool(term[0])})
        print(f"[play] episode {ep + 1}: final distance {100 * dists[-1]:.1f} cm, failed={bool(term[0])}", flush=True)
    writer.close()
    if clean is not None:
        clean.close()
    with open(os.path.splitext(args.video)[0] + ".json", "w") as fh:
        json.dump({"policy": args.policy, "checkpoint": args.checkpoint, "episodes": log}, fh, indent=1)
    print("PLAY_DONE", flush=True)


config.run_main(main)
