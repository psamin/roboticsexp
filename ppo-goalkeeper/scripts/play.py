"""PLAY: watch a goalkeeper policy defend repeated penalty shots in the rendered Isaac Sim scene.

Every keeper movement comes from the selected policy's actions through the same drive and physics as training;
nothing is scripted or animated. Shots are seeded, so different policies face the same sequence.

  # headless on the cluster: record MP4s for PPO and both baselines
  python scripts/play.py --checkpoint outputs/checkpoints/best.pt --policies ppo,analytical,random --episodes 12
  # with a display: real-time window
  python scripts/play.py --checkpoint outputs/checkpoints/best.pt --policies ppo --gui --realtime
"""
import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
parser.add_argument("--policies", default="ppo")
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--level", type=int, default=2)
parser.add_argument("--seed", type=int, default=3000)
parser.add_argument("--views", default="front,side")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--video_dir", default="outputs/videos")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
args.enable_cameras = True
app = AppLauncher(args).app

import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = {"ppo": "PPO policy (learned)", "analytical": "Analytical baseline", "random": "Random actions"}


def overlay(img, lines, flash=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(16, img.shape[0] // 30))
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(28, img.shape[0] // 14))
    except OSError:
        font = ImageFont.load_default(size=max(16, img.shape[0] // 30))
        big = ImageFont.load_default(size=max(28, img.shape[0] // 14))
    pad, lh = 14, int(font.size * 1.35)
    w = max(d.textlength(t, font=font) for t in lines) + 2 * pad
    d.rectangle([16, 16, 16 + w, 16 + 2 * pad + lh * len(lines) - (lh - font.size)], fill=(0, 0, 0, 140))
    for i, t in enumerate(lines):
        d.text((16 + pad, 16 + pad + i * lh), t, fill=(255, 255, 255, 255), font=font)
    if flash:
        tw = d.textlength(flash, font=big)
        d.text(((img.shape[1] - tw) / 2, img.shape[0] * 0.08), flash, font=big, fill=(255, 255, 255, 235),
               stroke_width=3, stroke_fill=(0, 0, 0, 200))
    return np.asarray(im)


def main():
    import imageio.v2 as iio
    import omni.usd

    from goalkeeper import config
    from goalkeeper.controllers.baselines import AnalyticalKeeper, RandomKeeper
    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import make_shots
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg
    from goalkeeper.physics import outcomes
    from goalkeeper.physics.shots import LEVEL_NAMES
    from goalkeeper.rl.ppo import PPO
    from goalkeeper.visualization import cameras as C
    from goalkeeper.visualization.render import Grabber, set_quality

    names = args.policies.split(",")
    agent, c = None, config.load()
    if "ppo" in names:
        agent, extra = PPO.load(args.checkpoint, device="cuda:0", load_optimizer=False)
        c = extra.get("config") or c
    c["env"]["num_envs"] = 1
    cfg = config.apply_env(GoalkeeperEnvCfg(), c)
    cfg.asset_paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=True)
    cfg.render_scene = True
    cfg.level = args.level
    cfg.kick_delay_s = 0.5
    cfg.hold_after_outcome_s = 0.9
    env = GoalkeeperEnv(cfg)
    stage = omni.usd.get_context().get_stage()
    origin = env.scene.env_origins[0].tolist()
    views = args.views.split(",")
    for v in views:
        p = C.PRESETS[v]
        C.define_camera(stage, f"/World/Cam_{v}", p["eye"], p["target"], p["focal"], args.width, args.height, origin)
    set_quality("rt")
    grab = None if args.no_video else Grabber({v: f"/World/Cam_{v}" for v in views}, args.width, args.height)
    os.makedirs(args.video_dir, exist_ok=True)

    shots = make_shots(args.level, args.episodes, args.seed, env.device, cfg.vmax, cfg.amax, env.step_dt)
    pick = lambda i: {k: v[i:i + 1] for k, v in shots.items()}
    gen = torch.Generator(device=env.device).manual_seed(args.seed)
    summary = {}
    for name in names:
        policy = {"ppo": (agent.act_deterministic if agent else None), "analytical": AnalyticalKeeper(env.drive),
                  "random": RandomKeeper(gen)}[name]
        writer = None
        if grab is not None:
            path = os.path.join(args.video_dir, f"goalkeeper_{name}_{LEVEL_NAMES[args.level]}.mp4")
            writer = iio.get_writer(path, fps=round(1 / env.step_dt), codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        env.forced_shots = pick(0)
        obs, _ = env.reset()
        env.pop_completed()
        env.forced_shots = pick(1 % args.episodes)
        done_eps, saves, on_target, flash, flash_left, results = 0, 0, 0, None, 0, []
        counted = False                          # current shot already added to the on-screen tally
        for _ in range(30):                      # let textures stream in before the first frame
            env.sim.render()
        while done_eps < args.episodes:
            t0 = time.time()
            obs, *_ = env.step(policy(obs["policy"]))
            rec = env.pop_completed()
            if rec is not None:
                oc = int(rec["outcome"][0]); tgt = bool(rec["on_target"][0])
                results.append({"outcome": outcomes.OUTCOME_NAMES[oc], "on_target": tgt})
                if not counted:
                    on_target += tgt
                    saves += tgt and oc == outcomes.SAVE
                counted = False
                done_eps += 1
                env.forced_shots = pick((done_eps + 1) % args.episodes)
            # tally and flash the result as soon as it is decided, while the ball is still in play
            live = int(env.outcome[0])
            if live != outcomes.NONE and not counted and rec is None:
                tgt = bool(env.shot["on_target"][0])
                on_target += tgt
                saves += tgt and live == outcomes.SAVE
                counted = True
                flash, flash_left = {outcomes.SAVE: "SAVE", outcomes.GOAL: "GOAL", outcomes.MISS: "MISS",
                                     outcomes.TIMEOUT: "TIME"}[live], int(1.0 / env.step_dt)
            if grab is not None:
                env.sim.render()
                frames = [grab.get(v) for v in views]
                if all(f is not None for f in frames):
                    shot_no = min(done_eps + 1, args.episodes)
                    lines = [LABELS[name], f"shot {shot_no}/{args.episodes}  ({LEVEL_NAMES[args.level]} shots)",
                             f"saves {saves}/{on_target} on target"]
                    frame = np.concatenate([overlay(frames[0], lines, flash)] + frames[1:], axis=1)
                    writer.append_data(frame)
            if flash_left > 0:
                flash_left -= 1
                if flash_left == 0:
                    flash = None
            if args.realtime:
                time.sleep(max(0.0, env.step_dt - (time.time() - t0)))
        if writer is not None:
            writer.close()
        summary[name] = {"episodes": args.episodes, "on_target": on_target, "saves": saves,
                         "save_rate": saves / max(on_target, 1), "results": results}
        print(f"[play] {name}: saves {saves}/{on_target} on target", flush=True)
    with open(os.path.join(args.video_dir, f"play_summary_{LEVEL_NAMES[args.level]}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("PLAY_DONE", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
