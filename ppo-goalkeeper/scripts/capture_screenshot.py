"""SCREENSHOT: render the finished environment in Isaac Sim (RTX) before training.

Loads the same environment class used for training, in render mode with one environment: the Jackal keeper on its
start line, the ball resting on the penalty mark (no shot yet), the futsal goal, turf, markings, sky. Nothing has
been learned at this point; no policy is loaded.

Writes outputs/screenshots/goalkeeper_hero.png and goalkeeper_angle.png, plus a JSON check of each image: size,
pixel statistics, and where the ball, robot and goal project in the frame.
"""
import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="outputs/screenshots")
parser.add_argument("--width", type=int, default=2560)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--mode", choices=("pt", "rt"), default="pt")
parser.add_argument("--spp", type=int, default=512)
parser.add_argument("--views", default="hero,angle")
parser.add_argument("--suffix", default="")
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import faulthandler
import traceback

import imageio.v2 as iio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faulthandler.dump_traceback_later(900, repeat=True)

NAMES = {"hero": "goalkeeper_hero", "angle": "goalkeeper_angle"}


def main():
    import omni.usd

    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import make_shots
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg
    from goalkeeper.physics import field as F
    from goalkeeper.visualization import cameras as C
    from goalkeeper.visualization.render import Grabber, image_stats, set_quality

    paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=True)
    cfg = GoalkeeperEnvCfg()
    cfg.scene.num_envs = 1
    cfg.asset_paths = paths
    cfg.render_scene = True
    cfg.level = 0
    cfg.kick_delay_s = 3600.0           # ball stays on the mark
    cfg.episode_length_s = 7200.0
    env = GoalkeeperEnv(cfg)
    stage = omni.usd.get_context().get_stage()
    origin = env.scene.env_origins[0].tolist()

    shot = make_shots(0, 1, 1, env.device, cfg.vmax, cfg.amax, env.step_dt)
    shot["start"][:] = torch.tensor([F.PENALTY_X, 0.0], device=env.device)
    shot["keeper_y"][:] = 0.0
    env.forced_shots = shot
    env.reset()
    for _ in range(15):                  # let the robot settle on its wheels
        env.step(torch.zeros(1, 1, device=env.device))

    views = args.views.split(",")
    W, H = args.width, args.height
    cams, geo = {}, {}
    for v in views:
        p = C.PRESETS[v]
        _, view = C.define_camera(stage, f"/World/Cam_{v}", p["eye"], p["target"], p["focal"], W, H, origin)
        cams[v], geo[v] = f"/World/Cam_{v}", (view, p["focal"])
    settings = set_quality(args.mode, spp_total=args.spp)
    grab = Grabber(cams, W, H)
    os.makedirs(args.out_dir, exist_ok=True)

    ball = (env.ball.data.root_pos_w[0]).tolist()
    keeper = (env.robot.data.root_pos_w[0]).tolist()
    o = origin
    landmarks = {
        "ball": (ball, F.BALL_RADIUS),
        "keeper_base": (keeper, 0.25),
        "keeper_panel_top": ((o[0] + F.PANEL_FRONT_X, keeper[1], o[2] + 0.43), 0.05),
        "left_post_top": ((o[0] + 0.04, o[1] + 1.56, o[2] + 2.1), 0.06),
        "right_post_top": ((o[0] + 0.04, o[1] - 1.56, o[2] + 2.1), 0.06),
        "right_post_base": ((o[0] + 0.04, o[1] - 1.56, o[2] + 0.02), 0.06),
        "net_back_centre": ((o[0] - 1.05, o[1], o[2] + 1.0), 0.2),
    }
    report = {"settings": settings, "width": W, "height": H, "mode": args.mode, "spp": args.spp, "views": {}}
    t0 = time.time()
    # warm up: shader compilation, texture streaming, and path-tracing accumulation
    n_frames = max(args.spp // 16, 1) + 40 if args.mode == "pt" else 120
    for i in range(n_frames):
        env.sim.render()
    for v in views:
        img = None
        for _ in range(30):
            img = grab.get(v)
            if img is not None and img.std() > 3:
                break
            env.sim.render()
        if img is None:
            report["views"][v] = {"error": "no image from annotator"}
            print(f"[screenshot] {v}: NO IMAGE", flush=True)
            continue
        path = os.path.join(args.out_dir, f"{NAMES.get(v, v)}{args.suffix}.png")
        iio.imwrite(path, img)
        view, focal = geo[v]
        marks = {}
        for name, (pt, radius) in landmarks.items():
            uv, inside = C.project(view, focal, W, H, pt)
            marks[name] = {"uv": [round(uv[0]), round(uv[1])] if uv else None, "in_frame": inside,
                           "radius_px": round(C.angular_size_px(view, focal, W, pt, radius), 1)}
        report["views"][v] = {"file": path, "bytes": os.path.getsize(path), **image_stats(img), "landmarks": marks,
                              "hfov_deg": round(C.hfov_deg(focal), 1)}
        print(f"[screenshot] {v}: {path} {img.shape} std {img.std():.1f} "
              + " ".join(f"{k}={m['uv']}{'' if m['in_frame'] else '(OUT)'}" for k, m in marks.items()), flush=True)
    report["render_wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, f"screenshot_report{args.suffix}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("SCREENSHOT_DONE", json.dumps({v: r.get("file") for v, r in report["views"].items()}), flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
