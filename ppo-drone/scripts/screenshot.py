"""SCREENSHOT: an Isaac Sim (RTX) render of the minimal scene.

Before training (no policy): the drone placed slightly above the floor, the target sphere at 1 m.
  python scripts/screenshot.py --out outputs/screenshots/drone_ppo_setup.png
After training: the PPO policy flies from a randomised start for --fly_s seconds, then the frame is rendered.
  python scripts/screenshot.py --checkpoint outputs/run/checkpoints/best.pt --out outputs/screenshots/drone_ppo_trained.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--out", default="outputs/screenshots/drone_ppo_setup.png")
p.add_argument("--checkpoint", default=None)
p.add_argument("--camera", default="closeup", help="closeup (frames the drone), or a preset in scene.CAMERAS")
p.add_argument("--width", type=int, default=2560)
p.add_argument("--height", type=int, default=1440)
p.add_argument("--mode", choices=("pt", "rt"), default="pt")
p.add_argument("--spp", type=int, default=256)
p.add_argument("--fly_s", type=float, default=5.0)
p.add_argument("--dump_settings", action="store_true", help="print renderer settings related to fog/exposure")
args, app = config.launch(p, cameras=True)


def main():
    import imageio.v2 as iio
    import numpy as np
    import omni.usd
    import torch

    from drone_ppo import scene as S
    from drone_ppo import task as T
    from drone_ppo.ppo import PPO

    c = config.load()
    agent = None
    if args.checkpoint:
        agent, extra = PPO.load(args.checkpoint, "cuda:0", load_optimizer=False)
        c = extra.get("config") or c
    env = config.make_env(c, 1, stage=1, render_scene=True)
    dev = env.device
    if agent is None:   # setup shot: drone just above the floor, target at the 1 m hover point (slightly offset)
        cond = {"stage": torch.zeros(1, dtype=torch.long, device=dev),
                "start": torch.tensor([[-0.12, -0.06, 0.16]], device=dev),
                "target": torch.tensor([[0.12, 0.10, 1.0]], device=dev),
                "rpy": torch.zeros(1, 3, device=dev), "start_vel": torch.zeros(1, 3, device=dev),
                "wind_frac": torch.zeros(1, 3, device=dev)}
        policy, steps = (lambda o: torch.zeros(1, 4, device=dev)), 3      # hover thrust, a few frames
    else:               # trained shot: a randomised start, flown by the learned policy
        cond = {"stage": torch.ones(1, dtype=torch.long, device=dev),
                "start": torch.tensor([[-0.15, -0.12, 0.8]], device=dev),
                "target": torch.tensor([[0.2, 0.12, 0.5]], device=dev),
                "rpy": torch.tensor([[0.12, -0.1, 0.6]], device=dev), "start_vel": torch.zeros(1, 3, device=dev),
                "wind_frac": torch.zeros(1, 3, device=dev)}
        policy, steps = agent.act_deterministic, int(args.fly_s / env.step_dt)
    env.forced = cond
    obs, _ = env.reset()
    for k in range(steps):
        obs, *_ = env.step(policy(obs["policy"]))
        # trained close-up: stop while the drone is closing in, so drone and target sphere are both visible
        if agent is not None and args.camera == "closeup" and k * env.step_dt > 0.4 and \
                float((env.robot.data.root_pos_w[0] - env.target_w[0]).norm()) < 0.12:
            break
    origin = env.scene.env_origins[0].tolist()
    cam_name = args.camera
    if cam_name == "closeup":
        o = torch.tensor(origin, device=dev)
        p_d = env.robot.data.root_pos_w[0] - o
        # frame the drone and its shadow on the floor (the shadow shows it is airborne)
        # camera above the drone looking down ~20 deg: drone in the upper third, its shadow in the lower part
        # (framings solved with scene.project for a 16:9 frame; see the screenshot JSON for the measured positions)
        x, y, z = p_d.tolist()
        if agent is None:
            cam = {"eye": (x + 0.47, y - 0.37, z + 0.19), "target": (x, y, 0.72 * z), "focal": 26.0}
        else:
            mid = (0.5 * (p_d + env.target_w[0] - o)).tolist()
            cam = {"eye": (x + 0.85, y - 0.68, z + 0.20), "target": (mid[0] - 0.02, mid[1] - 0.02, 0.62 * z),
                   "focal": 17.0}
    else:
        cam = S.CAMERAS[cam_name]
    stage = omni.usd.get_context().get_stage()
    view = S.define_camera(stage, "/World/Cam", cam["eye"], cam["target"], cam["focal"], args.width, args.height, origin)
    S.set_quality(args.mode, spp_total=args.spp)
    if args.dump_settings:
        import carb
        def walk(d, path):
            if isinstance(d, dict):
                for k, v in d.items():
                    walk(v, f"{path}/{k}")
            elif any(w in path.lower() for w in ("fog", "atmos", "vignet", "falloff", "horizon", "distance")):
                print(f"[settings] {path} = {d}", flush=True)
        walk(carb.settings.get_settings().get("/rtx"), "/rtx")
    grab = S.Grabber({"cam": "/World/Cam"}, args.width, args.height)
    for _ in range((args.spp // 16 + 40) if args.mode == "pt" else 120):
        env.sim.render()
    img = None
    for _ in range(30):
        img = grab.get("cam")
        if img is not None and img.std() > 3:
            break
        env.sim.render()
    if img is None:
        raise RuntimeError("no image from the annotator")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    iio.imwrite(args.out, img)
    drone = env.robot.data.root_pos_w[0].tolist()
    target = env.target_w[0].tolist()
    marks = {}
    for name, pt in (("drone", drone), ("target", target)):
        uv, inside = S.project(view, cam["focal"], args.width, args.height, pt)
        marks[name] = {"world": [round(x, 3) for x in pt], "uv": [round(x) for x in uv] if uv else None, "in_frame": inside}
    rep = {"file": args.out, "shape": list(img.shape), "mean": float(img.mean()), "std": float(img.std()),
           "mode": args.mode, "spp": args.spp, "camera": cam_name, "policy": "ppo" if agent else "none (before training)",
           "distance_to_target_m": float(np.linalg.norm(np.array(drone) - np.array(target))), "landmarks": marks}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    print("[screenshot]", json.dumps(rep), flush=True)
    print("SCREENSHOT_DONE", flush=True)


config.run_main(main)
