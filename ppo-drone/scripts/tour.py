"""TOUR: the trained policy flies a sequence of waypoints; recorded from a high diagonal follow camera.

The policy observes only the current target relative to itself, so a route is just a sequence of targets: when the
drone has stayed within --reach_cm of the current waypoint for --hold_s seconds (or --max_s has passed), the next
waypoint becomes the target. All motion comes from the policy's thrust/torque actions through the physics.

  python scripts/tour.py --checkpoint outputs/checkpoints/best.pt --video outputs/videos/drone_ppo_tour.mp4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
p.add_argument("--video", default="outputs/videos/drone_ppo_tour.mp4")
p.add_argument("--style", choices=("isaac", "white", "studio"), default="isaac")
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--reach_cm", type=float, default=6.0)
p.add_argument("--hold_s", type=float, default=0.4)
p.add_argument("--max_s", type=float, default=4.0)
p.add_argument("--render_passes", type=int, default=3, help="renders per video frame; >1 removes motion ghosting")
args, app = config.launch(p, cameras=True)

# env-frame waypoints (m): a loop at varying heights, hops of 0.6-0.9 m (inside the trained target offsets)
ROUTE = [(0.0, 0.0, 1.0), (0.6, 0.0, 1.2), (0.3, 0.6, 0.8), (-0.4, 0.5, 1.25), (-0.7, -0.1, 0.85),
         (-0.2, -0.65, 1.15), (0.55, -0.5, 0.8), (0.0, 0.0, 1.0)]


def main():
    import imageio.v2 as iio
    import numpy as np
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    from drone_ppo import scene as S
    from drone_ppo.ppo import PPO

    agent, extra = PPO.load(args.checkpoint, "cuda:0", load_optimizer=False)
    c = extra.get("config") or config.load()
    env = config.make_env(c, 1, stage=1, render_scene=True, scene_style=args.style, episode_length_s=300.0)
    dev = env.device
    stage = omni.usd.get_context().get_stage()
    o = np.asarray(env.scene.env_origins[0].tolist())
    route = [np.asarray(w) for w in ROUTE[1:]]
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Target")).MakeInvisible()     # waypoints drawn below instead

    def material(path, rgb, rough=0.35):
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m
    mats = {"next": material("/World/Looks/WpNext", (0.03, 0.85, 0.25)),
            "later": material("/World/Looks/WpLater", (0.14, 0.36, 0.2)),
            "done": material("/World/Looks/WpDone", (0.12, 0.12, 0.13)),
            "trail": material("/World/Looks/Trail", (0.16, 0.47, 0.84), 0.6)}
    spheres = []
    for i, w in enumerate(ROUTE[1:]):
        sp = UsdGeom.Sphere.Define(stage, f"/World/Waypoints/W{i}")
        sp.CreateRadiusAttr(0.03)
        UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(*(o + np.asarray(w))))
        UsdGeom.PrimvarsAPI(sp).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        spheres.append(sp)

    def paint(cur):
        for i, sp in enumerate(spheres):
            key = "next" if i == cur else ("done" if i < cur else "later")
            UsdShade.MaterialBindingAPI.Apply(sp.GetPrim()).Bind(mats[key])
    trail = UsdGeom.BasisCurves.Define(stage, "/World/Trail")
    trail.CreateTypeAttr(UsdGeom.Tokens.linear)
    trail.CreateWidthsAttr(Vt.FloatArray([0.006]))
    trail.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    UsdShade.MaterialBindingAPI.Apply(trail.GetPrim()).Bind(mats["trail"])
    UsdGeom.PrimvarsAPI(trail).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

    W, H = args.width, args.height
    S.define_camera(stage, "/World/CamTour", (2, -2, 3), (0, 0, 1), 28.0, W, H, o.tolist())
    S.set_quality("rt", auto_exposure=(args.style == "studio"))
    grab = S.Grabber({"tour": "/World/CamTour"}, W, H)
    cam_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/CamTour")).GetOrderedXformOps()[0]

    start = torch.tensor([ROUTE[0]], device=dev, dtype=torch.float32)
    env.forced = {"stage": torch.ones(1, dtype=torch.long, device=dev), "start": start, "target": start.clone(),
                  "rpy": torch.zeros(1, 3, device=dev), "start_vel": torch.zeros(1, 3, device=dev),
                  "wind_frac": torch.zeros(1, 3, device=dev), "start_ang_vel": torch.zeros(1, 3, device=dev)}
    obs, _ = env.reset()
    cur, held, t_wp = 0, 0, 0
    env.target_w[0] = torch.tensor(o + route[0], device=dev, dtype=torch.float32)
    paint(cur)
    look = o + np.asarray(ROUTE[0])
    pts = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", H // 34)
    except OSError:
        font = ImageFont.load_default(size=H // 34)
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    mk = lambda path: iio.get_writer(path, fps=round(1 / env.step_dt), codec="libx264", quality=8,
                                     macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    writer, clean = mk(args.video), mk(os.path.splitext(args.video)[0] + "_clean.mp4")
    for _ in range(20):
        env.sim.render()
    log, steps, k = [], 0, 0
    hold_steps, max_steps = int(args.hold_s / env.step_dt), int(args.max_s / env.step_dt)
    while cur < len(route):
        obs, _, term, _, _ = env.step(agent.act_deterministic(obs["policy"]))
        if bool(term[0]):
            log.append({"waypoint": cur, "event": "failed"})
            break
        pos = env.robot.data.root_pos_w[0].cpu().numpy().astype(float)
        d = float(np.linalg.norm(pos - (o + route[cur])))
        held = held + 1 if d < args.reach_cm / 100 else 0
        t_wp += 1
        if held >= hold_steps or t_wp >= max_steps:
            log.append({"waypoint": cur, "reached": held >= hold_steps, "time_s": t_wp * env.step_dt,
                        "final_dist_cm": 100 * d})
            cur, held, t_wp = cur + 1, 0, 0
            if cur < len(route):
                env.target_w[0] = torch.tensor(o + route[cur], device=dev, dtype=torch.float32)
            paint(cur)
        pts.append(Gf.Vec3f(*pos))
        trail.CreatePointsAttr(Vt.Vec3fArray(pts[-1500:]))
        trail.CreateCurveVertexCountsAttr(Vt.IntArray([len(pts[-1500:])]))
        look = 0.93 * look + 0.07 * pos
        eye = look + np.array([1.05, -1.05, 1.55])                     # ~48 degrees down, diagonal
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, look)), Gf.Vec3d(0, 0, 1))
        cam_xf.Set(view.GetInverse())
        for _ in range(args.render_passes):
            env.sim.render()
        img = grab.get("tour")
        if img is not None and k > 3:
            clean.append_data(img)
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im, "RGBA")
            lines = ["PPO policy (learned) flying a waypoint route",
                     f"waypoint {min(cur + 1, len(route))}/{len(route)}   distance {100 * d:5.1f} cm"]
            pad, lh = 12, int(font.size * 1.35)
            w = max(dr.textlength(t, font=font) for t in lines) + 2 * pad
            dr.rectangle([14, 14, 14 + w, 14 + 2 * pad + lh * len(lines) - (lh - font.size)], fill=(0, 0, 0, 130))
            for i, t in enumerate(lines):
                dr.text((14 + pad, 14 + pad + i * lh), t, fill=(255, 255, 255, 255), font=font)
            writer.append_data(np.asarray(im))
        k += 1
        steps += 1
    for _ in range(int(0.8 / env.step_dt)):                                # a short hover at the end
        obs, *_ = env.step(agent.act_deterministic(obs["policy"]))
        env.sim.render()
        img = grab.get("tour")
        if img is not None:
            clean.append_data(img)
            writer.append_data(img)
    writer.close()
    clean.close()
    with open(os.path.splitext(args.video)[0] + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "route_env_frame_m": ROUTE, "waypoints": log,
                   "seconds": steps * env.step_dt}, f, indent=1)
    for e in log:
        print("[tour]", json.dumps(e), flush=True)
    print("TOUR_DONE", flush=True)


config.run_main(main)
