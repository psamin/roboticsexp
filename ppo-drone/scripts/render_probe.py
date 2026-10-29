"""Diagnostic: why does the floor darken with distance? Renders the setup view under several variants and prints
floor brightness near/mid/far, every light prim on the stage, and lighting-mode settings."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--out", default="outputs/probe")
args, app = config.launch(p, cameras=True)


def main():
    import carb
    import imageio.v2 as iio
    import omni.usd
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    from drone_ppo import scene as S

    c = config.load()
    env = config.make_env(c, 1, stage=0, render_scene=True)
    dev = env.device
    env.forced = {"stage": torch.zeros(1, dtype=torch.long, device=dev),
                  "start": torch.tensor([[-0.12, -0.06, 0.30]], device=dev),
                  "target": torch.tensor([[0.12, 0.10, 1.0]], device=dev), "rpy": torch.zeros(1, 3, device=dev),
                  "start_vel": torch.zeros(1, 3, device=dev), "wind_frac": torch.zeros(1, 3, device=dev)}
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(1, 4, device=dev))
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
            vis = UsdGeom.Imageable(prim).ComputeVisibility() if prim.IsA(UsdGeom.Imageable) else "?"
            print(f"[light] {prim.GetPath()} {prim.GetTypeName()} active={prim.IsActive()} vis={vis} "
                  f"intensity={prim.GetAttribute('inputs:intensity').Get()}", flush=True)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/ground")):
        vis = UsdGeom.Imageable(prim).ComputeVisibility() if prim.IsA(UsdGeom.Imageable) else "-"
        print(f"[ground] {prim.GetPath()} {prim.GetTypeName()} vis={vis}", flush=True)

    def walk(d, path):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}/{k}")
        elif any(w in path.lower() for w in ("lightingmode", "viewlight", "cameralight", "ambient", "lightrig",
                                             "defaultlight", "domelight", "sceneDb")):
            print(f"[setting] {path} = {d}", flush=True)
    walk(carb.settings.get_settings().get("/rtx"), "/rtx")

    W, H = 640, 360
    cam = S.CAMERAS["setup"]
    S.define_camera(stage, "/World/Cam", cam["eye"], cam["target"], cam["focal"], W, H, env.scene.env_origins[0].tolist())
    grab = S.Grabber({"cam": "/World/Cam"}, W, H)
    os.makedirs(args.out, exist_ok=True)

    def shoot(name, mode="pt"):
        S.set_quality(mode, spp_total=64)
        for _ in range(60):
            env.sim.render()
        img = grab.get("cam")
        iio.imwrite(os.path.join(args.out, f"{name}.png"), img)
        rows = {r: float(img[int(r * H), :, :].mean()) for r in (0.95, 0.8, 0.65, 0.56, 0.52, 0.3)}
        print(f"[probe] {name:12s} " + " ".join(f"y{r:.2f}={v:6.1f}" for r, v in rows.items()), flush=True)

    from pxr import Sdf, UsdShade
    sh = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Floor/PBR"))
    for rough in (0.9, 0.5, 0.25, 0.0):
        sh.GetInput("roughness").Set(rough)
        shoot(f"rough_{rough}")
    sh.GetInput("roughness").Set(0.5)
    sh.GetInput("useSpecularWorkflow").Set(0)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    shoot("ior1_r0.5")
    print("PROBE_DONE", flush=True)


config.run_main(main)
