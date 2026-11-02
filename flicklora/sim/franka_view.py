"""Diagnostic: does the Franka render once physics is playing? Two variants: asset as-is, and instancing disabled."""
import os
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
import imageio.v2 as iio
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, UsdLux

try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import franka as FR
import scene as S

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for mode in ("asis", "noinstance"):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1500)
    g = UsdGeom.Cube.Define(stage, "/World/Floor")
    g.CreateSizeAttr(1.0)
    UsdGeom.Xformable(g).AddTranslateOp().Set((0.3, 0, -0.01))
    UsdGeom.Xformable(g).AddScaleOp().Set((1.4, 1.1, 0.02))
    FR.add_franka(stage, "/World/Franka", (0, 0, 0))
    if mode == "noinstance":
        n = 0
        for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                n += 1
        print(f"[{mode}] made {n} prims non-instanceable", flush=True)
    for _ in range(30):
        app.update()
    S.add_camera(stage, "/World/Cam", (1.4, 1.0, 0.9), (0.2, 0.0, 0.40), focal=24.0)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 240, rendering_dt=1 / 30)
    rp = rep.create.render_product("/World/Cam", (640, 480))
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach([rp])
    world.reset()
    t0, d = time.time(), None
    while time.time() - t0 < 240:
        for _ in range(8):
            world.step(render=False)
        world.render()
        d = an.get_data()
        if d is not None and getattr(d, "size", 0) > 0 and np.asarray(d)[..., :3].std() > 2:
            for _ in range(10):
                for _ in range(8):
                    world.step(render=False)
                world.render()
            d = an.get_data()
            break
    if d is not None and getattr(d, "size", 0) > 0:
        img = np.asarray(d)[:, :, :3]
        iio.imwrite(os.path.join(out, f"franka_{mode}.png"), img)
        # crude check: how much of the image differs from the background colours (sky + floor)?
        print(f"[{mode}] wrote image after {time.time()-t0:.0f}s, pixel std {img.std():.1f}", flush=True)
    else:
        print(f"[{mode}] NO IMAGE after {time.time()-t0:.0f}s", flush=True)
    an.detach()
    world.stop()
    World.clear_instance()
app.close()
