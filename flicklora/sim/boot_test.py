"""Minimal Isaac Sim boot + render test: python boot_test.py [experience.kit]"""
import sys
import time

from isaacsim import SimulationApp

exp = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ""
cfg = {"headless": True, "width": 256, "height": 256}
t0 = time.time()
app = SimulationApp(cfg, experience=exp) if exp else SimulationApp(cfg)
print(f"BOOT_OK {time.time()-t0:.1f}s exp={exp or 'default'}", flush=True)
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

stage = omni.usd.get_context().get_stage()
UsdGeom.Cube.Define(stage, "/World/Cube").CreateSizeAttr(0.1)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000)
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
UsdGeom.Xformable(cam).AddTransformOp().Set(
    Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.4, 0.4, 0.3), Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 1)).GetInverse())
rp = rep.create.render_product("/World/Cam", (128, 128))
an = rep.AnnotatorRegistry.get_annotator("rgb")
an.attach([rp])
for i in range(40):
    app.update()
d = np.asarray(an.get_data())
print("RENDER_OK", d.shape, float(d[..., :3].mean()) if d.size else "empty", flush=True)
app.close()
