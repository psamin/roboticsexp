"""Phase A (+ the first Phase B check): does the screw scene load, does it simulate, and can CONTACT drive the nut?

  --mode inspect       measure the CAD: thread pitch from the bolt helix, nut across-flats, lengths -> geometry.json
  --mode original      open the reference static_bolt.usda as-is (adds only a light + camera), record its driven
                       screw motion, and check rotation/travel coupling           -> original.mp4, original.json
  --mode ours_driven   our restyled horizontal-bolt scene, prismatic joint driven -> ours_driven.mp4
  --mode ours_passive  same scene, NO drive anywhere on the nut; a force-limited 1-DOF finger tip sweeps past the
                       hex corners. Pass = nut spins and travels only because of contact -> ours_passive.mp4/.json

Run with Isaac Sim 4.5 python (headless). Writes into --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["inspect", "original", "ours_driven", "ours_passive"])
ap.add_argument("--ref", required=True, help="reference repo dir (has static_bolt.usda, hex_bolt/)")
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--physics_hz", type=float, default=1200.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", type=str, default="960x540")
ap.add_argument("--push_speed", type=float, default=1.2)
ap.add_argument("--coulomb", type=float, default=None, help="thread Coulomb friction torque N*m (default: auto)")
ap.add_argument("--viscous", type=float, default=None, help="thread viscous friction N*m*s/rad (default: auto)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
W, H = map(int, args.res.split("x"))
BOLT = os.path.join(args.ref, "hex_bolt", "bolt.usd")
NUT = os.path.join(args.ref, "hex_bolt", "nut.usd")


def log(*a):
    print(f"[phase_a:{args.mode}]", *a, flush=True)


def dump(name, obj):
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
    log("wrote", name)


# ---------------------------------------------------------------- inspect needs only pxr, but pxr ships in isaacsim
if args.mode == "inspect":
    from isaacsim import SimulationApp      # makes pxr importable from the pip install
    app = SimulationApp({"headless": True})
    sys.path.insert(0, os.path.dirname(__file__))
    from scene import measure, nut_mass_inertia
    g, extra = measure(None, BOLT, NUT)
    m, izz, ixx = nut_mass_inertia(g)
    d = dict(g.__dict__, nut_mass_kg=m, nut_Izz=izz, nut_Ixx=ixx, **extra,
             gearing_deg_per_m=360.0 / g.pitch, reference_file_gearing=-240000,
             reference_file_implied_pitch_mm=360.0 / 240000 * 1000)
    log(json.dumps(d, indent=1, default=float))
    dump("geometry.json", d)
    app.close()
    sys.exit(0)

# ---------------------------------------------------------------- everything else needs the renderer
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": "RayTracedLighting",
                     "anti_aliasing": 3})
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
    API = "isaacsim.core (4.5)"
except ImportError:  # older namespace
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    API = "omni.isaac.core"
log("core API:", API)
sys.path.insert(0, os.path.dirname(__file__))
import scene as S

dt = 1.0 / args.physics_hz
substeps = max(1, int(round(args.physics_hz / args.fps)))
frames, stats = [], {"mode": args.mode, "physics_hz": args.physics_hz, "fps": args.fps, "api": API}


class Recorder:
    def __init__(self, cam_path):
        self.rp = rep.create.render_product(cam_path, (W, H))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot.attach([self.rp])

    def grab(self):
        d = self.annot.get_data()
        if d is None or getattr(d, "size", 0) == 0:
            return None
        return np.asarray(d)[:, :, :3].copy()


def write_video(name, imgs, fps):
    import imageio.v2 as iio
    path = os.path.join(args.out, name)
    with iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for im in imgs:
            w.append_data(im)
    iio.imwrite(path.replace(".mp4", "_frame0.png"), imgs[0])
    iio.imwrite(path.replace(".mp4", "_last.png"), imgs[-1])
    log("wrote", name, len(imgs), "frames")


def wait_loading():
    for _ in range(50):
        app.update()
    try:
        from isaacsim.core.utils.stage import is_stage_loading
    except ImportError:
        from omni.isaac.core.utils.stage import is_stage_loading
    t0 = time.time()
    while is_stage_loading() and time.time() - t0 < 120:
        app.update()


# ---------------------------------------------------------------- build / open the stage
ctx = omni.usd.get_context()
if args.mode == "original":
    ctx.open_stage(os.path.join(args.ref, "static_bolt.usda"))
    wait_loading()
    stage = ctx.get_stage()
    # render-only additions: the file relies on the editor's default light, which a headless run does not have
    if not any(p.HasAPI(UsdLux.LightAPI) or p.GetTypeName().endswith("Light") for p in stage.Traverse()):
        UsdLux.DomeLight.Define(stage, "/World/RecDome").CreateIntensityAttr(1000)
        dl = UsdLux.DistantLight.Define(stage, "/World/RecSun")
        dl.CreateIntensityAttr(1500)
    # the file's own saved Perspective view
    S.add_camera(stage, "/World/RecCam", eye=(0.0457, 0.0405, 0.0067), target=(0.0031, -0.0047, 0.0149),
                 focal=24.0)
    art_path, cam = "/World/NutAssembly", "/World/RecCam"
    gearing_expected = 240000.0
else:
    ctx.new_stage()
    wait_loading()
    stage = ctx.get_stage()
    g, _ = S.measure(None, BOLT, NUT)
    q_max = g.bolt_zmax - g.nut_zmin            # nut bottom at the bolt tip: off the thread
    q_min = g.bolt_head_top - g.nut_zmin + 0.0005
    q0 = q_max - 0.004                          # 4 mm of thread left (~4.4 turns at 0.907 mm pitch)
    sc = S.FlickScene(g=g, nut_q0=q0, q_min=q_min, q_max=q_max, nut_phase_deg=0.0,
                      driven_speed=0.004 if args.mode == "ours_driven" else None)
    S.build_flick_scene(stage, sc, BOLT, NUT, restyle=True)
    bx, by, bz = sc.bolt_pos
    xn = bx + q0 + (g.nut_zmin + g.nut_zmax) / 2
    stats.update(geometry=g.__dict__, q0=q0, q_min=q_min, q_max=q_max, nut_mass_inertia=S.nut_mass_inertia(g))
    if args.mode == "ours_passive":
        # tip (r=3 mm) bottom sits between the flat and corner radius: it catches corners, never the flats
        line = g.nut_r_flat + 0.25 * (g.nut_r_corner - g.nut_r_flat) + 0.003
        S.add_pusher(stage, sc, line_offset=line, speed=args.push_speed)
        stats.update(pusher_line_offset=line, push_speed=args.push_speed)
    # close-up 3/4 view from the bolt tip side, looking at the nut face and the finger path
    S.add_camera(stage, "/World/RecCam", eye=(xn + 0.05, -0.03, bz + 0.025), target=(xn - 0.003, 0, bz + 0.002),
                 focal=30.0)
    art_path, cam = sc.paths["art"], "/World/RecCam"
    gearing_expected = sc.paths["gearing_deg_per_m"]
    for _ in range(10):
        app.update()

world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=1.0 / args.fps)
art = Articulation(prim_path=art_path, name="nut_assembly")
world.scene.add(art)
rec = Recorder(cam)
world.reset()
names = list(art.dof_names)
log("DOFs:", names)
i_pri = next(i for i, n in enumerate(names) if "rismatic" in n)
i_rev = next(i for i, n in enumerate(names) if "evolute" in n)
if args.mode != "original":
    # world.reset() puts articulations back to joint zeros; put the nut back where the scene built it (q0, phase 0),
    # consistent with the thread constraint, before anything steps
    qs = np.zeros(len(names))
    qs[i_pri] = q0
    art.set_joint_positions(qs)
    art.set_joint_velocities(np.zeros(len(names)))
    if args.mode == "ours_driven":   # the state reset also cleared the slide drive's velocity target: re-apply it
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        vt = np.zeros(len(names))
        vt[i_pri] = sc.driven_speed
        art.get_articulation_controller().apply_action(ArticulationAction(joint_velocities=vt))
q_init = art.get_joint_positions().copy()
log("initial q:", q_init)

# warm up the renderer (first frames are often empty while shaders compile)
t0 = time.time()
for k in range(60):
    world.render()
    if k > 20 and rec.grab() is not None and rec.grab().max() > 0:
        break
log(f"renderer warm-up {time.time()-t0:.1f}s")

# thread friction (passive mode): purely dissipative, always opposes the nut's spin. Not a drive.
tau_c = tau_v = 0.0
if args.mode == "ours_passive":
    m, izz, ixx = S.nut_mass_inertia(g)
    tau_c = args.coulomb if args.coulomb is not None else izz * 60.0     # ~60 rad/s^2 Coulomb decel
    tau_v = args.viscous if args.viscous is not None else izz * 0.8
    stats.update(coulomb_Nm=tau_c, viscous_Nms=tau_v)

log_rows = []
nsteps = int(args.seconds * args.physics_hz)
t0 = time.time()
for k in range(nsteps):
    if tau_c or tau_v:
        w = float(art.get_joint_velocities()[i_rev])
        tau = -tau_c * math.tanh(w / 0.5) - tau_v * w
        eff = np.zeros(len(names))
        eff[i_rev] = tau
        art.set_joint_efforts(eff)
    render = (k % substeps == substeps - 1)
    world.step(render=False)          # physics only, exactly one physics_dt
    q = art.get_joint_positions()
    qd = art.get_joint_velocities()
    if k % 4 == 0:
        log_rows.append([k * dt, float(q[i_pri]), float(q[i_rev]), float(qd[i_pri]), float(qd[i_rev])])
    if render:
        world.render()                # renders without stepping physics
        im = rec.grab()
        if im is not None:
            frames.append(im)
wall = time.time() - t0
L = np.array(log_rows)
dq_pri = L[-1, 1] - L[0, 1]
dq_rev = L[-1, 2] - L[0, 2]
ratio = math.degrees(dq_rev) / dq_pri if abs(dq_pri) > 1e-7 else float("nan")
stats.update(
    dof_names=names, sim_seconds=args.seconds, wall_seconds=wall, frames=len(frames),
    axial_travel_mm=dq_pri * 1000, rotation_deg=math.degrees(dq_rev), rotation_turns=dq_rev / (2 * math.pi),
    measured_deg_per_m=ratio, expected_deg_per_m=gearing_expected,
    coupling_error_pct=abs(abs(ratio) - abs(gearing_expected)) / abs(gearing_expected) * 100 if ratio == ratio else None,
    peak_omega_rad_s=float(np.abs(L[:, 4]).max()), final_omega_rad_s=float(L[-1, 4]),
    frame_mean_brightness=float(np.mean([f.mean() for f in frames])) if frames else None,
)
if args.mode == "ours_passive":
    stats["contact_driven_pass"] = bool(abs(stats["rotation_turns"]) > 0.25 and abs(stats["axial_travel_mm"]) > 0.3
                                        and (stats["coupling_error_pct"] or 100) < 5)
log(json.dumps({k: v for k, v in stats.items() if k != "geometry"}, indent=1, default=float))
np.savetxt(os.path.join(args.out, f"{args.mode}_trace.csv"), L, delimiter=",",
           header="t,q_prismatic_m,q_revolute_rad,qd_prismatic,qd_revolute", comments="")
dump(f"{args.mode}.json", stats)
if frames:
    write_video(f"{args.mode}.mp4", frames, args.fps)
else:
    log("NO FRAMES CAPTURED (renderer problem) - physics results above are still valid")
app.close()
