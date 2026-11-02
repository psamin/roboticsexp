"""FrankaFlickEnv: the nut-flick task with a Franka Panda (Isaac Sim 5.1). Create SimulationApp first.

Same nut, thread model, friction, detach handoff and metrics as FlickEnv; only the robot and cameras change.
The bolt points along world +y, 0.55 m in front of the robot, so a joint-6 (wrist) swing sweeps the closed
fingertips along x across the top of the nut. Actions: 7 arm joint-position targets at 30 Hz (fingers held shut).
Cameras: camera1 = 3/4 close-up from the bolt-tip side, camera2 = wrist camera on the hand; hero / wide for videos.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import franka as FR
import scene as S
from flick_env import (ArticulationAction, FlickEnv, SingleArticulation, SingleRigidPrim, SingleXFormPrim, World)

BOLT_POS = (0.55, -0.028, 0.12)          # bolt CAD origin; nut center starts near y = 0
_ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
ENVIRONMENTS = {
    "warehouse": f"{_ASSETS}/Environments/Simple_Warehouse/full_warehouse.usd",
    "shelves": f"{_ASSETS}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "digital_twin": f"{_ASSETS}/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
}
LAB_TABLE = f"{_ASSETS}/Props/Mounts/SeattleLabTable/table.usd"


def _bbox(stage, prim):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def add_environment(stage, kind, top=((0.35, 0.0), (1.4, 1.1)), bench_h=0.90):
    """An industrial workbench (phenolic-resin top at z = 0, steel square-tube legs, apron rails, lower shelf) inside
    an Isaac Sim warehouse whose floor meets the legs at standard bench height. The warehouse comes from NVIDIA's
    asset library; the bench is built here so its top is exactly where the robot and bracket stand (NVIDIA's lab
    table asset includes a tall frame, which put its tabletop 0.7 m below the robot)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    (cx, cy), (sx, sy) = top
    m_top = S.preview_material(stage, "/World/Looks/PhenolicTop", (0.16, 0.17, 0.19), 0.0, 0.55)
    m_steel = S.preview_material(stage, "/World/Looks/BenchSteel", (0.58, 0.60, 0.63), 0.85, 0.35)
    m_shelf = S.preview_material(stage, "/World/Looks/ShelfGrey", (0.32, 0.34, 0.37), 0.5, 0.5)

    def box(path, center, size, mat):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        S.set_xform(c.GetPrim(), pos=center, scale=size)
        S.bind(c.GetPrim(), mat)
        return c
    t = 0.04
    box("/World/Env/Bench/Top", (cx, cy, -t / 2), (sx, sy, t), m_top)
    leg, inset = 0.05, 0.06
    for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        x, y = cx + dx * (sx / 2 - inset), cy + dy * (sy / 2 - inset)
        box(f"/World/Env/Bench/Leg{i}", (x, y, -(bench_h + t) / 2), (leg, leg, bench_h - t), m_steel)
    for i, (y, w) in enumerate([(cy - sy / 2 + inset, sx), (cy + sy / 2 - inset, sx)]):
        box(f"/World/Env/Bench/ApronX{i}", (cx, y, -t - 0.04), (w - 2 * inset, 0.03, 0.08), m_steel)
    for i, x in enumerate([cx - sx / 2 + inset, cx + sx / 2 - inset]):
        box(f"/World/Env/Bench/ApronY{i}", (x, cy, -t - 0.04), (0.03, sy - 2 * inset, 0.08), m_steel)
    box("/World/Env/Bench/Shelf", (cx, cy, -bench_h + 0.18), (sx - 2 * inset, sy - 2 * inset, 0.02), m_shelf)
    info = dict(bench_top=[sx, sy], bench_height=bench_h)
    if kind:
        holder = UsdGeom.Xform.Define(stage, "/World/Env/BuildingPlace")    # our placement; the asset keeps its own ops
        holder.AddTranslateOp().Set(Gf.Vec3d(0, 0, -bench_h))               # warehouse floor (z=0) under the legs
        env = UsdGeom.Xform.Define(stage, "/World/Env/BuildingPlace/Building").GetPrim()
        env.GetReferences().AddReference(ENVIRONMENTS[kind])
        n = 0
        for p in Usd.PrimRange(env):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                n += 1
        info["deinstanced_env_prims"] = n
        e = _bbox(stage, env)
        info["env_bbox"] = [[round(v, 2) for v in e.GetMin()], [round(v, 2) for v in e.GetMax()]]
    return info


def sample_franka_cfg(seed, split="train"):
    from flick_env import EpisodeConfig
    r = np.random.default_rng(seed)
    q0 = np.clip(FR.Q_READY + r.uniform(-0.15, 0.15, 7), FR.QMIN, FR.QMAX)
    return EpisodeConfig(
        seed=seed,
        remaining_mm=float(r.uniform(3.0, 6.5)),
        decel_coulomb=float(r.uniform(150, 450)),     # tip speed is ~0.6 m/s (wrist limit), so a looser thread
        decel_viscous=float(r.uniform(0.5, 2.0)),
        q_finger0=tuple(float(v) for v in q0),        # 7 arm joints for the Franka
        cam_jitter=tuple(float(v) for v in np.r_[r.uniform(-0.006, 0.006, 3), r.uniform(-0.003, 0.003, 3)]),
        max_steps=450,
    )


class FrankaFlickEnv(FlickEnv):
    def __init__(self, app, ref_dir, physics_hz=1200, ctrl_hz=30, res=256, hero=None, wide=False, env_kind="warehouse"):
        self.app = app
        self.physics_hz, self.ctrl_hz, self.res = physics_hz, ctrl_hz, res
        self.substeps = int(round(physics_hz / ctrl_hz))
        self.fs = None
        bolt_ref = os.path.join(ref_dir, "hex_bolt", "bolt.usd")
        nut_ref = os.path.join(ref_dir, "hex_bolt", "nut.usd")
        ctx = omni.usd.get_context()
        ctx.new_stage()
        for _ in range(10):
            app.update()
        self.stage = stage = ctx.get_stage()
        g, _ = S.measure(None, bolt_ref, nut_ref)
        self.g = g
        self.q_max = g.bolt_zmax - g.nut_zmin
        q_build = self.q_max - 0.00475
        self.sc = S.FlickScene(g=g, bolt_pos=BOLT_POS, nut_q0=q_build, q_min=g.bolt_head_top - g.nut_zmin + 0.0005,
                               q_max=self.q_max, axis="y", bench=((0.35, 0.0, -0.01), (1.4, 1.1, 0.02)))
        S.build_flick_scene(stage, self.sc, bolt_ref, nut_ref, restyle=True)
        if env_kind is not None:     # industry-standard surroundings: NVIDIA lab table + Isaac Sim warehouse
            info = add_environment(stage, env_kind if env_kind != "table" else None)
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/Bench")).MakeInvisible()   # keeps colliding, unseen
            print(f"[franka_env] environment {env_kind}: {info}", flush=True)
        self.gearing = self.sc.paths["gearing_deg_per_m"]
        self.mimic_offset = self.gearing * q_build
        bx, by, bz = BOLT_POS
        self.axis_z = bz
        self.nut_cx_of_q = lambda q: by + q + (g.nut_zmin + g.nut_zmax) / 2     # nut center, world y
        self.base_x = 0.0
        self.nut_x = bx
        self.r_band = (g.nut_r_flat, g.nut_r_corner)
        self.h_k = None

        # robot
        joints, links = FR.add_franka(stage, "/World/Franka", (0.0, 0.0, 0.0))
        for _ in range(20):
            app.update()
        hand = next((p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")) if p.GetName() == "panda_hand"), None)
        fingers = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")) if p.GetName() in ("panda_leftfinger", "panda_rightfinger")]
        tz = FR.fingertip_z(stage, hand, fingers) if hand is not None else None
        finger_tip = tz if tz is not None and 0.09 < tz < 0.13 else 0.1121
        from pxr import UsdShade
        self.tip_z = FR.add_flick_nub(stage, str(hand.GetPath()), finger_tip,
                                      phys_mat=UsdShade.Material.Get(stage, self.sc.paths["pm_tip"]),
                                      vis_mat=UsdShade.Material.Get(stage, self.sc.paths["m_skin"]))
        print(f"[franka_env] hand={hand.GetPath()} finger tip z={tz} (measured); nub tip z={self.tip_z:.4f}", flush=True)
        root = next((p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")) if p.HasAPI(UsdPhysics.ArticulationRootAPI)), None)
        self.robot_path = str(root.GetPath()) if root is not None else "/World/Franka"
        has_fixed = any(p.IsA(UsdPhysics.Joint) and p.GetName() == "rootJoint" or
                        (p.IsA(UsdPhysics.FixedJoint) and not p.GetRelationship("physics:body0").GetTargets())
                        for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")))
        if not has_fixed:      # make sure the base is bolted to the bench
            link0 = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")) if p.GetName() == "panda_link0")
            fj = UsdPhysics.FixedJoint.Define(stage, "/World/Franka/BaseFixed")
            fj.CreateBody1Rel().SetTargets([link0.GetPath()])
        print(f"[franka_env] articulation root {self.robot_path}, base fixed in asset: {has_fixed}", flush=True)
        S.add_free_nut(stage, self.sc, bolt_ref, nut_ref, park=(0.95, 0.45, 0.003))

        # cameras: world-fixed ones get per-episode jitter (camera1 only); the wrist camera rides on the hand
        nut0 = np.array([bx, self.nut_cx_of_q(q_build), bz])
        P = lambda d: tuple(nut0 + np.array(d))
        # camera1 / hero look from the bolt-tip side straight at the swing plane: the nub crossing the top of the nut
        # and the red index mark turning are both in view, and the hand no longer hides the contact
        self.cam_nominal = {"camera1": (P((0.10, 0.24, 0.09)), P((-0.01, 0.0, 0.03)), 20.0)}
        if hero:
            self.cam_nominal["hero"] = (P((0.12, 0.34, 0.08)), P((-0.01, 0.0, 0.05)), 22.0)
        if wide:
            self.cam_nominal["wide"] = ((1.6, 1.1, 0.95), (0.30, 0.0, 0.33), 22.0)
        for name, (eye, tgt, f) in self.cam_nominal.items():
            S.add_camera(stage, f"/World/Cam_{name}", eye, tgt, focal=f)
        # wrist camera: a world-level camera re-posed every frame from the hand's simulated pose (a camera parented
        # under the physics link rendered from the link's authored rest pose instead of following it)
        wc = UsdGeom.Camera.Define(stage, "/World/WristCam")
        wc.CreateFocalLengthAttr(14.0)
        wc.CreateHorizontalApertureAttr(20.955)
        wc.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
        self.wrist_local = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.075, 0.0, 0.03), Gf.Vec3d(0.0, 0.0, 0.15),
                                                   Gf.Vec3d(1, 0, 0)).GetInverse()
        self.wrist_op = UsdGeom.Xformable(wc.GetPrim()).AddTransformOp()
        self.wrist_op.Set(self.wrist_local)
        self.wrist_cam = str(wc.GetPath())
        for _ in range(10):
            app.update()

        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / physics_hz, rendering_dt=1.0 / ctrl_hz)
        self.nut_art = SingleArticulation(prim_path=self.sc.paths["art"], name="nut_assembly")
        self.finger = SingleArticulation(prim_path=self.robot_path, name="franka")
        self.free_nut = SingleRigidPrim(prim_path="/World/FreeNut", name="free_nut")
        self.nut_xf = SingleXFormPrim(prim_path=self.sc.paths["nut"], name="nut_xf")
        self.hand_xf = SingleXFormPrim(prim_path=str(hand.GetPath()), name="hand_xf")
        for o in (self.nut_art, self.finger, self.free_nut):
            self.world.scene.add(o)
        self.world.reset()
        dn = list(self.nut_art.dof_names)
        self.i_pri = next(i for i, n in enumerate(dn) if "rismatic" in n)
        self.i_rev = next(i for i, n in enumerate(dn) if "evolute" in n)
        fdn = list(self.finger.dof_names)
        print("[franka_env] robot dofs:", fdn, flush=True)
        self.f_idx = [fdn.index(f"panda_joint{i}") for i in range(1, 8)]
        self.grip_idx = [fdn.index(n) for n in ("panda_finger_joint1", "panda_finger_joint2") if n in fdn]
        self.nut_dofs = len(dn)
        self.ctrl = self.finger.get_articulation_controller()
        _, izz, _ = S.nut_mass_inertia(g)
        self.izz = izz

        self.rp, self.annot = {}, {}
        cams = {n: f"/World/Cam_{n}" for n in self.cam_nominal}
        cams["camera2"] = self.wrist_cam
        for name, path in cams.items():
            wh = tuple(hero) if name in ("hero", "wide") else (self.res, self.res)
            self.rp[name] = rep.create.render_product(path, wh)
            a = rep.AnnotatorRegistry.get_annotator("rgb")
            a.attach([self.rp[name]])
            self.annot[name] = a
        for _ in range(30):
            self.world.render()
        t0 = time.time()
        while not self._frames_ready() and time.time() - t0 < 180:
            for _ in range(self.substeps):
                self.world.step(render=False)
            self.world.render()
        print(f"[franka_env] cameras ready={self._frames_ready()} after {time.time()-t0:.1f}s", flush=True)
        self.nut_meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(self.sc.paths["nut"]))
                           if p.IsA(UsdGeom.Mesh) or p.GetName() == "Mark"]
        self._check_fk()

    def _check_fk(self):
        """Numpy URDF FK vs the simulator's own hand pose, at two configurations (catches frame mismatches)."""
        errs = []
        for q in (FR.Q_READY, FR.Q_READY + np.array([0.3, 0.2, -0.2, 0.3, 0.2, -0.3, 0.4])):
            qf = np.asarray(self.finger.get_joint_positions(), float).copy()
            qf[self.f_idx] = q
            self.finger.set_joint_positions(qf)
            self.world.step(render=False)
            pos, _ = self.hand_xf.get_world_pose()
            errs.append(float(np.linalg.norm(np.asarray(pos) - FR.fk(q)["hand"][:3, 3])))
        self.fk_error = max(errs)
        print(f"[franka_env] FK check: max hand position error {self.fk_error*1000:.2f} mm", flush=True)

    def _place_cameras(self, jitter):
        j = np.asarray(jitter)
        for name, (eye, tgt, f) in self.cam_nominal.items():
            jit = name == "camera1"
            e = np.asarray(eye) + (j[:3] if jit else 0)
            t = np.asarray(tgt) + (j[3:] if jit else 0)
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*e), Gf.Vec3d(*t), Gf.Vec3d(0, 0, 1))
            UsdGeom.Xformable(self.stage.GetPrimAtPath(f"/World/Cam_{name}")).GetOrderedXformOps()[0].Set(view.GetInverse())

    def _update_wrist_cam(self):
        pos, quat = self.hand_xf.get_world_pose()
        H = Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))),
                                       Gf.Vec3d(*map(float, pos)))
        self.wrist_op.Set(self.wrist_local * H)      # row-vector convention: local first, then the hand

    def render(self, names=None):
        if hasattr(self, "hand_xf"):
            self._update_wrist_cam()
        return super().render(names)

    def info(self):
        d = super().info()
        d["nut_center"] = np.array([self.nut_x, d["nut_cx"], self.axis_z])
        return d

    def observe(self):
        imgs = self.render()
        hero = imgs.pop("hero", None)
        wide = imgs.pop("wide", None)
        return {"images": imgs, "hero": hero, "wide": wide, "state": self.finger_q().astype(np.float32)}
