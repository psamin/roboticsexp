"""FlickEnv: Isaac Sim 4.5 environment for the nut-flick task (create SimulationApp before importing this).

Control: 30 Hz joint-position targets for the finger [slide_x (m), mcp, pip, dip (rad)].
Physics: physics_hz substeps per control step. Between substeps the thread friction torque is applied to the nut's
revolute DOF: tau = -tau_c*tanh(w/0.5) - c*w. It only ever opposes the spin (dissipative), so the nut accelerates
only through finger contact. Nothing drives the nut.
Detach: when the nut reaches the end of the thread its pose+velocity are copied to a free rigid-body nut, which
falls off under normal dynamics (the articulated nut is hidden and its collision switched off).
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.prims import RigidPrim as SingleRigidPrim, XFormPrim as SingleXFormPrim
    from omni.isaac.core.utils.types import ArticulationAction

import scene as S

TASK = "Flick the loosened nut until it spins off the bolt"
JOINTS = ["slide_x", "mcp", "pip", "dip"]


@dataclass
class EpisodeConfig:
    seed: int
    remaining_mm: float = 4.5        # thread left under the nut at start (sets height; angle follows the thread)
    decel_coulomb: float = 500.0     # rad/s^2 from Coulomb thread friction (looseness)
    decel_viscous: float = 1.0       # 1/s viscous thread friction
    q_finger0: tuple = (0.0, 0.6, 0.6, 0.4)
    cam_jitter: tuple = (0.0,) * 6   # dx,dy,dz eye; dx,dy,dz target (m)
    max_steps: int = 300

    @staticmethod
    def sample(seed: int, split: str = "train"):
        r = np.random.default_rng(seed)
        return EpisodeConfig(
            seed=seed,
            remaining_mm=float(r.uniform(3.0, 6.5)),
            decel_coulomb=float(r.uniform(300, 900)),
            decel_viscous=float(r.uniform(0.5, 2.0)),
            # mcp >= 0.5 rad keeps the fingertip well to the +y side of the nut at t=0 (never starts in contact)
            q_finger0=(float(r.uniform(-0.008, 0.008)), float(r.uniform(0.5, 1.0)), float(r.uniform(0.3, 1.1)),
                       float(r.uniform(0.2, 0.8))),
            cam_jitter=tuple(float(v) for v in np.r_[r.uniform(-0.004, 0.004, 3), r.uniform(-0.002, 0.002, 3)]),
        )


class FlickEnv:
    def __init__(self, app, ref_dir, physics_hz=1200, ctrl_hz=30, res=256, hero=None, fs: S.FingerSpec | None = None):
        self.app = app
        self.physics_hz, self.ctrl_hz, self.res = physics_hz, ctrl_hz, res
        self.substeps = int(round(physics_hz / ctrl_hz))
        self.fs = fs or S.FingerSpec()
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
        q_min = g.bolt_head_top - g.nut_zmin + 0.0005
        q_build = self.q_max - 0.00475
        self.sc = S.FlickScene(g=g, nut_q0=q_build, q_min=q_min, q_max=self.q_max, nut_phase_deg=0.0)
        S.build_flick_scene(stage, self.sc, bolt_ref, nut_ref, restyle=True)
        self.gearing = self.sc.paths["gearing_deg_per_m"]
        self.mimic_offset = self.gearing * q_build - 0.0
        bx, by, bz = self.sc.bolt_pos
        self.axis_z = bz
        self.nut_cx_of_q = lambda q: bx + q + (g.nut_zmin + g.nut_zmax) / 2
        # knuckle above the bolt axis: a moderately curled finger reaches the corner band at the bottom of its swing
        self.r_band = (g.nut_r_flat, g.nut_r_corner)
        R25, _ = S.finger_reach(self.fs, math.radians(25), math.radians(17))
        self.h_k = R25 + self.fs.r3 + g.nut_r_flat + 0.35 * (g.nut_r_corner - g.nut_r_flat)
        self.base_x = self.nut_cx_of_q(q_build)
        S.add_finger(stage, self.sc, self.fs, knuckle=(self.base_x, by, bz + self.h_k))
        S.add_free_nut(stage, self.sc, bolt_ref, nut_ref)
        xn = self.base_x
        # camera1: 3/4 close-up of the nut and the zone the fingertip sweeps through (shows axial travel)
        # camera2: end-on along the bolt from beyond its tip (shows the whole swing arc and the hex orientation)
        self.cam_nominal = {
            "camera1": ((xn + 0.070, -0.050, bz + 0.050), (xn - 0.004, 0.0, bz + 0.012), 24.0),
            "camera2": ((xn + 0.220, 0.000, bz + 0.045), (xn, 0.0, bz + 0.035), 24.0),
        }
        if hero:
            self.cam_nominal["hero"] = ((xn + 0.060, -0.060, bz + 0.040), (xn - 0.002, 0.0, bz + 0.014), 22.0)
        for name, (eye, tgt, f) in self.cam_nominal.items():
            S.add_camera(stage, f"/World/Cam_{name}", eye, tgt, focal=f)
        for _ in range(10):
            app.update()

        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / physics_hz, rendering_dt=1.0 / ctrl_hz)
        self.nut_art = SingleArticulation(prim_path=self.sc.paths["art"], name="nut_assembly")
        self.finger = SingleArticulation(prim_path="/World/Finger", name="finger")
        self.free_nut = SingleRigidPrim(prim_path="/World/FreeNut", name="free_nut")
        self.nut_xf = SingleXFormPrim(prim_path=self.sc.paths["nut"], name="nut_xf")
        for o in (self.nut_art, self.finger, self.free_nut):
            self.world.scene.add(o)
        self.world.reset()
        dn = list(self.nut_art.dof_names)
        self.i_pri = next(i for i, n in enumerate(dn) if "rismatic" in n)
        self.i_rev = next(i for i, n in enumerate(dn) if "evolute" in n)
        fdn = list(self.finger.dof_names)
        self.f_idx = [fdn.index(n) for n in ("SlideX", "MCP", "PIP", "DIP")]
        self.nut_dofs = len(dn)
        self.ctrl = self.finger.get_articulation_controller()
        _, izz, _ = S.nut_mass_inertia(g)
        self.izz = izz

        self.res_hero = hero
        self.rp, self.annot = {}, {}
        for name in self.cam_nominal:
            wh = tuple(hero) if name == "hero" else (self.res, self.res)
            self.rp[name] = rep.create.render_product(f"/World/Cam_{name}", wh)
            a = rep.AnnotatorRegistry.get_annotator("rgb")
            a.attach([self.rp[name]])
            self.annot[name] = a
        for _ in range(30):     # let materials load and shaders compile before the first real frame
            self.world.render()
        # RTX compiles shaders in the background: the cameras deliver nothing until that finishes (wall-clock
        # seconds, not a number of renders). Advance a control period at a time until every camera has a frame.
        import time as _time
        t0 = _time.time()
        while not self._frames_ready() and _time.time() - t0 < 180:
            for _ in range(self.substeps):
                self.world.step(render=False)
            self.world.render()
        print(f"[flick_env] cameras ready={self._frames_ready()} after {_time.time()-t0:.1f}s", flush=True)
        self.nut_meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(self.sc.paths["nut"]))
                           if p.IsA(UsdGeom.Mesh) or p.GetName() == "Mark"]

    # ------------------------------------------------------------------ helpers
    def _frames_ready(self):
        for n in self.annot:
            d = self.annot[n].get_data()
            if d is None or getattr(d, "size", 0) == 0:
                return False
        return True

    def nut_state(self):
        q = self.nut_art.get_joint_positions()
        qd = self.nut_art.get_joint_velocities()
        return float(q[self.i_pri]), float(q[self.i_rev]), float(qd[self.i_pri]), float(qd[self.i_rev])

    def finger_q(self):
        return np.asarray(self.finger.get_joint_positions(), dtype=np.float64)[self.f_idx]

    def finger_qd(self):
        return np.asarray(self.finger.get_joint_velocities(), dtype=np.float64)[self.f_idx]

    def _set_nut_visible(self, on: bool):
        for p in self.nut_meshes:
            im = UsdGeom.Imageable(p)
            im.MakeVisible() if on else im.MakeInvisible()
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(on)

    def _place_cameras(self, jitter):
        j = np.asarray(jitter)
        for name, (eye, tgt, f) in self.cam_nominal.items():
            e = np.asarray(eye) + (j[:3] if name != "hero" else 0)
            t = np.asarray(tgt) + (j[3:] if name != "hero" else 0)
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*e), Gf.Vec3d(*t), Gf.Vec3d(0, 0, 1))
            op = UsdGeom.Xformable(self.stage.GetPrimAtPath(f"/World/Cam_{name}")).GetOrderedXformOps()[0]
            op.Set(view.GetInverse())

    def render(self, names=None):
        # The rgb annotator hands back the last *completed* frame, so a single render returns the previous state
        # (episode 0 showed the last episode's bare bolt). Render twice so every image matches the current state.
        self.world.render()
        out = {}
        for attempt in range(6):            # an annotator can come back empty for the first frames: render again
            self.world.render()
            for n in names or list(self.annot):
                if out.get(n) is None:
                    d = self.annot[n].get_data()
                    out[n] = None if d is None or getattr(d, "size", 0) == 0 else np.asarray(d)[:, :, :3].copy()
            if all(v is not None for v in out.values()):
                break
        if not hasattr(self, "last_frame"):
            self.last_frame, self.empty_frames = {}, 0
        for n, v in out.items():
            if v is None:                   # still empty: reuse the last good frame and count it
                self.empty_frames += 1
                out[n] = self.last_frame.get(n)
            else:
                self.last_frame[n] = v
        return out

    # ------------------------------------------------------------------ episode API
    def reset(self, cfg: EpisodeConfig):
        self.cfg = cfg
        self.tau_c = self.izz * cfg.decel_coulomb
        self.c_visc = self.izz * cfg.decel_viscous
        q = self.q_max - cfg.remaining_mm / 1000
        q_rev = math.radians(self.gearing * q - self.mimic_offset)
        qn = np.zeros(self.nut_dofs)
        qn[self.i_pri], qn[self.i_rev] = q, q_rev
        self.nut_art.set_joint_positions(qn)
        self.nut_art.set_joint_velocities(np.zeros(self.nut_dofs))
        self._set_nut_visible(True)
        park = self.sc.paths["free_nut_park"]
        self.free_nut.set_world_pose(np.array(park), np.array([1.0, 0, 0, 0]))
        self.free_nut.set_linear_velocity(np.zeros(3))
        self.free_nut.set_angular_velocity(np.zeros(3))
        qf = np.zeros(len(self.finger.dof_names))
        qf[self.f_idx] = cfg.q_finger0
        self.finger.set_joint_positions(qf)
        self.finger.set_joint_velocities(np.zeros_like(qf))
        self.ctrl.apply_action(ArticulationAction(joint_positions=qf))
        self._place_cameras(cfg.cam_jitter)
        self.detached = False
        self.detach_info = {}
        self.sim_fail = None
        self.t = 0.0
        self.k = 0
        self.q_start = q
        self.trace = []   # physics-rate nut log: t, q, q_rev, qd, w
        for _ in range(3):
            self.world.step(render=False)
        # After a reset the rgb annotators return nothing for a while (the recorder used to reuse the previous
        # episode's last frame). Keep stepping physics with the finger holding still until every camera has a
        # fresh frame, so the first observation always shows this episode's scene.
        # The capture only refreshes once simulated time has advanced by a full camera frame (1/ctrl_hz), so each
        # try advances one control period of physics (finger holding its start pose, nut at rest) before rendering.
        for _ in range(8):
            for _ in range(self.substeps):
                self.world.step(render=False)
            self.world.render()
            if self._frames_ready():
                break
        return self.observe()

    def observe(self):
        imgs = self.render()
        hero = imgs.pop("hero", None)
        return {"images": imgs, "hero": hero, "state": self.finger_q().astype(np.float32)}

    def info(self):
        q, qr, qd, w = self.nut_state() if not self.detached else (self.q_max, 0.0, 0.0, 0.0)
        return dict(t=self.t, q=q, q_rev=qr, qd=qd, w=w, detached=self.detached, sim_fail=self.sim_fail,
                    nut_cx=self.nut_cx_of_q(q) - self.base_x, axial_mm=(q - self.q_start) * 1000,
                    remaining_mm=(self.q_max - q) * 1000)

    def _detach(self):
        q, qr, qd, w = self.nut_state()
        pos, quat = self.nut_xf.get_world_pose()
        self._set_nut_visible(False)
        self.free_nut.set_world_pose(np.asarray(pos), np.asarray(quat))
        self.free_nut.set_linear_velocity(np.array([max(qd, 0.02), 0.0, 0.0]))
        self.free_nut.set_angular_velocity(np.array([w, 0.0, 0.0]))
        self.detached = True
        self.detach_info = dict(t=self.t, w=w, qd=qd)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        qf = np.asarray(self.finger.get_joint_positions(), dtype=np.float64).copy()
        qf[self.f_idx] = a
        self.ctrl.apply_action(ArticulationAction(joint_positions=qf))
        eff = np.zeros(self.nut_dofs)
        dt = 1.0 / self.physics_hz
        for _ in range(self.substeps):
            if not self.detached:
                q, qr, qd, w = self.nut_state()
                if not all(map(math.isfinite, (q, qr, qd, w))) or abs(w) > 3000:
                    self.sim_fail = self.sim_fail or f"nut state blew up (w={w:.1f})"
                expect = math.radians(self.gearing * q - self.mimic_offset)
                if abs(qr - expect) > math.radians(10):
                    self.sim_fail = self.sim_fail or f"thread constraint violated by {math.degrees(qr-expect):.1f} deg"
                eff[self.i_rev] = -self.tau_c * math.tanh(w / 0.5) - self.c_visc * w
                self.nut_art.set_joint_efforts(eff)
                self.trace.append((self.t, q, qr, qd, w))
                if q >= self.q_max - 1e-4:
                    self._detach()
            self.world.step(render=False)
            self.t += dt
        self.k += 1
        return self.observe(), self.info()
