"""FlickLoRA scene: the reference nut/bolt CAD on a horizontal bolt, restyled, with a passive screw joint.

Reused from Tech-Multiverse/omniverse-nut-and-bolt-digital-twin (static_bolt.usda): the nut.usd / bolt.usd CAD and
the screw mechanism (bolt fixed to world -> prismatic NutSlide -> revolute Nut, rotation tied to travel by a
PhysxMimicJointAPI). Changed: the bolt sticks out horizontally from a bracket (so gravity does not back-drive the
screw), the prismatic joint has NO drive (the nut only moves when something pushes it), new materials, lighting,
bench, and a red index mark on the nut so rotation is easy to see on video.

Units: meters, kg, seconds. USD revolute angles are degrees, so mimic gearing is deg per meter.
Rigid-body prims are never scaled; the mm-unit CAD sits in a scaled child "Geom" so joint frames stay in meters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

STEEL = 7850.0  # kg/m^3


@dataclass
class Geometry:
    """Measured from the CAD meshes (mesh units are mm; values here are meters)."""
    bolt_zmin: float
    bolt_zmax: float
    bolt_r: float          # thread crest radius
    nut_zmin: float
    nut_zmax: float
    nut_r_corner: float    # hex corner radius
    nut_r_flat: float      # half across-flats
    nut_r_hole: float
    pitch: float           # m per revolution, from the bolt thread helix
    handed: str            # "right" or "left"
    pitch_strength: float  # 0..1 helix-fit quality
    bolt_head_top: float = 0.0   # CAD z where the head ends and the shank starts

    @property
    def nut_h(self):
        return self.nut_zmax - self.nut_zmin


def _mesh_points(stage, root_path):
    pts = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.IsA(UsdGeom.Mesh):
            p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            m = np.array(xf, dtype=np.float64)          # row-vector convention
            ph = np.c_[p, np.ones(len(p))] @ m
            pts.append(ph[:, :3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def estimate_pitch(p_mm):
    """Fit a helix to the bolt crest vertices: pick the pitch whose phase z/P - theta/2pi is most concentrated."""
    r = np.hypot(p_mm[:, 0], p_mm[:, 1])
    theta = np.arctan2(p_mm[:, 1], p_mm[:, 0])
    z = p_mm[:, 2]
    # threaded shank only: drop the head (widest radius) and keep vertices near the crest radius of the shank
    shank = r < np.percentile(r, 99) * 0.75 if r.max() > 1.4 * np.median(r) else np.ones_like(r, bool)
    rs = r[shank]
    crest = shank.copy()
    crest[shank] = rs > np.percentile(rs, 80)
    best = (0.0, None, None)
    for hand, sgn in (("right", 1.0), ("left", -1.0)):
        for P in np.arange(0.3, 3.0, 0.005):
            ph = 2 * np.pi * (z[crest] / P - sgn * theta[crest] / (2 * np.pi))
            R = abs(np.mean(np.exp(1j * ph)))
            if R > best[0]:
                best = (R, P, hand)
    R0, P0, hand = best
    sgn = 1.0 if hand == "right" else -1.0
    for P in np.arange(P0 - 0.01, P0 + 0.01, 0.0002):      # refine
        ph = 2 * np.pi * (z[crest] / P - sgn * theta[crest] / (2 * np.pi))
        R = abs(np.mean(np.exp(1j * ph)))
        if R > best[0]:
            best = (R, P, hand)
    return best[1] / 1000.0, best[2], float(best[0])


def measure(stage, bolt_ref, nut_ref):
    """Load the two CAD files into a scratch stage and measure them (all mesh units mm)."""
    s = Usd.Stage.CreateInMemory()
    b = s.DefinePrim("/B")
    b.GetReferences().AddReference(bolt_ref)
    n = s.DefinePrim("/N")
    n.GetReferences().AddReference(nut_ref)
    pb = _mesh_points(s, "/B")
    pn = _mesh_points(s, "/N")
    rb = np.hypot(pb[:, 0], pb[:, 1])
    rn = np.hypot(pn[:, 0], pn[:, 1])
    pitch, hand, strength = estimate_pitch(pb)
    # hex across-flats: flat faces have vertices only at the corners, so use the bbox (corners on x, flats on y)
    r_flat = min(np.abs(pn[:, 0]).max(), np.abs(pn[:, 1]).max())
    shank = rb < np.percentile(rb, 99) * 0.75 if rb.max() > 1.4 * np.median(rb) else np.ones_like(rb, bool)
    g = Geometry(
        bolt_zmin=pb[:, 2].min() / 1000, bolt_zmax=pb[:, 2].max() / 1000,
        bolt_r=np.percentile(rb[shank], 99) / 1000,
        nut_zmin=pn[:, 2].min() / 1000, nut_zmax=pn[:, 2].max() / 1000,
        nut_r_corner=rn.max() / 1000, nut_r_flat=r_flat / 1000, nut_r_hole=rn.min() / 1000,
        pitch=pitch, handed=hand, pitch_strength=strength)
    if rb.max() > 1.4 * np.median(rb):     # head = vertices wider than the shank
        head = rb > np.percentile(rb[shank], 99) * 1.15
        zh = pb[head, 2]
        # head is at one end; take the end nearest the head's centroid
        g.bolt_head_top = (zh.max() if abs(zh.mean() - pb[:, 2].min()) < abs(zh.mean() - pb[:, 2].max())
                           else zh.min()) / 1000
    return g, {"bolt_bbox_mm": [pb.min(0).tolist(), pb.max(0).tolist()],
               "nut_bbox_mm": [pn.min(0).tolist(), pn.max(0).tolist()], "n_bolt_pts": len(pb), "n_nut_pts": len(pn)}


def nut_mass_inertia(g: Geometry):
    """Hex prism with a hole, steel. Returns (mass, Izz, Ixx)."""
    a = g.nut_r_corner                          # hex side = corner radius
    area = 1.5 * math.sqrt(3) * a * a - math.pi * g.nut_r_hole ** 2
    m = STEEL * area * g.nut_h
    # polar moment of a regular hexagon about its center: (5*sqrt(3)/8) a^4, per unit thickness
    J_hex = 5 * math.sqrt(3) / 8 * a ** 4
    J_hole = math.pi / 2 * g.nut_r_hole ** 4
    izz = STEEL * g.nut_h * (J_hex - J_hole)
    ixx = 0.5 * izz + m * g.nut_h ** 2 / 12
    return m, izz, ixx


# --------------------------------------------------------------------------- materials / looks
def preview_material(stage, path, color, metallic, roughness, emissive=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def physics_material(stage, path, mu_s, mu_d, restitution):
    m = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    pm.CreateStaticFrictionAttr(mu_s)
    pm.CreateDynamicFrictionAttr(mu_d)
    pm.CreateRestitutionAttr(restitution)
    return m


def bind_physics(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def set_xform(prim, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:])))
    if scale is not None:
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))


def collision_tune(prim, contact=0.0005, rest=0.0):
    UsdPhysics.CollisionAPI.Apply(prim)
    c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    c.CreateContactOffsetAttr(contact)
    c.CreateRestOffsetAttr(rest)


def add_camera(stage, path, eye, target, focal=35.0, up=(0, 0, 1)):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))   # default near clip is 1 m: everything would vanish
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    x = UsdGeom.Xformable(cam.GetPrim())
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(view.GetInverse())
    return cam


# --------------------------------------------------------------------------- the scene
Q_X_AXIS = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)   # local +Z -> world +X (bolt axis)
Q_Y_AXIS = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)  # local +Z -> world +Y (Franka scene)


@dataclass
class FlickScene:
    g: Geometry
    bolt_pos: tuple = (0.0, 0.0, 0.10)     # bolt CAD origin in world; axis along world +X
    nut_q0: float = 0.0                    # starting prismatic coordinate (m along bolt axis, CAD frame)
    q_min: float = 0.0
    q_max: float = 0.0                     # end of thread: nut fully off when q reaches this
    driven_speed: float | None = None      # m/s on the prismatic joint (Phase A demo only)
    nut_phase_deg: float = 0.0             # hex orientation at start (independent of height via the mimic offset)
    axis: str = "x"                        # bolt axis: "x" (finger scene) or "y" (Franka scene)
    bench: tuple = ((0.1, 0.0, -0.01), (0.8, 0.6, 0.02))   # center, size
    paths: dict = field(default_factory=dict)

    @property
    def qaxis(self):
        return Q_X_AXIS if self.axis == "x" else Q_Y_AXIS

    def along(self, q):                    # world point on the bolt axis at CAD-z = q
        bx, by, bz = self.bolt_pos
        return (bx + q, by, bz) if self.axis == "x" else (bx, by + q, bz)


def build_flick_scene(stage, sc: FlickScene, bolt_ref, nut_ref, restyle=True):
    g = sc.g
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateSolverTypeAttr("TGS")
    px.CreateBounceThresholdAttr(0.02)
    px.CreateEnableCCDAttr(True)

    m_steel_dark = preview_material(stage, "/World/Looks/DarkSteel", (0.28, 0.29, 0.31), 1.0, 0.30)   # dark but threads still read
    m_zinc = preview_material(stage, "/World/Looks/YellowZinc", (0.86, 0.70, 0.28), 1.0, 0.22)
    m_mark = preview_material(stage, "/World/Looks/IndexRed", (0.85, 0.05, 0.03), 0.0, 0.5)
    m_bench = preview_material(stage, "/World/Looks/Walnut", (0.30, 0.18, 0.10), 0.0, 0.65)
    m_bracket = preview_material(stage, "/World/Looks/BlueSteel", (0.10, 0.22, 0.45), 0.6, 0.45)
    m_skin = preview_material(stage, "/World/Looks/Silicone", (0.93, 0.47, 0.30), 0.0, 0.55)
    pm_steel = physics_material(stage, "/World/Physics/SteelOnSteel", 0.35, 0.30, 0.15)
    pm_tip = physics_material(stage, "/World/Physics/SiliconeTip", 0.9, 0.8, 0.05)
    sc.paths.update(pm_tip="/World/Physics/SiliconeTip", m_skin="/World/Looks/Silicone")

    # lights: warm key + cool fill + soft dome
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(600)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2500)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.8))
    key.CreateAngleAttr(1.0)
    set_xform(key.GetPrim(), quat=_quat_from_euler(-50, 0, 35))
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(30000)
    fill.CreateRadiusAttr(0.03)
    set_xform(fill.GetPrim(), pos=(0.10, -0.25, 0.25))

    # bench + bracket (static colliders)
    bx, by, bz = sc.bolt_pos
    bench = UsdGeom.Cube.Define(stage, "/World/Bench")
    bench.CreateSizeAttr(1.0)
    set_xform(bench.GetPrim(), pos=sc.bench[0], scale=sc.bench[1])
    bind(bench.GetPrim(), m_bench)
    UsdPhysics.CollisionAPI.Apply(bench.GetPrim())
    bracket = UsdGeom.Cube.Define(stage, "/World/Bracket")
    bracket.CreateSizeAttr(1.0)
    plate_t = 0.006
    plate_c = sc.along(g.bolt_head_top + plate_t / 2)
    if sc.axis == "x":
        set_xform(bracket.GetPrim(), pos=(plate_c[0], 0, bz / 2 + 0.01), scale=(plate_t, 0.07, bz + 0.02))
    else:   # Franka scene: plate top only 12 mm above the bolt axis so the hand clears it
        set_xform(bracket.GetPrim(), pos=(plate_c[0], plate_c[1], (bz + 0.012) / 2), scale=(0.07, plate_t, bz + 0.012))
    bind(bracket.GetPrim(), m_bracket)

    # ---- articulation: Bolt (fixed) -> prismatic NutSlide -> revolute Nut (mimic-coupled)
    art = UsdGeom.Xform.Define(stage, "/World/NutAssembly").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(art)
    pa = PhysxSchema.PhysxArticulationAPI.Apply(art)
    pa.CreateEnabledSelfCollisionsAttr(False)          # nut/bolt hulls overlap by design; the joint is the thread
    pa.CreateSolverPositionIterationCountAttr(32)
    pa.CreateSolverVelocityIterationCountAttr(4)

    bolt = UsdGeom.Xform.Define(stage, "/World/NutAssembly/Bolt").GetPrim()
    set_xform(bolt, pos=sc.bolt_pos, quat=sc.qaxis)
    UsdPhysics.RigidBodyAPI.Apply(bolt)
    bgeom = UsdGeom.Xform.Define(stage, "/World/NutAssembly/Bolt/Geom").GetPrim()
    bgeom.GetReferences().AddReference(bolt_ref)
    set_xform(bgeom, scale=(0.001, 0.001, 0.001))
    for p in Usd.PrimRange(bgeom):
        if p.IsA(UsdGeom.Mesh):
            collision_tune(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
            bind_physics(p, pm_steel)
    if restyle:
        bind(bgeom, m_steel_dark)
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/NutAssembly/Bolt/FixedJoint")
    fj.CreateBody1Rel().SetTargets([bolt.GetPath()])
    fj.CreateLocalPos0Attr(Gf.Vec3f(*sc.bolt_pos))
    fj.CreateLocalRot0Attr(Gf.Quatf(*sc.qaxis))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    along = sc.along

    slide = UsdGeom.Xform.Define(stage, "/World/NutAssembly/NutSlide").GetPrim()
    set_xform(slide, pos=along(sc.nut_q0), quat=sc.qaxis)
    UsdPhysics.RigidBodyAPI.Apply(slide)
    ms = UsdPhysics.MassAPI.Apply(slide)
    ms.CreateMassAttr(0.001)
    ms.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9, 1e-9, 1e-9))
    pj = UsdPhysics.PrismaticJoint.Define(stage, "/World/NutAssembly/NutSlide/PrismaticJoint")
    pj.CreateAxisAttr("Z")
    pj.CreateBody0Rel().SetTargets([bolt.GetPath()])
    pj.CreateBody1Rel().SetTargets([slide.GetPath()])
    pj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    pj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    pj.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    pj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    pj.CreateLowerLimitAttr(sc.q_min)
    pj.CreateUpperLimitAttr(sc.q_max)
    if sc.driven_speed is not None:   # Phase A only: reproduce the original "driven" screw motion
        d = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
        d.CreateDampingAttr(1e4)
        d.CreateStiffnessAttr(0.0)
        d.CreateTargetVelocityAttr(sc.driven_speed)

    nut = UsdGeom.Xform.Define(stage, "/World/NutAssembly/Nut").GetPrim()
    qz = Gf.Rotation(Gf.Vec3d(0, 0, 1), sc.nut_phase_deg).GetQuat()
    qn = Gf.Quatd(*sc.qaxis) * qz
    set_xform(nut, pos=along(sc.nut_q0), quat=(qn.GetReal(), *qn.GetImaginary()))
    UsdPhysics.RigidBodyAPI.Apply(nut)
    m, izz, ixx = nut_mass_inertia(g)
    if sc.axis == "y":    # Franka scene: cap how fast the solver pushes overlaps apart (one-step 400+ rad/s spikes)
        PhysxSchema.PhysxRigidBodyAPI.Apply(nut).CreateMaxDepenetrationVelocityAttr(0.5)
    mn = UsdPhysics.MassAPI.Apply(nut)
    mn.CreateMassAttr(m)
    mn.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, izz))
    mn.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, (g.nut_zmin + g.nut_zmax) / 2))
    ngeom = UsdGeom.Xform.Define(stage, "/World/NutAssembly/Nut/Geom").GetPrim()
    ngeom.GetReferences().AddReference(nut_ref)
    set_xform(ngeom, scale=(0.001, 0.001, 0.001))
    for p in Usd.PrimRange(ngeom):
        if p.IsA(UsdGeom.Mesh):
            collision_tune(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
            bind_physics(p, pm_steel)
    if restyle:
        bind(ngeom, m_zinc)
        # index mark on the outer face (visual only): a thin red bar from the hole to one flat
        mk = UsdGeom.Cube.Define(stage, "/World/NutAssembly/Nut/Mark")
        mk.CreateSizeAttr(1.0)
        rm = (g.nut_r_hole + g.nut_r_flat) / 2
        set_xform(mk.GetPrim(), pos=(rm, 0, g.nut_zmax + 0.0001),
                  scale=(g.nut_r_flat - g.nut_r_hole, 0.0012, 0.0002))
        bind(mk.GetPrim(), m_mark)
    rj = UsdPhysics.RevoluteJoint.Define(stage, "/World/NutAssembly/Nut/RevoluteJoint")
    rj.CreateAxisAttr("Z")
    rj.CreateBody0Rel().SetTargets([slide.GetPath()])
    rj.CreateBody1Rel().SetTargets([nut.GetPath()])
    for a in ("LocalPos0", "LocalPos1"):
        getattr(rj, f"Create{a}Attr")(Gf.Vec3f(0, 0, 0))
    rj.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    rj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    rj.CreateLowerLimitAttr(-1e6)
    rj.CreateUpperLimitAttr(1e6)
    # PhysX mimic: q_rev + G*q_ref + offset = 0. Thread: q_rev = gearing*q_prism + phase (deg).
    # The start pose has q_rev = phase and q_prism = q0, so offset = gearing*q0 - phase keeps t=0 consistent.
    gearing = 360.0 / g.pitch * (1 if g.handed == "right" else -1)   # deg of nut per m of travel
    mj = PhysxSchema.PhysxMimicJointAPI.Apply(rj.GetPrim(), UsdPhysics.Tokens.rotX)
    mj.CreateGearingAttr(-gearing)        # same sign convention as the reference static_bolt.usda (-240000)
    mj.CreateOffsetAttr(gearing * sc.nut_q0 - sc.nut_phase_deg)
    mj.CreateReferenceJointRel().SetTargets([pj.GetPath()])
    sc.paths.update(art="/World/NutAssembly", nut="/World/NutAssembly/Nut", bolt="/World/NutAssembly/Bolt",
                    prismatic=str(pj.GetPath()), revolute=str(rj.GetPath()), gearing_deg_per_m=gearing)
    return sc


def _quat_from_euler(rx, ry, rz):
    r = Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
    q = r.GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def add_pusher(stage, sc: FlickScene, line_offset, speed, start_y=0.03, tip_r=0.003, max_force=30.0):
    """Phase A contact test: a 1-DOF articulated finger tip on a velocity-driven prismatic joint, sweeping along
    world -Y at a fixed height above the bolt axis so it clips the hex corners tangentially. A real body with
    mass and a force-limited drive: the nut only moves through contact."""
    g = sc.g
    bx, by, bz = sc.bolt_pos
    xn = bx + sc.nut_q0 + (g.nut_zmin + g.nut_zmax) / 2
    zline = bz + line_offset
    root = UsdGeom.Xform.Define(stage, "/World/Pusher").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateSolverPositionIterationCountAttr(32)
    base = UsdGeom.Xform.Define(stage, "/World/Pusher/Base").GetPrim()
    set_xform(base, pos=(xn, start_y, zline))
    UsdPhysics.RigidBodyAPI.Apply(base)
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/Pusher/Base/Fixed")
    fj.CreateBody1Rel().SetTargets([base.GetPath()])
    fj.CreateLocalPos0Attr(Gf.Vec3f(xn, start_y, zline))
    tip = UsdGeom.Xform.Define(stage, "/World/Pusher/Tip").GetPrim()
    set_xform(tip, pos=(xn, start_y, zline))
    UsdPhysics.RigidBodyAPI.Apply(tip)
    PhysxSchema.PhysxRigidBodyAPI.Apply(tip).CreateEnableCCDAttr(True)
    UsdPhysics.MassAPI.Apply(tip).CreateMassAttr(0.02)
    cap = UsdGeom.Capsule.Define(stage, "/World/Pusher/Tip/Pad")
    cap.CreateRadiusAttr(tip_r)
    cap.CreateHeightAttr(0.012)
    cap.CreateAxisAttr("X")
    collision_tune(cap.GetPrim())
    bind_physics(cap.GetPrim(), UsdShade.Material.Get(stage, sc.paths["pm_tip"]))
    bind(cap.GetPrim(), UsdShade.Material.Get(stage, sc.paths["m_skin"]))
    pj = UsdPhysics.PrismaticJoint.Define(stage, "/World/Pusher/Tip/Slide")
    pj.CreateAxisAttr("Y")
    pj.CreateBody0Rel().SetTargets([base.GetPath()])
    pj.CreateBody1Rel().SetTargets([tip.GetPath()])
    pj.CreateLowerLimitAttr(-0.08)
    pj.CreateUpperLimitAttr(0.01)
    d = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(200.0)
    d.CreateMaxForceAttr(max_force)
    d.CreateTargetVelocityAttr(-speed)
    sc.paths.update(pusher="/World/Pusher", pusher_joint=str(pj.GetPath()))
    return xn, zline


# --------------------------------------------------------------------------- Phase B: the finger
@dataclass
class FingerSpec:
    """A 4-DOF articulated finger: carriage slide along the bolt (x) + MCP/PIP/DIP flexion about x.
    Link lengths and masses are close to a human index finger; limits are those of a small fast servo finger."""
    L1: float = 0.040
    L2: float = 0.025
    L3: float = 0.018
    r1: float = 0.0065
    r2: float = 0.0060
    r3: float = 0.0055          # distal pad radius = fingertip radius
    m1: float = 0.012
    m2: float = 0.007
    m3: float = 0.006
    m_carriage: float = 0.08
    # drives (SI). Torque limits + joint speed limits are what bound the flick speed.
    k_slide: float = 600.0       # N/m
    c_slide: float = 14.0
    f_slide: float = 15.0        # N
    k_mcp: float = 2.0           # N*m/rad
    c_mcp: float = 0.02
    tau_mcp: float = 0.4         # N*m
    v_mcp: float = 15.0          # rad/s  -> fingertip <= ~1.2 m/s
    k_pip: float = 1.0
    c_pip: float = 0.01
    tau_pip: float = 0.2
    v_pip: float = 10.0
    slide_range: tuple = (-0.012, 0.012)


def finger_reach(fs: FingerSpec, pip, dip):
    """Distance knuckle -> fingertip-sphere center, and the angle of that vector from the proximal axis."""
    y = fs.L1 * 0 + fs.L2 * math.sin(pip) + fs.L3 * math.sin(pip + dip)
    z = fs.L1 + fs.L2 * math.cos(pip) + fs.L3 * math.cos(pip + dip)
    return math.hypot(y, z), math.atan2(y, z)


def _drive(prim, kind, k, c, fmax, vmax=None, angular=False):
    d = UsdPhysics.DriveAPI.Apply(prim, kind)
    s = math.pi / 180 if angular else 1.0      # USD angular drive gains are per degree
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(k * s)
    d.CreateDampingAttr(c * s)
    d.CreateMaxForceAttr(fmax)
    d.CreateTargetPositionAttr(0.0)
    if vmax is not None:
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(math.degrees(vmax) if angular else vmax)


def _body(stage, path, pos, mass, inertia=None, quat=(1, 0, 0, 0)):
    p = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_xform(p, pos=pos, quat=quat)
    UsdPhysics.RigidBodyAPI.Apply(p)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(mass)
    if inertia is not None:
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return p


def _capsule(stage, path, r, h, center_z, mat_vis, mat_phys, collide=True):
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateRadiusAttr(r)
    c.CreateHeightAttr(h)
    c.CreateAxisAttr("Z")
    set_xform(c.GetPrim(), pos=(0, 0, center_z))
    bind(c.GetPrim(), mat_vis)
    if collide:
        collision_tune(c.GetPrim(), contact=0.0008)
        bind_physics(c.GetPrim(), mat_phys)
    return c


def _revolute(stage, path, b0, b1, pos0, axis="X"):
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([b0.GetPath()])
    j.CreateBody1Rel().SetTargets([b1.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return j


def add_finger(stage, sc: FlickScene, fs: FingerSpec, knuckle, mcp0_deg=50.0):
    """Knuckle (MCP axis, along world x) at `knuckle`; at zero angles the finger hangs straight down (-z).
    Joint order: slide_x, mcp, pip, dip. Positive mcp swings the fingertip toward +y.
    The links are authored already swung to mcp0_deg so the finger never starts inside the nut."""
    kx, ky, kz = knuckle
    th = math.radians(mcp0_deg)
    qrot = (math.cos(th / 2), math.sin(th / 2), 0.0, 0.0)

    def down(L):   # knuckle + Rx(th) * (0, 0, -L)
        return (kx, ky + L * math.sin(th), kz - L * math.cos(th))
    m_alu = preview_material(stage, "/World/Looks/Anodized", (0.55, 0.57, 0.60), 0.9, 0.35)
    m_skin = UsdShade.Material.Get(stage, sc.paths["m_skin"])
    m_link = preview_material(stage, "/World/Looks/LinkGraphite", (0.12, 0.12, 0.13), 0.2, 0.5)
    pm_tip = UsdShade.Material.Get(stage, sc.paths["pm_tip"])
    root = UsdGeom.Xform.Define(stage, "/World/Finger").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    pa = PhysxSchema.PhysxArticulationAPI.Apply(root)
    pa.CreateEnabledSelfCollisionsAttr(False)
    pa.CreateSolverPositionIterationCountAttr(32)
    pa.CreateSolverVelocityIterationCountAttr(4)

    base = _body(stage, "/World/Finger/Base", (kx, ky, kz + 0.02), 0.2)
    blk = UsdGeom.Cube.Define(stage, "/World/Finger/Base/Rail")
    blk.CreateSizeAttr(1.0)
    set_xform(blk.GetPrim(), scale=(0.06, 0.016, 0.008))
    bind(blk.GetPrim(), m_alu)
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/Finger/Base/Fixed")
    fj.CreateBody1Rel().SetTargets([base.GetPath()])
    fj.CreateLocalPos0Attr(Gf.Vec3f(kx, ky, kz + 0.02))

    car = _body(stage, "/World/Finger/Carriage", (kx, ky, kz), fs.m_carriage, (2e-5, 2e-5, 2e-5))
    cb = UsdGeom.Cube.Define(stage, "/World/Finger/Carriage/Block")
    cb.CreateSizeAttr(1.0)
    set_xform(cb.GetPrim(), pos=(0, 0, 0.008), scale=(0.018, 0.02, 0.016))
    bind(cb.GetPrim(), m_alu)
    sj = UsdPhysics.PrismaticJoint.Define(stage, "/World/Finger/Carriage/SlideX")
    sj.CreateAxisAttr("X")
    sj.CreateBody0Rel().SetTargets([base.GetPath()])
    sj.CreateBody1Rel().SetTargets([car.GetPath()])
    sj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, -0.02))
    sj.CreateLowerLimitAttr(fs.slide_range[0])
    sj.CreateUpperLimitAttr(fs.slide_range[1])
    _drive(sj.GetPrim(), "linear", fs.k_slide, fs.c_slide, fs.f_slide)

    def inertia_rod(m, L, r):
        return (m * (L * L / 12 + r * r / 4), m * (L * L / 12 + r * r / 4), m * r * r / 2)

    p1 = _body(stage, "/World/Finger/Proximal", (kx, ky, kz), fs.m1, inertia_rod(fs.m1, fs.L1, fs.r1), qrot)
    UsdPhysics.MassAPI(p1).CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -fs.L1 / 2))
    _capsule(stage, "/World/Finger/Proximal/Shell", fs.r1, fs.L1, -fs.L1 / 2, m_link, pm_tip)
    p2 = _body(stage, "/World/Finger/Middle", down(fs.L1), fs.m2, inertia_rod(fs.m2, fs.L2, fs.r2), qrot)
    UsdPhysics.MassAPI(p2).CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -fs.L2 / 2))
    _capsule(stage, "/World/Finger/Middle/Shell", fs.r2, fs.L2, -fs.L2 / 2, m_link, pm_tip)
    p3 = _body(stage, "/World/Finger/Distal", down(fs.L1 + fs.L2), fs.m3, inertia_rod(fs.m3, fs.L3, fs.r3), qrot)
    UsdPhysics.MassAPI(p3).CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -fs.L3 / 2))
    _capsule(stage, "/World/Finger/Distal/Pad", fs.r3, fs.L3, -fs.L3 / 2, m_skin, pm_tip)
    PhysxSchema.PhysxRigidBodyAPI.Apply(p3).CreateEnableCCDAttr(True)

    j1 = _revolute(stage, "/World/Finger/Proximal/MCP", car, p1, (0, 0, 0))
    j1.CreateLowerLimitAttr(-75.0)
    j1.CreateUpperLimitAttr(75.0)
    _drive(j1.GetPrim(), "angular", fs.k_mcp, fs.c_mcp, fs.tau_mcp, fs.v_mcp, angular=True)
    j2 = _revolute(stage, "/World/Finger/Middle/PIP", p1, p2, (0, 0, -fs.L1))
    j2.CreateLowerLimitAttr(-5.0)
    j2.CreateUpperLimitAttr(100.0)
    _drive(j2.GetPrim(), "angular", fs.k_pip, fs.c_pip, fs.tau_pip, fs.v_pip, angular=True)
    j3 = _revolute(stage, "/World/Finger/Distal/DIP", p2, p3, (0, 0, -fs.L2))
    j3.CreateLowerLimitAttr(-5.0)
    j3.CreateUpperLimitAttr(90.0)
    _drive(j3.GetPrim(), "angular", fs.k_pip, fs.c_pip, fs.tau_pip, fs.v_pip, angular=True)
    sc.paths.update(finger="/World/Finger", finger_joints=[str(sj.GetPath()), str(j1.GetPath()), str(j2.GetPath()),
                                                           str(j3.GetPath())])
    return root


def add_free_nut(stage, sc: FlickScene, bolt_ref_unused, nut_ref, park=(0.35, 0.22, 0.003)):
    """A second, ordinary rigid-body nut parked on the bench out of view. When the articulated nut reaches the end
    of the thread we copy its pose and velocity onto this one and hide the articulated one: the nut leaves the
    bolt and falls under real dynamics (state handoff, not animation). Never collides with the bolt/thread nut."""
    g = sc.g
    p = UsdGeom.Xform.Define(stage, "/World/FreeNut").GetPrim()
    set_xform(p, pos=park, quat=(1, 0, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(p)
    m, izz, ixx = nut_mass_inertia(g)
    mm = UsdPhysics.MassAPI.Apply(p)
    mm.CreateMassAttr(m)
    mm.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, izz))
    mm.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, (g.nut_zmin + g.nut_zmax) / 2))
    geom = UsdGeom.Xform.Define(stage, "/World/FreeNut/Geom").GetPrim()
    geom.GetReferences().AddReference(nut_ref)
    set_xform(geom, scale=(0.001, 0.001, 0.001))
    for q in Usd.PrimRange(geom):
        if q.IsA(UsdGeom.Mesh):
            collision_tune(q)
            UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr("convexHull")
    bind(geom, UsdShade.Material.Get(stage, "/World/Looks/YellowZinc"))
    mk = stage.GetPrimAtPath("/World/NutAssembly/Nut/Mark")
    if mk:
        c = UsdGeom.Cube.Define(stage, "/World/FreeNut/Mark")
        c.CreateSizeAttr(1.0)
        for op in UsdGeom.Xformable(mk).GetOrderedXformOps():
            UsdGeom.Xformable(c).AddXformOp(op.GetOpType(), op.GetPrecision()).Set(op.Get())
        bind(c.GetPrim(), UsdShade.Material.Get(stage, "/World/Looks/IndexRed"))
    fp = UsdPhysics.FilteredPairsAPI.Apply(p)
    fp.CreateFilteredPairsRel().SetTargets([Sdf.Path(sc.paths["nut"]), Sdf.Path(sc.paths["bolt"])])
    sc.paths.update(free_nut="/World/FreeNut", free_nut_park=park)
    return p
