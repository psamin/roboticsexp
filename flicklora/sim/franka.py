"""Franka Panda for the flick task: Isaac Sim asset + AssembleNIST-style control + numpy kinematics/IK.

Control mirrors AssembleNIST (assembly_policy/sim/scene.py): libfranka-like PD gains
kp (600,600,600,600,250,150,50) N*m/rad, kd (50,50,50,50,30,25,15), torque limits 87 N*m x4 / 12 N*m x3,
gravity compensation (here: gravity off on the robot links), real joint speed limits (2.175 / 2.61 rad/s).

Kinematics follow the franka_description URDF exactly (the Isaac asset is converted from it):
  joint i: origin xyz/rpy below, then a rotation q_i about local z.  hand = link7 * T(z=0.107) * Rz(-pi/4).
"""
from __future__ import annotations

import math

import numpy as np

ASSET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
KP = np.array([600, 600, 600, 600, 250, 150, 50], float)
KD = np.array([50, 50, 50, 50, 30, 25, 15], float)
TAU = np.array([87, 87, 87, 87, 12, 12, 12], float)
VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], float)
QMIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
QMAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
P2 = math.pi / 2
_JOINTS = [((0, 0, 0.333), (0, 0, 0)), ((0, 0, 0), (-P2, 0, 0)), ((0, -0.316, 0), (P2, 0, 0)),
           ((0.0825, 0, 0), (P2, 0, 0)), ((-0.0825, 0.384, 0), (-P2, 0, 0)), ((0, 0, 0), (P2, 0, 0)),
           ((0.088, 0, 0), (P2, 0, 0))]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_ORIG = []
for xyz, rpy in _JOINTS:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    _ORIG.append(T)
_HAND = np.eye(4)
_HAND[:3, :3] = _rpy(0, 0, -math.pi / 4)
_HAND[2, 3] = 0.107


def fk(q, base=np.eye(4), tip_z=0.1121):
    """Returns dict: pivot6 (joint-6 origin), axis6, hand (4x4), tip (closed fingertip bottom point)."""
    T = base.copy()
    out = {}
    for i in range(7):
        T = T @ _ORIG[i]
        if i == 5:
            out["pivot6"], out["axis6"] = T[:3, 3].copy(), T[:3, 2].copy()
        c, s = math.cos(q[i]), math.sin(q[i])
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T = T @ Rz
    H = T @ _HAND
    out["hand"] = H
    out["tip"] = H[:3, 3] + H[:3, 2] * tip_z
    return out


def ik(q0, residual, iters=200, lam=1e-3, tol=1e-7):
    """Levenberg-Marquardt on an arbitrary residual(q) -> vector, numeric Jacobian, joint limits clamped."""
    q = np.clip(np.asarray(q0, float).copy(), QMIN, QMAX)
    r = residual(q)
    for _ in range(iters):
        J = np.empty((len(r), 7))
        for j in range(7):
            dq = np.zeros(7)
            dq[j] = 1e-6
            J[:, j] = (residual(q + dq) - r) / 1e-6
        step = np.linalg.solve(J.T @ J + lam * np.eye(7), -J.T @ r)
        qn = np.clip(q + step, QMIN, QMAX)
        rn = residual(qn)
        if rn @ rn < r @ r:
            q, r, lam = qn, rn, max(lam * 0.5, 1e-7)
        else:
            lam *= 10
        if r @ r < tol:
            break
    return q, float(r @ r)


def flick_config(q0, tip_target, base, tip_z, q_nom=Q_READY, w_reg=0.002):
    """Arm configuration at the bottom of a joint-6 swing: fingertip at tip_target, the joint-6 axis along world y
    (so the swing sweeps the tip along x, tangent to a nut on a y-axis bolt), the pivot->tip line vertical (this
    is the lowest point of the arc), and the finger-closing axis in the swing plane (the wide hand then lies
    along the swing, clear of the bracket behind the nut). That is exactly 7 constraints for 7 joints; the hand is
    tilted ~22 deg at the bottom because the wrist (joint 7) sits 0.088 m off the joint-6 axis."""
    tip_target = np.asarray(tip_target, float)

    def res(q):
        f = fk(q, base, tip_z)
        v = f["tip"] - f["pivot6"]
        v = v / np.linalg.norm(v)
        return np.concatenate([
            f["tip"] - tip_target,
            0.1 * (v - np.array([0, 0, -1.0])),
            0.1 * np.cross(f["axis6"], [0, 1.0, 0]),
            [0.1 * f["hand"][1, 1]],                      # hand y has no world-y component
            w_reg * (q - q_nom),
        ])
    return ik(q0, res)


def add_franka(stage, path, base_pos, asset=ASSET, mesh_variant="Quality"):
    """Reference the Isaac Franka asset, fix it to the world, apply AssembleNIST gains/limits, gravity off on links
    (= gravity compensation). Returns the joint prims dict {name: prim}."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.GetReferences().AddReference(asset)
    x = UsdGeom.Xformable(root)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*base_pos))
    vs = root.GetVariantSets()
    if "Mesh" in vs.GetNames():
        vs.GetVariantSet("Mesh").SetVariantSelection(mesh_variant)
    # The asset's link geometry is instanceable, and instanced meshes under physics-driven links do not render on
    # the DGX Spark nodes (the arm was invisible while its physics worked). De-instancing makes it render.
    for p in Usd.PrimRange(root):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    joints, links = {}, {}
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):   # the nub sits inside the closed fingers: no self-contact
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr().Set(False)
        n = p.GetName()
        if p.IsA(UsdPhysics.Joint):
            joints[n] = p
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            links[n] = p
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    for i in range(7):
        jp = joints.get(f"panda_joint{i + 1}")
        if jp is None:
            continue
        d = UsdPhysics.DriveAPI.Apply(jp, "angular")
        d.CreateStiffnessAttr().Set(float(KP[i] * math.pi / 180))     # USD angular gains are per degree
        d.CreateDampingAttr().Set(float(KD[i] * math.pi / 180))
        d.CreateMaxForceAttr().Set(float(TAU[i]))
        PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr().Set(float(math.degrees(VMAX[i])))
    for n in ("panda_finger_joint1", "panda_finger_joint2"):
        jp = joints.get(n)
        if jp is not None:      # keep the fingers shut hard: they are the flicking surface
            d = UsdPhysics.DriveAPI.Apply(jp, "linear")
            d.CreateStiffnessAttr().Set(20000.0)
            d.CreateDampingAttr().Set(200.0)
            d.CreateMaxForceAttr().Set(140.0)
            d.CreateTargetPositionAttr().Set(0.0)
    return joints, links


def fingertip_z(stage, hand_prim, finger_prims):
    """Fingertip bottom along the hand z axis, measured from the asset's own geometry (closed fingers)."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    zs = []
    for fp in finger_prims:
        rng = cache.ComputeRelativeBound(fp, hand_prim).ComputeAlignedRange()
        if not rng.IsEmpty():
            zs.append(rng.GetMax()[2])
    return max(zs) if zs else None


def add_flick_nub(stage, hand_path, tip_z, radius=0.006, protrude=0.005, phys_mat=None, vis_mat=None):
    """A silicone ball fixed under the closed fingertips (a common custom fingertip for dynamic tasks). The Franka's
    flat finger pads jam on a hex: after one corner is pushed, the next one rises into the 2 cm flat face. A round
    nub rolls over the corners like the finger's capsule did. Returns the new tip distance along the hand z axis."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    s = UsdGeom.Sphere.Define(stage, f"{hand_path}/FlickNub")
    s.CreateRadiusAttr(radius)
    UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, tip_z + protrude - radius))
    UsdPhysics.CollisionAPI.Apply(s.GetPrim())
    c = PhysxSchema.PhysxCollisionAPI.Apply(s.GetPrim())
    c.CreateContactOffsetAttr(0.0008)
    c.CreateRestOffsetAttr(0.0)
    if phys_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    if vis_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(vis_mat, UsdShade.Tokens.strongerThanDescendants)
    return tip_z + protrude
