"""SO-101 arm + table-tennis paddle: the MuJoCo Menagerie model with a paddle added, and batched torch kinematics.

The MuJoCo model is the single source of geometry and mass: `build_model()` loads Menagerie's `robotstudio_so101`
and attaches a paddle body to the gripper. `ArmKinematics` copies every body's frame, mass and inertia out of it and
runs forward kinematics, gravity torques and the joint-space inertia diagonal for thousands of arms at once.

Paddle frame: origin at the centre of the blade's top (hitting) surface; x along the handle, pointing away from the
gripper (forward); z = the face normal. The handle is clamped in the jaws; the jaws close along the face normal, so
the blade's flat sides face up and down when the gripper points forward.
"""
from dataclasses import dataclass
import os

import numpy as np
import torch

ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "robotstudio_so101",
                     "so101.xml")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_CLOSED_ON_HANDLE = 0.25          # rad; jaw angle when clamped on a ~25 mm handle (visual only, mass is tiny)


@dataclass
class PaddleSpec:
    reach: float = 0.13          # m, blade centre beyond the gripper's fingertip site, along the handle
    lift: float = 0.0            # m, blade surface above the fingertip site, along the face normal
    blade_a: float = 0.0785      # m, blade half-length along the handle (157 mm blade)
    blade_b: float = 0.075       # m, blade half-width (150 mm)
    blade_mass: float = 0.12     # kg, blade + rubber
    handle_mass: float = 0.05    # kg
    handle_len: float = 0.10     # m


def build_spec(paddle: PaddleSpec = PaddleSpec(), with_ball: bool = False):
    import mujoco
    spec = mujoco.MjSpec.from_file(ASSET)
    g = spec.body("gripper")
    site = next(s for s in spec.sites if s.name == "gripperframe")
    # gripper frame: z points back along the tool (wrist_roll axis), x is the jaws' closing direction
    origin = np.array(site.pos) + paddle.reach * np.array([0.0, 0.0, -1.0]) + paddle.lift * np.array([1.0, 0, 0])
    body = g.add_body(name="paddle", pos=origin.tolist())
    rot = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])     # paddle x, y, z in gripper coordinates
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    body.quat = quat.tolist()
    body.add_geom(name="blade", type=mujoco.mjtGeom.mjGEOM_BOX, size=[paddle.blade_a, paddle.blade_b, 0.0035],
                  pos=[0, 0, -0.0035], mass=paddle.blade_mass, rgba=[0.05, 0.05, 0.05, 1])
    body.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[paddle.handle_len / 2, 0.0125, 0.0125],
                  pos=[-paddle.blade_a - paddle.handle_len / 2 + 0.01, 0, -0.0035], mass=paddle.handle_mass,
                  rgba=[0.55, 0.35, 0.2, 1])
    if with_ball:
        b = spec.worldbody.add_body(name="ball", pos=[0.35, 0, 0.5])
        b.add_freejoint(name="ball")
        b.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.0027,
                   rgba=[1.0, 0.45, 0.05, 1])
    return spec


def build_model(paddle: PaddleSpec = PaddleSpec(), with_ball: bool = False):
    return build_spec(paddle, with_ball).compile()


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (N, 3, 3) about unit axes (3,) or (N, 3) by angles (N,) (Rodrigues)."""
    a = axis.expand(angle.shape[0], 3)
    c, s = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


class ArmKinematics:
    """Batched kinematics of the SO-101 + paddle, extracted from the compiled MuJoCo model.

    Per-environment variation: body masses, the paddle's mass and centre of mass, and the paddle mount transform.
    """

    def __init__(self, model, num_envs: int, device="cpu"):
        import mujoco
        self.N, self.device = num_envs, torch.device(device)
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        bodies = [b for b in range(1, model.nbody) if model.body(b).name != "ball"]
        self.names = [model.body(b).name for b in bodies]
        index = {b: i for i, b in enumerate(bodies)}
        self.parent = [index.get(int(model.body_parentid[b]), -1) for b in bodies]
        self.pos0 = t(model.body_pos[bodies])                                       # (B, 3) in parent frame
        self.rot0 = t(np.stack([_quat_to_mat(q) for q in model.body_quat[bodies]]))  # (B, 3, 3)
        self.ipos0 = t(model.body_ipos[bodies])
        self.irot = t(np.stack([_quat_to_mat(q) for q in model.body_iquat[bodies]]))
        self.inertia0 = t(model.body_inertia[bodies])
        self.mass0 = t(model.body_mass[bodies])
        # joints: the five arm joints drive bodies; the gripper is held at a fixed angle
        self.joint_body, self.fixed = {}, {}
        for b in bodies:
            if model.body_jntnum[b] == 0:
                continue
            j = int(model.body_jntadr[b])
            name, axis = model.joint(j).name, model.jnt_axis[j]
            assert np.allclose(model.jnt_pos[j], 0), "joint anchors are at body origins in this model"
            if name in ARM_JOINTS:
                self.joint_body[index[b]] = (ARM_JOINTS.index(name), t(axis))
            else:
                self.fixed[index[b]] = axis_angle(t(axis), torch.full((1,), GRIPPER_CLOSED_ON_HANDLE,
                                                                       device=self.device))[0]
        self.lower = t([model.jnt_range[model.joint(n).id][0] for n in ARM_JOINTS])
        self.upper = t([model.jnt_range[model.joint(n).id][1] for n in ARM_JOINTS])
        self.armature_model = t([model.dof_armature[model.joint(n).dofadr[0]] for n in ARM_JOINTS])
        self.paddle = index[model.body("paddle").id]
        # which bodies each arm joint carries (the joint's own body and all its descendants)
        B = len(bodies)
        jb = {ji: bi for bi, (ji, _) in self.joint_body.items()}
        carried = torch.zeros(5, B, device=self.device)
        for ji in range(5):
            for bi in range(B):
                k = bi
                while k != -1 and k != jb[ji]:
                    k = self.parent[k]
                carried[ji, bi] = float(k == jb[ji])
        self.carried = carried
        # per-environment copies (randomised by the environment)
        self.mass = self.mass0.expand(num_envs, B).clone()
        self.ipos = self.ipos0.expand(num_envs, B, 3).clone()
        self.paddle_pos = self.pos0[self.paddle].expand(num_envs, 3).clone()
        self.paddle_rot = self.rot0[self.paddle].expand(num_envs, 3, 3).clone()
        self.g = torch.tensor([0.0, 0.0, -9.81], device=self.device)

    FIELDS = ("pos0", "rot0", "ipos0", "irot", "inertia0", "mass0", "lower", "upper", "armature_model", "carried")

    def save(self, path):
        """Everything forward kinematics needs, so the real-arm runtime does not need MuJoCo."""
        torch.save({"names": self.names, "parent": self.parent, "paddle": self.paddle,
                    "joint_body": {k: (j, a.cpu()) for k, (j, a) in self.joint_body.items()},
                    "fixed": {k: v.cpu() for k, v in self.fixed.items()},
                    **{f: getattr(self, f).cpu() for f in self.FIELDS}}, path)

    @classmethod
    def load(cls, path, num_envs: int = 1, device="cpu"):
        sd = torch.load(path, map_location=device, weights_only=False)
        k = cls.__new__(cls)
        k.N, k.device = num_envs, torch.device(device)
        k.names, k.parent, k.paddle = sd["names"], sd["parent"], sd["paddle"]
        k.joint_body, k.fixed = sd["joint_body"], sd["fixed"]
        for f in cls.FIELDS:
            setattr(k, f, sd[f])
        B = len(k.names)
        k.mass = k.mass0.expand(num_envs, B).clone()
        k.ipos = k.ipos0.expand(num_envs, B, 3).clone()
        k.paddle_pos = k.pos0[k.paddle].expand(num_envs, 3).clone()
        k.paddle_rot = k.rot0[k.paddle].expand(num_envs, 3, 3).clone()
        k.g = torch.tensor([0.0, 0.0, -9.81], device=device)
        return k

    def forward(self, q: torch.Tensor):
        """q: (N, 5) arm joint angles. Returns a dict of world-frame quantities:
        body positions/rotations, joint anchors and axes, centres of mass, and the paddle frame."""
        N = q.shape[0]
        pos, rot = [None] * len(self.names), [None] * len(self.names)
        for i, p in enumerate(self.parent):
            if i == self.paddle:
                lp, lr = self.paddle_pos[:N], self.paddle_rot[:N]
            else:
                lp, lr = self.pos0[i].expand(N, 3), self.rot0[i].expand(N, 3, 3)
            if p == -1:
                P, Rp = lp, lr
            else:
                P = pos[p] + (rot[p] @ lp.unsqueeze(-1)).squeeze(-1)
                Rp = rot[p] @ lr
            if i in self.joint_body:
                ji, axis = self.joint_body[i]
                Rp = Rp @ axis_angle(axis, q[:, ji])
            elif i in self.fixed:
                Rp = Rp @ self.fixed[i]
            pos[i], rot[i] = P, Rp
        pos, rot = torch.stack(pos, 1), torch.stack(rot, 1)              # (N, B, 3), (N, B, 3, 3)
        com = pos + (rot @ self.ipos[:N].unsqueeze(-1)).squeeze(-1)
        anchors, axes = [], []
        for ji in range(5):
            bi = next(b for b, (j, _) in self.joint_body.items() if j == ji)
            anchors.append(pos[:, bi])
            axes.append((rot[:, bi] @ self.joint_body[bi][1]))
        return {"pos": pos, "rot": rot, "com": com, "anchor": torch.stack(anchors, 1), "axis": torch.stack(axes, 1),
                "paddle_o": pos[:, self.paddle], "paddle_R": rot[:, self.paddle]}

    def gravity_and_inertia(self, fk):
        """Gravity torque on each arm joint (the torque the servo must supply to hold still) and the diagonal of the
        joint-space inertia matrix (body inertia only; servo armature is added by the servo model)."""
        N = fk["com"].shape[0]
        r = fk["com"].unsqueeze(1) - fk["anchor"].unsqueeze(2)                   # (N, 5, B, 3)
        a = fk["axis"].unsqueeze(2)                                               # (N, 5, 1, 3)
        m = self.mass[:N].unsqueeze(1)                                            # (N, 1, B)
        # torque gravity exerts about the axis: ((c - p) x m g) . a
        g = self.g if self.g.dim() == 1 else self.g[:N, None, None, :]           # per-env gravity (table tilt)
        tg = (torch.cross(r, g.expand_as(r), dim=-1) * a).sum(-1) * m             # (N, 5, B)
        tau_g = -(tg * self.carried).sum(-1)                                      # servo must cancel it
        lever2 = torch.cross(a.expand_as(r), r, dim=-1).pow(2).sum(-1)            # |a x r|^2
        Rw = fk["rot"] @ self.irot                                                # principal axes in world
        proj = (Rw.unsqueeze(1) * fk["axis"][:, :, None, :, None]).sum(-2)       # (N,5,B,3): a . principal axis k
        rot_part = (proj.pow(2) * self.inertia0).sum(-1)
        M = ((m * lever2 + rot_part) * self.carried).sum(-1)
        return tau_g, M
