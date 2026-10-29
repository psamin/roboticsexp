"""Task definition shared by the environment, the baselines and the tests (plain PyTorch, no Isaac imports).

Frames: world z is up. "_b" quantities are in the drone's body frame. Gravity direction in the body frame,
g_b, is (0, 0, -1) when the drone is level.
"""
import math
from dataclasses import dataclass

import torch

GRAVITY = 9.81
HOVER_POINT = (0.0, 0.0, 1.0)     # stage-1 target, 1 m above the ground (env frame)

# ----------------------------------------------------------------------------------------------- actions
THRUST_TO_WEIGHT = 1.9            # Isaac Lab reference value for the Crazyflie
MOMENT_SCALE = 0.01               # N*m per unit action (reference value)


def thrust_from_action(a0: torch.Tensor, weight: float, twr: float = THRUST_TO_WEIGHT) -> torch.Tensor:
    """Hover-centred collective thrust: a0 = 0 -> weight, a0 = 1 -> twr * weight, a0 = -1 -> 0 (N)."""
    return weight * torch.where(a0 >= 0, 1.0 + a0 * (twr - 1.0), 1.0 + a0)


def action_from_thrust(thrust: torch.Tensor, weight: float, twr: float = THRUST_TO_WEIGHT) -> torch.Tensor:
    r = thrust / weight
    return torch.where(r >= 1.0, (r - 1.0) / (twr - 1.0), r - 1.0).clamp(-1.0, 1.0)


# ----------------------------------------------------------------------------------------------- observation
ACT_DIM = 4
OBS_NAMES = ("rel_x", "rel_y", "rel_z", "g_x", "g_y", "g_z", "vx", "vy", "vz", "wx", "wy", "wz",
             "a_thrust", "a_roll", "a_pitch", "a_yaw")
STATE_SCALE = torch.tensor([1.0] * 3 + [1.0] * 3 + [2.0] * 3 + [10.0] * 3)
OBS_CLIP = 5.0


def obs_dim(action_history: int = 1) -> int:
    """12 state numbers plus the last `action_history` actions (most recent first)."""
    return 12 + ACT_DIM * action_history


OBS_DIM = obs_dim(1)


def build_obs(rel_pos_b, grav_b, lin_vel_b, ang_vel_b, action_history):
    """action_history: (N, 4 * H), most recent action first."""
    state = torch.cat([rel_pos_b, grav_b, lin_vel_b, ang_vel_b], dim=-1) / STATE_SCALE.to(rel_pos_b.device)
    return torch.cat([state, action_history], dim=-1).clamp(-OBS_CLIP, OBS_CLIP)


def unpack_obs(obs):
    raw = obs[:, :12] * STATE_SCALE.to(obs.device)
    return raw[:, 0:3], raw[:, 3:6], raw[:, 6:9], raw[:, 9:12], obs[:, 12:16]


# ----------------------------------------------------------------------------------------------- rewards
@dataclass
class RewardWeights:
    distance: float = 15.0       # (1 - tanh(d / 0.8)), coarse pull toward the target
    precision: float = 5.0       # (1 - tanh(d / 0.1)), fine positioning
    lin_vel: float = 0.05        # |v|^2, multiplied up near the target
    near_target_gain: float = 4.0
    ang_vel: float = 0.01        # |w|^2
    upright: float = 1.0         # (1 + g_z): 0 when level, 2 when upside down
    effort: float = 0.05         # |a|^2 (a = 0 is hover thrust, zero torque)
    action_rate: float = 0.05    # |a - a_prev|^2
    success: float = 5.0         # one-time bonus for the first continuous hover of hold_s within radius
    crash: float = 5.0           # one-time penalty on crash / leaving the flight volume
    # unstable-flight penalties (0 in the baseline)
    ang_acc: float = 0.0         # |dw/dt / 100|^2: attitude wobble (angular acceleration, per 100 rad/s^2)
    lin_acc: float = 0.0         # |dv/dt / 10|^2: translational jerkiness (acceleration, per 10 m/s^2)


def reward_terms(w: RewardWeights, dist, lin_vel_b, ang_vel_b, grav_b, actions, prev_actions, success_now, failed,
                 dt: float, ang_acc=None, lin_acc=None) -> dict:
    near = torch.exp(-(dist / 0.3) ** 2)
    zero = torch.zeros_like(dist)
    return {
        "ang_acc": -w.ang_acc * (ang_acc / 100.0).pow(2).sum(-1) * dt if ang_acc is not None else zero,
        "lin_acc": -w.lin_acc * (lin_acc / 10.0).pow(2).sum(-1) * dt if lin_acc is not None else zero,
        "distance": w.distance * (1 - torch.tanh(dist / 0.8)) * dt,
        "precision": w.precision * (1 - torch.tanh(dist / 0.1)) * dt,
        "lin_vel": -w.lin_vel * (1 + w.near_target_gain * near) * lin_vel_b.pow(2).sum(-1) * dt,
        "ang_vel": -w.ang_vel * ang_vel_b.pow(2).sum(-1) * dt,
        "upright": -w.upright * (1 + grav_b[:, 2]) * dt,
        "effort": -w.effort * actions.pow(2).sum(-1) * dt,
        "action_rate": -w.action_rate * (actions - prev_actions).pow(2).sum(-1) * dt,
        "success": w.success * success_now.float(),
        "crash": -w.crash * failed.float(),
    }


# ----------------------------------------------------------------------------------------------- L2F-style reward
@dataclass
class L2FWeights:
    """Squared-cost reward from "Learning to Fly in Seconds" (Eschmann et al., RA-L 2024; arplaboratory/learning-to-fly):

        r = constant - scale * (position |e|^2 + orientation (1 - q_w^2) + linear_velocity |v|^2
                                + angular_velocity |w|^2 + action |a - a_hover|^2)

    Initial weights are the paper code's `reward_squared_position_only_torque`; its curriculum multiplies position by
    1.2 (cap 40), linear velocity by 1.4 (cap 1) and action by 1.4 (cap 1) at fixed intervals. Orientation here is the
    tilt-only form (1 + g_z) / 2 = sin^2(tilt / 2), which equals 1 - q_w^2 at zero yaw (heading is irrelevant to
    this task). Rewards are clipped at zero so ending an episode early is never better than flying badly.
    """
    position: float = 5.0
    orientation: float = 5.0
    linear_velocity: float = 0.01
    angular_velocity: float = 0.0
    action: float = 0.01
    scale: float = 0.5
    constant: float = 2.0
    reward_scale: float = 0.1     # overall multiplier so returns stay in the same range as the shaped reward
    non_negative: bool = True
    crash: float = 5.0
    success: float = 0.0          # optional hover bonus (0 = pure L2F)
    ang_acc: float = 0.0          # optional unstable-flight penalties, as in RewardWeights
    lin_acc: float = 0.0


def l2f_reward_terms(w: L2FWeights, err, lin_vel_b, ang_vel_b, grav_b, actions, success_now, failed, dt: float,
                     ang_acc=None, lin_acc=None) -> dict:
    k = w.reward_scale
    costs = {
        "l2f_position": w.position * err.pow(2).sum(-1),
        "l2f_orientation": w.orientation * 0.5 * (1 + grav_b[:, 2]),
        "l2f_lin_vel": w.linear_velocity * lin_vel_b.pow(2).sum(-1),
        "l2f_ang_vel": w.angular_velocity * ang_vel_b.pow(2).sum(-1),
        "l2f_action": w.action * actions.pow(2).sum(-1),
    }
    terms = {"l2f_constant": torch.full_like(costs["l2f_position"], k * w.constant)}
    terms.update({name: -k * w.scale * c for name, c in costs.items()})
    total = sum(terms.values())
    clipped = total.clamp(min=0.0) if w.non_negative else total
    terms["l2f_clip"] = clipped - total
    terms["ang_acc"] = -w.ang_acc * (ang_acc / 100.0).pow(2).sum(-1) * dt if ang_acc is not None else 0 * total
    terms["lin_acc"] = -w.lin_acc * (lin_acc / 10.0).pow(2).sum(-1) * dt if lin_acc is not None else 0 * total
    terms["success"] = w.success * success_now.float()
    terms["crash"] = -w.crash * failed.float()
    return terms


class PenaltyCurriculum:
    """Multiply chosen reward weights by fixed factors every `every` PPO iterations, up to caps (L2F-style).

    gate: escalate only while the hover-success rate of recent training episodes is at least this value, so the
    penalties tighten as fast as the policy keeps up (None = fixed schedule, as in the paper).
    """

    def __init__(self, weights, factors: dict, caps: dict, every: int, gate: float | None = None):
        self.w, self.factors, self.caps, self.every, self.gate = weights, dict(factors), dict(caps), every, gate
        self.steps = 0
        self.history = []

    def done(self):
        return all(getattr(self.w, k) >= self.caps[k] for k in self.factors)

    def step(self, iteration: int, success_rate: float) -> bool:
        if self.every <= 0 or iteration % self.every or self.done():
            return False
        if self.gate is not None and not (success_rate >= self.gate):
            return False
        for k, f in self.factors.items():
            setattr(self.w, k, min(getattr(self.w, k) * f, self.caps[k]))
        self.steps += 1
        self.history.append({"iteration": iteration, **{k: getattr(self.w, k) for k in self.factors}})
        return True

    def state_dict(self):
        return {"steps": self.steps, "history": self.history, "weights": {k: getattr(self.w, k) for k in self.factors}}

    def load_state_dict(self, sd):
        self.steps, self.history = sd["steps"], list(sd["history"])
        for k, v in sd["weights"].items():
            setattr(self.w, k, v)


# ----------------------------------------------------------------------------------------------- terminations
MIN_Z, MAX_Z, MAX_XY = 0.1, 2.5, 2.0


def failure(pos_env, grav_b):
    """Crash (too low, or tilted past 90 degrees) or leaving the flight volume. Time limits are separate."""
    crashed = (pos_env[:, 2] < MIN_Z) | (grav_b[:, 2] > 0.0)
    out = (pos_env[:, :2].norm(dim=-1) > MAX_XY) | (pos_env[:, 2] > MAX_Z)
    return crashed, out


# ----------------------------------------------------------------------------------------------- curriculum stages
@dataclass(frozen=True)
class Stage:
    name: str
    target_xy: float           # target x, y uniform in [-r, r] around the hover point
    target_z: tuple            # target height range
    start_offset: float        # start position: hover point + uniform [-s, s]^3
    tilt: float                # initial roll / pitch uniform [-t, t] rad
    yaw: float                 # initial yaw uniform [-y, y] rad
    start_vel: float           # initial linear velocity uniform [-v, v]^3
    wind: float                # constant horizontal disturbance force up to this fraction of the weight
    start_ang_vel: float = 0.0  # initial body rates uniform [-w, w]^3 rad/s


STAGES = (
    Stage("hover", 0.0, (1.0, 1.0), 0.10, 0.10, 0.2, 0.0, 0.0),
    Stage("reach", 0.5, (0.75, 1.25), 0.15, 0.15, 1.0, 0.1, 0.0),
    Stage("robust", 1.0, (0.5, 1.5), 0.30, 0.30, math.pi, 0.3, 0.05),
    # evaluation only (never trained on): L2F-style recovery from large tilts and initial motion
    Stage("recovery", 0.0, (1.0, 1.0), 0.20, math.pi / 4, math.pi, 1.0, 0.0, 1.0),
)
STAGE_NAMES = tuple(s.name for s in STAGES)
N_TRAIN_STAGES = 3


def sample_conditions(stages: torch.Tensor, generator=None) -> dict:
    """Initial state, target and disturbance for each entry of `stages` (long tensor), in the env frame."""
    dev, n = stages.device, stages.numel()
    rows = torch.tensor([[s.target_xy, *s.target_z, s.start_offset, s.tilt, s.yaw, s.start_vel, s.wind,
                          s.start_ang_vel] for s in STAGES], device=dev)[stages]
    u = lambda *shape: torch.rand(*shape, device=dev, generator=generator)
    sym = lambda half, k: (2 * u(n, k) - 1) * half.unsqueeze(-1)
    hover = torch.tensor(HOVER_POINT, device=dev)
    target = hover + torch.cat([sym(rows[:, 0], 2), (rows[:, 1] + (rows[:, 2] - rows[:, 1]) * u(n)).unsqueeze(-1)
                                - hover[2]], dim=-1)
    start = hover + sym(rows[:, 3], 3)
    rpy = torch.cat([sym(rows[:, 4], 2), sym(rows[:, 5], 1)], dim=-1)
    angle = 2 * math.pi * u(n)
    wind_mag = rows[:, 7] * u(n)
    wind = torch.stack([wind_mag * torch.cos(angle), wind_mag * torch.sin(angle), torch.zeros(n, device=dev)], -1)
    return {"stage": stages.clone(), "target": target, "start": start, "rpy": rpy,
            "start_vel": sym(rows[:, 6], 3), "wind_frac": wind, "start_ang_vel": sym(rows[:, 8], 3)}


def quat_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) quaternion from roll, pitch, yaw (extrinsic x-y-z)."""
    r, p, y = (rpy[:, i] / 2 for i in range(3))
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dim=-1)


# ----------------------------------------------------------------------------------------------- PD baseline
@dataclass
class PDGains:
    kp_pos: float = 8.0          # 1/s^2
    kd_pos: float = 5.0          # 1/s
    max_acc: float = 6.0         # m/s^2, commanded acceleration limit
    kp_att: float = 250.0        # 1/s^2
    kd_att: float = 25.0         # 1/s
    kd_yaw: float = 10.0         # 1/s


class PDHoverController:
    """Cascaded PD from the same 16 observations PPO gets: position PD -> desired thrust vector -> attitude PD.

    desired specific force f = a_cmd - g * g_b  (body frame), a_cmd = kp * e - kd * v (clipped)
    thrust = m * f_z ;  tilt error axis = z_b x f_hat = (-f_y, f_x, 0)/|f|
    torque = I * (kp_att * axis - kd_att * w)
    """

    def __init__(self, mass: float, inertia_diag, weight: float, gains: PDGains | None = None):
        self.m, self.I, self.W = mass, torch.as_tensor(inertia_diag, dtype=torch.float32), weight
        self.g = gains or PDGains()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        g = self.g
        e, gb, v, w, _ = unpack_obs(obs)
        a_cmd = g.kp_pos * e - g.kd_pos * v
        norm = a_cmd.norm(dim=-1, keepdim=True)
        a_cmd = a_cmd * (g.max_acc / norm.clamp(min=g.max_acc))
        f = a_cmd - GRAVITY * gb
        f_hat = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        thrust = (self.m * f[:, 2]).clamp(min=0.0)
        axis = torch.stack([-f_hat[:, 1], f_hat[:, 0]], dim=-1)
        I = self.I.to(obs.device)
        tau_xy = I[:2] * (g.kp_att * axis - g.kd_att * w[:, :2])
        tau_z = -I[2] * g.kd_yaw * w[:, 2:3]
        torques = torch.cat([tau_xy, tau_z], dim=-1)
        a0 = action_from_thrust(thrust, self.W).unsqueeze(-1)
        return torch.cat([a0, (torques / MOMENT_SCALE).clamp(-1, 1)], dim=-1)


class RandomPolicy:
    def __init__(self, generator=None):
        self.generator = generator

    def __call__(self, obs):
        return torch.rand(obs.shape[0], ACT_DIM, device=obs.device, generator=self.generator) * 2 - 1
