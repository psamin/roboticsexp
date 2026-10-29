"""Penalty-shot distributions (curriculum levels) and a vectorised sampler.

Shots are rolling: the ball starts on the ground near the penalty mark with a launch velocity aimed at a point on
the goal line, and a matching spin so it rolls without slipping.
"""
from dataclasses import dataclass

import torch

from . import field as F


@dataclass(frozen=True)
class ShotLevel:
    name: str
    speed: tuple[float, float]          # launch speed range, m/s
    target_half_width: float            # on-target aim point |y| on the goal line
    off_target_prob: float              # share of shots aimed wide of the posts
    start_jitter: tuple[float, float]   # ball placement noise around the mark (x, y), uniform half-ranges
    keeper_start: float                 # keeper starts uniformly in [-k, k]
    obs_noise_pos: float                # std of ball-position observation noise, m
    obs_noise_vel: float                # std of ball-velocity observation noise, m/s
    latency_prob: float                 # share of episodes with one control step of actuation latency


LEVELS = (
    ShotLevel("beginner", (3.0, 4.0), 0.60, 0.00, (0.05, 0.05), 0.05, 0.00, 0.00, 0.0),
    ShotLevel("intermediate", (3.5, 5.0), 1.00, 0.00, (0.10, 0.15), 0.25, 0.00, 0.00, 0.0),
    ShotLevel("advanced", (4.0, 6.0), 1.35, 0.10, (0.15, 0.25), 0.50, 0.02, 0.10, 0.5),
)
LEVEL_NAMES = tuple(lv.name for lv in LEVELS)
OFF_TARGET_RANGE = (1.75, 2.30)      # |y| aim range for wide shots (posts' inner edge is at 1.5)


def _table(device):
    rows = [[*lv.speed, lv.target_half_width, lv.off_target_prob, *lv.start_jitter, lv.keeper_start,
             lv.obs_noise_pos, lv.obs_noise_vel, lv.latency_prob] for lv in LEVELS]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def sample_shots(levels: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Sample one shot per entry of `levels` (long tensor of level indices)."""
    device = levels.device
    n = levels.numel()
    p = _table(device)[levels]

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)

    start_x = F.PENALTY_X + u(-p[:, 4], p[:, 4])
    start_y = u(-p[:, 5], p[:, 5])
    on_target = torch.rand(n, device=device, generator=generator) >= p[:, 3]
    side = torch.where(torch.rand(n, device=device, generator=generator) < 0.5, -1.0, 1.0)
    aim_on = u(-p[:, 2], p[:, 2])
    aim_off = side * u(torch.full_like(side, OFF_TARGET_RANGE[0]), torch.full_like(side, OFF_TARGET_RANGE[1]))
    target_y = torch.where(on_target, aim_on, aim_off)
    speed = u(p[:, 0], p[:, 1])
    d = torch.stack([-start_x, target_y - start_y], dim=-1)
    vel = speed[:, None] * d / d.norm(dim=-1, keepdim=True)
    keeper_y = u(-p[:, 6], p[:, 6])
    latency = (torch.rand(n, device=device, generator=generator) < p[:, 9]).long()
    return {
        "level": levels.clone(),
        "start": torch.stack([start_x, start_y], dim=-1),
        "target_y": target_y,
        "speed": speed,
        "vel": vel,
        "on_target": on_target,
        "keeper_y": keeper_y,
        "latency": latency,
        "obs_noise_pos": p[:, 7],
        "obs_noise_vel": p[:, 8],
    }


def rolling_spin(vel_xy: torch.Tensor, radius: float = F.BALL_RADIUS) -> torch.Tensor:
    """Angular velocity for rolling without slipping on z-up ground: w = (-vy/r, vx/r, 0)."""
    return torch.stack([-vel_xy[:, 1] / radius, vel_xy[:, 0] / radius, torch.zeros_like(vel_xy[:, 0])], dim=-1)


def time_to_reach(dist: torch.Tensor, vmax: float, amax: float) -> torch.Tensor:
    """Minimum time to cover `dist` from rest with speed and acceleration limits (no need to stop)."""
    d_acc = vmax * vmax / (2 * amax)
    return torch.where(dist <= d_acc, torch.sqrt(2 * dist / amax), vmax / amax + (dist - d_acc) / vmax)


def reachable(shots: dict[str, torch.Tensor], vmax: float, amax: float, reaction_s: float, ctrl_dt: float,
              margin: float = 0.05):
    """Whether the keeper can, in principle, block this shot.

    Straight-line prediction at the contact plane; the ball's arrival time ignores rolling resistance, which only
    gives the keeper more time, so this errs toward calling shots unreachable.
    """
    sx, sy = shots["start"][:, 0], shots["start"][:, 1]
    y_c = sy + (shots["target_y"] - sy) * (sx - F.CONTACT_X) / sx
    need = (y_c - shots["keeper_y"]).abs() - (F.PANEL_HALF_WIDTH + F.BALL_RADIUS - margin)
    need = need.clamp(min=0.0)
    in_range = y_c.abs() <= F.KEEPER_COVER - margin
    t_ball = (sx - F.CONTACT_X) / shots["vel"][:, 0].abs()
    t_keeper = time_to_reach(need, vmax, amax) + reaction_s + shots["latency"].float() * ctrl_dt
    return in_range & (t_keeper <= t_ball)
