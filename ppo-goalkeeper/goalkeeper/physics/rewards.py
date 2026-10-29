"""Reward terms. Each is returned separately so it can be logged on its own.

save / goal      +10 / -10 on the step the outcome is decided (misses and timeouts give 0).
progress         potential-based shaping toward the interception point, gamma * phi(s') - phi(s) with
                 phi = -|keeper_y - y*| and y* the predicted crossing of the contact plane, clipped to the keeper's
                 travel range. Potential-based shaping leaves the optimal policy unchanged, so the keeper cannot
                 farm it without saving balls.
action / jerk    small penalties on a^2 and (a - a_prev)^2.
"""
from dataclasses import dataclass

import torch

from . import field as F
from .outcomes import GOAL, SAVE


@dataclass
class RewardConfig:
    save: float = 10.0
    goal: float = -10.0
    progress: float = 1.0
    action: float = 0.01
    jerk: float = 0.02
    gamma: float = 0.99


def intercept_y(ball_xy: torch.Tensor, ball_vxy: torch.Tensor):
    """Predicted y where the ball centre reaches the contact plane, clipped to keeper travel; and validity."""
    approaching = (ball_vxy[:, 0] < -0.2) & (ball_xy[:, 0] > F.CONTACT_X)
    t = (ball_xy[:, 0] - F.CONTACT_X) / (-ball_vxy[:, 0]).clamp(min=0.2)
    y = (ball_xy[:, 1] + ball_vxy[:, 1] * t).clamp(-F.KEEPER_Y_LIMIT, F.KEEPER_Y_LIMIT)
    return y, approaching


def potential(keeper_y, y_star, valid):
    return torch.where(valid, -(keeper_y - y_star).abs(), torch.zeros_like(keeper_y))


def compute(cfg: RewardConfig, outcome, phi_prev, phi_now, action, prev_action) -> dict[str, torch.Tensor]:
    a, ap = action.squeeze(-1), prev_action.squeeze(-1)
    return {
        "save": cfg.save * (outcome == SAVE).float(),
        "goal": cfg.goal * (outcome == GOAL).float(),
        "progress": cfg.progress * (cfg.gamma * phi_now - phi_prev),
        "action": -cfg.action * a * a,
        "jerk": -cfg.jerk * (a - ap) ** 2,
    }
