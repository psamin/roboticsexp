"""Bounded lateral-velocity controller shared by every policy.

The action a in [-1, 1] asks for a lateral velocity a * vmax. The controller turns that into a velocity reference
that never exceeds vmax, changes by at most amax per second, and slows down early enough to stop at the travel
limits. The reference then becomes wheel-speed targets on the robot (see environments/goalkeeper_env.py).
"""
from dataclasses import dataclass

import torch

from . import field as F


@dataclass
class DriveConfig:
    vmax: float = 2.0             # m/s (the Jackal's rated top speed)
    amax: float = 6.0             # m/s^2
    y_limit: float = F.KEEPER_Y_LIMIT


def step_reference(action: torch.Tensor, y: torch.Tensor, v_ref: torch.Tensor, dt: float, cfg: DriveConfig):
    """Advance the velocity reference by one physics step."""
    v_cmd = action.clamp(-1.0, 1.0) * cfg.vmax
    room_pos = (cfg.y_limit - y).clamp(min=0.0)
    room_neg = (cfg.y_limit + y).clamp(min=0.0)
    v_cmd = torch.minimum(v_cmd, torch.sqrt(2 * cfg.amax * room_pos))
    v_cmd = torch.maximum(v_cmd, -torch.sqrt(2 * cfg.amax * room_neg))
    dv = (v_cmd - v_ref).clamp(-cfg.amax * dt, cfg.amax * dt)
    return v_ref + dv
