"""The policy's state observation: 9 normalised numbers.

The PPO policy and the analytical baseline both read this same vector (including any observation noise),
so neither sees anything the other does not.
"""
import torch

from . import field as F

OBS_DIM = 9
OBS_NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "keeper_y", "keeper_vy", "rel_y", "rel_x", "prev_action")
SCALE = torch.tensor([6.0, 1.5, 6.0, 2.0, 1.5, 2.0, 1.5, 6.0, 1.0])


def build(ball_xy, ball_vxy, keeper_y, keeper_vy, prev_action) -> torch.Tensor:
    raw = torch.stack([
        ball_xy[:, 0], ball_xy[:, 1], ball_vxy[:, 0], ball_vxy[:, 1], keeper_y, keeper_vy,
        ball_xy[:, 1] - keeper_y, ball_xy[:, 0] - F.KEEPER_X, prev_action,
    ], dim=-1)
    return raw / SCALE.to(raw.device)


def unpack(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    raw = obs * SCALE.to(obs.device)
    return {name: raw[:, i] for i, name in enumerate(OBS_NAMES)}
