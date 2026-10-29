"""Baseline goalkeepers. Both read the same observation vector as PPO and output the same action in [-1, 1],
which goes through the same bounded drive and the same simulated robot.
"""
import torch

from ..physics import field as F
from ..physics import observation
from ..physics.drive import DriveConfig


class AnalyticalKeeper:
    """Predict where the ball crosses the contact plane, then move there as fast as the drive limits allow.

    The command follows the braking curve v = sqrt(2 * amax * distance) so the keeper arrives without overshooting.
    When the ball is not approaching, it holds position.
    """

    def __init__(self, drive: DriveConfig | None = None, brake_margin: float = 0.9):
        self.drive = drive or DriveConfig()
        self.brake_margin = brake_margin

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        o = observation.unpack(obs)
        bx, by, bvx, bvy, ky = o["ball_x"], o["ball_y"], o["ball_vx"], o["ball_vy"], o["keeper_y"]
        approaching = (bvx < -0.2) & (bx > F.CONTACT_X - 0.05)
        t = (bx - F.CONTACT_X).clamp(min=0.0) / (-bvx).clamp(min=0.2)
        y_pred = (by + bvy * t).clamp(-self.drive.y_limit, self.drive.y_limit)
        y_des = torch.where(approaching, y_pred, ky)
        err = y_des - ky
        v_des = torch.sign(err) * torch.minimum(
            torch.full_like(err, self.drive.vmax),
            self.brake_margin * torch.sqrt(2 * self.drive.amax * err.abs()),
        )
        return (v_des / self.drive.vmax).clamp(-1, 1).unsqueeze(-1)


class RandomKeeper:
    """Uniform random action each control step."""

    def __init__(self, generator: torch.Generator | None = None):
        self.generator = generator

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.rand(obs.shape[0], 1, device=obs.device, generator=self.generator) * 2 - 1


class StillKeeper:
    """Never moves (zero action). Useful for physics checks."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 1, device=obs.device)
