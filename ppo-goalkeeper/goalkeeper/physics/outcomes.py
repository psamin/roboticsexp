"""Shot outcome rules, evaluated on the simulated ball state.

GOAL     the ball wholly crosses the goal line (centre behind x = -r) between the posts and under the bar.
SAVE     the keeper touched the ball and it did not go in: it rebounded away, stopped, or went wide/out.
MISS     the keeper never touched it and it did not go in: wide of the posts, off the post, stopped, or out.
TIMEOUT  nothing was decided before the episode time limit.
"""
import torch

from . import field as F

NONE, GOAL, SAVE, MISS, TIMEOUT = 0, 1, 2, 3, 4
OUTCOME_NAMES = {NONE: "none", GOAL: "goal", SAVE: "save", MISS: "miss", TIMEOUT: "timeout"}

AWAY_VX = 0.05        # m/s: ball moving back toward the kicker
STOPPED_SPEED = 0.15  # m/s
OUT_Y = 4.0           # m: ball far wide of the goal
OUT_X = F.PENALTY_X + 3.0


def classify(ball_pos: torch.Tensor, ball_vel: torch.Tensor, touched: torch.Tensor) -> torch.Tensor:
    """Outcome code per environment from the ball state (env frame) and whether the keeper has touched it."""
    x, y, z = ball_pos[:, 0], ball_pos[:, 1], ball_pos[:, 2]
    crossed = x < -F.BALL_RADIUS
    in_mouth = (y.abs() < F.GOAL_HALF_WIDTH) & (z < F.GOAL_HEIGHT)
    goal = crossed & in_mouth
    wide = crossed & ~in_mouth
    away = (ball_vel[:, 0] > AWAY_VX) & (x > 0.0)
    stopped = ball_vel[:, :2].norm(dim=-1) < STOPPED_SPEED
    out = (y.abs() > OUT_Y) | (x > OUT_X)
    finished = wide | away | stopped | out
    out_code = torch.where(touched, torch.full_like(x, SAVE, dtype=torch.long), torch.full_like(x, MISS, dtype=torch.long))
    none = torch.full_like(x, NONE, dtype=torch.long)
    return torch.where(goal, torch.full_like(none, GOAL), torch.where(finished, out_code, none))
