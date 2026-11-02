"""Actor and critic networks.

The actor is a tanh-squashed Gaussian. It samples u ~ N(mu(s), sigma) and acts with a = tanh(u), so every action
lies in (-1, 1) without clipping. Its log-probability includes the change-of-variables term:

    log pi(a|s) = sum_i [ log N(u_i; mu_i, sigma_i) - log(1 - tanh(u_i)^2) ]

The rollout buffer stores u itself, so the update evaluates exactly the same density that produced the action.
"""
import math

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


def mlp(in_dim: int, hidden, out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


def squash_log_prob(dist: Normal, u: torch.Tensor) -> torch.Tensor:
    """log pi(tanh(u)) for u drawn from dist, summed over action dimensions (numerically stable form).

    log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
    """
    log_det = 2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))
    return (dist.log_prob(u) - log_det).sum(-1)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), init_std: float = 0.5):
        super().__init__()
        self.mean = mlp(obs_dim, tuple(hidden), act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def dist(self, obs: torch.Tensor) -> Normal:
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(self.mean(obs), log_std.exp().expand(obs.shape[0], -1))

    def sample(self, obs: torch.Tensor):
        """Returns (pre-squash sample u, action a = tanh(u), log pi(a|s))."""
        d = self.dist(obs)
        u = d.sample()
        return u, torch.tanh(u), squash_log_prob(d, u)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        super().__init__()
        self.v = mlp(obs_dim, tuple(hidden), 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)
