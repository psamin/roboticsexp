"""Separate actor and critic MLPs."""
import math

import torch
from torch import nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
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


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy: an MLP gives the mean, a learned state-independent log std gives the spread.

    Samples are unbounded; the environment clips them to [-1, 1]. Log-probabilities are of the unclipped sample.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), init_std: float = 0.6):
        super().__init__()
        self.mean = mlp(obs_dim, tuple(hidden), act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def forward(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self(obs).log_prob(actions).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        super().__init__()
        self.v = mlp(obs_dim, tuple(hidden), 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)
