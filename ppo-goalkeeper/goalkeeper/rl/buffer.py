"""Fixed-size on-policy rollout storage, laid out (time, env)."""
import torch

from .gae import compute_gae


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, device):
        self.T, self.N = num_steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs = z(num_steps, num_envs, obs_dim)
        self.actions = z(num_steps, num_envs, act_dim)
        self.log_probs = z(num_steps, num_envs)
        self.rewards = z(num_steps, num_envs)
        self.dones = z(num_steps, num_envs)
        self.values = z(num_steps, num_envs)
        self.advantages = z(num_steps, num_envs)
        self.returns = z(num_steps, num_envs)
        self.step = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        t = self.step
        self.obs[t], self.actions[t], self.log_probs[t] = obs, actions, log_probs
        self.rewards[t], self.dones[t], self.values[t] = rewards, dones.float(), values
        self.step += 1

    @property
    def full(self) -> bool:
        return self.step >= self.T

    def compute_returns(self, last_values, gamma: float, lam: float):
        self.advantages, self.returns = compute_gae(self.rewards, self.values, self.dones, last_values, gamma, lam)

    def minibatches(self, num_minibatches: int, generator: torch.Generator | None = None):
        n = self.T * self.N
        flat = {
            "obs": self.obs.reshape(n, -1), "actions": self.actions.reshape(n, -1),
            "log_probs": self.log_probs.reshape(n), "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n), "returns": self.returns.reshape(n),
        }
        perm = torch.randperm(n, device=self.obs.device, generator=generator)
        size = n // num_minibatches
        for i in range(num_minibatches):
            idx = perm[i * size:(i + 1) * size]
            yield {k: v[idx] for k, v in flat.items()}

    def reset(self):
        self.step = 0
