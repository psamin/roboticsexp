"""On-policy rollout storage and Generalized Advantage Estimation."""
import torch


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """Advantages and discounted returns for a (T, N) rollout.

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
        R_t     = A_t + V(s_t)

    done_t marks the end of an episode on step t (crash or time limit). Time-limit truncations are not failures:
    the caller adds gamma * V(s_final) to their reward before calling this (see RolloutBuffer.add), so the value of
    the state the episode was cut off in still counts.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv, adv + values


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, device):
        z = lambda *s: torch.zeros(*s, device=device)
        self.T, self.N = num_steps, num_envs
        self.obs = z(num_steps, num_envs, obs_dim)
        self.raw_actions = z(num_steps, num_envs, act_dim)     # pre-tanh samples u
        self.log_probs = z(num_steps, num_envs)
        self.rewards = z(num_steps, num_envs)
        self.dones = z(num_steps, num_envs)
        self.values = z(num_steps, num_envs)
        self.advantages = z(num_steps, num_envs)
        self.returns = z(num_steps, num_envs)
        self.step = 0

    def add(self, obs, raw_actions, log_probs, rewards, terminated, truncated, values, gamma: float,
            truncated_values=None):
        """Store one step. For time-limit truncations, bootstrap: r += gamma * V(s_final)."""
        t = self.step
        r = rewards.clone()
        if truncated_values is not None:
            r = r + gamma * truncated_values * (truncated & ~terminated).float()
        self.obs[t], self.raw_actions[t], self.log_probs[t] = obs, raw_actions, log_probs
        self.rewards[t], self.dones[t], self.values[t] = r, (terminated | truncated).float(), values
        self.step += 1

    def compute_returns(self, last_values, gamma: float, lam: float):
        self.advantages, self.returns = compute_gae(self.rewards, self.values, self.dones, last_values, gamma, lam)

    def minibatches(self, num_minibatches: int):
        n = self.T * self.N
        flat = {"obs": self.obs.reshape(n, -1), "raw_actions": self.raw_actions.reshape(n, -1),
                "log_probs": self.log_probs.reshape(n), "values": self.values.reshape(n),
                "advantages": self.advantages.reshape(n), "returns": self.returns.reshape(n)}
        perm = torch.randperm(n, device=self.obs.device)
        size = n // num_minibatches
        for i in range(num_minibatches):
            idx = perm[i * size:(i + 1) * size]
            yield {k: v[idx] for k, v in flat.items()}

    def reset(self):
        self.step = 0
