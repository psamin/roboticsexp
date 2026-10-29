"""Generalised Advantage Estimation."""
import torch


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """Advantages and returns for a (T, N) rollout.

    dones[t] = 1 when the episode ended on step t; the next stored observation then belongs to a fresh episode,
    so nothing is bootstrapped across that boundary. last_values is V(s_T) for the state after the final step.
    Timeouts are treated as terminal: they are rare here (the ball normally decides the episode) and the value of
    a dead-ball state is close to zero.
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
