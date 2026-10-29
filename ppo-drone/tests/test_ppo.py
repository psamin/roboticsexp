"""PPO math (CPU, no Isaac): GAE, truncation bootstrapping, clipped objective, squashed-Gaussian log-probs and
action bounds, the update loop on a toy task, and checkpoint round-trips."""
import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from drone_ppo.networks import SquashedGaussianActor, squash_log_prob
from drone_ppo.ppo import PPO, PPOConfig, clipped_surrogate
from drone_ppo.rollout_buffer import RolloutBuffer, compute_gae


def gae_loop(r, v, d, last, gamma, lam):
    T, N = r.shape
    adv = np.zeros((T, N))
    for n in range(N):
        g = 0.0
        for t in reversed(range(T)):
            nv = last[n] if t == T - 1 else v[t + 1, n]
            delta = r[t, n] + gamma * nv * (1 - d[t, n]) - v[t, n]
            g = delta + gamma * lam * (1 - d[t, n]) * g
            adv[t, n] = g
    return adv


def test_gae_matches_reference_loop():
    g = torch.Generator().manual_seed(0)
    r, v = torch.randn(24, 7, generator=g), torch.randn(24, 7, generator=g)
    d = (torch.rand(24, 7, generator=g) < 0.15).float()
    last = torch.randn(7, generator=g)
    adv, ret = compute_gae(r, v, d, last, 0.99, 0.95)
    ref = gae_loop(r.numpy(), v.numpy(), d.numpy(), last.numpy(), 0.99, 0.95)
    np.testing.assert_allclose(adv.numpy(), ref, atol=1e-5)
    np.testing.assert_allclose(ret.numpy(), ref + v.numpy(), atol=1e-5)


def test_gae_lambda_limits():
    r = torch.tensor([[1.0], [1.0], [1.0]])
    v = torch.zeros(3, 1)
    d = torch.zeros(3, 1)
    adv1, _ = compute_gae(r, v, d, torch.tensor([10.0]), 0.5, 1.0)        # Monte Carlo with bootstrap
    assert adv1[0].item() == pytest.approx(1 + 0.5 + 0.25 + 0.125 * 10)
    adv0, _ = compute_gae(r, v, d, torch.tensor([10.0]), 0.5, 0.0)        # one-step TD
    assert adv0[:, 0].tolist() == pytest.approx([1.0, 1.0, 1 + 0.5 * 10])


def test_truncation_bootstraps_but_failure_does_not():
    buf = RolloutBuffer(1, 3, 2, 1, "cpu")
    term = torch.tensor([False, True, False])
    trunc = torch.tensor([True, True, False])
    buf.add(torch.zeros(3, 2), torch.zeros(3, 1), torch.zeros(3), torch.ones(3), term, trunc, torch.zeros(3), 0.9,
            truncated_values=torch.tensor([5.0, 5.0, 5.0]))
    assert buf.rewards[0].tolist() == pytest.approx([1 + 0.9 * 5.0, 1.0, 1.0])   # crash wins over time limit
    assert buf.dones[0].tolist() == [1.0, 1.0, 0.0]


def test_clipped_surrogate():
    old = torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    new = torch.log(torch.tensor([1.5, 0.9, 0.5, 1.1])).requires_grad_()
    loss, kl, cf = clipped_surrogate(new, old, adv, 0.2)
    assert loss.item() == pytest.approx(-(1.2 + 0.9 - 0.8 - 1.1) / 4)
    loss.backward()
    assert new.grad[0] == 0 and new.grad[2] == 0 and new.grad[1] != 0 and new.grad[3] != 0
    assert cf.item() == pytest.approx(0.5)
    ratio = torch.tensor([1.5, 0.9, 0.5, 1.1])
    assert kl.item() == pytest.approx(((ratio - 1) - ratio.log()).mean().item(), rel=1e-5)


def test_squashed_log_prob_matches_change_of_variables():
    """log pi(a) for a = tanh(u) equals the density of a obtained numerically from the density of u."""
    mu, sd = torch.tensor([[0.3]]), torch.tensor([[0.8]])
    d = Normal(mu, sd)
    u = torch.linspace(-3, 3, 41).reshape(-1, 1)
    lp = squash_log_prob(Normal(mu.expand(41, 1), sd.expand(41, 1)), u)
    eps = 1e-4
    a = torch.tanh(u)
    du_da = (torch.atanh((a + eps).clamp(-1 + 1e-7, 1 - 1e-7)) - torch.atanh((a - eps).clamp(-1 + 1e-7, 1 - 1e-7))) / (2 * eps)
    numeric = d.log_prob(u).squeeze(-1) + torch.log(du_da.squeeze(-1))
    mask = u.abs().squeeze(-1) < 2.5                                       # away from float saturation
    assert torch.allclose(lp[mask], numeric[mask], atol=2e-3)


def test_actor_actions_are_bounded_and_logp_consistent():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(16, 4, (128, 128), init_std=2.0)
    obs = torch.randn(5000, 16) * 3
    u, a, logp = actor.sample(obs)
    assert a.shape == (5000, 4) and (a.abs() < 1).all()
    assert torch.allclose(logp, squash_log_prob(actor.dist(obs), u))
    assert (actor.deterministic(obs).abs() < 1).all()


def _toy(agent, T=16, N=64):
    buf = RolloutBuffer(T, N, agent.obs_dim, agent.act_dim, "cpu")
    for _ in range(T):
        obs = torch.randn(N, agent.obs_dim)
        u, a, logp, v = agent.act(obs)
        r = -(a[:, 0] - 0.5 * torch.tanh(obs[:, 0])).pow(2)
        buf.add(obs, u, logp, r, torch.ones(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool), v, 0.99)
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    return buf


def test_ppo_learns_toy_task_with_finite_stats():
    torch.manual_seed(1)
    agent = PPO(3, 1, PPOConfig(lr=3e-3, adaptive_lr=False))
    x = torch.randn(2048, 3)
    err = lambda: (agent.act_deterministic(x)[:, 0] - 0.5 * torch.tanh(x[:, 0])).pow(2).mean().item()
    e0 = err()
    for _ in range(40):
        s = agent.update(_toy(agent))
        assert all(math.isfinite(v) for v in s.values())
    assert err() < 0.25 * e0


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    agent = PPO(16, 4, PPOConfig())
    buf = RolloutBuffer(2, 8, 16, 4, "cpu")
    for _ in range(2):
        o = torch.randn(8, 16)
        u, a, lp, v = agent.act(o)
        buf.add(o, u, lp, torch.randn(8), torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool), v, 0.99)
    buf.compute_returns(torch.zeros(8), 0.99, 0.95)
    agent.update(buf)
    p = tmp_path / "ck.pt"
    agent.save(p, iteration=3)
    b, extra = PPO.load(p)
    o = torch.randn(10, 16)
    assert torch.equal(agent.act_deterministic(o), b.act_deterministic(o))
    assert torch.equal(agent.value(o), b.value(o))
    assert extra == {"iteration": 3} and b.lr == pytest.approx(agent.lr)
