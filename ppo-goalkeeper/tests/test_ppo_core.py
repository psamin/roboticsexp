"""PPO math: GAE, the clipped objective, the Gaussian actor, the update loop, checkpoints. CPU only, no Isaac."""
import math

import numpy as np
import pytest
import torch

from goalkeeper.rl.buffer import RolloutBuffer
from goalkeeper.rl.gae import compute_gae
from goalkeeper.rl.networks import GaussianActor
from goalkeeper.rl.ppo import PPO, PPOConfig, clipped_surrogate


def gae_reference(r, v, d, last_v, gamma, lam):
    """Straightforward per-environment loop, written independently of the vectorised version."""
    T, N = r.shape
    adv = np.zeros((T, N))
    for n in range(N):
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_v[n] if t == T - 1 else v[t + 1, n]
            nonterm = 1.0 - d[t, n]
            delta = r[t, n] + gamma * nv * nonterm - v[t, n]
            gae = delta + gamma * lam * nonterm * gae
            adv[t, n] = gae
    return adv


def test_gae_matches_reference_with_episode_boundaries():
    g = torch.Generator().manual_seed(0)
    T, N = 17, 5
    r, v = torch.randn(T, N, generator=g), torch.randn(T, N, generator=g)
    d = (torch.rand(T, N, generator=g) < 0.2).float()
    last_v = torch.randn(N, generator=g)
    adv, ret = compute_gae(r, v, d, last_v, 0.99, 0.95)
    ref = gae_reference(r.numpy(), v.numpy(), d.numpy(), last_v.numpy(), 0.99, 0.95)
    np.testing.assert_allclose(adv.numpy(), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret.numpy(), ref + v.numpy(), rtol=1e-5, atol=1e-5)


def test_gae_special_cases():
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [0.25], [0.125]])
    d = torch.tensor([[0.0], [0.0], [1.0]])
    # lambda = 1, gamma = 1: advantage = reward-to-go - V
    adv, _ = compute_gae(r, v, d, torch.tensor([9.0]), 1.0, 1.0)
    assert torch.allclose(adv.squeeze(), torch.tensor([6.0, 5.0, 3.0]) - v.squeeze())
    # lambda = 0: advantage = one-step TD error; the terminal step does not bootstrap from last_values
    adv0, _ = compute_gae(r, v, d, torch.tensor([9.0]), 0.9, 0.0)
    td = torch.tensor([1 + 0.9 * 0.25 - 0.5, 2 + 0.9 * 0.125 - 0.25, 3 - 0.125])
    assert torch.allclose(adv0.squeeze(), td)


def test_clipped_surrogate_values_and_gradients():
    old = torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    new = torch.log(torch.tensor([1.5, 0.9, 0.5, 1.1])).requires_grad_()
    loss, kl, clip_frac = clipped_surrogate(new, old, adv, clip=0.2)
    # per-sample objectives: min(1.5, 1.2)=1.2 ; min(0.9, 0.9)=0.9 ; min(-0.5, -0.8)=-0.8 ; min(-1.1, -1.1)=-1.1
    assert loss.item() == pytest.approx(-(1.2 + 0.9 - 0.8 - 1.1) / 4)
    loss.backward()
    # clipped samples (0 and 2) carry no gradient; unclipped ones do
    assert new.grad[0] == 0 and new.grad[2] == 0
    assert new.grad[1] != 0 and new.grad[3] != 0
    assert clip_frac.item() == pytest.approx(0.5)
    ratio = torch.tensor([1.5, 0.9, 0.5, 1.1])
    assert kl.item() == pytest.approx(((ratio - 1) - ratio.log()).mean().item(), rel=1e-5)


def test_actor_sampling_and_log_prob():
    torch.manual_seed(0)
    actor = GaussianActor(9, 1, (128, 128), init_std=0.6)
    obs = torch.randn(4096, 9)
    dist = actor(obs)
    a = dist.sample()
    assert a.shape == (4096, 1)
    mu, std = actor.mean(obs), actor.log_std.exp()
    manual = -((a - mu) ** 2) / (2 * std ** 2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(actor.log_prob(obs, a), manual.sum(-1), atol=1e-5)
    assert ((a - mu) / std).std().item() == pytest.approx(1.0, abs=0.05)
    assert dist.entropy().sum(-1).mean().item() == pytest.approx(0.5 * math.log(2 * math.pi * math.e * 0.36), abs=1e-4)


def _toy_buffer(agent, T=32, N=64):
    """Contextual bandit: reward is high when the action matches -obs[0]."""
    buf = RolloutBuffer(T, N, agent.obs_dim, agent.act_dim, "cpu")
    for _ in range(T):
        obs = torch.randn(N, agent.obs_dim)
        a, logp, v = agent.act(obs)
        r = -(a.squeeze(-1) + obs[:, 0]) ** 2
        buf.add(obs, a, logp, r, torch.ones(N), v)
    buf.compute_returns(torch.zeros(N), agent.cfg.gamma, agent.cfg.lam)
    return buf


def test_ppo_update_improves_a_toy_task_and_stays_finite():
    torch.manual_seed(1)
    agent = PPO(3, 1, PPOConfig(lr=3e-3, epochs=4, num_minibatches=4, target_kl=None))
    test_obs = torch.randn(2048, 3)
    err0 = (agent.act_deterministic(test_obs).squeeze(-1) + test_obs[:, 0]).pow(2).mean().item()
    for _ in range(40):
        stats = agent.update(_toy_buffer(agent))
        assert all(math.isfinite(v) for v in stats.values())
    err1 = (agent.act_deterministic(test_obs).squeeze(-1) + test_obs[:, 0]).pow(2).mean().item()
    assert err1 < 0.25 * err0


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    agent = PPO(9, 1, PPOConfig())
    agent.update(_toy_buffer(agent, T=4, N=16))   # gives the optimiser some state
    path = tmp_path / "ck.pt"
    agent.save(path, iteration=7, best_save_rate=0.5)
    loaded, extra = PPO.load(path)
    obs = torch.randn(32, 9)
    assert torch.equal(agent.act_deterministic(obs), loaded.act_deterministic(obs))
    assert torch.equal(agent.value(obs), loaded.value(obs))
    assert extra == {"iteration": 7, "best_save_rate": 0.5}
    s0, s1 = agent.opt.state_dict()["state"], loaded.opt.state_dict()["state"]
    assert s0.keys() == s1.keys() and all(torch.equal(s0[k]["exp_avg"], s1[k]["exp_avg"]) for k in s0)
