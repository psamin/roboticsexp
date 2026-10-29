"""Reward shaping cannot be farmed, and the observation vector round-trips. CPU only."""
import torch

from goalkeeper.physics import observation, outcomes, rewards


def test_progress_shaping_telescopes():
    """With gamma = 1 the shaping terms over an episode sum to phi(end) - phi(start), whatever the keeper does."""
    cfg = rewards.RewardConfig(gamma=1.0, action=0.0, jerk=0.0)
    g = torch.Generator().manual_seed(0)
    y_star = torch.tensor([0.8])
    ky = torch.zeros(1)
    phi_prev = rewards.potential(ky, y_star, torch.tensor([True]))
    phi0 = phi_prev.clone()
    total = torch.zeros(1)
    for t in range(50):
        ky = ky + 0.1 * torch.randn(1, generator=g)          # arbitrary wandering
        last = t == 49
        phi_now = torch.zeros(1) if last else rewards.potential(ky, y_star, torch.tensor([True]))
        terms = rewards.compute(cfg, torch.tensor([outcomes.NONE]), phi_prev, phi_now, torch.zeros(1, 1), torch.zeros(1, 1))
        total += terms["progress"]
        phi_prev = phi_now
    assert torch.allclose(total, -phi0, atol=1e-5)      # depends only on the start state


def test_outcome_rewards_and_penalties():
    cfg = rewards.RewardConfig()
    z = torch.zeros(3)
    a = torch.tensor([[0.5], [0.0], [-1.0]])
    ap = torch.tensor([[0.5], [1.0], [1.0]])
    t = rewards.compute(cfg, torch.tensor([outcomes.SAVE, outcomes.GOAL, outcomes.MISS]), z, z, a, ap)
    assert t["save"].tolist() == [10.0, 0.0, 0.0] and t["goal"].tolist() == [0.0, -10.0, 0.0]
    assert torch.allclose(t["action"], -0.01 * a.squeeze(-1) ** 2)
    assert torch.allclose(t["jerk"], -0.02 * (a - ap).squeeze(-1) ** 2)


def test_observation_roundtrip():
    bxy, bv = torch.tensor([[5.0, 0.3]]), torch.tensor([[-4.0, 0.5]])
    obs = observation.build(bxy, bv, torch.tensor([0.2]), torch.tensor([-1.0]), torch.tensor([0.7]))
    assert obs.shape == (1, observation.OBS_DIM)
    o = observation.unpack(obs)
    assert abs(o["ball_x"] - 5.0) < 1e-5 and abs(o["keeper_vy"] + 1.0) < 1e-5 and abs(o["rel_y"] - 0.1) < 1e-5
    assert obs.abs().max() <= 1.0
