"""Task logic (CPU, no Isaac): thrust mapping, observations, rewards, failures, curriculum, and the PD baseline
flying a minimal rigid-body Crazyflie model."""
import math

import pytest
import torch

from drone_ppo import task as T
from drone_ppo.curriculum import Curriculum

MASS, INERTIA = 0.0282, (1.4e-5, 1.4e-5, 2.17e-5)
W = MASS * T.GRAVITY


def test_thrust_mapping_is_hover_centred_and_bounded():
    a = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    f = T.thrust_from_action(a, W)
    assert f.tolist() == pytest.approx([0.0, 0.5 * W, W, 1.45 * W, 1.9 * W])
    assert torch.allclose(T.action_from_thrust(f, W), a, atol=1e-6)
    assert T.action_from_thrust(torch.tensor([5 * W]), W).item() == 1.0


def test_observation_roundtrip_and_clip():
    rel, g, v, w, a = (torch.tensor([[0.3, -0.2, 0.1]]), torch.tensor([[0.0, 0.0, -1.0]]),
                       torch.tensor([[1.0, 0.0, -0.4]]), torch.tensor([[2.0, 0.0, 0.0]]), torch.zeros(1, 4))
    obs = T.build_obs(rel, g, v, w, a)
    assert obs.shape == (1, T.OBS_DIM)
    r2, g2, v2, w2, _ = T.unpack_obs(obs)
    assert torch.allclose(r2, rel) and torch.allclose(v2, v) and torch.allclose(w2, w)
    big = T.build_obs(rel * 100, g, v, w * 1000, a)
    assert big.abs().max() <= T.OBS_CLIP


def test_rewards_prefer_a_steady_hover_at_the_target():
    wts = T.RewardWeights()
    z3, z4 = torch.zeros(2, 3), torch.zeros(2, 4)
    level = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    dist = torch.tensor([0.0, 1.0])
    t = T.reward_terms(wts, dist, z3, z3, level, z4, z4, torch.zeros(2, dtype=torch.bool),
                       torch.zeros(2, dtype=torch.bool), 0.02)
    tot = sum(t.values())
    assert tot[0] > tot[1] > 0
    moving = T.reward_terms(wts, dist[:1], torch.tensor([[1.0, 0, 0]]), z3[:1], level[:1], z4[:1], z4[:1],
                            torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), 0.02)
    assert sum(moving.values()) < tot[0]
    upside = T.reward_terms(wts, dist[:1], z3[:1], z3[:1], torch.tensor([[0.0, 0.0, 1.0]]), z4[:1], z4[:1],
                            torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), 0.02)
    assert upside["crash"].item() == -wts.crash and upside["upright"].item() < 0


def test_failure_rules():
    pos = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.05], [2.5, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
    g = torch.tensor([[0, 0, -1.0]] * 4 + [[0, 0, 0.2]])
    crashed, out = T.failure(pos, g)
    assert crashed.tolist() == [False, True, False, False, True]
    assert out.tolist() == [False, False, True, True, False]


def test_stage_sampling_ranges():
    g = torch.Generator().manual_seed(0)
    for i, s in enumerate(T.STAGES):
        c = T.sample_conditions(torch.full((4000,), i), g)
        assert (c["target"][:, :2].abs() <= s.target_xy + 1e-6).all()
        assert (c["target"][:, 2] >= s.target_z[0] - 1e-6).all() and (c["target"][:, 2] <= s.target_z[1] + 1e-6).all()
        assert ((c["start"] - torch.tensor(T.HOVER_POINT)).abs() <= s.start_offset + 1e-6).all()
        assert (c["rpy"][:, :2].abs() <= s.tilt + 1e-6).all()
        assert (c["wind_frac"].norm(dim=-1) <= s.wind + 1e-6).all()
    q = T.quat_from_rpy(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    assert torch.allclose(q, torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]]), atol=1e-6)


def test_curriculum_promotes_on_success_rate():
    cur = Curriculum(3, (0.8, 0.8), window=100)
    st = torch.zeros(100, dtype=torch.long)
    assert cur.record(st, torch.zeros(100, dtype=torch.bool)) is None
    assert cur.record(st, torch.ones(100, dtype=torch.bool)) == 1
    s = cur.sample(5000, "cpu")
    assert set(s.unique().tolist()) == {0, 1}


# ------------------------------------------------------------------ minimal rigid-body quadrotor
def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rotmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def fly(policy, cond, seconds=6.0):
    n = cond["start"].shape[0]
    p, v, q, w = cond["start"].clone(), cond["start_vel"].clone(), T.quat_from_rpy(cond["rpy"]), torch.zeros(n, 3)
    I = torch.tensor(INERTIA)
    a = torch.zeros(n, 4)
    for k in range(int(seconds * 50)):
        R = rotmat(q)
        Rt = R.transpose(-1, -2)
        obs = T.build_obs((Rt @ (cond["target"] - p).unsqueeze(-1)).squeeze(-1),
                          (Rt @ torch.tensor([0, 0, -1.0]).expand(n, 3).unsqueeze(-1)).squeeze(-1),
                          (Rt @ v.unsqueeze(-1)).squeeze(-1), w, a)
        a = policy(obs).clamp(-1, 1)
        thrust = T.thrust_from_action(a[:, 0], W)
        tau = T.MOMENT_SCALE * a[:, 1:]
        for _ in range(2):                                   # 100 Hz physics, 50 Hz control
            R = rotmat(q)
            acc = (R @ torch.stack([torch.zeros(n), torch.zeros(n), thrust], -1).unsqueeze(-1)).squeeze(-1) / MASS
            acc = acc + torch.tensor([0, 0, -T.GRAVITY]) + cond["wind_frac"] * T.GRAVITY
            wdot = (tau - torch.linalg.cross(w, I * w)) / I
            v, p = v + 0.01 * acc, p + 0.01 * (v + 0.01 * acc)
            w = w + 0.01 * wdot
            q = q + 0.005 * quat_mul(q, torch.cat([torch.zeros(n, 1), w], -1))
            q = q / q.norm(dim=-1, keepdim=True)
    return (cond["target"] - p).norm(dim=-1)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_pd_controller_hovers_on_rigid_body_model(stage):
    g = torch.Generator().manual_seed(stage)
    cond = T.sample_conditions(torch.full((256,), stage), g)
    pd = T.PDHoverController(MASS, INERTIA, W)
    err = fly(pd, cond)
    assert torch.isfinite(err).all()
    assert (err < 0.1).float().mean() > 0.95, err.max()


def test_random_policy_does_not_hover():
    g = torch.Generator().manual_seed(0)
    cond = T.sample_conditions(torch.zeros(128, dtype=torch.long), g)
    err = fly(T.RandomPolicy(torch.Generator().manual_seed(1)), cond, seconds=3.0)
    assert (err < 0.1).float().mean() < 0.2
