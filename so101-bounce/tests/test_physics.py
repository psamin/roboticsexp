"""Physics checks (CPU, seconds): kinematics match MuJoCo, the ball bounces as specified, the arm holds still."""
import math

import mujoco
import numpy as np
import pytest
import torch

from bounce import ball as B
from bounce.env import BounceEnv, EnvCfg
from bounce.robot import GRIPPER_CLOSED_ON_HANDLE, ArmKinematics, build_model
from bounce.servo import ServoBank


def test_kinematics_gravity_inertia_match_mujoco():
    m = build_model()
    d = mujoco.MjData(m)
    kin = ArmKinematics(m, 16)
    q = np.random.default_rng(0).uniform(kin.lower.numpy(), kin.upper.numpy(), size=(16, 5)).astype(np.float32)
    fk = kin.forward(torch.tensor(q))
    tau_g, M = kin.gravity_and_inertia(fk)
    pid = m.body("paddle").id
    for i in range(16):
        d.qpos[:5], d.qpos[5], d.qvel[:] = q[i], GRIPPER_CLOSED_ON_HANDLE, 0
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        assert np.allclose(fk["paddle_o"][i].numpy(), d.xpos[pid], atol=1e-5)
        assert np.allclose(fk["paddle_R"][i].numpy(), d.xmat[pid].reshape(3, 3), atol=1e-5)
        assert np.allclose(tau_g[i].numpy(), d.qfrc_bias[:5], atol=1e-5)
        assert np.allclose(M[i].numpy(), np.diag(full)[:5] - m.dof_armature[:5], atol=1e-6)


def _paddle(n):
    R = torch.eye(3).expand(n, 3, 3).clone()
    return torch.zeros(n, 3), R


def test_restitution_on_still_paddle():
    o, R = _paddle(1)
    prm = {k: torch.tensor([v[0]]) for k, v in B.BALL_RANGES.items()}
    p, v, w = torch.tensor([[0, 0, 0.019]]), torch.tensor([[0, 0, -1.5]]), torch.zeros(1, 3)
    p, v, w, touch, impact, speed, _, _ = B.paddle_impact(p, v, w, o, R, o, R, prm, 0.0785, 0.075, 2.0, 0.002)
    assert bool(impact[0]) and v[0, 2].item() == pytest.approx(0.85 * 1.5, rel=1e-4)


def test_moving_paddle_adds_speed():
    """Paddle rising at 0.2 m/s: v_out = e (v_in + v_p) + v_p."""
    o, R = _paddle(1)
    prm = {k: torch.tensor([v[0]]) for k, v in B.BALL_RANGES.items()}
    dt = 0.002
    o_prev = o - torch.tensor([[0, 0, 0.2 * dt]])
    p, v, w = torch.tensor([[0, 0, 0.019]]), torch.tensor([[0, 0, -1.3]]), torch.zeros(1, 3)
    p, v, *_ = B.paddle_impact(p, v, w, o, R, o_prev, R, prm, 0.0785, 0.075, 2.0, dt)
    e = 0.85 - 0.02 * (1.5 - 1.5)
    assert v[0, 2].item() == pytest.approx(e * 1.5 + 0.2, rel=1e-3)


def test_tilted_paddle_rolls_ball_at_shell_rate():
    """On a slope, a rolling thin shell accelerates at g sin(theta) / (1 + 2/3)."""
    th = 0.05
    R = torch.tensor([[[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]])
    o = torch.zeros(1, 3)
    prm = {k: torch.tensor([v[0]]) for k, v in B.BALL_RANGES.items()}
    p, v, w = (B.RADIUS * R[..., 2]).clone(), torch.zeros(1, 3), torch.zeros(1, 3)
    g = torch.tensor([0, 0, -9.81])
    dt, T = 0.002, 0.5
    for _ in range(int(T / dt)):
        p, v = B.flight(p, v, w, torch.zeros(1), torch.zeros(1), dt, g)
        p, v, w, *_ = B.paddle_impact(p, v, w, o, R, o, R, prm, 0.0785, 0.075, 2.0, dt)
    speed = v.norm().item()
    assert speed == pytest.approx(9.81 * math.sin(th) / (1 + 2 / 3) * T, rel=0.05)


def test_servo_rate_limit_and_hold():
    s = ServoBank(1, 1, 16.0, "cpu")
    s.reset(torch.tensor([0]), torch.zeros(1, 1))
    q, dq = torch.zeros(1, 1), torch.zeros(1, 1)
    for _ in range(50):
        s.torque(torch.ones(1, 1), q, dq, 0.002)
    assert s.target.item() <= 5.6 * 0.1 + 1e-6                     # internal target moves at most max velocity


@pytest.mark.parametrize("variant", ["5v", "12v"])
def test_env_holds_ball_still_at_zero_action(variant):
    env = BounceEnv(EnvCfg(num_envs=8, device="cpu", servo_variant=variant))
    env.forced = {"difficulty": 0.0, "rest_start": True}
    env.reset_all()
    p0 = env.p_ball.clone()
    for _ in range(25):
        obs, r, term, trunc, _ = env.step(torch.zeros(8, 5))
    assert not bool(term.any())
    assert (env.p_ball - p0).norm(dim=-1).max().item() < 0.002
    assert torch.isfinite(obs["policy"]).all() and obs["policy"].shape == (8, env.obs_dim)
    assert obs["critic"].shape == (8, env.critic_dim)


def test_drop_rebound_height():
    env = BounceEnv(EnvCfg(num_envs=4, device="cpu"))
    env.forced = {"difficulty": 0.0, "rest_start": False}
    env.reset_all()
    fk = env.kin.forward(env.q)
    env.p_ball[:] = fk["paddle_o"] + torch.tensor([0, 0, 0.20 + B.RADIUS])
    env.v_ball[:], env.w_ball[:] = 0, 0
    heights = []
    for _ in range(40):
        env.step(torch.zeros(4, 5))
        heights.append((env.p_ball[:, 2] - env.o_prev[:, 2] - B.RADIUS).clone())
    apex = torch.stack(heights)[12:].max(0).values.mean().item()
    e = 0.85 - 0.02 * (math.sqrt(2 * 9.81 * 0.2) - 1.5)
    assert apex == pytest.approx(e * e * 0.20, rel=0.08)              # drag takes a few percent


def test_full_randomisation_is_finite_and_varied():
    env = BounceEnv(EnvCfg(num_envs=64, device="cpu", seed=3))
    env.forced = {"difficulty": 1.0, "rest_start": None}
    obs, _ = env.reset_all()
    assert env.blade[:, 0].std() > 0 and env.blade[:, 2].std() > 0 and env.kin.mass[:, env.kin.paddle].std() > 0
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(2 * torch.rand(64, 5) - 1)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all() and torch.isfinite(r).all()
    # every hold pose is level (the user levels the paddle when setting the hold pose)
    env.forced = {"difficulty": 1.0, "rest_start": True}
    env.reset_all()
    up = -env.g_env / env.g_env.norm(dim=-1, keepdim=True)                  # level = perpendicular to gravity
    cos = (env.R_prev[..., 2] * up).sum(-1).clamp(-1, 1)
    assert torch.rad2deg(torch.acos(cos)).max() < 1.5                       # level_error is at most ~0.9 deg


@pytest.mark.parametrize("variant,p_gain", [("5v", 16.0), ("12v", 16.0), ("12v", 32.0)])
def test_backlash_joint_model_is_stable(variant, p_gain):
    """Two-inertia joints with gear play stay bounded under aggressive random commands at full randomisation."""
    env = BounceEnv(EnvCfg(num_envs=64, device="cpu", seed=9, servo_variant=variant, p_gain=p_gain))
    env.forced = {"difficulty": 1.0, "rest_start": None}
    env.reset_all()
    gen = torch.Generator().manual_seed(1)
    for _ in range(150):
        env.step(2 * torch.rand(64, 5, generator=gen) - 1)
        assert torch.isfinite(env.q).all() and torch.isfinite(env.qm).all()
        assert env.qd.abs().max() < 30 and env.qdm.abs().max() < 30           # rad/s
        assert (env.qm - env.q).abs().max() < 0.1                             # play + gear wind-up stays small


def test_adr_moves_only_the_tested_group():
    env = BounceEnv(EnvCfg(num_envs=8, device="cpu", seed=1, difficulty=0.5))
    n = 300
    stats = {"boundary_g": torch.full((n,), 2.0), "survived": torch.ones(n)}
    env.update_curriculum(stats)
    assert env.levels[2] > 0.5 and env.levels[0] == env.levels[1] == env.levels[3] == 0.5
    stats = {"boundary_g": torch.full((n,), 0.0), "survived": torch.zeros(n)}
    env.update_curriculum(stats)
    assert env.levels[0] < 0.5 and env.adr_buf[0] == []
