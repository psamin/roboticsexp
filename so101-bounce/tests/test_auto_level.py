"""EnvCfg.auto_level (the runner's --auto-level): off changes nothing; on, a ball rolling down an invisibly tilted paddle
makes the wrist trim tilt the face back against it, within the clip."""
import math

import torch

from bounce.env import BounceEnv, EnvCfg


def _env(**kw):
    env = BounceEnv(EnvCfg(num_envs=8, device="cpu", seed=3, **kw))
    env.forced = {"difficulty": 0.0, "rest_start": True}
    env.reset_all()
    th = math.radians(2.0)                                        # gravity tilted: the face slopes down toward +x
    env.g_env[:] = 9.81 * torch.tensor([math.sin(th), 0.0, -math.cos(th)])
    pad = env.kin_nominal.forward(env.q_hist[:, 0] + env.q_ref)
    env.p_ball[:] = pad["paddle_o"] + 0.02 * pad["paddle_R"][..., 2]   # resting at the centre
    env.v_ball[:] = 0.0
    return env


def test_off_leaves_hold_command():
    env = _env()
    q0 = env.q_ref.clone()
    for _ in range(60):
        env.step(torch.zeros(env.N, 5))
    assert torch.equal(env.q_ref, q0)


def test_trims_against_the_roll():
    env = _env(auto_level=True)
    for _ in range(100):                                          # 2 s: the ball rolls toward +x
        env.step(torch.zeros(env.N, 5))
    trim = env.q_ref[:, 3:5] - env.al_qref0[:, 3:5]
    assert (trim.abs() <= env.cfg.al_clip + 1e-6).all()
    n = env.kin_nominal.forward(env.q_ref)["paddle_R"][:, :2, 2]
    assert float(n[:, 0].mean()) < -0.002                         # the face now leans back, uphill
