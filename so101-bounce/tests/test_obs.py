"""The real-arm observation builder (bounce/obs.py) must reproduce the simulator's observations exactly."""
import torch

from bounce.env import BounceEnv, EnvCfg
from bounce.obs import RealObs
from bounce.robot import ArmKinematics


import pytest


@pytest.mark.parametrize("history,cam_k", [(4, 4), (12, 12), (8, 16)])
def test_real_obs_matches_simulator(tmp_path, history, cam_k):
    env = BounceEnv(EnvCfg(num_envs=1, device="cpu", seed=4, history=history, cam_k=cam_k))
    env.forced = {"difficulty": 0.7, "rest_start": False}
    obs, _ = env.reset_all()
    env.kin_nominal.save(tmp_path / "kin.pt")
    kin = ArmKinematics.load(tmp_path / "kin.pt", 1)
    ro = RealObs(kin, env.q_ref[0], env.home_env[0], float(env.target_apex[0]), history, cam_k, bool(env.cam_side[0]))
    ro.joints(env.q_hist[0, 0] + env.q_ref[0])
    assert torch.allclose(ro.vector(float(env.t[0])), obs["policy"], atol=1e-5)
    fed = set()
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        a = 2 * torch.rand(1, 5, generator=gen) - 1
        obs, _, term, trunc, _ = env.step(a)
        if bool(term[0] | trunc[0]):
            break
        ro.acted(a)
        ro.joints(env.q_hist[0, 0] + env.q_ref[0])
        t = float(env.t[0])
        for i in range(env.cam_t.shape[1]):                 # detections that have arrived by now
            tc = float(env.cam_t[0, i])
            if tc > -1e8 and tc <= t - float(env.cam_lat[0]) and tc not in fed:
                fed.add(tc)
                ro.ball(tc, env.cam_pos[0, i])
        assert torch.allclose(ro.vector(t), obs["policy"], atol=1e-4), (ro.vector(t) - obs["policy"]).abs().max()
