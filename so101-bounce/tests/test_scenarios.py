"""Extra start types and evaluation options (held, beside, high drops, camera box): off by default."""
from pathlib import Path

import torch

from bounce.env import BounceEnv, EnvCfg


def _env(**kw):
    return BounceEnv(EnvCfg(num_envs=64, device="cpu", **kw))


def _run(env, steps):
    for _ in range(steps):
        env.step(torch.zeros(env.N, 5))


def test_held_ball_stays_then_falls():
    env = _env()
    env.forced = {"difficulty": 0.0, "rest_start": False, "side_cam": False, "drop_height": 0.15, "held_s": 0.4}
    env.reset_all()
    z0 = env.p_ball[:, 2].clone()
    _run(env, 10)                                       # 0.2 s: still held
    assert torch.allclose(env.p_ball[:, 2], z0)
    _run(env, 20)                                       # 0.6 s: released 0.2 s ago, falling
    assert bool((env.p_ball[:, 2] < z0 - 0.05).all())


def test_hidden_held_ball_is_not_seen():
    env = _env()
    env.forced = {"difficulty": 0.0, "rest_start": False, "side_cam": False, "drop_height": 0.15, "held_s": 0.5,
                  "held_hidden": True}
    env.reset_all()
    _run(env, 15)
    assert bool((env.cam_t < -1e8).all())               # no detection recorded while hidden


def test_high_drop_not_cut_by_ceiling_before_landing():
    env = _env(ceiling_after_land=True)
    env.forced = {"difficulty": 0.0, "rest_start": False, "side_cam": False, "drop_height": 0.38}
    env.reset_all()
    env.step(torch.zeros(env.N, 5))
    assert bool((env.steps == 1).all())                 # nobody terminated (and was reset) on the first step


def test_camera_box_and_training_starts():
    box = ((-0.25, -0.2, 0.7), (0.45, 0.2, 1.3))
    env = _env(overhead_cam_box=box, held_start_prob=0.3, beside_start_prob=0.3)
    env.reset_all()
    lo, hi = torch.tensor(box[0]), torch.tensor(box[1])
    assert bool(((env.cam_o >= lo) & (env.cam_o <= hi)).all())
    lat = (env.p_ball[:, :2] - env.home_env[:, :2]).norm(dim=-1)
    assert bool((env.hold_left > 0).any()) and bool((lat > 0.075).any())
    _run(env, 50)
    assert torch.isfinite(env.p_ball).all()


def test_startup_variation_options_run():
    """Round-8 start-up options (v14): every start type occurs, nothing goes non-finite, the arm holds while waiting."""
    import yaml
    v = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "variants.yaml"))["v14"]["env"]
    kw = {k: tuple(tuple(x) if isinstance(x, list) else x for x in val) if isinstance(val, list) else val
          for k, val in v.items()}
    env = BounceEnv(EnvCfg(num_envs=256, device="cpu", seed=2, **kw))
    env.level = 1.0
    env.reset_all()
    assert (env.wait_left > 0).any() and ((env.q - env.q_ref).abs() > 0.02).any()
    assert (env.v_ball[:, 2] > 0.5).any() and (env.v_ball[:, 2] < -1.0).any()
    for _ in range(60):
        obs, r, te, tr, _ = env.step(torch.zeros(256, 5))
        assert torch.isfinite(obs["critic"]).all() and torch.isfinite(r).all()


def test_launch_replay_is_the_runner_primitive_then_the_policy():
    from bounce import launch as LAUNCH
    env = BounceEnv(EnvCfg(num_envs=64, device="cpu", rest_start_prob=1.0, launch_replay_p=1.0, launch_jitter=0.0,
                           p_gain=32.0))
    env.launch_var[:] = 1
    env.reset_all()
    env.launch_var[:] = 1
    env.launch_scale[:] = 1.0
    seq = LAUNCH.actions(LAUNCH.DEFAULTS[32], 1)[:, 0]
    pol = torch.full((64, 5), 0.2)
    for t in range(seq.shape[0] + 1):
        env.step(pol)
        expect = seq[t] if t < seq.shape[0] else pol[0]
        alive = env.steps == t + 1
        assert torch.allclose(env.a_hist[alive, 0], expect.expand(int(alive.sum()), 5))


def test_no_detections_above_the_frame_edge():
    env = BounceEnv(EnvCfg(num_envs=32, device="cpu", rest_start_prob=0.0, view_top=(0.05, 0.05), view_top_p=1.0,
                           flight_h=(0.20, 0.20), cam_dropout=(0.0, 0.0, 0.0)))
    env.reset_all()
    env.cam_t[:] = -1e9
    env.step(torch.zeros(32, 5))
    assert bool((env.cam_t < -1e8).all())                    # ball 20 cm up, edge at 5 cm: nothing seen yet


def test_round9_options_off_by_default():
    c = EnvCfg()
    assert c.launch_replay_p == 0 and c.view_top_p == 0 and not c.beside_feasible and not c.toss_curriculum
    assert c.joint_offset_full_p == 0
