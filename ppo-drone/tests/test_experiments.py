"""Reward-shaping / curriculum experiment pieces (CPU, no Isaac): the L2F reward, its penalty curriculum, the
unstable-flight penalties, action-history observations, and the asymmetric critic through the full trainer loop."""
import pytest
import torch

from drone_ppo import task as T
from drone_ppo.ppo import PPO, PPOConfig
from drone_ppo.rollout_buffer import RolloutBuffer
from drone_ppo.trainer import Trainer
from tests.test_trainer_loop import FakeEnv


def test_l2f_reward_matches_formula_and_clips_at_zero():
    w = T.L2FWeights(position=5, orientation=5, linear_velocity=0.01, angular_velocity=0.0, action=0.01,
                     reward_scale=1.0, crash=0.0)
    err = torch.tensor([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]])
    v = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    zero3 = torch.zeros(2, 3)
    g = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    a = torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    no = torch.zeros(2, dtype=torch.bool)
    t = T.l2f_reward_terms(w, err, v, zero3, g, a, no, no, 0.02)
    total = sum(t.values())
    expected0 = 2.0 - 0.5 * (5 * 0.04 + 0 + 0.01 * 0.25 + 0 + 0.01 * 0.25)
    assert total[0].item() == pytest.approx(expected0)
    assert total[1].item() == pytest.approx(0.0)                 # 2 - 0.5 * 5 * 4 < 0 -> clipped
    assert t["l2f_clip"][1].item() == pytest.approx(8.0)         # the clip is logged as its own term


def test_l2f_orientation_is_tilt_only():
    w = T.L2FWeights(reward_scale=1.0)
    z = torch.zeros(1, 3)
    upside = T.l2f_reward_terms(w, z, z, z, torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1, 4),
                                torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), 0.02)
    assert upside["l2f_orientation"].item() == pytest.approx(-0.5 * 5 * 1.0)   # sin^2(90 deg) = 1


def test_penalty_curriculum_schedule_caps_and_gate():
    w = T.L2FWeights()
    pc = T.PenaltyCurriculum(w, {"position": 1.2, "action": 1.4}, {"position": 40.0, "action": 1.0}, every=10, gate=0.7)
    assert not pc.step(5, 1.0)                    # not on the schedule
    assert not pc.step(10, 0.5)                   # gated: policy not ready
    assert pc.step(10, 0.9)
    assert w.position == pytest.approx(6.0) and w.action == pytest.approx(0.014)
    for it in range(20, 2000, 10):
        pc.step(it, 1.0)
    assert w.position == 40.0 and w.action == 1.0 and pc.done()
    sd = pc.state_dict()
    w2 = T.L2FWeights()
    T.PenaltyCurriculum(w2, pc.factors, pc.caps, 10).load_state_dict(sd)
    assert w2.position == 40.0


def test_unstable_flight_penalties():
    w = T.RewardWeights(ang_acc=0.5, lin_acc=0.5)
    z3, z4 = torch.zeros(1, 3), torch.zeros(1, 4)
    g = torch.tensor([[0.0, 0.0, -1.0]])
    no = torch.zeros(1, dtype=torch.bool)
    calm = T.reward_terms(w, torch.zeros(1), z3, z3, g, z4, z4, no, no, 0.02, z3, z3)
    wobble = T.reward_terms(w, torch.zeros(1), z3, z3, g, z4, z4, no, no, 0.02,
                            torch.tensor([[200.0, 0.0, 0.0]]), torch.tensor([[10.0, 0.0, 0.0]]))
    assert calm["ang_acc"].item() == 0 and wobble["ang_acc"].item() == pytest.approx(-0.5 * 4 * 0.02)
    assert wobble["lin_acc"].item() == pytest.approx(-0.5 * 1 * 0.02)


def test_action_history_observation():
    assert T.obs_dim(1) == 16 and T.obs_dim(32) == 140
    z = torch.zeros(3, 3)
    hist = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128) / 1000
    o = T.build_obs(z, z, z, z, hist)
    assert o.shape == (3, 140)
    assert torch.allclose(T.unpack_obs(o)[4], hist[:, :4])       # most recent action first


def test_recovery_stage_is_eval_only_and_larger():
    assert T.STAGE_NAMES[T.N_TRAIN_STAGES:] == ("recovery",)
    c = T.sample_conditions(torch.full((2000,), 3), torch.Generator().manual_seed(0))
    assert c["rpy"][:, :2].abs().max() > 0.6 and c["start_ang_vel"].abs().max() > 0.8


class FakeCriticEnv(FakeEnv):
    """FakeEnv that also returns privileged critic observations, like DroneHoverEnv with asymmetric_critic."""
    EXTRA = 5

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.final_critic_obs = torch.zeros(self.num_envs, T.OBS_DIM + self.EXTRA)

    def _wrap(self, o):
        c = torch.cat([o["policy"], self.tgt, torch.zeros(self.num_envs, 2)], -1)
        o["critic"] = c
        return o

    def reset(self):
        o, e = super().reset()
        return self._wrap(o), e

    def step(self, a):
        o, r, term, trunc, e = super().step(a)
        self.final_critic_obs = torch.cat([self.final_obs, self.tgt, torch.zeros(self.num_envs, 2)], -1)
        return self._wrap(o), r, term, trunc, e


def test_asymmetric_critic_trains_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    env = FakeCriticEnv()
    agent = PPO(T.OBS_DIM, T.ACT_DIM, PPOConfig(), "cpu", critic_dim=T.OBS_DIM + FakeCriticEnv.EXTRA)
    s = Trainer(env, agent, str(tmp_path / "run"), 8, 0.03, 0.015, 0.01, 1).train()
    assert s["iterations"] > 0
    b, _ = PPO.load(tmp_path / "run" / "checkpoints" / "latest.pt")
    assert b.critic_dim == T.OBS_DIM + FakeCriticEnv.EXTRA and b.obs_dim == T.OBS_DIM
    buf = RolloutBuffer(2, 4, T.OBS_DIM, T.ACT_DIM, "cpu", critic_dim=b.critic_dim)
    assert next(buf.minibatches(1))["critic_obs"].shape[-1] == b.critic_dim
