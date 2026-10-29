"""Run the real Trainer and evaluation code end to end against a tiny fake environment with the same interface as
DroneHoverEnv (CPU, seconds). Catches wiring bugs before any cluster time is spent."""
import json
import os

import torch

from drone_ppo import task as T
from drone_ppo.curriculum import Curriculum
from drone_ppo.ppo import PPO, PPOConfig
from drone_ppo.trainer import Trainer


class FakeEnv:
    """Point mass steered by the first three action components toward a target; same API as DroneHoverEnv."""

    term_names = ("distance", "crash")

    def __init__(self, n=16, horizon=40):
        self.num_envs, self.device, self.step_dt = n, torch.device("cpu"), 0.02
        self.max_episode_length, self.max_episode_length_s = horizon, horizon * 0.02
        self.curriculum = Curriculum(3, (0.5, 0.5), window=32)
        self.forced, self.final_obs = None, torch.zeros(n, T.OBS_DIM)
        self._done = []
        self.pos, self.tgt = torch.zeros(n, 3), torch.zeros(n, 3)
        self.t = torch.zeros(n, dtype=torch.long)
        self.hold = torch.zeros(n, dtype=torch.long)
        self.ok = torch.zeros(n, dtype=torch.bool)
        self.ret = torch.zeros(n)
        self.stage = torch.zeros(n, dtype=torch.long)

    def _obs(self):
        o = torch.zeros(self.num_envs, T.OBS_DIM)
        o[:, :3] = self.tgt - self.pos
        return o

    def _reset(self, ids):
        n = len(ids)
        if self.forced is not None:
            c = {k: v[ids] for k, v in self.forced.items()}
        else:
            c = T.sample_conditions(self.curriculum.sample(n, "cpu"))
        self.pos[ids], self.tgt[ids], self.stage[ids] = c["start"], c["target"], c["stage"]
        self.t[ids], self.hold[ids], self.ok[ids], self.ret[ids] = 0, 0, False, 0.0

    def reset(self):
        self._reset(torch.arange(self.num_envs))
        return {"policy": self._obs()}, {}

    def step(self, a):
        self.pos += 0.05 * a[:, :3]
        self.t += 1
        d = (self.tgt - self.pos).norm(dim=-1)
        self.hold = torch.where(d < 0.1, self.hold + 1, torch.zeros_like(self.hold))
        self.ok |= self.hold >= 5
        term = self.pos.norm(dim=-1) > 3.0
        trunc = self.t >= self.max_episode_length
        r = -d * self.step_dt - 5.0 * term.float()
        self.ret += r
        ended = (term | trunc).nonzero().squeeze(-1)
        if ended.numel():
            self.final_obs[ended] = self._obs()[ended]
            nan = torch.full((ended.numel(),), float("nan"))
            rec = {"env_id": ended, "stage": self.stage[ended], "success": self.ok[ended], "crashed": term[ended],
                   "out_of_bounds": torch.zeros_like(term[ended]), "timed_out": trunc[ended] & ~term[ended],
                   "length_s": self.t[ended].float() * self.step_dt, "final_dist": d[ended], "mean_dist": d[ended],
                   "tail_dist": d[ended], "time_to_hover_s": nan, "r_distance": self.ret[ended],
                   "r_crash": torch.zeros(ended.numel()), "return": self.ret[ended]}
            self._done.append(rec)
            if self.forced is None:
                self.curriculum.record(rec["stage"], rec["success"])
            self._reset(ended)
        return {"policy": self._obs()}, r, term, trunc, {}

    def pop_completed(self):
        if not self._done:
            return None
        out = {k: torch.cat([r[k] for r in self._done]) for k in self._done[0]}
        self._done = []
        return out


def test_trainer_runs_evaluates_checkpoints_and_resumes(tmp_path):
    torch.manual_seed(0)
    env = FakeEnv()
    agent = PPO(T.OBS_DIM, T.ACT_DIM, PPOConfig(), "cpu")
    out = str(tmp_path / "run")
    s = Trainer(env, agent, out, rollout_steps=8, budget_min=0.04, eval_every_min=0.015, checkpoint_every_min=0.01,
                selection_seed=1).train()
    assert s["iterations"] > 0
    for f in ("metrics.csv", "eval_history.json", "summary.json", "checkpoints/best.pt", "checkpoints/latest.pt"):
        assert os.path.exists(os.path.join(out, f)), f
    hist = json.load(open(os.path.join(out, "eval_history.json")))
    assert len(hist) >= 2 and "hover_hover_success_rate" in hist[0]
    with open(os.path.join(out, "metrics.csv")) as f:
        assert len(f.readlines()) > 2
    agent2 = PPO(T.OBS_DIM, T.ACT_DIM, PPOConfig(), "cpu")
    s2 = Trainer(FakeEnv(), agent2, out, 8, s["wall_min"] + 0.01, 1.0, 1.0, 1,
                 resume=os.path.join(out, "checkpoints", "latest.pt")).train()
    assert s2["iterations"] > s["iterations"]
