"""PPO training loop: collect, compute GAE, update, log, evaluate on held-out conditions, checkpoint."""
import csv
import json
import math
import os
import time

import torch

from . import task as T
from .evaluation import evaluate, score
from .ppo import PPO
from .rollout_buffer import RolloutBuffer


class Trainer:
    def __init__(self, env, agent: PPO, out_dir: str, rollout_steps: int, budget_min: float, eval_every_min: float,
                 checkpoint_every_min: float, selection_seed: int, resume: str | None = None, config: dict | None = None):
        self.env, self.agent, self.out = env, agent, out_dir
        self.T, self.N = rollout_steps, env.num_envs
        self.budget_s, self.eval_every, self.ck_every = budget_min * 60, eval_every_min * 60, checkpoint_every_min * 60
        self.selection_seed, self.config = selection_seed, config or {}
        self.ck_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ck_dir, exist_ok=True)
        self.buf = RolloutBuffer(self.T, self.N, T.OBS_DIM, T.ACT_DIM, env.device)
        self.it, self.steps, self.prior_wall, self.best = 0, 0, 0.0, -1e9
        self.eval_history = []
        if resume:
            sd = torch.load(resume, map_location=env.device, weights_only=False)
            agent.load_state_dict(sd)
            ex = sd["extra"]
            self.it, self.steps, self.prior_wall, self.best = ex["iteration"], ex["steps"], ex["wall_s"], ex["best"]
            self.eval_history = ex.get("eval_history", [])
            env.curriculum.load_state_dict(ex["curriculum"])
            print(f"[train] resumed {resume} at iteration {self.it}", flush=True)
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(os.path.join(out_dir, "tb"))
        self.csv_f = open(os.path.join(out_dir, "metrics.csv"), "a", newline="")
        self.csv_w = None

    def wall(self):
        return self.prior_wall + time.time() - self.t0

    def extra(self):
        return {"iteration": self.it, "steps": self.steps, "wall_s": self.wall(), "best": self.best,
                "curriculum": self.env.curriculum.state_dict(), "eval_history": self.eval_history,
                "config": self.config}

    def run_eval(self, tag):
        t0 = time.time()
        res = evaluate(self.env, self.agent.act_deterministic, self.selection_seed)
        s = score(res)
        row = {"iteration": self.it, "steps": self.steps, "wall_min": self.wall() / 60, "score": s,
               **{f"{suite}_{k}": v for suite, r in res.items() for k, v in r.items()}}
        self.eval_history.append(row)
        for suite, r in res.items():
            for k in ("hover_success_rate", "final_dist_m", "crash_rate", "mean_return"):
                self.tb.add_scalar(f"eval_{suite}/{k}", r[k], self.steps)
        print(f"[eval:{tag}] it {self.it} {self.steps/1e6:.1f}M steps {self.wall()/60:.1f} min  " + "  ".join(
            f"{suite}: success {r['hover_success_rate']:.3f} dist {r['final_dist_m']:.3f} m crash {r['crash_rate']:.3f}"
            for suite, r in res.items()) + f"  score {s:.3f} ({time.time()-t0:.0f} s)", flush=True)
        if s > self.best:
            self.best = s
            self.agent.save(os.path.join(self.ck_dir, "best.pt"), **self.extra())
            print(f"[eval:{tag}] new best -> best.pt", flush=True)
        self.agent.save(os.path.join(self.ck_dir, "latest.pt"), **self.extra())
        with open(os.path.join(self.out, "eval_history.json"), "w") as f:
            json.dump(self.eval_history, f, indent=1)

    def log_iteration(self, stats, t_collect, t_total):
        env = self.env
        row = {"iteration": self.it, "steps": self.steps, "wall_min": self.wall() / 60,
               "sps_collect": self.N * self.T / t_collect, "sps": self.N * self.T / t_total, **stats,
               "stage": env.curriculum.stage, "stage_success_rate": env.curriculum.rate()[0]}
        ep = env.pop_completed()
        if ep is not None:
            row.update({"episodes": int(ep["success"].numel()), "ep_return": float(ep["return"].mean()),
                        "ep_length_s": float(ep["length_s"].mean()),
                        "success_rate": float(ep["success"].float().mean()),
                        "crash_rate": float((ep["crashed"] | ep["out_of_bounds"]).float().mean()),
                        "final_dist_m": float(ep["final_dist"].mean())})
            for k in env.term_names:
                row[f"rew_{k}"] = float(ep[f"r_{k}"].mean())
        for k, v in row.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                self.tb.add_scalar(f"train/{k}", v, self.steps)
        if self.csv_w is None:
            keys = list(row) + [k for k in ("episodes", "ep_return", "ep_length_s", "success_rate", "crash_rate",
                                            "final_dist_m") if k not in row] + \
                [f"rew_{k}" for k in env.term_names if f"rew_{k}" not in row]
            self.csv_w = csv.DictWriter(self.csv_f, fieldnames=keys, extrasaction="ignore")
            if self.csv_f.tell() == 0:
                self.csv_w.writeheader()
        self.csv_w.writerow(row)
        self.csv_f.flush()
        if self.it % 10 == 1:
            print(f"[train] it {self.it} {self.steps/1e6:.1f}M {self.wall()/60:.1f} min {row['sps']:.0f} sps "
                  f"stage {env.curriculum.stage} ({env.curriculum.rate()[0]:.2f}) "
                  f"success {row.get('success_rate', float('nan')):.2f} crash {row.get('crash_rate', float('nan')):.2f} "
                  f"ret {row.get('ep_return', float('nan')):.1f} | actor {stats['actor_loss']:.4f} "
                  f"critic {stats['critic_loss']:.4f} H {stats['entropy']:.2f} kl {stats['approx_kl']:.4f} "
                  f"lr {stats['lr']:.1e}", flush=True)

    def train(self):
        env, agent, buf, gamma = self.env, self.agent, self.buf, self.agent.cfg.gamma
        self.t0 = time.time()
        if not self.eval_history:
            self.run_eval("initial")                 # untrained policy, minute 0
        last_eval = last_ck = time.time()
        obs = env.reset()[0]["policy"]
        env.pop_completed()
        print(f"[train] {self.N} envs x {self.T} steps = {self.N*self.T} samples per update; "
              f"budget {self.budget_s/60:.0f} min", flush=True)
        while True:
            t0 = time.time()
            for _ in range(self.T):
                u, a, logp, v = agent.act(obs)
                nxt, r, term, trunc, _ = env.step(a)
                v_final = agent.value(env.final_obs) if bool(trunc.any()) else None
                buf.add(obs, u, logp, r, term, trunc, v, gamma, v_final)
                obs = nxt["policy"]
            buf.compute_returns(agent.value(obs), gamma, agent.cfg.lam)
            t_collect = time.time() - t0
            for name, x in (("obs", buf.obs), ("rewards", buf.rewards), ("actions", buf.raw_actions)):
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite {name} in the rollout")
            stats = agent.update(buf)
            buf.reset()
            if not all(math.isfinite(v) for v in stats.values()):
                raise RuntimeError(f"non-finite PPO statistics {stats}")
            self.it += 1
            self.steps += self.N * self.T
            self.log_iteration(stats, t_collect, time.time() - t0)
            done = self.wall() >= self.budget_s
            if done or time.time() - last_eval >= self.eval_every:
                self.run_eval("final" if done else "periodic")
                obs = env.reset()[0]["policy"]
                env.pop_completed()
                last_eval = time.time()
            elif time.time() - last_ck >= self.ck_every:
                agent.save(os.path.join(self.ck_dir, "latest.pt"), **self.extra())
                last_ck = time.time()
            if done:
                break
        summary = {"iterations": self.it, "steps": self.steps, "wall_min": self.wall() / 60, "best_score": self.best,
                   "num_envs": self.N, "rollout_steps": self.T, "curriculum": self.env.curriculum.state_dict(),
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("[train] done " + json.dumps(summary), flush=True)
        self.tb.close()
        self.csv_f.close()
        return summary
