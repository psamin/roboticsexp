"""PPO training loop: collect, GAE, update, log, raise the randomisation level, evaluate, checkpoint."""
import csv
import json
import math
import os
import time

import torch

from .env import FAIL_REASONS
from .evaluation import evaluate, score
from .ppo import PPO
from .rollout_buffer import RolloutBuffer


class Trainer:
    def __init__(self, env, agent: PPO, out_dir: str, rollout_steps: int, budget_min: float, eval_every_min: float,
                 checkpoint_every_min: float, eval_seed: int, resume: str | None = None, config: dict | None = None):
        self.env, self.agent, self.out = env, agent, out_dir
        self.T, self.N = rollout_steps, env.N
        self.budget_s, self.eval_every, self.ck_every = budget_min * 60, eval_every_min * 60, checkpoint_every_min * 60
        self.eval_seed, self.config = eval_seed, config or {}
        self.ck_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ck_dir, exist_ok=True)
        self.buf = RolloutBuffer(self.T, self.N, agent.obs_dim, agent.act_dim, env.device, agent.critic_dim)
        self.it, self.steps, self.prior_wall, self.best = 0, 0, 0.0, -1e9
        self.eval_history = []
        if resume:
            sd = torch.load(resume, map_location=env.device, weights_only=False)
            agent.load_state_dict(sd)
            ex = sd["extra"]
            self.it, self.steps, self.prior_wall, self.best = ex["iteration"], ex["steps"], ex["wall_s"], ex["best"]
            self.eval_history = ex.get("eval_history", [])
            env.level = ex.get("level", env.level)
            if ex.get("levels"):
                env.levels = list(ex["levels"])
            print(f"[train] resumed {resume} at iteration {self.it}, level {env.level:.2f}", flush=True)
        self.csv_f = open(os.path.join(out_dir, "metrics.csv"), "a", newline="")
        self.csv_w = None

    def wall(self):
        return self.prior_wall + time.time() - self.t0

    def extra(self):
        return {"iteration": self.it, "steps": self.steps, "wall_s": self.wall(), "best": self.best,
                "level": self.env.level, "levels": list(getattr(self.env, "levels", [])),
                "eval_history": self.eval_history, "config": self.config}

    def run_eval(self, tag):
        t0 = time.time()
        res = evaluate(self.env, self.agent.act_deterministic, self.eval_seed)
        s = score(res)
        row = {"iteration": self.it, "steps": self.steps, "wall_min": self.wall() / 60, "score": s,
               "level": self.env.level, **{f"{k}_{m}": v for k, r in res.items() for m, v in r.items()}}
        self.eval_history.append(row)
        lv = "/".join(f"{x:.2f}" for x in getattr(self.env, "levels", [self.env.level]))
        print(f"[eval:{tag}] it {self.it} {self.steps / 1e6:.0f}M steps {self.wall() / 60:.1f} min level "
              f"{lv} | " + " | ".join(
                  f"{k}: survive {r['survival']:.2f} bounces {r['bounces_mean']:.1f} launch {r['launch_rate']:.2f} "
                  f"apex err {r['apex_err_cm']:.1f} cm" for k, r in res.items()) + f" | score {s:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
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
               "sps": self.N * self.T / t_total, "sps_collect": self.N * self.T / t_collect, **stats}
        ep = env.pop_completed()
        if ep is not None:
            env.update_curriculum(ep)
            row.update({"episodes": int(ep["survived"].numel()), "ep_return": float(ep["ret"].mean()),
                        "ep_length_s": float(ep["length_s"].mean()), "survival": float(ep["survived"].mean()),
                        "bounces": float(ep["bounces"].mean()), "peak_load": float(ep["peak_load"].mean()),
                        "difficulty": float(ep["difficulty"].mean())})
            rs = ep["rest_start"] > 0
            row["launch_rate"] = float(ep["launched"][rs].mean()) if bool(rs.any()) else float("nan")
            for i, reason in enumerate(FAIL_REASONS[:-1]):
                row[f"fail_{reason}"] = float((ep["reason"] == i).float().mean())
            for k in ep:
                if k.startswith("r_"):
                    row[k] = float(ep[k].mean())
        row["level"] = env.level
        for name, lv in zip(("actuation", "sensing", "physics", "task"), getattr(env, "levels", [])):
            row[f"level_{name}"] = lv
        if self.csv_w is None:
            self.csv_w = csv.DictWriter(self.csv_f, fieldnames=list(row), extrasaction="ignore")
            if self.csv_f.tell() == 0:
                self.csv_w.writeheader()
        self.csv_w.writerow(row)
        self.csv_f.flush()
        if self.it % 10 == 1:
            print(f"[train] it {self.it} {self.steps / 1e6:.1f}M {self.wall() / 60:.1f} min {row['sps']:.0f} sps "
                  f"level {env.level:.2f} | survive {row.get('survival', float('nan')):.2f} bounces "
                  f"{row.get('bounces', float('nan')):.1f} launch {row.get('launch_rate', float('nan')):.2f} "
                  f"ret {row.get('ep_return', float('nan')):.1f} | drop {row.get('fail_dropped', float('nan')):.2f} "
                  f"view {row.get('fail_out_of_view', float('nan')):.2f} rest {row.get('fail_resting', float('nan')):.2f} "
                  f"overload {row.get('fail_overload', float('nan')):.2f} | H {stats['entropy']:.2f} "
                  f"kl {stats['approx_kl']:.4f} lr {stats['lr']:.1e} ev {stats['explained_variance']:.2f}", flush=True)

    def train(self):
        env, agent, buf, gamma = self.env, self.agent, self.buf, self.agent.cfg.gamma
        self.t0 = time.time()
        if not self.eval_history:
            self.run_eval("initial")
        last_eval = last_ck = time.time()
        o = env.reset_all()[0]
        obs, cobs = o["policy"], o["critic"]
        print(f"[train] {self.N} envs x {self.T} steps = {self.N * self.T} samples per update; "
              f"budget {self.budget_s / 60:.0f} min", flush=True)
        while True:
            t0 = time.time()
            for _ in range(self.T):
                u, a, logp, v = agent.act(obs, cobs)
                nxt, r, term, trunc, _ = env.step(a)
                v_final = agent.value(env.final_critic_obs) if bool(trunc.any()) else None
                buf.add(obs, u, logp, r, term, trunc, v, gamma, v_final, cobs)
                obs, cobs = nxt["policy"], nxt["critic"]
            buf.compute_returns(agent.value(cobs), gamma, agent.cfg.lam)
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
                o = env.reset_all()[0]
                obs, cobs = o["policy"], o["critic"]
                last_eval = time.time()
            elif time.time() - last_ck >= self.ck_every:
                agent.save(os.path.join(self.ck_dir, "latest.pt"), **self.extra())
                last_ck = time.time()
            if done:
                break
        summary = {"iterations": self.it, "steps": self.steps, "wall_min": self.wall() / 60, "best_score": self.best,
                   "level": env.level, "num_envs": self.N, "rollout_steps": self.T,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("[train] done " + json.dumps(summary), flush=True)
        self.csv_f.close()
        return summary
