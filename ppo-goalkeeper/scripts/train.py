"""TRAIN: PPO (from scratch, goalkeeper/rl) on the vectorised Isaac Lab goalkeeper environment, headless.

Stops cleanly at the wall-clock budget and keeps latest.pt and best.pt (best = highest mean on-target save rate
over the three shot levels, on a held-out selection set). Resume with --resume <run>/checkpoints/latest.pt.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--out", default="outputs/run")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--budget_min", type=float, default=None)
parser.add_argument("--eval_every_min", type=float, default=None)
parser.add_argument("--rollout_steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import faulthandler
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faulthandler.dump_traceback_later(900, repeat=True)


def main():
    from torch.utils.tensorboard import SummaryWriter

    from goalkeeper import config
    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import evaluate_policy
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg
    from goalkeeper.physics import outcomes
    from goalkeeper.physics.observation import OBS_DIM
    from goalkeeper.physics.shots import LEVEL_NAMES
    from goalkeeper.rl.buffer import RolloutBuffer
    from goalkeeper.rl.ppo import PPO

    c = config.load(args.config)
    if args.num_envs: c["env"]["num_envs"] = args.num_envs
    tr = c["train"]
    for k in ("budget_min", "eval_every_min", "rollout_steps", "seed"):
        if getattr(args, k) is not None:
            tr[k] = getattr(args, k)
    out = args.out
    ck_dir = os.path.join(out, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    torch.manual_seed(tr["seed"])

    paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=False)
    cfg = config.apply_env(GoalkeeperEnvCfg(), c)
    cfg.asset_paths = paths
    cfg.seed = tr["seed"]
    env = GoalkeeperEnv(cfg)
    dev, N, T = env.device, env.num_envs, tr["rollout_steps"]
    agent = PPO(OBS_DIM, 1, config.ppo_config(c), dev)
    buf = RolloutBuffer(T, N, OBS_DIM, 1, dev)

    it, global_step, prior_wall, best = 0, 0, 0.0, -1.0
    eval_history = []
    if args.resume:
        sd = torch.load(args.resume, map_location=dev, weights_only=False)
        agent.load_state_dict(sd)
        ex = sd["extra"]
        it, global_step, prior_wall, best = ex["iteration"], ex["global_step"], ex["wall_s"], ex["best_score"]
        env.curriculum.load_state_dict(ex["curriculum"])
        eval_history = ex.get("eval_history", [])
        print(f"[train] resumed {args.resume}: iteration {it}, {global_step} steps, {prior_wall/60:.1f} min", flush=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"config": c, "args": vars(args), "num_envs": N}, f, indent=1, default=str)

    writer = SummaryWriter(os.path.join(out, "tb"))
    csv_path = os.path.join(out, "metrics.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_w = None

    def extra():
        return {"iteration": it, "global_step": global_step, "wall_s": wall(), "best_score": best,
                "curriculum": env.curriculum.state_dict(), "eval_history": eval_history, "config": c}

    def run_eval(tag):
        nonlocal best
        t0 = time.time()
        res = evaluate_policy(env, agent.act_deterministic, seed=tr["selection_seed"])
        score = sum(r["save_rate"] for r in res.values()) / len(res)
        row = {"iteration": it, "global_step": global_step, "wall_min": wall() / 60, "score": score,
               **{f"{lv}_{k}": v for lv, r in res.items() for k, v in r.items()}}
        eval_history.append(row)
        for lv, r in res.items():
            writer.add_scalar(f"eval/save_rate_{lv}", r["save_rate"], global_step)
            writer.add_scalar(f"eval/goal_rate_{lv}", r["goal_rate"], global_step)
        writer.add_scalar("eval/score", score, global_step)
        print(f"[eval:{tag}] it {it} step {global_step} wall {wall()/60:.1f} min  " +
              "  ".join(f"{lv} {r['save_rate']:.3f}" for lv, r in res.items()) + f"  score {score:.3f}"
              f"  ({time.time()-t0:.1f} s)", flush=True)
        if score > best:
            best = score
            agent.save(os.path.join(ck_dir, "best.pt"), **extra())
            print(f"[eval:{tag}] new best {best:.3f} -> best.pt", flush=True)
        agent.save(os.path.join(ck_dir, "latest.pt"), **extra())
        with open(os.path.join(out, "eval_history.json"), "w") as f:
            json.dump(eval_history, f, indent=1)
        return env.reset()[0]["policy"]

    budget_s = tr["budget_min"] * 60
    t_start = time.time()
    wall = lambda: prior_wall + time.time() - t_start
    last_eval = last_ck = time.time()
    if not args.resume:
        run_eval("initial")              # the untrained policy, so the curve starts at minute 0
        last_eval = time.time()
    obs = env.reset()[0]["policy"]
    env.pop_completed()
    print(f"[train] {N} envs x {T} steps = {N*T} samples/iteration; budget {tr['budget_min']} min", flush=True)
    while True:
        t0 = time.time()
        for _ in range(T):
            a, logp, v = agent.act(obs)
            nxt, r, term, trunc, _ = env.step(a)
            buf.add(obs, a, logp, r, term | trunc, v)
            obs = nxt["policy"]
        buf.compute_returns(agent.value(obs), agent.cfg.gamma, agent.cfg.lam)
        t_collect = time.time() - t0
        if not (torch.isfinite(buf.obs).all() and torch.isfinite(buf.rewards).all() and torch.isfinite(buf.actions).all()):
            raise RuntimeError("non-finite observations, rewards or actions in rollout")
        stats = agent.update(buf)
        buf.reset()
        t_iter = time.time() - t0
        if not all(math.isfinite(v) for v in stats.values()):
            raise RuntimeError(f"non-finite PPO statistics: {stats}")
        it += 1
        global_step += N * T

        row = {"iteration": it, "global_step": global_step, "wall_min": wall() / 60,
               "sps_collect": N * T / t_collect, "sps_total": N * T / t_iter, **stats,
               "curriculum_level": env.curriculum.level, "curriculum_rate": env.curriculum.rate()[0]}
        ep = env.pop_completed()
        if ep is not None:
            row["episodes"] = int(ep["outcome"].numel())
            row["episode_len_s"] = float(ep["length"].float().mean()) * env.step_dt
            for code, name in outcomes.OUTCOME_NAMES.items():
                if code != outcomes.NONE:
                    row[f"frac_{name}"] = float((ep["outcome"] == code).float().mean())
            for k in env.term_names:
                row[f"rew_{k}"] = float(ep[f"r_{k}"].mean())
            for lv, name in enumerate(LEVEL_NAMES):
                m = (ep["level"] == lv) & ep["on_target"]
                row[f"train_save_{name}"] = float((ep["outcome"][m] == outcomes.SAVE).float().mean()) if m.any() else float("nan")
        for k, v in row.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                writer.add_scalar(f"train/{k}", v, global_step)
        if csv_w is None:
            fields = sorted(set(row) | {f"train_save_{n}" for n in LEVEL_NAMES} | {"episodes", "episode_len_s"} |
                            {f"frac_{n}" for n in outcomes.OUTCOME_NAMES.values() if n != "none"} |
                            {f"rew_{k}" for k in env.term_names})
            csv_w = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
            if csv_file.tell() == 0:
                csv_w.writeheader()
        csv_w.writerow(row)
        csv_file.flush()
        if it % 10 == 1:
            print(f"[train] it {it} step {global_step/1e6:.2f}M wall {wall()/60:.1f}m sps {row['sps_total']:.0f} "
                  f"lvl {env.curriculum.level} rate {env.curriculum.rate()[0]:.2f} "
                  f"save {row.get('frac_save', float('nan')):.2f} goal {row.get('frac_goal', float('nan')):.2f} "
                  f"pi {stats['policy_loss']:.3f} v {stats['value_loss']:.3f} H {stats['entropy']:.3f} "
                  f"kl {stats['approx_kl']:.4f} std {stats['action_std']:.3f}", flush=True)

        done = wall() >= budget_s
        if done or time.time() - last_eval >= tr["eval_every_min"] * 60:
            obs = run_eval("final" if done else "periodic")
            env.pop_completed()
            last_eval = time.time()
        elif time.time() - last_ck >= tr["checkpoint_every_min"] * 60:
            agent.save(os.path.join(ck_dir, "latest.pt"), **extra())
            last_ck = time.time()
        if done:
            break

    summary = {"iterations": it, "global_steps": global_step, "wall_min": wall() / 60, "best_score": best,
               "num_envs": N, "rollout_steps": T, "curriculum_history": env.curriculum.history,
               "final_level": env.curriculum.level, "device": torch.cuda.get_device_name(0)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[train] done", json.dumps(summary), flush=True)
    writer.close()
    csv_file.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
