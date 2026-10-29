"""EVALUATE: the trained PPO policy against the random and analytical baselines, on held-out shots.

Every policy faces the same seeded shots (test seed, distinct from the training stream and from the selection set
used to pick best.pt), with the same physics and drive limits. PPO acts deterministically (the Gaussian's mean).
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
parser.add_argument("--num_envs", type=int, default=1024, help="shots per level per policy")
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--out", default="outputs/logs/eval_results.json")
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from goalkeeper import config
    from goalkeeper.controllers.baselines import AnalyticalKeeper, RandomKeeper
    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import evaluate_policy
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg
    from goalkeeper.rl.ppo import PPO

    agent, extra = PPO.load(args.checkpoint, device="cuda:0", load_optimizer=False)
    c = extra.get("config") or config.load()
    c["env"]["num_envs"] = args.num_envs
    cfg = config.apply_env(GoalkeeperEnvCfg(), c)
    cfg.asset_paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=False)
    cfg.level = 0
    env = GoalkeeperEnv(cfg)
    gen = torch.Generator(device=env.device).manual_seed(args.seed)
    policies = {"ppo": agent.act_deterministic, "analytical": AnalyticalKeeper(env.drive), "random": RandomKeeper(gen)}
    results = {"checkpoint": args.checkpoint, "checkpoint_iteration": extra.get("iteration"),
               "checkpoint_steps": extra.get("global_step"), "test_seed": args.seed, "policies": {}}
    for name, pol in policies.items():
        results["policies"][name] = evaluate_policy(env, pol, seed=args.seed)
        print(f"[eval] {name:10s} " + "  ".join(f"{lv} save {r['save_rate']:.3f} goal {r['goal_rate']:.3f}"
                                                for lv, r in results["policies"][name].items()), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    lines = ["| policy | " + " | ".join(results["policies"]["ppo"].keys()) + " |",
             "|---|" + "---|" * len(results["policies"]["ppo"])]
    for name, res in results["policies"].items():
        lines.append(f"| {name} | " + " | ".join(f"{100*r['save_rate']:.1f}%" for r in res.values()) + " |")
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("On-target save rate, held-out test shots (%d per level)\n\n" % args.num_envs + "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
