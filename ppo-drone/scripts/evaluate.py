"""EVALUATE the trained PPO policy against the PD controller and random actions on the same held-out conditions
(test seed, separate from training and from the selection set used to choose best.pt).

  python scripts/evaluate.py --checkpoint outputs/run/checkpoints/best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="outputs/run/checkpoints/best.pt")
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=2000)
p.add_argument("--out", default="outputs/logs/eval_results.json")
p.add_argument("--policies", default="ppo,pd,random")
args, app = config.launch(p)


def main():
    import torch

    from drone_ppo import task as T
    from drone_ppo.evaluation import evaluate
    from drone_ppo.ppo import PPO

    agent, extra = PPO.load(args.checkpoint, "cuda:0", load_optimizer=False)
    c = extra.get("config") or config.load()
    env = config.make_env(c, args.num_envs, stage=0)
    pd = T.PDHoverController(env.mass, env.inertia_diag, env.weight)
    gen = torch.Generator(device=env.device).manual_seed(args.seed)
    pols = {"ppo": agent.act_deterministic, "pd": pd, "random": T.RandomPolicy(gen)}
    res = {"checkpoint": args.checkpoint, "iteration": extra.get("iteration"), "steps": extra.get("steps"),
           "test_seed": args.seed, "episodes_per_suite": env.num_envs, "policies": {}}
    pols = {k: v for k, v in pols.items() if k in args.policies.split(",")}
    for name, pol in pols.items():
        res["policies"][name] = evaluate(env, pol, args.seed, suites=("hover", "reach", "robust", "recovery"))
        for suite, r in res["policies"][name].items():
            print(f"[eval] {name:7s} {suite:8s} success {r['hover_success_rate']:.3f} crash {r['crash_rate']:.3f} "
                  f"final dist {r['final_dist_m']:.3f} m  last-2s dist {r['mean_dist_last_2s_m']:.3f} m  "
                  f"time to hover {r['time_to_hover_s']:.2f} s  |w| {r['ang_speed_last_2s']:.3f} rad/s  "
                  f"|da| {r['action_rate_last_2s']:.4f}  max tilt {r['max_tilt_deg']:.1f} deg", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    lines = ["| policy | suite | hover success | crash | final distance | mean distance, last 2 s | time to hover |",
             "|---|---|---|---|---|---|---|"]
    for name, suites in res["policies"].items():
        for suite, r in suites.items():
            lines.append(f"| {name} | {suite} | {100*r['hover_success_rate']:.1f}% | {100*r['crash_rate']:.1f}% | "
                         f"{100*r['final_dist_m']:.1f} cm | {100*r['mean_dist_last_2s_m']:.1f} cm | "
                         f"{r['time_to_hover_s']:.2f} s |")
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), "\nEVAL_DONE", flush=True)


config.run_main(main)
