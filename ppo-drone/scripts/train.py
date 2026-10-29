"""TRAIN: headless, vectorised PPO (implemented in drone_ppo/) on the Crazyflie hover task.

  python scripts/train.py --budget_min 60 --out outputs/run          # stops cleanly at the budget
  python scripts/train.py --resume outputs/run/checkpoints/latest.pt --out outputs/run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--config", default=None)
p.add_argument("--out", default="outputs/run")
p.add_argument("--num_envs", type=int, default=None)
p.add_argument("--budget_min", type=float, default=None)
p.add_argument("--eval_every_min", type=float, default=None)
p.add_argument("--rollout_steps", type=int, default=None)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--resume", default=None)
args, app = config.launch(p)


def main():
    import torch

    from drone_ppo import task as T
    from drone_ppo.ppo import PPO
    from drone_ppo.trainer import Trainer

    c = config.load(args.config)
    tr = c["train"]
    for k in ("budget_min", "eval_every_min", "rollout_steps", "seed"):
        if getattr(args, k) is not None:
            tr[k] = getattr(args, k)
    if args.num_envs:
        c["env"]["num_envs"] = args.num_envs
    torch.manual_seed(tr["seed"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"config": c, "args": vars(args)}, f, indent=1, default=str)
    env = config.make_env(c, seed=tr["seed"])
    agent = PPO(T.OBS_DIM, T.ACT_DIM, config.ppo_cfg(c), env.device)
    Trainer(env, agent, args.out, tr["rollout_steps"], tr["budget_min"], tr["eval_every_min"],
            tr["checkpoint_every_min"], tr["selection_seed"], args.resume, c).train()


config.run_main(main)
