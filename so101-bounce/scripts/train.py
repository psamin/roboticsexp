"""Train the bouncing policy with PPO.

  python scripts/train.py --out outputs/run [--variant p32] [--set train.budget_min=120 --set env.num_envs=8192]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml

from bounce.env import BounceEnv, EnvCfg
from bounce.ppo import PPO, PPOConfig
from bounce.trainer import Trainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def load_config(variant=None, sets=()):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "train.yaml")))
    if variant:
        cfg = merge(cfg, yaml.safe_load(open(os.path.join(ROOT, "configs", "variants.yaml")))[variant])
    for s in sets:
        key, val = s.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--resume")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.variant, a.set)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"variant": a.variant, **cfg}, f, indent=1)
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    env_kw = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["env"].items()}
    env = BounceEnv(EnvCfg(device=a.device, seed=t["seed"], **env_kw))
    p = dict(cfg["ppo"])
    p["hidden"] = tuple(p["hidden"])
    agent = PPO(env.obs_dim, 5, PPOConfig(**p), a.device, critic_dim=env.critic_dim)
    print(f"[train] device {a.device} ({torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu'}), "
          f"obs {env.obs_dim}, critic {env.critic_dim}, variant {a.variant}", flush=True)
    Trainer(env, agent, a.out, t["rollout_steps"], t["budget_min"], t["eval_every_min"], t["checkpoint_every_min"],
            t["eval_seed"], a.resume, cfg).train()


if __name__ == "__main__":
    main()
