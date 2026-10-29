"""Load configs/train.yaml and apply it to the environment and PPO configs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path=None) -> dict:
    with open(path or ROOT / "configs" / "train.yaml") as f:
        return yaml.safe_load(f)


def env_cfg(c: dict, **overrides):
    from .environment import DroneHoverEnvCfg
    cfg = DroneHoverEnvCfg()
    e = c["env"]
    cfg.scene.num_envs = e["num_envs"]
    cfg.success_radius, cfg.success_hold_s = e["success_radius"], e["success_hold_s"]
    cfg.rewards = dict(e["rewards"])
    cfg.curriculum_thresholds = tuple(e["curriculum"]["thresholds"])
    cfg.curriculum_window = e["curriculum"]["window"]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def ppo_cfg(c: dict):
    from .ppo import PPOConfig
    p = dict(c["ppo"])
    p["hidden"] = tuple(p["hidden"])
    return PPOConfig(**p)
