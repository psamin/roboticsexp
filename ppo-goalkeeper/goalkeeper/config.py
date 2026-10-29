"""Load configs/*.yaml and apply it to the environment and PPO configs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path=None) -> dict:
    with open(path or ROOT / "configs" / "train.yaml") as f:
        return yaml.safe_load(f)


def apply_env(env_cfg, c: dict):
    e = c["env"]
    env_cfg.scene.num_envs = e["num_envs"]
    env_cfg.vmax, env_cfg.amax = e["vmax"], e["amax"]
    r = e["reward"]
    env_cfg.reward_save, env_cfg.reward_goal = r["save"], r["goal"]
    env_cfg.reward_progress, env_cfg.reward_action, env_cfg.reward_jerk = r["progress"], r["action"], r["jerk"]
    env_cfg.gamma = c["ppo"]["gamma"]
    env_cfg.curriculum_thresholds = tuple(e["curriculum"]["thresholds"])
    env_cfg.curriculum_window = e["curriculum"]["window"]
    return env_cfg


def ppo_config(c: dict):
    from .rl.ppo import PPOConfig
    p = dict(c["ppo"])
    p["hidden"] = tuple(p["hidden"])
    return PPOConfig(**p)
