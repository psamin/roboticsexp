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
    for k in ("reward_mode", "l2f", "penalty_curriculum", "action_history", "motor_tau", "disturbance_force",
              "disturbance_torque", "asymmetric_critic"):
        if k in e:
            setattr(cfg, k, e[k])
    cfg.curriculum_thresholds = tuple(e["curriculum"]["thresholds"])
    cfg.curriculum_window = e["curriculum"]["window"]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if cfg.render_scene:
        cfg.scene.clone_in_fabric = False     # rendered prims must exist on the USD stage
    from . import task as T
    cfg.observation_space = T.obs_dim(cfg.action_history)
    return cfg


def apply_variant(c: dict, name: str | None, path=None) -> dict:
    """Merge a named variant from configs/variants.yaml into the base config (dicts merge recursively)."""
    if not name:
        return c
    with open(path or ROOT / "configs" / "variants.yaml") as f:
        variants = yaml.safe_load(f)

    def merge(base, over):
        out = dict(base)
        for k, v in over.items():
            out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        return out
    c = merge(c, variants[name])
    c["variant"] = name
    return c


def ppo_cfg(c: dict):
    from .ppo import PPOConfig
    p = dict(c["ppo"])
    p["hidden"] = tuple(p["hidden"])
    return PPOConfig(**p)


def make_env(c: dict, num_envs: int | None = None, **overrides):
    from .environment import DroneHoverEnv
    if num_envs is not None:
        c = {**c, "env": {**c["env"], "num_envs": num_envs}}
    return DroneHoverEnv(env_cfg(c, **overrides))


def launch(parser, headless=True, cameras=False):
    """Parse args and start Isaac Sim (must happen before importing anything from isaaclab.envs)."""
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = headless if not getattr(args, "gui", False) else False
    args.enable_cameras = cameras
    return args, AppLauncher(args).app


def run_main(main):
    """Run main(); print any traceback and exit hard (Kit can hang in shutdown)."""
    import os
    import sys
    import traceback
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
