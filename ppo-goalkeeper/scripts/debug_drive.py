"""Diagnostic: drive 4 keepers at full command and print what the wheels, base and panel do."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import omni.usd
    from pxr import UsdPhysics

    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import make_shots
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg

    cfg = GoalkeeperEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.asset_paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=False)
    cfg.level = 0
    env = GoalkeeperEnv(cfg)
    stage = omni.usd.get_context().get_stage()
    for i in range(min(2, env.num_envs)):
        j = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f"/World/envs/env_{i}/Keeper/goal_panel_joint"))
        print(f"env_{i} panel joint body0={j.GetBody0Rel().GetTargets()} body1={j.GetBody1Rel().GetTargets()}")
    env.kick_steps = 10 ** 6
    sh = make_shots(0, env.num_envs, 5, env.device, cfg.vmax, cfg.amax, env.step_dt)
    sh["keeper_y"][:] = 0.0
    env.forced_shots = sh
    env.reset()
    names = env.robot.body_names
    ip = names.index("goal_panel")
    for k in range(60):
        env.step(torch.full((env.num_envs, 1), 1.0 if k < 40 else -1.0, device=env.device))
        if k % 4 == 0 or k < 4:
            pos, yaw, vel = env._keeper_state()
            q = env.robot.data.root_quat_w
            roll = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            pitch = torch.asin((2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])).clamp(-1, 1))
            bp = env.robot.data.body_pos_w[:, ip] - env.scene.env_origins
            jv = env.robot.data.joint_vel[:, env.wheel_ids]
            for e in range(min(env.num_envs, 3)):
                print(f"k={k:2d} env{e} y={pos[e,1]:+.3f} x={pos[e,0]:.3f} z={pos[e,2]:.3f} yaw={math.degrees(yaw[e]):6.1f} "
                      f"roll={math.degrees(roll[e]):5.1f} pitch={math.degrees(pitch[e]):5.1f} vy={vel[e,1]:+.2f} "
                      f"vref={env.v_ref[e]:+.2f} wheel={[round(float(v),1) for v in jv[e]]} "
                      f"tgt={[round(float(v),1) for v in env.wheel_targets[e, env.wheel_ids]]} "
                      f"panel=({bp[e,0]:.3f},{bp[e,1]:+.3f},{bp[e,2]:.3f})", flush=True)
    print("DEBUG_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
