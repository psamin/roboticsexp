"""BUILD + VERIFY: physics checks and baseline goalkeepers, before any PPO training.

1. reset behaviour and observation/action dimensions
2. drive test: full-speed sweeps across the goal (speed, acceleration, heading and line drift, travel limit)
3. outcome checks under real physics: a parked keeper concedes, wide shots are misses, a still centred keeper's
   blocks are saves and agree with a geometric contact check
4. random / still / analytical keepers on held-out shots at every curriculum level
The analytical keeper must save at least 90% of on-target beginner shots.
"""
import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--asset_cache", default=os.environ.get("GK_ASSET_CACHE", "asset_cache"))
parser.add_argument("--out", default="outputs/logs/verify.json")
parser.add_argument("--seed", type=int, default=777)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import faulthandler
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faulthandler.dump_traceback_later(600, repeat=True)


def main():
    from goalkeeper.controllers.baselines import AnalyticalKeeper, RandomKeeper, StillKeeper
    from goalkeeper.environments.assets import build_all
    from goalkeeper.environments.evaluation import make_shots, run_shots, summarize
    from goalkeeper.environments.goalkeeper_env import GoalkeeperEnv, GoalkeeperEnvCfg
    from goalkeeper.physics import field as F
    from goalkeeper.physics import observation, outcomes
    from goalkeeper.physics.shots import LEVEL_NAMES

    paths = build_all(args.asset_cache, os.path.join(args.asset_cache, "usd"), with_render_assets=False)
    cfg = GoalkeeperEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.asset_paths = paths
    cfg.level = 0
    cfg.seed = args.seed
    env = GoalkeeperEnv(cfg)
    dev, n = env.device, env.num_envs
    report = {"num_envs": n, "physics_dt": env.physics_dt, "control_dt": env.step_dt}

    # 1. reset ---------------------------------------------------------------------------------------------
    shots = make_shots(0, n, args.seed, dev, cfg.vmax, cfg.amax, env.step_dt)
    env.forced_shots = shots
    obs, _ = env.reset()
    o = obs["policy"]
    bp = env.ball.data.root_pos_w - env.scene.env_origins
    kp = env.robot.data.root_pos_w - env.scene.env_origins
    report["reset"] = {
        "obs_shape": list(o.shape), "obs_finite": bool(torch.isfinite(o).all()),
        "action_space": str(env.single_action_space), "obs_dim_expected": observation.OBS_DIM,
        "ball_start_err_m": float((bp[:, :2] - shots["start"]).norm(dim=-1).max()),
        "ball_height_err_m": float((bp[:, 2] - F.BALL_RADIUS).abs().max()),
        "keeper_start_err_m": float((kp[:, 1] - shots["keeper_y"]).abs().max()),
        "bodies": env.robot.body_names, "joints": env.robot.joint_names,
        "keeper_mass_kg": float(env.robot.root_physx_view.get_masses()[0].sum()),
    }
    env.pop_completed()

    # 2. drive test (ball left resting on the mark) ----------------------------------------------------------
    env.kick_steps = 10 ** 6
    shots0 = dict(shots)
    shots0["keeper_y"] = torch.zeros(n, device=dev)
    env.forced_shots = shots0
    env.reset()
    ys, vys, yaws, xs, vrefs = [], [], [], [], []
    for k in range(100):
        a = torch.full((n, 1), 1.0 if k < 50 else -1.0, device=dev)
        env.step(a)
        pos, yaw, vel = env._keeper_state()
        ys.append(pos[:, 1].clone()); vys.append(vel[:, 1].clone()); yaws.append(yaw.clone())
        xs.append(pos[:, 0].clone()); vrefs.append(env.v_ref.clone())
    ys, vys, yaws, xs, vrefs = map(torch.stack, (ys, vys, yaws, xs, vrefs))
    dt = env.step_dt
    acc = (vys[1:] - vys[:-1]) / dt
    report["drive"] = {
        "max_speed_mps": float(vys.abs().max()),
        "speed_after_0.5s": float(vys[14].mean()),
        "peak_accel_mps2": float(acc[:12].mean(dim=1).max()),
        "time_to_1.5mps_s": float((vys.mean(dim=1) > 1.5).float().argmax()) * dt,
        "max_abs_y_m": float(ys.abs().max()),
        "y_limit_m": F.KEEPER_Y_LIMIT,
        "max_heading_err_deg": float(((yaws - torch.pi / 2).abs().max()) * 180 / torch.pi),
        "max_line_drift_m": float((xs - F.KEEPER_X).abs().max()),
        "mean_abs_track_err_mps": float((vys - vrefs).abs().mean()),
        "trace_y_env0": [round(float(v), 3) for v in ys[::5, 0]],
        "trace_vy_env0": [round(float(v), 3) for v in vys[::5, 0]],
    }
    env.kick_steps = 0

    # 3. outcome checks under physics ------------------------------------------------------------------------
    still, t0 = StillKeeper(), time.time()
    parked = make_shots(0, n, args.seed + 1, dev, cfg.vmax, cfg.amax, env.step_dt)
    parked["keeper_y"] = torch.full((n,), F.KEEPER_Y_LIMIT, device=dev)
    parked["target_y"] = -0.4 - 0.5 * torch.rand(n, device=dev)
    d = torch.stack([-parked["start"][:, 0], parked["target_y"] - parked["start"][:, 1]], -1)
    parked["vel"] = parked["speed"].unsqueeze(-1) * d / d.norm(dim=-1, keepdim=True)
    rec = run_shots(env, still, parked)
    report["parked_keeper_far_side"] = {
        "goal_share": float((rec["outcome"] == outcomes.GOAL).float().mean()),
        "outcomes": {outcomes.OUTCOME_NAMES[c]: int((rec["outcome"] == c).sum()) for c in range(5)},
        "ball_speed_at_line_mean": float(rec["ball_speed_at_line"][rec["outcome"] == outcomes.GOAL].mean()),
        "launch_speed_mean": float(rec["speed"].mean()),
    }
    wide = make_shots(2, n, args.seed + 2, dev, cfg.vmax, cfg.amax, env.step_dt)
    wide["on_target"][:] = False
    side = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    wide["target_y"] = side * (1.8 + 0.4 * torch.rand(n, device=dev))
    d = torch.stack([-wide["start"][:, 0], wide["target_y"] - wide["start"][:, 1]], -1)
    wide["vel"] = wide["speed"].unsqueeze(-1) * d / d.norm(dim=-1, keepdim=True)
    wide["keeper_y"] = torch.zeros(n, device=dev)
    wide["latency"][:] = 0
    rec = run_shots(env, still, wide)
    report["wide_shots"] = {outcomes.OUTCOME_NAMES[c]: int((rec["outcome"] == c).sum()) for c in range(5)}
    centred = make_shots(0, n, args.seed + 3, dev, cfg.vmax, cfg.amax, env.step_dt)
    centred["keeper_y"] = torch.zeros(n, device=dev)
    rec = run_shots(env, still, centred)
    s = summarize(rec)
    # a still keeper at y=0 should block exactly the shots whose path crosses the panel
    y_c = centred["start"][:, 1] + (centred["target_y"] - centred["start"][:, 1]) * \
        (centred["start"][:, 0] - F.CONTACT_X) / centred["start"][:, 0]
    covered = y_c.abs() < F.PANEL_HALF_WIDTH + F.BALL_RADIUS - 0.03
    clear = y_c.abs() > F.PANEL_HALF_WIDTH + F.BALL_RADIUS + 0.03
    report["still_centred_keeper"] = {
        **s,
        "saves_when_path_covered": float((rec["outcome"][covered] == outcomes.SAVE).float().mean()),
        "goals_when_path_clear": float((rec["outcome"][clear] == outcomes.GOAL).float().mean()),
        "wall_s": time.time() - t0,
    }

    # 4. baselines --------------------------------------------------------------------------------------------
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    policies = {"analytical": AnalyticalKeeper(env.drive), "random": RandomKeeper(gen), "still": still}
    report["baselines"] = {}
    for name, pol in policies.items():
        report["baselines"][name] = {}
        for lv in range(3):
            t0 = time.time()
            sh = make_shots(lv, n, args.seed + 10, dev, cfg.vmax, cfg.amax, env.step_dt)
            r = summarize(run_shots(env, pol, sh))
            r["wall_s"] = round(time.time() - t0, 2)
            report["baselines"][name][LEVEL_NAMES[lv]] = r
            print(f"[verify] {name:10s} {LEVEL_NAMES[lv]:12s} save {r['save_rate']:.3f} goal {r['goal_rate']:.3f} "
                  f"reachable {r['reachable_share']:.2f} (save|reach {r['save_rate_reachable']:.3f}) "
                  f"contact/geom agree {r['contact_vs_geom_agree']:.3f}  {r['wall_s']} s", flush=True)
    a0 = report["baselines"]["analytical"]["beginner"]["save_rate"]
    report["checks"] = {
        "obs_dims_ok": report["reset"]["obs_shape"] == [n, observation.OBS_DIM],
        "parked_keeper_concedes": report["parked_keeper_far_side"]["goal_share"] > 0.97,
        "wide_shots_are_misses": report["wide_shots"]["miss"] == n,
        "still_keeper_blocks_covered_paths": report["still_centred_keeper"]["saves_when_path_covered"] > 0.97,
        "drive_respects_travel_limit": report["drive"]["max_abs_y_m"] < F.KEEPER_Y_LIMIT + 0.05,
        "analytical_beginner_>=_0.90": a0 >= 0.90,
    }
    report["all_checks_pass"] = all(report["checks"].values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("reset", "drive", "parked_keeper_far_side", "wide_shots",
                                             "checks", "all_checks_pass")}, indent=1), flush=True)
    print("VERIFY_DONE", report["all_checks_pass"], flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
