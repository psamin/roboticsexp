"""VERIFY before training: environment checks, physics sanity, and baseline controllers on held-out conditions.

1. reset puts the drone and target where the conditions say; observation shape/finite; actions are clamped
2. zero thrust -> crash; a hard roll command -> flips and fails; hover thrust alone keeps it aloft briefly
3. the PD controller hovers until the time limit (no failure) in every stage
4. random and untrained-PPO baselines on the same held-out conditions
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo import config

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=777)
p.add_argument("--out", default="outputs/logs/verify.json")
args, app = config.launch(p)


def main():
    import torch

    from drone_ppo import task as T
    from drone_ppo.evaluation import make_conditions, run_episodes, summarize
    from drone_ppo.ppo import PPO

    c = config.load()
    env = config.make_env(c, args.num_envs, stage=0)
    dev, n = env.device, env.num_envs
    rep = {"num_envs": n, "physics_dt": env.physics_dt, "control_dt": env.step_dt,
           "mass_kg": env.mass, "weight_N": env.weight, "inertia_diag": env.inertia_diag,
           "max_episode_s": env.max_episode_length_s}

    # 1. reset, observation, action bounds
    cond = make_conditions(0, n, args.seed, dev)
    env.forced = cond
    obs, _ = env.reset()
    o = obs["policy"]
    org = env.scene.env_origins
    rep["reset"] = {
        "obs_shape": list(o.shape), "obs_finite": bool(torch.isfinite(o).all()),
        "start_err_m": float((env.robot.data.root_pos_w - org - cond["start"]).norm(dim=-1).max()),
        "target_err_m": float((env.target_w - org - cond["target"]).norm(dim=-1).max()),
        "bodies": env.robot.body_names, "joints": env.robot.joint_names}
    env.step(torch.full((n, 4), 10.0, device=dev))
    hi = float(env.thrust.max())
    env.step(torch.full((n, 4), -10.0, device=dev))
    lo = float(env.thrust.min())
    rep["action_bounds"] = {"thrust_max_N": hi, "thrust_min_N": lo, "expected_max_N": env.weight * T.THRUST_TO_WEIGHT,
                            "actions_clamped": bool(env.actions.abs().max() <= 1.0)}
    env.pop_completed()

    # 2. physics sanity
    const = lambda vec: (lambda ob: torch.tensor(vec, device=dev).expand(ob.shape[0], 4))
    fall = summarize(run_episodes(env, const([-1.0, 0, 0, 0]), make_conditions(0, n, args.seed + 1, dev)))
    flip = summarize(run_episodes(env, const([0.0, 1.0, 0, 0]), make_conditions(0, n, args.seed + 2, dev)))
    rep["zero_thrust"] = fall
    rep["full_roll_torque"] = flip

    # 3-4. controllers on held-out conditions
    pd = T.PDHoverController(env.mass, env.inertia_diag, env.weight)
    torch.manual_seed(args.seed)
    untrained = PPO(T.OBS_DIM, T.ACT_DIM, config.ppo_cfg(c), dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    policies = {"pd": pd, "random": T.RandomPolicy(gen), "untrained_ppo": untrained.act_deterministic}
    rep["baselines"] = {}
    for name, pol in policies.items():
        rep["baselines"][name] = {}
        for stage, suite in enumerate(T.STAGE_NAMES):
            t0 = time.time()
            r = summarize(run_episodes(env, pol, make_conditions(stage, n, args.seed + 10, dev)))
            r["wall_s"] = round(time.time() - t0, 1)
            if name == "pd" and stage == 0:
                rep["env_steps_per_s"] = round(n * r["episode_length_s"] / env.step_dt / r["wall_s"])
            rep["baselines"][name][suite] = r
            print(f"[verify] {name:13s} {suite:7s} success {r['hover_success_rate']:.3f} crash {r['crash_rate']:.3f} "
                  f"out {r['out_of_bounds_rate']:.3f} final dist {r['final_dist_m']:.3f} m "
                  f"(last 2 s {r['mean_dist_last_2s_m']:.3f}) len {r['episode_length_s']:.1f} s  {r['wall_s']} s",
                  flush=True)
    pdh = rep["baselines"]["pd"]
    rep["checks"] = {
        "obs_shape_ok": rep["reset"]["obs_shape"] == [n, T.OBS_DIM] and rep["reset"]["obs_finite"],
        "reset_places_drone_and_target": rep["reset"]["start_err_m"] < 1e-3 and rep["reset"]["target_err_m"] < 1e-5,
        "thrust_bounded": abs(hi - env.weight * T.THRUST_TO_WEIGHT) < 1e-6 and abs(lo) < 1e-9,
        "zero_thrust_crashes": fall["crash_rate"] > 0.99 and fall["episode_length_s"] < 1.5,
        "full_roll_fails": flip["crash_rate"] + flip["out_of_bounds_rate"] > 0.99,
        "pd_hovers_stage1": pdh["hover"]["hover_success_rate"] > 0.95 and pdh["hover"]["crash_rate"] < 0.01,
        "pd_hovers_all_stages": all(pdh[s]["hover_success_rate"] > 0.9 for s in T.STAGE_NAMES),
        "random_does_not_hover": rep["baselines"]["random"]["hover"]["hover_success_rate"] < 0.2,
    }
    rep["all_checks_pass"] = all(rep["checks"].values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("mass_kg", "weight_N", "inertia_diag", "reset", "action_bounds", "checks",
                                         "all_checks_pass", "env_steps_per_s")}, indent=1), flush=True)
    print("VERIFY_DONE", rep["all_checks_pass"], flush=True)


config.run_main(main)
