"""Held-out evaluation: fixed, seeded initial states and targets; one full episode per environment."""
import math

import torch

from . import task as T


def make_conditions(stage: int, n: int, seed: int, device) -> dict:
    g = torch.Generator(device=device).manual_seed(seed * 100 + stage)
    return T.sample_conditions(torch.full((n,), stage, dtype=torch.long, device=device), g)


@torch.no_grad()
def run_episodes(env, policy, conditions: dict) -> dict:
    n = env.num_envs
    env.forced = conditions
    obs, _ = env.reset()
    env.pop_completed()
    got = torch.zeros(n, dtype=torch.bool, device=env.device)
    out = {}
    for _ in range(int(env.max_episode_length) + 5):
        obs, *_ = env.step(policy(obs["policy"]))
        rec = env.pop_completed()
        if rec is None:
            continue
        ids = rec["env_id"]
        first = ~got[ids]
        for k, v in rec.items():
            if k not in out:
                out[k] = torch.zeros(n, dtype=v.dtype, device=v.device)
            out[k][ids[first]] = v[first]
        got[ids[first]] = True
        if got.all():
            break
    env.forced = None
    out["finished"] = got
    return out


def summarize(r: dict) -> dict:
    f = r["finished"]
    nan = float("nan")

    def mean(x, m=None):
        m = f if m is None else (m & f)
        x = x[m]
        x = x[~torch.isnan(x)] if x.is_floating_point() else x
        return float(x.float().mean()) if x.numel() else nan

    ok = ~(r["crashed"] | r["out_of_bounds"])
    return {
        "episodes": int(f.sum()),
        "hover_success_rate": mean(r["success"].float()),
        "crash_rate": mean(r["crashed"].float()),
        "out_of_bounds_rate": mean(r["out_of_bounds"].float()),
        "final_dist_m": mean(r["final_dist"]),
        "final_dist_median_m": float(r["final_dist"][f].median()) if f.any() else nan,
        "mean_dist_last_2s_m": mean(r["tail_dist"], ok),
        "time_to_hover_s": mean(r["time_to_hover_s"]),
        "episode_length_s": mean(r["length_s"]),
        "mean_return": mean(r["return"]),
        # stability over the last 2 s of episodes that did not fail
        "ang_speed_last_2s": mean(r["tail_ang_speed"], ok) if "tail_ang_speed" in r else nan,
        "yaw_rate_last_2s": mean(r["tail_yaw_rate"], ok) if "tail_yaw_rate" in r else nan,
        "ang_acc_last_2s": mean(r["tail_ang_acc"], ok) if "tail_ang_acc" in r else nan,
        "action_rate_last_2s": mean(r["tail_action_rate"], ok) if "tail_action_rate" in r else nan,
        "tilt_deg_last_2s": mean(r["tail_tilt_deg"], ok) if "tail_tilt_deg" in r else nan,
        "max_tilt_deg": mean(r["max_tilt_deg"]) if "max_tilt_deg" in r else nan,
    }


SUITES = {"hover": 0, "reach": 1, "robust": 2, "recovery": 3}


def evaluate(env, policy, seed: int, suites=("hover", "robust")) -> dict:
    return {s: summarize(run_episodes(env, policy, make_conditions(SUITES[s], env.num_envs, seed, env.device)))
            for s in suites}


def score(res: dict) -> float:
    """Model-selection score: mean hover-success rate over the suites, minus a small distance term."""
    vals = [r["hover_success_rate"] - 0.1 * min(r["final_dist_m"], 1.0) for r in res.values()]
    return sum(v for v in vals if not math.isnan(v)) / len(vals)
