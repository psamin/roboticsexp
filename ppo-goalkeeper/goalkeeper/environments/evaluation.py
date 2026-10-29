"""Run fixed, seeded shot sets through the environment and summarise the outcomes.

Each environment plays exactly one shot from the set; later auto-reset episodes are ignored. The same shots, the
same physics and the same drive limits are used for every policy, so results are directly comparable.
"""
import torch

from ..physics import outcomes
from ..physics.shots import LEVEL_NAMES, reachable, sample_shots


def make_shots(level: int, n: int, seed: int, device, vmax: float, amax: float, ctrl_dt: float):
    g = torch.Generator(device=device).manual_seed(seed * 1000 + level)
    s = sample_shots(torch.full((n,), level, dtype=torch.long, device=device), g)
    s["reachable"] = reachable(s, vmax, amax, ctrl_dt, ctrl_dt)
    return s


@torch.no_grad()
def run_shots(env, policy, shots: dict) -> dict:
    """Play one shot per environment; return per-shot records (index = environment id)."""
    n = env.num_envs
    assert shots["level"].numel() == n, "one shot per environment"
    env.forced_shots = shots
    obs, _ = env.reset()
    env.pop_completed()                          # drop anything recorded by the reset itself
    got = torch.zeros(n, dtype=torch.bool, device=env.device)
    out: dict = {}
    for _ in range(int(env.max_episode_length) + 10):
        obs, *_ = env.step(policy(obs["policy"]))
        rec = env.pop_completed()
        if rec is None:
            continue
        ids = rec["env_id"]
        first = ~got[ids]
        if first.any():
            for k, v in rec.items():
                if k not in out:
                    out[k] = torch.zeros(n, dtype=v.dtype, device=v.device)
                out[k][ids[first]] = v[first]
            got[ids[first]] = True
        if got.all():
            break
    env.forced_shots = None
    out["finished"] = got
    return out


def summarize(rec: dict) -> dict:
    on = rec["on_target"] & rec["finished"]
    oc = rec["outcome"]
    n_on = int(on.sum())
    reach = on & rec["reachable"]

    def rate(mask, code):
        m = int(mask.sum())
        return float(((oc == code) & mask).sum()) / m if m else float("nan")

    return {
        "shots": int(rec["finished"].sum()),
        "on_target": n_on,
        "save_rate": rate(on, outcomes.SAVE),
        "goal_rate": rate(on, outcomes.GOAL),
        "on_target_missed": rate(on, outcomes.MISS),
        "timeouts": int((oc == outcomes.TIMEOUT).sum()),
        "reachable_share": float(reach.sum()) / max(n_on, 1),
        "save_rate_reachable": rate(reach, outcomes.SAVE),
        "off_target_shots": int((~rec["on_target"] & rec["finished"]).sum()),
        "off_target_miss_rate": rate(~rec["on_target"] & rec["finished"], outcomes.MISS),
        "contact_vs_geom_agree": float((rec["touched_contact"] == rec["touched_geom"]).float().mean()),
        "mean_return": float(sum(rec[k] for k in rec if k.startswith("r_")).float().mean()),
    }


def evaluate_policy(env, policy, levels=(0, 1, 2), seed: int = 2000) -> dict:
    res = {}
    for lv in levels:
        shots = make_shots(lv, env.num_envs, seed, env.device, env.cfg.vmax, env.cfg.amax, env.step_dt)
        res[LEVEL_NAMES[lv]] = summarize(run_shots(env, policy, shots))
    return res
