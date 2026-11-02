"""Which randomised conditions does a checkpoint fail in? One episode per environment at a fixed difficulty, then
survival split by thirds of each parameter's sampled range.

  python scripts/failure_analysis.py --checkpoint best.pt [--envs 1024] [--difficulty 1] [--device cuda]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from bounce.env import FAIL_REASONS, BounceEnv, EnvCfg
from bounce.ppo import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, a.device, load_optimizer=False)
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()}
    for s in a.set:
        k, v = s.split("=", 1)
        cfg[k] = float(v)
    cfg.update(num_envs=a.envs, device=a.device, seed=a.seed)
    env = BounceEnv(EnvCfg(**cfg))
    env.forced = {"difficulty": a.difficulty, "rest_start": None}
    obs, _ = env.reset_all()
    sp = env.servo.p
    params = {
        "supply V": sp["vin"][:, 0], "motor kt (mean)": sp["kt"].mean(1), "armature (mean)": sp["armature"].mean(1),
        "friction (mean)": sp["fric_c"].mean(1), "max accel log10 (min)": sp["log_max_accel"].min(1).values,
        "D term (mean)": sp["kd_time"].mean(1), "action delay ms": env.delay_sub.float() * env.sub_dt * 1000,
        "camera fps": 1 / env.cam_period, "camera latency ms": env.cam_lat * 1000,
        "line-of-sight noise mm": env.sig_ray * 1000, "camera dropout": env.cam_drop,
        "side camera": env.cam_side.float(), "exposure ms": env.exposure * 1000,
        "burst rate": env.burst_in, "outlier rate": env.outlier_p_env, "age bias ms": env.age_bias_env * 1000,
        "backlash deg (mean)": torch.rad2deg(sp["backlash"].mean(1)), "servo strength (mean)": sp["strength"].mean(1),
        "load friction (mean)": sp["load_fric"].mean(1), "supply sag ohm": sp["supply_r"][:, 0],
        "USB glitch rate": env.glitch_p_env, "table tilt deg": torch.rad2deg(torch.acos(-env.g_env[:, 2] / 9.81)),
        "wind m/s": env.wind.norm(dim=-1), "paddle vibration deg": torch.rad2deg(env.pj_std),
        "blade half-length mm": env.blade[:, 0] * 1000,
        "camera bias mm": env.cam_bias.norm(dim=-1) * 1000, "restitution e0": env.ball_prm["e0"],
        "friction mu": env.ball_prm["mu"], "paddle mass g": env.kin.mass[:, env.kin.paddle] * 1000,
        "joint offset deg (max)": torch.rad2deg(env.q_off.abs().max(1).values),
        "rest start": env.rest_start.float(), "target apex cm": env.target_apex * 100,
    }
    params = {k: v.detach().cpu().numpy().copy() for k, v in params.items()}
    done = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    survived = np.zeros(env.N)
    reason = np.full(env.N, len(FAIL_REASONS) - 1)
    bounces = np.zeros(env.N)
    with torch.no_grad():
        for _ in range(env.max_steps + 1):
            obs, _, term, trunc, _ = env.step(agent.act_deterministic(obs["policy"]))
            st = env.pop_completed()
            if st is not None:
                for i, eid in enumerate(st["env_id"].long().tolist()):
                    if not done[eid]:
                        survived[eid] = float(st["survived"][i])
                        reason[eid] = int(st["reason"][i])
                        bounces[eid] = float(st["bounces"][i])
            done |= term | trunc
            if bool(done.all()):
                break
    print(f"{a.checkpoint}: difficulty {a.difficulty}, {env.N} episodes: survival {survived.mean():.3f}, "
          f"bounces {bounces.mean():.1f}; failures: " + ", ".join(
              f"{FAIL_REASONS[i]} {np.mean(reason == i):.3f}" for i in range(len(FAIL_REASONS) - 1)))
    print(f"{'parameter':26s} {'low third':>22s} {'middle third':>22s} {'high third':>22s}   spread")
    rows = []
    for k, v in params.items():
        if np.ptp(v) < 1e-9:
            continue
        edges = np.quantile(v, [0, 1 / 3, 2 / 3, 1])
        cells, rates = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v <= hi)
            rates.append(survived[m].mean())
            cells.append(f"{lo:7.3g}-{hi:<7.3g} {100 * survived[m].mean():4.0f}%")
        rows.append((max(rates) - min(rates), k, cells))
    for spread, k, cells in sorted(rows, reverse=True):
        print(f"{k:26s} " + " ".join(f"{c:>22s}" for c in cells) + f"   {100 * spread:4.0f} pts")


if __name__ == "__main__":
    main()
