"""Roll out a checkpoint and report what the bouncing actually looks like (CPU, seconds).

  python scripts/inspect_policy.py --checkpoint best.pt [--difficulty 0] [--rest] [--envs 16] [--save traj.npz]

Per bounce: the paddle surface height at impact (relative to home), the apex above home and above the paddle, the
time between impacts, and the paddle's vertical stroke; catches dribbling from a raised paddle and similar tricks.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from bounce import ball as B
from bounce.env import BounceEnv, EnvCfg
from bounce.ppo import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--rest", action="store_true", help="start every episode with the ball at rest")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--save")
    ap.add_argument("--set", action="append", default=[], help="env overrides, e.g. p_gain=32")
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, "cpu", load_optimizer=False)
    env_cfg = dict((extra.get("config") or {}).get("env", {}))
    for s in a.set:
        k, v = s.split("=", 1)
        env_cfg[k] = float(v)
    env_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in env_cfg.items()}
    env_cfg.update(num_envs=a.envs, device="cpu", episode_s=a.seconds)
    env = BounceEnv(EnvCfg(**env_cfg))
    env.forced = {"difficulty": a.difficulty, "rest_start": True if a.rest else None}
    obs, _ = env.reset_all()
    rec = {k: [] for k in ("ball", "paddle", "tilt", "q", "goal", "action")}
    alive = torch.ones(env.N, dtype=torch.bool)
    for _ in range(env.max_steps):
        act = agent.act_deterministic(obs["policy"])
        obs, r, term, trunc, _ = env.step(act)
        alive &= ~(term | trunc)
        rec["ball"].append(env.p_ball.clone())
        rec["paddle"].append(env.o_prev.clone())
        rec["tilt"].append(torch.rad2deg(torch.acos(env.R_prev[:, 2, 2].clamp(-1, 1))))
        rec["q"].append(env.q.clone())
        rec["goal"].append((env.q_ref + env.scale * act.clamp(-1, 1)).clone())
        rec["action"].append(act.clone())
        if not bool(alive.any()):
            break
    T = {k: torch.stack(v).numpy() for k, v in rec.items()}
    home = env.home_env.numpy()                           # (N, 3): each episode's paddle centre at its hold pose
    zb = T["ball"][..., 2] - B.RADIUS - home[:, 2]        # ball bottom above home
    zp = T["paddle"][..., 2] - home[:, 2]                 # paddle surface centre above home
    gap = zb - zp
    print(f"{a.checkpoint}: difficulty {a.difficulty}, {'rest' if a.rest else 'mixed'} starts, {env.N} episodes")
    print(f"paddle height vs home: mean {100 * zp.mean():+.1f} cm, range {100 * zp.min():+.1f}..{100 * zp.max():+.1f} cm; "
          f"tilt mean {T['tilt'].mean():.1f} deg, max {T['tilt'].max():.1f}")
    apex_home, apex_pad, periods, strokes = [], [], [], []
    for e in range(env.N):
        # bounce peaks (an impact lasts one physics step and falls between the 50 Hz samples)
        low_k, last_peak = 0, None
        for k in range(1, len(zb) - 1):
            if zb[k, e] < zb[low_k, e]:
                low_k = k
            if zb[k, e] >= zb[k - 1, e] and zb[k, e] > zb[k + 1, e] and zb[k, e] - zb[low_k, e] > 0.03:
                apex_home.append(zb[k, e])
                apex_pad.append(zb[k, e] - zb[low_k, e])
                if last_peak is not None:
                    periods.append((k - last_peak) * env.dt)
                    strokes.append(zp[last_peak:k + 1, e].max() - zp[last_peak:k + 1, e].min())
                last_peak, low_k = k, k
    if apex_home:
        f = lambda x: f"{100 * np.mean(x):.1f} cm (median {100 * np.median(x):.1f}, 10-90% {100 * np.percentile(x, 10):.1f}-{100 * np.percentile(x, 90):.1f})"
        print(f"bounces: {len(apex_home)} total, {len(apex_home) / env.N:.1f} per episode")
        print(f"  apex above home          {f(apex_home)}")
        print(f"  apex above launch point  {f(apex_pad)}")
        print(f"  time between peaks {1000 * np.mean(periods):.0f} ms (median {1000 * np.median(periods):.0f})")
        print(f"  paddle stroke per bounce {f(strokes)}")
    else:
        print("no bounces detected")
    print(f"target apex (above home): {100 * env.target_apex.mean():.1f} cm; episodes still running at the end: "
          f"{int(alive.sum())}/{env.N}")
    if a.save:
        np.savez_compressed(a.save, **T, home=home, dt=env.dt)


if __name__ == "__main__":
    main()
