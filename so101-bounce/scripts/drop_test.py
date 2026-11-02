"""Does the policy catch a ball dropped onto the paddle? Released at rest from fixed heights (ball bottom above the
paddle) and lateral offsets (random direction), overhead camera only, at the given randomisation levels. The
out-of-view ceiling is raised to --ceiling so starts above the training ceiling (0.30 m) can be measured.

  python scripts/drop_test.py --checkpoint best.pt [--envs 1024] [--device cuda]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bounce.env import FAIL_REASONS, BounceEnv, EnvCfg
from bounce.ppo import PPO

HEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40)
OFFSETS = (0.0, 0.03, 0.06)


def run(env, agent, forced):
    env.forced = forced
    obs, _ = env.reset_all()
    done = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    rows = []
    with torch.no_grad():
        for _ in range(env.max_steps + 1):
            obs, _, term, trunc, _ = env.step(agent.act_deterministic(obs["policy"]))
            st = env.pop_completed()
            if st is not None:
                keep = ~done[st["env_id"].long()]
                rows.append({k: v[keep] for k, v in st.items()})
            done |= term | trunc
            if bool(done.all()):
                break
    return {k: torch.cat([x[k] for x in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--difficulties", default="0,0.5")
    ap.add_argument("--ceiling", type=float, default=0.6)
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, a.device, load_optimizer=False)
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()}
    cfg.update(num_envs=a.envs, device=a.device, seed=7)
    env = BounceEnv(EnvCfg(**cfg))
    print(f"{a.checkpoint}: servo {cfg.get('servo_variant', '5v')}, P {cfg.get('p_gain')}; ball released at rest, "
          f"overhead camera only, {a.envs} episodes of {env.cfg.episode_s:.0f} s per row, ceiling {a.ceiling} m")
    print(f"{'diff':>4} {'drop cm':>7} {'off cm':>6} {'survive':>8} {'bounces':>8} {'apex err cm':>11} "
          f"{'1st bounce s':>12} {'main failure':>14}")
    for d in (float(x) for x in a.difficulties.split(",")):
        for h in HEIGHTS:
            for off in OFFSETS:
                s = run(env, agent, {"difficulty": d, "rest_start": False, "side_cam": False, "drop_height": h,
                                     "drop_offset": off, "ceiling": a.ceiling})
                launched = s["launched"] > 0
                t1 = float(s["launch_t"][launched].median()) if bool(launched.any()) else float("nan")
                fails = s["reason"][s["survived"] < 0.5].long()
                main_fail = "-" if fails.numel() == 0 else \
                    f"{FAIL_REASONS[int(fails.mode().values)]} {100 * fails.numel() / s['survived'].numel():.0f}%"
                print(f"{d:4.1f} {100 * h:7.0f} {100 * off:6.0f} {100 * float(s['survived'].mean()):7.0f}% "
                      f"{float(s['bounces'].mean()):8.1f} {100 * float(s['apex_err'][launched].mean()):11.1f} "
                      f"{t1:12.2f} {main_fail:>14}", flush=True)


if __name__ == "__main__":
    main()
