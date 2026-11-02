"""Does the policy cope with the ball placed anywhere on the paddle? Resting starts at fixed fractions of the way from
the blade centre to the rim (random direction), overhead camera only, at the given randomisation levels.

  python scripts/position_test.py --checkpoint best.pt [--envs 512] [--device cuda]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bounce.env import BounceEnv, EnvCfg
from bounce.ppo import PPO

RADII = (0.0, 0.25, 0.5, 0.65, 0.75, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--difficulties", default="0,0.5")
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, a.device, load_optimizer=False)
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()}
    cfg.update(num_envs=a.envs, device=a.device, seed=5)
    env = BounceEnv(EnvCfg(**cfg))
    print(f"{a.checkpoint}: ball resting at a fraction of the way to the rim (random direction), overhead camera only")
    print(f"{'difficulty':>10} {'radius':>7} {'launched':>9} {'survived 8 s':>13} {'bounces':>8}")
    for d in (float(x) for x in a.difficulties.split(",")):
        for r in RADII:
            env.forced = {"difficulty": d, "rest_start": True, "rest_radius": r, "side_cam": False}
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
            s = {k: torch.cat([x[k] for x in rows]) for k in rows[0]}
            print(f"{d:10.1f} {r:7.2f} {100 * float(s['launched'].mean()):8.0f}% {100 * float(s['survived'].mean()):12.0f}% "
                  f"{float(s['bounces'].mean()):8.1f}", flush=True)


if __name__ == "__main__":
    main()
