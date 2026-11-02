"""Is a start recoverable at all on this arm? For each start (a tossed or tapped ball), search open-loop arm commands
with the cross-entropy method from that exact state, with the simulator's exact physics and no camera (an oracle
far stronger than any policy). Success: the ball survives the horizon, makes at least 3 bounces over 3 cm and ends
within 4 cm of the paddle centre. If the oracle cannot do it, the case is beyond the arm and not worth training.
The checkpoint's own policy runs from the same start for comparison (its usual noisy camera, many episodes).

  python scripts/oracle_feasibility.py --checkpoint best.pt --servo 12v_r10 [--device cuda] [--pop 4096] [--iters 20]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bounce import ball as B
from bounce.env import BounceEnv, EnvCfg
from bounce.ppo import PPO

INTERFACE = ("ctrl_hz", "substeps", "history", "cam_k", "episode_s", "p_gain", "action_scale", "nominal_lift",
             "nominal_elbow", "paddle_box")
DELAY = {"12v_r10": (18.0, 5.0, 35.0), "12v_r8": (18.0, 5.0, 35.0)}

# start: ball centre height above the paddle centre (m), lateral offset (m, along +x), lateral velocity (m/s; angle
# from +x in degrees), vertical velocity, spin (rad/s; axis 'top' = horizontal, across the velocity; 'side' = vertical)
CASES = [
    ("tap at apex, 0.15 m/s sideways", dict(h=0.09, off=0.0, v=0.15, ang=90, vz=0.0, spin=0.0)),
    ("tap at apex, 0.30 m/s sideways", dict(h=0.09, off=0.0, v=0.30, ang=90, vz=0.0, spin=0.0)),
    ("tap at apex, 0.9 m/s down", dict(h=0.09, off=0.0, v=0.0, ang=0, vz=-0.9, spin=0.0)),
    ("tap at apex, 0.9 m/s up", dict(h=0.09, off=0.0, v=0.0, ang=0, vz=0.9, spin=0.0)),
    ("tap at apex, 0.15 sideways + 0.45 down", dict(h=0.09, off=0.0, v=0.15, ang=90, vz=-0.45, spin=0.0)),
    ("toss 0.3 m/s away, spin 15 top", dict(h=0.20, off=0.03, v=0.3, ang=0, vz=0.0, spin=15.0, axis="top")),
    ("toss 0.5 m/s toward centre, spin 30 top", dict(h=0.20, off=0.03, v=0.5, ang=180, vz=0.0, spin=30.0, axis="top")),
    ("toss 0.5 m/s sideways, spin 30 side", dict(h=0.20, off=0.03, v=0.5, ang=90, vz=0.0, spin=30.0, axis="side")),
    ("toss 0.5 m/s away, no spin", dict(h=0.20, off=0.03, v=0.5, ang=0, vz=0.0, spin=0.0)),
    ("toss 0.5 m/s away, spin 30 top", dict(h=0.20, off=0.03, v=0.5, ang=0, vz=0.0, spin=30.0, axis="top")),
    ("toss 0.5 m/s away, spin 30 back", dict(h=0.20, off=0.03, v=0.5, ang=0, vz=0.0, spin=-30.0, axis="top")),
]


ENVELOPE = [
    ("drop 20 cm centred, spin 30 top", dict(h=0.20, off=0.0, v=0.0, ang=0, vz=0.0, spin=30.0, axis="top")),
    ("drop 20 cm centred, spin 30 side", dict(h=0.20, off=0.0, v=0.0, ang=0, vz=0.0, spin=30.0, axis="side")),
    ("toss 0.1 m/s away, no spin", dict(h=0.20, off=0.03, v=0.1, ang=0, vz=0.0, spin=0.0)),
    ("toss 0.2 m/s away, no spin", dict(h=0.20, off=0.03, v=0.2, ang=0, vz=0.0, spin=0.0)),
    ("toss 0.2 m/s sideways, no spin", dict(h=0.20, off=0.03, v=0.2, ang=90, vz=0.0, spin=0.0)),
    ("toss 0.2 m/s toward centre, no spin", dict(h=0.20, off=0.03, v=0.2, ang=180, vz=0.0, spin=0.0)),
    ("toss 0.3 m/s toward centre, no spin", dict(h=0.20, off=0.03, v=0.3, ang=180, vz=0.0, spin=0.0)),
    ("toss 0.2 m/s sideways, spin 15 top", dict(h=0.20, off=0.03, v=0.2, ang=90, vz=0.0, spin=15.0, axis="top")),
    ("toss 0.1 m/s away, spin 15 top", dict(h=0.20, off=0.03, v=0.1, ang=0, vz=0.0, spin=15.0, axis="top")),
]


def place(env, st):
    """Put every env's ball in the same start state (relative to its paddle centre at the hold pose)."""
    N, dev = env.N, env.device
    a = math.radians(st["ang"])
    vlat = torch.tensor([st["v"] * math.cos(a), st["v"] * math.sin(a)], device=dev)
    p = env.home_env + torch.tensor([st["off"], 0.0, st["h"] + B.RADIUS], device=dev)
    v = torch.cat([vlat, torch.tensor([st["vz"]], device=dev)]).expand(N, 3).clone()
    w = torch.zeros(N, 3, device=dev)
    if st["spin"]:
        if st.get("axis") == "side":
            w[:, 2] = st["spin"]
        else:                                                      # horizontal, across the velocity
            w[:, 0], w[:, 1] = -st["spin"] * math.sin(a), st["spin"] * math.cos(a)
    env.p_ball[:], env.v_ball[:], env.w_ball[:] = p, v, w
    env.prev_vz[:] = v[:, 2]
    env.rest_start[:], env.landed[:] = False, False
    env.launch_z[:] = env.home_env[:, 2]
    env.hold_left[:] = 0.0


def rollout(env, actions):
    """actions (T, N, 5) open loop -> (survived s, bounces, final lateral offset m) per env, first episode only."""
    N, T, dt = env.N, actions.shape[0], 1.0 / env.cfg.ctrl_hz
    alive = torch.ones(N, dtype=torch.bool, device=env.device)
    surv = torch.zeros(N, device=env.device)
    bounces = torch.zeros(N, device=env.device)
    final = torch.full((N,), 1.0, device=env.device)
    for t in range(T):
        before = env.ep["bounces"].clone()
        _, _, term, trunc, _ = env.step(actions[t])
        got = torch.where(term | trunc, torch.zeros_like(before), env.ep["bounces"] - before).clamp(min=0)
        bounces += got * alive
        surv += dt * alive
        alive = alive & ~term
        lat = (env.p_ball[:, :2] - env.home_env[:, :2]).norm(dim=-1)
        final = torch.where(alive, lat, final)
    return surv, bounces, final, alive


def make_env(cfg, n):
    env = BounceEnv(EnvCfg(**{**cfg, "num_envs": n}))
    env.forced = {"difficulty": 0.0, "rest_start": False, "side_cam": False}
    env.reset_all()
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--servo", default="12v_r10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=1.5)
    ap.add_argument("--policy_eps", type=int, default=512)
    ap.add_argument("--cases", default="main", choices=["main", "envelope"])
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, a.device, load_optimizer=False)
    tr = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()
          if k in EnvCfg.__dataclass_fields__}
    cfg = {k: tr[k] for k in INTERFACE if k in tr}
    cfg.update(device=a.device, seed=5, servo_variant=a.servo, act_delay_ms=DELAY[a.servo], side_cam_prob=0.0,
               overhead_cam=(-0.068, 0.012, 0.732), ceiling_after_land=True)
    hz = cfg.get("ctrl_hz", 50)
    T = int(round(a.horizon * hz))
    knots = int(round(a.horizon / 0.1)) + 1
    print(f"{a.checkpoint}: oracle = CEM over open-loop actions ({knots} knots x 5 joints, linear between, |a| <= 1), "
          f"pop {a.pop}, {a.iters} iters, horizon {a.horizon} s, servo {a.servo}, difficulty 0 (deterministic physics)")
    det = make_env(cfg, 8)
    place(det, CASES[0][1])
    s1 = rollout(det, torch.zeros(T, 8, 5, device=a.device))
    print(f"determinism check (8 identical zero-action rollouts): survived {s1[0].tolist()}")
    print(f"\n{'start':42s} {'oracle':>7} {'surv s':>6} {'bounces':>7} {'end off':>7} {'pop ok':>6}  {'policy':>6}")
    for name, st in (CASES if a.cases == "main" else ENVELOPE):
        env = make_env(cfg, a.pop)
        mu = torch.zeros(knots, 5, device=a.device)
        sd = torch.full((knots, 5), 0.6, device=a.device)
        best = (-1e9, None)
        for it in range(a.iters):
            place(env, st)
            env.t[:], env.steps[:] = 0.0, 0
            K = (mu + sd * torch.randn(a.pop, knots, 5, device=a.device, generator=None)).clamp(-1, 1)
            if it == 0:
                K[0] = 0.0                                         # include "hold still"
            x = torch.linspace(0, knots - 1, T, device=a.device)
            i0 = x.floor().long().clamp(max=knots - 2)
            f = (x - i0)[None, :, None]
            acts = (K[:, i0] * (1 - f) + K[:, i0 + 1] * f).permute(1, 0, 2).contiguous()
            surv, bo, fin, alive = rollout(env, acts)
            ok = alive & (bo >= 3) & (fin < 0.04)
            score = surv + 0.3 * bo.clamp(max=6) - 5.0 * fin + 2.0 * ok.float()
            j = int(score.argmax())
            if float(score[j]) > best[0]:
                best = (float(score[j]), (float(surv[j]), float(bo[j]), float(fin[j]), bool(ok[j])))
            elite = score.topk(max(8, a.pop // 16)).indices
            mu = 0.3 * mu + 0.7 * K[elite].mean(0)
            sd = (0.3 * sd + 0.7 * K[elite].std(0)).clamp(min=0.05)
            pop_ok = float(ok.float().mean())
            env.reset_all()
        s, b, fe, okb = best[1]
        # the trained policy from the same start, with its noisy camera (difficulty 0)
        penv = make_env(cfg, a.policy_eps)
        place(penv, st)
        obs = penv._obs()
        alive = torch.ones(penv.N, dtype=torch.bool, device=penv.device)
        with torch.no_grad():
            for t in range(T):
                obs, _, term, trunc, _ = penv.step(agent.act_deterministic(obs["policy"]))
                alive = alive & ~term
        print(f"{name:42s} {'YES' if okb else 'no':>7} {s:6.2f} {b:7.0f} {100 * fe:6.1f}cm {100 * pop_ok:5.0f}%  "
              f"{100 * float(alive.float().mean()):5.0f}%", flush=True)


if __name__ == "__main__":
    main()
