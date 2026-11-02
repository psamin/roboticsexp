"""Tune a high, steered launch (bounce/launch.py) for dead paddles: a 15-20 cm first bounce from a ball resting
anywhere on the blade (centre, half-way, near the rim), possibly rolling (up to 0.3 m/s) on a paddle up to 2 deg off
level, with ball-paddle restitution 0.55-0.75. The search covers dip and strike shape and length, a command gain above
the policy's range (Launch.gain, up to 2: the servo's P loop saturates sooner) and strike steering from the ball's
offset and roll (Launch.steer_p / steer_v). Reports the paddle's peak upward speed.

  uv run python scripts/tune_launch_high.py --servo 12v_r10 --p_gain 32 [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bounce import ball as B                                # noqa: E402
from bounce.env import BounceEnv, EnvCfg                    # noqa: E402
from bounce.launch import DEFAULTS, NAMED, Launch, actions, steer, steer_jacobian   # noqa: E402
from bounce.robot import ArmKinematics                      # noqa: E402

DIP_MIN = [1]
WINDOW = 70                                                 # control steps watched (1.4 s at 50 Hz)
GROUPS = {"centre": 0.0, "half-way": 0.45, "rim": 0.85}     # rest radius as a fraction of the blade


def make_env(n, servo, p_gain, rest_radius, seed, roll=0.3, tilt=0.035, e_range=(0.55, 0.75), difficulty=0.0):
    env = BounceEnv(EnvCfg(num_envs=n, device="cpu", servo_variant=servo, p_gain=float(p_gain), seed=seed,
                           level_error=tilt, level_error_full_p=1.0, paddle_box=(0.12, 0.10),
                           launch_max_action=2.0))   # strikes rise ~8 cm
    g = torch.Generator().manual_seed(seed)
    e0 = e_range[0] + (e_range[1] - e_range[0]) * torch.rand(n, generator=g)
    sp = roll * torch.rand(n, generator=g)
    ang = 2 * np.pi * torch.rand(n, generator=g)

    def post(env_, ids):
        env_.ball_prm["e0"][ids] = e0[ids]
        env_.v_ball[ids, 0] = sp[ids] * torch.cos(ang[ids])
        env_.v_ball[ids, 1] = sp[ids] * torch.sin(ang[ids])
    env.forced = {"difficulty": difficulty, "rest_start": True, "rest_radius": rest_radius, "post_reset": post}
    env.reset_all()
    return env


def run(env, params, J):
    """params: list of Launch, one per env. Returns first apex above the paddle, drift at it, alive, peak paddle
    upward speed during the launch."""
    N = env.N
    off = env.p_ball[:, :2] - env.home_env[:, :2] + 0.01 * torch.randn(N, 2)
    vel = env.v_ball[:, :2] + 0.05 * torch.randn(N, 2)
    T = max(p.steps for p in params)
    seq = torch.zeros(T, N, 5)
    for i, p in enumerate(params):
        st = steer(p, J, env.scale, off[i:i + 1], vel[i:i + 1])
        s = actions(p, 1, steer=st)
        seq[:s.shape[0], i] = s[:, 0]
    alive = torch.ones(N, dtype=torch.bool)
    best = torch.full((N,), -1.0)
    drift = torch.zeros(N)
    padz = torch.zeros(N)
    vmax = torch.zeros(N)
    home = env.home_env.clone()
    z_prev = env.kin.forward(env.q)["paddle_o"][:, 2]
    done = torch.zeros(N, dtype=torch.bool)               # first apex reached: the launch is judged there
    for t in range(max(T, WINDOW)):
        a = seq[t] if t < T else torch.zeros(N, 5)
        _, _, term, _, _ = env.step(a)
        alive &= ~(term & ~done)
        z = env.kin.forward(env.q)["paddle_o"][:, 2]
        if t < T:
            vmax = torch.maximum(vmax, (z - z_prev) / env.dt)
        z_prev = z
        h = env.p_ball[:, 2] - B.RADIUS - home[:, 2]
        up = alive & ~done & (h > best)
        best = torch.where(up, h, best)
        drift = torch.where(up, (env.p_ball[:, :2] - home[:, :2]).norm(dim=-1), drift)
        padz = torch.where(up, (z - home[:, 2]).abs(), padz)
        done |= (best > 0.05) & (env.v_ball[:, 2] < 0)
        if bool(done.all()):
            break
    run.padz = padz
    return best, drift, alive & done, vmax


DRIFT_W = [2.0]
APEX = [0.15, 0.20]                                         # good first apex (m); --apex overrides
PAD_W = [0.0]                                               # score weight on the paddle's height above home at the apex
RET = [False]                                               # search a return stroke (Launch.ret / ret_steps)
RANK_OK = [False]                                           # rank candidates by their share of good launches first


def score(apex, drift, alive, padz=None):
    lo, hi = APEX
    ok = alive & (apex > lo) & (apex < hi) & (drift < 0.03)
    pen = PAD_W[0] * padz if (padz is not None and PAD_W[0]) else 0.0
    if padz is not None and PAD_W[0]:
        ok = ok & (padz < 0.03)
    return ok, torch.where(alive, -(apex - 0.5 * (lo + hi)).abs() - DRIFT_W[0] * drift - pen, torch.full_like(apex, -1.0))


def rand_params(rng, k, center=None, sigma=0.0):
    out = []
    for _ in range(k):
        if center is None:
            dip, strike = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            ds, ss, gain = int(rng.integers(DIP_MIN[0], 7)), int(rng.integers(2, 11)), float(rng.uniform(1.0, 2.0))
            sp, sv = float(rng.uniform(0, 3)), float(rng.uniform(0, 1))
        else:
            dip = np.array(center.dip) + rng.normal(0, sigma, 3)
            strike = np.array(center.strike) + rng.normal(0, sigma, 3)
            ds = int(np.clip(center.dip_steps + rng.integers(-1, 2), DIP_MIN[0], 7))
            ss = int(np.clip(center.strike_steps + rng.integers(-1, 2), 2, 11))
            gain = float(np.clip(center.gain + rng.normal(0, 0.1), 1.0, 2.0))
            sp = float(np.clip(center.steer_p + rng.normal(0, 0.3), 0, 4))
            sv = float(np.clip(center.steer_v + rng.normal(0, 0.1), 0, 1.5))
        if RET[0]:
            if center is None or not center.ret_steps:
                ret, rs = rng.uniform(-1, 1, 3), int(rng.integers(0, 6))
            else:
                ret = np.array(center.ret) + rng.normal(0, max(sigma, 0.15), 3)
                rs = int(np.clip(center.ret_steps + rng.integers(-1, 2), 0, 6))
        else:
            ret, rs = np.zeros(3), 0
        g = gain
        out.append(Launch(dip=tuple(np.round(np.clip(dip, -g, g), 3).tolist()), dip_steps=ds,
                          strike=tuple(np.round(np.clip(strike, -g, g), 3).tolist()), strike_steps=ss, settle_steps=2,
                          gain=round(gain, 2), steer_p=round(sp, 2), steer_v=round(sv, 2),
                          ret=tuple(np.round(np.clip(ret, -g, g), 3).tolist()), ret_steps=rs))
    return out


def evaluate(params, reps, servo, p_gain, seed, J, groups=GROUPS, difficulty=0.0):
    """Each candidate x reps envs in every start group. Returns good fraction, mean score, per-group results."""
    flat = [p for p in params for _ in range(reps)]
    res = {}
    for gi, (gname, r) in enumerate(groups.items()):
        env = make_env(len(flat), servo, p_gain, r, seed + 101 * gi, difficulty=difficulty)
        res[gname] = (*run(env, flat, J), run.padz)
    ok = torch.stack([score(a, d, al, pz)[0].float().view(len(params), reps).mean(1)
                      for a, d, al, _, pz in res.values()]).mean(0)
    val = torch.stack([score(a, d, al, pz)[1].view(len(params), reps).mean(1) for a, d, al, _, pz in res.values()]).mean(0)
    if RANK_OK[0]:
        val = ok + 0.1 * val
    return ok, val, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_gain", type=int, default=32)
    ap.add_argument("--servo", default="12v_r10")
    ap.add_argument("--candidates", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "launch_high.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dip_min", type=int, default=1, help="fewest dip steps searched (deeper dips: the ball falls)")
    ap.add_argument("--drift_w", type=float, default=2.0, help="score weight on drift at the first apex")
    ap.add_argument("--apex", default="0.15,0.20", help="good first apex range in m")
    ap.add_argument("--pad_w", type=float, default=0.0, help="score weight on the paddle's height above home at the "
                    "first apex (a return stroke brings it down before the hand-off)")
    ap.add_argument("--ret", action="store_true", help="search a return stroke after the strike")
    ap.add_argument("--start", default="", help="comma-separated NAMED launches to search around (local search only)")
    ap.add_argument("--rank_ok", action="store_true", help="rank by the share of good launches, score as tie-break")
    a = ap.parse_args()
    DIP_MIN[0], DRIFT_W[0], PAD_W[0], RET[0], RANK_OK[0] = a.dip_min, a.drift_w, a.pad_w, a.ret, a.rank_ok
    APEX[:] = [float(x) for x in a.apex.split(",")]
    K = dict(reps1=2, top=48, reps2=12, local=96, reps3=16, final=256) if not a.quick else \
        dict(reps1=1, top=4, reps2=2, local=4, reps3=2, final=8)
    if a.quick:
        a.candidates = 16
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    probe = BounceEnv(EnvCfg(num_envs=1, device="cpu", servo_variant=a.servo, p_gain=float(a.p_gain)))
    J = steer_jacobian(ArmKinematics(probe.model, 1), probe.q_nom)
    if a.start:                                            # local search around named launches
        cand = []
        for nm in a.start.split(","):
            c0 = Launch(**NAMED[nm].as_dict())
            cand += [c0] + rand_params(rng, a.candidates // len(a.start.split(",")), c0, 0.25)
    else:
        cand = rand_params(rng, a.candidates)
    if DEFAULTS[a.p_gain].dip_steps >= DIP_MIN[0]:
        cand.append(Launch(**{**DEFAULTS[a.p_gain].as_dict()}))
    ok, val, _ = evaluate(cand, K["reps1"], a.servo, a.p_gain, a.seed, J)
    top = [cand[i] for i in torch.argsort(val, descending=True)[:K["top"]].tolist()]
    print(f"stage 1: {len(cand)} candidates; best good {100 * ok.max():.0f}%", flush=True)
    ok, val, _ = evaluate(top, K["reps2"], a.servo, a.p_gain, a.seed + 1, J)
    best = top[int(torch.argmax(val))]
    print(f"stage 2: best robust good {100 * ok.max():.0f}%: {best}", flush=True)
    for sigma in (0.15, 0.07):
        loc = [best] + rand_params(rng, K["local"], best, sigma)
        ok, val, _ = evaluate(loc, K["reps3"], a.servo, a.p_gain, a.seed + 2, J)
        best = loc[int(torch.argmax(val))]
        print(f"stage 3 (sigma {sigma}): good {100 * ok.max():.0f}%", flush=True)
    print("launch:", json.dumps(best.as_dict()))
    report = {"launch": best.as_dict(), "p_gain": a.p_gain, "servo": a.servo, "eval": {}}
    for d in (0.0, 0.5):
        _, _, res = evaluate([best], K["final"], a.servo, a.p_gain, a.seed + 7, J, difficulty=d)
        for gname, (apex, drift, alive, vmax, padz) in res.items():
            okg = score(apex, drift, alive, padz)[0].float().mean().item()
            al = apex[alive]
            row = {"good": okg, "alive": alive.float().mean().item(),
                   "apex_cm_median": float(100 * al.median()) if len(al) else None,
                   "apex_cm_p10_p90": [float(100 * al.quantile(0.1)), float(100 * al.quantile(0.9))] if len(al) else None,
                   "drift_cm_median": float(100 * drift[alive].median()) if len(al) else None,
                   "paddle_up_speed_m_s": float(vmax.median()),
                   "paddle_above_home_cm_at_apex": float(100 * padz[alive].median()) if len(al) else None}
            report["eval"][f"d={d} {gname}"] = row
            print(f"  d={d} {gname:8s}: good {100 * okg:.0f}%, alive {100 * row['alive']:.0f}%, apex cm median "
                  f"{row['apex_cm_median']}, drift cm {row['drift_cm_median']}, paddle up-speed {row['paddle_up_speed_m_s']:.2f} m/s, "
                  f"paddle {row['paddle_above_home_cm_at_apex']} cm above home at the apex")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
