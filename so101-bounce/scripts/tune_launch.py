"""Tune the dead-stop launch primitive (bounce/launch.py) in the batched simulator, then check the hand-off to a policy.
CPU only, a few seconds per stage.

  uv run python scripts/tune_launch.py --p_gain 32 --servo 12v_r8
  uv run python scripts/tune_launch.py --p_gain 32 --check "dip=..." --handoff policies/v12-p32.pt

Each candidate runs from a ball resting at the paddle centre. Scored on the first bounce: apex 7-13 cm above the paddle
(target 10), within 3 cm of the centre at the apex, and the episode still alive. Ball-paddle restitution is drawn from
0.60-0.85 (the user's paddle is "hard"); robustness stages also use difficulty 0.5.
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
from bounce.launch import Launch, actions                   # noqa: E402
from bounce.ppo import PPO                                  # noqa: E402

WINDOW = 60                                                 # control steps watched from the start (1.2 s at 50 Hz)
BACK_W = [0.0]                                              # score weight per m of first-apex drift toward -x (the
                                                            # handle and the base); > 0 also requires < 1 cm of it
KEEPOUT = [None]                                            # m: a ball this far behind the paddle centre (-x) within
                                                            # 4 cm of paddle height counts as lost (hits the handle)
ENV_EXTRA = {}                                              # EnvCfg overrides for every env here (e.g. paddle_box)


def make_env(n, servo, p_gain, difficulty, e0, seed=0, base=None):
    kw = {**ENV_EXTRA, **(base or {})}
    kw.update(num_envs=n, device="cpu", servo_variant=servo, p_gain=float(p_gain), seed=seed)
    env = BounceEnv(EnvCfg(**kw))
    e0 = torch.as_tensor(e0, dtype=torch.float32).expand(n).clone()

    def post(env_, ids):
        env_.ball_prm["e0"][ids] = e0[ids]
    env.forced = {"difficulty": difficulty, "rest_start": True, "rest_radius": 0.0, "post_reset": post}
    env.reset_all()
    return env


def run_primitives(env, seq, extra=0):
    """seq (T, N, 5). Returns first-apex height above the paddle (m), drift at that apex (m), alive through the
    window, the paddle's distance from home at the end, and the signed x drift at the apex (m, + = away from the base)."""
    N = env.N
    alive = torch.ones(N, dtype=torch.bool)
    best = torch.full((N,), -1.0)
    drift = torch.zeros(N)
    dx = torch.zeros(N)
    home = env.home_env.clone()
    T = max(seq.shape[0], WINDOW + extra)
    for t in range(T):
        a = seq[t] if t < seq.shape[0] else torch.zeros(N, 5)
        _, _, term, _, _ = env.step(a)
        alive &= ~term
        if KEEPOUT[0] is not None:
            po = env.kin.forward(env.q)["paddle_o"]
            alive &= ~((env.p_ball[:, 0] - po[:, 0] < -KEEPOUT[0]) & (env.p_ball[:, 2] < po[:, 2] + 0.04))
        h = env.p_ball[:, 2] - B.RADIUS - home[:, 2]
        up = alive & (h > best)
        best = torch.where(up, h, best)
        drift = torch.where(up, (env.p_ball[:, :2] - home[:, :2]).norm(dim=-1), drift)
        dx = torch.where(up, env.p_ball[:, 0] - home[:, 0], dx)
    pad = (env.kin.forward(env.q)["paddle_o"] - home).norm(dim=-1)
    return best, drift, alive, pad, dx


def score(apex, drift, alive, dx=None):
    ok = alive & (apex > 0.07) & (apex < 0.13) & (drift < 0.03)
    val = -(apex - 0.10).abs() - 2 * drift
    if dx is not None and BACK_W[0] > 0:
        ok = ok & (dx > -0.01)
        val = val - BACK_W[0] * (-dx).clamp(min=0)
    return ok, torch.where(alive, val, torch.full_like(val, -1.0))


def seq_from(params, n_each):
    """Per-env sequences for a list of Launch params, each repeated n_each times."""
    T = max(p.steps for p in params)
    out = torch.zeros(T, len(params) * n_each, 5)
    for i, p in enumerate(params):
        s = actions(p, n_each)
        out[:s.shape[0], i * n_each:(i + 1) * n_each] = s
    return out


def random_params(rng, k, center=None, sigma=0.0):
    out = []
    for _ in range(k):
        if center is None:
            dip, strike = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            ds, ss = int(rng.integers(0, 6)), int(rng.integers(2, 10))     # slow loaded joints: long strikes
        else:
            dip = np.clip(np.array(center.dip) + rng.normal(0, sigma, 3), -1, 1)
            strike = np.clip(np.array(center.strike) + rng.normal(0, sigma, 3), -1, 1)
            ds = int(np.clip(center.dip_steps + rng.integers(-1, 2), 0, 6))
            ss = int(np.clip(center.strike_steps + rng.integers(-1, 2), 1, 10))
        out.append(Launch(dip=tuple(np.round(dip, 3).tolist()), dip_steps=ds,
                          strike=tuple(np.round(strike, 3).tolist()), strike_steps=ss, settle_steps=2))
    return out


def robust(params, servo, p_gain, reps, rng, diffs=(0.0, 0.5)):
    """Fraction of good first bounces per candidate over restitution 0.60-0.85 and the given difficulties."""
    res = []
    for d in diffs:
        n = len(params) * reps
        e0 = torch.tensor(rng.uniform(0.60, 0.85, n), dtype=torch.float32)
        env = make_env(n, servo, p_gain, d, e0, seed=int(rng.integers(1 << 30)))
        apex, drift, alive, _, dx = run_primitives(env, seq_from(params, reps))
        ok, val = score(apex, drift, alive, dx)
        res.append((ok.view(len(params), reps).float().mean(1), val.view(len(params), reps).mean(1),
                    apex.view(len(params), reps), drift.view(len(params), reps), alive.view(len(params), reps)))
    frac = torch.stack([r[0] for r in res]).mean(0)
    val = torch.stack([r[1] for r in res]).mean(0)
    return frac, val, res


def summary(name, apex, drift, alive, dx):
    a = apex[alive]
    print(f"  {name}: good {100 * score(apex, drift, alive, dx)[0].float().mean():.0f}%, alive {100 * alive.float().mean():.0f}%, "
          f"apex cm median {100 * a.median():.1f} (p10 {100 * a.quantile(0.1):.1f}, p90 {100 * a.quantile(0.9):.1f}), "
          f"drift cm median {100 * drift[alive].median():.1f}, x drift cm median {100 * dx[alive].median():+.1f} "
          f"(p10 {100 * dx[alive].quantile(0.1):+.1f})" if len(a) else f"  {name}: none alive")


def handoff(ckpt, p, servo, n, diff, seed=0, use_launch=True):
    """From rest, the primitive first (or not), then the policy: 8 s survival, the share that gets the ball 7 cm above
    the paddle within 2 s (a real launch, not a dribble), and the median time to get there."""
    agent, extra = PPO.load(ckpt, "cpu", load_optimizer=False)
    ecfg = (extra.get("config") or {}).get("env", {})
    base = {k: tuple(v) if isinstance(v, list) else v for k, v in ecfg.items() if k in EnvCfg.__dataclass_fields__}
    rng = np.random.default_rng(seed)
    env = make_env(n, servo, base.get("p_gain", 16), diff, torch.tensor(rng.uniform(0.6, 0.85, n), dtype=torch.float32),
                   seed=seed, base=base)
    seq = actions(p, n) if use_launch else torch.zeros(0, n, 5)
    alive = torch.ones(n, dtype=torch.bool)
    reach_t = torch.full((n,), float("nan"))
    home = env.home_env.clone()
    obs = env._obs()
    for t in range(env.max_steps):
        a = seq[t] if t < seq.shape[0] else agent.act_deterministic(obs["policy"])
        obs, _, term, trunc, _ = env.step(a)
        alive &= ~term
        h = env.p_ball[:, 2] - B.RADIUS - home[:, 2]
        new = alive & torch.isnan(reach_t) & (h > 0.07)
        reach_t[new] = (t + 1) * env.dt
        if bool((term | trunc).all()):
            break
    fast = (reach_t <= 2.0).float().mean().item()
    med = float(np.nanmedian(reach_t.numpy())) if (~torch.isnan(reach_t)).any() else float("nan")
    return alive.float().mean().item(), fast, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_gain", type=int, default=32)
    ap.add_argument("--servo", default="12v_r8")
    ap.add_argument("--candidates", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--handoff", nargs="*", default=[], help="policy checkpoints to test the hand-off with")
    ap.add_argument("--check", default=None, help="skip tuning: JSON of a Launch to evaluate")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "launch_tune.json"))
    ap.add_argument("--quick", action="store_true", help="tiny sizes: a smoke test of the script")
    ap.add_argument("--back_w", type=float, default=0.0, help="penalty per m of first-apex drift toward -x")
    ap.add_argument("--keepout", type=float, default=None, help="m behind the paddle centre that counts as the handle")
    ap.add_argument("--paddle_box", default="", help="xy,z paddle limit (m), e.g. 0.12,0.10")
    ap.add_argument("--evals", default="12v", help="extra servo sets for the final evaluation, comma-separated")
    a = ap.parse_args()
    BACK_W[0], KEEPOUT[0] = a.back_w, a.keepout
    if a.paddle_box:
        ENV_EXTRA["paddle_box"] = tuple(float(x) for x in a.paddle_box.split(","))
    K = dict(top=8, reps=4, local=15, final=64, hand=16) if a.quick else dict(top=64, reps=24, local=127, final=1024, hand=256)
    if a.quick:
        a.candidates = 64
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    if a.check:
        best = Launch(**json.loads(a.check))
    else:
        # stage 1: broad random search, one env per candidate, e0 0.72, nominal
        cand = random_params(rng, a.candidates)
        env = make_env(len(cand), a.servo, a.p_gain, 0.0, 0.72, seed=a.seed)
        apex, drift, alive, _, dx = run_primitives(env, seq_from(cand, 1))
        _, val = score(apex, drift, alive, dx)
        top = [cand[i] for i in val.argsort(descending=True)[:K["top"]].tolist()]
        print(f"stage 1: {len(cand)} candidates; best single-run apex {100 * apex[val.argmax()]:.1f} cm")
        # stage 2: robustness of the top 64
        frac, v2, _ = robust(top, a.servo, a.p_gain, K["reps"], rng)
        order = (frac + 0.1 * v2).argsort(descending=True).tolist()
        best = top[order[0]]
        print(f"stage 2: best robust good-bounce rate {100 * frac[order[0]]:.0f}%")
        # stage 3: local refinement around the best
        for sigma in (0.15, 0.07):
            local = [best] + random_params(rng, K["local"], best, sigma)
            frac, v3, _ = robust(local, a.servo, a.p_gain, max(4, K["reps"] * 2 // 3), rng)
            i = int((frac + 0.1 * v3).argmax())
            best = local[i]
            print(f"stage 3 (sigma {sigma}): good-bounce rate {100 * frac[i]:.0f}%")
    print("primitive:", json.dumps(best.as_dict()))
    # final evaluation on fresh seeds: this servo set and the old 12 V set, nominal and difficulty 0.5
    report = {"launch": best.as_dict(), "p_gain": a.p_gain, "eval": {}}
    for servo in dict.fromkeys([a.servo] + [s for s in a.evals.split(",") if s]):
        for d in (0.0, 0.5):
            n = K["final"]
            env = make_env(n, servo, a.p_gain, d, torch.tensor(rng.uniform(0.6, 0.85, n), dtype=torch.float32),
                           seed=int(rng.integers(1 << 30)))
            apex, drift, alive, pad, dx = run_primitives(env, seq_from([best], n))
            summary(f"{servo} d={d}", apex, drift, alive, dx)
            ok, _ = score(apex, drift, alive, dx)
            report["eval"][f"{servo} d={d}"] = {"good": ok.float().mean().item(), "alive": alive.float().mean().item(),
                                                "apex_cm_median": 100 * apex[alive].median().item() if alive.any() else None,
                                                "x_drift_cm_median": 100 * dx[alive].median().item() if alive.any() else None}
    for ck in a.handoff:
        for d in (0.0, 0.5):
            with_l = handoff(ck, best, a.servo, K["hand"], d, seed=11, use_launch=True)
            without = handoff(ck, best, a.servo, K["hand"], d, seed=11, use_launch=False)
            print(f"  hand-off {Path(ck).name} {a.servo} d={d}: survival {100 * without[0]:.0f}% -> {100 * with_l[0]:.0f}%, "
                  f"7 cm bounce within 2 s {100 * without[1]:.0f}% -> {100 * with_l[1]:.0f}%, "
                  f"median time to it {without[2]:.2f} -> {with_l[2]:.2f} s  (without -> with the launch)")
            report["eval"][f"handoff {Path(ck).name} d={d}"] = {"without": without, "with": with_l}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
