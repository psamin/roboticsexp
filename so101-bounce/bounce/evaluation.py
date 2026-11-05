"""Held-out evaluation: fixed seeds, fixed randomisation level, one episode per environment."""
import torch

from .env import FAIL_REASONS

SUITES = {
    # the user's hardware today: overhead C920 only (no side camera), measured-centre parameters, no taps
    "nominal": {"difficulty": 0.0, "rest_start": None, "side_cam": False},
    "full": {"difficulty": 1.0, "rest_start": None},                       # everything randomised, either camera setup
    "moderate": {"difficulty": 0.5, "rest_start": None, "side_cam": False},  # half-range variation, overhead only
    "user_hw": {"difficulty": 1.0, "rest_start": None, "side_cam": False}, # full randomisation, overhead camera only
    "launch": {"difficulty": 1.0, "rest_start": True},                     # every episode starts with the ball at rest
    "off_center": {"difficulty": 1.0, "rest_start": True, "rest_radius": 0.65},   # resting 55-75 % out to the rim
}
# opt-in through train.eval_suites (not in the default score)
EXTRA_SUITES = {"easy": {"difficulty": 0.25, "rest_start": None, "side_cam": False}}


@torch.no_grad()
def evaluate(env, policy, seed: int, suites=SUITES):
    out = {}
    for name, forced in suites.items():
        env.gen.manual_seed(seed)
        env.forced = dict(forced)
        obs, _ = env.reset_all()
        done_once = torch.zeros(env.N, dtype=torch.bool, device=env.device)
        rows = []
        for _ in range(env.max_steps + 1):
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
            new = (term | trunc) & ~done_once
            stats = env.pop_completed()
            if stats is not None and bool(new.any()):
                keep = new[stats["env_id"].long()]
                rows.append({k: v[keep] for k, v in stats.items()})
            done_once |= term | trunc
            if bool(done_once.all()):
                break
        s = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
        res = {"episodes": int(s["survived"].numel()),
               "survival": float(s["survived"].mean()),
               "bounces_mean": float(s["bounces"].mean()),
               "bounces_median": float(s["bounces"].median()),
               "length_s": float(s["length_s"].mean()),
               "apex_err_cm": 100 * float(s["apex_err"][s["bounces"] > 0].mean()) if bool((s["bounces"] > 0).any())
               else float("nan"),
               "lateral_cm": 100 * float(s["lateral"].mean()),
               "peak_load": float(s["peak_load"].mean())}
        rs = s["rest_start"] > 0
        res["launch_rate"] = float(s["launched"][rs].mean()) if bool(rs.any()) else float("nan")
        for i, reason in enumerate(FAIL_REASONS[:-1]):
            res[f"fail_{reason}"] = float((s["reason"] == i).float().mean())
        out[name] = res
    env.forced = None
    env.gen.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,))))
    env.reset_all()
    return out


def score(res):
    """Model selection: survival and bounce count across suites (bounces capped at 25 per 8 s episode)."""
    vals = [r["survival"] + min(r["bounces_mean"], 25.0) / 25.0 for r in res.values()]
    return sum(vals) / len(vals)
