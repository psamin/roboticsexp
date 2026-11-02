"""Ping-pong ball: flight with drag and Magnus lift, and impulse-based impacts with the paddle face.

Analytic rather than a general contact solver, as in table-tennis RL work (DeepMind 2024, Sony AI 2026): a 2.7 g ball
needs sub-millisecond steps in a soft-contact solver, while an impulse model gets the restitution exactly and is cheap.

Impact at the contact point (relative to the paddle surface moving under it):
  normal:     v_n' = -e * v_n                      (e drops ~0.02 per m/s of impact speed on rubber)
  tangential: friction impulse drives the contact point toward rolling, capped by Coulomb mu * J_n;
              for a thin spherical shell I = 2/3 m r^2, so full rolling needs J_t = -0.4 m v_t
Below a small approach speed the impact is perfectly inelastic, so a resting ball stays put (rolls on a tilt).
"""
import torch

RADIUS = 0.02
MASS = 0.0027
SHELL_K = 2.0 / 3.0                     # I / (m r^2) for a thin shell
REST_SPEED = 0.05                        # m/s, slower approaches do not bounce

# name: (nominal, low, high); per-episode, sampled like the servo parameters
BALL_RANGES = {
    "e0": (0.85, 0.72, 0.92),            # restitution at 1.5 m/s: worn/pimpled rubber to fresh inverted rubber
    "e_slope": (-0.02, -0.04, 0.0),      # per m/s
    "mu": (0.25, 0.08, 0.6),             # dusty or long-pimple rubber to tacky rubber
    "drag": (0.14, 0.10, 0.18),          # 1/m: 0.5 rho Cd A / m with Cd 0.4-0.6
    "magnus": (0.003, 0.0, 0.008),       # (w x v) coefficient
    "edge": (0.05, 0.0, 0.15),           # restitution lost at the rim (sweet spot vs edge)
    "e_noise": (0.0, 0.0, 0.03),         # std of per-impact restitution scatter
    "kick": (0.0, 0.0, 0.05),            # m/s, std of per-impact sideways scatter
    "roll": (1.0, 0.6, 1.0),             # how far one impact goes toward rolling (tangential restitution)
}


def flight(p, v, w, drag, magnus, dt, g, wind=None):
    """g: (3,) or (N, 3) (a tilted table tilts gravity relative to the robot); wind: (N, 3) air velocity."""
    rel = v if wind is None else v - wind
    speed = rel.norm(dim=-1, keepdim=True)
    a = g - drag[:, None] * speed * rel + magnus[:, None] * torch.cross(w, rel, dim=-1)
    v = v + a * dt
    return p + v * dt, v


def paddle_impact(p, v, w, o, R, o_prev, R_prev, prm, half_a, half_b, shape, dt, e_noise=None, kick=None):
    """Resolve ball-paddle contact after a flight step. Paddle frame: R columns (along handle, across, normal).

    The blade outline is a superellipse |du/a|^shape + |dv/b|^shape <= 1 (shape 2 = oval, higher = squarer).
    Restitution falls toward the rim (prm["edge"] * normalised radius^2). e_noise (N,) and kick (N, 3) are this
    step's random restitution change and sideways velocity kick, applied only on impact (ball seams, defects).
    Returns updated (p, v, w) and per-env: touching, impact (approach faster than REST_SPEED), approach speed,
    in-plane offsets (du, dv) of the ball centre from the blade centre."""
    u, side, n = R[..., 0], R[..., 1], R[..., 2]
    rel = p - o
    s = (rel * n).sum(-1)
    du, dv = (rel * u).sum(-1), (rel * side).sum(-1)
    on_blade = (du / half_a).abs().pow(shape) + (dv / half_b).abs().pow(shape) <= 1.0
    touching = on_blade & (s < RADIUS) & (s > -0.5 * RADIUS)
    # velocity of the paddle surface under the ball
    pt = o + du[:, None] * u + dv[:, None] * side
    pt_prev = o_prev + du[:, None] * R_prev[..., 0] + dv[:, None] * R_prev[..., 1]
    vp = (pt - pt_prev) / dt
    rn = -RADIUS * n
    vc = v + torch.cross(w, rn, dim=-1) - vp
    vn = (vc * n).sum(-1)
    approaching = touching & (vn < 0)
    speed_in = (-vn).clamp(min=0)
    rim2 = (du / half_a).pow(2) + (dv / half_b).pow(2)
    e = prm["e0"] + prm["e_slope"] * (speed_in - 1.5) - prm["edge"] * rim2
    if e_noise is not None:
        e = e + e_noise
    e = e.clamp(0.3, 0.95)
    e = torch.where(speed_in < REST_SPEED, torch.zeros_like(e), e)
    jn = torch.where(approaching, -(1 + e) * vn * MASS, torch.zeros_like(vn))           # >= 0
    vt = vc - vn[:, None] * n
    jt = -prm["roll"][:, None] * MASS * SHELL_K / (1 + SHELL_K) * vt
    jt_norm = jt.norm(dim=-1).clamp(min=1e-9)
    jt = jt * torch.clamp(prm["mu"] * jn / jt_norm, max=1.0)[:, None]
    jt = torch.where(approaching[:, None], jt, torch.zeros_like(jt))
    v = v + (jn[:, None] * n + jt) / MASS
    if kick is not None:
        hit = approaching & (speed_in >= REST_SPEED)
        kick_t = kick - (kick * n).sum(-1, keepdim=True) * n
        v = v + torch.where(hit[:, None], kick_t, torch.zeros_like(v))
    w = w + torch.cross(rn, jt, dim=-1) / (SHELL_K * MASS * RADIUS ** 2)
    p = p + torch.where(touching[:, None], (RADIUS - s).clamp(min=0)[:, None] * n, torch.zeros_like(p))
    return p, v, w, touching, approaching & (speed_in >= REST_SPEED), speed_in, du, dv
