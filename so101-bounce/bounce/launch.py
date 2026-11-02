"""Scripted launch from a dead stop: drop the paddle away so the resting ball floats, then strike it on the way back
up with a flick, then hand over to the policy with the ball in flight.

Why: the real servos are speed-limited (~2.3 rad/s, servo set "12v_r8"). Moving the paddle straight up with the face
level needs ~10 rad of elbow per metre, so a level push tops out near 0.25 m/s, while a 10 cm bounce needs the ball
to leave at 1.4 m/s. Two things help: a strike adds the rebound (v_out = (1 + e) v_paddle + e v_ball_in), and swinging
lift, elbow and wrist flex the same way (a flick) moves the blade centre several times faster than a level push.

A primitive is a sequence of policy actions at the control rate (goal = q_ref + action_scale * a, |a| <= 1; pan and
wrist roll stay 0), so the simulator, the rehearsal and the real runner execute exactly the same thing. The numbers
come from scripts/tune_launch.py.
"""
from dataclasses import asdict, dataclass

import torch


@dataclass
class Launch:
    dip: tuple = (0.0, 0.0, 0.0)      # action (lift, elbow, wrist flex) held while the paddle drops away
    dip_steps: int = 2
    strike: tuple = (0.0, 0.0, 0.0)   # action held for the upward stroke
    strike_steps: int = 3
    settle_steps: int = 2             # hold command (a = 0) before the policy takes over
    gain: float = 1.0                 # largest |action|: above 1 the goal overshoots the policy's range, so the servo's
                                      # P loop saturates sooner and the torque-capped joints accelerate harder
    steer_p: float = 0.0              # strike face tilt (rad of face normal) per m of ball offset from the centre
    steer_v: float = 0.0              # ... per m/s of ball rolling speed; tilts toward the centre (see steer())
    ret: tuple = (0.0, 0.0, 0.0)      # action held after the strike to bring the paddle back down toward home before
    ret_steps: int = 0                # the policy takes over (a hard strike leaves it ~8 cm up while the ball drifts)

    def as_dict(self):
        return asdict(self)

    @property
    def steps(self):
        return self.dip_steps + self.strike_steps + self.ret_steps + self.settle_steps


# per servo P gain, tuned in simulation on the refit 12 V servo set "12v_r8" (scripts/tune_launch.py, job 13220):
# first bounce 9.1-9.2 cm (p10-p90 8.8-10.1) with restitution 0.60-0.85, drift 1-2 cm, at nominal parameters.
# Both dip by swinging lift, elbow and wrist flex the same way, then strike back with a flick (about 0.1 s each).
# Round 10: retuned on "12v_r10" (torque-capped loaded joints, fitted to the real-ball runs; job 13324): the 12v_r8 P32
# launch moved the real paddle at 0.45 m/s instead of ~1 m/s. P32: first bounce 10.7 cm median (p10-p90 8.3-11.7),
# 98 % good at nominal. P16: 8.4 cm, 100 % good. On the faster 12v_r8/12v sets these overshoot; the runner adapts the
# strike length from the first bounce, and training replays strikes of 3-8 steps (launch_strike_range).
DEFAULTS = {
    32: Launch(dip=(0.181, 0.66, 0.51), dip_steps=1, strike=(0.04, -0.629, -0.548), strike_steps=5, settle_steps=2),
    16: Launch(dip=(-0.147, 0.056, 0.554), dip_steps=5, strike=(-0.649, 0.192, -0.842), strike_steps=8, settle_steps=2),
}
# Round 12: higher, steered launches for dead paddles (scripts/tune_launch_high.py on 12v_r10, P32; starts at the
# centre, half-way and near the rim, rolling up to 0.3 m/s, paddle up to 2 deg off, e 0.55-0.75). Commands above
# the policy's range (gain) drive the paddle to ~1.0 m/s, up from 0.67 with the round-10 launch.
#   r12:      first bounce 12.3-12.5 cm median (rim 11.2), drift 3.7-4.5 cm (rim 6.6); at d = 0.5 72-84 % reach the apex
#   r12_deep: 4-step dip so the ball falls first: 14.9-15.2 cm (rim 12.4), drift 5.4-6.4 cm (rim 8.1); d = 0.5 ~50 %
# Selected by name (EnvCfg.launch_name, scenario_test --launch_name); DEFAULTS stay the round-10 launch.
NAMED = {
    "r12": Launch(dip=(0.345, 0.547, -0.861), dip_steps=1, strike=(-0.242, -0.513, -0.728), strike_steps=5,
                  settle_steps=2, gain=1.56, steer_p=0.82, steer_v=0.26),
    "r12_deep": Launch(dip=(-0.298, -0.597, -0.802), dip_steps=4, strike=(-0.286, 0.199, 0.737), strike_steps=5,
                       settle_steps=2, gain=1.94, steer_p=2.36, steer_v=0.86),
}


def get(p_gain, name=""):
    """The launch to run: a named one, else the tuned default for this P gain."""
    return NAMED[name] if name else DEFAULTS[int(round(p_gain))]


# round 9 values (12v_r8): P32 dip (0.27, 0.596, 0.849) x1, strike (-0.185, -0.516, -0.88) x3;
#                          P16 dip (0.557, 0.93, 0.706) x2, strike (-0.552, -0.961, -0.598) x5


def actions(p: Launch, n: int = 1, device="cpu", steer=None):
    """(T, n, 5) actions for the primitive (pan and wrist roll 0). steer (n, 2): extra wrist flex / roll actions added
    during the strike (from steer())."""
    g = float(getattr(p, "gain", 1.0))

    def a(v, s=None):
        out = torch.zeros(n, 5, device=device)
        out[:, 1:4] = torch.tensor(v, dtype=torch.float32, device=device)
        if s is not None:
            out[:, 3:5] = out[:, 3:5] + s
        return out.clamp(-g, g)
    st = None if steer is None else torch.as_tensor(steer, dtype=torch.float32, device=device).reshape(n, 2)
    seq = [a(p.dip)] * p.dip_steps + [a(p.strike, st)] * p.strike_steps + \
        [a(getattr(p, "ret", (0, 0, 0)))] * int(getattr(p, "ret_steps", 0)) + [a((0, 0, 0))] * p.settle_steps
    return torch.stack(seq)


def steer_jacobian(kin, q):
    """(2, 2): d(face normal x, y) / d(wrist flex, wrist roll) at sim joint angles q (5,)."""
    q = torch.as_tensor(q, dtype=torch.float32).reshape(1, 5)
    n0 = kin.forward(q)["paddle_R"][0, :2, 2]
    cols = []
    for j in (3, 4):
        dq = q.clone()
        dq[0, j] += 1e-3
        cols.append((kin.forward(dq)["paddle_R"][0, :2, 2] - n0) / 1e-3)
    return torch.stack(cols, 1)


def steer(p: Launch, J, scale, off_xy, vel_xy):
    """(n, 2) wrist flex / roll actions that tilt the face during the strike so the ball, offset off_xy (n, 2) m from
    the paddle centre and rolling at vel_xy (n, 2) m/s, flies back toward the centre. A ball leaves along the face
    normal, so the normal leans away from the offset: d n_xy = -(steer_p * off + steer_v * vel)."""
    if not (p.steer_p or p.steer_v):
        return None
    off_xy = torch.as_tensor(off_xy, dtype=torch.float32)
    dev = off_xy.device
    want = -(p.steer_p * off_xy + p.steer_v * torch.as_tensor(vel_xy, dtype=torch.float32, device=dev))
    dq = torch.linalg.solve(torch.as_tensor(J, dtype=torch.float32, device=dev), want.T).T   # (n, 2) radians
    sc = torch.tensor([float(scale[3]), float(scale[4])], dtype=torch.float32, device=dev)
    g = float(getattr(p, "gain", 1.0))
    return (dq / sc).clamp(-g, g)
