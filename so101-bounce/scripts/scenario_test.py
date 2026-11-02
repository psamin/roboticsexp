"""Every simulatable start and failure case from research/scenarios_and_failures.md, for one checkpoint.

All scenarios run at once: each gets its own slice of `--per` environments, and an evaluation hook sets that slice's
start (resting / dropped / beside / tossed / held) and its one injected fault after every reset. The arm model is the
user's: 12 V servos refitted to the real arm ("12v_r8", 18 ms command delay), overhead camera only, at the
calibrated real mount with the measured latency. Every checkpoint is tested in this same base setup (only its
control interface: rate, history, P gain, action scale, is taken from the checkpoint), so policies trained with
different randomisation are compared fairly. A ball dropped from above the 30 cm out-of-view ceiling is not failed
until it has touched the paddle. Resting rows are repeated "+ launch": the runner's scripted dip-and-strike
(bounce/launch.py) runs first, then the policy. Rows are printed per randomisation level (and per forced target
height with --targets).

  python scripts/scenario_test.py --checkpoint best.pt [--per 512] [--difficulties 0,0.5] [--device cuda]
      [--servo 12v_r8] [--launch both|on|off] [--targets 7,10,13,15] [--only "row;row"]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bounce import ball as B
from bounce import launch as LAUNCH
from bounce.robot import ArmKinematics          # noqa: E402
from bounce.env import FAIL_REASONS, BounceEnv, EnvCfg
from bounce.ppo import PPO

INTERFACE = ("ctrl_hz", "substeps", "history", "cam_k", "episode_s", "p_gain", "action_scale", "nominal_lift",
             "nominal_elbow", "paddle_box")       # paddle_box: a higher launch's strike may rise 8-10 cm
DELAY = {"12v_r10": (18.0, 5.0, 35.0), "12v_r10w": (18.0, 5.0, 35.0), "12v_r8": (18.0, 5.0, 35.0), "12v": (30.0, 8.0, 40.0),
         "5v": (8.0, 4.0, 40.0)}
REAL_CAM = (-0.068, 0.012, 0.732)          # real/camera_calib.json, 2026-10-31
REAL_CAM_BASE_MS = (14.0, 8.0, 50.0)       # + 1.5 frames at 30 fps = 64 ms: 62 ms measured capture-to-read + finder

DEG = math.pi / 180

# starts: kind rest (radius as a fraction of the blade) or flight (height of the ball bottom, lateral offset,
# sideways speed, vertical speed, spin, held seconds, held hidden)
REST_C = {"kind": "rest", "r": 0.1}
DROP15 = {"kind": "flight", "h": 0.15, "off": 0.0}
STARTS = {
    "rest centre": REST_C,
    "rest half-way": {"kind": "rest", "r": 0.5},
    "rest near rim": {"kind": "rest", "r": 0.85},
    "rest rolling 0.2 m/s": {"kind": "rest", "r": 0.3, "roll": 0.2},
    "rest near rim, rolling 0.15 m/s": {"kind": "rest", "r": 0.85, "roll": 0.15},
    "drop 10 cm": {"kind": "flight", "h": 0.10, "off": 0.0},
    "drop 20 cm, 6 cm off": {"kind": "flight", "h": 0.20, "off": 0.06},
    "drop 40 cm": {"kind": "flight", "h": 0.40, "off": 0.0},
    "beside 8 cm, 10 cm up": {"kind": "flight", "h": 0.10, "off": 0.08, "vlat": 0.2},
    "beside 12 cm, 20 cm up": {"kind": "flight", "h": 0.20, "off": 0.12, "vlat": 0.3},
    "beside 15 cm, 30 cm up": {"kind": "flight", "h": 0.30, "off": 0.15, "vlat": 0.3},
    "beside 8 cm, 20 cm up (reachable)": {"kind": "flight", "h": 0.20, "off": 0.08},
    "beside 12 cm, 30 cm up (reachable)": {"kind": "flight", "h": 0.30, "off": 0.12},
    "beside 15 cm, 40 cm up (reachable)": {"kind": "flight", "h": 0.40, "off": 0.15},
    "tossed 0.5 m/s + spin": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.5, "spin": 30.0},
    "tossed 0.3 m/s + spin 15": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.3, "spin": 15.0},
    "spin 30 rad/s, dropped centred": {"kind": "flight", "h": 0.15, "off": 0.0, "spin": 30.0},
    "bounced in (up 1.5 m/s)": {"kind": "flight", "h": 0.05, "off": 0.02, "vz": 1.5},
    # round 14: the oracle's recoverable envelope (scripts/oracle_feasibility.py), direction relative to the offset
    "toss 0.2 m/s toward centre": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.2, "vdir": "toward"},
    "toss 0.3 m/s toward centre": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.3, "vdir": "toward"},
    "toss 0.3 m/s toward centre, from 10 cm": {"kind": "flight", "h": 0.10, "off": 0.03, "vlat": 0.3, "vdir": "toward"},
    "toss 0.3 m/s toward centre, from 30 cm": {"kind": "flight", "h": 0.30, "off": 0.03, "vlat": 0.3, "vdir": "toward"},
    "toss 0.2 m/s sideways": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.2, "vdir": "side"},
    "toss 0.3 m/s sideways": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.3, "vdir": "side"},
    "toss 0.2 m/s away": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.2, "vdir": "away"},
    "toss 0.1 m/s away + spin 15 top": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.1, "vdir": "away",
                                        "spin": 15.0, "axis": "top"},
    "drop 20 cm centred, spin 30 top": {"kind": "flight", "h": 0.20, "off": 0.0, "spin": 30.0, "axis": "top"},
    "drop 20 cm centred, spin 30 side": {"kind": "flight", "h": 0.20, "off": 0.0, "spin": 30.0, "axis": "side"},
    "toss 0.5 m/s toward centre + spin 30 top": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.5, "vdir": "toward",
                                                 "spin": 30.0, "axis": "top"},
    "toss 0.5 m/s sideways + spin 30 side": {"kind": "flight", "h": 0.20, "off": 0.03, "vlat": 0.5, "vdir": "side",
                                             "spin": 30.0, "axis": "side"},
    "held 0.5 s visible": {**DROP15, "held": 0.5, "hidden": False},
    "held 1 s hidden": {**DROP15, "held": 1.0, "hidden": True},
}


def _rand_dir2(env, n):
    a = 2 * math.pi * env._u(n)
    return torch.stack([torch.cos(a), torch.sin(a)], -1)


def tilt(deg):
    def f(env, ids):
        n, th = ids.numel(), deg * DEG
        d = _rand_dir2(env, n) * math.sin(th)
        env.g_env[ids] = 9.81 * torch.cat([d, torch.full((n, 1), -math.cos(th), device=env.device)], -1)
    return f


def setv(attr, value):
    def f(env, ids):
        getattr(env, attr)[ids] = value
    return f


def compose(*fs):
    def f(env, ids):
        for g in fs:
            g(env, ids)
    return f


def cam_pos(x, y, z):
    def f(env, ids):
        env.cam_o[ids] = torch.tensor([x, y, z], device=env.device)
    return f


def dim15(env, ids):
    env.cam_period[ids] = 1 / 15
    env.exposure[ids] = 0.9 / 15
    env.sig_perp[ids] = env.sig_perp[ids] * 2.5
    env.cam_lat[ids] = 0.12


def wind(speed):
    def f(env, ids):
        env.wind[ids] = torch.cat([speed * _rand_dir2(env, ids.numel()), torch.zeros(ids.numel(), 1, device=env.device)], -1)
    return f


def strength(s):
    def f(env, ids):
        env.servo.p["strength"][ids] = env.servo.p["strength"][ids] * s
    return f


def joint_offset(deg):
    def f(env, ids):
        env.q_off[ids] = deg * DEG * (2 * (env._u(ids.numel(), 5) > 0.5).float() - 1)
    return f


def cam_calib(rot_deg, bias_m):
    def f(env, ids):
        n = ids.numel()
        ax = env._n(n, 3)
        ax = ax / ax.norm(dim=-1, keepdim=True) * rot_deg * DEG
        th = ax.norm(dim=-1, keepdim=True)
        k = ax / th
        K = torch.zeros(n, 3, 3, device=env.device)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
        s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
        env.cam_R[ids] = torch.eye(3, device=env.device) + s * K + (1 - c) * K @ K
        b = env._n(n, 3)
        env.cam_bias[ids] = bias_m * b / b.norm(dim=-1, keepdim=True)
    return f


def paddle_mass(kg):
    def f(env, ids):
        env.kin.mass[ids, env.kin.paddle] = kg
    return f


def taps(hz):
    def f(env, ids):
        env.push_p[ids] = hz * env.sub_dt
    return f


# faults: injected at reset, one per row, on the start that exposes them most
FAULTS = {
    "paddle 1 deg off level": (tilt(1.0), {"kind": "rest", "r": 0.3}),
    "paddle 2 deg off level": (tilt(2.0), {"kind": "rest", "r": 0.3}),
    "paddle 0 deg off level": (tilt(0.0), {"kind": "rest", "r": 0.3}),
    "paddle 3 deg off level": (tilt(3.0), {"kind": "rest", "r": 0.3}),
    "paddle 4 deg off level": (tilt(4.0), {"kind": "rest", "r": 0.3}),
    "camera latency 120 ms": (setv("cam_lat", 0.12), DROP15),
    "camera latency 150 ms": (setv("cam_lat", 0.15), DROP15),
    "timestamps 50 ms late": (setv("age_bias_env", 0.05), DROP15),
    "timestamps 100 ms late": (setv("age_bias_env", 0.10), DROP15),
    "depth noise 7 cm": (setv("sig_ray", 0.07), DROP15),
    "depth scale +10 %": (setv("depth_err", 0.10), DROP15),
    "camera above paddle": (cam_pos(0.35, 0.0, 1.10), DROP15),
    "camera in front": (cam_pos(0.60, 0.0, 0.90), DROP15),
    "camera low behind": (cam_pos(-0.25, 0.0, 0.70), DROP15),
    "15 fps, dim": (dim15, DROP15),
    "detection bursts": (compose(setv("burst_in", 0.03), setv("burst_out", 0.2)), DROP15),
    "false detections 2 %": (setv("outlier_p_env", 0.02), DROP15),
    "camera calib 3 deg, 2 cm": (cam_calib(3.0, 0.02), DROP15),
    "hand taps 1 Hz": (taps(1.0), DROP15),
    "wind 0.5 m/s": (wind(0.5), DROP15),
    "command delay 40 ms": (None, DROP15),                 # set below (depends on sub_dt)
    "USB stalls 3 %": (setv("glitch_p_env", 0.03), DROP15),
    "servos 15 % weaker (hot)": (strength(0.85), DROP15),
    "joint calibration 2 deg": (joint_offset(2.0), DROP15),
    "ball restitution 0.75": (None, DROP15),               # set below
    "paddle 220 g": (paddle_mass(0.22), DROP15),
    "dead paddle e 0.65, resting": (None, REST_C),          # set below
    "dead paddle e 0.65, dropped 15 cm": (None, DROP15),
    "dead paddle e 0.55, resting": (None, REST_C),
    "bad day: 1.5 deg, 120 ms, +50 ms, cam above": (compose(tilt(1.5), setv("cam_lat", 0.12),
                                                            setv("age_bias_env", 0.05), cam_pos(0.35, 0.0, 1.10)),
                                                    {"kind": "rest", "r": 0.3}),
}


def set_start(env, ids, st):
    n, dev = ids.numel(), env.device
    fk = env._fk_ids(env.q[ids], ids)
    o, R = fk["paddle_o"], fk["paddle_R"]
    if st["kind"] == "rest":
        r = (st["r"] + 0.05 * (2 * env._u(n) - 1)).clamp(0, 0.95)
        d, bl = _rand_dir2(env, n), env.blade[ids]
        p = o + B.RADIUS * R[..., 2] + (bl[:, 0] * r * d[:, 0])[:, None] * R[..., 0] + \
            (bl[:, 1] * r * d[:, 1])[:, None] * R[..., 1]
        v = torch.zeros(n, 3, device=dev)
        if st.get("roll"):                                        # already rolling along the face
            dr = _rand_dir2(env, n)
            v = st["roll"] * (dr[:, :1] * R[..., 0] + dr[:, 1:] * R[..., 1])
        w = torch.zeros(n, 3, device=dev)
        rest = torch.ones(n, dtype=torch.bool, device=dev)
    else:
        d_off = _rand_dir2(env, n)
        lat = st.get("off", 0.0) * d_off
        p = env.home_env[ids] + torch.cat([lat, torch.full((n, 1), st["h"] + B.RADIUS, device=dev)], -1)
        vd = {"toward": -d_off, "away": d_off, "side": torch.stack([-d_off[:, 1], d_off[:, 0]], -1)}.get(
            st.get("vdir"), None)
        vd = _rand_dir2(env, n) if vd is None else vd
        v = torch.cat([st.get("vlat", 0.0) * vd, torch.full((n, 1), st.get("vz", 0.0), device=dev)], -1)
        if st.get("axis") == "side":                              # spin about the vertical
            w = st.get("spin", 0.0) * torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3).clone()
        elif st.get("axis") == "top":                             # horizontal, across the direction of travel
            hz = torch.cat([-vd[:, 1:2], vd[:, 0:1], torch.zeros(n, 1, device=dev)], -1)
            w = st.get("spin", 0.0) * hz
        else:
            wd = env._n(n, 3)
            w = st.get("spin", 0.0) * wd / wd.norm(dim=-1, keepdim=True)
        rest = torch.zeros(n, dtype=torch.bool, device=dev)
    env.p_ball[ids], env.v_ball[ids], env.w_ball[ids], env.p_hold[ids] = p, v, w, p
    env.prev_vz[ids] = v[:, 2]
    env.rest_start[ids], env.landed[ids] = rest, rest
    env.launch_z[ids] = torch.where(rest, p[:, 2] - B.RADIUS, env.home_env[ids, 2])
    env.hold_left[ids] = float(st.get("held", 0.0))
    env.held_hidden[ids] = bool(st.get("hidden", False)) and st.get("held", 0.0) > 0


def restitution(e):
    def f(env, ids):
        env.ball_prm["e0"][ids] = e
    return f


def build_rows(env, launch="both"):
    """(name, fault, start, launch) rows; resting rows get a "+ launch" twin (or are replaced with launch="on")."""
    rows = [(name, None, st) for name, st in STARTS.items()]
    delay = int(round(0.040 / env.sub_dt))
    for name, (fault, st) in FAULTS.items():
        if name == "command delay 40 ms":
            fault = setv("delay_sub", delay)
        elif name == "ball restitution 0.75":
            fault = restitution(0.75)
        elif name.startswith("dead paddle e 0.65"):
            fault = restitution(0.65)
        elif name.startswith("dead paddle e 0.55"):
            fault = restitution(0.55)
        rows.append((name, fault, st))
    out = []
    for name, fault, st in rows:
        rest = st["kind"] == "rest"
        if launch != "on" or not rest:
            out.append((name, fault, st, False))
        if launch != "off" and rest:
            out.append((name + " + launch", fault, st, True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per", type=int, default=512)
    ap.add_argument("--difficulties", default="0,0.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="", help="row names separated by ; (quick checks)")
    ap.add_argument("--servo", default="12v_r8", help="servo set in bounce/servo.py the arm is simulated with")
    ap.add_argument("--launch", default="both", choices=["both", "on", "off"],
                    help="resting rows with the scripted launch (on), without (off), or both")
    ap.add_argument("--targets", default="", help="target apex heights in cm, e.g. 7,10,13,15 (default: random 7-15)")
    ap.add_argument("--launch_name", default="", help="named launch in bounce/launch.py NAMED (default: DEFAULTS)")
    ap.add_argument("--sched", default="", help="high-then-low target: start_cm,high_s,ramp_s (e.g. 16,1.5,1.0); "
                    "the --targets value is the steady target")
    ap.add_argument("--auto_level", action="store_true", help="the runner's --auto-level wrist trim (EnvCfg.auto_level)")
    ap.add_argument("--ci", action="store_true", help="also print the 95 %% interval of each survival rate")
    ap.add_argument("--al_gain", type=float, default=None, help="auto-level gain (EnvCfg.al_gain)")
    ap.add_argument("--al_alpha", type=float, default=None, help="auto-level EMA weight per detection")
    ap.add_argument("--camera", default="real", choices=["real", "trained"],
                    help="overhead camera at the calibrated real mount and latency, or the env's default nominal")
    a = ap.parse_args()
    agent, extra = PPO.load(a.checkpoint, a.device, load_optimizer=False)
    trained = {k: tuple(v) if isinstance(v, list) else v for k, v in (extra.get("config") or {}).get("env", {}).items()
               if k in EnvCfg.__dataclass_fields__}
    cfg = {k: trained[k] for k in INTERFACE if k in trained}    # same test setup for every checkpoint
    probe = BounceEnv(EnvCfg(**{**cfg, "num_envs": 1, "device": "cpu"}))
    rows = build_rows(probe, a.launch)
    if a.only:
        keep = {s.strip() for s in a.only.split(";")}
        rows = [r for r in rows if r[0] in keep]
    S, E = len(rows), a.per
    cfg.update(num_envs=S * E, device=a.device, seed=11, servo_variant=a.servo, act_delay_ms=DELAY[a.servo],
               side_cam_prob=0.0, ceiling_after_land=True, auto_level=a.auto_level)
    if a.al_gain is not None:
        cfg["al_gain"] = a.al_gain
    if a.al_alpha is not None:
        cfg["al_alpha"] = a.al_alpha
    if a.camera == "real":
        cfg.update(overhead_cam=REAL_CAM, cam_base_ms=REAL_CAM_BASE_MS)
    p_gain = int(round(float(cfg.get("p_gain", 16.0))))
    if a.launch_name or p_gain in LAUNCH.DEFAULTS:
        cfg["launch_max_action"] = max(1.0, float(getattr(LAUNCH.get(p_gain, a.launch_name), "gain", 1.0)))
    env = BounceEnv(EnvCfg(**cfg))
    env.held_on = True
    sched = [float(x) for x in a.sched.split(",")] if a.sched else None
    if sched:
        env.sched_any = True
        env.cfg.target_ramp_s = sched[2]
    seq = LAUNCH.actions(LAUNCH.get(p_gain, a.launch_name), 1, a.device)[:, 0] \
        if (a.launch_name or p_gain in LAUNCH.DEFAULTS) else None
    slot_of = torch.div(torch.arange(env.N, device=env.device), E, rounding_mode="floor")
    lmask = torch.tensor([r[3] for r in rows], device=env.device)[slot_of]
    if seq is None and bool(lmask.any()):
        raise SystemExit(f"no tuned launch for P {p_gain} in bounce/launch.py")
    target = [None]

    def hook(env_, ids):
        k = torch.div(ids, E, rounding_mode="floor")
        for s in torch.unique(k).tolist():
            sub = ids[k == s]
            name, fault, st, _ = rows[s]
            set_start(env_, sub, st)
            if fault is not None:
                fault(env_, sub)
            if target[0] is not None:
                env_.target_apex[sub] = target[0]
            if sched:
                env_.sched[sub] = True
                env_.tgt_steady[sub] = env_.target_apex[sub]
                env_.tgt_start[sub] = sched[0] / 100
                env_.t_high[sub] = sched[1]
                env_.target_apex[sub] = sched[0] / 100

    tr_env = (extra.get("config") or {}).get("env", {})
    print(f"{a.checkpoint}: trained servo {tr_env.get('servo_variant', '5v')}, P {cfg.get('p_gain')}; tested on servo "
          f"set {a.servo} ({DELAY[a.servo][0]:.0f} ms command delay), overhead camera only at the "
          f"{'calibrated real mount, 64 ms latency' if a.camera == 'real' else 'default mount'}; "
          f"{E} episodes of {env.cfg.episode_s:.0f} s per row", flush=True)
    targets = [float(x) / 100 for x in a.targets.split(",")] if a.targets else [None]
    for d in (float(x) for x in a.difficulties.split(",")):
        for tgt in targets:
            target[0] = tgt
            env.forced = {"difficulty": d, "rest_start": None, "side_cam": False, "post_reset": hook}
            obs, _ = env.reset_all()
            seq_all = None
            if seq is not None:                                   # the runner's --launch, steered per episode
                lp = LAUNCH.get(p_gain, a.launch_name)
                st = None
                if getattr(lp, "steer_p", 0) or getattr(lp, "steer_v", 0):
                    Jst = LAUNCH.steer_jacobian(ArmKinematics(env.model, 1, env.device), env.q_nom)
                    off = env.p_ball[:, :2] - env.home_env[:, :2] + 0.01 * env._n(env.N, 2)
                    vel = env.v_ball[:, :2] + 0.05 * env._n(env.N, 2)
                    st = LAUNCH.steer(lp, Jst.to(env.device), env.scale, off, vel).to(env.device)
                seq_all = LAUNCH.actions(lp, env.N, a.device, steer=st)
            done = torch.zeros(env.N, dtype=torch.bool, device=env.device)
            parts = []
            with torch.no_grad():
                for t in range(env.max_steps + 1):
                    act = agent.act_deterministic(obs["policy"])
                    if seq is not None and t < seq.shape[0]:
                        act = torch.where(lmask[:, None], seq_all[t], act)   # the runner's --launch
                    obs, _, term, trunc, _ = env.step(act)
                    new = (term | trunc) & ~done
                    st = env.pop_completed()
                    if st is not None and bool(new.any()):
                        keep = new[st["env_id"].long()]
                        parts.append({k: v[keep] for k, v in st.items()})
                    done |= term | trunc
                    if bool(done.all()):
                        break
            s = {k: torch.cat([x[k] for x in parts]) for k in parts[0]}
            slot = torch.div(s["env_id"].long(), E, rounding_mode="floor")
            print(f"\ndifficulty {d}" + ("" if tgt is None else f", target apex {100 * tgt:.0f} cm"))
            print(f"{'scenario':46s} {'survive':>7} {'bounces':>7} {'apex err':>8} {'1st bounce':>10} {'peak load':>9}  "
                  f"{'steady (share, s)':>17}  main failure")
            for i, (name, _, _, _) in enumerate(rows):
                m = slot == i
                sv, bo = s["survived"][m], s["bounces"][m]
                launched = s["launched"][m] > 0
                ae = 100 * float(s["apex_err"][m][bo > 0].mean()) if bool((bo > 0).any()) else float("nan")
                t1 = float(s["launch_t"][m][launched].median()) if bool(launched.any()) else float("nan")
                fails = s["reason"][m][sv < 0.5].long()
                mf = "-" if fails.numel() == 0 else \
                    f"{FAIL_REASONS[int(fails.mode().values)]} {100 * fails.numel() / max(1, sv.numel()):.0f}%"
                stt = s["steady_t"][m] if "steady_t" in s else torch.full_like(sv, float("nan"))
                got = ~torch.isnan(stt)
                st_med = float(stt[got].median()) if bool(got.any()) else float("nan")
                ci = f" +-{100 * 1.96 * math.sqrt(max(float(sv.mean()) * (1 - float(sv.mean())), 1e-4) / sv.numel()):.0f}" \
                    if a.ci else ""
                print(f"{name:46s} {100 * float(sv.mean()):6.0f}%{ci} {float(bo.mean()):7.1f} {ae:7.1f}cm {t1:9.2f}s "
                      f"{float(s['peak_load'][m].mean()):9.2f}  {100 * float(got.float().mean()):6.0f}% {st_med:6.2f}s  {mf}",
                      flush=True)


if __name__ == "__main__":
    main()
