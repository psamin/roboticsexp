"""Bounce a ping-pong ball on the real SO-101 with a trained policy. You run it; nothing else may hold the port.

  uv run python real/bounce_run.py --checkpoint policies/best.pt --virtual-ball   # dry run: real arm, simulated ball
  uv run python real/bounce_run.py --checkpoint policies/best.pt                  # the real ball (overhead camera)

Before the first real-ball run: `real/sysid.py set-hold` (hold pose) and `real/calibrate_camera.py` (camera pose).

What happens:
 1. Start with the arm folded at rest. It writes the servo settings the policy was trained with (P gain, and
    Acceleration 0 = no ramp) and reads them back; any mismatch stops the run.
 2. It glides to the hold pose and trims the command until the measured joints sit at the hold pose (sag
    correction, like the simulator's hold command), so the paddle is level.
 3. Press Enter to arm. Camera mode: put, drop or bounce the ball onto the paddle; the policy starts as soon as two
    camera frames see it over the paddle, and after a drop it glides back to holding and waits for the next ball. Virtual mode: a simulated ball starts resting on the paddle.
 4. The policy runs at its training rate until the ball is lost, --seconds pass, or you press Ctrl-C.
 5. The arm glides back to the hold pose; press Enter (ball removed) and it folds back to rest and turns off.
Every tick is logged to real/logs/run-<time>.npz (joints, goals, actions, detections, timing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import arm as A                                                    # noqa: E402
from rig import BALL_RADIUS, CAMERA_CALIB as rig_calib, CONFIG, BallFinder, Camera, CameraModel, camera_index, lock_camera, _jmap as rig_jmap, deg_to_sim, sim_to_deg   # noqa: E402
from bounce import ball as B                                       # noqa: E402
from bounce.env import EnvCfg                                      # noqa: E402
from bounce import launch as LAUNCH                                # noqa: E402
from bounce.obs import RealObs                                     # noqa: E402
from bounce.ppo import PPO                                         # noqa: E402
from bounce.robot import ArmKinematics, build_model                # noqa: E402

TEMP_LIMIT = 60
START_GAP_S = 0.15        # policy on at 2 detections this recent over the paddle (resting, dropped or bounced in)
LOST_S = 1.0              # no detection this long: back to holding
START_LATERAL = 0.16      # m from the paddle centre: drops beside the paddle start the policy too
STATIC_S = 1.5            # a "ball" that has not moved this long while active is a static orange blob


class VirtualBall:
    """The simulator's ball, driven by the real paddle's measured pose; emits camera-like detections."""

    def __init__(self, fk, fps=30.0, latency=0.035, noise=(0.0015, 0.0015, 0.03), start="rest", e0=None, seed=None,
                 tilt_deg=(0.0, 0.0)):
        tx, ty = np.radians(tilt_deg)                             # a paddle mounted off level by this much (tests)
        cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
        self.Rt = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32) @ \
            torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        o, R = fk["paddle_o"], fk["paddle_R"] @ self.Rt
        self.p = o + BALL_RADIUS * R[..., 2]
        self.v, self.w = torch.zeros(1, 3), torch.zeros(1, 3)
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2 * np.pi)
        side = torch.tensor([[np.cos(a), np.sin(a), 0.0]], dtype=torch.float32)
        up = torch.tensor([[0.0, 0.0, 1.0]])
        if start == "drop":                                   # let go 15 cm above, near the centre
            self.p = self.p + 0.15 * up + 0.02 * side
        elif start == "beside":                               # let go 15 cm above, 10 cm off centre
            self.p = self.p + 0.15 * up + 0.10 * side
        elif start == "toss":                                 # tossed in from the side with spin
            self.p = self.p + 0.10 * up + 0.06 * side
            self.v = -0.3 * side
            self.w = torch.tensor(rng.uniform(-20, 20, (1, 3)), dtype=torch.float32)
        self.prm = {k: torch.tensor([v[0]]) for k, v in B.BALL_RANGES.items()}
        if e0 is not None:
            self.prm["e0"] = torch.tensor([float(e0)])        # a dead (hard) paddle returns less of each hit
        self.o_prev, self.R_prev = o, R
        self.g = torch.tensor([0.0, 0.0, -9.81])
        self.period, self.latency, self.noise = 1.0 / fps, latency, torch.tensor(noise)
        self.next_frame, self.pending = 0.0, []
        self.started = False

    def advance(self, fk, t0, t1, substeps=10):
        o1, R1 = fk["paddle_o"], fk["paddle_R"] @ self.Rt
        if not self.started:                                  # first sample: no paddle speed yet
            self.o_prev, self.R_prev, self.started = o1, R1, True
            return
        dt = (t1 - t0) / substeps
        for k in range(1, substeps + 1):
            a = k / substeps
            o = self.o_prev + a * (o1 - self.o_prev)
            R = self.R_prev + a * (R1 - self.R_prev)              # small motions: linear blend is fine
            o_p = self.o_prev + (a - 1 / substeps) * (o1 - self.o_prev)
            R_p = self.R_prev + (a - 1 / substeps) * (R1 - self.R_prev)
            self.p, self.v = B.flight(self.p, self.v, self.w, self.prm["drag"], self.prm["magnus"], dt, self.g)
            self.p, self.v, self.w, *_ = B.paddle_impact(self.p, self.v, self.w, o, R, o_p, R_p, self.prm,
                                                         0.0785, 0.075, 2.2, dt)
            t = t0 + k * dt
            if t >= self.next_frame:
                self.pending.append((t, (self.p[0] + self.noise * torch.randn(3)).numpy()))
                self.next_frame = t + self.period
        self.o_prev, self.R_prev = o1, R1

    def arrived(self, now):
        out = [d for d in self.pending if d[0] + self.latency <= now]
        self.pending = [d for d in self.pending if d[0] + self.latency > now]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--virtual-ball", action="store_true", help="real arm, simulated ball (safe dry run)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long bouncing (default 20 s with --virtual-ball; with the camera it keeps "
                         "running, re-arming after every drop, until Ctrl-C)")
    ap.add_argument("--target-cm", type=float, default=10.0, help="bounce height above the paddle")
    ap.add_argument("--target-start-cm", type=float, default=None,
                    help="start each attempt aiming this high (easier to get going), then ramp down to --target-cm")
    ap.add_argument("--target-hold-s", type=float, default=1.5, help="with --target-start-cm: seconds at the start height")
    ap.add_argument("--target-ramp-s", type=float, default=1.0, help="with --target-start-cm: seconds to ramp down")
    ap.add_argument("--virtual-start", default="rest", choices=["rest", "drop", "beside", "toss"],
                    help="with --virtual-ball: resting, dropped 15 cm, dropped 10 cm beside the paddle, or tossed in")
    ap.add_argument("--virtual-tilt-deg", type=float, nargs=2, default=(0.0, 0.0),
                    help="with --virtual-ball: the simulated paddle sits this far off level (about its x, y axes)")
    ap.add_argument("--auto-level", action="store_true",
                    help="while bouncing, slowly tilt the hold pose against the ball's average drift (at most 3 deg)")
    ap.add_argument("--virtual-e", type=float, default=None,
                    help="with --virtual-ball: ball-paddle restitution (default 0.85; lower = a deader paddle)")
    ap.add_argument("--launch-steps", type=int, default=None,
                    help="strike length in control steps (default: the tuned value; it adapts during the run)")
    ap.add_argument("--launch-lean", type=float, default=0.0,
                    help="tilt the paddle face this many degrees forward (away from the handle) during the strike; "
                         "use it when launches throw the ball back toward the handle")
    ap.add_argument("--record", action="store_true",
                    help="save the overhead camera video (detections circled, state stamped) next to the log")
    ap.add_argument("--launch", action="store_true",
                    help="start a resting ball with the scripted dip-and-strike (bounce/launch.py), then the policy")
    ap.add_argument("--vb-hz", type=float, default=200.0,
                    help="with --virtual-ball: how often the simulated ball samples the paddle (extra joint reads "
                         "between control steps; 50 = once per step). Fast strokes need more than 50.")
    a = ap.parse_args()
    limit = a.seconds or (20.0 if a.virtual_ball else float("inf"))
    agent, extra = PPO.load(a.checkpoint, "cpu", load_optimizer=False)
    ecfg = (extra.get("config") or {}).get("env", {})
    cfg = EnvCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ecfg.items() if k in EnvCfg.__dataclass_fields__})
    hz, p_gain, scale = cfg.ctrl_hz, int(cfg.p_gain), np.array(cfg.action_scale)
    launch = None
    if a.launch:
        if p_gain not in LAUNCH.DEFAULTS:
            raise SystemExit(f"no tuned launch for P gain {p_gain} in bounce/launch.py (scripts/tune_launch.py)")
        lp = LAUNCH.Launch(**LAUNCH.DEFAULTS[p_gain].as_dict())
        if a.launch_steps:
            lp.strike_steps = a.launch_steps
        if a.launch_lean:                                        # + wrist flex tilts the face normal forward (+x)
            s0, s1, s2 = lp.strike
            lp.strike = (s0, s1, s2 + np.radians(a.launch_lean) / scale[3])
            print(f"launch strike leans {a.launch_lean:+.1f} deg forward")
        launch = list(LAUNCH.actions(lp)[:, 0].numpy())         # the same actions the simulator ran
    launch_q, t_launched, relaunches = [], None, 0
    kin = ArmKinematics(build_model(), 1)
    lower, upper = kin.lower.numpy() + 0.02, kin.upper.numpy() - 0.02
    hold = json.loads(A.HOLD.read_text())
    print(f"policy {a.checkpoint}: {hz:.0f} Hz, P gain {p_gain}, history {cfg.history}, detections {cfg.cam_k}")
    cam = finder = model = vb = writer = None
    if not a.virtual_ball:
        cc = CONFIG["cameras"]["overhead"]
        cam = Camera(camera_index(cc), cc["width"], cc["height"], cc["fps"], CONFIG.get("camera_latency_s", 0.035))
        lock_camera(cam, cc)
        if a.record:
            import cv2
            video = A.HERE / "logs" / f"run-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            (A.HERE / "logs").mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), cc["fps"], (cc["width"], cc["height"]))
            print(f"recording to {video}")
        model = CameraModel()
        made_with = json.loads(rig_calib.read_text()).get("joint_map_offset_deg")
        if made_with is None or np.abs(np.array(made_with) - rig_jmap()[1]).max() > 0.05:
            print("WARNING: the camera calibration predates the current wrist offsets (re-levelled since?): ball positions "
                  "may be off by a few cm. Rerun `uv run python real/calibrate_camera.py` with the ball taped on.")
        px = getattr(model, "ball_px", None)
        size = {"min_r": 0.5 * px, "max_r": 2.5 * px} if px else {}
        finder = BallFinder(CONFIG.get("ball_colour", "orange"), **size)   # rejects the blade and other large blobs
    b = A.open_bus()
    rest = A.hold_here(b)
    log = {k: [] for k in ("t", "q_deg", "goal_deg", "action", "state", "loop_ms")}
    dets_log, ball_log = [], []
    t_start = time.perf_counter()
    try:
        got = A.set_registers(b, {"P_Coefficient": p_gain, **A.BOUNCE_REGISTERS})
        bad = {r: v for r, v in got.items() if any(x != {"P_Coefficient": p_gain, **A.BOUNCE_REGISTERS}[r]
                                                   for x in v.values())}
        if bad:
            raise RuntimeError(f"servo registers did not take: {bad}")
        print(f"registers set and read back: {got}")
        print(f"to the hold pose ({A.glide(b, hold, 3.0):.1f} deg)")
        cmd = dict(hold)                                          # trim out the sag so the measured pose is the hold
        for _ in range(int(3.0 * 50)):                            # slow integral (3/s): the servos lag ~0.1 s
            pos = A.read_pos(b)
            for n in A.ARM:
                cmd[n] = float(np.clip(cmd[n] + 0.06 * (hold[n] - pos[n]), hold[n] - 12, hold[n] + 12))
            b.sync_write("Goal_Position", cmd)
            time.sleep(0.02)
        pos = A.read_pos(b)
        q_ref = deg_to_sim(cmd)
        q_meas = deg_to_sim(pos)
        fk0 = kin.forward(torch.tensor(q_meas, dtype=torch.float32)[None])
        home = fk0["paddle_o"][0]
        if finder is not None:                                   # only look around the paddle (not at the arm)
            finder.roi = model.paddle_roi(home.numpy(), size=(cc["width"], cc["height"]))
            print(f"ball search window (pixels): {finder.roi}")
        tilt = np.degrees(np.arccos(np.clip(float(fk0["paddle_R"][0, 2, 2]), -1.0, 1.0)))
        print(f"holding: sag trimmed {max(abs(cmd[n] - hold[n]) for n in A.ARM):.1f} deg; model says the paddle is "
              f"tilted {tilt:.1f} deg (level by hand = joint map OK if this is small)")
        trim, drift, t_trim = np.zeros(5), np.zeros(2), 0.0      # --auto-level: wrist trim and the ball's mean drift
        t_det = -1.0                                              # capture time of the last detection folded in

        def new_obs():
            return RealObs(kin, q_ref + trim, home, a.target_cm / 100, cfg.history, cfg.cam_k, side_camera=False)

        def normal_xy(qq):
            return kin.forward(torch.tensor(qq, dtype=torch.float32)[None])["paddle_R"][0, :2, 2].numpy().astype(float)
        ro = new_obs()
        ro.joints(q_meas)
        input("Enter to arm" + (" (simulated ball)" if a.virtual_ball else "; then put the ball on the paddle whenever "
              "you like. It keeps running and re-arms after every drop; Ctrl-C to stop") + " ... ")
        if a.virtual_ball:
            vb = VirtualBall(fk0, latency=CONFIG.get("camera_latency_s", 0.035), start=a.virtual_start, e0=a.virtual_e,
                             tilt_deg=a.virtual_tilt_deg)
            state = "active"
            if launch and a.virtual_start == "rest":
                launch_q = list(launch)
        else:
            state, rounds, ignore = "armed", 0, None
        period = 1.0 / hz
        t_active = t_next = time.perf_counter()
        last_frame = -1
        hot_n = 0
        vb_t = t_next
        while True:
            t_next += period
            now = time.perf_counter()
            tick0 = now
            pos = A.read_pos(b)
            q = deg_to_sim(pos)
            ro.joints(q)
            fk = kin.forward(torch.tensor(q, dtype=torch.float32)[None])
            if vb is not None:
                vb.advance(fk, vb_t - t_active, now - t_active)
                vb_t = now
                for tc, p in vb.arrived(now - t_active):
                    ro.ball(t_active + tc, p)
                    dets_log.append((t_active + tc - t_start, *p))
                ball_log.append((now - t_start, *vb.p[0].numpy()))
                if float(vb.p[0, 2]) < float(fk["paddle_o"][0, 2]) - 0.05:
                    print("simulated ball dropped")
                    break
            else:
                frame, tc, n = cam.get()
                if frame is not None and n != last_frame:
                    last_frame = n
                    d = finder.find(frame)
                    if d is not None:
                        p = model.ball_position(*d)
                        ro.ball(tc, p)
                        dets_log.append((tc - t_start, *p))
                    if writer is not None:
                        img = frame.copy()
                        if finder.roi is not None:
                            cv2.rectangle(img, finder.roi[:2], finder.roi[2:], (255, 160, 0), 1)
                        if d is not None:
                            cv2.circle(img, (int(d[0]), int(d[1])), int(d[2]), (0, 0, 255), 2)
                        cv2.putText(img, f"{tc - t_start:6.2f}s {state}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2)
                        writer.write(img)
                recent = [x for x in ro.det if now - x[0] < LOST_S]
                if state == "armed":
                    rel = [(x[1] - home).numpy() for x in ro.det if now - x[0] < START_GAP_S][-2:]
                    over = len(rel) == 2 and all(np.hypot(r[0], r[1]) < START_LATERAL and -0.05 < r[2] < 0.5 for r in rel)
                    if over and ignore is not None and np.hypot(*(rel[-1][:2] - ignore[:2])) < 0.02:
                        over = False                                   # the static blob found earlier
                    if over:
                        old, ro = ro, new_obs()               # fresh history; keep the detections already seen
                        ro.joints(q)
                        for tc_, p_ in old.det:
                            ro.ball(tc_, p_)
                        state, t_active, rounds, out_n, last_det = "active", now, rounds + 1, 0, None
                        resting = all(abs(r[2]) < 0.04 for r in rel) and np.hypot(*(rel[1][:2] - rel[0][:2])) < 0.015
                        launch_q = list(launch) if (launch and resting) else []
                        t_launched, relaunches = (now if launch_q else None), 0
                        print(f"ball on the paddle: policy on (#{rounds})" + (", launching it" if launch_q else ""))
                elif state == "active":
                    if recent and recent[-1][0] != last_det:            # 2 new detections in a row off the paddle
                        last_det = recent[-1][0]
                        off = (recent[-1][1] - home).numpy()
                        below = float(recent[-1][1][2] - fk["paddle_o"][0, 2]) < -0.08   # under the paddle as it is now
                        out_n = out_n + 1 if (np.hypot(off[0], off[1]) > cfg.max_lateral or below) else 0
                    if launch and t_launched is not None and now - t_launched > 0.9:   # judge the launch's first bounce
                        hs = [float(x[1][2] - home[2]) - BALL_RADIUS for x in ro.det if x[0] > t_launched]
                        top = max(hs) if hs else 0.0
                        step = 1 if top < 0.06 else -1 if top > 0.15 else 0
                        lp.strike_steps = int(np.clip(lp.strike_steps + step, 2, 8))
                        launch = list(LAUNCH.actions(lp)[:, 0].numpy())
                        print(f"  launch reached {100 * top:.0f} cm -> strike {lp.strike_steps} steps next time")
                        t_launched = None
                    span = [x[1].numpy() for x in ro.det if now - x[0] < STATIC_S]
                    still = len(span) >= 10 and np.ptp(np.array(span)[:, :2], 0).max() < 0.01 and \
                        np.ptp(np.array(span)[:, 2]) < 0.04
                    on_paddle = still and abs(float(span[-1][2] - fk["paddle_o"][0, 2]) - BALL_RADIUS) < 0.04 and \
                        np.hypot(*(span[-1][:2] - home[:2].numpy())) < 0.08
                    if launch and on_paddle and not launch_q and relaunches < 3 and now - t_active > 0.8 and \
                            all(now - x[0] < 0.8 for x in ro.det[-10:]):   # ball left sitting on the paddle: pop it again
                        launch_q, t_launched, relaunches = list(launch), now, relaunches + 1
                        print(f"  ball resting on the paddle: launching again ({relaunches})")
                    static = now - t_active > STATIC_S and still and \
                        (not on_paddle or (now - t_active > 4.0 and (not launch or relaunches >= 3)))   # gave up on it
                    if not recent or out_n >= 2 or static:
                        why = ("no detection" if not recent else "ball off the paddle" if out_n >= 2 else
                               "detection has not moved for 1.5 s: not a ball? ignoring that spot")
                        print(f"ball lost after {now - t_active:.1f} s ({why}): back to holding; put the ball back on")
                        if static:
                            ignore = span[-1] - home.numpy()               # relative to the paddle, like `rel`
                        state, t_lost, goal_lost = "returning", now, np.array(goal, float)
                        if a.auto_level:
                            print(f"  auto-level trim: wrist flex {np.degrees(trim[3]):+.2f}, roll {np.degrees(trim[4]):+.2f} deg")
            if state == "active":
                if a.target_start_cm is not None:                 # high first, then the steady height
                    f = float(np.clip((now - t_active - a.target_hold_s) / max(a.target_ramp_s, 1e-3), 0.0, 1.0))
                    ro.target = (a.target_start_cm + f * (a.target_cm - a.target_start_cm)) / 100
                act = launch_q.pop(0) if launch_q else agent.act_deterministic(ro.vector(now))[0].numpy()
                if a.auto_level and ro.det and ro.det[-1][0] != t_det:   # once per new detection (frames arrive
                    t_det = ro.det[-1][0]                                # ~62 ms after capture)
                    off = (ro.det[-1][1][:2] - fk["paddle_o"][0, :2]).numpy()
                    if np.hypot(*off) < 0.08:                          # where on the blade the ball sits, smoothed ~1 s
                        drift = 0.97 * drift + 0.03 * off
                if a.auto_level and now - t_trim > 0.1 and np.hypot(*drift) > 0.004:
                    t_trim, qe = now, q_ref + trim                     # a tilted face sends the ball downhill: tilt back
                    n0 = normal_xy(qe)
                    J = np.stack([(normal_xy(qe + 1e-3 * np.eye(5)[j]) - n0) / 1e-3 for j in (3, 4)], 1)
                    trim[3:5] = np.clip(trim[3:5] + np.linalg.solve(J, -0.5 * drift * 0.1), -0.052, 0.052)
                    ro.q_ref = torch.tensor(q_ref + trim, dtype=torch.float32)[None]
                goal = np.clip(q_ref + trim + scale * act, lower, upper)
                b.sync_write("Goal_Position", sim_to_deg(goal))
                ro.acted(act)
            elif state == "returning":                              # ramp back to the hold command (0.4 s)
                s = min(1.0, (now - t_lost) / 0.4)
                act, goal = np.zeros(5), goal_lost + s * (q_ref + trim - goal_lost)
                b.sync_write("Goal_Position", sim_to_deg(goal))
                if s >= 1.0:
                    state = "armed"
            else:
                act, goal = np.zeros(5), q_ref
            log["t"].append(now - t_start)
            log["q_deg"].append([pos[n] for n in A.ARM])
            log["goal_deg"].append(list(sim_to_deg(goal).values()))
            log["action"].append(act)
            log["state"].append(state)
            log["loop_ms"].append(1000 * (time.perf_counter() - tick0))
            if state == "active" and now - t_active > limit:
                print(f"{limit:.0f} s done")
                break
            if len(log["t"]) % int(hz) == 0:
                temp = b.sync_read("Present_Temperature", A.ARM, normalize=False)
                hot_n = hot_n + 1 if max(temp.values()) > TEMP_LIMIT else 0   # a lone bad read (e.g. 111 C) is a glitch
                if hot_n >= 2:
                    raise RuntimeError(f"servo temperature {temp}")
            if vb is not None and a.vb_hz > hz:                   # extra paddle samples for the simulated ball
                while t_next - time.perf_counter() > 1.0 / a.vb_hz + 0.002:
                    time.sleep(max(0.0, 1.0 / a.vb_hz - 0.002))
                    tn = time.perf_counter()
                    q_s = torch.tensor(deg_to_sim(A.read_pos(b)), dtype=torch.float32)[None]
                    vb.advance(kin.forward(q_s), vb_t - t_active, tn - t_active)
                    vb_t = tn
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = time.perf_counter()                      # overran: do not try to catch up
    except (KeyboardInterrupt, RuntimeError) as e:
        print("stopping:", e)
    finally:
        try:
            if a.auto_level:
                print(f"auto-level trim at the end: wrist flex {np.degrees(trim[3]):+.2f}, roll {np.degrees(trim[4]):+.2f} deg")
            print(f"back to the hold pose ({A.glide(b, hold, 1.5):.1f} deg)")
            if not a.virtual_ball:
                A.pause("Take the ball off the paddle, press Enter (the arm then folds back and turns off)...")
            A.set_registers(b, A.LEROBOT_DEFAULTS)
            A.glide(b, {n: rest[n] for n in A.ARM}, 3.0)
        finally:
            A.park(b)
            if cam is not None:
                cam.close()
            if writer is not None:
                writer.release()
        (A.HERE / "logs").mkdir(exist_ok=True)
        out = A.HERE / "logs" / f"run-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        lm = np.array(log["loop_ms"]) if log["loop_ms"] else np.zeros(1)
        np.savez_compressed(out, **{k: np.array(v) for k, v in log.items()}, detections=np.array(dets_log),
                            virtual_ball=np.array(ball_log), checkpoint=a.checkpoint, hz=hz)
        print(f"log: {out}  ({len(log['t'])} ticks, loop {np.median(lm):.1f} ms median, {np.percentile(lm, 99):.1f} ms "
              f"p99, {len(dets_log)} detections)")


if __name__ == "__main__":
    main()
