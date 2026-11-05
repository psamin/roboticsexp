"""real/bounce_run.py in camera mode, closed loop, without the arm or a camera: the simulator's arm on a fake servo
bus (as in real/rehearse.py) and a fake camera watching a simulated ball that a scripted hand places, holds, drops
or takes away. Checks the runner's start / lost / re-arm logic in the cases the simulator cannot express.

  uv run python scripts/runner_scenarios.py policies/v12-base.pt [scenario ...] [-- runner flags, e.g. --launch]

Each scenario runs in real time (~6-9 s after a 3 s settle). Reported per scenario: when the policy switched on
relative to the ball becoming free, how many bounces the ball made while the policy ran, runner "lost" calls made
while the ball was still in play, re-arms, how far the paddle moved while the ball was still held, and the paddle's
pitch (lift + elbow + wrist flex vs the hold pose) at the end of each wait after a try.
"""
import builtins
import contextlib
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real"))
sys.path.insert(0, str(ROOT))
src = (ROOT / "real" / "rehearse.py").read_text()
exec(src[:src.index("ck = sys.argv[1]")])                 # SimBus and its imports (A, rig, BounceEnv, EnvCfg, torch)
import numpy as np                                         # noqa: E402
from bounce import ball as B                               # noqa: E402
from bounce.ppo import PPO                                 # noqa: E402

CAM = np.array([-0.17, 0.0, 0.94])                         # camera position for the depth-error direction
LAT = rig.CONFIG.get("camera_latency_s", 0.035)            # measured camera latency; the runner assumes it too
FLAGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


class DelayBus(SimBus):
    """SimBus with the fitted 30 ms command delay: each 20 ms tick runs 10 ms on the goal from two writes ago and
    10 ms on the previous one."""

    def __init__(self, env, start_deg):
        super().__init__(env, start_deg)
        self.queue = [self.goal.clone()] * 3

    def sync_write(self, name, values, normalize=True, num_retry=0):
        self.writes += 1
        g = self.queue[-1][0].numpy().copy()
        q = rig.deg_to_sim({**rig.sim_to_deg(g), **{k: v for k, v in values.items() if k in rig.ARM}})
        self.queue = self.queue[-2:] + [torch.tensor(q, dtype=torch.float32)[None]]
        self.goal = self.queue[0]
        self._advance(0.01)
        self.goal = self.queue[1]
        self._advance(0.01)


class World:
    """Ball + scripted hand + camera. Phases (time after Enter, action, params) are applied in order."""

    def __init__(self, env, phases, fps=30.0, lat_true=LAT, depth_sd=0.0, perp_sd=0.002, drop_p=0.0,
                 outlier_p=0.0, seed=0):
        self.env, self.phases = env, sorted(phases, key=lambda x: x[0])
        self.fps, self.lat_true, self.depth_sd, self.perp_sd = fps, lat_true, depth_sd, perp_sd
        self.drop_p, self.outlier_p = drop_p, outlier_p
        self.rng = np.random.default_rng(seed)
        self.prm = {k: torch.tensor([v[0]]) for k, v in B.BALL_RANGES.items()}
        self.g = torch.tensor([0.0, 0.0, -9.81])
        self.state = "absent"                              # absent / held / free / blob
        self.visible_held = True
        self.gap_until = -1.0
        self.p = self.v = self.w = None
        self.t, self.T0 = 0.0, None
        self.next_frame, self.pending, self.n = 0.0, [], 0
        fk = env.kin.forward(env.q)
        self.o_prev, self.R_prev = fk["paddle_o"].clone(), fk["paddle_R"].clone()
        self.home = fk["paddle_o"][0].numpy().copy()
        self.track = []
        self.events = {"free_t": None, "bounces": [], "dropped_t": None, "held_motion": 0.0, "removed": []}

    def _apply(self, t):
        while self.phases and self.phases[0][0] <= t:
            _, act, prm = self.phases.pop(0)
            if act == "hold":
                self.state, self.visible_held = "held", prm.get("visible", True)
                self.p = torch.tensor(self.home + np.array(prm["rel"]), dtype=torch.float32)[None]
                self.v, self.w = torch.zeros(1, 3), torch.zeros(1, 3)
            elif act == "rest":
                fk = self.env.kin.forward(self.env.q)
                o, R = fk["paddle_o"], fk["paddle_R"]
                dx, dy = prm.get("rel", (0.0, 0.0))
                self.p = o + B.RADIUS * R[..., 2] + dx * R[..., 0] + dy * R[..., 1]
                self.v, self.w = torch.zeros(1, 3), torch.zeros(1, 3)
                self.state = "free"
                self.events["free_t"] = t
            elif act == "free":
                self.state = "free"
                self.v = torch.tensor([prm.get("v", (0.0, 0.0, 0.0))], dtype=torch.float32)
                self.events["free_t"] = t
            elif act == "remove":
                self.state = "absent"
                self.events["removed"].append(t)
            elif act == "gap":
                self.gap_until = t + prm["s"]
            elif act == "blob":
                self.state = "blob"

    def advance(self, now):
        dt = 0.002
        fk = self.env.kin.forward(self.env.q)                  # paddle now; interpolated from the last call
        o1, R1 = fk["paddle_o"], fk["paddle_R"]
        if not hasattr(self, "o_last"):
            self.o_last, self.R_last, self.t_last = o1, R1, self.t
        span = max(now - self.t_last, 1e-6)
        while self.t + dt <= now:
            self.t += dt
            self._apply(self.t)
            a = min(1.0, (self.t - self.t_last) / span)
            o = self.o_last + a * (o1 - self.o_last)
            R = self.R_last + a * (R1 - self.R_last)
            if self.state == "held":
                self.events["held_motion"] = max(self.events["held_motion"],
                                                 float((o[0] - torch.tensor(self.home)).norm()))
            if self.state == "free":
                self.p, self.v = B.flight(self.p, self.v, self.w, self.prm["drag"], self.prm["magnus"], dt, self.g)
                self.p, self.v, self.w, touch, impact, speed, *_ = B.paddle_impact(
                    self.p, self.v, self.w, o, R, self.o_prev, self.R_prev, self.prm, 0.0785, 0.075, 2.2, dt)
                if bool(impact[0]) and float(speed[0]) > 0.25:
                    self.events["bounces"].append(self.t)
                if self.events["dropped_t"] is None and float(self.p[0, 2]) < float(o[0, 2]) - 0.04:
                    self.events["dropped_t"] = self.t
                if float(self.p[0, 2]) < B.RADIUS:                      # on the table
                    self.p[0, 2], self.v[:] = B.RADIUS, 0.0
            self.o_prev, self.R_prev = o, R
            if self.state in ("free", "held") and int(self.t / dt) % 5 == 0:
                lat = float((self.p[0, :2] - o[0, :2]).norm())
                self.track.append((self.t, lat, float(self.p[0, 2] - o[0, 2])))
            if self.t >= self.next_frame:
                self.next_frame += 1.0 / self.fps
                self._capture()

        self.o_last, self.R_last, self.t_last = o1, R1, now

    def in_play(self, t):
        """Ball over the blade and above it at time t (true state)."""
        near = [x for x in self.track if abs(x[0] - t) < 0.02]
        return bool(near) and near[-1][1] < 0.09 and near[-1][2] > -0.01

    def _capture(self):
        if self.state == "absent" or self.t < self.gap_until or self.rng.random() < self.drop_p:
            return
        if self.state == "held" and not self.visible_held:
            return
        if self.state == "blob":                                         # a static orange thing on the paddle
            pos = self.home + np.array([0.0, 0.0, 0.01])
        else:
            pos = self.p[0].numpy().astype(float).copy()
            ray = (pos - CAM) / np.linalg.norm(pos - CAM)
            pos = pos + self.rng.normal(0, self.perp_sd, 3) + ray * self.rng.normal(0, self.depth_sd)
            if self.rng.random() < self.outlier_p:
                d = self.rng.normal(0, 1, 3)
                d[2] = abs(d[2])
                pos = pos + d / np.linalg.norm(d) * self.rng.uniform(0.03, 0.08)
        self.pending.append((self.t + self.lat_true, self.t, pos))

    def frame(self, now):
        ready = [x for x in self.pending if x[0] <= now]
        if ready:
            self.pending = [x for x in self.pending if x[0] > now]
            self.n += 1
            return ready[-1]
        return None


SCENARIOS = {
    # name: (phases, world kwargs, seconds after Enter)
    "drop 15 cm, hand hides it": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}), (1.0, "free", {})], {}, 7),
    "held 1 s in view, then let go": ([(0.5, "hold", {"rel": (0, 0, 0.17)}), (1.5, "free", {})], {}, 7.5),
    "set down resting, 3 cm off": ([(0.5, "rest", {"rel": (0.03, 0.0)})], {}, 6.5),
    "dropped 12 cm beside": ([(0.5, "hold", {"rel": (0.0, 0.12, 0.17), "visible": False}), (1.0, "free", {})], {}, 6),
    "depth noise 4 cm": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}), (1.0, "free", {})],
                         {"depth_sd": 0.04}, 7),
    "0.7 s detection gap in play": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}), (1.0, "free", {}),
                                     (3.0, "gap", {"s": 0.7})], {}, 7),
    "15 fps camera": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}), (1.0, "free", {})], {"fps": 15.0}, 7),
    "true latency 100 ms (runner assumes config)": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}),
                                                (1.0, "free", {})], {"lat_true": 0.100}, 7),
    "hand false detections 10 %": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}), (1.0, "free", {})],
                                   {"outlier_p": 0.10}, 7),
    "ball taken away, dropped again 0.3 s after": ([(0.5, "hold", {"rel": (0, 0, 0.17), "visible": False}),
                                                    (1.0, "free", {}), (3.0, "remove", {}),       # runner gives up
                                                    (4.2, "hold", {"rel": (0, 0, 0.17), "visible": False}),   # ~4.0 s
                                                    (4.3, "free", {})], {}, 8.5),
    "static orange blob on the paddle, no ball": ([(0.5, "blob", {})], {}, 4),
}


def run(name, ck, env, hold):
    phases, kw, seconds = SCENARIOS[name]
    world = World(env, [tuple(p) for p in phases], **kw)
    T = {"enter": None}

    class Cam:
        def __init__(self, *a, **k):
            pass

        def get(self):
            now = time.perf_counter()
            if T["enter"] is None:
                return None, now, 0
            t = now - T["enter"]
            if t > seconds:
                raise KeyboardInterrupt
            world.advance(t)
            f = world.frame(t)
            if f is None:
                return (world.last if hasattr(world, "last") else None), getattr(world, "last_t", now), world.n
            avail, cap, pos = f
            world.last, world.last_t = ("det", pos), T["enter"] + avail - LAT   # runner: read time - latency
            return world.last, world.last_t, world.n

        def close(self):
            pass

    class Finder:
        def __init__(self, *a, **k):
            pass

        def find(self, frame):
            return None if frame is None else tuple(frame[1])

    class Model:
        def ball_position(self, x, y, z):
            return np.array([x, y, z])

        def paddle_roi(self, home, **kw):                  # the fake camera sees only the ball
            return None

    def fake_input(*a):
        if T["enter"] is None and "Enter to arm" in (a[0] if a else ""):
            T["enter"] = time.perf_counter()
        return ""

    import bounce_run as R
    R.Camera, R.BallFinder, R.CameraModel = Cam, Finder, Model
    R.camera_index = lambda cc: 0                          # no real camera in the harness
    R.lock_camera = lambda cam, cc: None                   # (and leave the real camera's settings alone)
    R.camera_view_check = lambda frames: (True, "harness: no camera view to check")
    builtins.input = fake_input
    sys.argv = ["bounce_run", "--checkpoint", ck, *FLAGS]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        R.main()
    text = out.getvalue()
    # times come from the log: first tick of each activation
    log = sorted((A.HERE / "logs").glob("run-*.npz"))[-1]
    d = np.load(log, allow_pickle=True)
    t_log, st = d["t"], d["state"]
    t_enter_log = t_log[0]
    act_starts = [float(t_log[i] - t_enter_log) for i in range(len(st)) if st[i] == "active" and (i == 0 or st[i - 1] != "active")]
    lost = [float(t_log[i] - t_enter_log) for i in range(len(st)) if st[i] == "returning" and (i == 0 or st[i - 1] != "returning")]
    ev = world.events
    active_t = [float(t_log[i] - t_enter_log) for i in range(len(st)) if st[i] == "active"]
    bounces = [b for b in ev["bounces"] if active_t and any(abs(b - a) < 0.03 for a in active_t)]
    false_lost = [x for x in lost if world.in_play(x)]
    q, pitch = d["q_deg"], []                               # paddle pitch at the end of each wait after a try
    for i in [i for i in range(1, len(st)) if st[i] == "armed" and st[i - 1] == "returning"]:
        j = next((k for k in range(i, len(st)) if st[k] != "armed"), len(st))
        seg = [k for k in range(i, j) if t_log[k] > t_log[j - 1] - 0.5]
        pitch.append((round(float(t_log[j - 1] - t_log[i]), 1),
                      round(float(sum(q[seg, c].mean() - hold[rig.ARM[c]] for c in (1, 2, 3))), 2)))   # lift+elbow+flex
    return {"scenario": name, "free_t": ev["free_t"], "policy_on": act_starts, "lost": lost, "false_lost": false_lost,
            "dropped_t": ev["dropped_t"], "bounces": len(bounces), "held_motion_cm": 100 * ev["held_motion"],
            "rearm_s_pitch_deg": pitch, "log": log.name}


def main():
    ck = sys.argv[1]
    names = [n for n in sys.argv[2:sys.argv.index("--") if "--" in sys.argv else None]] or list(SCENARIOS)
    ecfg = (PPO.load(ck, "cpu", load_optimizer=False)[1].get("config") or {}).get("env", {})
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in ecfg.items() if k in EnvCfg.__dataclass_fields__}
    env = BounceEnv(EnvCfg(**{**kw, "num_envs": 1, "device": "cpu", "servo_variant": "12v_r11"}))   # the real arm
    env.forced = {"difficulty": 0.0, "rest_start": True}
    env.reset_all()
    hold = rig.sim_to_deg(env.q_nom.numpy())
    tmp = Path(tempfile.mkdtemp())
    A.HOLD = tmp / "hold_pose.json"
    A.HOLD.write_text(json.dumps(hold))
    A.glide = lambda b, target, seconds=3.0, hz=50: (b.sync_write("Goal_Position", target), 0.0)[1]
    A.HERE = tmp                                           # run logs stay out of real/logs
    results = []
    for name in names:
        bus = DelayBus(env, dict(hold))
        A.open_bus = lambda bus=bus: bus
        r = run(name, ck, env, hold)
        results.append(r)
        fr = r["free_t"]
        on = ", ".join(f"{x - fr:+.2f}" if fr is not None else f"{x:.2f}" for x in r["policy_on"]) or "never"
        print(f"{name:44s} on at {on:18s} (s from ball free) | bounces {r['bounces']:2d} | lost calls {len(r['lost'])} "
              f"(in play {len(r['false_lost'])}) | ball fell {'-' if r['dropped_t'] is None else f'{r['dropped_t'] - (fr or 0):.1f} s'}"
              f" | paddle moved {r['held_motion_cm']:.1f} cm while held"
              + (f" | re-armed (s waited, pitch deg): {r['rearm_s_pitch_deg']}" if r["rearm_s_pitch_deg"] else ""),
              flush=True)
    out = ROOT / "outputs" / "runner_scenarios.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
