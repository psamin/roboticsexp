"""Rehearse real/bounce_run.py without the arm: a fake servo bus whose arm is the simulator's (STS3215 model, gear
play, gravity); each Goal_Position write advances it by one control period. Virtual-ball mode. Run this after any
change to the runner (it caught a runaway sag-trim loop before it reached the real arm).

  uv run python real/rehearse.py policies/<checkpoint>.pt [--virtual-start drop] [--virtual-e 0.7]
"""
import builtins, json, math, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real")); sys.path.insert(0, str(ROOT))
import numpy as np, torch
import arm as A, rig
from bounce.env import BounceEnv, EnvCfg

class SimBus:
    def __init__(self, env, start_deg):
        self.env, self.writes = env, 0
        e = env
        self.goal = torch.tensor(rig.deg_to_sim(start_deg), dtype=torch.float32)[None]
        e.q[:] = self.goal; e.qm[:] = self.goal; e.qd[:] = 0; e.qdm[:] = 0
        e.servo.reset(torch.tensor([0]), self.goal)
        self.reg = {}
    def _advance(self, seconds):
        e = self.env
        n = max(1, int(round(seconds / e.sub_dt)))
        fk = e.kin.forward(e.q)
        for _ in range(n):
            tau_g, M = e.kin.gravity_and_inertia(fk)
            sp = e.servo.p
            tau, _ = e.servo.torque(self.goal, e.q, e.qd, e.sub_dt, dq_motor=e.qdm)
            gap = e.qm - e.q; half = 0.5 * sp["backlash"]
            tc = sp["gear_k"] * torch.sign(gap) * (gap.abs() - half).clamp(min=0) + sp["gear_c"] * (e.qdm - e.qd) * (gap.abs() > half)
            tc = torch.where(gap > half, tc.clamp(min=0), torch.where(gap < -half, tc.clamp(max=0), torch.zeros_like(tc)))
            e.qdm = e.qdm + (tau + e.servo.friction(e.qdm, tc) - tc) / sp["armature"] * e.sub_dt
            e.qm = e.qm + e.qdm * e.sub_dt
            e.qd = e.qd + (tc - tau_g) / (M + 1e-3) * e.sub_dt
            e.q = (e.q + e.qd * e.sub_dt).clamp(e.kin.lower, e.kin.upper)
            fk = e.kin.forward(e.q)
    def sync_read(self, name, motors=None, normalize=True, num_retry=0):
        if name == "Present_Position":
            d = rig.sim_to_deg(self.env.q[0].numpy()); d["gripper"] = 10.0; return d
        return {m: 30 for m in (motors or A.NAMES)}
    def sync_write(self, name, values, normalize=True, num_retry=0):
        self.writes += 1
        g = self.goal[0].numpy().copy()
        q = rig.deg_to_sim({**rig.sim_to_deg(g), **{k: v for k, v in values.items() if k in rig.ARM}})
        self.goal = torch.tensor(q, dtype=torch.float32)[None]
        self._advance(1 / 50)
    def write(self, r, m, v, normalize=True, num_retry=0): self.reg[(r, m)] = v
    def read(self, r, m, normalize=True, num_retry=0): return self.reg.get((r, m), 0)
    def enable_torque(self, *a, **k): pass
    def disable_torque(self, *a, **k): pass
    def disconnect(self, disable_torque=True): print(f"[sim bus] {self.writes} goal writes")

ck = sys.argv[1]
from bounce.ppo import PPO
ecfg = (PPO.load(ck, "cpu", load_optimizer=False)[1].get("config") or {}).get("env", {})   # the policy's own sim settings
kw = {k: tuple(v) if isinstance(v, list) else v for k, v in ecfg.items() if k in EnvCfg.__dataclass_fields__}
if "--servo" in sys.argv:                                   # e.g. --servo 12v_r8: rehearse on another servo set
    i = sys.argv.index("--servo"); kw["servo_variant"] = sys.argv[i + 1]; del sys.argv[i:i + 2]
env = BounceEnv(EnvCfg(**{**kw, "num_envs": 1, "device": "cpu"})); env.forced = {"difficulty": 0.0, "rest_start": True}; env.reset_all()
hold = rig.sim_to_deg(env.q_nom.numpy())
tmp = Path(tempfile.mkdtemp()); A.HOLD = tmp / "hold_pose.json"; A.HOLD.write_text(json.dumps(hold))
rest = dict(hold); rest["gripper"] = 10.0
A.open_bus = lambda: SimBus(env, rest)
builtins.input = lambda *a: ""
import bounce_run
sys.argv = ["bounce_run", "--checkpoint", ck, "--virtual-ball", "--seconds", "8", "--vb-hz", "50", *sys.argv[2:]]   # the sim
# bus only moves the arm on goal writes, so extra paddle samples between steps would see it frozen   # extra options pass through
bounce_run.main()
