"""Fit the simulator's servo model to a real-arm log from real/bounce_run.py: replay the logged goal positions through
the servo + gear play + gravity model (the same dynamics as BounceEnv.step) for many candidate parameter sets at once,
and keep the set whose simulated joint angles best match the measured ones.

  uv run python scripts/fit_servo.py real/logs/run-<time>.npz [more logs ...] --vin 11.7 [--device cuda]

Fitted: kt, R (strength fixed at 1, vin fixed at the measured supply), armature, the command delay and the firmware
speed limit. Everything else stays at the nominal of the chosen servo set. Several logs are fitted together (each
replayed at its own P gain, read from the checkpoint name); errors are reported per gain and joint, overall and on
the fast strokes (the top 20 % of measured joint speeds). One 2 s log: about a minute on CPU; minutes of logs: GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml                                                         # noqa: E402
from bounce.env import EnvCfg                                       # noqa: E402
from bounce.robot import ArmKinematics, build_model                 # noqa: E402
from bounce.servo import SERVO_VARIANTS, ServoBank                  # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_JM = yaml.safe_load((ROOT / "real" / "config.yaml").read_text()).get("joint_map") or {}


def deg_to_sim(pos):
    """LeRobot degrees -> simulator radians with the rig's joint map (as in real/rig.py, minus its camera imports)."""
    sign, off = np.array(_JM.get("sign", [1] * 5), float), np.array(_JM.get("offset_deg", [0] * 5), float)
    return np.deg2rad(sign * (np.array([pos[n] for n in ARM], float) - off))


# search box (log-uniform): name -> (low, high)
BOX = {"kt": (0.8, 4.0), "R": (1.5, 20.0), "armature": (0.01, 0.10), "delay_ms": (1.0, 40.0),
       "max_velocity": (2.5, 8.0)}


def load(path):
    """Every logged tick (holding, active and ramping back all have the commanded goal), the hold command, and the
    P gain the run used (from the checkpoint name)."""
    d = np.load(path, allow_pickle=True)
    act = d["state"] == "active"
    t = d["t"] - d["t"][0]
    q = np.stack([deg_to_sim(dict(zip(ARM, r))) for r in d["q_deg"]])
    goal = np.stack([deg_to_sim(dict(zip(ARM, r))) for r in d["goal_deg"]])
    scale = np.array(EnvCfg().action_scale)
    q_ref = np.median(goal[act] - scale * d["action"][act], 0)  # the hold command before the policy started
    p_gain = 32.0 if "p32" in str(d["checkpoint"]) else 16.0
    return t, q, goal, q_ref, p_gain


def replay(prm, t, goal, q0, q_ref, p_gain, base, sub_dt=0.002, settle_s=1.0, device="cpu"):
    """prm: name -> (N,) tensor. Returns simulated joint angles (N, K, J) at the log's tick times."""
    N, K, J = next(iter(prm.values())).shape[0], len(t), 5
    kin = ArmKinematics(build_model(EnvCfg().paddle), N, device)
    sv = ServoBank(N, J, p_gain, device, ranges=base)
    prm = {k: v.to(device) for k, v in prm.items()}
    for k, v in prm.items():
        if k in sv.p:
            sv.p[k] = v[:, None].expand(N, J).clone()
    delay = prm["delay_ms"] / 1000
    t_ = torch.tensor(t, dtype=torch.float32, device=device)
    G = torch.tensor(goal, dtype=torch.float32, device=device)
    qr = torch.tensor(q_ref, dtype=torch.float32, device=device)
    q = torch.tensor(q0, dtype=torch.float32, device=device).expand(N, J).clone()
    qd = torch.zeros(N, J, device=device)
    fk = kin.forward(q)
    tau_g, _ = kin.gravity_and_inertia(fk)
    sp = sv.p
    qm = q + tau_g / sp["gear_k"] + torch.sign(tau_g) * 0.5 * sp["backlash"]
    qdm = torch.zeros(N, J, device=device)
    sv.reset(torch.arange(N, device=device), qr.expand(N, J))
    out = torch.zeros(N, K, J, device=device)
    n_steps = int(np.ceil((settle_s + t[-1]) / sub_dt)) + 2
    k_next = 0
    for i in range(n_steps):
        s = -settle_s + i * sub_dt
        while k_next < K and s >= t[k_next] - 1e-9:
            out[:, k_next] = q
            k_next += 1
        idx = torch.searchsorted(t_, (s - delay).contiguous(), right=True) - 1          # latest goal that has arrived
        g = torch.where((idx >= 0)[:, None], G[idx.clamp(min=0)], qr.expand(N, J))
        tau_g, M = kin.gravity_and_inertia(fk)
        tau, _ = sv.torque(g, q, qd, sub_dt, dq_motor=qdm)
        gap = qm - q
        half = 0.5 * sp["backlash"]
        tc = sp["gear_k"] * torch.sign(gap) * (gap.abs() - half).clamp(min=0) + sp["gear_c"] * (qdm - qd) * (gap.abs() > half)
        tc = torch.where(gap > half, tc.clamp(min=0), torch.where(gap < -half, tc.clamp(max=0), torch.zeros_like(tc)))
        qdm = qdm + (tau + sv.friction(qdm, tc) - tc) / sp["armature"] * sub_dt
        qm = qm + qdm * sub_dt
        qd = qd + (tc - tau_g) / (M + 1e-3) * sub_dt
        qn = q + qd * sub_dt
        lim = (qn < kin.lower) | (qn > kin.upper)
        q = qn.clamp(kin.lower, kin.upper)
        qd = torch.where(lim, torch.zeros_like(qd), qd)
        fk = kin.forward(q)
    return out.cpu()


def rms_deg(sim, q):
    """Per-candidate overall rms (N,) and per-joint rms (N, J), in degrees, skipping the first tick."""
    e = np.degrees((sim - torch.tensor(q, dtype=torch.float32)[None]).numpy()[:, 1:])
    return np.sqrt((e ** 2).mean((1, 2))), np.sqrt((e ** 2).mean(1))


def sq_err(sim, q, fast):
    """Per-candidate squared errors (deg^2) summed over ticks: all (N, J) and fast-stroke ticks only (N, J)."""
    e2 = np.degrees((sim - torch.tensor(q, dtype=torch.float32)[None]).numpy()[:, 1:]) ** 2
    return e2.sum(1), (e2 * fast[None, 1:, :]).sum(1)


def sample(n, box, gen):
    return {k: torch.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * torch.rand(n, generator=gen)) for k, (lo, hi) in box.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--vin", type=float, default=11.7, help="measured supply voltage (Present_Voltage / 10)")
    ap.add_argument("--base", default="5v", help="servo set the unfitted parameters come from")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fix", action="append", default=[], help="name=value: hold one of the fitted parameters fixed")
    a = ap.parse_args()
    fixed = {k: float(v) for k, v in (f.split("=") for f in a.fix)}
    logs = []
    for path in a.logs:
        t, q, goal, q_ref, p_gain = load(path)
        speed = np.abs(np.gradient(q, t, axis=0))
        fast = (speed > np.quantile(speed, 0.8, axis=0)[None]).astype(float)    # fast strokes, per joint
        logs.append((Path(path).stem, t, q, goal, q_ref, p_gain, fast))
        print(f"{Path(path).stem}: P{p_gain:.0f}, {len(t)} ticks over {t[-1]:.1f} s; measured lag rms deg "
              f"{np.round(np.degrees(np.sqrt(((goal - q) ** 2).mean(0))), 1)}")
    base = SERVO_VARIANTS[a.base]
    gains = sorted({lg[5] for lg in logs})

    def evaluate(prm, servo_base):
        """Summed squared errors per gain: {gain: (all (N, J), fast (N, J), ticks, fast ticks (J,))}."""
        out = {g: [0.0, 0.0, 0, 0.0] for g in gains}
        for name, t, q, goal, q_ref, p_gain, fast in logs:
            e_all, e_fast = sq_err(replay(prm, t, goal, q[0], q_ref, p_gain, servo_base, device=a.device), q, fast)
            o = out[p_gain]
            o[0], o[1], o[2], o[3] = o[0] + e_all, o[1] + e_fast, o[2] + len(t) - 1, o[3] + fast[1:].sum(0)
        return out

    def report(label, out, i):
        for g in gains:
            e_all, e_fast, n_t, n_f = out[g]
            print(f"  {label} P{g:.0f}: rms {np.sqrt(e_all[i].sum() / (n_t * 5)):.2f} deg, per joint "
                  f"{np.round(np.sqrt(e_all[i] / n_t), 2)}; fast strokes per joint {np.round(np.sqrt(e_fast[i] / n_f), 2)}")

    def total(out):
        return np.sqrt(sum(o[0].sum(-1) for o in out.values()) / sum(o[2] * 5 for o in out.values()))

    nom = {k: torch.tensor([v[0]]) for k, v in SERVO_VARIANTS["12v"].items()}
    nom["delay_ms"] = torch.tensor([30.0])
    o0 = evaluate(nom, SERVO_VARIANTS["12v"])
    print(f"current 12 V model (round 7 nominal): rms {total(o0)[0]:.2f} deg")
    report("12v", o0, 0)
    gen = torch.Generator().manual_seed(0)
    box = {k: v for k, v in BOX.items() if k not in fixed}
    best = None
    for rnd in range(a.rounds):
        prm = sample(a.n, box, gen)
        prm["vin"], prm["strength"] = torch.full((a.n,), a.vin), torch.ones(a.n)
        prm.update({k: torch.full((a.n,), v) for k, v in fixed.items()})
        out = evaluate(prm, base)
        r = total(out)
        top = np.argsort(r)[: max(8, a.n // 50)]
        if best is None or r[top[0]] < best[0]:
            best = (r[top[0]], {k: float(v[top[0]]) for k, v in prm.items()}, out, int(top[0]))
        box = {k: (float(prm[k][top].min()) * 0.9, float(prm[k][top].max()) * 1.1) for k in box}
        print(f"round {rnd + 1}: best rms {r[top[0]]:.2f} deg; top-{len(top)} spread " +
              ", ".join(f"{k} {lo:.3g}-{hi:.3g}" for k, (lo, hi) in box.items()), flush=True)
    r, p, out, i = best
    p["stall_Nm"] = p["strength"] * p["kt"] * p["vin"] * 0.97 / p["R"]
    p["no_load_rad_s"] = p["vin"] * 0.97 / p["kt"]
    print(f"fitted: rms {r:.2f} deg")
    report("fit", out, i)
    print(json.dumps({k: round(v, 4) for k, v in p.items()}))


if __name__ == "__main__":
    main()
