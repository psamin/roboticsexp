"""Evaluate a policy (served by train/policy_server.py) or the scripted expert on the held-out test configurations.

  python eval_policy.py --ref REF --out OUT --tag r32 --port 5555 --n 30 --seed0 90000
  python eval_policy.py --ref REF --out OUT --tag expert --expert --flick_sign -1
Same seeds -> identical test cases for every rank. Policy failures and simulation failures are counted separately.
Saves per-episode metrics JSON, nut traces, and the hero video of the best successful rollout.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed0", type=int, default=90000)
ap.add_argument("--expert", action="store_true")
ap.add_argument("--flick_sign", type=int, default=-1)
ap.add_argument("--physics_hz", type=float, default=1200)
ap.add_argument("--max_steps", type=int, default=0, help="0 = the robot's default (300 finger, 450 Franka)")
ap.add_argument("--hero", default="960x540")
ap.add_argument("--robot", choices=["finger", "franka"], default="finger")
ap.add_argument("--env", default="warehouse")
args = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 3})
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert import FlickExpert
from flick_env import EpisodeConfig, FlickEnv
from metrics import nut_metrics


class PolicyClient:
    def __init__(self, port):
        for _ in range(600):
            try:
                self.s = socket.create_connection(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(1)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, hdr, payload=b""):
        b = json.dumps(hdr).encode()
        self.s.sendall(struct.pack("!I", len(b)) + b + payload)
        n = struct.unpack("!I", self._recv(4))[0]
        return json.loads(self._recv(n))

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.s.recv(n - len(buf))
            if not c:
                raise ConnectionError
            buf += c
        return bytes(buf)

    def reset(self):
        self._send({"cmd": "reset"})

    def act(self, obs):
        arrs = [("observation.images.camera1", obs["images"]["camera1"]),
                ("observation.images.camera2", obs["images"]["camera2"]),
                ("observation.state", obs["state"].astype(np.float32))]
        hdr = {"cmd": "act", "arrays": [(k, list(v.shape), str(v.dtype)) for k, v in arrs]}
        r = self._send(hdr, b"".join(np.ascontiguousarray(v).tobytes() for _, v in arrs))
        return np.asarray(r["action"], np.float64), r["infer_ms"]

    def close(self):
        try:
            self._send({"cmd": "bye"})
        except Exception:
            pass


def write_mp4(path, frames, fps=30):
    import imageio.v2 as iio
    with iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


if args.robot == "franka":
    from franka_env import FrankaFlickEnv, sample_franka_cfg
    from franka_expert import FrankaFlickExpert
    env = FrankaFlickEnv(app, args.ref, physics_hz=args.physics_hz, hero=tuple(map(int, args.hero.split("x"))),
                         env_kind=None if args.env == "none" else args.env)
    sample_cfg, Expert = sample_franka_cfg, FrankaFlickExpert
else:
    env = FlickEnv(app, args.ref, physics_hz=args.physics_hz, hero=tuple(map(int, args.hero.split("x"))))
    sample_cfg, Expert = EpisodeConfig.sample, FlickExpert
client = None if args.expert else PolicyClient(args.port)
out = os.path.join(args.out, args.tag)
os.makedirs(out, exist_ok=True)
episodes, best = [], (None, -1e9)
for i in range(args.n):
    cfg = sample_cfg(args.seed0 + i, "test")
    cfg.max_steps = args.max_steps or cfg.max_steps
    obs = env.reset(cfg)
    if client:
        client.reset()
    ex = Expert(env, np.random.default_rng(cfg.seed + 7), flick_sign=args.flick_sign) if args.expert else None
    hero, infer, post, t0 = [], [], 0, time.time()
    for k in range(cfg.max_steps):
        if ex:
            a = ex.act(obs["state"], env.info())
        else:
            a, ms = client.act(obs)
            infer.append(ms)
        if obs["hero"] is not None:
            hero.append(obs["hero"])
        obs, info = env.step(a)
        if info["sim_fail"]:
            break
        if info["detached"]:
            post += 1
            if post >= 24:
                break
    m = nut_metrics(env.trace, pitch=env.g.pitch)
    fail_kind = None
    if env.sim_fail:
        fail_kind = "sim"            # physics blew up / thread constraint violated: not the policy's fault
    elif not env.detached:
        fail_kind = "policy"
    rec = dict(seed=cfg.seed, success=fail_kind is None, fail_kind=fail_kind, sim_fail=env.sim_fail,
               completion_s=env.detach_info["t"] if env.detached else None, steps=k + 1,
               infer_ms_mean=float(np.mean(infer)) if infer else None, wall_s=time.time() - t0, cfg=asdict(cfg),
               **{kk: v for kk, v in m.items() if kk != "flicks"}, flicks=m.get("flicks", []))
    episodes.append(rec)
    np.save(os.path.join(out, f"trace_{cfg.seed}.npy"), np.asarray(env.trace, np.float32)[::4])
    print(f"[{args.tag}] seed {cfg.seed}: success={rec['success']} fail={fail_kind} flicks={rec['n_flicks']} "
          f"turns={rec['total_turns']:.2f} axial={rec['axial_mm']:.2f}mm peak_w={rec['peak_w']:.0f} "
          f"t={rec['completion_s']}", flush=True)
    score = (1e3 if rec["success"] else 0) + rec["axial_mm"] - (rec["completion_s"] or 99)
    if score > best[1] and hero:
        best = (cfg.seed, score)
        write_mp4(os.path.join(out, "best_rollout.mp4"), hero)
    if i < 2 and hero:
        write_mp4(os.path.join(out, f"rollout_{cfg.seed}.mp4"), hero)

S = [e for e in episodes if e["success"]]
summary = dict(
    tag=args.tag, n=len(episodes), success_rate=len(S) / len(episodes),
    policy_failures=sum(e["fail_kind"] == "policy" for e in episodes),
    sim_failures=sum(e["fail_kind"] == "sim" for e in episodes),
    mean_turns_per_flick=float(np.mean([e["mean_dturns_per_flick"] for e in episodes if e["n_flicks"]] or [0])),
    mean_peak_w=float(np.mean([e["peak_w"] for e in episodes])),
    mean_peak_w_10ms=float(np.mean([e.get("peak_w_10ms", e["peak_w"]) for e in episodes])),
    mean_sustained_w=float(np.mean([e["sustained_w"] for e in episodes])),
    mean_axial_mm=float(np.mean([e["axial_mm"] for e in episodes])),
    mean_flicks_success=float(np.mean([e["n_flicks"] for e in S])) if S else None,
    mean_completion_s=float(np.mean([e["completion_s"] for e in S])) if S else None,
    infer_ms=float(np.mean([e["infer_ms_mean"] for e in episodes if e["infer_ms_mean"]] or [0])),
    best_seed=best[0],
)
json.dump(dict(summary=summary, episodes=episodes), open(os.path.join(out, "eval.json"), "w"), indent=1, default=float)
print("SUMMARY", json.dumps(summary, default=float), flush=True)
if client:
    client.close()
app.close()
