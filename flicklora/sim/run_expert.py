"""Run the scripted expert: Phase B checks (few episodes, hero video, both flick directions) or Phase C collection.

  python run_expert.py --mode test    --ref REF --out OUT [--physics_hz 1200]
  python run_expert.py --mode collect --ref REF --out OUT --split train --n 100 --seed0 0
Raw episode layout (converted to LeRobot by train/to_lerobot.py):
  OUT/<split>/ep_<seed>/{camera1.mp4, camera2.mp4, lowdim.npz, meta.json}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["test", "collect"], required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=100, help="successful episodes wanted")
ap.add_argument("--max_attempts", type=int, default=0)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--physics_hz", type=float, default=1200)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--flick_sign", type=int, default=0, help="0 = auto-detect from a test flick")
ap.add_argument("--hero", default="960x540")
ap.add_argument("--robot", choices=["finger", "franka"], default="finger")
ap.add_argument("--env", default="warehouse", help="Franka surroundings: warehouse | shelves | digital_twin | table | none")
args = ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 3})
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert import FlickExpert
from flick_env import TASK, EpisodeConfig, FlickEnv

if args.robot == "franka":
    from franka_env import FrankaFlickEnv, sample_franka_cfg
    from franka_expert import FrankaFlickExpert
    sample_cfg = sample_franka_cfg
    make_expert = lambda env, rng, sign: FrankaFlickExpert(env, rng, flick_sign=sign)
    JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
else:
    sample_cfg = EpisodeConfig.sample
    make_expert = lambda env, rng, sign: FlickExpert(env, rng, flick_sign=sign)
    JOINT_NAMES = ["slide_x", "mcp", "pip", "dip"]
from metrics import nut_metrics


def log(*a):
    print("[run_expert]", *a, flush=True)


def write_mp4(path, frames, fps=30, quality=9):
    import imageio.v2 as iio
    with iio.get_writer(path, fps=fps, codec="libx264", quality=quality, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def run_episode(env, cfg, sign, record_hero=False, post_steps=24):
    obs = env.reset(cfg)
    ex = make_expert(env, np.random.default_rng(cfg.seed + 7), sign)
    rec = dict(cam1=[], cam2=[], hero=[], wide=[], state=[], action=[], t=[], q=[], w=[], phase=[])
    post = 0
    for k in range(cfg.max_steps):
        info = env.info()
        a = ex.act(obs["state"], info)
        rec["cam1"].append(obs["images"]["camera1"])
        rec["cam2"].append(obs["images"]["camera2"])
        if record_hero and obs["hero"] is not None:
            rec["hero"].append(obs["hero"])
        if record_hero and obs.get("wide") is not None:
            rec["wide"].append(obs["wide"])
        rec["state"].append(obs["state"])
        rec["action"].append(np.asarray(a, np.float32))
        rec["t"].append(info["t"])
        rec["q"].append(info["q"])
        rec["w"].append(info["w"])
        rec["phase"].append(ex.phase)
        obs, info = env.step(a)
        if info["sim_fail"]:
            break
        if info["detached"]:
            post += 1
            if post >= post_steps:
                break
    m = nut_metrics(env.trace, pitch=env.g.pitch)
    res = dict(seed=cfg.seed, success=bool(env.detached and not env.sim_fail), detached=env.detached,
               sim_fail=env.sim_fail, steps=len(rec["action"]), duration_s=len(rec["action"]) / env.ctrl_hz,
               expert_flicks=ex.flicks, expert_misses=ex.misses, detach_t=getattr(env, "detach_info", {}).get("t"), cfg=asdict(cfg),
               **{k: v for k, v in m.items() if k != "flicks"}, flicks=m.get("flicks", []))
    return res, rec


if args.robot == "franka":
    env = FrankaFlickEnv(app, args.ref, physics_hz=args.physics_hz, res=args.res,
                         hero=tuple(map(int, args.hero.split("x"))) if args.hero else None, wide=(args.mode == "test"),
                         env_kind=None if args.env == "none" else args.env)
else:
  env = FlickEnv(app, args.ref, physics_hz=args.physics_hz, res=args.res,
               hero=tuple(map(int, args.hero.split("x"))) if args.hero else None)
log(f"env ready ({args.robot}): corner band {env.r_band[0]*1000:.2f}-{env.r_band[1]*1000:.2f} mm, "
    f"pitch {env.g.pitch*1000:.3f} mm, Izz {env.izz:.3e}")
os.makedirs(args.out, exist_ok=True)

sign = args.flick_sign
if sign == 0:   # which sweep direction unscrews? one flick each way from the same start, pick the +travel one
    probe = {}
    for s in (-1, +1):
        cfg = sample_cfg(424242)
        cfg.max_steps = 45 if args.robot == "finger" else 100
        r, rec = run_episode(env, cfg, s, record_hero=(args.mode == "test"))
        probe[s] = r
        log(f"probe sign {s:+d}: axial {r['axial_mm']:+.3f} mm, turns {r['total_turns']:+.2f}, peak w {r['peak_w']:.1f}")
        if args.mode == "test" and rec["hero"]:
            write_mp4(os.path.join(args.out, f"probe_sign{s:+d}.mp4"), rec["hero"])
        if args.mode == "test" and rec["wide"]:
            write_mp4(os.path.join(args.out, f"probe_sign{s:+d}_wide.mp4"), rec["wide"])
    sign = max(probe, key=lambda s: probe[s]["axial_mm"])
    json.dump({str(k): v for k, v in probe.items()}, open(os.path.join(args.out, "probe.json"), "w"), indent=1, default=float)
log("flick sign:", sign)

if args.mode == "test":
    results = []
    for i in range(6):
        cfg = sample_cfg(1000 + i)
        t0 = time.time()
        r, rec = run_episode(env, cfg, sign, record_hero=True)
        r["wall_s"] = time.time() - t0
        results.append(r)
        log(json.dumps({k: v for k, v in r.items() if k not in ("cfg", "flicks")}, default=float))
        if rec["hero"]:
            write_mp4(os.path.join(args.out, f"expert_{cfg.seed}_hero.mp4"), rec["hero"])
        if rec["wide"]:
            write_mp4(os.path.join(args.out, f"expert_{cfg.seed}_wide.mp4"), rec["wide"])
        write_mp4(os.path.join(args.out, f"expert_{cfg.seed}_cam1.mp4"), rec["cam1"])
        write_mp4(os.path.join(args.out, f"expert_{cfg.seed}_cam2.mp4"), rec["cam2"])
        np.save(os.path.join(args.out, f"expert_{cfg.seed}_trace.npy"), np.asarray(env.trace, np.float32))
    json.dump(dict(flick_sign=sign, physics_hz=args.physics_hz, results=results),
              open(os.path.join(args.out, f"phaseB_{int(args.physics_hz)}hz.json"), "w"), indent=1, default=float)
    log("success", sum(r["success"] for r in results), "/", len(results))
else:
    d = os.path.join(args.out, args.split)
    os.makedirs(d, exist_ok=True)
    max_att = args.max_attempts or int(args.n * 1.6) + 5
    ok, att, summary = 0, 0, []
    while ok < args.n and att < max_att:
        seed = args.seed0 + att
        att += 1
        cfg = sample_cfg(seed, args.split)
        r, rec = run_episode(env, cfg, sign)
        summary.append({k: v for k, v in r.items() if k != "flicks"})
        log(f"[{args.split}] seed {seed}: success={r['success']} flicks={r['n_flicks']} steps={r['steps']} "
            f"sim_fail={r['sim_fail']}  ({ok + r['success']}/{args.n})")
        if not r["success"]:
            continue
        if any(f is None for f in rec["cam1"] + rec["cam2"]):
            log(f"seed {seed}: dropped, a camera never produced a frame")
            continue
        ok += 1
        e = os.path.join(d, f"ep_{seed:05d}")
        os.makedirs(e, exist_ok=True)
        write_mp4(os.path.join(e, "camera1.mp4"), rec["cam1"])
        write_mp4(os.path.join(e, "camera2.mp4"), rec["cam2"])
        np.savez(os.path.join(e, "lowdim.npz"), joint_names=np.asarray(JOINT_NAMES), state=np.asarray(rec["state"], np.float32),
                 action=np.asarray(rec["action"], np.float32), t=np.asarray(rec["t"]), nut_q=np.asarray(rec["q"]),
                 nut_w=np.asarray(rec["w"]), phase=np.asarray(rec["phase"]))
        json.dump(dict(r, task=TASK, flick_sign=sign, physics_hz=args.physics_hz, robot=args.robot), open(os.path.join(e, "meta.json"), "w"),
                  indent=1, default=float)
    json.dump(dict(split=args.split, wanted=args.n, successes=ok, attempts=att, flick_sign=sign,
                   empty_frames_filled=getattr(env, "empty_frames", 0), episodes=summary),
              open(os.path.join(d, "collect_summary.json"), "w"), indent=1, default=float)
    log(f"collected {ok}/{args.n} successes in {att} attempts")
app.close()
