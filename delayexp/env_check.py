"""Milestone 1: load PandaPickCube, print its interfaces, run a short
random-action rollout, and render it headlessly to MP4.

    uv run python -m delayexp.env_check --out runs/env_check-local
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENV_NAME = "PandaPickCube"
# Upstream's success test for the sibling task PandaPickCubeCartesian
# (pick_cartesian.py: _get_success). PandaPickCube itself defines no success.
SUCCESS_THRESHOLD_M = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    # train-jax-ppo's --impl defaults to "jax" and overrides the env config's "warp".
    parser.add_argument("--impl", default="jax", choices=["jax", "warp"])
    parser.add_argument("--steps", type=int, default=75, help="control steps to roll out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Must be set before mujoco creates a GL context: EGL on headless Linux GPU
    # nodes, CGL on macOS.
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "cgl")

    import imageio_ffmpeg
    import jax
    import mediapy
    import mujoco
    import numpy as np
    from mujoco_playground import registry

    from delayexp import provenance

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}  impl={args.impl}")

    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config_overrides={"impl": args.impl})
    m = env.mj_model

    print(f"\n== Timing ==")
    print(f"sim_dt={env.sim_dt}s  ctrl_dt={env.dt}s  n_substeps={env.n_substeps}  "
          f"episode_length={config.episode_length} control steps "
          f"= {config.episode_length * env.dt:.2f}s  action_scale={config.action_scale}")

    print(f"\n== Actuators (action_size={env.action_size}, nu={m.nu}) ==")
    actuators = []
    for i in range(m.nu):
        target_id = int(m.actuator_trnid[i, 0])
        if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            target = m.joint(target_id).name
        else:
            target = m.tendon(target_id).name
        lo, hi = (float(v) for v in m.actuator_ctrlrange[i])
        row = {
            "index": i,
            "name": m.actuator(i).name,
            "target": target,
            "ctrlrange": [lo, hi],
            "gainprm0": float(m.actuator_gainprm[i, 0]),
            "biasprm012": [float(v) for v in m.actuator_biasprm[i, :3]],
        }
        actuators.append(row)
        print(f"  u[{i}] {row['name']:<10} -> {target:<14} ctrlrange=[{lo:+.4f}, {hi:+.4f}]  "
              f"gainprm[0]={row['gainprm0']:.1f}  biasprm[:3]={row['biasprm012']}")

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    t0 = time.perf_counter()
    state = reset_fn(jax.random.PRNGKey(args.seed))
    obs0 = np.asarray(state.obs)
    reset_first_call_s = time.perf_counter() - t0

    # Layout of PandaPickCube._get_obs, in concatenation order.
    segments = [
        ("data.qpos", m.nq),
        ("data.qvel", m.nv),
        ("gripper_pos", 3),
        ("gripper_mat[3:]", 6),
        ("box_mat[3:]", 6),
        ("box_pos - gripper_pos", 3),
        ("target_pos - box_pos", 3),
        ("target_mat[:6] - box_mat[:6]", 6),
        ("ctrl - robot_qpos[:nu]", m.nu),
    ]
    print(f"\n== Observation: shape {obs0.shape} dtype {obs0.dtype} "
          f"(nq={m.nq}, nv={m.nv}, joints: {[m.joint(j).name for j in range(m.njnt)]}) ==")
    start = 0
    obs_layout = []
    for name, size in segments:
        print(f"  obs[{start:>2}:{start + size:>2}] {name}")
        obs_layout.append({"name": name, "start": start, "stop": start + size})
        start += size
    if start != obs0.shape[-1]:
        sys.exit(f"Observation layout mismatch: segments sum to {start}, obs has {obs0.shape[-1]}")

    print(f"\nstate.info keys: {sorted(state.info)}")
    print(f"state.metrics keys: {sorted(state.metrics)}")

    actions = jax.random.uniform(
        jax.random.PRNGKey(args.seed + 1), (args.steps, env.action_size), minval=-1.0, maxval=1.0
    )
    states = [state]
    t0 = time.perf_counter()
    state = step_fn(state, actions[0])
    float(state.reward)  # block until the device finishes
    step_first_call_s = time.perf_counter() - t0
    states.append(state)

    t0 = time.perf_counter()
    for action in actions[1:]:
        state = step_fn(state, action)
        states.append(state)
    float(state.reward)
    step_steady_s = (time.perf_counter() - t0) / max(args.steps - 1, 1)

    box_id = m.body("box").id
    gripper_id = m.site("gripper").id
    # reset() builds data without a forward pass, so body/site positions are all
    # zero in the reset state. Measure distances from the first step on.
    reset_xpos_all_zero = bool(np.all(np.asarray(states[0].data.xpos) == 0))
    rewards = np.array([float(s.reward) for s in states[1:]])
    dones = np.array([float(s.done) for s in states[1:]])
    box_to_target = np.array([
        np.linalg.norm(np.asarray(s.info["target_pos"]) - np.asarray(s.data.xpos[box_id]))
        for s in states[1:]
    ])
    gripper_to_box = np.array([
        np.linalg.norm(np.asarray(s.data.xpos[box_id]) - np.asarray(s.data.site_xpos[gripper_id]))
        for s in states[1:]
    ])
    print(f"\n== Random-action rollout ({args.steps} control steps) ==")
    print(f"reset state data.xpos all zero: {reset_xpos_all_zero}"
          + ("  -> position-derived obs entries are 0 at t=0" if reset_xpos_all_zero else ""))
    print(f"reset first call (incl. JIT): {reset_first_call_s:.2f}s   "
          f"step first call (incl. JIT): {step_first_call_s:.2f}s   "
          f"steady step (unbatched): {step_steady_s * 1e3:.2f} ms")
    print(f"reward: sum={rewards.sum():.3f} min={rewards.min():.3f} max={rewards.max():.3f}  "
          f"any done={bool(dones.any())}")
    print(f"box->target distance: after step 1={box_to_target[0]:.3f} m  min={box_to_target.min():.3f} m  "
          f"(success would need < {SUCCESS_THRESHOLD_M} m)")
    print(f"gripper->box distance: after step 1={gripper_to_box[0]:.3f} m  min={gripper_to_box.min():.3f} m")

    render_every = 2
    t0 = time.perf_counter()
    frames = np.stack(env.render(states[::render_every], height=240, width=320))
    render_s = time.perf_counter() - t0
    fps = 1.0 / (env.dt * render_every)
    video_path = args.out / "random_rollout.mp4"
    mediapy.write_video(video_path, frames, fps=fps)
    mediapy.write_image(args.out / "first_frame.png", frames[0])
    frame_std = float(frames.std())
    print(f"\n== Render ==")
    print(f"{len(frames)} frames {frames.shape[1:]} in {render_s:.2f}s -> {video_path} at {fps:.0f} fps")
    print(f"pixel mean={frames.mean():.1f} std={frame_std:.1f} (a blank render has std near 0)")
    if frame_std < 1.0:
        sys.exit("Rendered frames look blank.")

    summary = {
        "env_name": ENV_NAME,
        "impl": args.impl,
        "seed": args.seed,
        "env_config": config.to_dict(),
        "n_substeps": env.n_substeps,
        "obs_shape": list(obs0.shape),
        "obs_layout": obs_layout,
        "actuators": actuators,
        "timing_s": {
            "reset_first_call": reset_first_call_s,
            "step_first_call": step_first_call_s,
            "step_steady_unbatched": step_steady_s,
            "render": render_s,
        },
        "rollout": {
            "steps": args.steps,
            "reward_sum": float(rewards.sum()),
            "any_done": bool(dones.any()),
            "min_box_to_target_m": float(box_to_target.min()),
        },
        "frames": {"count": len(frames), "shape": list(frames.shape[1:]), "std": frame_std},
        "provenance": provenance.collect(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
