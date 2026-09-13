"""Evaluate a trained PandaPickCube PPO checkpoint over many seeded episodes,
optionally with observation delay, and render a few episodes to MP4.

    uv run python -m delayexp.evaluate \
        --checkpoint runs/<run>/train/PandaPickCube-<time>/checkpoints \
        --out runs/eval-<name> --episodes 256 --delays none

Every condition in one call starts from the same initial states (the same reset
keys), and the policy is deterministic, so conditions differ only in what the
policy observes.
"""

import argparse
import functools
import json
import os
import sys
import types
from pathlib import Path

ENV_NAME = "PandaPickCube"
# Upstream's success test for the sibling task PandaPickCubeCartesian
# (pick_cartesian.py: _get_success). PandaPickCube itself defines no success.
SUCCESS_THRESHOLD_M = 0.05


def parse_delays(text):
    delays = []
    for item in text.split(","):
        item = item.strip()
        if item == "none":
            delays.append(None)
        elif item.isdigit():
            delays.append(int(item))
        else:
            raise argparse.ArgumentTypeError(f"delay must be 'none' or an int >= 0, got {item!r}")
    return delays


def condition_name(delay):
    return "nodelay" if delay is None else f"d{delay}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="seed for the episode reset keys")
    parser.add_argument("--delays", type=parse_delays, default=[None],
                        help="comma list of 'none' (no buffer) and delay steps, e.g. none,0,1,2")
    parser.add_argument("--videos", type=int, default=2, help="episodes to render per condition")
    parser.add_argument("--impl", default="jax", choices=["jax", "warp"])
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "cgl")

    import imageio_ffmpeg
    import jax
    import jax.numpy as jp
    import mediapy
    import numpy as np
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    from delayexp import checkpoints, obs_delay, provenance

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = checkpoints.resolve_checkpoint(args.checkpoint.resolve())
    params, network = checkpoints.load_ppo(ckpt)
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config_overrides={"impl": args.impl})
    episode_length = config.episode_length
    box_id = env.mj_model.body("box").id
    n_render = min(args.videos, args.episodes)
    print(f"checkpoint {ckpt}\njax backend={jax.default_backend()}  episodes={args.episodes}  "
          f"seed={args.seed}  ctrl_dt={env.dt}s  episode_length={episode_length}")

    reset_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    initial_state = jax.jit(jax.vmap(env.reset))(reset_keys)
    np.savez(args.out / "initial_states.npz",
             reset_keys=np.asarray(reset_keys),
             qpos=np.asarray(initial_state.data.qpos),
             target_pos=np.asarray(initial_state.info["target_pos"]))

    def render_fields(state):
        d = state.data
        return {"qpos": d.qpos[:n_render], "qvel": d.qvel[:n_render], "mocap_pos": d.mocap_pos[:n_render],
                "mocap_quat": d.mocap_quat[:n_render], "xfrc_applied": d.xfrc_applied[:n_render]}

    def make_rollout(delay):
        step_env = jax.vmap(env.step)
        policy_key = jax.random.PRNGKey(0)  # unused: the policy is deterministic

        def rollout(state):
            if delay is None:
                buffer = None
            else:
                buffer = jax.vmap(functools.partial(obs_delay.init_buffer, delay_steps=delay))(state.obs)
            alive = jp.ones(state.done.shape, dtype=bool)

            def step(carry, _):
                state, buffer, alive = carry
                if delay is None:
                    policy_obs = state.obs
                else:
                    buffer, policy_obs = jax.vmap(
                        functools.partial(obs_delay.push_and_read, delay_steps=delay))(buffer, state.obs)
                action, _ = policy(policy_obs, policy_key)
                next_state = step_env(state, action)
                out = {
                    "action": action,
                    "reward": next_state.reward,
                    "done": next_state.done,
                    "box_to_target": jp.linalg.norm(
                        next_state.info["target_pos"] - next_state.data.xpos[:, box_id], axis=-1),
                    "alive": alive,  # episode had not terminated before this step
                    "render": render_fields(next_state),
                }
                alive = alive & (next_state.done == 0)
                return (next_state, buffer, alive), out

            _, outs = jax.lax.scan(step, (state, buffer, alive), None, length=episode_length)
            return outs

        return jax.jit(rollout)

    conditions = []
    for delay in args.delays:
        name = condition_name(delay)
        print(f"\n== {name} ==")
        outs = jax.tree.map(np.asarray, make_rollout(delay)(initial_state))

        alive = outs["alive"]                                  # (T, N)
        dist = outs["box_to_target"]                           # (T, N), after step k
        success_steps = (dist < SUCCESS_THRESHOLD_M) & alive
        success = success_steps.any(axis=0)                    # (N,)
        first_success_step = np.where(success, success_steps.argmax(axis=0), -1)
        time_to_success_s = np.where(success, (first_success_step + 1) * env.dt, np.nan)
        terminated_early = ((outs["done"] > 0) & alive).any(axis=0)
        episode_return = (outs["reward"] * alive).sum(axis=0)

        np.savez(args.out / f"episodes-{name}.npz",
                 success=success, first_success_step=first_success_step, time_to_success_s=time_to_success_s,
                 terminated_early=terminated_early, episode_return=episode_return,
                 box_to_target=dist, alive=alive, actions=outs["action"])

        times = time_to_success_s[success]
        summary = {
            "condition": name,
            "delay_steps": delay,
            "delay_ms": None if delay is None else delay * env.dt * 1000.0,
            "episodes": args.episodes,
            "success_count": int(success.sum()),
            "success_rate": float(success.mean()),
            "time_to_success_s": None if times.size == 0 else {
                "mean": float(times.mean()), "median": float(np.median(times)),
                "min": float(times.min()), "max": float(times.max())},
            "terminated_early_count": int(terminated_early.sum()),
            "mean_return": float(episode_return.mean()),
        }
        conditions.append(summary)
        print(json.dumps(summary, indent=2))

        initial = render_fields(initial_state)
        for e in range(n_render):
            traj = [types.SimpleNamespace(data=types.SimpleNamespace(**{k: np.asarray(v[e]) for k, v in initial.items()}))]
            traj += [types.SimpleNamespace(data=types.SimpleNamespace(**{k: v[t, e] for k, v in outs["render"].items()}))
                     for t in range(episode_length)]
            frames = env.render(traj[::2], height=480, width=640)
            path = args.out / f"episode{e}-{name}.mp4"
            mediapy.write_video(path, frames, fps=1.0 / (env.dt * 2))
            print(f"wrote {path}  (success={bool(success[e])})")

    report = {
        "env_name": ENV_NAME,
        "checkpoint": str(ckpt),
        "impl": args.impl,
        "seed": args.seed,
        "episodes": args.episodes,
        "ctrl_dt_s": env.dt,
        "sim_dt_s": env.sim_dt,
        "episode_length": episode_length,
        "success_threshold_m": SUCCESS_THRESHOLD_M,
        "success_definition": "any control step, before early termination, with ||box_pos - target_pos|| < threshold",
        "conditions": conditions,
        "provenance": provenance.collect(),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
