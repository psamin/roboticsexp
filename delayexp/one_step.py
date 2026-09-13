"""Trace ONE control step of a trained policy on PandaPickCube and print the
real arrays at each stage:

  observation -> normalizer -> policy MLP -> tanh-Normal -> action
  -> actuator targets (ctrl) -> n_substeps physics steps -> reward

    uv run python -m delayexp.one_step --checkpoint runs/<run>/train/PandaPickCube-<time>/checkpoints

A teaching aid we added. Apart from delayexp.checkpoints (our workaround for a brax
checkpoint-config bug), everything it calls is upstream brax or MuJoCo Playground.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="policy steps to run before the traced step (t=0 has zeroed kinematics)")
    args = parser.parse_args()

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    from delayexp import checkpoints
    from delayexp.evaluate import ENV_NAME

    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    ckpt = checkpoints.resolve_checkpoint(args.checkpoint.resolve())
    params, network = checkpoints.load_ppo(ckpt)  # params: [RunningStatisticsState, policy, value]
    normalizer, policy_params = params[0], params[1]
    dist = network.parametric_action_distribution
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=True)

    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config_overrides={"impl": "jax"})
    m = env.mj_model

    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(args.seed))
    for _ in range(args.warmup_steps):
        state = step_fn(state, policy(state.obs, None)[0])

    obs = state.obs
    print(f"checkpoint {ckpt}")
    print(f"\n[t={args.warmup_steps}, sim time {float(state.data.time):.3f}s]")

    print(f"\n1. OBSERVATION  state.obs  shape {obs.shape} {obs.dtype}   (pick.py: PandaPickCube._get_obs)")
    print(f"   obs[46:49] box_pos - gripper_pos : {np.asarray(obs[46:49])}")
    print(f"   obs[49:52] target_pos - box_pos  : {np.asarray(obs[49:52])}")
    print(f"   obs[58:66] ctrl - robot qpos     : {np.asarray(obs[58:66])}")

    normalized = (obs - normalizer.mean) / normalizer.std
    print(f"\n2. NORMALIZER  (obs - mean) / std, mean/std shape {normalizer.mean.shape}   (brax running_statistics.normalize)")
    print(f"   normalized obs range [{float(normalized.min()):.2f}, {float(normalized.max()):.2f}]")

    print("\n3. POLICY MLP parameters   (brax networks.make_policy_network, swish activations)")
    for path, leaf in jax.tree_util.tree_leaves_with_path(policy_params):
        print(f"   {jax.tree_util.keystr(path):<45} {leaf.shape}")
    logits = network.policy_network.apply(normalizer, policy_params, obs)
    print(f"   output 'logits' shape {logits.shape} = 2 x action_size (first half loc, second half scale)")

    normal = dist.create_dist(logits)
    deterministic = dist.mode(logits)
    sampled = dist.sample(logits, jax.random.PRNGKey(123))
    print("\n4. ACTION DISTRIBUTION  tanh(Normal(loc, softplus(raw)+0.001))   (brax distribution.NormalTanhDistribution)")
    print(f"   loc              : {np.asarray(normal.loc)}")
    print(f"   scale            : {np.asarray(normal.scale)}")
    print(f"   eval action  = tanh(loc)            : {np.asarray(deterministic)}")
    print(f"   train action = tanh(loc + scale*eps) : {np.asarray(sampled)}   (one sample)")

    ctrl_before = state.data.ctrl
    lowers, uppers = m.actuator_ctrlrange.T
    ctrl_after = jp.clip(ctrl_before + deterministic * config.action_scale, lowers, uppers)
    print(f"\n5. ACTION -> ACTUATOR TARGETS  ctrl <- clip(ctrl + {config.action_scale} * action)   (pick.py: PandaPickCube.step)")
    print(f"   {'actuator':<10} {'joint':<14} {'qpos now':>9} {'ctrl before':>12} {'action':>8} {'ctrl after':>11}")
    for i in range(m.nu):
        joint = int(m.actuator_trnid[i, 0])
        qpos = float(state.data.qpos[m.jnt_qposadr[joint]])
        print(f"   {m.actuator(i).name:<10} {m.joint(joint).name:<14} {qpos:>9.4f} {float(ctrl_before[i]):>12.4f} "
              f"{float(deterministic[i]):>8.3f} {float(ctrl_after[i]):>11.4f}")

    next_state = step_fn(state, deterministic)
    box_id = m.body("box").id
    print(f"\n6. PHYSICS  {env.n_substeps} x mjx.step at sim_dt={env.sim_dt}s, ctrl held   (mjx_env.step)")
    print(f"   sim time {float(state.data.time):.3f}s -> {float(next_state.data.time):.3f}s  (one ctrl_dt={env.dt}s)")
    print(f"   ctrl after step equals computed targets: {bool(np.allclose(next_state.data.ctrl, ctrl_after))}")
    print(f"   arm qpos change : {np.asarray(next_state.data.qpos[:7] - state.data.qpos[:7])}")
    print(f"   box position    : {np.asarray(state.data.xpos[box_id])} -> {np.asarray(next_state.data.xpos[box_id])}")

    print("\n7. REWARD  sum of scale * term   (pick.py: PandaPickCube._get_reward)")
    for name, scale in config.reward_config.scales.items():
        print(f"   {name:<20} raw {float(next_state.metrics[name]):.3f} x {scale}")
    print(f"   reward {float(next_state.reward):.3f}   done {float(next_state.done)}")


if __name__ == "__main__":
    main()
