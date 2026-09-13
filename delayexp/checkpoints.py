"""Load brax PPO checkpoints written by upstream train-jax-ppo."""

import json
import sys

from brax.training import networks as brax_networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks


def resolve_checkpoint(path):
    """Accept a step directory, or a checkpoints/ directory (use its latest step)."""
    if (path / "ppo_network_config.json").exists():
        return path
    steps = sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if not steps:
        sys.exit(f"No checkpoint step directories under {path}")
    return steps[-1]


def load_ppo(ckpt):
    """Return (params, ppo_network) for a checkpoint step directory.

    params is [RunningStatisticsState, policy params, value params].

    We rebuild the network from ppo_network_config.json ourselves because brax
    0.14.2's checkpoint.load_config (used by ppo checkpoint.load_policy) raises
    KeyError(None) for kernel-init kwargs saved as null, such as
    "mean_kernel_init_fn": null. brax's save() skips None for those keys, but
    load_config() does not.
    """
    params = ppo_checkpoint.load(ckpt)
    saved = json.loads((ckpt / "ppo_network_config.json").read_text())

    kwargs = {k: v for k, v in saved["network_factory_kwargs"].items() if v is not None}
    if "activation" in kwargs:
        kwargs["activation"] = brax_networks.ACTIVATION[kwargs["activation"]]
    for key in [k for k in kwargs if k.endswith("_init_fn")]:
        kwargs[key] = brax_networks.KERNEL_INITIALIZER[kwargs[key]]

    normalize = running_statistics.normalize if saved["normalize_observations"] else (lambda obs, _: obs)
    # The saved observation_size doesn't round-trip through JSON; the normalizer mean has the obs shape.
    observation_size = int(params[0].mean.shape[-1])
    network = ppo_networks.make_ppo_networks(
        observation_size, saved["action_size"], preprocess_observations_fn=normalize, **kwargs
    )
    return params, network
