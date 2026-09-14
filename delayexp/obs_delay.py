"""Evaluation-only observation delay. Not implemented yet: both functions raise
NotImplementedError, so delayexp.evaluate supports only --delays none for now.

Goal: at control step t, the policy acts on the observation from step t - d,
while the simulator keeps advancing every control step. Nothing sleeps and
physics never pauses; the policy just sees a stale observation.

Conventions the evaluator (delayexp/evaluate.py) relies on:
  * Each function handles ONE environment. obs has shape (66,) for
    PandaPickCube. The evaluator vmaps over environments.
  * delay_steps (d) is a Python int >= 0, fixed for a whole evaluation. Static
    shapes are required because this runs inside jax.jit and jax.lax.scan.
  * Step t=0 is the observation returned by env.reset. push_and_read is called
    once per control step, before the policy runs, with that step's observation.
  * One control step is env.dt = 0.02 s, so d steps means 20*d ms of staleness.
  * With d = 0, push_and_read must return `obs` unchanged, so zero delay
    reproduces the baseline exactly.
"""

import jax


def init_buffer(first_obs: jax.Array, delay_steps: int) -> jax.Array:
    """Create the buffer at the start of an episode.

    Args:
      first_obs: the observation from env.reset, shape (obs_dim,).
      delay_steps: d >= 0.

    Returns:
      A buffer array able to serve the observation from d steps ago.
    """
    # TODO: decide what the buffer holds before d real observations exist.
    raise NotImplementedError


def push_and_read(
    buffer: jax.Array, obs: jax.Array, delay_steps: int
) -> tuple[jax.Array, jax.Array]:
    """Record this step's observation and return the one the policy should act on.

    Args:
      buffer: the value returned by init_buffer or by the previous call.
      obs: observation of the current simulator state, shape (obs_dim,).
      delay_steps: d >= 0, the same value passed to init_buffer.

    Returns:
      (new_buffer, delayed_obs): delayed_obs has shape (obs_dim,). For t >= d it
      is the observation from step t - d.
    """
    # TODO
    raise NotImplementedError
