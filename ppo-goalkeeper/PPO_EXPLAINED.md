# PPO, as implemented here

Proximal Policy Optimization is an on-policy actor-critic method. The policy collects a batch of experience,
then takes several gradient steps on it while a clipped objective keeps each update close to the policy that
collected the data. Every piece below is plain PyTorch in [goalkeeper/rl/](goalkeeper/rl/).

## The loop

1. **Collect.** 1024 simulated goalkeepers each act for 32 control steps (30 Hz), giving 32,768 transitions per
   update ([scripts/train.py](scripts/train.py)). For each step the buffer stores the observation, the sampled
   action, its log-probability under the current policy, the reward, whether the episode ended, and the critic's
   value estimate ([buffer.py](goalkeeper/rl/buffer.py)).
2. **Score the actions.** Generalized Advantage Estimation turns rewards and value estimates into advantages
   ([gae.py](goalkeeper/rl/gae.py)):

   δₜ = rₜ + γ·V(sₜ₊₁)·(1 − doneₜ) − V(sₜ)  
   Aₜ = δₜ + γλ·(1 − doneₜ)·Aₜ₊₁  
   returns Rₜ = Aₜ + V(sₜ)

   γ = 0.99 discounts the future; λ = 0.95 trades bias against variance. Nothing is bootstrapped across an
   episode boundary.
3. **Update.** Four epochs over the batch in four shuffled minibatches ([ppo.py](goalkeeper/rl/ppo.py)). Per
   minibatch, advantages are normalised to zero mean and unit variance, then:

   - **Clipped surrogate.** With the probability ratio r = π_new(a|s) / π_old(a|s),
     L_policy = −E[min(r·A, clip(r, 1−ε, 1+ε)·A)], ε = 0.2. Once an action's probability has moved more than
     20% in the direction its advantage favours, that sample stops contributing gradient.
   - **Value loss.** ½·E[(V(s) − R)²] trains the critic toward the GAE returns.
   - **Entropy bonus.** −0.003·H(π) keeps some exploration.
   - Gradients are clipped to norm 1.0, separately for actor and critic; Adam, learning rate 3e-4.
4. **Watch the step size.** Approximate KL, E[(r − 1) − log r], is logged every minibatch. If an epoch's mean KL
   exceeds 1.5 × 0.03, the remaining epochs for that batch are skipped.

## The networks

Separate actor and critic MLPs, each with two hidden layers of 128 units and ELU activations
([networks.py](goalkeeper/rl/networks.py)). The actor outputs the mean of a Gaussian over the single action;
a learned, state-independent log standard deviation sets its spread. Training samples from the Gaussian;
evaluation and playback use the mean (deterministic).

## The task, from the policy's point of view

- **Observation (9 numbers):** ball position and velocity on the ground plane, keeper position and velocity
  along the goal line, ball position relative to the keeper, and the previous action, each divided by a fixed
  scale ([observation.py](goalkeeper/physics/observation.py)).
- **Action (1 number in [−1, 1]):** a requested sideways speed, a × 2.0 m/s. A bounded drive limits it to
  6 m/s² of acceleration and brakes early enough to stop at the travel limits; the result becomes wheel-speed
  targets for the Jackal's four wheels ([drive.py](goalkeeper/physics/drive.py),
  [goalkeeper_env.py](goalkeeper/environments/goalkeeper_env.py)).
- **Reward:** +10 for a save, −10 for a goal, potential-based shaping toward the predicted interception point,
  and small penalties on large or jerky actions ([rewards.py](goalkeeper/physics/rewards.py)). Potential-based
  shaping sums to a constant over an episode, so it guides early learning without changing which policy is best.

## Why PPO fits this problem

The simulator runs thousands of keepers in parallel on one GPU, so fresh on-policy data is cheap and
sample-efficiency matters less than stable updates. The action is continuous and low-dimensional, which suits a
Gaussian policy. The clipped objective makes large learning rates and many reuse epochs safe, so the run needs
little tuning.
