# PPO, as implemented in this repository

Proximal Policy Optimization trains a stochastic policy from batches of its own experience. After each batch it
takes several gradient steps, and a clipped objective stops any single action's probability from moving far from
the policy that collected the data. Everything below is plain PyTorch in [drone_ppo/](drone_ppo/); no RL library is
imported.

## 1. The policy and the value function ([networks.py](drone_ppo/networks.py))

Two separate MLPs, each with two hidden layers of 128 ELU units.

- **Actor.** For observation s it outputs a mean μ(s) ∈ ℝ⁴; a learned, state-independent log σ sets the spread.
  An action is sampled as u ~ 𝒩(μ(s), σ²) and squashed, a = tanh(u), so thrust and torques always lie in (−1, 1).
  The log-probability of the executed action follows from the change of variables:

  log π(a|s) = Σᵢ [ log 𝒩(uᵢ; μᵢ, σᵢ) − log(1 − tanh²(uᵢ)) ],

  computed in the stable form log(1 − tanh² u) = 2(log 2 − u − softplus(−2u)). The buffer stores u, so the update
  re-evaluates exactly the density that produced each action (no atanh of saturated values).
- **Critic.** V(s), a scalar estimate of the discounted return from s.
- **Deterministic evaluation** uses a = tanh(μ(s)).

## 2. Collecting a batch ([trainer.py](drone_ppo/trainer.py), [rollout_buffer.py](drone_ppo/rollout_buffer.py))

4,096 simulated Crazyflies each act for 24 control steps (50 Hz), giving 98,304 transitions per update. Per step the
buffer stores s, u, log π(a|s), r, the episode-end flag, and V(s).

Episodes end in two different ways, handled differently:

- **Failure** (crash, flipped, left the flight volume): the episode's future is genuinely zero.
- **Time limit** (10 s): the drone was cut off, not finished. The environment keeps the observation of the final
  state before resetting, and the trainer adds γ·V(s_final) to that step's reward. Without this, a drone hovering
  perfectly until the time limit would look, to the critic, like one that crashed into nothing.

## 3. Advantages: Generalized Advantage Estimation

With done_t = 1 when the episode ended on step t:

δ_t = r_t + γ V(s_{t+1})(1 − done_t) − V(s_t)  
A_t = δ_t + γλ (1 − done_t) A_{t+1}  
R_t = A_t + V(s_t)

γ = 0.99 (horizon of roughly 100 steps = 2 s), λ = 0.95. A_t says how much better the action was than the critic
expected; R_t is the critic's regression target. Tests check this against an independent per-environment loop and
against the λ = 0 (TD) and λ = 1 (Monte Carlo) limits ([tests/test_ppo.py](tests/test_ppo.py)).

## 4. The update ([ppo.py](drone_ppo/ppo.py))

For 4 epochs, the batch is shuffled into 4 minibatches. For each minibatch:

1. **Advantage normalisation:** Â = (A − mean A) / (std A + 1e-8).
2. **Probability ratio:** r(θ) = π_θ(a|s) / π_old(a|s) = exp(log π_θ − log π_old).
3. **Clipped surrogate (actor loss):** L_clip = −E[ min( r Â, clip(r, 1 − ε, 1 + ε) Â ) ], ε = 0.2.
   When the advantage is positive the objective stops rewarding increases of r beyond 1.2; when negative, decreases
   below 0.8. Those samples contribute zero gradient, which bounds how far one update moves the policy.
4. **Critic loss:** L_V = ½ E[(V_θ(s) − R)²].
5. **Entropy bonus:** −c_H · H[𝒩(μ, σ)] (c_H = 0 in the default config; the Gaussian's spread is learned).
6. **Total:** L = L_clip + c_V L_V − c_H H, c_V = 1. Gradients are clipped to norm 1.0, separately for actor and
   critic, and applied with Adam.
7. **Approximate KL:** KL ≈ E[(r − 1) − log r]. The learning rate starts at 3e-4 and adapts per minibatch: divided
   by 1.5 when KL > 2 × 0.01, multiplied by 1.5 when KL < 0.5 × 0.01 (bounded to [1e-5, 1e-3]). This is the same
   schedule Isaac Lab's reference quadcopter configuration uses.

## 5. Checkpoints and evaluation

Checkpoints hold both networks, the Adam state, the current learning rate, the curriculum stage and the full config
([ppo.py](drone_ppo/ppo.py) `save` / `load`; round-trip tested). Every 4 minutes the deterministic policy flies one full
episode from each of 4,096 fixed held-out conditions (seeded separately from training); `best.pt` is the checkpoint
with the highest held-out hover-success score, not the highest training reward.

## 6. The task the policy sees

- **Observation (16):** target position relative to the drone, in the body frame; gravity direction in the body
  frame (attitude); linear and angular velocity in the body frame; the previous action. Each is divided by a fixed
  scale and clipped to ±5 ([task.py](drone_ppo/task.py)).
- **Action (4):** collective thrust and roll/pitch/yaw torques, applied as an external wrench on the Crazyflie body
  exactly as in Isaac Lab's quadcopter task. Thrust is centred on hover: a₀ = 0 is the drone's weight, +1 is 1.9×
  weight, −1 is zero. Torques are 0.01 N·m per unit.
- **Reward:** distance and precision terms, penalties on speed (stronger near the target), spin, tilt, control
  effort and action changes, a one-time bonus for the first continuous second within 10 cm of the target, and a
  one-time crash penalty. Each term is logged separately.
