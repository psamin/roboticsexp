# Reward shaping and curriculum experiments

Everything here is switchable from [configs/variants.yaml](configs/variants.yaml); no code changes are needed to add
a variant. All variants share the physics, the three task stages (hover → reach → robust) and the held-out tests,
so differences come from the reward, the reward curriculum, and what the policy and critic observe.

## What can be varied

| knob | where | what it does |
|---|---|---|
| shaped reward weights | `env.rewards` | distance, precision, velocity (stronger near the target), body rate, tilt, effort, action rate, success bonus, crash |
| **unstable-flight penalties** | `env.rewards.ang_acc`, `lin_acc` | penalise angular acceleration (attitude wobble, per 100 rad/s²) and linear acceleration (jerk, per 10 m/s²), squared |
| **L2F reward** | `env.reward_mode: l2f`, `env.l2f` | r = c − s·(C_p‖e‖² + C_q·sin²(tilt/2) + C_v‖v‖² + C_ω‖ω‖² + C_a‖a‖²), clipped at 0, from *Learning to Fly in Seconds* |
| **penalty curriculum** | `env.penalty_curriculum` | every N PPO updates multiply chosen weights by factors up to caps; optional `gate` = only while measured hover success ≥ gate |
| action history | `env.action_history` | last H actions in the observation (paper: 32) |
| asymmetric critic | `env.asymmetric_critic` | critic also sees disturbance force/torque, actuator state, height and distance |
| disturbances | `env.disturbance_force`, `disturbance_torque` | per-episode random constant force (fraction of weight) and torque (N·m), training only |
| motor lag | `env.motor_tau` | first-order lag on thrust and torques (s) |

The L2F constants come from the paper's code ([arplaboratory/learning-to-fly](https://github.com/arplaboratory/learning-to-fly)):
initial weights `reward_squared_position_only_torque` (position 5, orientation 5, linear velocity 0.01, action 0.01,
scale 0.5, constant 2), curriculum in `src/steps/curriculum.h` (every 100k steps: position ×1.2 up to 40, linear
velocity ×1.4 up to 1, action ×1.4 up to 1). The paper's ablation found that without the curriculum "the policies are
not able to fly early on" (1/10 vs 10/10 real-world flights at 300k steps). Differences from the paper here: PPO
instead of TD3; the curriculum step is every 20 PPO updates (~2M transitions) and gated on hover success; tilt-only
orientation cost because heading does not matter for this task; rewards clipped at zero and a crash penalty, because
this task's targets can start up to ~1.4 m away and a negative per-step reward would otherwise make crashing early
the best option.

## Metrics

Every evaluation runs 1,024 fixed episodes per suite: **hover**, **reach**, **robust** (the training stages) and
**recovery**, which is never trained on (start tilted up to 45°, moving up to 1 m/s and 1 rad/s). Besides hover
success, crash rate and distance, the stability columns average the last 2 s of each non-failed episode:
body rate |ω|, wobble |dω/dt|, action jitter |a_t − a_{t−1}|, and the largest tilt in the episode.

## Running

```bash
python scripts/train.py --variant stability --budget_min 20 --out outputs/exp/stability/train
python scripts/evaluate.py --checkpoint outputs/exp/stability/train/checkpoints/best.pt --policies ppo \
    --out outputs/exp/stability/eval.json
python scripts/compare_experiments.py --exp outputs/exp --out outputs/exp/report
# on the cluster, one job per variant:
VARIANT=stability sbatch --export=ALL cluster/experiment.sbatch
```

<!-- EXPERIMENT RESULTS -->
