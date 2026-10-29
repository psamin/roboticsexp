# PPO Drone: a Crazyflie learns to hover with PPO written from scratch

A Crazyflie quadcopter in NVIDIA Isaac Sim learns to fly to a target and hover there, trained with Proximal Policy
Optimization implemented directly in PyTorch (no RL library). The environment is adapted from Isaac Lab's
open-source `Isaac-Quadcopter-Direct-v0` task: same Crazyflie asset, same thrust/torque interface, same physics rates.

<!-- RESULTS -->

## The setup

| | |
|---|---|
| Simulator | Isaac Sim 5.1, Isaac Lab 2.3.2 (direct workflow), GPU PhysX, 100 Hz physics, 50 Hz control |
| Hardware | NVIDIA DGX Spark (GB10, aarch64), Georgia Tech Futurama cluster |
| Robot | Bitcraze Crazyflie 2.x (NVIDIA asset via Isaac Lab's `CRAZYFLIE_CFG`), moved only by an external thrust + torque wrench on its body |
| Scene | flat light-grey floor, neutral grey background, one green target sphere (visual only) |
| Observation (16) | target position in the body frame, gravity direction in the body frame, linear and angular velocity (body frame), previous action |
| Action (4) | collective thrust (hover-centred: 0 = weight, +1 = 1.9 x weight, -1 = 0) and roll/pitch/yaw torque (0.01 N·m per unit), tanh-squashed |
| Episode | 10 s; fails on a crash (below 0.1 m or tilted past 90°) or leaving a 2 m radius / 2.5 m ceiling; the time limit is a truncation, not a failure |
| Success | staying within 10 cm of the target for 1 continuous second |
| PPO | actor and critic MLPs [128, 128], GAE (γ 0.99, λ 0.95), clip 0.2, 4 epochs x 4 minibatches, lr 3e-4 with KL-adaptive schedule, truncation bootstrapping |

### Curriculum (advances at 80% hover success over the last 2,048 episodes)

| stage | target | start | extras |
|---|---|---|---|
| 1 hover | fixed, 1 m above the floor | within 10 cm, tilt up to 0.1 rad | |
| 2 reach | x, y within 0.5 m; height 0.75-1.25 m | within 15 cm of the hover point, tilt 0.15 rad, small velocity | |
| 3 robust | x, y within 1 m; height 0.5-1.5 m | within 30 cm, tilt 0.3 rad, any heading, up to 0.3 m/s | constant side force up to 5% of weight |

## Layout

```text
ppo-drone/
├── PPO_EXPLAINED.md          the algorithm as implemented, with equations and pointers into the code
├── configs/train.yaml        environment, reward, curriculum, PPO and training settings
├── drone_ppo/
│   ├── environment.py        Isaac Lab DirectRLEnv (adapted from Isaac-Quadcopter-Direct-v0)
│   ├── task.py               thrust mapping, observations, reward terms, failures, curriculum stages, PD baseline
│   ├── networks.py           tanh-squashed Gaussian actor, critic
│   ├── rollout_buffer.py     rollout storage, GAE
│   ├── ppo.py                clipped-objective update, adaptive lr, checkpoints
│   ├── trainer.py            training loop, logging, held-out evaluation, best/latest checkpoints, resume
│   ├── evaluation.py         seeded held-out conditions and metrics
│   └── scene.py, plots.py    render-only scene and capture helpers, plots
├── scripts/                  verify_env, screenshot, train, evaluate, play, make_plots
├── tests/                    PPO math and task logic (CPU), including the PD controller on a rigid-body model
├── cluster/                  Slurm jobs for the DGX Spark nodes
└── outputs/                  screenshots, checkpoints, logs, plots, videos
```

## Commands

Isaac Sim 5.1 needs Linux with glibc 2.35+ and an RTX-class NVIDIA GPU; `cluster/common.sh` has the exact install
(Python 3.11, `isaacsim[all,extscache]==5.1.0`, torch 2.9.0 cu130 on aarch64, Isaac Lab v2.3.2 from source, numpy 1.26.4).

```bash
python -m pytest -q tests                                       # PPO math + task logic, CPU, seconds
python scripts/verify_env.py                                    # env checks, physics sanity, PD / random baselines
python scripts/screenshot.py --out outputs/screenshots/drone_ppo_setup.png
python scripts/train.py --budget_min 60 --out outputs/run       # headless, stops at the budget
python scripts/train.py --resume outputs/run/checkpoints/latest.pt --out outputs/run
python scripts/evaluate.py --checkpoint outputs/run/checkpoints/best.pt
python scripts/make_plots.py --run outputs/run --eval outputs/logs/eval_results.json
python scripts/play.py --checkpoint outputs/run/checkpoints/best.pt --video outputs/videos/drone_ppo_demo.mp4
python scripts/play.py --checkpoint outputs/run/checkpoints/best.pt --gui            # live, with a display
python scripts/screenshot.py --checkpoint outputs/run/checkpoints/best.pt --out outputs/screenshots/drone_ppo_trained.png
```

Loading a checkpoint in your own code:

```python
from drone_ppo.ppo import PPO
agent, extra = PPO.load("outputs/run/checkpoints/best.pt", device="cuda:0", load_optimizer=False)
action = agent.act_deterministic(obs)        # obs: (N, 16) tensor from the environment
```

On the Futurama cluster the whole sequence runs as one job: `sbatch cluster/pipeline.sbatch`.

## Credits

The environment structure, Crazyflie configuration and thrust/torque interface come from Isaac Lab
(BSD-3-Clause, `isaaclab_tasks/direct/quadcopter`); the Crazyflie model is from NVIDIA's Isaac Sim asset library.
