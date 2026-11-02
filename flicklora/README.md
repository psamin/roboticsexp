# FlickLoRA: how does LoRA rank affect learning a fast, contact-rich flick?

A simulated finger flicks a loosened hex nut so it spins off a fixed bolt (Isaac Sim 5.1). A scripted expert
records demonstrations; SmolVLA (`lerobot/smolvla_base`) is adapted with LoRA at ranks 8 / 32 / 128 (alpha = r),
and compared with the unadapted base model, a full fine-tune, and the expert itself on 30 identical held-out tests.

## Where it runs (Futurama)

| Stage | Node | Why |
|---|---|---|
| Simulation, demos, eval | DGX Spark (`gpu:gb10:1`, aarch64) | The only nodes where Isaac Sim runs. x86 RTX nodes: 4.5 segfaults on driver 610; 5.x/6.x need glibc 2.35, RHEL 9 has 2.34 |
| Training | L40 (`gpu:l40:1`, x86) | LeRobot 0.6.1, ~0.35 s/step at batch 32 |

`cluster/common.sh` builds two throwaway envs per job in `/dev/shm`: Isaac Sim 5.1 (Python 3.11) and
LeRobot 0.6.1 `[smolvla,peft]` (Python 3.12). Results are written straight to `~/flicklora-results`.

## Run it

```bash
sbatch flicklora/cluster/phase_a.sbatch          # A: reference scene + our scene + contact-only test + comparison video
sbatch flicklora/cluster/collect.sbatch          # B+C: expert test videos, then 100 train + 10 val demos (raw)
bash   flicklora/cluster/run_ablation.sh ~/flicklora-results/raw-<collect job>   # D+E: train r8/r32/r128 (+full) x 2 seeds, eval all
python flicklora/train/analyze.py --results <copy of flicklora-results> --out <dir>   # table + plots
```
Smoke test (already passed): `MODE=lora RANK=32 STEPS=300 DATASET=lerobot/svla_so100_pickplace sbatch flicklora/cluster/train_lora.sbatch`.

## What was reused, what changed

Assets and screw mechanism come from
[Tech-Multiverse/omniverse-nut-and-bolt-digital-twin](https://github.com/Tech-Multiverse/omniverse-nut-and-bolt-digital-twin)
(commit `3089cc9`; the repo has no license file): `hex_bolt/nut.usd`, `hex_bolt/bolt.usd`, and the
`static_bolt.usda` idea of bolt-fixed -> prismatic slide -> revolute nut tied together by a `PhysxMimicJointAPI`.

| | Reference `static_bolt.usda` | FlickLoRA (`sim/scene.py`) |
|---|---|---|
| Thread coupling | 240,000 °/m = 1.5 mm pitch | **396,913 °/m = 0.907 mm**, measured from the bolt mesh helix (1/4"-28 UNF, right-hand) |
| What moves the nut | a velocity drive on the slide | **nothing**: no drive; only finger contact. Thread friction (Coulomb + viscous) only ever opposes spin |
| Bolt | vertical, gravity back-drives the screw | horizontal from a bracket |
| Look | grey defaults | dark steel bolt, zinc-yellow nut, red index mark, walnut bench, blue bracket, warm key light |
| Nut leaving the bolt | stops at the slide limit | pose + velocity handed to a free rigid-body nut, which falls off |

## Franka Panda in a warehouse cell (Oct 26 2026)

Same task with a Franka Panda (Isaac asset, de-instanced to render on Spark) on a steel workbench inside NVIDIA's
Isaac Sim `full_warehouse`. Control as in AssembleNIST: kp (600,600,600,600,250,150,50), kd (50,50,50,50,30,25,15),
87/12 N*m torque limits, gravity compensation, 2.175/2.61 rad/s joint limits. The flick is a joint-6 swing (fingertip
~0.6 m/s) with a 6 mm silicone nub under the closed fingers (`sim/franka.py`, `sim/franka_env.py`,
`sim/franka_expert.py`). Cameras: 3/4 view from the bolt-tip side + wrist camera. Demos: `raw-franka-12587`.

| Policy | Success | 95% CI | Seeds | Time to detach (s) | Trainable params | Train (L40) | Peak GPU |
|---|---|---|---|---|---|---|---|
| Base SmolVLA | 1/30 (3%) | 1-17% | – | – | – | – | – |
| LoRA r=8 | 16/60 (27%) | 17-39% | 47% / 7% | 7.8 | 0.37 M | 59 min | 8.1 GiB |
| LoRA r=32 | 36/60 (60%) | 47-71% | 50% / 70% | 9.2 | 1.49 M | 60 min | 8.1 GiB |
| LoRA r=128 | 50/60 (83%) | 72-91% | 73% / 93% | 7.2 | 5.94 M | 59 min | 8.1 GiB |
| Full fine-tune (1 seed) | 27/30 (90%) | 74-97% | – | 7.5 | 99.9 M | 61 min | 8.5 GiB |
| Scripted expert | 30/30 (100%) | 89-100% | – | 5.9 | – | – | – |

No simulation failures. Neighbouring ranks' pooled intervals don't overlap, but seed spread is large (rank 8: 47% vs
7%). Run: `ROBOT=franka sbatch --export=ALL cluster/collect.sbatch`, then
`ROBOT=franka SIGN=1 bash cluster/run_ablation.sh <raw dir>`.

Flick counts on the results page are recounted from each episode's nut trace (spin-ups within 0.35 s = one flick),
the same way for both robots. With that rule the finger's turns per flick are flat across ranks (0.86, 0.90, 0.90);
the earlier 0.86 -> 1.02 trend was an artifact of the old event split.

## Finger result (Oct 25 2026)

30 held-out test configs (seeds 90000-90029), identical for every policy; LoRA rows pool training seeds 1000 and 2000.
Full write-up with videos: https://claude.ai/artifact/QHedcNhiEEDfZEco7JC7Pd

| Policy | Success | 95% CI | Turns / flick | Time to detach (s) | Trainable params | Train time | Peak GPU |
|---|---|---|---|---|---|---|---|
| Base SmolVLA | 0/30 (0%) | 0-11% | 0.05 | – | – | – | – |
| LoRA r=8 | 41/60 (68%) | 56-79% | 0.86 | 5.23 | 0.37 M | 59 min | 8.2 GiB |
| LoRA r=32 | 49/60 (82%) | 70-89% | 0.91 | 4.82 | 1.49 M | 58 min | 8.2 GiB |
| LoRA r=128 | 55/60 (92%) | 82-96% | 1.02 | 4.18 | 5.94 M | 59 min | 8.3 GiB |
| Full fine-tune (1 seed) | 28/30 (93%) | 79-98% | 0.97 | 4.85 | 99.9 M | 61 min | 8.8 GiB |
| Scripted expert | 30/30 (100%) | 89-100% | 0.99 | 4.29 | – | – | – |

No simulation failures in any run. Rank 8 vs 128 is outside noise; rank 32 overlaps both. Caveat: alpha = r and one
shared LR, so rank and effective step size are tied.

## Measured along the way

| Check | Result |
|---|---|
| Reference scene loads and screws | 2.38 turns, 3.6 mm in 4 s; rotation/travel = 237,910 °/m vs 240,000 set (0.9%) |
| Contact alone drives our nut | one 1.2 m/s finger-tip pass -> 98 rad/s, 4.4 turns, 4.0 mm; rotation/travel = 396,908 vs 396,913 °/m (0.001%) |
| LoRA smoke test r=32 (300 steps, L40) | 1.49 M trainable of 450 M (0.33%), 37 modules, 8.4 GB peak, loss 0.63 -> 0.20 |
| Scripted expert (Phase B, 6 test configs) | 6/6 detach the nut, 2-6 flicks each; one flick ~ 110-155 rad/s then a free coast (~1.1 turns) |
| Demos (Phase C, `raw-12412`) | 100/101 train + 10/10 val successful; 14,302 frames; 5.3 flicks and 4.0 s per demo on average |
| LoRA sizes (alpha = r) | r8: 371,328 (0.083%) · r32: 1,485,312 (0.33%) · r128: 5,941,248 (1.32%) trainable, same 37 modules |

### Gotchas found on the way (and fixed)

- The rgb annotator returns the last *completed* frame and nothing for a few renders after a reset: raw frame 0
  of each demo is stale (the converter drops it), and `reset()` now waits for fresh frames before observing.
- A fingertip sweep at radius r only catches a hex corner within acos(r / r_corner) of vertical; aiming between the
  flat and corner radius missed forever whenever a flat faced up. The expert aims at the flat radius and deeper
  after a miss.
- `world.reset()` puts articulations back to joint zeros; the env re-applies the nut's start state explicitly.
- SmolVLA executes 50-step chunks by default (1.7 s blind); eval re-plans every 10 steps for every policy.

LoRA targets (LeRobot's SmolVLA default, logged per run in `peft_modules.json`): `q_proj`, `v_proj` in all 16
action-expert layers + `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`.

## Files

| File | Role |
|---|---|
| `sim/scene.py` | CAD measurement (pitch fit), scene build, finger, free nut |
| `sim/flick_env.py` | 30 Hz env: joint-target actions, 1200 Hz physics, thread friction, detach handoff, sim-failure checks |
| `sim/expert.py` | scripted flick expert (approach, aim, flick, reload, repeat) |
| `sim/run_expert.py` | Phase B test videos / Phase C demo collection |
| `sim/eval_policy.py`, `train/policy_server.py` | Phase E: sim client + SmolVLA server over localhost |
| `sim/metrics.py` | flick detection, turns per flick, peak/sustained ω, axial travel |
| `train/to_lerobot.py` | raw episodes -> LeRobot v3 dataset |
| `train/peft_inspect.py` | which modules get adapters, trainable params |
| `train/analyze.py` | comparison table + plots |
