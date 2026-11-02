# Real-world and sim-to-real RL on low-cost arms (LeRobot SO-100/SO-101), for ping-pong-ball bouncing — as of 2026-10-30

Scope and provenance:
- LeRobot facts were verified in the user's local checkout `/Users/praneethsamineni/Documents/GitHub/lerobot`: v0.6.2, commit e624f3f7, 2026-09-24. GitHub links point at `main`, which may have moved since.
- Papers were read in arXiv/ar5iv HTML. "(via sub-researcher)" marks findings gathered by a delegated research agent that I did not re-fetch myself.
- I re-verified these load-bearing claims myself: FlashSAC, Hamed et al. 2026, JuggleRL, Squint, HIL-SERL, SERL, Ploeger, and the LeRobot code.
- Numbers marked **[calc]** are my own idealized physics: point-mass ball, massive paddle, no drag or spin. They are not sourced.

## Q1. Zero-shot sim-to-real for a dynamic skill: what do people actually do? (priority question)

### Takeaway
No public work has transferred a dynamic or ballistic RL skill zero-shot to an SO-100/101. Every documented SO-arm zero-shot RL success is quasi-static pick, lift or stack at 10–50 Hz. The closest dynamic analogues are JuggleRL (quadrotor ball juggling) and DeepMind table tennis. Both transferred zero-shot, both used a tracked ball state rather than pixels, and both made four things decisive:
- an identified, randomized ball restitution;
- a modelled latency;
- a smoothness term;
- a broad ball-state distribution.

On legged and aerial robots, actuator-lag modelling is repeatedly the difference between a policy that works and one that "could not complete a single step".

### Cited Findings
**Actuator and system ID**
- Hwangbo et al. 2019 (ANYmal):
  - An actuator network fed the history of position error and velocity at t, t−0.01 and t−0.02 s. It was trained on data collected in under 4 minutes at 400 Hz (>1M samples).
  - Torque error: 0.740 Nm, against 3.55 Nm for an ideal-actuator model. Policies trained on the ideal model "could not complete a single step".
  - Rule of thumb: the history should be "sufficiently longer than the sum of all communication delays and the mechanical response time".
  - Source: [Hwangbo et al. 2019](https://ar5iv.labs.arxiv.org/html/1901.08652) (via sub-researcher).
- Tan et al. 2018 (Minitaur, PPO):
  - Measured latency of 3 ms in the PD loop and "typically 15-19ms" in the controller.
  - Verbatim: "Without either of them [actuator model, latency], the learned controllers do not work on the real robot."
  - Source: [Tan et al. 2018](https://ar5iv.labs.arxiv.org/html/1804.10332) (via sub-researcher).
- "Learning to Fly in Seconds" (Eschmann et al., RA-L 2024):
  - Motor delay is modelled as a first-order low-pass filter with a 0.15 s time constant. Without it, no usable policy was obtained.
  - Without action history: 5/10 flights succeeded and trajectory tracking was 0/10.
  - The policy transferred to a Crazyflie "even without domain randomization". Training took about 18 s and about 300k steps (TD3, 100 Hz sim).
  - Source: [Eschmann et al.](https://arxiv.org/html/2311.13081v2) (via sub-researcher).
- NVIDIA's SO-101 sim-to-real course (SAGE):
  - Replays real motion files in Isaac Sim and compares per-joint position, velocity and torque. The motion files include `actuator_bandwidth.txt` and `oscillation_low_freq.txt`.
  - GapONet learns a delta-action model from 8 h of real SO-101 data. No before/after numbers are published.
  - Sources: [NVIDIA SAGE page](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html), [sage repo](https://github.com/isaac-sim2real/sage) (via sub-researcher).
- so101-mjlab fits STS3215 parameters from a 50 Hz step-response sysid: Kp 18.33, Kd 1.34, effort 1.47 N·m, frictionloss 0.20, armature 0.0211. The repo includes the sysid script. [so101-mjlab](https://github.com/vssinghh/so101-mjlab) (via sub-researcher)
- ASAP trains a delta-action model on real data and reports beating SysID and DR baselines on tracking error. Only the abstract was read. [ASAP](https://arxiv.org/abs/2502.01143) (via sub-researcher)

**Domain randomization ranges reported for ball tasks**
- JuggleRL (Tsinghua; arXiv 2509.24892, v2 dated 14 Jan 2026), verified:
  - Setup: PPO, 4096 envs, 50 Hz policy, about 2B frames in under 3 h on one RTX 4090. Ball and drone state come from 120 Hz motion capture with Kalman-filtered velocities.
  - DR: restitution [0.75, 0.90]. Identified restitution was 0.82 in the racket "sweet spot" (radius 5.5 cm) and about 0.64 at the edge.
  - Real-world ablation, average hits:
    | Variant | Average hits |
    |---|---|
    | Full system | 311.0 |
    | w/o restitution DR | 6.8 |
    | w/o horizontal-position DR | 0.5 |
    | w/o height DR | 20.3 |
    | w/o low-latency comms | 200.2 |
    | w/o smoothness reward | 21.3 |
  - Model-based planner baseline (Müller et al. 2011): 3.1 average.
  - Source: [JuggleRL](https://arxiv.org/html/2509.24892).
- JuggleRL's reward terms: hit +50, apex-in-range +50, relative-position 1/(1+max(d_xy, 0.2)), contact 10·exp(−5·d_axis), smoothness 2.0·exp(−ε_t) on the action difference, plus deviation and spin penalties. It does not describe sim latency modelling or observation-noise injection. [JuggleRL](https://arxiv.org/html/2509.24892)
- DeepMind table tennis (arXiv 2408.03906, v3 May 2025):
  - Ball state comes from two 125 Hz cameras and a neural tracker; the policy never sees pixels.
  - Observations: 8 timesteps (about 0.14 s) of ball position and velocity plus joint positions. The policy is a roughly 10k-parameter dilated-gated 1D CNN outputting joint velocities at 50 Hz.
  - Sim latency is randomized around measured means (ms): ball observation 40, arm observation 29, arm action 71. Total system latency is about 100 ms.
  - Result: 45% of 29 matches won.
  - Source: [DeepMind TT](https://arxiv.org/html/2408.03906v3) (via sub-researcher).
- Reactive ball catching, FR3 arm, zero-shot (arXiv 2605.09789):
  - DR: restitution 0.4–0.7, friction 0–0.1, ball radius 2–4 cm. Gains were identified on the real robot rather than randomized.
  - Result: 41/60 (68%), against 8/60 for an end-to-end baseline that was "highly sensitive to observation noise". Control at 20 Hz.
  - Source: [reactive catching](https://arxiv.org/html/2605.09789v1) (via sub-researcher).
- Humanoid-Gym DR (Table III): system delay 0–10 ms, friction 0.1–2.0, motor strength 95–105%, joint-position noise σ 0.05 rad, joint-velocity noise σ 0.5 rad/s. [Humanoid-Gym](https://arxiv.org/html/2404.05695v1) (via sub-researcher)
- Rudin et al. 2021 (legged_gym): 4096 envs, PPO, 50 Hz policy, previous actions in the observation, uniform noise (e.g. joint position ±0.01 rad, joint velocity ±1.5 rad/s). Flat terrain trains in under 4 min. [Rudin et al.](https://ar5iv.labs.arxiv.org/html/2109.11978) (via sub-researcher)
- Dactyl randomized forces on the object itself. Removing those "unmodeled effects" dropped the real-robot median consecutive successes from 13 to 2. [OpenAI Dactyl](https://ar5iv.labs.arxiv.org/html/1808.00177) (via sub-researcher)

**Smoothing**
- DroQ "A Walk in the Park": a low-pass filter on PD targets "degrades learning efficiency". [Smith et al. 2022](https://ar5iv.labs.arxiv.org/html/2208.07860) (via sub-researcher)
- A smoothness benchmark found filtering "can lead to even larger oscillations in the raw outputs". LipsNet+L2C2 gave 26.8% smoother control with ≤2.8% return loss. [arXiv 2410.16632](https://arxiv.org/html/2410.16632v1) (via sub-researcher)
- CAPS smoothness regularization gave an "almost 80% reduction in power consumption" on a real quadrotor. [CAPS](https://arxiv.org/abs/2012.06644) (via sub-researcher)
- Squint (SO-101) scaled actions by 0.15 and ran at 30 Hz on the real arm against 10 Hz in sim, which "provides smoother trajectories". [Squint](https://arxiv.org/html/2602.21203) (via sub-researcher)

**History, privileged teachers, distillation**
- RMA's adaptation module reads the last 50 steps (0.5 s) of state and action. Sim success: RMA 73.5%, plain DR 62.4%, expert 76.2%. Base policy at 100 Hz. [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034) (via sub-researcher)
- Miki et al. 2022: a GRU belief-encoder student distilled from a privileged PPO teacher. [Miki et al.](https://ar5iv.labs.arxiv.org/html/2201.08117) (via sub-researcher)
- HIB reports teacher/student/DR-only numbers, but only for DMC and locomotion: Quadruped Walk 973 / 864 / 224; stairs success 98.0% / 94.2%. [HIB](https://arxiv.org/html/2305.18464v2) (via sub-researcher)

**Training budgets for comparable work**
| Work | Envs | Budget | GPU |
|---|---|---|---|
| [JuggleRL](https://arxiv.org/html/2509.24892) | 4096 | about 2B frames, under 3 h | RTX 4090 |
| [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md) (SO100 RGB grasp) | 1024 | 25–40M steps, about 1 h | 4090 |
| [so101-mjlab](https://github.com/vssinghh/so101-mjlab) (lift) | 2048 | 49.15M steps | — |
| [Squint](https://arxiv.org/abs/2602.21203) | — | 15 min per run, most tasks converge in under 6 min | RTX 3090 |

All via sub-researcher except JuggleRL and Squint's time, which I verified.

**Sim-to-sim gating**
- Humanoid-Gym validates Isaac Gym policies in a calibrated MuJoCo. Its sine-sweep trajectories are "very close after calibration", shown only in a figure. I found no quantitative evidence that a sim2sim pass predicts real success. [Humanoid-Gym](https://arxiv.org/html/2404.05695v1) (via sub-researcher)
- lerobot-sim2real warns: "Over training can sometimes lead to worse policies!" and "some checkpoints might do better than others despite having the same performance in simulation." [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md) (via sub-researcher)

### Inferences
- **The recommended plan centres on zero-shot transfer.** It follows the JuggleRL and table-tennis template, adapted to slow position servos. The checklist is ordered: do not train seriously before steps 1–4 are done.
  1. **Servo system ID first.**
     - Use the exact command path you will deploy: LeRobot `sync_write("Goal_Position")` at your control rate, with LeRobot's P/I/D of 16/0/32 (see Q2).
     - Log steps, chirps (or `actuator_bandwidth`-style sweeps) and square waves per joint while carrying the paddle.
     - Fit pure delay, a first-order lag (or 2nd-order), velocity and acceleration saturation, backlash and deadband. Alternatively fit a small actuator net: Hwangbo shows a few minutes of data suffice.
     - Implement it in Isaac Lab as an explicit/custom actuator model (or a learned net), not `ImplicitActuatorCfg` with copied gains. Published SO-101 sim gains disagree by about 250×; see Q2.
  2. **Measure end-to-end latency.** Time the full path: C920 frame → ball detection → policy → serial write → motion onset. One method is to film the arm and an LED with a high-fps phone camera. Randomize latency in sim around the measured value, with a margin of about ±1 control step as in so101-mjlab's 0–2 step delay. A 30 fps webcam adds ≥33 ms of frame period alone **[calc]**.
  3. **Measure ball–paddle restitution and friction.** Drop-test with a high-fps phone video, at the paddle centre and at the edge. JuggleRL found 0.82 vs 0.64 across one racket, and restitution DR was their single most important term. Randomize about ±0.075 around the measured value, JuggleRL-style.
  4. **Build the ball-state estimator before training.**
     - Fit a ballistic Kalman filter to 3D ball position from the top camera, plus the wrist camera if useful.
     - In sim, the policy should see what the estimator outputs: noisy, delayed and possibly dropped position and velocity at the camera rate. The critic should see true state.
  5. **Sim design.** PPO, 4096 envs, physics dt ≤ 2–5 ms, policy at 30–50 Hz. Actions are joint-position deltas fed through the identified servo model.
     - Actor input: about 0.1–0.2 s of joint-position and action history, plus the estimated ball state.
     - Asymmetric critic inputs: true ball state, restitution, delays.
     - Rewards: an action-rate/smoothness term (JuggleRL: removing it cut hits from 311 to 21.3), and an early-termination penalty on a drop.
  6. **DR list.**
     - Restitution and ball friction.
     - Ball start offset, height and velocity (see Q5).
     - Latency, plus camera-rate jitter and dropouts.
     - Servo lag, gain, effort and velocity-limit scale (about ±10–20%).
     - Backlash of about 0.5–1° (Menagerie models ±0.5°; a bench test measured 0.87°).
     - Paddle mass and inertia; arm link masses (one rebuild measured +20% vs the sim mass).
     - Camera extrinsics error.
     - Random impulses on the ball (Dactyl-style).
  7. **Gates before touching hardware.**
     - (a) Real-to-sim replay: send the same command sequence to the real arm and to sim, and compare joint traces. The SAGE approach does this.
     - (b) Sim-to-sim: evaluate the frozen policy in MuJoCo with an independently sysid'd SO-101 (Menagerie or so101-mjlab). This is a weak gate with no quantitative validation found.
     - (c) Robustness sweeps in sim at DR extremes plus 1.5× latency.
     - (d) A ball-free hardware dry run: replay the policy's commanded trajectories open-loop and check tracking error and servo temperature.
     - Treat checkpoint choice as noisy and test several (lerobot-sim2real).
- **Budget.** With your H200 or L40S, expect JuggleRL-scale budgets: roughly 1–2B frames, a few GPU-hours per run. This is far from a bottleneck. The bottleneck is sysid and perception fidelity (inference from the budgets table).
- **Pixels vs tracker.** No dynamic ball task I found fed pixels to the policy at deployment. Use a tracker. Image-based policies on SO-101 (Squint, lerobot-sim2real) are quasi-static.

### Gaps
- No published step-response, bandwidth or deadband data for the STS3215. so101-mjlab's `sysid_sts3215.json` exists but was not opened.
- No paper on arm-based paddle bouncing trained in sim and deployed zero-shot, and none on a hobby-servo arm.
- Restitution of a ping-pong ball on whatever paddle the user mounts is unknown and must be measured.
- C920 end-to-end latency and LeRobot sync_read/sync_write round-trip latency at 1 Mbps: no measurement found.
- No isolated latency-randomization ablation with numbers; Tan et al. report actuator model and latency together.

## Q2. Sim-to-real RL on SO-100/SO-101 and similar hobby-servo arms: what transferred, what failed, how actuators were modelled

### Takeaway
Zero-shot RL transfer on SO-arms is documented only for slow tasks:
- Squint: 8 visual manipulation tasks, 73/80.
- lerobot-sim2real: SO100 RGB grasp.
- so101-mjlab: lift.

HIL-SERL sim pretraining in Isaac Sim followed by real use "didn't transfer at all" for a vision policy. The actuator models in public sims are mutually inconsistent and mostly not identified from data, so the user must run their own system ID.

### Cited Findings
- **LeRobot servo configuration** (v0.6.2, verified locally in `so_follower.py` `configure()`):
  - Every motor is set to position mode with `P_Coefficient=16`, `I_Coefficient=0`, `D_Coefficient=32`.
  - Only the gripper gets `Max_Torque_Limit 500` ("50% of max torque to avoid burnout"), `Protection_Current 250` and `Overload_Torque 25`. Arm joints keep the defaults.
  - Optional `max_relative_target` clamps each step's command relative to the present position.
  - Sources: [so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [config_so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py).
- The SO-101 follower "uses 6x STS3215 motors with 1/345 gearing." [LeRobot so101 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx) (verified locally)
- The Feetech register table exposes `Max_Temperature_Limit` (addr 13), `Present_Temperature` (63), `Acceleration` (41), `Maximum_Acceleration` (85) and `Return_Delay_Time` (7). Default values were not checked. [tables.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py) (verified locally)
- **STS3215 bench numbers** (secondary: a GitHub issue citing the Robo9 bench test of a 12 V C018 servo):
  - No-load speed 45.6 RPM ≈ 4.78 rad/s.
  - "stable dynamic torque ≈ 15 kg·cm ≈ 1.47 N·m"; backlash ≈ 0.87° against a spec of ≤ 0.5°; position error under load 1.8–2.6°.
  - Heated to 71 °C in continuous ±90° oscillation.
  - Sources: [le_snail #12](https://github.com/robintomar1/le_snail/issues/12), [Robo9 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/) (via sub-researcher).
  - On a non-SO-101 robot built on these servos, "every joint tracks about 100 ms behind its target". [le_snail #12](https://github.com/robintomar1/le_snail/issues/12) (via sub-researcher)
- **Public sim actuator models disagree** (all via sub-researcher, each read in source):

  | Source | Kp | Kd | Effort / force | Notes |
  |---|---|---|---|---|
  | [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/so_arm101.py) (`ImplicitActuatorCfg`) | 200 → 50 | 80 → 20 | `effort_limit_sim=1.9` | `velocity_limit_sim=1.5` rad/s |
  | [MuJoCo Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) | 998.22 | kv 2.731 | forcerange ±2.94 | backlash joint ±0.5° |
  | [TRS SO-ARM100 MJCF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/joints_properties.xml) | 17.8 | 0 | ±3.35 | — |
  | [ManiSkill SO100](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py) | 1e3 | 1e2 | force_limit 100 | about 34× stall torque |

  - Menagerie's gains come from a DC-motor derivation that assumes servo P = 16. The derivation warns that "if this gain is changed then these model values need to be recalculated." [RBE501 README](https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/README.md) (via sub-researcher)
- **isaac_so_arm101** (reach and lift tasks): the README's Sim2Real section says "_Work in progress._" [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (via sub-researcher)
- **lerobot-sim2real** (ManiSkill3, SO100 grasp-cube from RGB, zero-shot):
  - PPO with 1024 envs. "around 25 to 40 million [steps] are enough, which take about an hour of training on a 4090."
  - DR: random robot colour, a greenscreen of the real background, and camera alignment by sim/real overlay.
  - The main failure causes are calibration and camera-alignment errors. No SO100 success rate is given.
  - Source: [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md) (via sub-researcher).
- **Squint** (Almuzairee & Christensen, RA-L 2026, arXiv 2602.21203; v1 24 Feb 2026, v2 4 Sep 2026):
  - Visual SAC on an 8-task SO-101 set in ManiSkill3. "We train policies for 15 minutes on a single RTX 3090 GPU, with most tasks converging in under 6 minutes." (verified)
  - Real zero-shot result: 73/80 (91.3%). Reach, Lift and Place are 9–10/10; Stack Can 6/10. (via sub-researcher)
  - DR included joint noise "σ=5°" and colour jitter. Without colour jitter, real success fell to 58/80. (via sub-researcher)
  - Control: 10 Hz delta-joint in sim, 30 Hz on the real arm. (via sub-researcher)
  - Sources: [Squint abstract](https://arxiv.org/abs/2602.21203), [Squint HTML](https://arxiv.org/html/2602.21203).
- **Patil et al.** (SO-101 HIL-SERL, blog dated 28 Mar 2026):
  - Isaac Sim (LeIsaac SO-101) training: about 1 h gave 24,205 optimization steps; 16 h gave 168,000 steps at about 7.5–8 Hz optimization frequency.
  - Sim-to-real transfer of the vision policy: "This was a disaster it didnt transfer at all." Causes: robot colour, camera mounting, calibration, shadows, cube colour.
  - About 4 h of real fine-tuning got "some successful picks". From scratch on hardware with tight EE bounds: "a working policy in about 40 mins".
  - Mujoco/LeRobot use degrees and Isaac uses radians, which caused porting friction.
  - Source: [Patil blog](https://indraneelpatil.github.io/blog/2026/hil-serl).
- **trunc8** (Feb 2026 Seeed hackathon) reports the same arc: about 1 h sim cube pick, 16 h sim → failed transfer, about 4 h real fine-tune, about 40 min from scratch with tight bounds. [trunc8](https://trunc8.github.io/projects/hil-serl-lerobot)
- **NVIDIA's SO-101 course** uses GR00T imitation learning, not RL. It calls hobby-servo backlash "that accumulates through the kinematic chain" a major gap. [NVIDIA course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) (via sub-researcher)

### Inferences
- What has transferred on SO-arms is visual quasi-static manipulation, using heavy visual DR and action scaling. Nothing in public evidence shows SO-arm servos tracking fast trajectories in sim-matched ways. For bouncing, servo fidelity, not visual fidelity, is the main risk.
- **[calc] Paddle-tilt precision.** To shift the ball 5 cm/s laterally per bounce at a 20 cm bounce height, the paddle needs only about 0.8° of tilt. That is the same size as the measured backlash (0.87°) and below the loaded position error (1.8–2.6°). Lateral drift control is therefore at the servo's precision floor. A slightly concave paddle (see Q4) can provide passive centring.
- **[calc] Vertical speed is not the problem; launch from rest is.**
  - Sustaining a 20 cm bounce with restitution 0.8 needs only about 0.22 m/s upward paddle speed at impact.
  - A single-joint "throw" from rest is speed-limited. At the isaac_so_arm101 velocity cap (1.5 rad/s, 0.3 m lever) the apex is about 1 cm. At the ~4.7 rad/s no-load speed it is about 10 cm.
  - A "pump-up" launch works within limits. Repeated small hops at a paddle speed of 0.3 m/s converge to about 37 cm (e = 0.8) or about 15 cm (e = 0.7).

### Gaps
- The voltage variant (7.4 V vs 12 V STS3215) of the user's follower is unknown. Speed and torque differ by variant, and the le_snail and Robo9 figures are for 12 V.
- ManiSkill3's 91.6% sim2real figure: unverified which arm it refers to.
- Whether `Acceleration` is written by LeRobot (one source claims 254) is unverified. It is not set in the `configure()` code I read.
- No dynamic-task RL transfer exists on any SO-arm.

## Q3. Which RL algorithm: massively parallel PPO vs FastTD3/FastSAC/FlashSAC, SAPG, PQN, PQL, SAC/DroQ/RLPD, DreamerV3/TD-MPC2

### Takeaway
For a low-dimensional (5-DoF), short-horizon task in GPU-parallel sim, keep PPO. The user's existing implementation is a good fit:
- The newest off-policy method, FlashSAC (RSS'26), says it only "slightly outperforms PPO" on low-dimensional tasks.
- A July 2026 controlled study found PPO reached usable policies sooner in wall-clock time than SAC or TD-MPC2.
- PPO is the algorithm behind both zero-shot ball-juggling and legged successes.

Off-policy methods (DroQ/RLPD at UTD around 20) matter only if real-world fine-tuning is ever added. The user's goal is zero-shot.

### Cited Findings
- **FlashSAC** (Kim et al., arXiv 2604.04539; v1 6 Apr 2026, v2 15 May 2026), verified in the paper body:
  - "On low-dimensional tasks, FlashSAC slightly outperforms PPO" (§5.1).
  - G1 sim-to-real: "stable real-world locomotion after approximately 20 minutes of training" vs "PPO requires about 3 hours". Stairs: about 4 h vs "nearly 20 hours".
  - Setup: 1024 envs, batch 2048, UTD "2/1024". Off-policy methods trained for 50M steps, PPO for 200M.
  - Source: [FlashSAC](https://arxiv.org/html/2604.04539v2).
- **Hamed, Naik, Bellinger, Mahmood** (arXiv 2607.17326, 19 Jul 2026), verified, abstract only, no numbers: PPO produced a usable policy sooner in wall-clock than SAC and TD-MPC2, and "domain randomization affects all three algorithms in a similar way." [Hamed et al.](https://arxiv.org/abs/2607.17326)
- **FastTD3** (Seo et al., arXiv 2505.22642, May 2025) (via sub-researcher):
  - Solves HumanoidBench tasks "in under 3 hours on a single A100". Batch 32,768, low UTD, distributional critic.
  - Sim-to-real on a Booster T1.
  - Smoothness caveat: with the PPO-tuned reward it produced an "undeployable gait... abrupt arm movements" on G1. Stronger smoothness penalties fixed it.
  - The authors say FastSAC "tends to be unstable during training".
  - Source: [FastTD3](https://arxiv.org/html/2505.22642v3).
- **FastSAC/FastTD3 humanoid follow-up** (arXiv 2512.01996, Dec 2025): locomotion trains "in just 15 minutes with a single RTX 4090". DR included action delay, PD gains, mass, friction and pushes. Results were "competitive or superior to PPO" on tracking. [Seo et al. 2025b](https://arxiv.org/html/2512.01996v1) (via sub-researcher)
- **SAPG** (ICML 2024) beats PPO only at very large env counts: 24,576 envs and 2e10 samples. Regrasping: PPO 1.25 vs SAPG 35.7 successes. Its rationale is PPO saturating from "duplicated data" far above 4096 envs. [SAPG](https://arxiv.org/html/2407.20230) (via sub-researcher)
- **PQN** handles discrete actions only, so it does not apply directly. [PQN](https://arxiv.org/abs/2407.04811) (via sub-researcher)
- **PQL** (ICML 2023): 4096 envs; "fastest policy learning in five out of six tasks" vs PPO, SAC(n) and DDPG(n). It uses mixed per-env exploration noise σ ∈ [0.05, 0.8]. [PQL](https://arxiv.org/html/2307.12983v1) (via sub-researcher)
- **DroQ**: SAC with a dropout and LayerNorm critic, UTD 20, on a real A1 at 20 Hz. It "learns to traverse each given terrain with roughly 20,000 samples (20 minutes)". [Smith et al.](https://ar5iv.labs.arxiv.org/html/2208.07860) (via sub-researcher)
- **RLPD**: 50/50 online/offline batches, LayerNorm critic, ensemble of 10 critics with 2 subsampled, UTD 20. [RLPD](https://ar5iv.labs.arxiv.org/html/2302.02948) (via sub-researcher)
- **Model-based** (via sub-researcher):
  - DreamerV3 handles "over 150 diverse tasks" with one configuration. [DreamerV3](https://arxiv.org/abs/2301.04104)
  - DayDreamer taught a real quadruped to walk "in only 1 hour". [DayDreamer](https://arxiv.org/abs/2206.14176)
  - TD-MPC2: 104 tasks. [TD-MPC2](https://arxiv.org/abs/2310.16828)
  - No evidence found of either on fast contact-timing tasks or at 4096-env parallel scale.
- **JuggleRL**, the closest task analogue, used PPO with 4096 envs. [JuggleRL](https://arxiv.org/html/2509.24892)
- DeepMind table tennis used BGS (evolution strategies) "because we observed that it produced policies with relatively smooth actions". [DeepMind TT](https://arxiv.org/html/2408.03906v3) (via sub-researcher)

### Inferences
- **Recommendation: PPO in sim**, with the user's existing tanh-Gaussian, GAE, KL-adaptive-LR, asymmetric-critic code.
  - JuggleRL is direct evidence that PPO at 4096 envs solves a paddle-juggling task in under 3 GPU-hours.
  - Switching to FlashSAC or FastTD3 promises at best a slight speedup on a 5-DoF task, and FastTD3 has a documented smoothness risk.
  - At this env count, SAPG's advantage does not apply.
- Consider FastTD3/FlashSAC only if PPO plateaus. Even then, keep the smoothness reward, because off-policy policies were jerkier without it.
- **If real fine-tuning is ever added** (not the current goal): use RLPD/DroQ-style SAC at UTD of about 10–20, seeded with sim rollouts as prior data. LeRobot's SAC defaults to UTD 1 and 2 critics (see Q6).
- Model-based RL adds wall-clock cost and no demonstrated benefit for this task class.

### Gaps
- No RL algorithm comparison on ball bouncing or juggling specifically.
- No controlled PPO vs off-policy smoothness comparison.
- FastTD3's C51 settings and PQL's DClaw "33×" figure: not verified (possible rendering error).

## Q4. Policy, observation and action design: history vs recurrent, asymmetric critic, pixels vs tracked ball, action-space choice (joint targets / EE+IK / per-bounce stroke / residual on mirror law)

### Takeaway
Use the following, with ball state coming from an external tracker:
- an MLP over a short history of observations and actions (about 0.1–0.2 s);
- an asymmetric critic with privileged ball state and physics parameters;
- joint-position delta targets at 30–50 Hz through an identified servo model.

That is the recipe that transferred for drones, table tennis and juggling. Keep an analytic mirror-law or "negative acceleration at impact" controller as a baseline. It is also the base for residual RL if pure RL fails to transfer.

### Cited Findings
- **Action history**: without it, Learning to Fly in Seconds got trajectory tracking 0/10 and a 4.5× larger hover error (1.18 vs 0.26). [Eschmann et al.](https://arxiv.org/html/2311.13081v2) (via sub-researcher)
- **Asymmetric critic**:
  - Pinto et al. 2017: the "critic is trained on full states while the actor... gets rendered images", transferring "without training on any real world data". [Pinto et al.](https://arxiv.org/abs/1710.06542) (via sub-researcher)
  - Learning to Fly's critic also sees rotor speeds and the disturbance. [Eschmann et al.](https://arxiv.org/html/2311.13081v2) (via sub-researcher)
- **Recurrent and history encoders**:
  - RMA uses a 0.5 s history encoder. Feeding the raw history directly to the base policy gave "unnatural gaits". [RMA](https://ar5iv.labs.arxiv.org/html/2107.04034)
  - Miki et al. use a GRU student. [Miki et al.](https://ar5iv.labs.arxiv.org/html/2201.08117)
  - Both via sub-researcher.
- **Tracked ball state, not pixels**:
  - DeepMind TT: 125 Hz cameras → ball tracker → an 8-step history of ball position and velocity → 50 Hz joint-velocity actions. [DeepMind TT](https://arxiv.org/html/2408.03906v3) (via sub-researcher)
  - JuggleRL: 120 Hz mocap with Kalman-filtered velocity → 50 Hz PPO. [JuggleRL](https://arxiv.org/html/2509.24892)
- **Per-bounce/episodic open-loop primitives**:
  - Ploeger, Lutter, Peters (CoRL 2020), two-ball juggling on a 4-DoF Barrett WAM:
    - The policy is an open-loop Gaussian over joint via-points (21 parameters), tracked by a low-gain PD.
    - Episodic REPS with a binary reward: 20 episodes × 25 rollouts, "56 minutes of experience" on the real robot.
    - Best run: about 33 min and more than 4,500 catches.
  - They avoided sim because rigid-body sims miss actuator saturation, cable dynamics and ball contact. Policies even failed to transfer between identical WAMs.
  - A naive closed-loop adaptation reduced performance. The authors judged closed-loop network policies unsafe without stability guarantees.
  - The cyclic pattern is "prepended with an initial stroke movement".
  - Source: [Ploeger et al.](https://ar5iv.labs.arxiv.org/html/2010.13483).
- **Analytic controllers**:
  - The "mirror law" (Buehler, Koditschek, Kindlmann): the paddle trajectory is a mirrored and scaled version of the ball's trajectory about the desired impact height. Rizzi & Koditschek extended it to 3D, reportedly sustaining "thousands and thousands of impacts".
  - Primary PDF not read (403); this description comes from search-result summaries. [core.ac.uk PDF](https://core.ac.uk/download/76389447.pdf), [Rizzi 1991 exhibit](https://www.docketalarm.com/cases/PTAB/IPR2023-01426/ABB_Inc._v._RoboticVISIONTech_Inc/01-04-2024-Patent_Owner/Exhibit-2006-EX2006___Rizzi_1991)
- **Open-loop stability** (abstract-level sources):
  - Schaal & Atkeson showed stable periodic bouncing "is sufficiently characterized by a negative acceleration of the racket at the moment of impact". [Schaal et al. 1996](https://is.mpg.de/publications/schaal_jmb_1996), [Schaal & Atkeson 1993](https://is.mpg.de/publications/schaal_iicra_1993)
  - A human follow-up reported the lowest variability at impact accelerations between −4 and −1 m/s². Search summary only. [Sternad et al. 2001](https://is.mpg.de/publications/sternad_jephpp_2001)
  - The ETH "blind juggler" (Reist & D'Andrea) is open-loop stable through paddle curvature plus impact deceleration. Search summary only; the PDF failed to load. [Reist & D'Andrea](https://typeset.io/pdf/bouncing-an-unconstrained-ball-in-three-dimensions-with-a-4zog40ij6p.pdf)
- **HIL-SERL's dynamic tasks** (Jenga whipping, object flipping) used a 3-DoF feedforward wrench output at 10 Hz into a 1 kHz torque-level controller. [HIL-SERL](https://arxiv.org/html/2410.21845)
  - The SO-101's STS3215s are position-mode servos; LeRobot sets `Operating_Mode=POSITION`. [so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)

### Inferences
**Action-space comparison for slow position servos** (all inference):

| Option | Sample efficiency | Transfer on slow servos | Reacts to nudges | Verdict |
|---|---|---|---|---|
| Joint-position deltas, 30–50 Hz, closed-loop | Irrelevant in 4096-env sim | Good **if** the servo model is identified; directly matches the hardware interface | Yes, every step | **Primary choice** |
| EE/paddle-pose targets via IK (placo) | Similar | Adds IK latency and singularities; a 5-DoF arm cannot reach arbitrary 6-DoF paddle poses; hides joint limits from the policy | Yes | Avoid as the main action space; maybe use for the hold-level pose |
| Per-bounce parametrized stroke (event-based: pick tilt plus stroke velocity at apex) | Highest (Ploeger: 21 params, about 1 h real) | Robust, because each stroke can be tracked open-loop and is smooth | Only once per bounce (about 0.3–0.5 s) | Strong fallback, especially if 30 fps tracking is too noisy for 50 Hz closed loop |
| Residual RL on a mirror-law / negative-acceleration analytic controller | High | The analytic base supplies a stable limit cycle; RL learns servo-lag compensation | Yes | Best hedge if pure RL fails to transfer |

- **Observations.**
  - Actor: estimated ball position and velocity (plus optionally the predicted time-to-impact) from the tracker; joint positions; and the last 3–10 actions and joint readings (0.1–0.2 s at 50 Hz). The length should exceed the measured delay plus the servo time constant, per Hwangbo's rule.
  - Critic: true ball state, restitution, latency and servo parameters.
  - Use an MLP first. A GRU adds little over explicit history for short delays; DeepMind TT used an 8-step CNN.
- **Raw pixels.** Not recommended. At 30 fps a 0.4 s bounce gives only about 12 frames **[calc]**, and motion blur at about 2 m/s ball speed (≈6.7 cm/frame at 30 fps **[calc]**) makes pixel policies harder to transfer. Every successful dynamic ball system used a tracker.
- **Paddle hardware.** A slightly concave paddle gives passive centring (blind-juggler principle). That relaxes the about 0.8° tilt-precision requirement computed in Q2.
- **Sensing upgrade.** Consider a higher-fps camera (60–120 fps, manual exposure) for ball tracking. Both zero-shot ball systems used 120–125 Hz sensing.

### Gaps
- No paper compares the action spaces above on the same ball task.
- The history length N_H in Learning to Fly is unverified.
- Mirror-law and blind-juggler details come from abstract and search summaries, not full texts.

## Q5. Required behaviours: start at rest on a level paddle, hold while a hand is in view, launch, then recover from nudges — how to train and structure it

### Takeaway
Train one always-on bounce policy from a broad initial-state mix:
- ball at rest on the paddle;
- ball in flight with random height, velocity and offset;
- ball just after a nudge.

Also apply random impulses to the ball during episodes. Wrap the policy in a small outer supervisor driven by a hand detector (e.g. MediaPipe) that gates *pre-launch only*.

There is a requirements conflict: "hold still while a hand is in view" and "react immediately when the user nudges the ball by hand" contradict each other, because the hand is in view during a nudge. The supervisor must apply the hold rule only before the first launch, or only while the ball is at rest.

### Cited Findings
- **DeepMimic reference-state initialization**: "For the backflip, without RSI, the policy never learns to perform a full mid-air flip... it performs a small backwards hop." Also: "Early termination proves to be crucial." [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717) (via sub-researcher)
- **Dactyl**: episodes start with the object resting on the palm (a contact-rich resting start) and transfer to the real robot. Random object forces are applied with per-episode probability log-uniform in 0.1%–10% per step, σ = 1 m/s² × mass, decaying 0.99 per 80 ms. Removing them dropped the real-robot median from 13 to 2. [Dactyl](https://ar5iv.labs.arxiv.org/html/1808.00177) (via sub-researcher)
- **Legged pushes**:
  - Rudin et al.: pushes "every 10 s", base accelerated "up to ±1 m/s". [Rudin](https://ar5iv.labs.arxiv.org/html/2109.11978)
  - OP3 soccer: impulses of 5–15 "Nm" for 0.05–0.15 s. Without randomization and perturbations, agents fell "with every one or two steps" on hardware. [OP3 soccer](https://ar5iv.labs.arxiv.org/html/2304.13653)
  - ANYmal recovery: training from random drops gave >97% success over more than 100 tests. [Lee et al. 2019](https://ar5iv.labs.arxiv.org/html/1901.07517)
  - All via sub-researcher.
- **Ball distributions**:
  - JuggleRL: ball z₀ ∈ [1.5, 2.0] m, x/y offset ±0.07 m. It always starts from a falling ball, never at rest. [JuggleRL](https://arxiv.org/html/2509.24892) (via sub-researcher, consistent with my fetch)
  - DeepMind TT sampled jointly from 14.2k rally and 3.4k serve real ball states, preserving correlations between dimensions. [DeepMind TT](https://arxiv.org/html/2408.03906) (via sub-researcher)
- **Launch phase precedent**: Ploeger's juggling prepends "an initial stroke movement that quickly enters the limit cycle"; starting the cyclic pattern directly gave "a bad first throw". [Ploeger](https://ar5iv.labs.arxiv.org/html/2010.13483) (via sub-researcher)
- **Supervisor plus skills**:
  - DeepMind TT chose one of 17 skills once per incoming ball. "switching LLCs mid-swing resulted in policies ending up in states that were outside of their training distribution." [DeepMind TT](https://arxiv.org/html/2408.03906)
  - ANYmal parkour: a 5 Hz selector over 50 Hz skills; switches were "smooth and unnoticeable". [ANYmal parkour](https://ar5iv.labs.arxiv.org/html/2306.14874)
  - ANYmal recovery used separate policies, because one multi-task policy showed "ungraceful behaviors". [Lee et al. 2019](https://ar5iv.labs.arxiv.org/html/1901.07517)
  - All via sub-researcher.
- **Skill chaining needs matching hand-off states**: T-STAR raised chair_ingolf from 0% to 56% success by regularizing terminal and initial state sets. [T-STAR](https://ar5iv.labs.arxiv.org/html/2111.07999) (via sub-researcher)
- **Distillation alternative**: OP3 soccer distilled skill teachers into one policy that was "more diverse, fluent, and robust than simply composing the skills". [OP3 soccer](https://ar5iv.labs.arxiv.org/html/2304.13653) (via sub-researcher)
- **Stand-still conditioning in legged RL**: the standing reward is gated at commanded speed < 0.1 [BeamDojo](https://arxiv.org/pdf/2502.10363). Jitter at zero command is reported only anecdotally. (via sub-researcher)
- **MediaPipe HandLandmarker**: 17.12 ms CPU / 12.27 ms GPU on a Pixel 6. In VIDEO/LIVE_STREAM mode palm detection re-runs only when tracking is lost. [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) (via sub-researcher)

### Inferences
**Recommended structure** (inference):
- **Supervisor (outer state machine, about 30 Hz):** `IDLE_HOLD → ARMED → ACTIVE → (DROP → IDLE_HOLD)`.
  - IDLE_HOLD: send a fixed, level paddle pose with a plain position command, not the policy. This sidesteps the "jitter at zero command" failure mode.
  - IDLE_HOLD → ARMED once the ball is detected at rest on the paddle and no hand has been seen for N frames (e.g. 0.5 s).
  - ARMED → ACTIVE starts the RL policy.
  - In ACTIVE, ignore hand presence. Exit only on a ball drop or loss, or on a safety trip (temperature, torque, joint limits).
- **One RL policy covers launch and bouncing.** Episodes initialize as about 30–40% ball resting on a level paddle at random offset and about 60–70% ball in flight (random apex height, lateral offset and velocity, spin optional).
  - The policy learns its own pump-up launch from rest. Q2 **[calc]** suggests pumping beats a single throw on these servos.
  - This avoids T-STAR-style hand-off mismatch between separate launch and bounce policies.
- **Nudge recovery.** Add random ball impulses (Dactyl-style, log-uniform per-episode probability) and "human-like" velocity kicks during flight and at rest. Use early termination on a drop.
- **Hold or go flag.** A "go" input bit into the policy is unnecessary if the supervisor owns the hold. If the user wants the policy itself to hold, train with a flag that is randomly toggled mid-episode, gate the reward, and add a strong action-rate penalty. Flag-ignoring and mode-switch failure modes are undocumented.
- **Separate launch and bounce policies.** This option is only worth it if the single policy fails to learn the launch. If so, train the bounce policy on the launch policy's terminal states.

### Gaps
- No paddle-bouncing paper starts from a ball at rest on the paddle.
- None uses a height curriculum, and none quantifies recovery from hand nudges.
- MediaPipe latency on a Mac CPU is unpublished.
- No measured data on mode-flag failure modes.

## Q6. LeRobot HIL-SERL: what is implemented, which robots, workflow, results, and suitability for ball bouncing

### Takeaway
LeRobot v0.6.2 ships a complete HIL-SERL stack. It is documented with SO-100 configs and SO-101 leader-arm interventions, and community teams have run it on SO-101 for cube picking in about 40 min to 4 h.

It is designed for 10 Hz, image-based, end-effector-delta control with human resets. That makes it a poor fit for bouncing:
- A 0.4 s flight gives only about 4 policy steps **[calc]**.
- Success classifiers on single frames cannot see bounces.
- The position servos rule out the wrench control the original used for dynamic tasks.

### Cited Findings
**What is implemented** (all verified in the local v0.6.2 checkout)
- The docs promise you will "master training a policy with RL on a real robot in just a few hours". Components:
  - offline demos plus a vision reward classifier;
  - a distributed SAC actor/learner (`lerobot.rl.actor` / `lerobot.rl.learner`, gRPC);
  - human interventions;
  - joint/EE bounds and crop-ROI preprocessing.
  - Source: [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx).
- Requirements: the `hilserl` extra, "A gamepad (recommended) or keyboard", an optional leader arm, and "A URDF file for the robot for the kinematics package". [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx)
- Kinematics imports `placo`. [kinematics.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py)
- **Env config defaults**:
  - `fps: int = 10` ("Control frequency"), `control_mode: "gamepad"`.
  - `crop_params_dict` and `resize_size`; the docs advise 128×128 (or 64×64).
  - `end_effector_bounds` and `end_effector_step_sizes`, `terminate_on_success: True`, `success_threshold: 0.5`.
  - Source: [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx).
- **Robots**:
  - The doc examples use `so100_follower` / `so100_leader` and configs `env_config_so100.json` / `train_config_hilserl_so100.json`.
  - The SO101 leader is documented for interventions: its "reduced gears" let it track the follower, so "taking over is much smoother than the gearless SO100". Press `space` to intervene, `s` for success, `esc` for failure.
  - EE control for SO arms lives in `robots/so_follower/robot_kinematic_processor.py`.
  - Sources: [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx), [so_follower dir](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots/so_follower).
- **Workflow**:
  1. Find EE bounds with a leader arm.
  2. Record demos with `python -m lerobot.rl.gym_manipulator` (example `num_episodes_to_record: 15`).
  3. Pick crops with `lerobot.rl.crop_dataset_roi`.
  4. Optionally train a reward classifier with `lerobot-train` (example 5000 steps, 3×128×128 inputs). "Training a reward classifier is optional," and success can be annotated manually.
  5. Run learner plus actor; intervene with the gamepad trigger.
  - Source: [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx).
- **Doc results**: no success-rate table. With interventions, "the number of steps where the policy starts achieving the maximum reward is cut by a quarter" vs without (from a plot). [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx)
- **Tuning advice**: `temperature_init` start at 1e-2 (code default 1.0); weight push every 4 s by default (suggests 1–2 s); `storage_device` `"cuda"`. [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx), [configuration_sac.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rl/algorithms/sac/configuration_sac.py)
- **SAC defaults**: actor/critic/temperature lr 3e-4, discount 0.99, `num_critics 2`, `utd_ratio 1`, `grad_clip_norm 40`, `critic_target_update_weight 0.005`. [configuration_sac.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rl/algorithms/sac/configuration_sac.py)
- **Data mixing**: `OnlineOfflineMixer(online_ratio=1.0)` mixes online and offline buffers, which is RLPD-style when the ratio is set to 0.5. [data_mixer.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rl/data_sources/data_mixer.py)
- **Sim**: `gym_hil` is "a Franka Panda robot simulation based on MuJoCo" (PandaPickCube), at fps 10. No SO-101 sim env ships with it. [hilserl_sim.mdx](https://huggingface.co/docs/lerobot/hilserl_sim)
- **Related**: v0.6.2 also has `hil_data_collection.mdx`, i.e. human-in-the-loop intervention data for imitation-learning policies (DAgger-like). [hil_data_collection.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hil_data_collection.mdx)

**Community results on SO-101** (Patil, Mar 2026)
- Manual keyboard rewards were used, no classifier. Training was interrupted by "camera issues or robot issues", and `max_ee_step_m` was set to 0.05 m.
- The author's verdict: IL reached similar performance with about 2 h of data collection.
- Source: [Patil](https://indraneelpatil.github.io/blog/2026/hil-serl).

### Inferences
- **Not suitable for bouncing as shipped.**
  - The 10 Hz loop gives 3–5 decisions per bounce **[calc]**.
  - The 128×128 cropped images at 10 Hz alias the ball's motion.
  - The EE-delta IK path adds latency.
  - The binary per-frame success signal does not capture "keep bouncing".
  - Every dropped ball needs a human reset.
- Raising `fps` to 30–50 is possible in config, but the stack was only validated at 10 Hz. Untested.
- HIL-SERL's sweet spot is precise, quasi-static or episodic skills on the SO-101 (e.g. placing the ball on the paddle).

### Gaps
- No official LeRobot benchmark of HIL-SERL success rates or training times on SO-100/SO-101.
- Whether `gym_manipulator` sustains 30–50 Hz with cameras on a Mac: not tested.
- The default `online_ratio` used by the learner config (vs the class default) was not traced.

## Q7. Original HIL-SERL (Luo et al., 2024) and SERL (2024): sample efficiency, dynamic tasks, control rate, reward classifier

### Takeaway
HIL-SERL reached 100% success on 12 real tasks in 1–6 h each, using 20–30 demos plus human interventions. This includes two dynamic tasks: Jenga whipping in 1.25 h and object flipping in 1 h.

Every policy ran at 10 Hz on top of a 1 kHz impedance or wrench controller. The dynamic tasks were essentially open-loop, so the "dynamic" results do not imply reactive 10 Hz closed-loop control of a ballistic object.

### Cited Findings
- **Training time and success rate** (Table 1a):

  | Task | Training time | BC → HIL-SERL success |
  |---|---|---|
  | RAM insertion | 1.5 h | 29 → 100% |
  | USB grasp-insertion | 2.5 h | 26 → 100% |
  | Car dashboard assembly | 2 h | 41 → 100% |
  | Timing belt assembly | 6 h | 2 → 100% |
  | **Jenga whipping** | **1.25 h** | 8 → 100% |
  | **Object flipping** | **1 h** | 46 → 100% |
  | Average | — | 49.7 → 100% |

  - Cycle time improved by 1.8× on average.
  - Source: [HIL-SERL](https://arxiv.org/html/2410.21845).
- **Control**: "an impedance controller running at 1000 Hz, which accepts 10 Hz setpoints computed by the policy".
  - The dynamic tasks output a 3-DoF feedforward wrench (Fx, Fz, τz or τy) at 10 Hz into a 1 kHz controller.
  - Jenga whipping needs "a highly dynamic open-loop behavior". The Jenga policy's Gaussian std stays near 0.
  - Source: [HIL-SERL](https://arxiv.org/html/2410.21845).
- **Data**:
  - Demo buffers are "usually on the range of 20-30" demonstrations. Jenga used 30 demos, no interventions, and human annotation of reward at episode end.
  - Baselines received 200 demos.
  - Source: [HIL-SERL](https://arxiv.org/html/2410.21845).
- **Reward classifier**:
  - Typically "200 positive data points and 1000 negative data points", about 10 trajectories, about 5 minutes to collect.
  - Pretrained ResNet-10 plus a 2-layer MLP with cross-entropy, trained for 100 iterations at lr 3e-4.
  - Accuracy 95–99%, with extra false-positive and false-negative samples added for some tasks.
  - Source: [HIL-SERL](https://arxiv.org/html/2410.21845).
- **Algorithm**: built on RLPD, sampling "equally between prior data and on-policy data". Grippers use a separate discrete (DQN-style) critic with a small gripper-action penalty. Images are 128×128 after cropping. [HIL-SERL](https://arxiv.org/html/2410.21845)
- **Resets**: human resets for Jenga whipping, cable clipping, dashboard, handover and belt; scripted for flipping, RAM, SSD, USB and IKEA. [HIL-SERL](https://arxiv.org/html/2410.21845)
- **Baselines** (Table 1b), object flipping: HIL-SERL 100%, "No itv" 100%, Residual RL 97%, IBRL 95%, DP 56%. Interventions mattered for precise tasks rather than for flipping. [HIL-SERL](https://arxiv.org/html/2410.21845)
- **Stated limitations**: it is uncertain whether the method extends to "significantly longer horizons". Extensive randomization and unstructured environments were not tested. [HIL-SERL](https://arxiv.org/html/2410.21845)
- **SERL** (Luo et al., ICRA 2024, arXiv 2401.16013): PCB assembly, cable routing and object relocation, with policies in "25 to 50 minutes" on average at "perfect or near-perfect" success. [SERL](https://arxiv.org/abs/2401.16013)

### Inferences
- The two dynamic HIL-SERL tasks are one-shot, open-loop motions with a single outcome, and they relied on torque-level control. Ball bouncing is a sustained, closed-loop limit cycle with an outcome every 0.3–0.5 s. HIL-SERL's evidence therefore does not transfer to bouncing on position servos.
- Flipping's "No itv" result of 100% suggests that for dynamic open-loop skills, demos plus RL suffice without interventions.

### Gaps
- Robot hardware is not named in the sections I read; Franka is widely reported but was not seen in the text.
- Exact intervention counts per task are not stated.

## Q8. RLPD and sample-efficient off-policy RL on real robots; real-world episodes and hours for dynamic skills

### Takeaway
Real-world dynamic skills learned from scratch or near-scratch take about 20 min to 6 h of robot time:
- DroQ quadruped walking: about 20 min / 20k samples.
- HIL-SERL Jenga whipping: 1.25 h.
- Ploeger two-ball juggling: 56 min of experience over 500 rollouts, with a launcher and expert initialization.

The closest analogue, Ploeger, avoided sim entirely because the sim could not capture the dynamics. That is a caution for the zero-shot plan, and the reason sysid comes first in Q1.

### Cited Findings
- **RLPD**: symmetric 50/50 sampling, LayerNorm, a 10-critic ensemble, UTD 20. It reports 2.5× over the prior best on Adroit Door. [RLPD](https://ar5iv.labs.arxiv.org/html/2302.02948) (via sub-researcher)
- **DroQ walk**: "roughly 20,000 samples (20 minutes)" on an A1 at 20 Hz. Bounding actions around default joints was essential. [Smith et al.](https://ar5iv.labs.arxiv.org/html/2208.07860) (via sub-researcher)
- **DayDreamer**: a quadruped walked "in only 1 hour" in the real world. [DayDreamer](https://arxiv.org/abs/2206.14176) (via sub-researcher)
- **Ploeger juggling**:
  - 20 episodes × 25 rollouts. Binary reward: 1 while both balls stay above 0.6 m.
  - A ball launcher with OptiTrack-triggered episode start; corrupted rollouts were repeated.
  - Safety measures: low gains, joint boxes, a funnel end-effector, sand-filled non-bouncing balls, and KL-bounded updates.
  - Learned vs hand-tuned juggling duration: 106.82 s vs 66.52 s average (120 s cap).
  - The conclusion also says learning "still takes up to 55 hours", which conflicts with "56 minutes" and is likely a typo.
  - Source: [Ploeger et al.](https://ar5iv.labs.arxiv.org/html/2010.13483).
- **HIL-SERL**: 1–6 h per task (Q7). **SERL**: 25–50 min (Q7).

### Inferences
- If the user ever adds real-world fine-tuning, budget about 1–4 h of supervised robot time per attempt series. A ball-return mechanism would be essential (see Q9).

### Gaps
- No real-world RL paper on a ping-pong ball paddle bouncing task with a low-cost arm.

## Q9. Safety and practical constraints on the real SO-101: overheating, torque limits, resets, realistic attempts per hour

### Takeaway
The real risks are servo heating under sustained oscillation and backlash-limited precision. LeRobot leaves arm-joint torque and current limits at defaults; it only limits the gripper. Resets are human (pick up the ball, place it on the paddle).

These constraints are why real-world RL from scratch is unattractive here. They still matter for zero-shot evaluation runs.

### Cited Findings
- An STS3215 heated to 71 °C in continuous ±90° oscillation (bench test, 12 V). [le_snail #12](https://github.com/robintomar1/le_snail/issues/12), [Robo9](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/) (via sub-researcher; secondary)
- LeRobot applies `Max_Torque_Limit 500`, `Protection_Current 250` and `Overload_Torque 25` only `if motor == "gripper"`. `Present_Temperature` and `Max_Temperature_Limit` registers are readable and writable via the bus. [so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [tables.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py) (verified locally)
- `max_relative_target` caps per-step position jumps for safety. [config_so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py) (verified locally)
- **Real-world HIL-SERL on SO-101**:
  - Hardware interruptions from "camera issues or robot issues".
  - "Resetting the environment is tedious."
  - The author expected the robot "will break before this learns anything" when training from scratch without tight bounds.
  - Source: [Patil](https://indraneelpatil.github.io/blog/2026/hil-serl).
- Ploeger cites high-acceleration "wear and tear" and collision damage risk as limitations. [Ploeger](https://ar5iv.labs.arxiv.org/html/2010.13483)

### Inferences
- **Attempts per hour** **[calc/inference]**: each drop needs about 5–15 s of human pick-and-place, and an attempt lasts about 2–30 s. A dedicated operator therefore gets roughly 100–400 attempts per hour. Bounce attempts are cheap in sim and expensive on the real arm, which reinforces the zero-shot plan.
- **Deployment safety for evaluation runs**:
  - Log `Present_Temperature` each second and stop above a threshold (e.g. 60–65 °C).
  - Set `max_relative_target` per joint to the sim's per-step action bound.
  - Consider lowering the arm-joint torque limit for early runs.
  - Keep the policy's action bounds inside a joint box (Ploeger, DroQ).
  - Make bouncing duty cycles short, e.g. 30–60 s runs followed by cooling.
- **Paddle mass**: a heavier paddle adds load to the wrist servos. isaac_so_arm101 gives them the lowest gains: wrist_flex Kp 80 / Kd 30, wrist_roll Kp 50 / Kd 20. Keep the paddle light (e.g. foam-core plus rubber) and include its mass in sysid.

### Gaps
- The Max_Temperature_Limit default value and the servo's thermal time constant were not found.
- No measured SO-101 duty-cycle limits for continuous oscillation.

## Q10. "Reward policy / reward model": learned success classifier or VLM reward vs an analytic reward from ball tracking

### Takeaway
For bouncing, a learned reward model adds nothing:
- In sim, the exact ball state gives a dense analytic reward: hit and apex-height bonuses, centring, smoothness, and drop termination, exactly as JuggleRL used.
- On the real arm under zero-shot deployment, no reward is needed at all. The ball tracker gives the evaluation metric (consecutive bounces).

LeRobot's reward models (image classifier; SARM, Robometer and TOPReward progress/VLM models) target quasi-static tasks and RA-BC demonstration weighting. They are too slow and coarse for 0.3–0.5 s bounce events.

### Cited Findings
- The HIL-SERL classifier predicts success from images, typically trained on 200 positive and 1000 negative frames, reaching 95–99% accuracy. It is used where hand-designing rewards from pixels is hard. [HIL-SERL](https://arxiv.org/html/2410.21845)
- LeRobot's classifier "uses a pretrained vision model" and outputs binary or multi-class success. "Training a reward classifier is optional." [hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx) (verified locally)
- **LeRobot v0.6.2 reward models** (verified locally):
  - `rewards/classifier`, `rewards/sarm`, `rewards/robometer`, `rewards/topreward`.
  - SARM docs: "Stage-Aware Reward Modeling... for long-horizon robot manipulation tasks", predicting progress 0→1 from video for "weighted imitation learning (RA-BC)" and possibly RL. [sarm.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/sarm.mdx)
  - TOPReward is "a zero-shot reward model" using log P("True" | video + instruction) from **Qwen3-VL-8B-Instruct**. [topreward.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/topreward.mdx), paper [arXiv 2602.19313](https://arxiv.org/abs/2602.19313)
- JuggleRL's analytic reward (hit, apex, relative position, contact, smoothness, spin) transferred zero-shot to 311–404 average hits. [JuggleRL](https://arxiv.org/html/2509.24892)
- Ploeger used a binary analytic reward from mocap (balls > 0.6 m). [Ploeger](https://ar5iv.labs.arxiv.org/html/2010.13483)
- Patil recommends "we should stop assigning manual rewards" and suggests rewards based on robot state. [Patil](https://indraneelpatil.github.io/blog/2026/hil-serl)

### Inferences
- "Run RL and reward policy" most likely refers to HIL-SERL's "reward classifier". For bouncing it is the wrong tool:
  - Bouncing is defined by events in time (impacts, apex heights), not by a single frame.
  - The ball tracker the controller already needs gives an exact reward.
  - An 8B VLM is far too slow for per-step rewards at 30–50 Hz.
- A learned model would only make sense as a cheap real-world **evaluation** or failure detector (e.g. "ball off paddle") if tracking were unavailable. Tracking is required anyway for the policy's observations.
- What the user may actually need is a **ball detector/tracker**: a colour threshold or small CNN plus a ballistic Kalman filter. That is the "model" worth training, not a reward model.

### Gaps
- No paper evaluates learned or VLM rewards on juggling or bouncing. This is absence of evidence, not a negative result.
- TOPReward's inference latency was not measured here.
