# Servo fidelity, not RL, decides zero-shot bouncing

**Recommendation.** Train a closed-loop PPO policy in MuJoCo, using MJX or MuJoCo Warp on the free H200. Train it against a servo model fitted on your own arm with the paddle bolted on. Feed it a tracked ball state, not pixels. Randomize ball restitution, the ball's lateral start state and latency around values you have measured. This is how the only two zero-shot ball systems worked, JuggleRL and DeepMind's table tennis, adapted here to slow hobby position servos.

**The task is easy mechanically.** A steady 10 cm bounce needs only **0.11–0.20 m/s of paddle speed at impact and a stroke of about 1–2 cm** (*calc*).

**It is hard in four places that no RL algorithm can fix:**

- **Servo speed.** At LeRobot's gains the STS3215 position loop behaves like a **~2–3 Hz system**, while bounces happen at **2.5–5 Hz**.
- **Launch.** Launching from rest needs the paddle to drop **faster than gravity**. LeRobot's `Acceleration=254` setting, which I confirmed it writes to every motor, probably caps acceleration below that.
- **Load.** A 150–190 g paddle loads the elbow and shoulder to **~80% of stall** on a 5 V supply.
- **Camera.** The 30 fps C920 sees a 10 cm bounce in only **~9 blurred frames**.

**The reward should be analytic and computed from simulator state.** A learned reward model adds nothing for this task.

**No one has published RL ball bouncing on a real SO-101.** The closest is a self-reported, simulation-only SO-101 project. Treat a first real result of **~10 consecutive bounces** as the milestone, and everything beyond it as earned.

**Before any training code is written, you need to:**

- read the servo and power-supply labels;
- replace the gripper's hold with a bolted adapter;
- buy a ~$60, 120 fps global-shutter camera for a side view;
- settle on a ~10 cm apex;
- plan to test P=32 and `Acceleration=0` during system identification.

*How to read this.* Numbers tagged *calc* are the research team's own derivations from cited inputs, not published measurements. *Measured locally* means a researcher ran the test on your Mac's CPU. Items marked *our proposal* are thresholds or ranges we suggest, not literature values. The first two sections are the action layer, about 7 minutes of reading; the evidence after them takes about 13 more. That is over your usual 15-minute budget, because the question covers twelve sub-topics.

## Six decisions gate everything else

| # | Decision | Do now | Recommendation | Why |
|---|---|---|---|---|
| 1 | **Servo variant and supply voltage** | Read the follower servo labels: C001 is 7.4 V / 19.5 kg·cm; C018 or C047 is 12 V / 30 kg·cm. Read the PSU label. | **7.4 V servos on a 5 V brick:** move to a supply at the servos' 7.4 V rating, after checking the bus-adapter board's input rating. **12 V servos on 12 V:** keep them. **Strongest option:** replace all six servos with 12 V parts plus a 12 V ≥5 A PSU, since 7.4 V and 12 V servos cannot share a bus supply. | At 5 V a C001 gives ≈1.35 N·m and ≈3.7 rad/s (*calc*, linear scaling). At 7.4 V it gives 1.91 N·m; the 12 V part gives 2.94 N·m. The standard kit ships 5 V/4 A adapters ([TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100), [Waveshare](https://www.waveshare.com/wiki/ST3215_Servo), [Feetech datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)). |
| 2 | **Paddle mount and handle length** | Weigh the paddle and find its centre of mass (CoM) by balancing it on a knife edge. | Cut the handle to a **~3 cm stub** and **bolt it to a printed adapter** on the wrist-roll flange or the gripper body. Do not grip it with the jaws. Keep the face horizontal and the handle along the wrist-roll axis. Keep the wrist housing behind or below the blade plane. Optionally strip the backhand rubber, which saves 30–60 g. | **Holding is the problem.** The jaws can resist ≈0.08–0.26 N·m of twisting; the paddle needs ≈0.18 N·m static and **0.35–0.43 N·m at launch** (*calc*), so it will slip. **The mount moves the CoM** from ≈0.26 m to ≈0.13–0.16 m from the wrist-flex axis. |
| 3 | **Extra camera** | Before buying, confirm the Mac delivers 120 fps from it: `ffmpeg -f avfoundation -list_options true -i "<n>"`. | **Arducam B0332 (OV9281, mono, global shutter, UVC, ~$60)** as a side camera at paddle height, against a black backdrop. Take the colour B0385 if you prefer colour. Keep the C920 overhead. | Estimating height from the ball's apparent size in the overhead view gives σ ≈ 1.3–2.7 cm per frame (*calc*), which is unusable. A side view resolves ≈0.83 mm/px. At 120 fps a 10 cm bounce gets ~34 frames ([welectron](https://welectron.com/Arducam-B0332-120fps-Global-Shutter-USB-Camera-Board)). |
| 4 | **Apex height** | Pick the nominal height. | **10 cm nominal**, training over a 7–15 cm band. Use 15 cm if you stay on 5 V and P=16. | **5 cm** is the hardest timing problem: 4.95 Hz, and only 6 control steps per flight at 30 Hz. **20 cm** is the hardest precision and field-of-view problem: 28 mm of drift per degree of tilt. **Lower bounces** survive longer on a flat paddle (*calc*). |
| 5 | **Servo gain and ramp** | Test both values, don't just switch. | Identify the servos at P=16 and P=32, and at `Acceleration` 254 and 0. Expect to deploy **P=32 and `Acceleration=0`** on the bouncing joints if the paddle does not oscillate. | **Gain:** P=32, the factory default, roughly doubles usable bandwidth (*calc*). **Ramp:** if a third-party register table is right, `Acceleration=254` is a ≈38.6 rad/s² cap, which is below g at any lever of 25 cm or less, so the arm could not launch ([register analysis](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)). |
| 6 | **Ball, rubber colour and light** | Buy orange 40 mm balls and flicker-free LED panels. | Orange ball, **black rubber face up**, white board overhead, black backdrop behind the side camera. | Strongest colour contrast for detection. Red rubber with an orange ball is the risky pair. Short exposures need bright light. |

**The voltage and mount decisions interact through the launch, so read the labels first.** A ball resting on a paddle leaves it only when the paddle accelerates downward faster than g at the ball's location (*calc*).

- **Long lever, 5 V.** The wrist hinge alone can drive a long-lever paddle (0.25 m) down at **12.4–14 m/s²**, because gravity helps.
- **Short mount, 5 V.** The short mount that saves the shoulder brings the lever to 0.15 m, and the wrist reaches only **8.2–9.6 m/s²**, which is below g.
- **Short mount, 7.4 V.** The same short mount reaches **10.9–12.8 m/s²**.

At 5 V, then, the short mount fixes holding but breaks launching; at 7.4 V or 12 V it gets both. In a typical forearm-horizontal pose at 5 V, a 170 g paddle with its CoM 0.25 m out loads the elbow and shoulder to **81% of stall**. That sits right at the servo's factory overload threshold of 80%, above which the servo drops to about 20% torque after 2 s. With the whole arm extended, the shoulder would need **126% of stall** (*calc* from URDF masses).

**Two items to keep in reserve, not buy now:**

- **A shallow printed concave insert or rim.** It restores the passive lateral centring that a flat paddle lacks.
- **Three or four piezo discs under the blade.** They give sub-millisecond impact timing.

## Seven phases, each closed by a measurable gate

**Who does what.** Claude writes every script and gives the commands. You run every live session on the arm, and nothing else opens the serial port while a run is live.

**This plan stays zero-shot.** The real measurements in Phases 1 and 4 are system identification; the policy's weights never see real data. When a real run fails, Claude replays its logs in simulation to find the model term that diverged, fixes the model or widens the randomization, and retrains. That is the measure-then-randomize loop JuggleRL and DeepMind followed.

| Phase | Where / who | Deliverables | Gate before moving on |
|---|---|---|---|
| **0 Hardware** | You, at the bench | Labels recorded; supply sorted; adapter printed and paddle bolted on; paddle mass and CoM; camera ordered; lights in place; a "level" home pose set with a phone level app on the paddle | Arm holds the bounce pose (forearm near vertical, paddle level) for 5 min with `Present_Load` under ~60%, temperature flat, and no overload trip (*our proposal*) |
| **1 Measure** | You run Claude's scripts (~1 h live, of which ~25–30 min is identification) | Register dump; bus benchmark (p50/p95/p99, error count); per-joint logs of steps, sine dwells at 1–6 Hz, chirps, max-effort moves and slow triangles, at P16/P32 × `Acceleration` 254/0; ball-drop restitution at the centre and rim, filmed with a 240 fps phone; post-impact lateral scatter; camera latency via an LED test | Bus write+read p99 well inside the 20 ms control period, with no blocking retries. Measured downward paddle acceleration at the ball location > g with `Acceleration=0`; if not, revisit Decision 1 |
| **2 Sim + fit** | Claude; least-squares fit on the Mac (seconds), CMA-ES on cluster CPU | Simulator model: Menagerie SO-101, plus paddle bodies from your measurements, plus a ball contact tuned by hand, plus an explicit servo model with delay; fitted parameters; a replay tool | Replay of held-out chirps and a multi-joint trace: **phase error < 15° and gain error < 15% across 2.5–5 Hz; launch peak velocity within 10%**. These thresholds are the servo researcher's proposal |
| **3 Perception** | Claude writes it, you run it | Frame grabbers that always take the newest frame; colour (HSV) ball detector; ballistic Kalman filter (EKF) with bounce handling and latency forward-prediction; intrinsic calibration and extrinsics to the robot base; hand detector; logged latency distribution | Ball 3D bias under ~1 cm against known positions; detection > ~98% in the bounce volume; no false "hand clear" in a scripted test (*our proposals*) |
| **4 Physics check with a real ball** | You, running a scripted stroke | Open-loop periodic stroke that hits the ball while decelerating, run on the real arm with the ball; the same command stream run in sim | Real apex sequence and lateral drift fall inside the sim's randomized spread (*our proposal*). This is the first test of the servo and contact models together |
| **5 RL** | H200 via `sbatch` | Throughput job; curriculum runs of 0.5–2B steps; ablations; 3–5 candidate checkpoints | In sim: median ≥ 100 bounces at nominal settings; ≥ 30 at the randomization extremes with 1.5× latency; ≥ 90% successful launches from rest; ≥ 80% nudge recoveries (*our proposals*). Isaac Lab sim-to-sim as a weak extra check |
| **6 Deploy** | You at the arm, Claude's commands | Supervisor plus policy runner, using a custom bus loop with no retries; dry run on the real arm with a simulated "virtual ball"; then the real ball | **Virtual-ball dry run:** tracking error inside the sim's distribution, temperature under 60–65 °C over 30–60 s runs. **Real ball:** 10 consecutive bounces → 50 → recovery from a nudge |

**The Phase 6 dry run is borrowed from Büchler's hybrid sim-and-real training.** There, recorded balls were replayed virtually while the real robot acted ([Büchler et al.](https://arxiv.org/abs/2006.05935)). Here a simulated ball reacts to the real paddle's measured joint angles. That tests servo tracking, timing and heat with nothing to drop.

Because public checkpoints that score the same in sim can transfer differently, test several of them on the arm ([lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)).

## The evidence, topic by topic

### Speed is plentiful; bandwidth, launch and payload are not

| Apex above paddle | Impact speed | Bounce period | Bounce rate | Paddle speed at impact (e 0.85 / 0.75) | Minimum stroke, peak-to-peak (e 0.85) | Frames per flight at 30 / 120 fps | Control steps per flight at 50 Hz | Landing drift per 1° tilt |
|---|---|---|---|---|---|---|---|---|
| 5 cm | 0.99 m/s | 202 ms | 4.95 Hz | 0.080 / 0.141 m/s | 5.2 mm | 6 / 24 | 10 | 7 mm |
| 10 cm | 1.40 m/s | 286 ms | 3.50 Hz | 0.114 / 0.200 m/s | 10.3 mm | 9 / 34 | 14 | 14 mm |
| 15 cm | 1.72 m/s | 350 ms | 2.86 Hz | 0.139 / 0.245 m/s | 15.5 mm | 10 / 42 | 17 | 21 mm |
| 20 cm | 1.98 m/s | 404 ms | 2.48 Hz | 0.161 / 0.283 m/s | 20.6 mm | 12 / 48 | 20 | 28 mm |

All rows are *calc*: steady-state ballistics with a heavy paddle, where paddle speed v_p = v(1−e)/(1+e) and landing drift ≈ 8hθ for apex h and tilt θ. "e" is the paddle–ball coefficient of restitution.

**Speed is not the limit.** A single joint delivers 0.5–1.3 m/s at the paddle (*calc*). The no-load speed is 5.45 rad/s at 7.4 V ([datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)), and the firmware rate limit was identified at **5.07–5.29 rad/s** ([Rhoban BAM](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V)).

**Bandwidth is the limit.** BAM identified the STS3215 with a large reflected rotor inertia (armature) of **0.022–0.026 kg·m²**, which outweighs the paddle's own 0.004–0.018 kg·m². Linearizing BAM's model at LeRobot's P=16 gives a ~2.1–3.2 Hz system (*calc*). At 3.5 Hz it passes **only 0.38–0.42 of the commanded amplitude at 5 V**, lagging 116–125°. At 7.4 V it passes 0.62–0.67; at P=32 and 7.4 V the response is about 1.3 (resonant). Each 15 ms of dead time costs another 27° at 5 Hz. So the policy must command strokes 2–5× larger than it wants, about a quarter to a third of a period early. A simulator that does not reproduce this curve teaches the wrong timing.

**Height stability comes for free if you hit while decelerating.** Schaal, Sternad and Atkeson showed that stable periodic bouncing "is sufficiently characterized by a negative acceleration of the racket at the moment of impact" ([MPI](https://is.mpg.de/publications/schaal_jmb_1996)). A review gives a Lyapunov-predicted optimum of −5 to −2 m/s², and humans converged to about **−4 m/s²** ([PMC3498056](https://pmc.ncbi.nlm.nih.gov/articles/PMC3498056/)). ETH's Blind Juggler bounced balls with no sensors at all, using a slightly concave paddle ([ETH IDSC](https://idsc.ethz.ch/research-dandrea/research-projects/archive/juggling-machines.html)).

**Lateral stability does not come for free.** A flat paddle has no restoring force, so tilt errors random-walk the ball off the blade. A Monte Carlo model gives a median of **7–17 bounces at a 10 cm apex with 0.5° of random tilt error, and 4–7 at 1°** (*calc*, model not data). That is why a flat store-bought paddle needs closed-loop ball sensing, as every working flat-plate bouncer has used ([Octo-Bouncer](https://www.electrondust.com/2020/03/01/the-octo-bouncer/)).

**Tilt control sits at the servo's precision floor.** One encoder step is 0.088°, about 1.2 mm of drift at a 10 cm apex. A 12 V STS3215 bench test measured **0.87° of backlash** and 1.8–2.6° of position error under load ([Robo9](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/), via [le_snail #12](https://github.com/robintomar1/le_snail/issues/12)).

**Launch from rest.** A direct throw needs the paddle at √(2gh) (1.40 m/s for 10 cm) and then braking harder than g. That is beyond any single joint (*calc*). Two launches are feasible:

- **Drop-and-hit.** The paddle drops faster than g so the ball floats free, falls 1–2 cm, and is struck at 0.44–0.61 m/s for a 10 cm apex.
- **Small hop plus pumping.** A small hop followed by 1–3 pumping hits of 0.24–0.38 m/s each.

**Heat is a real constraint.** Steady bouncing through the wrist costs 0.8–1.3 N·m, which is **60–95% of 5 V stall** (*calc*). The same 12 V bench servo reached **71 °C** in continuous oscillation ([Robo9](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)), against a 70 °C default cut-off.

### MuJoCo reproduces the bounce, but only with hand-set contact parameters

**Use the MuJoCo Menagerie `robotstudio_so101` model.** It was derived from TheRobotStudio's model, its wrist_roll limit was fixed on 2026-09-28, and all public SO-101 assets share its CAD link masses: 0.103 / 0.104 / 0.079 / 0.087 kg for the upper arm, lower arm, wrist and gripper ([Menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)). The paddle weighs more than the wrist and gripper together, so your paddle measurements matter more than refining the arm's links. Model the paddle as a real-mass child body, not a fixed or kinematically driven one:

- **Head:** a cylinder of radius ≈0.077 m and half-thickness 0.007 m.
- **Handle:** a box, at your measured offset.

**Menagerie's defaults will not bounce a 2.7 g ball.** Its 5 ms timestep plus default contact gave no bounce at all (*measured locally*). The research team's CPU tests found a setup that works ([bounce_test.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bounce_test.py)):

- **Settings:** direct-format contact `solref="-1e5 -32.7"` on the ball, a 1 ms timestep, and `shellinertia="true"` for the hollow ball.
- **Restitution:** 0.834–0.850 for impacts at 1–3 m/s, stable over 8 bounces.
- **Moving paddle:** exit speed 6.23 m/s against an expected 6.25 m/s.
- **MJX:** matches MuJoCo C within ~1%.

**Three traps:**

- **Positive-format `solref`** is averaged with the paddle's default, which collapses restitution to 0.13–0.27.
- **Contacts too stiff for the timestep** inject energy. Too soft (`k=1e4`) and the ball tunnels through the paddle.
- **Default friction rows** produce only ~30% of the correct spin. Fix this with `solreffriction="-1 -300"` on an explicit contact pair.

The MuJoCo docs call the direct format "the recommended format for system identification" ([MuJoCo](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)). The robo-ping-pong SO-101 project independently found that at 200 Hz "the solver injects energy" and moved to 2 kHz ([robo-ping-pong](https://github.com/mariapatni/robo-ping-pong)).

**Add drag.** Use quadratic drag with C_d ≈ 0.5; it trims a 30 cm apex by about 4% (*calc*). MJX-JAX supports only the inertia fluid model, so apply drag through `xfrc_applied` there ([MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)).

**Restitution on rubber is about 0.84–0.86 at these speeds.** Sony AI fits **e = 0.878 − 0.020·v_n**, which gives 0.858 at 1 m/s and 0.838 at 2 m/s ([Sony AI 2026](https://arxiv.org/html/2606.28805)). Gossard's 8,194-bounce study finds similar slopes and says constant-parameter models underperform ([Gossard](https://arxiv.org/html/2604.11349)). The difference between the centre and the rim of a paddle (0.82 against 0.64 on JuggleRL's racket) dwarfs the speed dependence. Measure your own paddle at the centre and near the rim.

**Neither MuJoCo nor PhysX can model rubber's tangential springback** (e_t ≈ 0.82). If Phase 4 shows that spin matters, swap in an analytic impulse model for the ball–paddle contact.

**Isaac Lab is the cross-check, not the primary simulator.** It is the only path that reuses [ppo-drone](/Users/praneethsamineni/Documents/GitHub/roboticsexp/ppo-drone) unchanged, but it runs only on the DGX Sparks. PhysX defaults also kill this task:

- the 0.5 m/s bounce threshold ignores slow contacts ([Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sim/simulation_cfg.py));
- restitution defaults to 0 with "average" combining;
- continuous collision detection runs on the CPU only ([PhysX](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html)).

No one tested a PhysX bounce in this research.

**MJX against MuJoCo Warp.** On a Franka benchmark, MuJoCo Warp ran at **3.54M physics steps/s against MJX's 1.91M**. But MJX was capped at 512 envs in a contact-heavy free-fall scene ([GPUSimBench](https://arxiv.org/html/2607.13059)), and Warp's bounce physics is unverified. Let the first H200 job run the bounce tests on both and pick.

**Bridging to your PyTorch PPO.** Two options, untested:

- pass JAX arrays to the PyTorch trainer zero-copy via DLPack;
- use MuJoCo Playground's Brax PPO.

### Every public SO-101 servo model is wrong for this task

| Source | Stiffness | Damping | Torque limit | Note |
|---|---|---|---|---|
| [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml) | kp 998 | kv 2.73 | ±2.94 N·m | Saturates at 0.17° of error, against ~21° implied by BAM: about 100× too stiff |
| [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/so_arm101.py) | 50–200 | 20–80 | 1.9 N·m | Velocity cap 1.5 rad/s (≈3.5× too slow); no armature (too agile) |
| [ManiSkill SO-100](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py) | 1e3 | 1e2 | 100 N·m | ~34× the real stall torque |
| [so101-mjlab](https://github.com/vssinghh/so101-mjlab) (fitted at 50 Hz) | 18.33 | 1.34 | 1.47 N·m | Armature 0.0211; no payload; reported via a sub-researcher |
| [BAM m1](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V) (identified) | firmware-style | — | — | Armature 0.026, v_max 5.29 rad/s, delay 1.05 ms, Coulomb friction 0.052 N·m; recorded with D=0 |

**Modelling the actuator is the lesson every zero-shot robot paper repeats.**

- **Tan et al.:** "Without either of them [actuator model, latency], the learned controllers do not work on the real robot" ([Tan et al.](https://ar5iv.labs.arxiv.org/html/1804.10332)).
- **Hwangbo et al.:** policies trained on an ideal actuator "could not complete a single step". An actuator network trained on under 4 minutes of data cut torque error from 3.55 to 0.74 N·m ([Hwangbo et al.](https://ar5iv.labs.arxiv.org/html/1901.08652)).

**Build the servo model as an explicit chain, in BAM's structure** ([BAM actuator.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/actuator.py)), and fit it to your Phase 1 logs:

1. Command dead time and zero-order hold.
2. A rate-limited internal target (~5.2 rad/s).
3. An optional acceleration ramp.
4. A clipped proportional-derivative duty cycle, ±0.97.
5. A DC-motor torque model.
6. Armature plus Coulomb and viscous friction.
7. Torque limits, ±1-step deadband, backlash and 0.088° quantization.

Initialize from BAM m1. Treat BAM's motor constant and resistance as a fitted pair: together they imply 3.4 N·m of stall against the datasheet's 1.91 N·m. An actuator network is the fallback if the fit misses.

**Write the identification loop on the raw `FeetechMotorsBus`.** `get_observation()` reads position only.

- **Re-write deployment registers after connecting.** LeRobot overwrites them on connect: `configure()` calls `configure_motors()`, which writes `Acceleration=254` and `Maximum_Acceleration=254` (verified in LeRobot 0.6.1 and 0.6.2 locally; [feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)).
- **Read back the torque limits.** Your README notes P12/D40 could not lift the arm ([README](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/README.md)). LeRobot limits torque only on the gripper, so confirm the arm joints are at the factory `Max_Torque=1000`.
- **Expect the bus to be fast but its tail to be dangerous.**
  - **Speed:** at 1 Mbps a six-servo position read takes ≈0.62 ms on the wire and a goal write ≈0.26 ms (*calc*). A loop of 100–200 Hz is plausible but unverified.
  - **Tail:** LeRobot's patched timeout adds 50 ms per bad packet, and `num_read_retries=2` can stall the loop ≥100 ms, half a 5 cm bounce.
  - **Fix:** deploy with `num_retry=0`, and on an error reuse the last state.

### PPO at 50 Hz on tracked ball state, with an asymmetric critic

| Choice | Recommendation | Evidence |
|---|---|---|
| Algorithm | PPO, reusing [ppo-drone](/Users/praneethsamineni/Documents/GitHub/roboticsexp/ppo-drone/drone_ppo) (tanh-Gaussian, GAE, KL-adaptive learning rate, asymmetric critic, curriculum) | JuggleRL used PPO at 4096 envs ([JuggleRL](https://arxiv.org/html/2509.24892)). FlashSAC "slightly outperforms PPO" on low-dimensional tasks ([FlashSAC](https://arxiv.org/html/2604.04539v2)). PPO reached a usable policy sooner in wall-clock than SAC or TD-MPC2 ([Hamed et al.](https://arxiv.org/abs/2607.17326)). FastTD3 produced jerky, "undeployable" motion until smoothness penalties were strengthened ([FastTD3](https://arxiv.org/html/2505.22642v3)). |
| Control rate | **50 Hz**, trained at the deployment rate with ±10% jitter. 30 Hz only if the bus benchmark fails. | robo-ping-pong's SO-101 policy fell from **58.6 bounces at 60 Hz to 5.6 at 30 Hz** when not trained at 30 Hz (self-reported, sim) ([robo-ping-pong](https://github.com/mariapatni/robo-ping-pong)) |
| Actions | Deltas on the five arm joints' position targets, clipped per step and mirrored by LeRobot's `max_relative_target` setting; gripper unpowered | Matches the servo interface. End-effector control via inverse kinematics adds latency, and LeRobot's 10 Hz HIL-SERL stack gives only 3–5 decisions per bounce ([hilserl.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx)) |
| Actor observations | Ball position and velocity from the EKF, in the paddle frame, plus time since the last camera frame; measured joint positions; last commanded targets; **8–10 steps (0.16–0.2 s) of history** | DeepMind fed 8 steps of ball and joint state to small CNNs ([DeepMind 2024](https://arxiv.org/html/2408.03906)). The history should exceed total delay plus mechanical response ([Hwangbo](https://ar5iv.labs.arxiv.org/html/1901.08652)). Without action history, quadrotor trajectory tracking was 0/10 ([Eschmann et al.](https://arxiv.org/html/2311.13081v2)). |
| Critic observations | Everything the actor sees, plus true ball state and spin, restitution, latencies, servo parameters and mount tilt bias | Asymmetric actor-critic ([Pinto et al.](https://arxiv.org/abs/1710.06542)) |
| Network | Small MLP; add a GRU only if the MLP plateaus | DeepMind's low-level policies had ~10k parameters with 3 ms CPU inference |
| Fallbacks | (a) Residual RL on top of an analytic "hit while decelerating" stroke. (b) A once-per-bounce stroke parameterization. | Ploeger juggled for 33 min with a 21-parameter open-loop policy ([Ploeger et al.](https://arxiv.org/abs/2010.13483)) |

**The policy sees a tracked ball, not pixels.** Every dynamic ball system feeds the policy tracked ball state. Every image-based SO-arm success is quasi-static: Squint went 73/80 zero-shot on manipulation ([Squint](https://arxiv.org/html/2602.21203)), and an Isaac-trained SO-101 vision policy "didnt transfer at all" ([Patil](https://indraneelpatil.github.io/blog/2026/hil-serl)).

**Run the same EKF code in simulation.** Feed it simulated detections with noise, latency, frame drops and blur bias, so the actor sees in training exactly what it will see on the Mac.

**Run a scripted baseline first.** Keep the analytic stroke controller as a baseline in sim before RL, because Phase 4 reuses it on the real arm.

### An analytic reward beats any learned reward model here

**No reward model is needed.** In simulation the exact ball state yields a dense analytic reward. On the real arm, zero-shot deployment needs no reward at all; the tracker's consecutive-bounce count is the metric. LeRobot's reward tools target quasi-static tasks:

- the HIL-SERL image classifier, trained on ~200 positive and 1,000 negative frames ([HIL-SERL](https://arxiv.org/html/2410.21845));
- VLM rewards such as TOPReward, built on Qwen3-VL-8B ([topreward.mdx](https://github.com/huggingface/lerobot/blob/main/docs/source/topreward.mdx)).

A bounce is an event lasting milliseconds, not a frame. HIL-SERL's "dynamic" Jenga-whipping result was an open-loop torque command sent at 10 Hz into a 1 kHz controller, which is not comparable. The model actually worth having is the ball detector, and colour thresholding already handles that in about a millisecond.

| Reward term | Form | Why |
|---|---|---|
| Valid-hit bonus | Paid on each contact once the next apex is known, only if the apex lands in a 7–15 cm band and the ball rose more than 4 cm | Paying per bounce let a policy exploit an 8 cm "dribble" ([robo-ping-pong](https://github.com/mariapatni/robo-ping-pong)). JuggleRL paid +50 per hit and +50 per in-band apex. |
| Good-flight time and alive term | Per second of in-band flight, plus +0.5/s | robo-ping-pong's fix for the dribble exploit |
| Centred contact | 10·exp(−5d) for impact distance d from the paddle centre | Removing it cut JuggleRL's real-robot mean from 311 to 55.3 hits |
| Workspace centring | −1.0 × the paddle's distance from home | Removing JuggleRL's lateral penalty caused failure |
| Smoothness | 2·exp(−‖Δa‖) plus a small joint-acceleration penalty | Removing it: **311 → 21.3 hits** on the real system ([JuggleRL](https://arxiv.org/html/2509.24892)) |
| Effort and heat | Penalty on simulated duty cycle² above ~60% (*our proposal*) | 60–95% of stall during bouncing, 70 °C cut-off |
| Ceiling and termination | Terminate if the apex exceeds ~25 cm (keeps the ball in camera view), or the ball falls below the paddle plane, leaves the blade footprint or leaves the workspace; also on joint limits or self-collision | JuggleRL terminated on ground contact or leaving the workspace |

**Curriculum** (*our proposal*, combining two sources):

1. Start with in-flight drops of 10–20 cm over the paddle centre.
2. Widen the drop position in 6 cm rings, unlocking each ring when the current one averages 12 bounces. Filter out unreachable drops; robo-ping-pong's v3 stalled on them.
3. Add ball-at-rest starts at 30–40% of episodes.
4. Add nudges.
5. Ramp the randomization to its full ranges.
6. Throughout, resample start types in inverse proportion to their success rate, as DeepMind did ([DeepMind 2024](https://arxiv.org/html/2408.03906)).

**Optional shaping toward −g/2 paddle acceleration at impact.** It is physically motivated, but no one has tested it with RL, so treat it as an ablation.

### Randomize restitution, lateral start and latency above all

| Parameter | Starting range | Evidence |
|---|---|---|
| Paddle–ball restitution | Measured mean ±0.05–0.075, lower toward the rim | **Without it: 311 → 6.8 hits** ([JuggleRL](https://arxiv.org/html/2509.24892)). +2% ball restitution cost DeepMind 25% ([RSS 2023](https://arxiv.org/html/2309.03315)) |
| Ball start position, height, velocity | ±7 cm laterally, widened by the curriculum | **Without horizontal randomization: 311 → 0.5 hits** |
| Camera latency | Per-episode sample around the measured mean and σ, plus frame drops | Sim at 50% of real latency scored 1.33 against 1.83 ([RSS 2023](https://arxiv.org/html/2309.03315)). Without latency, transfer "completely failed" ([i-Sim2Real](https://arxiv.org/pdf/2207.06572v3)) |
| Action dead time, control-rate jitter | 0–1 extra step; ±10% | robo-ping-pong used 27–33 Hz and a 1-frame lag |
| Ball observation error | Zero-mean noise per frame plus a per-episode bias up to ~1 cm | A 4 cm bias cost DeepMind 36%; zero-mean noise was tolerated |
| Servo model | Gain, armature and friction ±20%; v_max; deadband; backlash 0.5–1° | Fitted model ± its residuals; Robo9 measured 0.87° of backlash |
| Ball mass and drag | 2.5–2.9 g; C_d 0.40–0.52 | [robo-ping-pong design doc](https://github.com/mariapatni/robo-ping-pong/blob/main/docs/reward-design.md) |
| Friction and spin | `solreffriction` and μ ranges; initial spin | Contact spin of ~15 rad/s kicks the next bounce sideways by 0.1–0.2 m/s (*calc*) |
| Random ball pushes | Per-episode probability log-uniform; also human-like taps | Removing pushes cut Dactyl's real-robot median from 13 to 2 ([Dactyl](https://ar5iv.labs.arxiv.org/html/1808.00177)) |
| Paddle mass, CoM, mount tilt; supply voltage; camera extrinsics | ±10%; ±1 cm; ±1°; ±10%; a few mm and ~1° (*our proposals*) | — |

**Anchor the means to your measurements, then randomize around them.** DeepMind found its measured values did worse than tuned ones, and suspected unmodelled spin.

### A side camera at paddle height fixes the weakest sensor

**The C920 is the bottleneck.** It runs at most 30 fps ([Logitech](https://logitech.com/en-za/shop/p/c920-pro-hd-webcam)), and on your Mac its manual exposure does not hold once streaming ([camera_presets.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_presets.py)). Auto-exposure near 1/30 s smears the ball **33–66 mm at impact speed** (*calc*). The estimated photon-to-state latency is **~45–115 ms**, against **~12–30 ms** for a 120 fps global-shutter board (*calc*). Neither has been measured on your Mac; the only anchors are one generic webcam at 108 ms glass-to-glass ([perrytsao](https://github.com/perrytsao/Webcam-Latency-Measurement)) and warnings that the C920's H.264 stream adds latency ([vMix forum](https://forums.vmix.com/posts/m65054-Video-lag-with-Logitech-c920-camera)).

DeepMind's table-tennis system used 125 fps cameras with 4 ms exposure and 40 ms ball latency, and its performance fell sharply at 50 fps ([RSS 2023](https://arxiv.org/html/2309.03315)). The Octo-Bouncer bounced a ping-pong ball "for hours" at 120 fps ([Octo-Bouncer](https://www.electrondust.com/2020/03/01/the-octo-bouncer/)).

**Placement.** Keep the C920 overhead, behind the base, for x/y position and hand gating; overhead x/y resolution is good, ~1–1.5 mm/px. Add the new camera for height:

- slightly above the paddle plane, about 0.5 m from the paddle;
- looking across the paddle from your left or right, perpendicular to the arm's forward axis;
- facing a black backdrop.

An orthogonal pair tolerates unsynchronized USB cameras: each axis comes mostly from one camera. A side-by-side stereo pair with a 16 ms offset would put **3.3 cm of error** on a 2 m/s ball (*calc*).

**The wrist camera is not useful as the primary sensor.** With the paddle mounted it mostly sees the blade edge-on.

**Keep the bounce volume in view.** At 5–20 cm apex the ball moves at most ~80 px in the overhead image (*calc*). Frame the side camera to cover up to ~25 cm, matching the simulator's termination ceiling.

| Camera | fps | Shutter | Mac path | Price | Verdict |
|---|---|---|---|---|---|
| C920 (owned) | 30 | rolling | UVC | — | Overhead x/y and hand gate |
| [Arducam B0332 OV9281](https://core-electronics.com.au/120fps-global-shutter-usb-camera.html) | 120 claimed (one listing says 100) | global | UVC | ~$60 | **Buy**, as the side camera |
| [Arducam B0385 OV9782](https://core-electronics.com.au/120fps-global-shutter-color-usb-camera-board.html) | 120 claimed | global | UVC | ~AU$88 | Colour alternative |
| [PS3 Eye](https://psmoveapi.readthedocs.io/en/latest/camera.html) | 187 at 320×240 | unverified | libusb driver, not AVFoundation | <$10 used | Fallback if the Arducam fails on macOS |

**Inference is fast; latency is the camera's, not the compute's.** Measured locally on the M3 Pro with synthetic frames:

| Step | Time |
|---|---|
| Decode 640×480 MJPEG | 0.57 ms |
| HSV detect + connected components, full frame | 1.17 ms |
| HSV detect inside a predicted region of interest | 0.08 ms |
| Hand detection by frame difference | 0.12 ms |

Source: [bench.py](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bench.py). A small MLP policy adds well under a millisecond (estimate), so a full cycle costs a few milliseconds.

**Perception pipeline:**

1. Grab the newest frame on its own thread, dropping stale buffered frames.
2. Threshold the ball's colour (HSV) inside a region predicted by the filter; fall back to the full frame if the ball is lost.
3. Gate blobs by area and circularity, and take the centroid.
4. Run a ballistic EKF with drag.
5. Detect bounces from the paddle's forward kinematics.
6. Propagate the state forward by the measured latency plus half a control period.

**Calibrate carefully.** Use a checkerboard for intrinsics and a ChArUco or AprilTag board, or forward-kinematics touch points, for extrinsics. Systematic bias, not jitter, is what breaks transfer.

**Hand gate.** Compare each overhead frame against a fixed reference image of the empty board. Subtract a robot mask rendered from the current joint angles; your [camera_check.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_check.py) already renders matching MuJoCo views. Debounce over ~3 frames. Ask yourself to reach in from the sides or the far edge.

- **Avoid adaptive background models (MOG2).** They absorb a stationary hand into the background, and are 25× slower here.
- **Use MediaPipe only as a veto.** It costs ~12–17 ms on a Pixel 6 ([Google](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)) and ~9 ms on an M4, per a third-party README ([fasthands](https://pypi.org/project/fasthands/)).

### One policy launches, bounces and recovers; a supervisor owns the hand

**Your requirements conflict.** "Wait while a hand is in view" contradicts "react immediately to a nudge", because the hand is in view during the nudge.

**Resolve it with an outer supervisor running at ~30 Hz.** Its states run IDLE_HOLD → ARMED → ACTIVE → DROP:

- **IDLE_HOLD:** a plain level-paddle position command, not the policy.
- **IDLE_HOLD → ARMED:** once the ball rests on the paddle and no hand has been seen for ~0.3–0.5 s.
- **ARMED → ACTIVE:** hands control to the policy.
- **Override:** in any state, if the EKF says the ball is airborne above the blade and will land on it within ~0.5 s, the policy runs regardless of the hand.
- **ACTIVE:** ignores hands entirely, and exits only on a drop or a safety trip (temperature, load, joint box).

**Train one policy for launch, bouncing and recovery.** Start 30–40% of episodes from a ball at rest at a random offset on a level paddle, and the rest in flight. Separate launch and bounce policies risk hand-off mismatch: one skill-chaining task rose from 0% to 56% success only after terminal and initial states were regularized ([T-STAR](https://ar5iv.labs.arxiv.org/html/2111.07999)). Ploeger found that starting the cyclic pattern without an initial stroke gave "a bad first throw" ([Ploeger](https://arxiv.org/abs/2010.13483)).

**The launch is the regime the simulator knows least**, because it saturates torque. If the real launch fails while bouncing works, fall back to a scripted drop-and-hit tuned on the arm, and train the bounce policy on that launch's terminal states.

**Train for nudges with taps in simulation.** Use Dactyl-style random pushes, plus human-like taps on a resting ball: 0.3–1.0 m/s vertical and ±0.2 m/s lateral (*our proposal*). Re-acquiring a tapped ball takes ~67–100 ms at 30 fps, and on top of the latency that consumes most of a 286 ms flight. At 120 fps about 200 ms remain (*calc*). That is another reason to buy the camera.

### One H200 GPU covers the training budget

**Training compute is cheap.** JuggleRL trained **2B frames in under 3 hours on one RTX 4090** with 4096 envs ([JuggleRL](https://arxiv.org/html/2509.24892)). An SO-100 RGB grasp needed 25–40M steps, about an hour ([lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)). With 20 physics substeps per 50 Hz policy step, expect **~50k–300k policy steps/s on one H200** (*calc*, unverified), so 1B steps takes ~1–6 h.

| Job | Resources (`sbatch`) | Walltime | Needs your explicit OK |
|---|---|---|---|
| Physics and throughput check (MJX vs Warp, 1k/4k/16k envs) | `-p general --gres=gpu:h200:1` | 1 h | No (I will still state it) |
| Servo-model fit with CMA-ES over all traces | `-p general`, CPU only | 1–2 h | No |
| PPO curriculum run | `-p general --gres=gpu:h200:1` | 4–8 h | Yes if the walltime is long |
| Ablation or seed sweep | 4× H200, or an array | 4–8 h | **Yes** |
| Isaac Lab sim-to-sim check | `--gres=gpu:gb10:1` (DGX Spark) | 2–4 h | No |

**Cluster hygiene follows your existing [ppo-drone/cluster](/Users/praneethsamineni/Documents/GitHub/roboticsexp/ppo-drone/cluster) scripts.** Caches go in `/dev/shm`; results go straight to home through `/nethome-instruction`. The Mac does only the seconds-long fits, perception and deployment.

## Where the notes disagree or rest on thin sources

| Topic | Disagreement | Status |
|---|---|---|
| Does LeRobot write `Acceleration=254`? | The servo note says yes. The low-cost-arms note says unverified, and "not in configure()". | **Verified yes during synthesis** (0.6.1 and 0.6.2). What 254 *means* rests on a third-party register wiki. Measure it in Phase 1. |
| JuggleRL training time | "Under 33 hours" against "under 3 hours" | arXiv HTML doubles digits: it renders "22 billion… 33 hours… RTX 40904090". The correct reading is 2B frames, under 3 h, RTX 4090. The same artifact explains Ploeger's "55 hours", meaning 5 h. |
| Blind Juggler impact acceleration | a = −g/2, decelerating (search snippet), against "accelerating impact" (another note) | Unresolved; the full T-RO text was not read. The decelerating-impact evidence from Schaal and Sternad is independent, so the recommendation stands. |
| Stability window −2g(1+e²)/(1+e)² < a < 0 | Rebuilt from garbled text by one researcher, from memory by two others | Unverified. Treat the −g/2 target as a heuristic. |
| STS3215 top speed | 4.5 rad/s (robo-ping-pong README), 1.66–2.74 (its own design doc), 1.5 (Isaac), 5.07–5.29 (BAM, identified) | Use BAM as the prior and measure. |
| Backlash | "No measurement exists" against ±0.5° (Menagerie, unsourced) and 0.87° (Robo9, 12 V, secondary) | Measure it with an external marker. |
| Register 36 meaning | LeRobot: "25% torque when overloaded". Wiki: the overload *threshold*, default 80%. | Read back the registers. |
| JuggleRL ball | "Mocap marker" against "47.2 g ball, not a ping-pong ball" | Its restitution values do not transfer directly. |

**Single-source or self-reported items:**

- robo-ping-pong: simulation only, self-reported.
- The NII arm-juggling poster: claims zero-shot transfer, gives no numbers ([NII](https://www.nii.ac.jp/event/upload/oh2026/F06.pdf)).
- so101-mjlab's parameters and le_snail's "100 ms behind target" (a non-SO-101 robot): both via a sub-researcher.
- MediaPipe latency on Apple Silicon: a third-party README.
- C920 latency: anecdotes only.
- The Arducam's 120 fps on macOS: unverified.

**The research team's own calculations, not published measurements:** all the bounce kinematics, the bandwidth estimates, the launch and torque tables, the lateral drift, the camera geometry and latency budgets, the flat-paddle Monte Carlo, and the throughput estimate.

**Researcher-run tests (CPU, synthetic frames):** the MuJoCo restitution results and the OpenCV timings. They come from session scratch scripts in [the scratchpad](/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts), which is not permanent. Copy them into the repo to keep them.

## Risks and open questions

| Risk | Early warning | Mitigation |
|---|---|---|
| The servo model cannot match gain and phase at 2.5–5 Hz (BAM was recorded with D=0; LeRobot uses D=32) | Phase 2 gate fails | Actuator network on the logs; P=32; raise the apex to 15 cm |
| The arm cannot launch (no faster-than-g drop) | Phase 1 max-effort test | 7.4 V or 12 V supply; longer contact lever; multi-joint drop; scripted launch |
| Overload or thermal trips | Load above 80% for more than 2 s; temperature above 60 °C | Voltage; short mount; near-vertical forearm pose; effort penalty; 30–60 s runs |
| Bus tail latency | p99 and maximum time in the benchmark | Custom loop with no retries; 30 Hz fallback |
| Tilt precision at the backlash floor | Lateral drift in Phase 4 | Backlash randomization; concave insert in reserve |
| Unmodelled spin and rubber springback | Phase 4 scatter outside the sim's envelope | Fit post-impact noise from real data, as Kulchenko and Todorov did ([ADPRL 2011](https://www.roboti.us/lab/papers/KulchenkoADPRL11.pdf)); analytic impulse model |
| Arm-specific transfer | — | Ploeger's policy "fails immediately" on a second, identical WAM. Re-identify after any hardware change. |

**Open questions only measurement can settle:**

- your servo variant and supply;
- the real effect of `Acceleration=0` and of register 86;
- the firmware control period (register 81) and the filtering on `Present_Velocity`;
- end-to-end latency of the C920 and of the new camera on this Mac;
- your paddle's restitution at the centre and rim, and its post-impact scatter;
- whether MJX and Warp honour `solreffriction`;
- training throughput on GB10.

**Calibrating expectations.** No one has done this on a real SO-101. Reference points from comparable systems:

| System | Result |
|---|---|
| Solo ball-juggling quadrocopter | Averaged 3 hits ([Müller et al.](https://hiperlab.berkeley.edu/wp-content/uploads/2018/05/2011_QuadrocopterBallJuggling.pdf)) |
| Delta robot (Kulchenko and Todorov) | About 25 bounces, with two balls |
| JuggleRL | Averaged 311 hits |

## Conclusion

**This is a measurement problem wearing an RL costume.** The policy itself is cheap: a few H200 hours, with your existing PPO trainer. The outcome is decided by four things:

- whether the servo model reproduces 2.5–5 Hz gain and phase;
- whether the supply voltage allows a faster-than-g drop;
- whether the paddle is bolted rather than gripped;
- whether a camera sees the ball's height with millimetre precision and tens of milliseconds of latency.

**The most informative single experiment is Phase 4: a scripted stroke with a real ball, compared against the same commands in simulation.** If the simulation's randomized spread covers what the real ball does there, RL transfer becomes a matter of training time. If it doesn't, no amount of RL or randomization will close the gap, and the fix is in the servo model, the mount or the camera. A working result would also be the first public RL ball-bouncing demonstration on SO-101 hardware.
