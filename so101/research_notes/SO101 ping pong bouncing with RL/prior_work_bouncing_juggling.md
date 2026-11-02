# Prior work on robot ball bouncing, juggling and paddling (plus RL table tennis), for an SO-101 ping-pong bouncing policy

Scope: classic passive-stability juggling, RL juggling and bouncing from 2020 to 2026, RL table tennis with sim-to-real, low-cost hardware, and reward and curriculum design. Status markers: **[read]** means I read the primary full text, **[abstract/snippet]** means I saw only an abstract, search snippet or secondary coverage, and **[self-reported]** means a non-peer-reviewed GitHub project.

## Q1. Classic and passive-stability approaches: what makes bouncing stable, and which methods need ball sensing?

### Takeaway
Vertical bouncing is passively stable without any ball sensing. Two conditions make it so: the paddle must still be moving upward at impact but **decelerating** (negative acceleration), and the paddle must be slightly **concave**, which handles lateral stability. ETH's Blind Juggler bounced a ball in 3D with no sensors at all, using curvature c = 0.36 1/m and an impact acceleration of −g/2. The mirror-law family, by contrast, needs full ball state from vision.

### Cited Findings
**Blind Juggler (Reist and D'Andrea, ETH; ICRA 2009, IEEE T-RO 28(6):1228–1243, 2012)**
- The robot juggles an unconstrained ball "without feedback at heights of up to 2 m." It does this with a parabolic aluminium paddle driven by a single linear motor. Open-loop stability comes from two design parameters: the paddle curvature and the paddle acceleration at impact. These parameters set the eigenvalues of a linearized one-bounce perturbation map. The model has nine ball states, spin included, and the parameters were optimized against measured process noise. — [T-RO 2012 metadata/abstract](https://manuscript.isc.ac/Inventory/49/1254978.htm); [ResearchGate abstract](https://www.researchgate.net/publication/260635126_Design_and_Analysis_of_a_Blind_Juggling_Robot) **[abstract/snippet]**
- The first prototype used **curvature c = 0.36 (1/m)** and **paddle acceleration a_P = −g/2**. — search snippet of the [ETH-hosted T-RO PDF](https://blindjuggler.ethz.ch/wp-content/uploads/2024/09/tro2012.pdf), which now returns 404 **[snippet only; I could not open the full text to check how "curvature" is defined, so verify]**
- ETH's project page says the machines juggle "without any cameras, microphones, or any other sensors to detect the balls." It describes the paddle as "slightly concave" so the ball does not fall off, and says the paddle motion keeps bounce height consistent. Later variants were the Swinging (pendulum) Blind Juggler and a four-leaf "Cloverleaf" paddle that juggles up to four balls. — [ETH IDSC Juggling Machines](https://idsc.ethz.ch/research-dandrea/research-projects/archive/juggling-machines.html) **[read]**
- Reist and D'Andrea are quoted as saying that the main noise source is "stochastic deviations of the impact parameters, i.e. ball roundness and CR". They also "specifically called out table tennis balls as generating too much noise in the horizontal degrees of freedom at contact." — quoted in [Kulchenko and Todorov, ADPRL 2011](https://www.roboti.us/lab/papers/KulchenkoADPRL11.pdf) **[read]**
- I could not verify the stroke amplitude, stroke frequency or exact paddle dimensions. The blindjuggler.org domain is now a squatted gambling page.

**Negative acceleration at impact (Schaal, Sternad, Atkeson)**
- Schaal and Atkeson (ICRA 1993, pp. 913–918) present case studies of open-loop stable control for dynamic manipulation, including juggling. — [MPI record](https://is.mpg.de/publications/schaal_iicra_1993) **[abstract]**. A later paper describes the 1993 robot as "position controlled rather than using the ball's state as an input." — [Plooij, Wolfslag and Wisse, ICRA 2014](https://filelist.tudelft.nl/3mE/Organisatie/Afdelingen/BmechE/DBL/Plooij_and_Wolfslag_and_Wisse_ICRA2014.pdf) **[snippet]**
- Schaal, Sternad and Atkeson (J. Motor Behavior 1996) show that stable periodic bouncing is "sufficiently characterized by a negative acceleration of the racket at the moment of impact." The task dynamics make such a pattern stable even when the actuator motion is open-loop. — [MPI record](https://is.mpg.de/publications/schaal_jmb_1996) **[abstract/snippet]**
- Period-one bouncing is passively stable when the racket acceleration at impact satisfies **−2g(1+α²)/(1+α)² < r̈ < 0**, where α is the coefficient of restitution. The optimal acceleration, which maximizes the stability margin, "lies in the middle of the range." A Lyapunov analysis predicted values "between −5 and −2 m/s²". In human practice, measured values were all negative and "asymptoted to a value of −4 m/s²". The model simulations used a 0.8 m target height and α = 0.6–0.71. — [Sternad and colleagues' review, PMC3498056](https://pmc.ncbi.nlm.nih.gov/articles/PMC3498056/) **[read; the equation text was garbled in extraction. I reconstructed the inequality and it agrees with the numeric range given for α = 0.71, about −10.1 to 0 m/s²]**
- Sternad, Duarte, Katsumata and Schaal (JEP:HPP 2001) found that human variability was lowest at impact accelerations **between −4 and −1 m/s²**. Haptic information gave more stable performance than visual information alone. — [MPI record](https://is.mpg.de/publications/sternad_jephpp_2001) **[abstract via search; the page was bot-blocked]**
- Ronsse, Lefèvre and Sepulchre (IEEE T-RO 22(1), 2006) extend sensorless stabilization to 2D, using a sinusoidally vibrated "wedge billiard". Orbits that are unstable in an elastic fixed wedge become exponentially stable in the inelastic vibrating wedge. They stabilize only within a restricted amplitude range, and a lower restitution coefficient shrinks the stability region. A period-two orbit ran for "more than 50 impacts" in the lab. — [Ronsse et al. 2006 PDF](https://orbi.uliege.be/bitstream/2268/33349/1/ronsse06.pdf) **[read]**. A follow-up adds impact-time feedback: "Rhythmic Feedback Control of a Blind Planar Juggler." — [ResearchGate](https://www.researchgate.net/publication/3450443_Rhythmic_Feedback_Control_of_a_Blind_Planar_Juggler) **[title only; details unverified]**

**Mirror law (Bühler, Koditschek, Kindlmann ~1990; Rizzi and Koditschek)**
- "Mirror algorithms" are nonlinear feedback laws. The authors adopted them after local controllers based on linearized analysis failed in experiments. The law specifies robot position as a function of the ball's position and velocity: a map from the ball's phase space to robot joint space that generates a reference trajectory. — [Penn repository: A Family of Robot Control Strategies for Intermittent Dynamical Environments](https://repository.upenn.edu/handle/20.500.14332/33535); [Progress in spatial robot juggling](https://repository.upenn.edu/handle/20.500.14332/33587) **[search snippets; pages returned 403]**
- Rizzi and Koditschek's spatial juggler used real-time stereo at 60 Hz under structured lighting. A 1993 paper says the controller needed a ~1 kHz estimate of ball position and velocity, from the CYCLOPS vision system. — same Penn sources **[snippet]**
- Buehler et al. and Rizzi and Koditschek are classed as **closed-loop paddle juggling** (1-ball and 2-ball). Aboaf et al. learned paddle juggling parametrically in **160 attempts**. — [Lee, Wang, Atkeson et al. 2026, Table 1](https://arxiv.org/html/2608.26800v1) **[read]**

**Closed-loop sensing-based bouncers (for contrast)**
- **Quadrocopter ball juggling** (Müller, Lupashin, D'Andrea, IROS 2011) hit a ping-pong ball with a racket on a quadrocopter. Motion capture ran at 200 Hz and commands at 67 Hz. The system estimated drag online (K_D ≈ 0.079 m⁻¹ against 0.06 m⁻¹ calculated) and racket restitution (≈ 0.76, with a central sweet spot). Two-vehicle rallies reached 17 recorded consecutive hits, 160 hits over 24 rallies in 8 min, and ~140 hits unrecorded. **Solo juggling averaged 3 hits (max 14)**, because one vehicle has the least time between impacts. Failure causes were input saturation, unpredictable bounces and tracking errors. — [Müller et al. 2011](https://hiperlab.berkeley.edu/wp-content/uploads/2018/05/2011_QuadrocopterBallJuggling.pdf) **[read]**
- Kulchenko and Todorov (ADPRL 2011) used a 3-DoF Delta haptic robot with a table-tennis paddle in a tiny 0.36 m × 0.30 m cylinder workspace, with Vicon at 240 Hz. With hand-crafted parameters the real robot bounced **two balls "only for about 25 bounces."** They fitted per-axis Gaussian noise on post-impact ball velocity from real data and injected it in simulation. — [Kulchenko and Todorov 2011](https://www.roboti.us/lab/papers/KulchenkoADPRL11.pdf) **[read]**

### Inferences
- **Which approaches need ball sensing.** None do for the Blind Juggler (fully blind), Schaal and Atkeson's open-loop paddle, or the Ronsse wedge. Impact time only is used by the Ronsse 2007 follow-up and possibly a closed-loop Blind Juggler extension, both unverified. Full ball position and velocity is required by the mirror law (60 Hz stereo), the quadrocopter (200 Hz mocap) and Kulchenko (240 Hz Vicon).
- **The stability band for a ping-pong ball (my arithmetic).** With paddle restitution α ≈ 0.85, the period-one band is about **−9.9 < a_impact < 0 m/s²**. Its midpoint, ≈ −4.9 m/s² ≈ −g/2, matches the Blind Juggler's choice of −g/2. For any α from 0.6 to 1 the band runs from about −1.0g to −1.1g up to 0. Aiming at roughly −g/2 is therefore a robust default.
- **Paddle speed needed at impact (my arithmetic).** For a steady vertical bounce of apex height h above the paddle, the paddle speed at impact is u = √(2gh)·(1−α)/(1+α). For h = 0.30 m: α = 0.85 gives u ≈ 0.20 m/s, and α = 0.75 gives ≈ 0.35 m/s.
- **A sinusoidal stroke that meets both conditions.** For h = 0.30 m the bounce period is ≈ 0.49 s. Hitting u = 0.2 m/s and a = −4.9 m/s² needs an amplitude of ≈ 3.4 cm (≈ 7 cm peak-to-peak), a peak paddle speed of ≈ 0.43 m/s and a peak acceleration of ≈ 5.5 m/s². These are modest demands, the same order as what hobby servos can deliver; see Q4.
- **Lateral sensitivity (my arithmetic).** A 1° paddle tilt adds ≈ (1+α)·√(2gh)·sin θ ≈ 0.08 m/s of horizontal ball velocity at h = 0.3 m, about **4 cm of drift per bounce per degree**. Paddle orientation error from servo backlash or deadband is therefore a first-order lateral disturbance. Curvature, or a learned tilt correction, has to cancel it.
- The ETH note that ping-pong balls are especially noisy horizontally argues for strong lateral randomization in simulation and for measuring post-impact velocity scatter on the real paddle, as Kulchenko did.

### Gaps
- I could not read the Blind Juggler's stroke amplitude, frequency, paddle diameter, how curvature is defined, or its consecutive-bounce and duration statistics, because the full text could not be fetched. Get the T-RO PDF via IEEE or the library.
- I did not confirm whether any Reist and D'Andrea version used impact-timing feedback (for example a microphone). The ETH page says no sensors.
- The Bühler–Koditschek 1990/1994 papers were not read in full, so the mirror-law gains, performance and hit counts are unverified.

## Q2. RL and learning-based juggling and bouncing (2020–2026): designs, rewards and results

### Takeaway
Only two real-robot results are close to "RL keeps a ball bouncing on a paddle": **JuggleRL** (2025, a quadrotor with a racket, PPO in Isaac Sim, zero-shot transfer, up to 462 consecutive hits) and the self-reported, simulation-only **robo-ping-pong SO-101 keep-up** project. The strongest real-arm juggling result, Ploeger et al. 2020, deliberately used an **open-loop** policy with only a binary reward and learned on hardware rather than in simulation. Both successful real systems had either a passively stabilizing end-effector or carefully calibrated restitution with randomization.

### Cited Findings
**Ploeger, Lutter, Peters: "High Acceleration RL for Real-World Juggling with Binary Rewards" (CoRL 2020)** — [arXiv 2010.13483](https://arxiv.org/abs/2010.13483) **[read]**
- The robot is a 4-DoF Barrett WAM juggling 2 balls, with accelerations "of up to 8g". The balls are 75 mm Russian-style, partly sand-filled with 37.5 g so they do not bounce, caught in a funnel end-effector with a 170 mm opening that "passively compensates" for trajectory errors. A launcher 3 m above the floor releases the second ball.
- The policy is **open-loop**: a Gaussian over joint-space via-points (position, velocity, duration), interpolated with cubic splines and tracked by a low-gain PD controller with gravity compensation. Symmetry and zero via-point velocity cut it from 48 to **21 parameters**. It was initialized by hand-setting the via-points, because kinesthetic teaching cannot reach the accelerations.
- The authors explain why they avoided closed-loop control: they "do not dare to apply a real-time closed-loop network policy" at 8g. A naive task-space adaptation to observed balls "even reduced system performance".
- The reward is binary, r = 1 while both balls are ≥ 0.6 m above the floor (measured by OptiTrack, used for the reward only). The optimizer is episodic REPS with an added reverse-KL constraint, KL bound 2, and 25 rollouts per update; 10 rollouts often converged to bad optima in simulation.
- **Results:** 20 episodes × 25 rollouts took **56 min of experience**, and the authors say "learning still takes up to 5 hours" of wall-clock time. The final policy averaged 106.82 s out of a 120 s cap, against 66.51 s for hand-tuning, and ran for up to **33 min (~4500 catches)**.
- **Sim-to-real lessons:** learning in MuJoCo "is actually more challenging" than on the real robot, because "the stabilizing passive dynamics … could not be modeled accurately". The learned policy "fails immediately when transferred to a different Barrett WAM." OptiTrack tracking suffered from occlusion and from wear on the reflective tape.

**JuggleRL: quadrotor with racket (2025)** — [arXiv 2509.24892](https://arxiv.org/html/2509.24892) **[read]**
- **Observation (24-D):** quadrotor position, rotation matrix and velocity, plus ball position, velocity and relative position. The ball is a mocap marker with Kalman-filtered velocity. Mocap runs at 120 Hz and the policy at **50 Hz**. **Action:** collective thrust plus body rates, executed by a PID.
- **Reward, summed with unit weights:**
  - hit bonus +50 when contact height is inside a band;
  - apex bonus +50 when the apex lands in a band ([2.85, 3.2] m in simulation, [1.6, 1.85] m on the real robot);
  - relative-position term 1/(1+max(d_xy, 0.2));
  - centre-contact term 10·exp(−5·d_axis) on hits;
  - lateral penalty −1.0·d_quad;
  - smoothness term 2·exp(−‖a_t − a_{t−1}‖);
  - yaw penalty −10·|ψ|.
- **Termination:** ball touches the ground or leaves the workspace; 500-step cap.
- **Calibration and randomization:** racket restitution was measured by drop tests at 0.82 in the 5.5 cm sweet spot and ≈ 0.64 outside it. Domain randomization covered restitution U[0.75, 0.90], initial ball height U[1.5, 2.0] m and initial XY offsets ±0.07 m. **Drag was not modelled.**
- **Training:** PPO in OmniDrones/Isaac Sim, 4096 environments, ~2 billion frames on one RTX 4090. The paper says "under 33 hours", which may be a typo.
- **Real results:** fixed 1.65 m release averaged **311 hits** (max 414); varied 1.5–1.95 m release averaged 403.6 (max **462**). The model-based baseline (Müller et al.) averaged 3.1 hits (max 14).
- **Real ablations (mean hits, against 311):**

| Removed | Mean hits |
|---|---|
| Horizontal randomization | **0.5** |
| Restitution randomization | **6.8** |
| Height randomization | 20.3 |
| Smoothness reward | 21.3 |
| Contact reward | 55.3 |
| Position reward | 58.7 |
| Lateral or yaw penalty | failure |

**Other learning-based juggling (real robots)**
- Ploeger and Peters: 5-ball cascade, open-loop, regularized memory-based learning in **1–2 attempts**. Lee, Wang, Atkeson, Rizzi and Rojas (RAI Institute/CMU, arXiv Aug 2026): closed-loop 3-ball cascade with multi-fingered hands in **7–10 attempts**, averaging 53 s of physical interaction for the cascade. Their perception is **30 Hz RGB plus time-of-flight with ~0.1 s latency**, plus parabolic flight fitting. Without their reachability constraint, 89% of planner solutions were unsafe. — [arXiv 2608.26800](https://arxiv.org/html/2608.26800v1) **[read]**
- Older learning-based paddle juggling, as classified in Ploeger's Table 1: imitation (Kober et al. 2010), model learning (Aboaf et al. 1989) and evolutionary strategies with dense fitness (Schaal and Sternad 1993). — [Ploeger et al. 2020](https://arxiv.org/abs/2010.13483) **[read]**
- Ball-in-a-cup: SAC-X learned it on a real Sawyer "from raw pixels within three days" in a fully automated, safety-engineered cell. — [Ploeger et al. summarizing Schwab et al. 2019](https://arxiv.org/abs/2010.13483); [Schwab et al. arXiv 1902.04706](https://arxiv.org/abs/1902.04706) **[read via Ploeger; primary abstract via search]**
- Kendama: a UR5e learned to catch the ball in a kendama cup with RL in simulation and transferred to the real arm. — [arXiv 2003.06751](https://arxiv.org/pdf/2003.06751v1.pdf) **[snippet]**. Diabolo: an analytical diabolo model was built for learning in simulation and run on a real dual-arm robot. — [arXiv 2011.09068](https://arxiv.org/pdf/2011.09068.pdf) **[snippet]**
- DribbleBot (legged soccer dribbling): PPO with ~4000 parallel simulated robots, sim-to-real. — [arXiv 2304.01159](https://arxiv.org/abs/2304.01159); [Robohub](https://robohub.org/a-four-legged-robotic-system-for-playing-soccer-on-various-terrains) **[secondary; reward terms not extracted]**

### Inferences
- Two designs work for real-world bouncing and juggling. One is open-loop with passive stability plus episodic tuning on hardware (Ploeger; Blind Juggler). The other is closed-loop with accurate ball state plus heavy randomization of restitution and lateral initial conditions (JuggleRL). JuggleRL's ablations show that **restitution and horizontal randomization matter more than any reward term**.
- JuggleRL's reward structure carries over to a paddle on an arm almost unchanged:
  - sparse hit bonus;
  - sparse apex-in-band bonus;
  - centred-contact bonus;
  - lateral-distance penalty;
  - action-smoothness term, whose removal cost 93% of performance on the real system.
- Ploeger found that simulation lacked the stabilizing passive dynamics. A concave paddle may help in the real world even if the simulation does not reproduce it faithfully, but that is a hypothesis to test.

### Gaps
- I found no peer-reviewed paper (2020–2026) on **RL ball bouncing on a paddle with a fixed-base arm** transferred to hardware. I also found no learned legged or humanoid "keepy-uppy". Searches turned up dribbling, shooting and goalkeeping, not juggling.
- "Task-Error Residual Learning for Real-Robot Five-Ball Juggling" ([arXiv 2606.16978](https://arxiv.org/pdf/2606.16978)) turned up in a search, but I did not read it.
- I did not extract DribbleBot's exact reward terms or randomization ranges, because the PDF exceeded the fetch limit.

## Q3. Table tennis with RL and sim-to-real: ball physics, latency and perception rates

### Takeaway
The Google/DeepMind table-tennis line (i-Sim2Real 2022, RSS 2023 case study, "human-level" 2024) shows four things. **Simulating measured latency is essential**: without it "sim-to-real transfer completely failed". Matching the latency distribution beats minimizing it. Policies are highly sensitive to restitution: +2% on ball restitution cost 25% of performance. Small per-step policies (~1k–10k parameters) at 50–100 Hz on 8-step histories of ball and joint state were enough.

### Cited Findings
**i-Sim2Real (Abeyruwan et al., CoRL 2022)** — [arXiv 2207.06572](https://arxiv.org/pdf/2207.06572v3) **[read]**
- **Robot:** ABB IRB 120T plus a 2-axis Festo gantry (8 DoF).
- **Observation:** an 8×11 history (ball xyz plus 8 joints over 8 steps).
- **Action:** 8 joint velocities at **75 Hz**, from a 3-layer 1-D dilated gated CNN. Simulation is in PyBullet.
- **Ball model:** ẍ = g − K_d‖ẋ‖ẋ with **m = 0.0027 kg, C_d = 0.47, ρ = 1.29 kg/m³, A = 1.256×10⁻³ m²**. Spin is not modelled. Restitution and friction were estimated empirically.
- **Noise and latency:** uniform ball-observation noise of **2× the ball diameter** per step. "We also found it necessary to simulate sensor latency, otherwise sim-to-real transfer completely failed"; latencies are per-component Gaussians measured on the robot.
- **Reward:** sparse evaluation reward of +1 for a hit and +1 for landing on the human side (max +2), and −2 on a robot fault. The simulation training reward adds:
  - landing-near-target bonus up to 5;
  - net-clearance bonus;
  - episodic jerk, acceleration and velocity "proxies for faulting" (weights 0.3/0.3/0.4);
  - joint-angle safety term (weight 1);
  - self/table collision penalty;
  - paddle-height and style penalties (simulation only).
- **Result:** cooperative rallies of **22 hits on average and 150 at best**. For 80% of players, rallies were 70–175% longer than the sim-to-real baseline.

**Robotic Table Tennis: A Case Study into a High Speed Learning System (D'Ambrosio et al., RSS 2023)** — [RSS PDF](https://www.roboticsproceedings.org/rss19/p006.pdf); [arXiv 2309.03315](https://arxiv.org/abs/2309.03315) **[read]**
- **Measured latency** (µ ± σ, ms):

| Component | Latency (ms) |
|---|---|
| Ball observation | 40 ± 8.2 |
| ABB observation | 29 ± 8.2 |
| Festo observation | 33 ± 9.0 |
| ABB action | 71 ± 5.7 |
| Festo action | 64.5 ± 11.5 |

  In simulation, each component's latency is sampled per episode and observations and actions are linearly interpolated to it.
- **Policy rate and size:** "20Hz – 100Hz have been explored, but 100Hz is used for most experiments". Policies are ≈ 1k-parameter gated dilated CNNs over the past 8 joint and ball states; a 13M-parameter LSTM also ran at 60 Hz.
- **Cameras:** 125 FPS at 1280×1024, 4 ms exposure, 388 µs camera latency.
- **Latency ablation:** the baseline (measured) latency scored 1.83/2.0 zero-shot. The next best, **50% of measured latency, scored 1.33**. At 0%, 20% and 150% performance was "very poor" and inconsistent.
- **Perception thresholds:** performance stayed at baseline until a "heavy dropoff at **50 FPS and 150 ms**". Zero-mean noise degraded performance gracefully. A non-zero-mean bias of 4 cm (one ball diameter) caused a 36% drop in reward.
- **Physical sensitivity:** removing table-restitution randomization cost 14%. **+2% ball restitution cost 25%**, and +8% table restitution cost 36%. Measured restitution and mass values did *worse* than tuned ones; the authors suspect unmodelled spin.
- **Training scale:** BGS (an evolution-strategy method) for 10,000 iterations, about 6 billion environment steps.

**Achieving Human-Level Competitive Robot Table Tennis (D'Ambrosio et al., 2024)** — [arXiv 2408.03906](https://arxiv.org/html/2408.03906) **[read, partial]**
- **Robot:** ABB IRB 1100 on 4 m × 2 m Festo gantries.
- **Low-level skill policies:** joint velocities at **50 Hz** from an 8-step history of ball position and velocity, joints and style. Each is a ~10k-parameter CNN with 3 ms CPU inference.
- **Perception:** 2 Ximea cameras at **125 Hz**.
- **Simulation:** MuJoCo with a fluid drag model. The paddle rubber is modelled as orthogonal spring-damper joints, and restitution damping is bimodal for topspin and underspin. Actuator parameters were system-identified per joint.
- **Transfer:** zero-shot, with randomization of observation noise, latency and table/paddle damping and friction. A 2.8k-parameter FiLM adapter corrected topspin returns.
- **Reward:** return rate plus shaping for net height, paddle-joint angle at contact and a reference pose.
- **Training:** BGS, ~2.4B steps per generalist. Ball categories were resampled in inverse proportion to their return rate.
- **Result:** won 45% of 29 matches, including 100% against beginners, 55% against intermediates and 0% against advanced players.

**Other table-tennis work**
- Büchler et al. (T-RO 2022) used pneumatic muscle arms. The reward is on returned-ball speed, with no safety constraints in the RL algorithm, because the muscles' compliance keeps the robot safe. In hybrid sim-and-real training (HYSR), recorded balls are replayed virtually while the real robot acts. The robot learned to return and smash real balls without touching a real ball during training. — [arXiv 2006.05935](https://arxiv.org/abs/2006.05935); [DFKI record](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/13546) **[abstract]**
- HITTER (2025) put a model-based ball-trajectory and strike planner on top of an RL whole-body controller on a humanoid, reaching **up to 106 consecutive shots** against a human. — [arXiv 2508.21043](https://arxiv.org/abs/2508.21043v2) **[abstract via search; the Unitree G1 detail and the 96.2% hit rate are from an AI-summary site and unverified]**
- Sony AI "Ace" (Nature 652:886–891, 2026):
  - perception: 9 active-pixel cameras for ball 3D position, plus 3 event-camera gaze-control systems for spin;
  - control: model-free RL;
  - spin: ">75% return rate up to spins of 450 rad/s";
  - matches: won 3 of 5 against elite players. — [AIhub](https://aihub.org/2026/04/22/sony-ai-table-tennis-robot-outplays-elite-human-players/) **[secondary]**
  - Reports conflict on the professional matches: one report says it lost both but won a game, while [heise](https://heise.de/-11269518) says 1 of 7 matches was won. A snippet-level claim of a 32 ms (31.25 Hz) policy period and 200 Hz tracking with 10.2 ms latency is **unverified**: the full Nature text was paywalled.
- TuneNet tuned ball restitution from real drop videos, then used the simulator to choose a bounce shot for a Kinova Jaco: 62% success. — [arXiv 1907.11200](https://arxiv.org/pdf/1907.11200) **[snippet]**

### Inferences
- These systems all run per-step closed-loop RL. Their recipe is to measure latency per component and simulate it as a sampled distribution, and to randomize restitution while anchoring its mean to measurement. Pairing a low-dimensional state of ball position plus joints over an 8-step history with a tiny network was enough. The SO-101's tens of milliseconds of latency is the same order as the 29–71 ms per component in the RSS system.
- On perception rate: table tennis tolerated frame rates down to ~50 FPS when the ball crosses the table in ~0.4 s, which is ~20 frames per approach. A 0.3 m bounce has a ~0.5 s flight, so 30 FPS gives ~15 frames per cycle (my arithmetic). That is near the threshold but plausibly workable, especially since the 2026 RAI juggler worked with 30 Hz / ~0.1 s-latency vision. The ball moves ~8 cm per frame near impact at 2.4 m/s, so the policy must predict, not react.
- Bias in ball position (triangulation or calibration offset) hurt far more than zero-mean noise. Camera extrinsic calibration therefore matters more than detector jitter.

### Gaps
- I did not extract the 2024 DeepMind paper's numeric simulated latency values; it cites the RSS method. Reward weights were not stated in the parts I read.
- I could not verify the Sony Ace control rate, latency or simulation details from the full Nature text.

## Q4. Dynamic and ball tasks on SO-100/SO-101 and other hobby-grade hardware

### Takeaway
I found **no peer-reviewed or verified real-hardware SO-101 ball-bouncing result**. The closest is the self-reported, simulation-only **robo-ping-pong** keep-up policy for an SO-101: PPO with curriculum and randomization, 34.8–62.8 mean bounces in simulation, with results collapsing when control dropped from 60 Hz to 30 Hz unless the policy was trained at 30 Hz. Hobby machines do bounce ping-pong balls reliably: T-Kuhn's stepper-driven Octo-Bouncer bounces for "hours" with a 120 FPS camera.

### Cited Findings
**robo-ping-pong (mariapatni), SO-101 keep-up in MuJoCo** — [GitHub README](https://github.com/mariapatni/robo-ping-pong) **[self-reported; simulation only; phase 1, step 2; no hardware bounce results]**
- **Model and servos:** the SO-101 model is vendored from mujoco_menagerie. STS3215 actuators use kp = 998.22, kv = 2.731, force range ±2.94 N·m. The paddle (150 mm) replaces the gripper on the wrist-roll horn.
- **Ball:** 40 mm and 2.7 g, with manual quadratic drag at C_d = 0.47 and no spin in v1.
- **Simulation timestep:** **2 kHz integration was needed**, because "at 200 Hz the solver injects energy and the ball leaves the room."
- **Observation (33-D):** joint positions, commanded targets and velocities; ball position and velocity relative to the paddle face; face normal; face position relative to the centre; last action; control period.
- **Action:** 5 joint-position targets with a 4.5 rad/s slew limit (said to be STS3215 top speed).
- **Reward, paid per second of good flight:**
  - apex between **20 and 35 cm**;
  - travelling toward the centre, then straight up there;
  - landing near the centre;
  - catch bonus;
  - **+0.5/s alive**.
  - A bounce counts only after the ball rises **4 cm**, so "buzzing a resting ball earns nothing".
- **Curriculum and filtering:** drops start at the centre and widen in **6 cm rings**. A ring unlocks when drops from the current ring average **12 bounces**. Drops are filtered to those the arm can reach in time.
- **Randomization:** ball mass and drag; paddle restitution (also decaying per hit); 2° takeoff scatter; servo gains; **control rate 27–33 Hz**; sensor noise with **1-frame lag**; mid-flight lateral nudges.

| Run | Change | Self-reported result (sim) |
|---|---|---|
| v1 | First policy, old paddle, centre drops | 62.8 bounces, but **an 8 cm "dribble"** exploit, because the reward paid per bounce |
| v2 | Physics and sensor randomization | **58.6 bounces at 60 Hz, 5.6 at 30 Hz** |
| v3 | New paddle, ~30 Hz, per-second scoring, curriculum | **34.8 bounces**; half of episodes reached the 20 s cap; stalled on unreachable drops |
| v4 | Catchable drops only | "training" (no result) |

- The project lists open issues for its next version (v5). The ball model should use hollow-shell inertia, 2/3·m·r², rather than a solid sphere, which made spin "about 1.7x too easy". The paddle's assumed 0.85 restitution is "the most important unmeasured number in keep-up". Reach is ~67% of the half-table, out to x = 0.48 m. — [README](https://github.com/mariapatni/robo-ping-pong)
- The same repo's rally design document lists SO-101 velocity constraints of **1.92, 1.75, 1.69, 1.66 and 2.74 rad/s** (at 0.85× limit) and current constraints of 1.8 A instantaneous and 0.9 A RMS. It also lists a randomization table: paddle restitution U[0.78, 0.90], C_d U[0.40, 0.52], ball mass U[2.5, 2.9] g, kp U[700, 1200], kv U[1.9, 3.6], 0–1 step action dead time, ball observation noise of 5 mm plus 1-step latency. — [docs/reward-design.md](https://github.com/mariapatni/robo-ping-pong/blob/main/docs/reward-design.md) **[design doc, no runs]**. **Conflict:** these joint velocity limits are well below the README's 4.5 rad/s, and the source of either figure is not stated.

**STS3215 servo specs (vendor, no-load)**
- 12 V / 30 kg·cm variant: **0.222 s/60°** (≈ 4.7 rad/s), 30 kg·cm stall, 2.7 A stall current. — [RobotShop](https://robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215)
- 7.4 V / 19.5 kg·cm variant: 0.238 s/60° at 6 V (≈ 4.4 rad/s). — [Core Electronics](https://core-electronics.com.au/catalog/product/view/sku/CE09383) **[vendor listings, not measured under load]**

**Other SO-101 and hobby dynamic tasks**
- SO-101 ball throwing into a basket with ACT, from 30 teleoperated demos recorded at 30 Hz: 96–100% at 20 cm and 47–60% at 40 cm. No speed figures are given. — [nxp0999 repo](https://github.com/nxp0999/Robotics-SO-101-Imitation-Learning) **[self-reported]**
- **Octo-Bouncer** (T-Kuhn):
  - drive: 4 NEMA-17 steppers with 5:1 planetary gearboxes on a flat 8 mm acrylic plate;
  - camera: See3CAM_CU135 at **120 FPS**, 640×480, with OpenCV Hough-circle detection;
  - control: predicts the hit position, starts the upward stroke when the ball is within a set distance, and tilts the plate for lateral correction;
  - result: bounces a ping-pong ball for "hours";
  - noted problems: noisy frame-to-frame velocity at high FPS, ~5° usable tilt due to joint play, and missed steps without a gearbox. — [electrondust Octo-Bouncer](https://www.electrondust.com/2020/03/01/the-octo-bouncer/) **[self-reported]**
  - The 2026 six-arm successor, Hexa-Bouncer, uses 2 cameras; no bounce count is stated. — [PopSci](https://www.popsci.com/technology/hexa-bouncer-over-engineered-robot/)
- A vision-based table-tennis juggling robot (Ji et al., *Mathematics* 12(11):1634, 2024) uses **two 250 FPS industrial cameras**, optimization-based and PD controllers analyzed with Poincaré maps, and controls bounce height. — [RePEc](https://ideas.repec.org/a/gam/jmathe/v12y2024i11p1634-d1400277.html); [MDPI](https://www.mdpi.com/2227-7390/12/11/1634) **[abstract; MDPI returned 403, so the bounce count is unknown]**

### Inferences
- The kinematics are probably not the bottleneck (my arithmetic, from Q1). A 30 cm bounce needs ≈ 0.2–0.35 m/s of paddle speed at impact and ≈ 5 m/s² of deceleration. At a ~0.25 m lever arm that is ≈ 1–2 rad/s of joint speed, within even the conservative 1.66–2.74 rad/s limits.
- The binding constraints are more likely these four:
  - timing and latency, because the ball covers ~8 cm per 30 FPS frame near impact;
  - the servo's internal position-loop lag and backlash, which turn into paddle-tilt error of ~4 cm lateral drift per degree;
  - the 30 Hz command rate;
  - restitution uncertainty.
- The robo-ping-pong 60 Hz → 30 Hz collapse (58.6 → 5.6) shows that **the policy must be trained at the deployment rate with rate jitter**. That matches the table-tennis finding that latency must be matched, not just minimized.
- A cheaper fallback suggested by the classic work: a fixed periodic stroke with a concave paddle. Amplitude about 3–4 cm, with the phase set so impact happens at about −g/2. A small learned or feedback correction on top would adjust tilt and phase. This puts the task inside SO-101 limits and needs ball sensing only for the corrections. It is a hypothesis to test.

### Gaps
- I found no measured SO-101/STS3215 step response, loaded speed or command-to-motion latency in the sources, so the arm's tens-of-ms latency figure is unverified here.
- I found no LeRobot-community video or blog of an SO-100/SO-101, Koch or ALOHA-class arm bouncing or juggling a ball on real hardware.
- I did not find C920 end-to-end capture latency.

## Q5. Recommended reward shaping and curricula for bouncing: what does the evidence support?

### Takeaway
Across the systems here, the setups that worked share a pattern. They give a sparse reward for each valid hit or apex inside a height band, plus dense centring and lateral terms, plus an action-smoothness term and an alive or time term. They gate bounce credit to prevent "dribbling" exploits and terminate when the ball falls below the paddle or leaves the workspace. They start from easy, centred drops and widen the spread on a success gate. Randomizing restitution and lateral initial conditions matters as much as the reward itself.

### Cited Findings
**Reward terms that were tested**
- Height-band apex bonus (+50), hit bonus (+50), centred-contact bonus, lateral penalty, smoothness and yaw. Removing smoothness dropped real performance from 311 to 21.3 hits, and removing the lateral penalty caused failure. — [JuggleRL](https://arxiv.org/html/2509.24892) **[read]**
- Paying per bounce let the policy exploit an 8 cm dribble (62.8 "bounces"). The fix was to pay per second of good flight with an apex band of 20–35 cm, count a bounce only after the ball rises 4 cm, and add +0.5/s alive. — [robo-ping-pong](https://github.com/mariapatni/robo-ping-pong) **[self-reported]**
- A binary "alive" reward (both balls above 0.6 m) works only with a good **expert initialization**, because "most random initialization would not make any contact with the balls". The authors chose binary rewards to avoid hand-tuning a target ball trajectory that may not fit the robot. — [Ploeger et al. 2020](https://arxiv.org/abs/2010.13483) **[read]**
- An MPC cost for paddle bouncing used a target post-contact ball velocity at contact, a centring term ‖p_xy‖² (w = 100), a paddle height target (w = 36000) and paddle-under-ball alignment ‖p_xy − b_xy‖² (w = 100). Optimizing the few hand-crafted weights first, then adding more, worked better than optimizing 345 parameters at once, which overfit. — [Kulchenko and Todorov 2011](https://www.roboti.us/lab/papers/KulchenkoADPRL11.pdf) **[read]**
- Safety and smoothness proxies (episodic jerk, acceleration, velocity, joint-angle and collision penalties) were used in simulation and swapped for an actual fault penalty on the real robot. — [i-Sim2Real Table 5](https://arxiv.org/pdf/2207.06572v3) **[read]**

**Curricula and initial-state distributions**
- Centre drops first, widened in 6 cm rings once the current ring averages 12 bounces. Drops were filtered to reachable ones after v3 stalled on unreachable drops. — [robo-ping-pong](https://github.com/mariapatni/robo-ping-pong) **[self-reported]**
- No explicit curriculum, but the initial ball height was randomized over U[1.5, 2.0] m with ±7 cm XY offsets. Removing height randomization dropped performance from 311 to 20.3 hits. — [JuggleRL](https://arxiv.org/html/2509.24892) **[read]**
- Ball categories resampled in inverse proportion to their success rate ([DeepMind 2024](https://arxiv.org/html/2408.03906)). Anchoring the ball distribution to the real world matters, but precision does not ([RSS 2023](https://www.roboticsproceedings.org/rss19/p006.pdf)). **[read]**

**Termination**
- Ball touches the ground or leaves the workspace, with a 500-step cap. — [JuggleRL](https://arxiv.org/html/2509.24892)
- Episodes capped at 20 s (half of v3 episodes reached it). — [robo-ping-pong](https://github.com/mariapatni/robo-ping-pong)

**Sim-to-real techniques that measurably mattered**

| Technique | Evidence |
|---|---|
| Restitution randomization around a measured mean | Without it: 311 → 6.8 hits ([JuggleRL](https://arxiv.org/html/2509.24892)) |
| Lateral/initial-state randomization | Without it: 311 → 0.5 hits ([JuggleRL](https://arxiv.org/html/2509.24892)) |
| Measured per-component latency in sim | Without it transfer "completely failed" ([i-S2R](https://arxiv.org/pdf/2207.06572v3)); 50% latency 1.33 vs 1.83 ([RSS](https://www.roboticsproceedings.org/rss19/p006.pdf)) |
| Train at deployment control rate with jitter | 58.6 bounces at 60 Hz vs 5.6 at 30 Hz ([robo-ping-pong](https://github.com/mariapatni/robo-ping-pong), self-reported) |
| Fine simulation timestep for a 2.7 g ball | 200 Hz injects energy; 2 kHz used ([robo-ping-pong](https://github.com/mariapatni/robo-ping-pong)) |
| Post-impact velocity noise fitted from real data | Kulchenko and Todorov fitted per-axis Gaussians from 240 Hz Vicon ([ADPRL 2011](https://www.roboti.us/lab/papers/KulchenkoADPRL11.pdf)) |
| Ball observation noise (uniform, 2× diameter) | Used in [i-S2R](https://arxiv.org/pdf/2207.06572v3); zero-mean noise tolerated, bias not ([RSS](https://www.roboticsproceedings.org/rss19/p006.pdf)) |
| Online drag/restitution estimation | K_D ≈ 0.079 m⁻¹, racket restitution ≈ 0.76 ([Müller 2011](https://hiperlab.berkeley.edu/wp-content/uploads/2018/05/2011_QuadrocopterBallJuggling.pdf)) |

### Inferences
- **Reward recipe.** Synthesized from JuggleRL and robo-ping-pong; not validated together.
  - (1) Sparse bonus per paddle contact, credited only if the next apex is inside a band (for example 20–35 cm above the paddle) and the rise exceeds a few cm.
  - (2) Dense lateral-distance penalty, from ball and paddle to the workspace centre.
  - (3) Contact-near-paddle-centre bonus.
  - (4) Action-rate/smoothness penalty.
  - (5) Small alive-per-second term.
  - (6) Terminate when the ball falls below the paddle plane by a margin, or leaves the workspace.
  - Avoid paying per bounce without an apex gate, which invites the dribble exploit.
- **Curriculum.**
  - Drop the ball from a low height (about 15–30 cm) directly above the paddle centre.
  - Widen the lateral spread and height range on a success gate, such as mean bounces ≥ N.
  - Always filter out unreachable drops.
  - Optionally tighten the apex band over training.
- **Physics to measure on the real rig before training.**
  - Paddle–ball restitution, which RSS showed is very sensitive and robo-ping-pong flagged as its key unknown.
  - Post-impact lateral scatter, which ETH flagged as large for ping-pong balls.
  - Command-to-motion latency and camera-to-observation latency.
  - Then randomize around these values instead of guessing.
- **Passive-stability prior.** The literature supports giving the policy an inductive bias toward impacting at negative paddle acceleration (about −g/2), and possibly using a concave paddle. One option is a reward term or action parameterization around a periodic stroke. Neither has been tested with RL; it is an inference from Q1.

### Gaps
- No source directly compares reward variants for paddle bouncing on an arm beyond JuggleRL's quadrotor ablations and robo-ping-pong's self-reported runs.
- No source tested a concave paddle in an RL simulation-to-real pipeline.
- Whether open-loop stroke parameterization outperforms per-step joint-target policies on low-bandwidth servos is untested in the literature I found.
