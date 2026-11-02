# Domain-randomisation critique: SO-101 ping-pong bouncer

Date: 2026-10-30. Scope: what the simulator does **not** vary, varies too narrowly, or varies the wrong way, and how other projects vary the same things. I read the code on 2026-10-30 while env.py was being edited, so line numbers may drift. The developer's in-flight additions (tilt, wind, per-impact scatter, rim falloff, wobble, supply sag, USB glitches, jitter, extrinsic rotation, blur lag, outliers, timestamp noise, start spread) are already in the code. I comment on them only where the chosen range or model looks wrong.

Provenance tags: **[code]** read in this repo. **[calc]** my own arithmetic; the scripts are in the session scratchpad. **[paper]** I opened the source in this session. **[notes]** taken from the project research notes in [so101/research_notes](../../so101/research_notes/SO101%20ping%20pong%20bouncing%20with%20RL/), which I did not re-verify. **[memory]** a well-known paper cited from memory, not re-read. **[U]** unverified.

## Bottom line

1. **Timing is the crux, and the sim tells the policy the truth about timing.** Detection ages are the true capture ages plus zero-mean noise. On the real arm the age has to be estimated from arrival time, and that estimate carries a per-session bias. Fix this first.
2. **Several "nominal" values describe hardware you do not have, or settings you have not chosen.** The camera nominal is 60 fps (the C920 tops out at 30). Camera z-noise is 3 mm, which needs a side camera. Supply voltage is 6.2 V, between the kit's 5 V and 7.4 V. The servo acceleration range excludes the cap implied by LeRobot's Acceleration=254. The model's servo is also ~1.8× stronger than the datasheet. The "nominal" suite and the early curriculum all train on these values.
3. **The biggest physics gaps are in the gearbox and the grip, not the ball.** No backlash, no load-dependent friction, no grip slip, and no compliant-mount dynamics. The new paddle-wobble term is a noise process whose velocity depends on the time step: it injects ~6.6 cm/s of surface-velocity noise, comparable to the stroke itself.
4. **The camera error model is in the wrong coordinates.** It uses independent world-axis Gaussians. The real error lies mostly along the camera ray, changes with blur and occlusion, and comes in bursts.
5. **The randomisation method:** one global level, uniform boxes, independent parameters, no real-data gate. That is a known weakness compared with per-parameter ADR and posterior-based DR. It is fixable cheaply once the sysid logs exist.

## 1. Prioritised gaps

Impact is my judgement of the effect on zero-shot transfer. Ranges in the "Proposed" column are starting points to replace once measured.

| # | Factor | Currently | What others do | Proposed range, distribution, correlation | Impact and why | Where in code |
|---|---|---|---|---|---|---|
| 1 | **Detection age bias** (estimated vs true capture time) | Ages are true capture time + per-frame zero-mean noise ≤3 ms; availability uses the true latency. [env.py:503](../bounce/env.py#L503), [env.py:519-527](../bounce/env.py#L519). The real side needs `t_capture` ([obs.py:36](../bounce/obs.py#L36)), which a webcam does not give exactly. | DeepMind samples each latency component **per episode** from a Gaussian at the **measured** μ±σ (ball obs 40±8.2 ms). Policies trained at 50% or 150% of the true latency transferred much worse **[paper]** ([DeepMind 2023](https://arxiv.org/abs/2309.03315)). OpenAI adds a per-episode correlated term to every observation **[paper]** ([Rubik's Cube 2019](https://arxiv.org/abs/1910.07113), App. B.4). | Per-episode age bias b ~ U(−15, +15) ms if latency is LED-measured, ±30 ms if not. Add b to every reported age, independent of the true latency. Keep the per-frame term. | **High.** At 1.4 m/s, 15 ms is 2.1 cm of extrapolation error **[calc]**. Impact timing is already the binding constraint (servo lag 74–143° in the bounce band **[notes]**). | `_camera`: `stamp = t + bias_env + noise`. |
| 2 | **Camera error geometry** | Gaussian per world axis: σ_xy 0.5–5 mm, σ_z 0.5–20 mm. σ is constant per episode, i.i.d. per frame, with no depth-scale error. [env.py:48-49](../bounce/env.py#L48), [env.py:496-498](../bounce/env.py#L496). | A 4 cm bias cost 36% of reward, while doubling zero-mean noise barely mattered **[paper]** ([DeepMind 2023](https://arxiv.org/abs/2309.03315)). OpenAI's o·n₀+n₁+n₂ includes a per-episode multiplicative term **[paper]** ([Rubik's Cube](https://arxiv.org/abs/1910.07113)). | Simulate in camera coordinates: project the true ball into the deployed camera(s); add σ_uv ~ logU(0.3, 1.5) px and radius σ_r ~ logU(0.2, 1.0) px, inflated by blur length / ball radius; back-project with **nominal** intrinsics and extrinsics. Add a per-episode focal-length/ball-radius scale error of ±3% (depth = f·R/r). Take σ from BallNet's residuals. | **High.** Overhead depth-from-diameter gives σ 1.3–2.7 cm along a ray ~60° below horizontal **[notes]**, so z and horizontal errors are strongly correlated. A per-axis model teaches the policy the wrong filter. | [env.py:493](../bounce/env.py#L493). Reuse `project()` from [make_detector_data.py:79](../scripts/make_detector_data.py#L79). |
| 3 | **Gear backlash / lost motion** | Not modelled. There is only a firmware dead zone of 0–0.005 rad ([servo.py:31](../bounce/servo.py#L31), [servo.py:85](../bounce/servo.py#L85)). | Dactyl and Rubik's Cube randomise a motor-backlash model per episode **[paper]** ([Dactyl 2018](https://arxiv.org/abs/1808.00177), [Rubik's Cube](https://arxiv.org/abs/1910.07113) B.2). Menagerie defines ±0.5° but never applies it **[code]** ([so101.xml:41-43](../assets/robotstudio_so101/so101.xml#L41)). A bench test measured 0.87° on a 12 V STS3215 **[notes]** ([Robo9](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)). | Total lost motion U(0.2°, 1.0°) per joint per episode. Model it as a dead zone between motor-side and link-side angle (two-inertia joint, armature on the motor side). | **High.** At wrist_flex, the dynamic stroke torque (0.8–0.9 N·m) exceeds the gravity preload (0.36–0.53 N·m) **[notes]**, so the gears reverse every bounce. 0.5° gives 7–14 mm of landing drift at 10–20 cm apex **[notes]**, plus ~17 ms of reversal dead time at 0.5 rad/s **[calc]**. | Substep loop [env.py:384-395](../bounce/env.py#L384), plus a `q_motor` state in `ServoBank`. |
| 4 | **Load-dependent gearbox friction** | Coulomb + viscous only (BAM model m1), [servo.py:92-93](../bounce/servo.py#L92). | BAM's richer STS3215 fits (m4/m6) include load friction ≈0.21–0.24 × transmitted torque **[notes]** ([BAM](https://github.com/Rhoban/bam)). | f = −(f_c + μ_L·\|τ_load\|)·tanh(q̇/0.02) − f_v·q̇, with μ_L ~ U(0, 0.3) per joint. Alternatively, make BAM m6 the nominal. Check BAM's exact definition before coding it. | **High** for shoulder and elbow. A 0.8–1.1 N·m gravity load adds 0.2–0.27 N·m of friction, 4–5× the Coulomb term **[calc]**. That changes small-stroke tracking, sag hysteresis and the hold command q_ref. | `friction(dq, tau_g)`, called at [env.py:390](../bounce/env.py#L390). |
| 5 | **Paddle slipping/rotating in the jaws** | Mount offsets are fixed per episode ([env.py:257-268](../bounce/env.py#L257)). | No direct precedent. Dactyl applies random forces to the object for unmodelled effects; removing all unmodelled-effect randomisations dropped the median real-robot result from 13 to 2 **[paper]** ([Dactyl](https://arxiv.org/abs/1808.00177)). | Fix it mechanically (clamp or adapter), then measure. If the stock gripper stays: a stick-slip joint about the handle axis, breakaway torque U(0.08, 0.3) N·m, cumulative within the episode. | **High** with the stock gripper. The jaw friction couple is ~0.08–0.26 N·m against the 0.22–0.43 N·m needed **[notes]**. LeRobot's 25% gripper overload setting can also trip after ~2 s of stall **[notes]**. The drift is non-stationary, so randomising it is no substitute. | Per-substep update of `kin.paddle_rot`. |
| 6 | **Acceleration register** | log₁₀ a_max ~ U(2.2, 3.9), i.e. 160–8000 rad/s² ([servo.py:28](../bounce/servo.py#L28)). This excludes the ~39 rad/s² cap implied by Acceleration=254, which [arm.py](../real/arm.py) restores as its default. | — | Choose the deployed value and verify it in sysid. Centre the range on the measurement. If 254 stays, the range must include ~39 rad/s². | **High** for launch: braking harder than g needs 39–73 rad/s² at 0.25–0.135 m levers **[notes]**. Medium for steady bouncing. | `SERVO_RANGES`; the real runtime must write the register. |
| 7 | **Peak torque** | BAM's kt/R give 2.86 N·m stall at 6.2 V, against a datasheet-scaled 1.60 N·m ([servo.py:21-22](../bounce/servo.py#L21) vs [servo.py:95-97](../bounce/servo.py#L95)) **[calc]**. The overload check uses the datasheet figure ([env.py:396](../bounce/env.py#L396)). | BAM itself warns that kt and R are a fitted pair **[notes]**. | Measure max effort with the paddle on. Fit kt/R jointly. Sample stall torque and back-EMF damping as two correlated quantities rather than kt and R independently. | **Med-High.** Launches and recoveries may rely on peak torque the real servo does not have. | [servo.py](../bounce/servo.py). |
| 8 | **Dropout structure** | i.i.d. Bernoulli, 0–15% per frame ([env.py:50](../bounce/env.py#L50), [env.py:504](../bounce/env.py#L504)). | Dactyl masks or freezes markers to simulate occlusion **[paper]** ([Dactyl](https://arxiv.org/abs/1808.00177)). | Two-state (Gilbert–Elliott) dropout with mean burst 1–4 frames. Add geometric occlusion by the wrist/gripper "view shadow" (≈0.58·h on the face **[notes]**) from the deployed camera pose, and a miss probability that rises with blur length. | **Med-High.** Real misses cluster near impact (fast ball, next to the paddle), which is exactly when timing matters. | `_camera`. |
| 9 | **Compliant mount vs "wobble" noise** | OU process on face angle, τ = 20 ms, σ ≤ 6 mrad. Its rate enters the impact as surface velocity ([env.py:404-408](../bounce/env.py#L404), [ball.py:56-58](../bounce/ball.py#L56)). | DeepMind models paddle rubber as two orthogonal passive spring-damper joints **[paper]** ([DeepMind 2024](https://arxiv.org/abs/2408.03906)). | Replace it with a 2-DOF damped oscillator (pitch/roll): f_n ~ logU(8, 40) Hz, ζ ~ U(0.05, 0.3), driven by the paddle's acceleration and by impacts. Measure with a tap test. | **Med-High.** The current term injects 6.6 cm/s of normal surface velocity at 5 cm off centre (2 ms substeps), 9.4 cm/s at 1 ms. That is a step-size artefact ~4× the physical σ/τ, and as large as the 7–20 cm/s stroke speed needed at 10 cm apex **[calc]**. | Same lines; add oscillator state. |
| 10 | **Joint-reading latency** | q is read instantly at the end of the step ([env.py:473-475](../bounce/env.py#L473)). | DeepMind randomises robot-observation latency (29±8.2 ms) separately from action latency (71±5.7 ms) **[paper]**. Isaac Lab's `DelayedPDActuator` resamples an integer delay per reset **[code]** ([actuator_pd.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py)). | q_obs delay U(0, 10) ms via a substep ring buffer, centred on the `sysid.py check` read times. | **Med.** The FK paddle pose and the stroke phase the policy reads from q history are skewed. | New ring buffer in `step`. |
| 11 | **Tangential rubber response** | The impulse drives contact slip to rolling (grip fraction α = 0.4), Coulomb-capped ([ball.py:72-74](../bounce/ball.py#L72)). | Sony AI fits a tangential restitution e_t ≈ 0.82, giving α = (1+e_t)/(1+κ) ≈ 0.73 **[notes]** ([Sony AI 2026](https://arxiv.org/abs/2606.28805)). Gossard 2026: constant-parameter bounce models underperform across rubbers **[notes]** ([arXiv 2604.11349](https://arxiv.org/abs/2604.11349)). | jt = −α·m·v_t (capped by μ·j_n), α ~ U(0.35, 0.75) per episode, correlated with rubber type (row of §3.4). Sony's fit is at table-tennis speeds; extrapolation to 1–2 m/s is [U]. | **Med.** α sets the spin carried into the next bounce: 0.1–0.2 m/s of lateral kick per 0.5 m/s of slip **[notes]**. | [ball.py:72](../bounce/ball.py#L72). |
| 12 | **Blade flatness / curvature** | Flat face; only i.i.d. per-impact kicks ([ball.py:77-80](../bounce/ball.py#L77)). | The blind juggler uses paddle curvature for lateral stability **[notes]** (Reist & D'Andrea). | Per-episode curvature κ_u, κ_v ~ U(−0.2, 0.2) m⁻¹: sagitta ≤ 0.6 mm, slope ≤ 0.6° at 5 cm **[calc]**. Measure with a straightedge and narrow the range. | **Med.** This is a systematic, position-dependent tilt of the same size as the 0.4–1.4° needed to recentre the ball **[notes]**. A convex face destabilises. I.i.d. kicks cannot represent it. | Local normal in `paddle_impact`: n − κ_u·du·u − κ_v·dv·side. |
| 13 | **Per-frame latency variation** | Camera latency is fixed per episode ([env.py:319](../bounce/env.py#L319)); frame period jitter is 3% ([env.py:510](../bounce/env.py#L510)). | Peng 2018 varies the timestep every step around a per-episode rate **[paper]** ([arXiv 1710.06537](https://arxiv.org/abs/1710.06537)). OpenAI combines per-episode and per-step delay factors **[paper]**. | latency = L₀ + k·T_frame, with k ∈ {0,1,2} per frame and P(k ≥ 1) ~ U(0, 0.3) per episode (driver buffering). Frames stay in order. | **Med.** Webcam buffering adds 0–2 frames **[notes]**. | Store a per-detection latency at capture; use it in `_obs`. |
| 14 | **Detector: robot colour** | The arm is always Menagerie yellow (1, 0.82, 0.12) ([so101.xml:67-79](../assets/robotstudio_so101/so101.xml#L67)); [make_detector_data.py:26-64](../scripts/make_detector_data.py#L26) never changes it. Its hue (~0.14) sits inside the "yellow ball" range 0.12–0.16 ([visual.py:19](../bounce/visual.py#L19)). | ManiSkill/lerobot-sim2real randomise robot colour **[code]** ([grasp_cube.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py)). Squint fell from 73/80 to 58/80 real successes without colour jitter **[notes]**. | Random PLA palette for printed parts (orange, yellow, white, black, red, grey), with ~25% weight on the real arm's colour. Servos dark grey to black. | **High** (for the detector). The real arm colour may never have been seen, and these parts sit next to the ball. | `build()`: set the material rgba. |
| 15 | **Detector: backgrounds and textures** | Flat-colour floor and walls, no textures, six primitive distractors ([make_detector_data.py:37-62](../scripts/make_detector_data.py#L37)). | Random textures on everything **[memory]** ([Tobin 2017](https://arxiv.org/abs/1703.06907)). lerobot-sim2real overlays a greenscreen of the real background **[code]**. Cut-paste compositing **[memory]** ([Dwibedi 2017](https://arxiv.org/abs/1708.01642)). | 50% procedural textures (noise, checker, wood, gradient). 30–50% composites over ~200 real frames from the deployed camera (arm in random poses, no ball), using the segmentation mask. | **High** (for the detector). This is the largest single appearance gap: real desks have cables, papers, a laptop, shadows. | `build()` / `postprocess()`. |
| 16 | **Detector: camera artefacts** | No lens distortion, no roll (a MuJoCo free camera has none), no defocus, no rolling shutter, i.i.d. Gaussian noise, native 320×240 render ([make_detector_data.py:22](../scripts/make_detector_data.py#L22), [:67-76](../scripts/make_detector_data.py#L67), [:97-106](../scripts/make_detector_data.py#L97)). | Dactyl randomises intrinsics, contrast and per-pixel noise **[paper]**. ManiSkill SO100: camera ±2.5 cm, FOV ±2° **[code]**. | Radial k₁ ~ U(−0.25, 0.05), or the calibrated C920 model. In-plane roll ±5°. Defocus σ 0–1 px. Poisson–Gaussian noise plus mild sharpening. LED flicker banding at short exposures. Render at 640×480 and downscale with the deployment code. | **Med.** | `postprocess()`, `camera()`. |
| 17 | **Detector: hands, glare, second ball** | Two finger-sized skin capsules; ellipsoids are forced non-round; no glossy rubber glare; no second ball ([make_detector_data.py:49-62](../scripts/make_detector_data.py#L49)). | Distractor randomisation is standard **[memory]** ([Tobin 2017](https://arxiv.org/abs/1703.06907)). | Hand plus forearm shapes (8–10 cm palms, partly occluding). Randomise material specular/shininess. Either render an occasional second ball as background, or make "one ball in view" an operating rule. | **Med.** Hand taps put a hand next to the ball. | `build()`. |
| 18 | **Taps decoupled from perception** | Taps are pure velocity impulses ([env.py:409-412](../bounce/env.py#L409)); outliers and dropouts are independent of them ([env.py:499-504](../bounce/env.py#L499)). | — | When a tap fires, raise dropout to 30–70% and the outlier rate 5–10× for 0.2–0.6 s around it. | **Med.** Recovery after a tap is when perception is worst. | `step` / `_camera`. |
| 19 | **Rolling resistance on rubber** | None: a resting ball rolls on any tilt ([ball.py:10](../bounce/ball.py#L10)). | — | Equivalent angle U(0, 0.5°) [U]. Measure by tilting the paddle. | **Low-Med.** With level error up to 0.86° ([env.py:78](../bounce/env.py#L78)), the sim ball rolls ~4 cm in 1 s **[calc]**, while tacky rubber may hold it. That changes rest starts. | Rest handling in `paddle_impact`. |
| 20 | **Thermal state** | Not modelled; kt and R are drawn independently ([servo.py:21-22](../bounce/servo.py#L21)). | — | Latent T ~ U(25, 65) °C: R×(1+0.0039ΔT), kt×(1−0.001…0.002·ΔT) (magnet type [U]). This replaces the independent kt/R draws. | **Low-Med.** STS3215 reached 71 °C in continuous oscillation **[notes]**. | `ServoBank.sample`. |
| 21 | **Link masses** | ±10% uniform, independent per body; rotational inertia is not rescaled ([env.py:254-255](../bounce/env.py#L254), [robot.py:208](../bounce/robot.py#L208)). | Dactyl 0.5–1.5× **[paper]**. Isaac Lab now uses ±25% **log-uniform** ("scale-invariant … geometric mean 1.0") **[code]** ([velocity_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py)). One SO-arm rebuild weighed +20% vs sim **[notes]**. | A common print factor logU(0.8, 1.25) × per-part logU(0.95, 1.05). Servos ±3%. Scale inertia with mass. | **Low.** Armature dominates, and gravity sag is relevelled. | `_reset`. |
| 22 | **Episode-start arm state** | Exact steady state: q̇ = 0 and servo target = q_ref ([env.py:288-289](../bounce/env.py#L288)). | Isaac Lab resets joints with a position scale of 0.5–1.5 **[code]**. | q̇ ~ N(0, 0.05 rad/s), servo target ±0.5°, and sometimes start mid-stroke. | **Low.** | `_reset`. |
| 23 | **Global randomisation level** | One scalar scales ~45 ranges plus the task difficulty. The gate is survival of episodes with d ≥ 0.7·level; the level is capped at 1 ([env.py:575-586](../bounce/env.py#L575)). | OpenAI ADR uses per-parameter bounds, boundary sampling p_b = 0.5, buffer m = 240, Δ = 0.02 **[paper]**. Static DR is "usually inferior" to adaptive schemes, which need target-domain data **[paper]** ([Muratore 2022](https://www.frontiersin.org/articles/10.3389/frobt.2022.799893/full)). | See §2(f): group-level ADR, plus one-at-a-time boundary evaluations. | **Med.** Hurts diagnosability and leads to over- or under-randomisation. | `update_curriculum`, `_reset`. |
| 24 | **Evaluation gates** | Three suites at d ∈ {0, 1} ([evaluation.py:6-10](../bounce/evaluation.py#L6)). | DeepMind sensitivity sweeps **[paper]**. Humanoid-Gym sim-to-sim **[notes]**. lerobot-sim2real warns that checkpoints with equal sim scores differ on hardware **[notes]**. | Real-log replay gate, sim-to-sim, detector-in-the-loop, one-at-a-time sweeps (including 1.5× latency), 60 s episodes. | **High** as a process: it is the only way to know the DR covers reality. | [evaluation.py](../bounce/evaluation.py). |

## 2. By area

### (a) Physics and dynamics

- **The ball model is the strongest part.** e(v), rim falloff, per-impact scatter, drag, Magnus, tilt and wind all match or exceed what the table-tennis and juggling work does. The e0 range [0.72, 0.92] is wider than JuggleRL's [0.75, 0.90] **[paper]** ([JuggleRL](https://arxiv.org/abs/2509.24892)).
- **Restitution must be measured, not just widened.** JuggleRL anchors its range on a measured value; removing restitution DR took real hits from 311 to 6.8 **[paper]**. In DeepMind's table-tennis study, +2% ball restitution cost 25% of reward **[notes]**. Bouncing is closed-loop per bounce and so more forgiving, but a range this wide still buys a conservative policy.
- **Missing contact physics:** tangential springback (row 11), position-dependent face normal (row 12), rolling resistance (row 19), and ball–gripper/handle collisions (the ball passes through the handle region). The missing reaction impulse on the arm is negligible (~0.5% of effective paddle mass **[notes]**).
- **The arm is where reality and the sim differ most:** backlash (row 3), load friction (row 4), grip slip (row 5), mount compliance as dynamics (row 9), masses (row 21).
- The level solve now uses tilted gravity ([env.py:279](../bounce/env.py#L279)), which is consistent with levelling by phone. Table tilt of ±1.5° per axis is generous but harmless.

### (b) Actuation, servo, bus, timing

- **Good:** the BAM structure (rate limit, saturation, back-EMF, armature), per-joint parameters, supply sag with total current (per substep, [servo.py:87-88](../bounce/servo.py#L87)), command delay plus per-step jitter, stale-command glitches.
- **Gaps:**
  - The acceleration range excludes 254 (row 6).
  - The model's torque ceiling vs the datasheet (row 7).
  - No joint-reading latency (row 10).
  - No thermal coupling (row 20).
  - Overload protection triggers on torque / datasheet stall ([env.py:396](../bounce/env.py#L396)). The firmware may instead use duty (Present_Load is duty) or current **[notes]**, and on overload the real servo drops to ~20% torque rather than ending the episode. Treat this as [U] until sysid logs show what trips it.
- **Glitches:** glitch_p is U(0, 3%) per step, so the mean at full difficulty is 0.75 glitches/s, which is probably high. Use log-uniform 10⁻⁴–10⁻² per step. Raise the rate with total current, since LeRobot notes corrupted packets "especially when several joints move at once" **[notes]**. A real LeRobot timeout blocks ≥50 ms (~3 steps; 5+ with retries **[notes]**), so 1-step freezes model USB scheduling hiccups, not timeouts.
- **Comparison:** legged work randomises motor strength by ±5% and delay 0–10 ms (Humanoid-Gym **[notes]**); Peng 2018 randomised gains 0.5–2× and the per-step timestep **[paper]**. Your ranges are wider, which is justified only until sysid exists (§2f).

### (c) Sensing, perception and the observation pipeline

- Rows 1, 2, 8 and 13 are the core issues: the age bias, the geometry, the bursts and the per-frame latency.
- **Camera rotation error:** `cam_rot_err` rotates about the paddle home point ([env.py:497](../bounce/env.py#L497)). A real extrinsic rotation error pivots about the camera centre, which also translates the home point. At 0.75 m, 1.5° is 2 cm **[calc]**, against `cam_bias` ±8 mm. That is acceptable if you calibrate with a board at the paddle, but say so explicitly.
- **Blur lag** (±10 ms per episode, [env.py:69](../bounce/env.py#L69)) should follow from exposure and the timestamp convention (mid-exposure vs frame start), and from fps (row 2 / §3.4). It should not be independent.
- **Missing, low priority:** rolling-shutter row timing (~2.5 ms per 10 cm of height **[calc]**), and camera shake from desk vibration correlated with the strokes [U].
- **Observation history is too short for the randomisation.** Four steps (80 ms) of joints/actions and four detections cannot identify a delay of up to 40 ms, servo strength, or e. The policy is therefore forced to be robust rather than adaptive.
  - Peng 2018 needed an LSTM **[paper]**; RMA reads 0.5 s **[notes]**.
  - Chen et al. show memory matters for DR **[paper]** ([arXiv 2110.03239](https://arxiv.org/abs/2110.03239)).
  - Evaluate the `hist12` variant, or longer, before narrowing ranges.

### (d) Task, initial state and disturbances

- **Start distribution:** rest starts out to 75% of the blade, flight starts ±7 cm, spin ±20 rad/s and taps are all in line with JuggleRL. JuggleRL's ±7 cm horizontal DR was its single most important term (311 → 0.5 hits without it) **[paper]**.
- **Gaps:** taps are decoupled from perception (row 18); the arm starts at exact steady state (row 22); 8 s episodes hide slow drift (lateral creep, heating, slip), so add a 60 s suite. A human dropping the ball from 20–40 cm is not covered either: starts are 3–18 cm at ±1 m/s ([env.py:305-309](../bounce/env.py#L305)) and anything above 30 cm terminates. Train it or rule it out.
- **The single difficulty scalar also scales tap rate, spread and spin** ([env.py:299-312](../bounce/env.py#L299), [env.py:325](../bounce/env.py#L325)). The curriculum therefore mixes "harder task" with "wider DR".

### (e) Appearance randomisation for the detector

- **Already good:** procedural paddles and rubber colours, ball hues, lights, camera families, white balance/gamma/noise/JPEG, segmentation-based visibility labels, partial-occlusion modes.
- **Gaps, in order:** fixed robot colour (row 14), no textures or real backgrounds (row 15), camera artefacts (row 16), hands/glare/second ball (row 17).
- **Two smaller issues:**
  1. Each image re-aims the camera at paddle + 8 cm ±4 cm ([make_detector_data.py:76](../scripts/make_detector_data.py#L76)), so the ball is almost always near the image centre. The deployed camera is fixed. Aim at the hold pose with the deployment pose jittered, so ball pixels cover the whole frame.
  2. Exposures ≥4 ms use exactly 5 sub-frames ([make_detector_data.py:178](../scripts/make_detector_data.py#L178)). At 33 ms and 3 m/s that leaves 2.5 cm between ghost copies of a 4 cm ball **[calc]**, a stroboscopic look the real sensor never makes. Use n_sub ≥ blur_px / (r_px/2), and blur the paddle too.
- **The detector has no real-image validation.** Hand-label ~200 frames from the deployed camera. Report recall, false-positive rate and centre/radius error by ball speed. Feed those residuals into row 2's noise model.

### (f) The randomisation method itself

**Global ADR level.** Yes, this is a known weakness relative to per-parameter ADR.

- OpenAI keeps two bounds per parameter. They evaluate one bound at a time (p_b = 0.5), buffer 240 results, and expand or shrink each bound independently **[paper]** ([Rubik's Cube](https://arxiv.org/abs/1910.07113)).
- Here, one scalar moves ~45 ranges, the tap rate and the start spread together. The consequences:
  1. The worst parameter (e.g. 40 ms delay at 4.5 V with a 0.22 kg paddle) caps every other range.
  2. You cannot tell which parameter is the bottleneck.
  3. The gate is measured on episodes where every range is near its widest at once, so joint corner cases set it.
  4. The level is capped at 1, so this is a curriculum toward hand-picked bounds, not "automatic" DR. It cannot discover that a range is too narrow.
- **Cheap fix:** give each group (servo, timing, camera noise, ball/rubber, mount/paddle, disturbances) its own level and boundary-test buffer, and separate task difficulty from DR width. At every evaluation, sweep each parameter to lo, hi and 1.5× hi with the others at nominal, DeepMind-style **[paper]**.

**Uniform vs log-uniform.**

- Many ranges span 4–40× (cam_noise_z, delays, latency, dropout, glitch, μ, fps). Uniform sampling puts most mass at the top end: at full difficulty, P(σ_z > 5 mm) = 0.77, P(delay > 16 ms) = 0.67 and P(fps > 60) = 0.63 **[calc]**.
- Dactyl uses log-uniform for damping and gains **[paper]**; Isaac Lab now uses log-uniform for mass **[code]**.
- Use log-uniform for scale parameters. Once measured, use a truncated normal around the measurement (DeepMind's latency model **[paper]**).

**Per-episode vs per-step.** Muratore notes that episodic randomisation is the norm without theoretical justification, and that per-step randomisation adds variance **[paper]**. Choose by physics: per episode for hardware constants and the age bias; per frame or step for latency buffering, dropout bursts, glitches and supply voltage (already per step); within-episode processes for grip slip and temperature. The i.i.d. per-impact seam scatter is acceptable as a first cut, although real seams are fixed to the ball and coupled to spin.

**Correlations.** Independent boxes are a superset of the correlated reality, so they are conservative rather than invalid. Their cost is capacity spent on impossible combinations. They are wrong only where a correlation carries information the policy should use. See §3.4 for the specific couplings.

**Posterior-based DR once sysid exists.**

- Fit the servo model to the [sysid.py](../real/sysid.py) logs (least squares or CMA-ES, bootstrapped over segments for a covariance), then sample from that posterior inflated ~2× instead of uniform boxes. This is the BayesSim / DROPO idea **[paper]** ([BayesSim](https://arxiv.org/abs/1906.01728), [DROPO](https://arxiv.org/abs/2201.08434)), and it captures the kt–R–armature correlations automatically. SimOpt ([arXiv 1810.05687](https://arxiv.org/abs/1810.05687)) and BayRn ([arXiv 2003.02471](https://arxiv.org/abs/2003.02471)) close the loop with real rollouts **[memory]**.
- Xie et al. transferred a quadruped with no dynamics randomisation once the model was accurate **[paper]** ([arXiv 2011.02404](https://arxiv.org/abs/2011.02404)): accuracy can substitute for width, not vice versa.

**Evaluation gates before hardware**, in order:

1. **Real-log replay** of the sysid goal streams through `ServoBank`: the real trace must lie inside the ensemble's 5–95% band (gain/phase error <15%/15° at 2.5–5 Hz, the notes' proposal).
2. **Sim-to-sim** in full MuJoCo contact physics, to catch exploits of the analytic impulse model (wobble artefact, REST_SPEED).
3. **Detector-in-the-loop:** renders at the camera rate through BallNet, to check the analytic camera noise.
4. **One-at-a-time sweeps** (including 1.5× latency) and 60 s episodes.
5. **Ball-free hardware replay** of policy commands, checking tracking and temperature.

## 3. Varied, but probably the wrong way

1. **Nominals centred on unverified or unowned hardware.**
   - cam_fps 60, but the C920 maxes at 30 **[notes]** ([env.py:46](../bounce/env.py#L46)).
   - cam_noise_z 3 mm, which only a side camera gives ([env.py:49](../bounce/env.py#L49)).
   - cam_latency 35 ms, against a C920 estimate of 45–115 ms **[notes]**.
   - vin 6.2 V. Reality is bimodal (5 V brick or 7.4 V supply) and readable from Present_Voltage ([servo.py:30](../bounce/servo.py#L30)).
   - The acceleration range excludes 254 (row 6).
   - The "nominal" suite ([evaluation.py:7](../bounce/evaluation.py#L7)) therefore measures a robot that does not exist.
2. **Asymmetric ranges shift the mean.** Samples are nominal + d·(U(lo, hi) − nominal) ([env.py:112-114](../bounce/env.py#L112)), so at d = 1 the mean is the range centre, not the nominal. Delay averages 22 ms vs a nominal of 8; camera latency 57.5 ms vs 35.
3. **Paddle wobble as an OU process on angle** feeds a step-size-dependent velocity into impacts (row 9).
4. **Independent where coupled:**
   - **Camera profile:** fps, exposure (blur), latency and noise are hardware-coupled. Today the sim can draw "120 fps with 100 ms latency and a 10 ms blur lag". Sample a camera profile, then jitter it.
   - **Rubber type:** e0, μ, rim falloff and α. The code comments describe one joint "worn/pimpled vs fresh/tacky" variable ([ball.py:21-23](../bounce/ball.py#L21)).
   - **Servo constants:** kt and R are a fitted pair; temperature moves R, kt and friction together.
   - **Paddle:** mass, blade size, CoM and grip reach.
   - **Glitches:** their rate depends on load.
5. **Wrong time scale or noise structure:** per-episode constants that drift within a real run (grip, rows 5/13/20); i.i.d. events that are really bursty or state-dependent (rows 8, 18); truth leaking into the detection age (row 1); overload on datasheet torque while the model can exceed it ~1.8× (row 7); linear, independent masses (row 21); detector framing and 5-copy blur (§2e). The critic also gets `diff` ([env.py:537](../bounce/env.py#L537)), which is fine, as long as it never reaches the actor.

## 4. Measure on the real arm first

Randomisation cannot stand in for these measurements. Each one either resolves a bimodal unknown or sets a nominal value the whole distribution hangs on.

1. **Servo variant and supply.** Read the servo label (C001 7.4 V vs 12 V) and Present_Voltage at rest and under the hold load. This collapses vin from 4.5–8.4 V to ±5%.
2. **Acceleration register.** Decide the deployed value, then measure max acceleration and braking with the paddle on, at both 254 and 0 (sysid `run` covers both tunings).
3. **Per-joint frequency response at 2.5–5 Hz** at the deployed P/D, with the paddle mounted, plus a max-effort step to get the real torque ceiling (rows 4, 7).
4. **Backlash and compliance at the paddle face.** Film small reversals at 240 fps, or tape a phone IMU to the blade and do a tap test for f_n and ζ (rows 3, 9).
5. **Grip.** Draw a mark across the jaw and handle, run 2 minutes of strokes and check for creep. If it moves, clamp it; do not randomise it (row 5).
6. **Camera.** Actual fps and exposure on macOS, and end-to-end latency μ/σ (LED test, ~50 trials). Find out which timestamp the capture path really exposes (rows 1, 13).
7. **Detector on ~200 real labelled frames:** recall, false positives, centre/radius error by speed, and dropout burst lengths (rows 2, 8, 14-17).
8. **Ball–paddle COR at the centre and the rim, and per-bounce scatter.** Drop from 30 cm, film at 240 fps. Add one oblique or spinning drop for the tangential response (rows 11 and the e0/edge/e_noise ranges).
9. **Paddle mass, CoM, blade dimensions and flatness.** Kitchen scale, knife edge, ruler, straightedge. This shrinks the 0.08–0.22 kg mass range and the shape ranges to ±5% (row 12).
10. **Bus timing and error rate** from `sysid.py check`, over minutes and under load. This sets glitch_p, ctrl_jitter and the joint-read latency (row 10).

After these, re-centre every nominal on the measurement. Narrow the ranges to the measurement uncertainty times 2–3, and keep wide ranges only for what genuinely varies between sessions: taps, the ball, lighting.

## Sources

Opened in this session:
- [OpenAI, Solving Rubik's Cube with a Robot Hand (2019)](https://arxiv.org/abs/1910.07113)
- [OpenAI, Learning Dexterous In-Hand Manipulation (2018)](https://arxiv.org/abs/1808.00177)
- [Peng et al. 2018](https://arxiv.org/abs/1710.06537)
- [DeepMind, Robotic Table Tennis case study (2023)](https://arxiv.org/abs/2309.03315)
- [DeepMind, Human-level competitive table tennis (2024)](https://arxiv.org/abs/2408.03906)
- [JuggleRL](https://arxiv.org/abs/2509.24892)
- [Muratore et al. 2022 review](https://www.frontiersin.org/articles/10.3389/frobt.2022.799893/full)
- [BayesSim](https://arxiv.org/abs/1906.01728)
- [DROPO](https://arxiv.org/abs/2201.08434)
- [Chen et al., Understanding DR](https://arxiv.org/abs/2110.03239)
- [Xie et al., Dynamics randomization revisited](https://arxiv.org/abs/2011.02404)
- [legged_gym config](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- [Isaac Lab velocity env](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py)
- [Isaac Lab DelayedPDActuator](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py)
- [ManiSkill SO100 grasp-cube DR](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py)
- [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real)
- [isaac_so_arm101 reach cfg](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/tasks/reach/reach_env_cfg.py): no dynamics DR, ±0.01 joint noise only.

Via the project notes: BAM, Sony AI 2026, Gossard 2026, Humanoid-Gym, Hwangbo 2019, Tan 2018, RMA, Squint, Robo9, robo-ping-pong, Reist & D'Andrea.

From memory, not re-read: Tobin 2017, Dwibedi 2017, SimOpt, BayRn, Active DR ([arXiv 1904.04762](https://arxiv.org/abs/1904.04762)).
