# Domain-randomisation critique, second pass: SO-101 ping-pong bouncer

Date: 2026-10-30. Scope: the code as it stands after the developer acted on the [first pass](randomization_critique.md). I re-read [env.py](../bounce/env.py), [servo.py](../bounce/servo.py), [ball.py](../bounce/ball.py), [robot.py](../bounce/robot.py), [visual.py](../bounce/visual.py), [obs.py](../bounce/obs.py), [detector.py](../bounce/detector.py), [make_detector_data.py](../scripts/make_detector_data.py) (including the single-renderer change made during this review), [train_detector.py](../scripts/train_detector.py), [evaluation.py](../bounce/evaluation.py) and [real/arm.py](../real/arm.py).

**Provenance tags:**
- **[code]** read in this repo.
- **[run]** measured by running the env on CPU (512 envs, a few seconds; the scripts are in the session scratchpad).
- **[calc]** my own arithmetic.
- **[paper]** source opened in this session or in the first pass.
- **[web]** forum or issue page opened in this session.
- **[memory]** cited from memory, not re-read.
- **[U]** unverified.

## Bottom line

1. **The new physics is mostly sound.** The two-inertia gear model is stable at the 2 ms substep: max ω·dt = 0.98, against an explicit-Euler limit of 2, and the stiffest mode is wrist roll at 78 Hz **[run]**. The paddle oscillator's noise scaling is now correct, so the step-size artefact is gone. The rubber types, the burst dropouts and the age bias are implemented as proposed. Three problems remain in the new code:
   - The vibration pivots about the blade centre instead of the jaw.
   - The gear contact can pull.
   - The detector renders keep MuJoCo's default headlight on.
2. **Four nominal or range choices still describe hardware you do not have.** Each one biases both training and the "nominal" evaluation suite:
   - **Side camera:** it is present in 50% of episodes even at difficulty 0.
   - **Supply voltage:** vin runs up to 8.4 V, so the mean at full difficulty is 6.4 V. 73% of episodes are above 5.5 V, and the mean stall torque is 1.8× the 5 V datasheet value **[run]**.
   - **Acceleration register:** the range still excludes the ~39 rad/s² that Acceleration=254 implies. [arm.py:25](../real/arm.py#L25) still writes 254.
   - **Overhead depth noise:** its nominal of 1.5 cm is 2–4× too small for a 320×240 detector **[calc]**.
3. **Lighting is coupled in the policy env but not in the detector data.** In env.py, light sets exposure, which sets blur, noise and dropout, and the signs are right. The image generator instead draws scene light, gain, exposure and flicker independently. Real auto-exposure couples them, and that coupled joint distribution is what the detector will meet. The C920's auto-exposure behaviour (fps drop, anti-flicker exposure steps, drift within a session) is not modelled anywhere.
4. **The per-group ADR works, but its four levels will move almost in lockstep.** Every group is gated on the same survival signal. There is no boundary sampling and no buffer reset, and task difficulty sits in the same signal. The cheapest high-value addition is evaluation gates rather than ADR changes: a user-hardware suite, a lighting sweep, one-at-a-time sweeps, and a radius-error metric for the detector.

## 1. Prioritised table

| # | Factor | Current state | Problem | Proposed fix (range, distribution) | Impact | Where |
|---|---|---|---|---|---|---|
| 1 | **Acceleration register** | log₁₀ a_max ~ U(2.2, 3.9) (160–8000 rad/s²); the runtime writes Acceleration=254 ([servo.py:28](../bounce/servo.py#L28), [arm.py:25](../real/arm.py#L25)) | Unfixed from pass 1. If 254 (≈39 rad/s², [U]) is deployed, the sim has never seen it, and the launch/brake strokes need ≥39–73 rad/s² **[notes]**. | **Decide first.** Write Acceleration=0 at runtime start and read it back, then keep the range. If 254 must stay, train with log₁₀ ~ U(1.45, 1.75). | **High** | [arm.py](../real/arm.py), `SERVO_RANGES` |
| 2 | **Overhead depth-from-size noise** | σ_ray ~ U(0.8, 3.0) cm, nominal 1.5 cm, × dim (1–2.5); white per frame ([env.py:56](../bounce/env.py#L56), [env.py:392](../bounce/env.py#L392)) | At 320×240 the ball radius is 4.7–7.4 px from the overhead camera (0.84 m, ray 52° below horizontal) **[run]**. Depth σ = z·σ_r/r gives 2.8–4.4 cm at σ_r = 0.25 px and 5.7–8.9 cm at 0.5 px **[calc]**. The real error is also correlated in time and biased by blur and bloom. White noise averages out over 4 detections; correlated error does not. DeepMind found that bias costs far more than zero-mean noise **[paper]**. | Nominal 3.5 cm, logU(2, 7) cm, until the detector's radius residuals are measured. Split the variance 50/50 between white noise and an OU process (τ ~ logU(0.05, 0.3) s). Make σ_r grow with blur: σ_r·(1 + streak_px/2r). Add a radius bias κ·streak, with κ fitted on the held-out set (a white ball blooms in bright light, so it reads closer). | **High.** About 80% of the ray error is vertical, and the apex target is 7–15 cm with σ = 3 cm. | `_camera` [env.py:589-593](../bounce/env.py#L589); new radius metric in [train_detector.py:35](../scripts/train_detector.py#L35) |
| 3 | **Side camera that does not exist** | `side_cam_prob = 0.5`, applied at every difficulty ([env.py:48](../bounce/env.py#L48), [env.py:384](../bounce/env.py#L384)) | Half of the "nominal" suite and of model selection ([evaluation.py:7](../bounce/evaluation.py#L7), [:54](../bounce/evaluation.py#L54)) runs on hardware you lack. Capacity also goes to two sensing regimes. | Set 0.0 for the deployment run, or 0.2 if a side camera is planned. Force `cam_side=False` in every selection suite. | **High** (selection and evaluation) | `EnvCfg`, `forced` |
| 4 | **Supply voltage** | vin U(4.5, 8.4) per arm, nominal 5.0, plus strength U(0.55, 1.05) ([servo.py:30](../bounce/servo.py#L30), [:33](../bounce/servo.py#L33)) | The 5 V brick is fixed and known, yet the training mean is 6.4 V. The 5–95% stall band is 1.38–3.58 N·m, against 1.29 N·m (datasheet at 5 V) and 1.84 N·m nominal **[run]**. Asymmetric ranges shift the mean (pass 1 §3.2). The policy never sees vin. | vin ~ U(4.75, 5.25); keep sag through `supply_r` ~ U(0.05, 0.35) Ω. Or read Present_Voltage once per session and feed vin/5 to the actor with ±0.1 V noise, which turns the wide range from conservative into adaptive. | **Med-High** | `SERVO_RANGES`, `_obs`, [obs.py](../bounce/obs.py) |
| 5 | **Lighting → camera in the env** | Light ~ U(0,1) per episode; exposure = T·(0.1 + 0.9(1−light)); noise and dropout × (1 + 1.5(1−light)); latency independent of exposure ([env.py:387-398](../bounce/env.py#L387)) | The signs are right. Still missing:<br>• Uniform "light" means uniform exposure (3–33 ms), but room light is log-distributed.<br>• Exposure cannot exceed one frame, so there is no fps drop.<br>• Anti-flicker exposure steps are not modelled.<br>• Mid-exposure delay is not added to latency.<br>• Exposure is constant within an episode, whereas real auto-exposure drifts and jumps when a hand enters.<br>• Bright light is treated as perfect, with no glare. | See §2: log-lux, an auto-exposure (AE) model, a 15 fps mode, 1/120 s exposure steps, latency += t_exp/2, AE lag, and a glare term. | **Med-High.** In dim light this is the dominant sensing regime. | `_reset` sensing block, `_camera` |
| 6 | **Detector lighting chain** | Scene level logU(0.08, 1.3); gain logU(0.3, 2.5); blur exposure ∈ {2, 5, 10, 20, 33} ms; flicker in 20%. All independent ([make_detector_data.py:36](../scripts/make_detector_data.py#L36), [:165](../scripts/make_detector_data.py#L165), [:244](../scripts/make_detector_data.py#L244), [:167](../scripts/make_detector_data.py#L167)) | The sweep is broad, which follows Tobin. But it is unstructured: "dim scene + 2 ms + clean" is common, while "dim + 33 ms streak + gain noise", the real dim case, is rare. The headlight and spot-aim bugs (§3) further shrink the dim tail. | Drive everything from one sampled lux through the AE model of §2. Keep ±0.5 stop of AE error for breadth. | **High** (for the detector) | `build`, `postprocess`, `main` |
| 7 | **Paddle vibration pivot** | Roll and pitch both rotate the face about the blade centre ([env.py:485-490](../bounce/env.py#L485)) | Grip and jaw compliance pivot at the jaw, 6–16 cm behind the centre. A pitch θ̇ moves the centre at reach·θ̇, about 3.3 cm/s at σ = 2 mrad and 20 Hz, roughly 30% of the 11 cm/s stroke **[calc]**. In the sim, centred impacts feel nothing. | Pitch about the jaw: o_eff = o + reach·(R_eff[:,:,0] − R[:,:,0]), and pass o_eff and its previous value into `paddle_impact`. Keep roll about the handle axis. Optionally drive the mode with paddle acceleration plus impacts, not white noise only. | **Med-High** | [env.py:490](../bounce/env.py#L490), [:498](../bounce/env.py#L498) |
| 8 | **Detector: real context** (SDR) | Procedural textures, flat walls, two finger capsules ([make_detector_data.py:85-88](../scripts/make_detector_data.py#L85)); the camera re-aims at the paddle ([:110](../scripts/make_detector_data.py#L110)) | Unfixed from pass 1. Prakash et al. show that context-aware placement beats both plain DR and real BDD100K data on KITTI **[paper]**. Tremblay et al. 2018 is the plain-DR paper; SDR is Prakash et al. 2019, not Tremblay. | 30–50% composites over ~200 real frames from the fixed C920 pose. Hands as palm + forearm (8–10 cm palm), entering from the tap side during taps. Fixed deployment camera ± 3 cm / 3°, not re-aimed. | **High** (for the detector) | `build`, `camera` |
| 9 | **Joint-reading latency** | q is read exactly at the step end ([env.py:556](../bounce/env.py#L556)) | Unfixed from pass 1. DeepMind randomises observation latency separately from action latency **[paper]**. | Substep ring buffer with delay U(0, 10) ms, centred on the `sysid.py check` timings. | **Med** | `step`, `_measure_q` |
| 10 | **Tangential rubber response** | Grip fraction fixed at 0.4 ([ball.py:72](../bounce/ball.py#L72)) | Unfixed from pass 1. It is not tied to the new rubber types. | α ~ U(0.35, 0.75), drawn per rubber type (lower for long pimples). | **Med** | `paddle_impact`, rubber table |
| 11 | **Taps → misses** | Taps raise false detections only ([env.py:596](../bounce/env.py#L596)) | A hand in view also occludes the ball and shifts auto-exposure. | While `hand_t > 0`: miss probability + U(0.2, 0.5) and an AE step ×U(0.7, 1.4). Bias the outlier direction toward the tap's side. | **Med** | `_camera` |
| 12 | **Per-frame latency** | Fixed per episode ([env.py:389](../bounce/env.py#L389)) | Unfixed from pass 1. Webcam buffering adds 0–2 frames. | Per frame k ∈ {0, 1, 2} extra frames, P(k ≥ 1) ~ U(0, 0.3). Store the latency per detection. | **Med** | `_camera`, `_obs` |
| 13 | **Arm dynamics fidelity** | Diagonal joint inertia only ([robot.py:209](../bounce/robot.py#L209)); now on the link side of the gear | Before the change, armature (0.026) swamped the dropped off-diagonal terms. Link-side inertias are 0.0003–0.045 kg·m² **[run]**, so the omitted coupling (M_ij·q̈_j, e.g. 0.25 N·m on the elbow during a stroke **[calc]**) is now comparable. | Gate, not DR: a sim-to-sim check of the arm alone against `mj_step` on the Menagerie model, with matching armature and backlash approximated. | **Med** | new test |
| 14 | **Defocus / AF hunting** | None | The C920 autofocuses and hunts when a hand enters [U]. | Gaussian defocus σ ~ U(0, 1.5) px in 30% of images. | **Med-Low** | `postprocess` |
| 15 | **Ball and arm materials** | Default geom material (specular 0.5, shininess 0.5), lit by the headlight | Produces a fixed highlight at the camera-facing point, a shortcut the detector can learn. | Ball specular U(0, 0.5), shininess U(0.05, 0.8). PLA arm specular U(0.05, 0.35). Turn the headlight off so arm/paddle shadows on the ball become visible. | **Med-Low** | `build` |
| 16 | **Flicker banding** | Period 10–60 rows, depth 3–15%, independent of exposure ([make_detector_data.py:167-170](../scripts/make_detector_data.py#L167)) | 120 Hz flicker (60 Hz US mains) over a 15–30 ms readout gives 2–4 bands per frame, i.e. a 67–133-row period at 240 rows. Bands appear only when exposure < 8.3 ms. Cheap LEDs modulate up to 100% **[memory]** (IEEE 1789). | Period = 240·8.33/t_read rows, t_read ~ U(15, 30) ms. Depth m·(1 − t_exp/8.33 ms), m ~ U(0, 0.6). Only for artificial light with t_exp < 8.3 ms. | **Low-Med** | `postprocess` |
| 17 | **Grip slip** | Fixed mount per episode | Unfixed from pass 1. The new oscillator is no substitute for creep. | Measure first: mark the handle and check after 2 min of strokes. Clamp it if it moves. | **High if it slips** | — |
| 18 | **Blade curvature, rolling resistance, thermal state, inertia scaled with mass** | None; `inertia0` is not scaled ([robot.py:208](../bounce/robot.py#L208)) | Unfixed from pass 1 (rows 12, 19, 20, 21). | As pass 1: κ ~ U(−0.2, 0.2) m⁻¹; rolling resistance 0–0.5°; latent T ~ U(25, 65) °C; inertia × mass scale. | **Low-Med** | ball.py, servo.py, robot.py |
| 19 | **Ball–gripper/handle contact** | Ball passes through everything except the blade | A rim-side bounce toward the handle has no collision. | Sphere-vs-capsule test on handle and gripper, e = 0.5, or terminate as "dropped". | **Low-Med** | `paddle_impact` |
| 20 | **Supply current limit** | Linear sag, clamped at 0.5·vin ([servo.py:94](../bounce/servo.py#L94)) | A 5 V / 4 A brick folds back above its limit, and several servos stalling at ~2 A each can trip it [U]. | Brownout: if Σ\|I\| > I_lim ~ U(3, 6) A, then vin → 0.6·vin for 20–100 ms. Read the brick's rating. | **Low-Med** | `ServoBank.torque` |
| 21 | **Long episodes, high drops** | 8 s; drops 3–18 cm ([env.py:366](../bounce/env.py#L366)) | Unfixed from pass 1. | A 60 s eval suite. Drops of 20–40 cm, or an operating rule against them. | **Low-Med** | evaluation.py |

## 2. Lighting: what to change

**In env.py**, replace `light` by a physically scaled auto-exposure (AE) chain sampled in the sensing group. All constants marked [U] should be refit from the real lighting experiment below.

- lux ~ logU(40, 2000), nominal 300. Homes at night run ~50–150 lux, offices ~300–500, near a window >1000 **[memory]** (EN 12464-1 office 500 lux).
- AE target exposure t_req = t₃₀₀·(300/lux), with t₃₀₀ ~ logU(8, 30) ms [U].
- **fps drop.** UVC auto-exposure often runs past the frame period in dim light and lowers fps **[web]** ([Blue Robotics #233](https://github.com/bluerobotics/companion/issues/233); not C920-specific). If t_req > T, then with probability p_drop ~ U(0, 1)·d_s the camera goes to 15 fps and the exposure cap becomes 66 ms.
- **Anti-flicker steps.** With probability 0.8 the light is artificial. Then, when t_exp > 8.33 ms, round t_exp down to a multiple of 1/120 s; this is the C920's 60 Hz power-line setting [U].
- **Gain.** gain = clamp(t_req/t_exp, 1, 8). Scale σ_perp, σ_r and dropout by √gain, not by the fixed 1–2.5×.
- **Latency.** cam_lat += t_exp/2, from mid-exposure to readout. blur_k then only encodes the timestamp convention.
- **AE dynamics.** log t_req follows an OU process (σ = 0.15, τ = 1 s), plus the hand-in-view step from row 11.
- **Glare.** For a white ball when lux > 1000: outliers + U(0, 0.02) and radius bias + U(0, 3)%.

**In make_detector_data.py**, use the same chain in image space. This follows Tremblay et al. 2018, which randomises lighting and textures non-realistically, and the SDR lesson that structure matters **[paper]**.

- Lights: CCT ~ U(2700, 6500) K per light from a blackbody table. Today's palette spans only ~3000–5500 K: at warm = 0 it gives (1, 0.92, 0.85), which is not 6500 K **[calc]**.
- 20% "window" scenes: one directional light at U(3, 10)× the level with CCT 5500–7500 K. Clipped highlights are allowed.
- AWB set from the dominant light, ±8% error, so secondary lights leave a colour cast.
- Normalise the linear render to the AE target × logN(0, 0.35), clip, then add noise σ² = a·gain·I + (b·gain)².
- Motion blur from the same t_exp, with n_sub = max(3, ⌈streak_px/(r_px/2)⌉). This also fixes the 5-copy stepping (pass 1 §2e).
- Store lux, t_exp, gain and streak_px in the labels. Bucket evaluation by these, not by mean brightness: mean brightness confounds light with scene albedo ([train_detector.py:56-61](../scripts/train_detector.py#L56)).

**On the real arm**, run one cheap experiment that answers the user's question (1) directly:

- Light four setups: room light at night; room + desk lamp; an LED panel; daylight.
- Log lux at the paddle with a phone (±20% [U]).
- Record 60 s of hand-bounced ball at each, logging:
  - frame intervals, which show whether the camera drops to 15 fps;
  - mean brightness;
  - detection rate;
  - centre and radius residuals on 50 hand-labelled frames.
- Fit t₃₀₀, the 15 fps lux threshold, and σ_r(lux, streak) from these logs, then narrow §2's ranges.

Two runtime habits help regardless:
- Abort or flag if the median frame interval exceeds 40 ms.
- Apply UVC controls only *after* the capture session opens. On macOS, UVC exposure controls are unreliable and tend to revert ([CameraController #69](https://github.com/Itaybre/CameraController/issues/69) **[web]**; consistent with your observation).

## 3. Bugs or regressions found in the new code

1. **The vibration pivots about the blade centre** ([env.py:490](../bounce/env.py#L490)). Only roll about the handle axis should pivot there; pitch should pivot at the jaw. As a result, impacts at the centre get zero vibration velocity. Fix in row 7.
2. **The gear contact can pull** ([env.py:465-466](../bounce/env.py#L465)). The damping term c·(q̇m − q̇) acts even while the gears separate. In 1.1% of engaged samples, the gear torque had the opposite sign to the gap **[run]**. Effect: a "sticky" backlash that shortens reversal dead time. Fix: τ_c = sign(gap)·max(0, k(|gap| − b/2) + c·sign(gap)·(q̇m − q̇)).
3. **The MuJoCo headlight is on in the detector renders.** Its defaults are ambient 0.1, diffuse 0.4, specular 0.5 **[run]**, and so101.xml does not override them. Consequences:
   - The darkest sampled scenes (level 0.08) are lit mostly by a camera-mounted shadowless light, so "very dim" is not dim.
   - Arm and paddle shadows on the ball are washed out.
   - Every ball gets a highlight facing the camera.
   Fix: `spec.visual.headlight.active = 0`, or diffuse ≤ 0.1 in 30% of scenes to model monitor glow.
4. **Spot lights aim from a second random point** ([make_detector_data.py:41-42](../scripts/make_detector_data.py#L41)). `dir` uses a fresh `rng.uniform`, not the light's own `pos`, so many spots, with cutoffs of 30–80°, miss the scene. Fix: dir = (target + N(0, 0.2)) − pos.
5. **The ADR gate lags and is biased** ([env.py:698](../bounce/env.py#L698)):
   - Completed episodes are compared against the *current* level, with no buffer.
   - Survivors take 400 steps (12.5 iterations) to complete, while failures complete quickly. During a ramp the "hard" set therefore over-represents early failures.
   - All four groups share one survival signal, which includes task difficulty, so the levels move together.
   Fix:
   - Boundary sampling: in 20% of resets, set exactly one group to d_g = L_g and tag the episode with L_g.
   - A per-group buffer of 256 episodes, cleared after every change (OpenAI ADR: m = 240, p_b = 0.5 **[paper]**).
   - A relative threshold: expand if boundary performance ≥ 0.9× the concurrent non-boundary baseline; shrink if < 0.6×.
   - Task gets its own absolute gate. This is how legged_gym separates its terrain curriculum from DR **[memory]**.
6. **The "nominal" values do not match the user's hardware** (rows 1–4): side camera in 50%, vin mean 6.4 V, Acceleration=254 excluded, overhead depth σ optimistic.
7. **Minor:**
   - Lens distortion k₁ is normalised by f = W/2 = 160 px while the render's focal length is 200–313 px ([make_detector_data.py:135](../scripts/make_detector_data.py#L135)). The effective k₁ is therefore 1.6–3.8× the stated range, up to −0.46. Labels stay consistent. Normalise by the true f.
   - Coulomb friction under the tanh has slope·dt/J_m up to 1.5 at hold and above 2 at stroke torques **[run/calc]**: a bounded 2-cycle chatter. Use a Karnopp stick model if it shows up in sysid replay.
   - `cam_noise_px` is in metres, so the name is misleading.
   - `look["light"]` ([visual.py:58](../bounce/visual.py#L58)) is unused on the detector path.

What I checked and found correct:
- **_lerp_log:** the log-uniform draw, the 20% exact zeros when lo = 0, and the shrink toward nominal [code].
- **Burst-dropout states:** mean burst 1.7–5 frames, at most 13% of time in the bad state [calc].
- **Age bias:** it shifts true latency relative to the reported age, which matches a runtime that estimates t_capture = arrival − L_assumed [code].
- **Gear preload at reset:** the motor side is wound up correctly, and the hold drift at zero action is < 5 mrad in 1 s [run].
- **Hunting at hold:** 0.1–0.3 mrad std, negligible [run].
- **Rubber-type draws:** they collapse to the nominal at d = 0.
- **Strength scaling:** it scales stall torque and back-EMF damping together, consistent with a gear-efficiency loss.

## 4. Evaluation gates to add (in order)

1. **user_hw**: `cam_side=False`, vin U(4.8, 5.2), fps 30, Acceleration as deployed, everything else nominal. This is the model-selection metric in place of the current mixed average.
2. **user_hw_full**: the same at d = 1.
3. **Lighting sweep**: lux ∈ {40, 100, 300, 1000, 2000}, plus a forced 15 fps run. Report survival, bounces and apex error per level. Pass if survival at the measured room lux is ≥ 0.9× nominal.
4. **One-at-a-time sweeps** at lo, hi and 1.5×hi, others nominal, as DeepMind does **[paper]**: latency, age bias ±40 ms, overhead depth σ ×2, strength 0.5, a_max 39 rad/s², backlash 1.5°, glitch 10⁻² per step. [failure_analysis.py](../scripts/failure_analysis.py) stratifies at d = 1, which is correlational; these sweeps are causal.
5. **Per-group boundary**: each group at level 1 with the others at 0, to show which group binds.
6. **Detector, synthetic**: recall, false-positive rate, centre σ, and **radius bias and σ** (missing today; it sets row 2), by lux, gain, streak and ball × arm colour. Include white ball on white arm.
7. **Detector, real**: ≥200 labelled frames over the four lighting setups. Suggested targets: recall ≥ 95%, false positives ≤ 1%, centre σ ≤ 1 px, radius σ ≤ 0.3 px.
8. **Detector-in-the-loop**: 30 fps rendered sequences run through BallNet. Compare its empirical noise and dropout quantiles with env.py's analytic model at matched lux.
9. **Arm sim-to-sim** against MuJoCo with the full mass matrix (row 13), plus a 60 s suite.

## Sources

- [Prakash et al. 2019, Structured Domain Randomization (arXiv 1810.10093)](https://arxiv.org/abs/1810.10093): context-aware placement; beats plain DR and real BDD100K on KITTI.
- [Tremblay et al. 2018, Training Deep Networks with Synthetic Data (arXiv 1804.06516)](https://arxiv.org/abs/1804.06516): lighting, pose and textures randomised non-realistically; fine-tuning on real data beats real data alone. Its flying-distractor and background details are **[memory]**.
- [Tobin et al. 2017 (arXiv 1703.06907)](https://arxiv.org/abs/1703.06907): random textures; 1.5 cm localisation.
- [Blue Robotics companion #233](https://github.com/bluerobotics/companion/issues/233) and [CameraController #69](https://github.com/Itaybre/CameraController/issues/69): UVC auto-exposure fps drop, and unreliable C920 exposure control on macOS.
- Via the first pass, opened there: OpenAI Rubik's Cube ADR, Dactyl, DeepMind table tennis 2023/2024, JuggleRL, Peng 2018, Muratore 2022, Isaac Lab and legged_gym configs.
