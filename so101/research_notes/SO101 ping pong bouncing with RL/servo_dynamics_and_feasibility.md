# SO-101 / Feetech STS3215: servo dynamics, LeRobot control timing, and feasibility of bouncing a ping-pong ball on a store-bought paddle

Conventions used throughout: **[D]** = datasheet / vendor claim, **[M]** = measured or identified by someone on real hardware, **[S]** = read directly from source code (LeRobot 0.6.1 verified in the local install at `/Users/praneethsamineni/dream-env/lib/python3.12/site-packages/lerobot`), **[C]** = my own calculation (math shown), **[U]** = unverified. Scope updates from the coordinator are folded in: apex heights 5/10/15/20 cm; regular 150–190 g paddle; episode starts with the ball resting on a level paddle (launch phase); zero-shot sim-to-real, which requires a sysid procedure.

**Bottom line.** Keeping the ball bouncing low (5–20 cm) needs very little paddle *speed* (0.05–0.28 m/s) or *stroke* (3–36 mm peak-to-peak). The hard parts are elsewhere:
1. **Bandwidth.** The STS3215 position loop at LeRobot's P=16 behaves like a ~2–3 Hz second-order system, but bounces happen at 2.5–5 Hz (estimated from the BAM-identified model). The servo's lag must be modelled exactly in sim.
2. **Launch from rest.** It needs 1–2 m/s plus braking harder than 1 g. That exceeds what one joint can do with a full paddle at 5 V, and LeRobot's `Acceleration=254` setting (~39 rad/s²) forbids it outright at short levers.
3. **Mounting.** Holding a 150–190 g paddle in the gripper puts its CoM ~25 cm from the wrist-flex axis and loads the shoulder and elbow to ~80% of 5 V stall in a typical pose. The gripper will not hold it rigidly.

---

## 1. Which STS3215 variants does the SO-101 ship with, per joint, and what are their datasheet limits?

### Takeaway
The SO-101 **follower** uses 6× STS3215 **1/345, 7.4 V (C001)** by default; a 12 V / 30 kg·cm variant (C018/C047) is an optional upgrade. The **leader** mixes 1/191, 1/345 and 1/147 gearing and is always 7.4 V. Datasheet figures: 7.4 V version = 19.5 kg·cm (1.91 N·m) stall and 52 rpm at 7.4 V. 12 V version = 30 kg·cm (2.94 N·m) stall and 45 rpm (0.222 s/60°) at 12 V. The standard kit runs the 7.4 V servos from a **5 V** supply, which costs roughly ⅓ of both torque and speed.

### Cited Findings
- [D] "The follower arm uses 6x STS3215 motors with 1/345 gearing." Leader per-joint table: Shoulder Pan (ID 1) 1/191; Shoulder Lift (2) 1/345; Elbow Flex (3) 1/191; Wrist Flex (4) 1/147; Wrist Roll (5) 1/147; Gripper (6) 1/147 — [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- [D] BOM: two-arm kit = 7× 1/345 (C001), 2× 1/191 (C044), 3× 1/147 (C046); the follower-only table lists 6× 1/345 (C001). "The leader arm is always 7.4V for the SO101." "The 7.4V has a stall torque of 16.5kg.cm at 6V (and likely slightly less for a 5V power supply)." "The 12V version has a stall torque of 30kg.cm." "…you will also have to buy a 12V 5A+ power supply instead of a 5V one." — [TheRobotStudio SO-ARM100 README](https://github.com/TheRobotStudio/SO-ARM100)
- [D] Standard SO-ARM101 kit power is "2x 5V/4A power adapter" (secondary press source) — [CNX Software](https://www.cnx-software.com/2025/05/02/so-arm101-open-source-dual-robotic-arm-kit-works-with-hugging-faces-lerobot/?amp=1). "Pro" kits ship 12 V followers, but vendor listings disagree on PSU current (12 V/1 A vs 12 V/2 A vs 12 V/5 A) — [OpenELab SO-101 Pro listing](https://openelab.io/products/seeed-studio-so-101-pro-servo-motor-kit-lerobot) (search-result summary; not opened).
- [D] Feetech STS3215 evaluation sheet (2020): "7.4V 19KG.CM … Stall Torque：19.5kg.cm@7.4V; Stall Speed：52RPM@7.4V" (the "stall speed" label is presumably the no-load speed, [U]). 1:345 copper gear train; "360 degree absolute position 4096 bit precision, the highest position resolution is 0.088 degrees". Four modes: 0 position, 1 closed-loop speed, 2 open-loop speed/PWM, 3 step. Multi-turn ±7 turns. Protections: overload, overcurrent, overvoltage, overheat. 45.2×24.7×35 mm — [Feetech STS3215 datasheet (Mantech mirror)](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)
- [D] Waveshare ST3215 (12 V class): input 6–12.6 V ("12V is recommended"), "up to 30kg.cm@12V", no-load speed "0.222sec/60°（45RPM）@12V", no-load current 200 mA, stall current 2.7 A, resolution 360°/4096, 1 Mbps, IDs 0–253 — [Waveshare wiki ST3215](https://www.waveshare.com/wiki/ST3215_Servo)
- [D] ST-3215-**C047** (12 V) data used in a MuJoCo gain derivation: stall 30 kg·cm, stall current 2.7 A, rated 12 V, no-load current 180 mA, no-load speed 45 rpm — [Gregory119 calc_actuator_model.py](https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/calc_actuator_model.py)
- [D] 12 V "STS3215-C018 12V 30kg.cm TTL serial bus servo for SO-ARM100/101" is sold as the SO-ARM upgrade part — [OpenELab C018 listing](https://openelab.io/da/products/feetech-sts3215-c018-servo-12v) (title only)
- [S] LeRobot defines every SO follower joint as `Motor(id, "sts3215", …)`, with resolution 4096 and default baud 1 000 000 — [lerobot so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [lerobot feetech tables.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py)
- [D] Max temperature limit register default 70 °C — [Feetech STS memory-table analysis (third-party)](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)

### Inferences
- [C] Unit conversions (1 kg·cm = 0.0981 N·m):

  | Variant / supply | Stall torque | No-load speed |
  |---|---|---|
  | C001 7.4 V servo @ 7.4 V | 19.5 kg·cm = **1.91 N·m** [D] | 52 rpm = **5.45 rad/s** [D] |
  | C001 @ 6 V | 16.5 kg·cm = 1.62 N·m [D] | ≈ 42 rpm ≈ 4.4 rad/s [C, linear in V] |
  | C001 @ 5 V (standard kit PSU) | ≈ 13.75 kg·cm ≈ **1.35 N·m** [C, scaled from 6 V figure] | ≈ 35 rpm ≈ **3.7 rad/s** [C] |
  | C018/C047 12 V servo @ 12 V | 30 kg·cm = **2.94 N·m** [D] | 45 rpm = **4.71 rad/s** [D] |

- The 12 V servo gives ~1.5× (vs 7.4 V) to ~2.2× (vs 5 V) torque, at a slightly lower top speed than the 7.4 V servo run at 7.4 V. For this task torque is the binding limit, not speed (§5–§8), so the 12 V servo is the more useful upgrade.
- The 7.4 V and 12 V servos cannot share one supply. A 12 V PSU would over-volt the 7.4 V parts (the datasheet lists overvoltage protection). So an upgrade means swapping all the follower servos on that bus.
- **Action for the user:** read the servo label (C001 vs C018/C047) and the PSU label. Every margin below depends on 5 V vs 7.4 V vs 12 V.

### Gaps
- No primary datasheet found for the C001's stall current, no-load current or rated (continuous) torque. The 12 V "rated torque 10 kg·cm" figure appeared only in a search snippet and was not verified.
- No torque–speed curves at 5 V. The 5 V numbers above are linear scalings [U].

---

## 2. How does the STS3215 internal position loop work, and what does LeRobot 0.6.1 write into it?

### Takeaway
In position mode the firmware does three things:
1. It moves an internal target toward `Goal_Position`, rate-limited to a maximum velocity (~3400 steps/s ≈ 5.2 rad/s) and ramped by the `Acceleration` register.
2. It runs a P(+D) loop on that target.
3. It outputs a PWM duty cycle to the DC motor, clipped at ~97%.

LeRobot 0.6.1 sets P=16 (factory default is 32), I=0, D=32, `Acceleration`=254 and `Maximum_Acceleration`=254, and `Return_Delay_Time`=0. Only the gripper gets a 50% torque limit. If the third-party unit table is right, `Acceleration`=254 is a **~38.6 rad/s² cap**, not "maximum acceleration". That is decisive for launching the ball.

### Cited Findings
- [S] `SOFollowerConfig`: `position_p_coefficient: int = 16`, `position_i_coefficient: int = 0`, `position_d_coefficient: int = 32`, `num_read_retries: int = 2`. The retry comment reads: "Feetech buses can occasionally return a corrupted status packet ("Incorrect status packet!"), especially when several joints move at once" — [lerobot config_so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py)
- [S] `configure()` writes `Operating_Mode=POSITION` and P/I/D to every motor. **For the gripper only** it writes `Max_Torque_Limit=500` ("50% of max torque to avoid burnout"), `Protection_Current=250` ("50% of max current") and `Overload_Torque=25` ("25% torque when overloaded") — [lerobot so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- [S] `FeetechMotorsBus.configure_motors(return_delay_time=0, maximum_acceleration=254, acceleration=254)`. Comments: "By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 …). We ensure this is reduced to the minimum of 2µs (value of 0)", and "Set 'Maximum_Acceleration' to 254 to speedup acceleration and deceleration of the motors" — [lerobot feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)
- [S] LeRobot 0.6.0 (local uv cache) hard-coded the same values, with the comment "Set P_Coefficient to lower value to avoid shakiness (Default is 32)" and "Set I_Coefficient and D_Coefficient to default value 0 and 32" — [lerobot so_follower.py history](https://github.com/huggingface/lerobot/commits/main/src/lerobot/robots/so_follower/so_follower.py)
- [S] Control-table addresses (STS/SMS): P 21, D 22, I 23, Minimum_Startup_Force 24, CW/CCW_Dead_Zone 26/27, Protection_Current 28, Operating_Mode 33, Protective_Torque 34, Protection_Time 35, Overload_Torque 36, Torque_Enable 40, Acceleration 41, Goal_Position 42, Goal_Time 44, Goal_Velocity 46, Torque_Limit 48, Present_Position 56, Present_Velocity 58, Present_Load 60, Present_Voltage 62, Present_Temperature 63, Present_Current 69. Factory area: "DTs" 81 "(ms)", Maximum_Velocity_Limit 84, Maximum_Acceleration 85, "Acceleration multiplier in effect when acceleration is 0" 86 — [lerobot tables.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py)
- [D] Third-party register analysis:
  - Gains: Kp "Proportional gain (scaled by 1/4)", Kd "Derivative gain (scaled by 1/8)".
  - **Acceleration (41): range 0–254, unit "8.7 degrees/s²", "0 indicates maximum acceleration"**, default 0.
  - Torque: Maximum Torque (16) default 1000 in 0.1% units. Protection Current (28) default 511 × 6.5 mA.
  - Overload: Holding Torque after overload (34) default 20%; Protection Time (35) default 200 × 10 ms = 2 s; **Overload Torque Threshold (36) default 80%**; Overcurrent Protection Time default 200 × 10 ms.
  - Dead zones default 1 step (0.087°). Torque_Enable=2 is a "damping mode". Goal speed unit 0.732 rpm.

  Source: [Feetech STS memory-table analysis](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- [D] Datasheet: "Acceleration start stop function: speed and acceleration value can be set." On overload, "the unloading force lasts for 2S (20% of the default blocking force). Until a new command is triggered". Load feedback is "the voltage duty cycle of the motor, … 1000 = 100%". Current feedback "1 = 6.5mA". Speed feedback is in "steps … per second", with "50steps/second=0.732RPM" — [Feetech STS3215 datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)
- [M] BAM's identified firmware model, from its source: "The firmware rate-limits the target position it feeds to its P controller". It computes `duty = (q_target_smooth − q) · kp · error_gain(=0.166) · error_gain_ratio`, clipped to `max_pwm=0.97`, with both constants "determined using an oscilloscope and STS3215 actuators". Default max velocity is `3400·2π/4096` rad/s, and voltage = `vin · duty` — [Rhoban BAM feetech/actuator.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/actuator.py)

### Inferences
- [C] `Acceleration=254` × 8.7°/s² = 2210°/s² ≈ **38.6 rad/s²** (equivalently 254 × 100 steps/s² × 2π/4096 = 39.0 rad/s²). LeRobot believes 254 "speeds up" the servo, but under the wiki's semantics 254 is the *largest finite ramp*, and 0 removes the ramp. A joint capped at 38.6 rad/s² moves a paddle point at lever r with at most 38.6·r m/s². At r = 0.15 / 0.25 m that is 5.8 / 9.6 m/s², **below g in both cases**, so a launch by braking (§6) is impossible at LeRobot defaults. The register must be measured (§9) and probably set to 0 for this task.
- [C] Using BAM's error gain, the duty cycle saturates at |error| = 0.97/(kp·0.166) rad: **20.8° at P=16** and 10.4° at P=32. Bouncing strokes are 1–5° at the wrist, so the servo will run in its linear (P-controller) region for steady bouncing but saturate for launch-sized moves.
- Conflicting register semantics: LeRobot labels address 36 `Overload_Torque` and comments "25% torque when overloaded", but the wiki says address 36 is the overload **threshold** (default 80%) and address 34 is the post-protection torque (default 20%). Under the wiki reading, LeRobot makes the gripper enter protection after >2 s above 25% load. Either way, a gripper squeezing a paddle handle (a stalled motor) will fall back to low torque (§8).
- The user's own README says the softer gains P12/D40 could not lift the folded arm "under the 50% torque limit" ([so101/README.md](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/README.md)). LeRobot 0.6.1 writes that 50% limit only to the gripper, and no torque-limit writes exist in the user's code (grep). So arm joints should be at factory `Max_Torque=1000`; worth confirming by reading register 16/48 per joint.

### Gaps
- No official Feetech manual was retrieved: `doc.feetech.cn` is JavaScript-rendered. Unconfirmed: `Goal_Velocity=0` semantics in position mode (assumed "max speed"), D-term scaling, the `DTs` firmware control period, and how the acceleration ramp interacts with goals streamed at 30–100 Hz. All are [U] and should be read back or measured on the arm (§9).

---

## 3. What read/write rates and command-to-motion latency are achievable through LeRobot on a Mac?

### Takeaway
On the wire, a 6-servo `sync_read` of positions takes ~0.62 ms and a `sync_write` of goals ~0.26 ms at 1 Mbps, so the bus itself allows well over 500 Hz. I found **no published benchmark** of LeRobot SO-101 loop rates on macOS. Community practice is 15–30 Hz, and one issue links 60 Hz to more bus dropouts. LeRobot's failure path waits ≥50 ms per bad packet, which is an outsized hazard for a 200–400 ms bounce. Command-to-motion dead time is a few ms (BAM-identified 1–5 ms + bus), but the servo's ~2–3 Hz dynamics dominate the effective lag.

### Cited Findings
- [S] Default baud `DEFAULT_BAUDRATE = 1_000_000`; `Return_Delay_Time` written to 0 (2 µs) — [lerobot feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)
- [S] LeRobot patches the Feetech SDK packet timeout to `tx_time_per_byte*packet_length + tx_time_per_byte*3.0 + 50` ms. Comment: "It fixes https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6 … we rely on the (unofficial) [SDK on PyPI] which needs patching" — [lerobot feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)
- [S] `get_observation()` does one `sync_read("Present_Position", num_retry=num_read_retries)` and logs "read state: {dt_ms:.1f}ms" at DEBUG level. `send_action()` does one `sync_write("Goal_Position", …)`, plus an extra `sync_read` only if `max_relative_target` is set. **Velocity is not read** — [lerobot so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- [M] Issue #3131, "SO-101 follower disconnects during teleoperation": a maintainer suggested dropping from 60 fps to 30 fps. The reporter said lower frequency "seems to reduce the number of disconnects, but the issue is still present". The maintainer added: "Feetech motors can sometimes have weird behaviors, and they are very very hard to debug". Caveat: the reporter's servos were **Hiwonder HX-30HM/HX-10HM**, not STS3215 — [lerobot issue #3131](https://github.com/huggingface/lerobot/issues/3131)
- [M] Recurrent "[TxRxResult] Incorrect status packet!" and "There is no status packet" reports — [lerobot issue #1010](https://github.com/huggingface/lerobot/issues/1010), [lerobot issue #3299](https://github.com/huggingface/lerobot/issues/3299)
- [M] The ManiSkill zero-shot SO-100 sim2real tutorial recommends "for the SO100 hardware to stick to a control frequency of 15 which is a good balance of speed with accuracy/safety" (`--control-freq=15`) — [lerobot-sim2real tutorial](https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md)
- [M] The user's own setup: "Control loop drops to 4–10 Hz, cameras time out while recording", caused by CPU AV1 encoding at "63 ms CPU per frame per camera" and fixed with `h264_videotoolbox` (1.5 ms). Their rollouts use `--fps=30` — [so101/README.md](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/README.md), [so101/rollout.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/rollout.py)
- [M] BAM-identified `command_delay` for the STS3215: 0.00105 s (model m1); 0.00494 s (m4); 0.00498 s (m6) — [BAM params feetech_sts3215_7_4V](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V)
- [D] SO-101 USB enumerates as `/dev/tty.usbmodem…` on macOS; the Waveshare board needs its jumpers on channel "B" (USB) — [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)

### Inferences
- [C] Wire time at 1 Mbps, 10 bits/byte:

  | Transaction | Bytes | Time |
  |---|---|---|
  | `sync_read` Present_Position × 6 | instruction 14 B (FF FF FE len inst addr n + 6 IDs + chk) + 6 status × 8 B | **0.62 ms** |
  | Same, reading position + velocity (4 B per servo) | — | 0.74 ms |
  | `sync_write` Goal_Position × 6 | 26 B | **0.26 ms** |

  The bus ceiling is >1 kHz. Real loops will be limited by the USB-serial round trip (full-speed USB has 1 ms frames, so plausibly 1–4 ms per transaction on macOS [U]) and by Python overhead. A **100–200 Hz read+write loop is plausible but unverified**; benchmark it (§9 step 1).
- [C] Steps per bounce, by apex height:

  | Rate | 5 cm | 10 cm | 15 cm | 20 cm |
  |---|---|---|---|---|
  | 30 Hz | 6 | 9 | 10 | 12 |
  | 50 Hz | 10 | 14 | 17 | 20 |
  | 100 Hz | 20 | 29 | 35 | 40 |

  At 30 Hz a 5 cm bounce gives the policy only ~6 decisions per flight. 50 Hz or more is advisable, provided the camera supplies matching information (adjacent scope).
- Tail latency is the bigger risk. A corrupted packet costs ≥50 ms (timeout), and `num_read_retries=2` can stretch that to ≥100 ms: half the flight time of a 5 cm bounce. For bouncing, use non-blocking error handling: on failure, reuse the last state and flag it, rather than retrying synchronously.
- Effective command-to-motion latency ≈ policy compute + one write (~1–3 ms [U]) + firmware cycle (`DTs`, unknown) + ~1–5 ms (BAM) ≈ **5–15 ms of dead time [U]**. The servo's second-order lag then adds −74° to −143° of phase at 2.5–5 Hz (§4–5), which equals 80–100 ms at 2.5 Hz and ~75 ms at 5 Hz.

### Gaps
- No public benchmark of `so101_follower` loop rate or `sync_read` latency on macOS was found (searches of LeRobot issues and the web returned nothing quantitative).
- The USB-serial chip on the user's controller board and its macOS latency are unknown [U].
- `DTs` (firmware control period) is unknown; read register 81.

---

## 4. What measured dynamics, system identifications and sim actuator models exist for STS3215 / SO-100 / SO-101?

### Takeaway
The only rigorous public identification is **Rhoban BAM's** STS3215 7.4 V model (pendulum testbench, models m1–m6). It finds:
- a rate-limited target at **5.07–5.29 rad/s**;
- a large reflected **armature ≈ 0.022–0.026 kg·m²**, which dominates the inertia of any SO-101 joint, paddle included;
- Coulomb friction ≈ 0.05 N·m, viscous 0.03–0.06 N·m·s/rad;
- command delay 1–5 ms.

The popular sim models disagree wildly with BAM and with each other:
- MuJoCo Menagerie kp=998 N·m/rad is ~100× stiffer than BAM implies;
- Isaac Lab caps velocity at 1.5 rad/s and has no armature;
- ManiSkill allows 100 N·m.

None is usable as-is for a dynamic bouncing task.

### Cited Findings
- [M] BAM identified parameters, Feetech STS3215 7.4 V — [BAM params](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V), [BAM actuator list](https://bam.readthedocs.io/en/latest/usage/actuators.html):
  - m1: kt 1.1776, R 2.4787 Ω, armature 0.02608 kg·m², max_velocity 5.288 rad/s, command_delay 0.00105 s, friction_base 0.0515 N·m, friction_viscous 0.0599
  - m4: kt 1.2653, R 2.4724, armature 0.02336, max_velocity 5.071, command_delay 0.00494, friction_base 0.0519, friction_stribeck 0.0111, load_friction_base 0.208, dtheta_stribeck 0.443, alpha 9.99, viscous 0.0338
  - m6: kt 1.2753, R 2.7534, armature 0.02156, max_velocity 5.096, command_delay 0.00498, friction_base 0.0533, load_friction_motor 0.0449, load_friction_external 0.2436, quad terms ~0.003–0.008, viscous 0.0282
  - Also identified: Waveshare ST3025 12 V (m1: kt 1.620, R 3.73, armature 0.00436) and Dynamixel MX-64, MX-106, XL-320, XL-330 (useful Koch-arm analogues).
- [M] BAM recording method: single servo on a pendulum testbench (`--mass`, `--length`), `--kp` default 32, **D set to 0**, predefined trajectories (e.g. `lift_and_drop`). Each sample logs position, speed, input volts, goal_position, torque_enable, and the timestamp at the midpoint of the read — [BAM feetech/record.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/record.py)
- [S] MuJoCo Menagerie `robotstudio_so101/so101.xml`:
  - `sts3215` class: joint `damping="0.60" frictionloss="0.052" armature="0.028"`; `<position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>`.
  - Comment: "These gains are not a 1-to-1 mapping of the servo gains used in Lerobot … calculated according to [Gregory119] … assuming that the servo proportional gain is set to 16."
  - Backlash class: "+/- 0.5° of backlash – not added to any joints for now".

  Source: [Menagerie so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)
- [S] Gregory119 derivation: kp = K_m·G_p·(180/π)/R and kv = K_m·K_b/R, assuming "the servo proportional gain … uses units of gain/deg". The script assumes R = 1 Ω, K_m = (30 kg·cm)/(2.7 A), and K_b from 12 V, 180 mA and 45 rpm — [Gregory119 README](https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/README.md), [calc script](https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/calc_actuator_model.py)
- [S] Isaac Lab `isaac_so_arm101` SO-101 `ImplicitActuatorCfg`: `effort_limit_sim=1.9`, `velocity_limit_sim=1.5`; stiffness shoulder_pan 200, shoulder_lift 170, elbow 120, wrist_flex 80, wrist_roll 50; damping 80/65/45/30/20; gripper 2.5 N·m, K 60, D 20; **no armature** — [isaac_so_arm101 so_arm101.py](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/so_arm101.py)
- [S] ManiSkill SO-100 `pd_joint_pos`: stiffness 1e3, damping 1e2, `force_limit=100` — [ManiSkill so_100.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py)
- [S] SO-101 URDF (Isaac repo):
  - Joint origins: shoulder_lift→elbow (−0.11257, −0.028) m; elbow→wrist_flex (−0.1349, 0.0052); wrist_flex→wrist_roll (0, −0.0611, 0.0181); wrist_roll→gripper_frame (−0.0079, 0, −0.0981).
  - Link masses: upper arm 0.103 kg, lower arm 0.104, wrist 0.079, gripper 0.087, moving jaw 0.012, shoulder 0.100.
  - Joint `effort="10" velocity="10"` (placeholders).

  Source: [so_arm101.urdf](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf)
- [M] Zero-shot RGB sim2real of SO-100 cube grasping has been shown with ManiSkill + LeRobot at 15 Hz — [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real)
- [M] Generic recipe: tune simulator armature, friction and motor delay with CMA-ES until simulated joint trajectories match real ones, then check by overlaying plots — [UWLab Sim2Real SysID](https://uw-lab.github.io/UWLab/main/source/publications/omnireset/sim2real.html). Actuator models can also be fitted from encoder motion alone via differentiable simulation — [arXiv 2604.10351](https://arxiv.org/pdf/2604.10351)

### Inferences
- [C] Linearising BAM m1 at LeRobot's P=16, with the D-term ignored because BAM recorded with D=0:
  - stiffness K = kt·(V/R)·P·0.166·1.005 ≈ **9.4 N·m/rad at 7.4 V** and **6.3 N·m/rad at 5 V**;
  - damping B = kt²/R + viscous ≈ 0.56 + 0.06 ≈ **0.62 N·m·s/rad**;
  - with J = armature + load ≈ 0.024–0.035 kg·m², the natural frequency is ω_n ≈ **2.1–3.2 Hz** at P=16 (3.1–4.2 Hz at P=32), with ζ ≈ 0.5–0.8.

  This is the key number for bouncing (§5).
- [C] Armature dominance. Paddle inertia about the wrist-flex axis is 0.004–0.018 kg·m² (§8); BAM's armature is 0.022–0.026 and Menagerie's 0.028. Joint acceleration is therefore mostly limited by the servo's own rotor. A sim without armature (Isaac config) makes joints several times too agile. Its 1.5 rad/s velocity cap, against a real ≈5.1–5.3 rad/s, simultaneously makes them ~3.5× too slow.
- [C] Menagerie's kp = 998 N·m/rad saturates its ±2.94 N·m force range at 0.17° of error. BAM implies saturation at ~21° (P=16). The ~100× gap comes from Gregory119 treating the P register as "volts per degree" with R = 1 Ω. Menagerie's damping 0.60, frictionloss 0.052 and armature 0.028 look close to BAM m1 (0.56 back-EMF + 0.06 viscous, 0.052, 0.026), but their provenance is not stated [U].
- [C] BAM m1's implied stall torque, kt·0.97·7.4/R ≈ 3.4 N·m, is ~1.8× the 1.91 N·m datasheet figure. Treat BAM's kt and R as a fitted pair, not physical motor constants, and refit on the user's arm and supply voltage (§9).
- Recommended sim actuator, implemented as an explicit/custom actuator rather than PhysX/MuJoCo implicit PD, in this order:
  1. command dead time τ_d (+ zero-order hold at the policy rate);
  2. rate-limited internal target (v_max ≈ 5.2 rad/s);
  3. optional acceleration ramp if `Acceleration` ≠ 0;
  4. duty = clip(K_e·(target − q) − K_d·q̇, ±0.97);
  5. τ = kt·(V·duty − kt·q̇)/R;
  6. armature, Coulomb + viscous friction;
  7. per-joint torque and current limits.

  Initialise from BAM m1 and refit on the arm. This is exactly BAM's `STS3215Actuator.compute_control` structure plus delay.

### Gaps
- No published SO-100/SO-101 sysid with a payload, no measured step responses of SO-101 joints, and no BAM parameters for the 12 V STS3215 were found.
- BAM was identified with D=0, while LeRobot uses D=32, so the D-term's effect is unmodelled.
- No measured backlash value exists (Menagerie's ±0.5° is unsourced).

---

## 5. Feasibility of steady low bouncing at apex 5, 10, 15, 20 cm

### Takeaway
Sustaining a bounce needs only **0.05–0.28 m/s** of paddle upward speed at impact and a **3–36 mm** stroke, well within the arm's ≥0.5 m/s. The binding constraint is **timing and bandwidth**: bounce frequency is 2.5–5 Hz, but the STS3215 loop at P=16 is a ~2–3 Hz system, so commanded strokes are attenuated to 0.2–0.96× and lag 74–143° (+14–27° per 15 ms of dead time). Lower bounces are *easier* mechanically (smaller strokes, less lateral sensitivity) but *harder* in timing (5 Hz, 6 steps per flight at 30 Hz). Apex **10–15 cm** looks like the best compromise.

### Cited Findings
- [D] Ball: 40 mm diameter, 2.7 g — [Laws of Table Tennis 2026-27](https://www.tabletennisengland.co.uk/content/uploads/2026/07/Laws-of-Table-Tennis-2026-27.pdf) (search-result summary)
- [D] Plastic ball bounce spec: 240–265 mm rebound from a 305 mm drop onto a steel block (ITTF ball document as quoted) — [PingSkills: The table tennis ball bounce](https://www.pingskills.com/table-tennis-forum/the-table-tennis-ball-bounce). Older celluloid-era wording gives 24–26 cm with COR 0.89–0.92 — [Sportsmatik](https://sportsmatik.com/sports-corner/sports-equipage-detail/table-tennis-ball)
- [M] Racket COR has been measured for 15 rackets, with and without rubber, on a pseudo-rigid wall, but the values were not retrievable from the abstract — [Karasawa et al. 2021, Trans. JSME](https://www.jstage.jst.go.jp/article/transjsme/87/902/87_21-00145/_article/-char/en). Rubber-sheet COR falls as ball speed rises and depends on incidence angle (Sakai, Yasutomi, Saiki 2017–18, search summary) — [JSME symposium 2018 C-7](https://www.jstage.jst.go.jp/article/jsmeshd/2018/0/2018_C-7/_article/-char/en)
- [M] Humans bouncing a ball on a racket hit with **negative (decelerating) paddle acceleration at impact**, which gives passive stability against perturbations — [Schaal, Sternad & Atkeson 1996, J. Motor Behavior](https://data.zhenzhi.com/details/1/869351250); "To obtain this passive stability, the ball has to be impacted with negative acceleration" — [Ronsse et al. 2006](https://orbi.uliege.be/bitstream/2268/33349/1/ronsse06.pdf). A blind robot juggler (Reist & D'Andrea) instead stabilises with paddle shape and an *accelerating* impact — [Reist & D'Andrea](https://typeset.io/pdf/bouncing-an-unconstrained-ball-in-three-dimensions-with-a-4zog40ij6p.pdf)

### Inferences
- [C] Kinematics. Impact plane = paddle height at impact; steady state, so impact speed equals launch speed. v = √(2gh), flight T = 2v/g. The paddle is effectively infinitely heavy (effective mass at the contact point ≈ J/r² ≈ 0.53 kg vs 2.7 g ball, ratio 0.5%), so v_out = e·v_in + (1+e)·v_p and, at steady state, **v_p = v(1−e)/(1+e)**.

  | apex h | impact speed v | flight T | bounce freq | v_p, e=0.75 | v_p, e=0.85 | v_p, e=0.90 |
  |---|---|---|---|---|---|---|
  | 5 cm | 0.99 m/s | 202 ms | 4.95 Hz | 0.141 m/s | 0.080 | 0.052 |
  | 10 cm | 1.40 | 286 ms | 3.50 Hz | 0.200 | 0.114 | 0.074 |
  | 15 cm | 1.72 | 350 ms | 2.86 Hz | 0.245 | 0.139 | 0.090 |
  | 20 cm | 1.98 | 404 ms | 2.48 Hz | 0.283 | 0.161 | 0.104 |

  Control steps per bounce are in the §3 table (6–40 depending on height and rate).
- [C] Minimum stroke. For a sinusoidal paddle at the bounce frequency, z = A·sin(ωt) with ω = 2π/T, impacting at peak velocity: A = v_p/ω. Peak-to-peak stroke:

  | apex | e=0.75 | e=0.85 | e=0.90 |
  |---|---|---|---|
  | 5 cm | 9.1 mm | 5.2 mm | 3.4 mm |
  | 10 cm | 18.2 | 10.3 | 6.7 |
  | 15 cm | 27.3 | 15.5 | 10.1 |
  | 20 cm | 36.4 | 20.6 | 13.4 |

  Peak paddle acceleration A·ω² = **πg(1−e)/(1+e)**, which is independent of h: 4.40 / 2.50 / 1.62 m/s² for e = 0.75 / 0.85 / 0.90. A passively stable (decelerating) impact needs a slightly larger A than this minimum. An impulsive stroke (paddle idle, then a quick flick) needs proportionally higher acceleration.
- [C] Speed capability. A single joint at 3.7 rad/s (5 V estimate) to 5.2 rad/s (firmware cap) gives 0.50–0.70 m/s at a 0.135 m lever (elbow→wrist) and 0.93–1.30 m/s at 0.25 m (wrist→paddle). Required v_p ≤ 0.28 m/s, so **speed is not limiting** for steady bouncing.
- [C] Bandwidth. Linearised BAM model (§4) with J = 0.027–0.033 kg·m²:

  | Setting | 2.5 Hz | 3.5 Hz | 5 Hz |
  |---|---|---|---|
  | P=16, 5 V | \|G\| 0.64–0.65, −92…−100° | 0.38–0.42, −116…−125° | 0.19–0.23, −136…−143° |
  | P=16, 7.4 V | 0.93–0.96, −74…−83° | 0.62–0.67, −105…−116° | 0.31–0.36, −132…−140° |
  | P=32, 7.4 V | 1.2–1.3 (resonant), −39…−42° | 1.3, −67…−78° | 0.8–0.9, −112…−125° |

  Add −360°·f·τ_d for dead time (−27° at 5 Hz for 15 ms). Consequences: the policy must command strokes 2–5× larger than desired and ~¼–⅓ of a period early, and the sim must reproduce this gain/phase curve or the learned timing will be wrong. Raising P to 32 (the factory default; it is configurable via `position_p_coefficient`) roughly doubles usable bandwidth, at the cost of the "shakiness" LeRobot cites.
- [C] Torque for the minimum sinusoidal stroke via the wrist hinge (m_p 0.17 kg, CoM 0.15–0.25 m): about 0.8–0.9 N·m at e = 0.85 and 1.1–1.3 N·m at e = 0.75, gravity included. That is **60–95% of 5 V stall**, so it is thermally significant; see §8.
- [C] Impact loads are negligible: angular impulse 1.2–2.5 mN·m·s gives Δω ≈ 0.04–0.08 rad/s.
- Lower vs higher apex, for this arm:

  | | 5 cm | 10–15 cm | 20 cm |
  |---|---|---|---|
  | Stroke / speed | smallest | moderate | largest (still small) |
  | Bounce freq vs servo ω_n (~2.5 Hz) | 2× above, heavy attenuation | 1.1–1.4× above | ≈ equal |
  | Steps per bounce @ 50 Hz | 10 | 14–17 | 20 |
  | Landing drift per 1° tilt (§7) | 7 mm | 14–21 mm | 28 mm |
  | Perception + actuation latency share (~100 ms assumed [U]) | ~50% of flight | 29–35% | 25% |

  Net: 5 cm is the hardest *timing* problem, 20 cm the hardest *precision and camera-FOV* problem. Start curriculum training around 10–15 cm.

### Gaps
- COR of the user's actual paddle/ball pair is unknown; published racket values were not retrieved. Measure it (§9, step g).
- The closed-form passive-stability window for paddle acceleration at impact, which I recall as −2g(1+e²)/(1+e)² < a_p < 0, could not be verified from sources [U].
- The D=32 term and the firmware's ramp/rate limiter are not in the linearised bandwidth estimate.

---

## 6. Launching the ball from rest on a level paddle

### Takeaway
The ball leaves the paddle only once the paddle's acceleration drops below −g. A direct "throw" therefore needs the paddle at **v_sep = √(2gh) = 0.99 / 1.40 / 1.72 / 1.98 m/s** for 5/10/15/20 cm, followed by braking harder than 1 g. With a full-size paddle this is marginal for 5 cm and infeasible for 10–20 cm on a 5 V/7.4 V arm. It is impossible at LeRobot's `Acceleration=254` at levers ≤ 25 cm. Two strategies are far easier:
- **Drop-and-hit:** drop the paddle at slightly more than 1 g so the ball floats free, then hit it at 0.25–0.4 m/s.
- **Small throw + pump:** throw to ~2 cm, then add 0.22–0.40 m/s per hit.

### Cited Findings
- [S] LeRobot writes `Acceleration=254` to every motor at connect — [lerobot feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)
- [D] Acceleration unit 8.7°/s²; 0 = maximum (no ramp) — [Feetech STS memory-table analysis](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- [M] Firmware velocity cap ~3400 steps/s ≈ 5.2 rad/s; identified 5.07–5.29 rad/s — [BAM actuator.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/actuator.py), [BAM params](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V)
- [M] Armature 0.0216–0.0261 kg·m² (BAM); 0.028 (Menagerie) — [BAM params](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V), [Menagerie so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)

### Inferences
- [C] Separation physics. While the paddle's acceleration a_p ≥ −g the ball stays in contact. At the first instant a_p < −g, the ball separates with the paddle's velocity v_sep. Its apex above the separation point is v_sep²/2g, so **v_sep = √(2gh)**. Stroke = d₁ (accelerate from rest at a_up: v²/2a_up) + d₂ (paddle overshoot after separation while braking at a_dn > g: v²/2a_dn). Note d₂ → h as a_dn → g: the paddle chases the ball upward.

  | apex | v_sep | a_up 10 m/s², a_dn 1.2g | a_up 10, a_dn 2g | a_up 20, a_dn 2g |
  |---|---|---|---|---|
  | 5 cm | 0.99 m/s | 4.9 + 4.2 = 9.1 cm, t_acc 99 ms | 7.4 cm | 5.0 cm, 50 ms |
  | 10 cm | 1.40 | 18.1 cm, 140 ms | 14.8 cm | 9.9 cm, 70 ms |
  | 15 cm | 1.72 | 27.2 cm, 172 ms | 22.2 cm | 14.9 cm, 86 ms |
  | 20 cm | 1.98 | 36.3 cm, 198 ms | 29.6 cm | 19.8 cm, 99 ms |

- [C] Joint speed needed at separation:
  - wrist hinge, paddle contact at r = 0.15 / 0.25 m: 6.6 / 4.0 rad/s (5 cm), 9.3 / 5.6 (10 cm), 11.4 / 6.9 (15 cm), 13.2 / 7.9 (20 cm);
  - elbow translation (0.135 m lever): 7.3 rad/s (5 cm) to 14.7 rad/s (20 cm).

  Against the ~5.2 rad/s cap (≈3.7 at 5 V), **only a 5 cm throw with a ≥0.25 m lever is reachable by one joint.** Larger throws need several joints summing velocity (v = Σ ω_i·r_i).
- [C] Braking > g requires joint deceleration > g/r: 73 rad/s² at 0.135 m, 65 at 0.15, 55 at 0.18, 39 at 0.25, 33 at 0.30. `Acceleration=254` caps this at 38.6 rad/s², which rules out every lever ≤ 0.25 m. **Set `Acceleration=0` for launch (and model it in sim).**
- [C] Torque-limited accelerations with `Acceleration=0`, single joint, m_p = 0.17 kg, armature 0.0216–0.0261, from α = (τ_stall ∓ τ_gravity)/J. Gravity helps braking an upward stroke and resists the upward push.

  | Strategy | Supply | Upward accel at paddle | Braking / downward accel at paddle |
  |---|---|---|---|
  | Wrist hinge, CoM 0.15 m | 5 V | 4.7–5.5 m/s² | **8.2–9.6 m/s²** (< g) |
  | Wrist hinge, CoM 0.15 m | 7.4 V | 7.5–8.7 | 10.9–12.8 |
  | Wrist hinge, CoM 0.15 m | 12 V servo | 12–15 | 16–19 |
  | Wrist hinge, CoM 0.25 m | 5 V | 5.4–6.1 | **12.4–14.0** (> g) |
  | Wrist hinge, CoM 0.25 m | 7.4 V | 9.1–10.3 | 16–18 |
  | Elbow translation, wrist compensating, CoM 0.25 m | 5 V | **1.0–1.2** | 9.8–11.3 |
  | Elbow translation, wrist compensating, CoM 0.25 m | 7.4 V | 3.3–3.8 | 12–14 |

  In the translation strategy the wrist servo must also counter-rotate its own armature: ≈ J_arm·α_e ≈ 1.7–2.3 N·m at those decelerations, which exceeds its stall. So pure-translation launches at > g are not achievable. The hinge launch needs a 7–10 cm accelerating stroke, i.e. ~20° of wrist swing at r = 0.25 m, so the paddle is far from level while the ball rides on it, which encourages rolling.
- [C] **Drop-and-hit launch.** Drive the paddle downward at a_p < −g (motor and gravity add, so the wrist hinge at 0.25 m reaches 12–14 m/s² even at 5 V). The gap grows as ½(k−1)g·t²: 2.5 mm in 50 ms at 1.2 g, 6.1 mm at 1.5 g. Let the ball fall 1–2 cm (v_in 0.44–0.63 m/s), then strike upward with:

  | target apex | 1 cm fall | 2 cm fall |
  |---|---|---|
  | 5 cm | 0.31–0.38 m/s | 0.22–0.30 m/s |
  | 10 cm | 0.53–0.61 | 0.44–0.53 |
  | 20 cm | 0.83–0.94 | 0.75–0.86 |

  (range over e = 0.9…0.75). This converts the launch into an ordinary hit and is feasible for 5–10 cm on the stock arm.
- [C] **Pumping.** To raise the apex from h₁ to h₂ in one hit, v_p = (v₂ − e·v₁)/(1+e): 0.22–0.30 m/s (2→5 cm), 0.27–0.38 m/s (5→10), 0.24–0.38 m/s (10→15), 0.23–0.40 m/s (15→20). A low first launch plus 1–3 pumping hits reaches any target apex within the arm's speed envelope.
- For RL: reward or curriculum the launch separately. It is a discontinuous, torque-saturated manoeuvre, so its sim fidelity depends on torque limits, armature and the `Acceleration` register, the parameters least well known today.

### Gaps
- The true maximum acceleration with `Acceleration=0` (and what `Acceleration_Multiplier`, reg 86, does) is unknown; measure it.
- Plug-braking torque (back-EMF adds to the drive voltage when reversing) may exceed stall but is current-limited by `Protection_Current` (default 511 × 6.5 mA ≈ 3.3 A [D, third-party]). Not quantified.
- Ball rolling and rubber friction during the pre-launch stroke were not modelled.

---

## 7. Lateral control: how much paddle tilt corrects (or causes) how much drift?

### Takeaway
At steady state a paddle tilted by θ gives the ball a horizontal speed ≈ 2vθ. **Landing drift ≈ 8hθ** and apex drift ≈ 4hθ: per degree that is 7 / 14 / 21 / 28 mm of landing drift at 5 / 10 / 15 / 20 cm. Recentring a 1 cm error in one bounce needs 1.43° / 0.72° / 0.48° / 0.36°. One encoder step (0.088°) equals 0.6–2.5 mm of landing error, and ±0.5° of backlash up to 3.5–14 mm. At launch, tilt matters half as much (apex drift ≈ 2hθ) if the paddle moves along its normal.

### Cited Findings
- [D] Position resolution 0.088°/step (4096/rev) — [Feetech STS3215 datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)
- [D] Default dead zone ±1 step (0.087°) — [Feetech STS memory-table analysis](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- [U] Backlash "+/- 0.5°" (stated in the model, not measured) — [Menagerie so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)

### Inferences
- [C] Bounce derivation, assuming a frictionless tangential impact with no spin. Relative incoming speed w = v_in + v_p along the vertical. Normal component reversed with factor e, tangential component kept, so after impact v_x = (1+e)·w·sinθ·cosθ ≈ (1+e)·w·θ. At steady state (1+e)·w = 2v, so v_x ≈ 2vθ. Flight time 2v/g gives landing drift Δx = 4v²θ/g = **8hθ** and apex drift **4hθ**.

  | apex | landing drift / 1° | apex drift / 1° | tilt to recentre 1 cm in one bounce | landing error per 0.088° step | per 0.5° backlash |
  |---|---|---|---|---|---|
  | 5 cm | 7.0 mm | 3.5 mm | 1.43° | 0.61 mm | 3.5 mm |
  | 10 cm | 14.0 | 7.0 | 0.72° | 1.23 | 7.0 |
  | 15 cm | 20.9 | 10.5 | 0.48° | 1.84 | 10.5 |
  | 20 cm | 27.9 | 14.0 | 0.36° | 2.46 | 14.0 |

  To cancel an incoming lateral velocity u: θ ≈ −u/(2v).
- [C] Launch:
  - If the contact point moves along the paddle normal (wrist hinge; the axis lies in the paddle plane), the ball leaves with v_x = v_sep·sinθ. Apex drift = 2hθ (1.75 / 3.5 / 5.2 / 7.0 mm per degree), and landing back at launch height = 4hθ.
  - If the paddle translates purely vertically while tilted, tilt adds no v_x at separation. But the ball (thin shell, I = ⅔mr²) rolls downhill at 0.6·(g + a_p)·sinθ during the stroke; e.g. 1° for 100 ms at a_p = g gives Δv_x ≈ 0.02 m/s, or ≈2 mm drift by a 5 cm apex.
- Tilt axes: fore-aft tilt comes from wrist_flex, left-right from wrist_roll (the paddle face is horizontal and the handle points forward). Wrist_roll is lightly loaded: the paddle's inertia about its long axis is ~2.6 × 10⁻⁴ kg·m², much smaller than the armature.
- With the wrist-hinge stroke, v_p ∝ contact radius r, so where the ball lands along the paddle changes the outgoing speed. This couples lateral position to apex height and should be in the sim.
- Rubber friction and ball spin, ignored here, will change the tangential response. Expect the real tilt gain to differ from 2vθ, and identify it from ball tracks.

### Gaps
- Real tangential COR and spin coupling for rubber are not quantified.
- No measured SO-101 backlash.

---

## 8. Payload: a store-bought 150–190 g paddle on the SO-101 wrist (torque, grip, heat)

### Takeaway
A typical assembled bat weighs ~150–190 g: blade 65–100 g plus two rubbers, head ≈ 157 × 150 mm, handle ≈ 100 mm. **Held by its handle in the gripper, its CoM sits ≈ 0.25–0.27 m from the wrist-flex axis**, not 12–18 cm; only a short flange adapter in place of the gripper achieves ~15 cm. At 5 V that loads:
- the wrist flex to ~40% of stall;
- the elbow and shoulder to **~80% of stall** in a forearm-horizontal pose, i.e. at the factory 80% overload threshold;
- the shoulder **above stall** with the arm fully extended.

The gripper cannot hold the handle rigidly (≤50% torque in LeRobot, dropping to ~20% under sustained stall). Use a printed clamp/adapter, cut the handle, and ideally run 7.4 V or 12 V servos.

### Cited Findings
- [D] Example blade: TSP Levant, "Blade Head: 157mm length by 150mm wide; Handle Length: 100mm; Weight: 84g", ±7 g wood variation — [Megaspin TSP Levant](https://www.megaspin.net/store/default.asp?pid=tsp-levant). Butterfly Zhang Jike ALC: 82–88 g, 150 × 157 mm — [thegodofsports listing](https://thegodofsports.com/products/butterfly-zhang-jike-alc-table-tennis-blade)
- [M] An assembled bat at 183 g is "quite normal" / "quite average". "Blades usually range approximatively from 65 to 100g", "rubbers from 30 to 60g" — [PingSkills forum: how heavy should your bat be](https://www.pingskills.com/table-tennis-forum/how-heavy-should-your-bat-be). Decathlon notes recreational bat weight "may vary (+/-20 g)" (search snippet) — [Decathlon TTR 50 listing](https://decathlon.com.kw/products/table-tennis-ittf-club-all-round-recreational-bat-ttr-50-2)
- [S] SO-101 link geometry and masses: see the URDF entry in §4 — [so_arm101.urdf](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf). Isaac config comments put the moved masses at "Wrist Pitch … (~0.24kg)" and "Elbow … (~0.38kg)" — [isaac_so_arm101 so_arm101.py](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/so_arm101.py)
- [S] LeRobot gripper limits: `Max_Torque_Limit=500` (50%), `Protection_Current=250`, `Overload_Torque=25` — [lerobot so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- [D] On overload the servo unloads to 20% of blocking force for 2 s "until a new command is triggered" — [Feetech STS3215 datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf). Defaults: overload threshold 80%, protection time 2 s, holding torque after overload 20%, max temperature 70 °C — [Feetech STS memory-table analysis](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- [M] On the user's arm, even without a payload, softer gains (P12/D40) plus heavy damping left "not enough force to lift against gravity", and they reverted to P16/I0/D32 — [so101/README.md](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/README.md)

### Inferences
- [C] Where the paddle CoM ends up:
  - Wrist_flex axis → gripper TCP ≈ 0.061 + 0.098 = 0.159 m (URDF).
  - Estimated paddle CoM ≈ 155 mm from the butt, assuming ~30 g of handle wood centred 50 mm from the butt and ~140 g of head + rubbers centred 178 mm from it [U].
  - **Handle gripped at mid-length:** CoM ≈ 0.159 + 0.05 + 0.055 ≈ **0.26 m** from the wrist-flex axis.
  - **Handle cut to a ~3 cm stub and gripped:** ≈ 0.21–0.23 m.
  - **Gripper removed, blade stub bolted to the wrist-roll flange via a printed adapter:** ≈ 0.13–0.16 m.

  So the coordinator's 12–18 cm is only reached with the flange-adapter option.
- [C] Static gravity torques, paddle level and pointing forward. Pose A: upper arm vertical, forearm horizontal. Pose X: whole arm horizontal (worst case).

  | m_p | CoM from wrist axis | wrist_flex | elbow = shoulder (pose A) | shoulder (pose X) |
  |---|---|---|---|---|
  | 0.15 kg | 0.12 m | 0.29 N·m | 0.82 | 1.41 |
  | 0.17 | 0.15 | 0.36 | 0.92 | 1.53 |
  | 0.17 | 0.25 | 0.53 | 1.09 | 1.70 |
  | 0.19 | 0.18 | 0.45 | 1.04 | 1.66 |
  | 0.19 | 0.30 | 0.67 | 1.26 | 1.89 |

  As a fraction of stall (5 V ≈ 1.35, 7.4 V 1.91, 12 V servo 2.94 N·m), for the realistic case m_p = 0.17 kg at 0.25 m:
  - wrist 39 / 28 / 18%;
  - elbow & shoulder in pose A **81 / 57 / 37%**;
  - shoulder in pose X **126% (cannot hold) / 89 / 58%**.
- [C] Wrist_roll: if the handle is coaxial with the roll axis, the static torque is ≈ 0 (0.025 N·m per 15 mm of CoM offset). Its load inertia is tiny (~2.6 × 10⁻⁴ kg·m²), so roll is the least-stressed joint and a good actuator for left-right tilt.
- [C] Dynamic torque (wrist hinge, m_p 0.17 kg; armature dominates J):
  - steady bouncing at e = 0.85: ≈ 0.8–0.9 N·m;
  - steady bouncing at e = 0.75: ≈ 1.1–1.3 N·m;
  - launch: stall-level torque.

  Through the elbow/shoulder, add 1.0–1.1 N·m of gravity in pose A. At 5 V the arm spends much of each bounce cycle at 60–90% of stall.
- **Overload and overheating risk:**
  - Sustained load above the overload threshold (factory 80%) for more than 2 s triggers protection, and the joint drops to ~20% torque. In pose A at 5 V the elbow and shoulder hold ~81% statically, so **the arm may sag or drop the paddle simply from holding the pose.**
  - Repeated strokes at 60–90% duty heat the motors (I²R) toward the 70 °C cut-off.
  - Monitor `Present_Temperature` and `Present_Load`, keep episodes short with cool-down gaps, and prefer poses that keep the forearm closer to vertical. Example: wrist directly above the elbow, flexed 90° so the paddle points forward. Then elbow gravity torque ≈ m_p·g·d_p ≈ 0.42 N·m at 0.25 m.
- [C] **Gripper hold.** Moment the grip must resist about the grip point (CoM ≈ 0.105 m beyond it), M = m(g+a)d: 0.18 N·m static, 0.22 N·m during steady strokes (2.5 m/s²), 0.35–0.43 N·m during launch (10–14 m/s²). Capability, a rough estimate with unverified inputs (jaw lever ~5–7 cm, μ ≈ 0.5 PLA-on-wood, ~4 cm contact length, gripper torque 20–50% of ~1.35 N·m): friction couple ≈ 0.08–0.26 N·m [U]. **Expect the handle to rotate or slip in the jaws.** Even without slip, the jaw compliance becomes an unmodelled spring at impact.
  - Recommendation: a printed adapter that clamps or bolts the handle (or a cut stub) to the gripper body/fixed jaw, or replaces the gripper with a wrist-roll flange mount. Add tape or zip-ties as a backup.
  - Then disable gripper torque (or leave it lightly closed) so the gripper servo isn't stalled into protection or heated by the 25% LeRobot threshold.
- [C] **Cutting the handle short:**
  - moves the CoM 3–7 cm closer;
  - saves ~10–20 g of handle wood [U];
  - cuts paddle inertia by ~30–50% and static wrist torque by ~0.05–0.12 N·m;
  - reduces elbow/shoulder torque by the same lever change × m_p·g.

  The dynamic benefit at the wrist is modest because armature dominates; the static relief at the elbow and shoulder is the main gain. Removing the backhand rubber saves another 30–60 g. Both are legal for this purpose.
- Supply: if the arm is 7.4 V servos on the kit's 5 V PSU, moving to a 7.4 V supply (within the servos' rating [U]) adds ~40% torque and speed. A 4 A, 5 V supply shared by six servos, each able to draw ~2.7 A at stall (12 V spec), can also sag under simultaneous high load [U].

### Gaps
- Gripper maximum opening, jaw length and clamp force on the SO-101 are not documented in the sources found. Measure whether a 25–30 mm handle even fits.
- Thermal time constants of the STS3215 (minutes to 70 °C at a given duty) are unknown.
- The user's servo variant and PSU are unknown.
- Real paddle mass split and CoM: weigh the paddle and balance it on a knife edge.

---

## 9. A ≤30-minute system-identification procedure for zero-shot sim-to-real (SO-101 + paddle, through LeRobot)

### Takeaway
Run a raw `FeetechMotorsBus` loop, not `robot.get_observation()`, which reads position only. Use the deployment register settings with the paddle mounted. On each primary joint (wrist_flex, elbow_flex, shoulder_lift, wrist_roll), run step ladders, sine dwells at the bounce frequencies (2–6 Hz), a chirp, max-effort moves and slow triangles, logging commanded vs measured position, velocity, load, current and temperature with timestamps around every transaction. Fit a BAM-style model (dead time → rate limit → [accel ramp] → saturated P(D) duty → DC motor → armature + friction) by replaying the logged command stream in sim. Validate on held-out chirps and a bounce-like multi-joint trace. Gain/phase at 2.5–5 Hz is the acceptance metric that matters.

### Cited Findings
- [S] LeRobot bus API: `sync_read(data_name, motors=None, *, normalize=True, num_retry=0)`, `sync_write(data_name, values, *, normalize=True, num_retry=0)`, `write(data_name, motor, value, *, normalize=True, num_retry=0)` — [lerobot motors_bus.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py)
- [M] BAM's protocol: pendulum of known mass and length on one servo; predefined trajectories; logs position, speed, input volts, goal and torque-enable; timestamp = midpoint of the read; kp fixed (32) and D=0 during recording; fit with `python -m bam.fit` per actuator and model (m1–m6). Raw data published as `feetech_sts3215_raw.zip` — [BAM record.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/record.py), [BAM README](https://github.com/Rhoban/bam), [BAM actuators doc](https://bam.readthedocs.io/en/latest/usage/actuators.html)
- [M] Validation by tuning sim armature, friction and delay with CMA-ES to match real joint trajectories, then overlaying plots — [UWLab Sim2Real SysID](https://uw-lab.github.io/UWLab/main/source/publications/omnireset/sim2real.html). Step-response comparison of real vs simulated actuators, normalised 0–1; delay in the damping loop hurt stability more than delay in the stiffness loop — [arXiv 1701.05929](https://arxiv.org/pdf/1701.05929)
- [D] Feedback registers: Present_Load = duty in 0.1% (1000 = 100%); Present_Current 6.5 mA/unit; Present_Velocity in steps/s; Present_Voltage 0.1 V — [Feetech STS3215 datasheet](https://www.mantech.co.za/datasheets/products/sts3215-200620a-1.pdf)

### Inferences
**Procedure** (total ≈ 25–30 min, run from the Mac; this is an I/O-bound check, not compute):

0. **Prep, 3 min.**
   - Mount the paddle exactly as it will be deployed.
   - Connect via `SO101Follower`, then write the deployment settings explicitly: P, I, D, `Acceleration` (test both 254 and 0), `Return_Delay_Time=0`.
   - Read back and save: those registers, plus `Goal_Velocity`, `Torque_Limit`/`Max_Torque_Limit`, `Protection_Current`, `Overload_Torque`, `Protection_Time`, `Protective_Torque`, `DTs` (81), `Maximum_Velocity_Limit` (84), `Maximum_Acceleration` (85), `Acceleration_Multiplier` (86), firmware version, and `Present_Voltage`.
   - Weigh the paddle and locate its CoM.
1. **Bus benchmark, 2 min.** 2000 iterations each of (a) `sync_read Present_Position`, (b) `sync_write Goal_Position` + `sync_read Present_Position`, (c) the same with `Present_Velocity` added. Record per-iteration dt (p50/p95/p99/max) and error counts. This sets the identification sampling rate and the deployable control rate.
2. **Per-joint tests**, with the other joints holding the bounce pose. Raw steps (`normalize=False`). Stream goals at the planned policy rate (e.g. 50 Hz, zero-order hold) and sample as fast as step 1 allows.
   - a. *Step ladder* (~45 s/joint): ±0.5°, 1°, 2°, 5°, 10°, 20°, 40° about the bounce pose, 0.5 s holds, both directions, with and against gravity. Gives dead time, rise time, overshoot, deadband, the velocity plateau (v_max) and the accel limit.
   - b. *Sine dwells* (~60 s/joint): 1, 2, 2.5, 3, 3.5, 4, 5, 6 Hz, 2 s each, at two amplitudes: paddle-tip motion ≈ 5 mm and ≈ 20 mm (≈ 1.1° and 4.6° at a 0.25 m lever). Gives the gain/phase (Bode) curve in the bounce band and its amplitude dependence.
   - c. *Chirp* (~20 s/joint): 0.5→8 Hz over 8 s, at small and large amplitude. Used for validation.
   - d. *Max-effort moves* (wrist_flex, elbow; ~30 s): 30–40° steps and a scripted launch-like "drop" with `Acceleration` = 254 vs 0. Gives the real acceleration and braking limits, and current/load peaks.
   - e. *Slow triangle* (~20 s/joint): ±3° at 0.25 Hz through the gravity-load reversal. Gives deadband, stiction and Coulomb friction.
   - f. *Paddle-off repeat* of (a)+(b) on wrist_flex only (~2 min). Separates armature from paddle inertia (ΔJ is known from the paddle's mass and CoM).
   - g. *Ball drop on the held paddle* (~3 min): drop from ~30 cm at the centre and edge, filmed with the overhead camera at its highest fps (a phone at 240 fps is better). COR e = √(h_rebound/h_drop) per location. This captures paddle + arm compliance.
   - h. *Multi-joint validation trace* (~1 min): a scripted bounce-like stroke (wrist + elbow), plus one drop-and-hit launch.

   Budget: 4 joints × ~3 min + extras ≈ 25 min. Keep `Present_Temperature` < ~60 °C and pause if needed.
3. **Logging per iteration:**
   - `t_w0`, `t_w1` (around `sync_write`) and `t_r0`, `t_r1` (around `sync_read`), from `time.perf_counter()`;
   - goal (raw steps), Present_Position, Present_Velocity;
   - every ~10th iteration: Present_Load, Present_Current, Present_Voltage, Present_Temperature;
   - bus errors (catch, flag, don't retry synchronously).

   Use the read midpoint as the sample time (BAM). Save to `.npz`/parquet with the register dump. Minimal loop skeleton (API verified in 0.6.1):
   ```python
   bus = robot.bus  # after robot.connect()
   for k, (t_cmd, goal) in enumerate(schedule):          # goal: dict motor -> raw steps
       while time.perf_counter() < t0 + t_cmd: pass
       a = time.perf_counter(); bus.sync_write("Goal_Position", goal, normalize=False); b = time.perf_counter()
       pos = bus.sync_read("Present_Position", normalize=False); c = time.perf_counter()
       vel = bus.sync_read("Present_Velocity", normalize=False); d = time.perf_counter()
       log.append((a, b, c, d, goal, pos, vel))
   ```
4. **Fit**, on the Mac in seconds per joint, or on the cluster if using CMA-ES over many traces:
   - *(i) Quick black-box per joint:* dead time τ_d (cross-correlation on small steps), second-order ω_n and ζ (fit to the sine-dwell gain/phase), v_max (step-velocity plateau), a_max (initial velocity slope on large steps), deadband (triangle).
   - *(ii) Physical BAM-style model:* simulate the logged goal stream with the same timing (ZOH at the command rate, delay τ_d), then minimise position (and velocity) error with `scipy.optimize.least_squares` or CMA-ES. Parameters: τ_d, K_e (= kp·error_gain), K_d, kt, R (or the product kt/R and kt²/R), armature, Coulomb, viscous, v_max, accel ramp. Initialise from BAM m1. Include gravity and the paddle via the sim's own rigid-body model, so the fitted parameters are actuator-only.
5. **Validate:**
   - Replay held-out chirps and the multi-joint trace in sim; overlay sim vs real q(t).
   - Compare gain and phase at 2.5 / 3.5 / 5 Hz, step overshoot and settling, and v_max / a_max.
   - Suggested acceptance criteria (my proposal): phase error < 15° and gain error < 15% across 2.5–5 Hz; launch-trace peak velocity within 10%.
   - Then domain-randomise around the fit (e.g. K_e, armature, friction ±20%, delay ± one control period, COR ±0.05) and re-check that the policy works across that range in sim.

**Identification limits specific to the STS3215:**
- *Quantisation:* 0.088°/step. A 5 mm stroke at 0.25 m is only ≈13 steps peak-to-peak, so use the larger sine amplitude for phase estimates. Finite-difference velocity from 1-step changes is coarse: 0.08 / 0.15 / 0.31 rad/s at 50 / 100 / 200 Hz. Prefer fitting position, or use `Present_Velocity`, whose internal filtering and window are unknown [U].
- *Readback timing:* the sample instant is unknown within the transaction (±½ of ~1–3 ms) plus macOS USB jitter. Midpoint stamping and many cycles average this out; dead-time estimates finer than ~2 ms are not trustworthy.
- *Encoder location:* the magnetic encoder reads the output shaft, so gear backlash and compliance inside the loop are not directly visible. Use an external measurement (overhead-camera marker on the paddle at high fps, or a phone IMU taped to the paddle) for impact compliance and backlash.
- *Dead zone:* the default ±1 step means the servo will not correct sub-step errors. Fit it explicitly.
- *Extra registers cost time:* each additional `sync_read` costs ≈0.6 ms of wire time plus a USB round-trip. Read load, current and temperature at a lower rate.

### Gaps
- No one has published this procedure, or validation metrics, specifically for the SO-101 with a payload. The acceptance thresholds above are my proposal, not literature values.
- `Present_Velocity` filtering and `DTs` are undocumented in the sources found.
- Whether `Acceleration=0` removes the ramp entirely on the user's firmware must be verified in step 2d.
