# How a run can start and how it can fail (SO-101 ping-pong bouncing, real rig)

Rig as of 2026-10-31:
- **Arm:** 12 V SO-101; the paddle is taped in the jaws, with wrist offsets fitted by `real/level.py` (flex 5.4°, roll 95.5°).
- **Camera:** an overhead C920 at 30 fps. An HSV orange finder gives `(u, v, r)`, and depth comes from the ball's radius in pixels.
- **Camera pose:** measured with `real/calibrate_camera.py`.
- **Runner:** `real/bounce_run.py` runs at 50 Hz on the Mac over USB. Its state machine goes armed → active → returning:
  - start: 2 detections within 0.15 s, within 8 cm laterally of home and 5 cm below to 50 cm above it;
  - lost: no detection for 0.6 s, or the latest detection is more than 18 cm out laterally or more than 5 cm below home;
  - returning: a 0.4 s ramp back to the hold command.
- **Policies:** trained in `bounce/env.py`. Ranges below are written (nominal, low, high) with the randomisation group that scales them; a group at level 0 means nominal.

How to read the tables:
- **Likelihood:** how often the case should come up on this rig (H, M or L).
- **Test:** how it is exercised.
  - *knob* means an existing env option.
  - *forced* means an eval-only option added for this study (`env.forced[...]`, `scripts/scenario_test.py`).
  - *harness* means the fake-bus and fake-camera runner test (`scripts/runner_scenarios.py`).

## A. Ways a run starts

| # | Start | Likelihood | In training? | Test |
|---|---|---|---|---|
| A1 | Ball set down at the centre, at rest | H | Yes. `rest_start_prob` 0.35 | forced `rest_start`, `rest_radius` 0.1 |
| A2 | Set down off-centre (about half way to the rim) | H | Yes. `rest_spread` up to 0.75 of the blade (task group) | `rest_radius` 0.5 |
| A3 | Set down near the rim (85-95 %) | M | Partly. Training goes only to 75 % at full task level | `rest_radius` 0.85 |
| A4 | Set down on a paddle that is not quite level, so the ball is already rolling when the policy starts | **H**. Hand levelling with a phone is good to about 1-2°, and this session saw a ball roll off at about 1° | Partly. `level_error` 0.015 rad (0.9°) on the task group, `table_tilt` 0.026 rad on the physics group | forced gravity tilt 1/2/4° after reset |
| A5 | Dropped from 5-20 cm over the centre | H | Yes. 65 % of starts are in flight: 3-18 cm, vz ±1 m/s, up to ±7 cm off-centre | forced `drop_height` (drop test done: 98-100 % for 10-20 cm) |
| A6 | Dropped from 30-50 cm | M | **No.** Training starts at most 18 cm up, and the 30 cm ceiling ends the episode | `drop_height` 0.3/0.4 with the ceiling raised (v12-base 57-96 % at 30 cm, 24-28 % at 40 cm) |
| A7 | Dropped or bounced **beside** the paddle (8-15 cm off-centre), so the arm must move to catch it | M (the user asked for it) | **No.** At most 7 cm off-centre | forced `drop_offset` 0.08-0.15 plus sideways speed |
| A8 | Tossed in with sideways speed (0.3-0.5 m/s) and spin | M | Partly. Sideways speed up to 0.15 m/s, spin up to 20 rad/s | forced `drop_vlat`, `drop_spin` |
| A9 | Bounced in by hand, so the ball arrives moving upward | M | Yes. vz up to +1 m/s | forced `drop_vz` +1.5 |
| A10 | **Held above the paddle where the camera can see it, then released after 0.1-1 s** | **H** (the user's habit) | **No.** In training the ball always moves from t = 0. The runner triggers on the held ball, so the policy runs on a ball hovering in place | harness (trigger time, paddle motion while held); new forced `held_s` |
| A11 | Held with the hand hiding it, then released | M | N/A. The runner starts about 70 ms after the ball is first seen falling, which is the same as A5 with a short lag | harness |
| A12 | Hand still hovering after the release: false detections near the ball, occlusion | H | Partly. `hand_outlier_p` 0.15 per frame for 0.4 s after a tap, sensing group only, so none at nominal | forced `held_s` plus hand outliers |
| A13 | Ball arrives while the arm is still on its 0.4 s return ramp | M | Runner only: detections are ignored until armed | harness |
| A14 | A second ball, or the same ball still bouncing on the table nearby, after a drop | M | Runner only. The start zone excludes balls more than 5 cm below home | harness |
| A15 | Ball placed before the runner's 3 s sag trim has finished | L | Runner only. Arming waits for Enter after the trim | none needed |

## B. Ways it fails

### Disturbances

| # | Failure | Likelihood | In training? | Test |
|---|---|---|---|---|
| B1 | Hand taps and nudges during play | H | Yes. `push_rate_hz` 0.4 on the task group, `push_std` 0.15-0.45 m/s | forced taps at 1 Hz at nominal |
| B2 | Air currents from AC or a fan | L-M | Yes. `wind` 0.3 m/s on the physics group | wind 0.5 m/s |
| B3 | Table bumps or base vibration | L | Partly. `paddle_jitter` up to 0.002 rad | not tested |

### Servos and hardware

| # | Failure | Likelihood | In training? | Test |
|---|---|---|---|---|
| B4 | 12 V servos are 3.4× stiffer than the 5 V model the v5 policies learned on | **certain** for v5 | v12 policies: yes (`servo_variant` 12v, fitted to 1.45° rms). v5: no | compare v5 and v12 under the 12 V servo model |
| B5 | Servo heating over a long session weakens the servos by 10-15 % | M | Yes. `strength` 0.85-1.15 | strength 0.85 |
| B6 | An arm joint trips overload protection (the gripper did this session). After a trip the servo reports an error and LeRobot treats it as missing | L-M | Partly. The `overload` termination teaches the policy to avoid it. The runner does not handle a mid-run trip: a read error ends the run | peak load from the sims |
| B7 | Gravity sag of the hold pose | certain | Yes. Hold command plus sag trim, `supply_r` | none needed |
| B8 | Gear backlash | certain | Yes. `backlash` 0.2-1° | none needed |
| B9 | USB stalls (commands and readings freeze for 1-4 steps) | M | Yes. `glitch_p` up to 3 % per step | glitches 3 % at nominal |
| B10 | Mac loop overruns (median 4-6 ms, p99 9-13 ms against a 20 ms period) | L | Yes. `ctrl_jitter_ms` 4 | not needed |
| B11 | Command delay (fitted 30 ms) | certain | Yes. `act_delay_ms` (30, 8, 40) in v12 | 40 ms |

### Perception

| # | Failure | Likelihood | In training? | Test |
|---|---|---|---|---|
| B12 | Camera latency above the model. Forum reports put C920 latency at 4.5-5.5 frames, about 150-180 ms, depending on the capture pipeline | **M-H** (not measured on this Mac) | **Capped at 120 ms** (`cam_lat` clamp) | latency 120 and 150 ms |
| B13 | **Wrong frame timestamps.** The runner assumes 35 ms from capture to read (`latency_s`). If the true figure is 100 ms, every detection's age is about 65 ms too young | **H** until latency is measured | **No.** `age_bias` ±25 ms only | age bias +50 and +100 ms |
| B14 | Depth-from-radius error. Mask shading and blur bias the radius; ±0.5 px on 12 px is 4 % of about 0.8 m, roughly 3 cm | H | Yes. `overhead_depth` 2-7 cm noise plus drift, `depth_scale` ±3 % | depth noise 7 cm; scale error 10 % |
| B15 | Dim light: auto exposure gives 15 fps, long exposure and blur | M (evenings) | Yes. `light`; 15 fps half the time when light < 0.3 | 15 fps and dim |
| B16 | Detection dropout bursts (glare, rim, motion blur) | M | Yes. `burst_in` up to 0.03, `burst_out` down to 0.2 | bursts at maximum |
| B17 | **False positives.** Skin, orange objects, red rubber, or orange tape or logos on the blade. A blob that sits still on the paddle would trigger the runner at once and look like a resting ball forever. The blade's radius is about 31-43 px at 0.8-1.1 m, and `BallFinder.max_r` is 40 | M | Partly. `outlier_p` up to 2 %, hand outliers | harness (static blob); one check with the real camera: run the finder on a frame with no ball |
| B18 | Hand occluding the ball | M | Yes (dropouts and occlusion) | covered by B16 |
| B19 | Motion blur at 2 m/s impacts | M | Yes. `blur_k`, `blur_miss_p` | covered by B15 |

### Calibration and setup

| # | Failure | Likelihood | In training? | Test |
|---|---|---|---|---|
| B20 | Camera pose error from `calibrate_camera` (synthetic: 1.3 cm median and 3.4 cm p90 error at 10 cm above the paddle) | M | Yes. `cam_rot_err` 1.5°, `cam_bias` 8 mm | rotation 3°, bias 2 cm |
| B21 | **Camera mounted somewhere other than the training spot** (−0.17, 0, 0.94 m ±10 cm). The user's mount is unknown; the failed calibration put it near x 0.25 m | **H** | **No** | camera above the paddle, in front, and low behind the base |
| B22 | Wrong focal length or `radius_scale`, so depth is scaled wrong | M | `depth_scale` ±3 % | scale error 10 % (B14) |
| B23 | **Wrist offset error after levelling: paddle 1-3° off level.** The model was 7.8° off before `level.py` was run | **H** | Partly (`level_error` 0.9°) | tilt 1/2/4° (A4) |
| B24 | Calibration error on another joint | M | Yes. `joint_offset` 2.3° on the sensing group | offset 2° on every joint |
| B25 | Paddle shifts in the tape during a session | M | No | covered by the tilt test |
| B26 | Paddle mass or reach differs (taped deeper or shallower) | M | Yes. Mass 80-220 g, reach 6-16 cm | mass 220 g; reach 8 cm |
| B27 | Ball or rubber restitution differs (old ball, worn rubber) | M | Yes. e0 0.72-0.92, rubber types | e0 0.75 |

### Runner logic (`real/bounce_run.py`)

| # | Failure | Likelihood | In training? | Test |
|---|---|---|---|---|
| B28 | Premature trigger on a held ball (A10) | H | No | harness |
| B29 | **Lost-ball false alarm.** "Lost" uses one noisy detection: dz < −5 cm, or more than 18 cm laterally. With 3-7 cm depth noise, a ball at the bottom of a bounce sits about 2 cm above home, so one bad frame ends an active run. In sim, termination uses the true ball position | **H** | N/A | harness with depth noise |
| B30 | Re-arm while the ball is still in play after a 0.6 s detection gap. The return ramp pulls the paddle away | M | N/A | harness: 0.7 s burst |
| B31 | Start zone too narrow for beside-paddle starts (8 cm) | M | N/A | recommendation |
| B32 | Stale detections copied into the fresh observation at activation | L. The newest two detections are fresh by construction | N/A | harness |

## Incidents from this session and what fixed them

| Incident | Cause | Fix | Residual risk |
|---|---|---|---|
| Arm whipped during calibration | 90° wrist-roll map error | `joint_map` offsets and `level.py` | Refit is needed after re-taping the paddle (B23) |
| Camera "below the table" | Mirrored PnP solution from almost coplanar points | Overhead initial guesses, rejection checks, wider poses | B20 |
| Ball rolled off a resting paddle | About 1° tilt | Calibration now tapes the ball | A4/B23 |
| Gripper stopped answering | Overload trip | `use_gripper: false`, paddle taped | B6 |
| Policy over-hit | 5 V model on a 12 V arm | v12 policies | B4 |

## Sources

- [Robotics Center SO-101 community](https://admin.roboticscenter.ai/hardware/so-101/community) and [troubleshooting](https://oscardvs.github.io/haller_ws/docs/troubleshooting/) pages: an STS3215 stops after an overload until it gets a new command or a power cycle; check 12 V supply polarity and cables when a servo goes missing.
- [OBS forum, C910/C920 latency](https://obsproject.com/forum/threads/webcam-latency-question-logitech-c910-c920.10517/latest): about 4.5-5.5 frames at 30 fps (anecdotal). [Hackernoon, measuring webcam latency with OpenCV](https://hackernoon.com/how-to-measure-the-latency-of-a-webcam-with-opencv-1a3d4a86558): screen-flip method, about 200 ms on a C930e.
- [JuggleRL (arXiv 2509.24892)](https://arxiv.org/abs/2509.24892): sim-to-real juggling. System identification, randomised ball position and restitution, and cutting perception latency were what mattered.
- This session: `real/logs/run-*.npz` (real-arm dry runs), `outputs/v12/*` (12 V evaluations and drop tests), `research/randomization_critique*.md`.
