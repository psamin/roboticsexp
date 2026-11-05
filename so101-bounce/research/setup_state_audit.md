# Rig setup state audit (2026-11-04)

Why the saved real-arm setup goes stale between sessions, why the paddle looked "slanted" tonight, and what the code
should check. Code citations are at commit a988c69 (real/ is unchanged since 35a04d5). Nothing here was measured with
the arm's serial port open; numbers come from run logs, recorded video, the camera calibration files and git.

## Summary

- The saved files are correct and do reproduce the levelled hold. hold_pose.json, joint_map and the servo calibration
  match backups/v1. hold_pose.json equals the current code's q_nom plus joint_map to within 0.04 deg. The first armed
  period of each run sits at -0.16 deg pitch (Nov 2 21:23) and -0.43 / +0.01 deg (tonight).
- What goes stale is physical and unmonitored, mainly the overhead camera mount. It moved at least four times across
  two sessions: before 20:50 Nov 2, between 21:26 and 21:36 Nov 2 (unnoticed), between Nov 2 and 23:10 tonight, and by
  hand at 23:23. The only staleness check (real/bounce_run.py:173-176) compares wrist offsets, not the camera. It has
  printed on every real run since Nov 2 21:21 and cannot fire for a camera move.
- Tonight's camera position, read through the Nov 2 calibration, reported the ball about +1.9 cm toward the blade tip,
  +1.2 cm left and +1.4 cm high. This was confirmed two independent ways: image registration, and the 23:31
  recalibration.
- The earlier "new paddle sits 1.5-3.6 cm off the model" result (research/user_asks_coverage.md:275-286, commit
  09f0ea1) was measured on frames taken after the unnoticed 21:27-21:36 camera move. With the camera at its calibrated
  position (Nov 2 21:22-21:26 frames), the model outline lands on the new blade to within about 1 cm.
- Most important fix: a startup check that the camera has not moved relative to a reference frame saved at
  calibration. Next most important: a calibration/setup fingerprint stored in every run log.

## 1. State a real run depends on

"Detected today" means the runner notices without a human looking.

| State | Stored in | Last written | Invalidated by | Detected today? |
|---|---|---|---|---|
| Servo calibration (homing offset, min/max) | Servo EEPROM, plus real/calibration/so_follower/my_follower.json and LeRobot's copy | JSON Oct 8 (LeRobot), copied Oct 31, commit 9206c70. Identical to backups/v1 and LeRobot's copy (verified) | lerobot-calibrate or another tool rewriting EEPROM, servo swap, horn re-seated | Partly. real/arm.py:47-49 compares the two JSON files only. EEPROM is never read; LeRobot's `is_calibrated` (third_party/lerobot/src/lerobot/motors/feetech/feetech.py:228-245) is unused |
| P gain, Acceleration | Servo RAM (EEPROM P = 16 at power-up) | Every run | Power cycle | Yes. real/bounce_run.py:186-191 writes, reads back, stops on mismatch |
| D/I gains, torque and protection limits | Servo EEPROM | Never by this project | Another tool reconfiguring the servos | No. Only read by sysid check (real/arm.py:137-146) |
| Supply voltage (servo model 12v_r11 assumes 12 V) | Physical PSU | n/a | PSU swap or sag | No. Present_Voltage read only in real/speed_test.py:38 and real/sysid.py:118 |
| Servo temperature | Runtime | Every second | Heat | Yes. real/bounce_run.py:373-377 |
| Serial port, camera identity | real/config.yaml:2, :12 | Oct 31 | Different adapter or camera | Yes. Fails loudly (real/arm.py:38-41, :54; real/rig.py:56-73) |
| Wrist offsets (paddle angle in the jaws) | real/config.yaml:9 joint_map | fa0922d, Nov 2 21:21 | Paddle re-taped, turned or slipped; new paddle | No. The "model says the paddle is tilted" print (real/bounce_run.py:208-210) runs FK with the same offsets that define level, so it reads ~0 whatever the real paddle does (tonight 0.4, 0.0, 0.0 deg) |
| Hold pose | real/hold_pose.json | fa0922d, Nov 2 21:21 (= q_nom + offsets) | Same as wrist offsets. Also real/sysid.py:47-54 `set-hold` writes raw joint angles, a different convention, and real/bounce_run.py:6 and real/rig.py:5 still point to it | Partly. Inconsistency with joint_map shows in the print at real/bounce_run.py:208-210, but nothing stops the run |
| Blade position in the jaws | real/config.yaml:19 paddle_offset_cm [0, 0] | 3fc4a63, Nov 4 02:01 | New paddle, re-taping | No. real/camera_check.py is visual only, and its reading was confounded by the camera move (section 2) |
| Paddle identity and restitution | Not stored | n/a | Paddle swap (old white handle until ~21:13 Nov 2, new black handle after) | No |
| Gripper clamp | real/grip.json | Oct 31 | n/a: unused with use_gripper false (real/arm.py:27, :110) | n/a |
| Camera pose (extrinsics, camera relative to arm base) | real/camera_calib.json R, t | 0956b70, Nov 2 20:50. Rewritten 23:31 tonight (uncommitted) | Mount bumped or re-aimed, cable pulled at replug, arm base moved on the table | No. real/bounce_run.py:173-176 checks joint_map_offset_deg, which changes when the paddle is re-levelled, not when the camera moves |
| Camera intrinsics K (nominal FOV x fitted focal, no distortion) | real/camera_calib.json | Same | Capture mode or resolution change, focus breathing | No. The frame size is never compared with the calibration's image_size (real/rig.py:76-92, :249-255) |
| Ball size scale (radius_scale, rad) | real/camera_calib.json | Same | Different ball or colour, exposure change | No. The calibration does not record which ball was used |
| Camera latency | real/config.yaml:16 (written by real/calibrate_camera.py:138-146) | 62 ms (Nov 1), 48 ms (Nov 2 20:50), 62 ms (tonight 23:31) | Exposure and light (auto exposure is on), USB path | No. Not measured at runtime. Tonight's 23:11 and 23:15 runs used 48 ms |
| Exposure, white balance, focus | Camera RAM | Every open | Replug | Yes, re-set on each open (real/rig.py:134-146). Focus locked at 30 on tonight's runs, 35 at calibration |
| Static scene reference (what the room should look like from the camera) | Not stored anywhere | n/a | n/a | No. This is the missing piece |
| Runner flags (--auto-level, --target-cm, --launch...) and config used | Command line | Every run | n/a | No. The npz stores the checkpoint name only (real/bounce_run.py:410-411) |
| Run logs | real/logs/ (gitignored, .gitignore:5) | Every run | n/a | The harness (scripts/runner_scenarios.py) writes the same run-*.npz names into real/logs. Recorded runs 20261102-212458, -213605 and -214053 have video but no npz |

### Camera position history (registration on the left wall, rows 40-300, cols 0-200, plus the jaws and blade)

| Period | Shift vs calibration-era view: wall | At the jaws/blade | Evidence |
|---|---|---|---|
| Nov 2 before 20:50 | moved | - | commits 0956b70 ("recalibrated camera after it moved") and 7c3be15 |
| Nov 2 20:50-21:26 (calibration made here) | 0 | 0 | videos 211230, 212201, 212458 |
| Nov 2 21:36-21:51 | (-11.9, -0.7) px | (-7, +1.5) px | videos 213605, 214053. Moved between 21:26:38 and 21:36:05; run 212949's video from that window is unreadable |
| Nov 4 23:10-23:19 | (-10, -18 to -19) px | (-4 to -6, -17 to -21) px | camera-check 231018, videos 231109 and 231521 |
| Nov 4 23:23 (after the hand fix) | (-7.7, -25) px | - | camera-check 232305 |
| Nov 4 23:31 recalibration | - | paddle centre (-7.0, -19.3) px from the old calibration | real/camera_calib.json: camera 1.2 deg rotated and 1.5 cm moved from Nov 2 |

When the camera did not move, registration repeats to within 0.3 px inside a run and about 1 px between runs on the
same night. Every move seen was 11-25 px. The wall and the arm shift by different amounts (about 5 px in x for both
moves), so a wall shift is a reliable move detector but not an exact correction at the paddle. Note that
camera-check-20261104-015513.png and -015707.png were made from Nov 2 recordings (the 21:36-21:40 and 21:20-21:26
views). The 231018 and 232305 checks were taken with the arm folded: real/camera_check.py:58-63 always draws the hold
pose, so those outlines sit over empty table.

## 2. The "slanted" paddle

### Candidate causes, ranked

| Rank | Cause | Evidence | Verdict |
|---|---|---|---|
| 1 | Camera moved | Wall moved (-10, -19) px, blade and jaws (-5, -19) +- (2, 4) px. Independently confirmed by the 23:31 recalibration: (-7.0, -19.3) px, 1.2 deg rotation. The blade appears ~19 px higher in the frame, and the Nov 2 model outline no longer lands on it. On Nov 2 21:22 it did (outline fit IoU 0.93-0.94, see method) | Verified. Changes what the camera sees and biases the ball position; does not tilt the real paddle |
| 2 | Hold wanders after each return | First armed period is at the hold on both nights. Later armed periods: Nov 2 21:23 pitch p50 -2.1 deg (tip up); tonight +0.5 / +0.9 deg, p90 +1.5 (tip down). This comes from deadband and load (elbow +0.8 to +1.8 deg), not from a saved file. The sag trim runs only once, at startup (real/bounce_run.py:193-199) | Verified, but at most ~1.5 deg tonight. Tip-down sends a set-down ball toward the tip |
| 3 | Policy tilting during play | Active pitch p10/p90: tonight -5.0/+2.7 and -5.1/+2.6 deg; Nov 2 21:23 -6.3/+3.9. Roll: tonight -3/+4; Nov 2 -8.4/+3.1 | Visible (+-5 deg) but not new. If anything, smaller tonight |
| 4 | Paddle turned in the jaws | Blade tip vs jaws, sideways: stable to +-0.5 px across Nov 2 and tonight (<= 0.2 deg yaw). Blade vs model at the calibrated camera: within ~1 cm. Pitch/roll in the jaws below ~3-5 deg cannot be resolved from the overhead view (+-4 px pose noise) | No evidence. Not excluded at 1-3 deg; needs the ball-roll test (real/level_ball.py, 0.4 deg) |
| 5 | Stale camera calibration | It was correct for 20:50-21:26 Nov 2 and wrong from 21:36 on. That is cause 1, not a separate one. The "predates the wrist offsets" mismatch (8.8 vs 10.8 deg flex) is the paddle re-level; it moves the calibration's FK points by <= 1 cm and does not make the camera pose wrong | Consequence of 1. Fixed at 23:31 (uncommitted) |
| 6 | Servo calibration | Files identical, unchanged since Oct 8. After removing the camera shift, the jaws and blade show no jump beyond pose noise. EEPROM not read | Unlikely. EEPROM unverified |

What the user saw cannot be pinned down: a ~19 px move of the view (cause 1) and a 1-3 deg change in the held pitch
between sessions (cause 2) are the only verified differences. Neither makes the real paddle visibly slanted by
itself, while the policy's own +-5 deg during play (cause 3) is visible on both nights.

### What the camera shift does to the ball position

Ball at the hold-pose paddle centre (camera depth 0.68 m, radius ~15 px), using real/rig.py:270-274 with the Nov 2
calibration. Values are reported minus true, in cm, in the base frame: +x toward the blade tip, +y left, +z up.

| Shift of the scene at the paddle | x | y | z | |xy| |
|---|---|---|---|---|
| Tonight, image registration (-5, -19) px | +1.8 | +1.0 | +1.4 | 2.1 |
| Tonight, from the 23:31 recalibration (-7.0, -19.3) px | +1.9 | +1.2 | +1.4 | 2.2 |
| Wall shift taken as uniform (-10, -19) px | +1.7 | +1.6 | +1.4 | 2.4 |
| Nov 2 21:36-21:51 (-7, +1.5) px | -0.3 | +0.9 | -0.1 | 0.9 |
| 1 px | 0.13 | 0.13 | 0.1 | - |

At 10 cm above the paddle the error is about 12 % smaller. In simulation, a camera bias of this size barely changed
v23-p32-learn16 survival (14-21 mm: 16 % vs 1-11 mm: 17 %, research/user_asks_coverage.md:281-284), so the bias is a
setup error worth catching but probably not the main cause of tonight's drops (simulation evidence only).

### The forward drift

Forward losses are not new. Of the activations of 1.5 s or longer, the last detection was beyond the blade tip in 4 of
6 on Nov 2 21:23 (calibrated camera), 5 of 9 at 23:13 and 4 of 4 at 23:19 tonight. Tonight's bias reports the ball
~1.9 cm further toward the tip than it is. A policy centring the reported ball would therefore keep the real ball
toward the handle, not the tip (inference, not simulated). So the camera move does not explain the forward drift.
Open candidates: the post-return tip-down hold tonight (cause 2), and the 48 ms latency used by runs 1-2 against
62 ms measured at 23:31.

### Tonight's recalibration (23:31)

One pose (16) is an outlier: radius 11.1 px against 14.4 px at the identical pose 1. It passed the 0.6-1.6 ratio gate
at real/calibrate_camera.py:93 and raised the reported depth rms from 0.71 to 4.62 cm. Refitting without it changes
the camera position by 2 mm and radius_scale from 1.042 to 1.040, so the written calibration is usable. The arm also
returned to hold with a 6.3 deg error at the end of the calibration.

## 3. Checks to add (most important first)

| # | Check | Catches | Where | Threshold / cost |
|---|---|---|---|---|
| 1 | **Camera-moved check.** calibrate_camera.py saves a grey reference frame and a static-region box (left wall, which correlated at 0.93-0.98 when unmoved). The runner phase-correlates the live frame against it right after opening the camera, before opening the bus | Every camera move seen here (11-25 px) | real/calibrate_camera.py:150, real/bounce_run.py:164-172 | Warn at 2 px (~0.3 cm), stop at 5 px. If the correlation response is < 0.3, say "reference region changed, check by eye". About 20 lines; reuses cv2.phaseCorrelate |
| 2 | **Blade-at-hold check** (camera relative to arm). After the sag trim, fit the projected model outline to the dark blade (as in the method below), or compare with a reference silhouette saved at calibration | Camera moves, the arm base moving on the table (check 1 misses this), blade offset or yaw in the jaws | real/bounce_run.py:200-210 | Stop if the best shift is > 6 px (pose noise is +-4 px). About 40 lines |
| 3 | **Setup fingerprint instead of the offset warning.** camera_calib.json records the reference-frame hash, image size, ball colour, paddle id and latency. The run npz stores config.yaml text, calib `created`, the hold pose, the command-line args and the results of checks 1, 2, 4 | Silent setup changes between runs; makes every log traceable | Replace real/bounce_run.py:173-176; extend :410-411 | Warn only when a check fails, so warnings mean something. The current one fired on every real run since Nov 2 21:21 |
| 4 | **Servo EEPROM check** in open_bus: `bus.read_calibration()` against the project copy (refuse on mismatch), Present_Voltage >= 11 V, D_Coefficient = 32 | Recalibration by another tool, wrong PSU, reconfigured gains | real/arm.py:44-62 | Read-only, about 10 lines |
| 5 | **Hold/joint-map consistency.** Turn the tilt print into a stop at > 1 deg; drop the `sysid.py set-hold` pointers | hold_pose.json and joint_map out of step (e.g. after set-hold) | real/bounce_run.py:6, :208-210; real/rig.py:5 | A few lines |
| 6 | **Check-only paddle level.** A level_ball.py mode that reports the ball-roll tilt and writes nothing, run at the start of a session | Paddle pitch/roll in the jaws (the only instrument that sees it, 0.4 deg) | real/level_ball.py:151-162 | Moves the arm: needs the user's OK each time |
| 7 | Calibration gates: reject a pose whose radius is > 10 % off the others at similar height; require depth rms <= 1.5 cm | Outliers like tonight's pose 16 | real/calibrate_camera.py:93, :223 | A few lines |
| 8 | Logs: harness runs under their own name; save the npz even when a run is killed | Real and harness logs being confused; lost logs | scripts/runner_scenarios.py, real/bounce_run.py:392-413 | Small |
| 9 | Optional: re-trim the sag for ~1 s in the armed state after each return | 1-2 deg hold wander between attempts (2 deg halves sim survival, research/user_asks_coverage.md:39) | real/bounce_run.py:356-363 | Small; behaviour change, not a check |

Checks 1 and 3 matter most. The camera mount is the part that keeps moving, and today nothing records what the camera
saw at calibration. Status while this was written: uncommitted edits to real/rig.py and real/calibrate_camera.py add
check 1 (camera_ref.png, the same left-wall box, 4 px limit, frame-size check) and part of check 3 (`calib_inputs`:
joint map, paddle offset, SHA-1 of the servo-calibration JSON). The JSON hash does not cover the servo EEPROM (check 4),
and the wall box does not see the arm base moving (check 2). Check 2 is the stronger version of 1, because it measures at the paddle and also catches arm-base
and blade moves. Check 4 closes the one remaining saved-state gap (EEPROM).

## Method

- Frames: frame 0 (armed) of each recorded run, plus later frames, and the camera-check PNGs. No files were written
  under real/; scratch scripts are in the session scratchpad.
- Camera shift: cv2.phaseCorrelate with a Hanning window on the left-wall region, and template matching
  (TM_CCOEFF_NORMED, sub-pixel) on the jaws bracket and blade, against the Nov 2 21:24 armed frame.
- Blade vs model: the hold-pose blade outline (same construction as real/camera_check.py:58-69) is slid over the dark
  blade mask (grey < 70, top 75 % of the outline) and the best IoU is kept. Results, model -> real, in px: Nov 2 21:22-21:24
  (-1..-3, 0..+6); Nov 2 21:36-21:40 (-10, +11..+12); tonight (-7, -12..-15). IoU 0.93-0.94 throughout.
- Pose effects: FK of the median armed joints per run (real/rig.py joint map, bounce.robot.ArmKinematics), projected
  with the calibration. These amount to <= 6 px between nights at the paddle centre.
- Ball bias: real/rig.py `CameraModel.project` / `ball_position` with the pixel shift applied and the radius unchanged.
- Real runs: those with video (Nov 2 21:06-21:56, Nov 4 23:11-23:19). The Nov 4 01:52-02:56 and Nov 2 23:03-23:14
  npz files have no video and look like harness runs (zero joint variance or a 4.75 s start); they were excluded.

## Unverified

- Servo EEPROM contents (the port was not opened).
- Paddle pitch/roll in the jaws below ~3-5 deg.
- Why the camera moved (bump, or cable pull at the USB replug).
- Whether the user judged "slanted" from the camera feed or by eye.
- That a +1.9 cm x bias makes the policy hold the ball toward the handle.
- Why three recorded Nov 2 runs have no npz.
