# Scenario test results (2026-10-31)

The cases come from [scenarios_and_failures.md](scenarios_and_failures.md).

## How the tests were run

**Simulator: `scripts/scenario_test.py`** (cluster job 13198, H200)
- All policies were tested on the model of **this arm**: 12 V servos, the fitted 30 ms command delay, overhead camera only.
- Each row is 512 episodes of 8 s.
- Two difficulty levels: d = 0 is nominal (the simulator matches the hardware); d = 0.5 has every randomised parameter at half range.
- A ball dropped from above the 30 cm out-of-view ceiling is only failed after it has touched the paddle.
- Raw output: `outputs/scenario_tests/*.txt`.

**Runner: `scripts/runner_scenarios.py`** (local, about 10 s per case)
- This runs the real `real/bounce_run.py` in camera mode against fakes:
  - a fake arm bus with the 12 V servo model and the 30 ms command delay;
  - a fake camera watching a simulated ball;
  - a scripted hand that holds, drops, removes or gaps the ball.
- Runs last 6-8 s.
- Raw output: `outputs/runner_scenarios-*.json`.

## 1. Simulator: survival on the 12 V arm model

| Scenario | v5-base | v5-p32 | v12-base | v12-p32 | v5-base (d 0.5) | v5-p32 (d 0.5) | v12-base (d 0.5) | v12-p32 (d 0.5) |
|---|---|---|---|---|---|---|---|---|
| rest centre | 14% | 63% | 100% | 100% | 11% | 26% | 59% | 69% |
| rest half-way | 22% | 56% | 99% | 100% | 12% | 24% | 61% | 67% |
| rest near rim | 29% | 63% | 100% | 100% | 14% | 25% | 57% | 64% |
| drop 10 cm | 33% | 70% | 100% | 100% | 11% | 29% | 58% | 63% |
| drop 20 cm, 6 cm off | 38% | 61% | 99% | 100% | 14% | 25% | 50% | 56% |
| drop 40 cm | 12% | 16% | **28%** | **64%** | 12% | 14% | 24% | 36% |
| beside 8 cm, 10 cm up | 11% | 17% | **42%** | **42%** | 4% | 12% | 23% | 26% |
| beside 12 cm, 20 cm up | 3% | 5% | **10%** | **11%** | 1% | 2% | 4% | 6% |
| beside 15 cm, 30 cm up | 0% | 0% | **0%** | **1%** | 0% | 1% | 0% | 1% |
| tossed 0.5 m/s + spin 30 rad/s | 0% | 0% | **1%** | **1%** | 0% | 0% | 0% | 1% |
| bounced in (up 1.5 m/s) | 27% | 70% | 100% | 100% | 15% | 26% | 58% | 59% |
| held 0.5 s visible, then let go | 31% | 76% | 100% | 100% | 14% | 34% | 55% | 57% |
| held 1 s hidden, then let go | 40% | 67% | 100% | 100% | 16% | 30% | 60% | 68% |
| paddle 1° off level | 10% | 53% | 96% | 98% | 7% | 20% | 46% | 51% |
| paddle 2° off level | 3% | 39% | **60%** | **75%** | 1% | 13% | 24% | 26% |
| paddle 4° off level | 0% | 0% | **0%** | **4%** | 0% | 0% | 1% | 0% |
| camera latency 120 ms | 9% | 5% | 100% | 100% | 5% | 11% | 47% | 47% |
| camera latency 150 ms | 0% | 1% | 98% | 93% | 2% | 3% | 34% | 40% |
| timestamps 50 ms late | 10% | 22% | 91% | **60%** | 3% | 14% | 28% | 31% |
| timestamps 100 ms late | 0% | 1% | **11%** | **40%** | 0% | 1% | 8% | 17% |
| depth noise 7 cm | 27% | 44% | 100% | 98% | 13% | 24% | 54% | 57% |
| depth scale +10 % | 21% | 33% | 100% | 100% | 9% | 20% | 52% | 65% |
| camera above the paddle (0.35, 0, 1.1) | 3% | 11% | **72%** | **76%** | 1% | 6% | 20% | 29% |
| camera in front (0.6, 0, 0.9) | 0% | 4% | **25%** | **22%** | 0% | 1% | 8% | 12% |
| camera low behind (−0.25, 0, 0.7) | 5% | 16% | 95% | 92% | 7% | 13% | 42% | 48% |
| 15 fps, dim | 3% | 4% | **75%** | 97% | 0% | 5% | 24% | 43% |
| detection dropout bursts | 11% | 30% | 94% | 96% | 6% | 17% | 49% | 54% |
| false detections 2 % | 18% | 34% | 96% | 95% | 9% | 20% | 48% | 53% |
| camera calibration off by 3°, 2 cm | 35% | 52% | 100% | 100% | 14% | 24% | 53% | 55% |
| hand taps at 1 Hz | 2% | 8% | **19%** | **18%** | 1% | 4% | 12% | 11% |
| wind 0.5 m/s | 37% | 55% | 100% | 100% | 15% | 32% | 62% | 63% |
| command delay 40 ms | 24% | 41% | 100% | 100% | 10% | 21% | 55% | 58% |
| USB stalls 3 % | 11% | 25% | 85% | 99% | 10% | 20% | 40% | 57% |
| servos 15 % weaker (hot) | 93% | 97% | 100% | 100% | 38% | 48% | 44% | 53% |
| joint calibration 2° | 10% | 27% | 86% | 79% | 5% | 11% | 35% | 34% |
| ball restitution 0.75 | 74% | 91% | 100% | 100% | 32% | 49% | 54% | 66% |
| paddle 220 g | 85% | 96% | 99% | 100% | 24% | 40% | 25% | 51% |
| "bad day": 1.5° tilt + 120 ms latency + 50 ms late timestamps + camera above | 0% | 0% | **7%** | **12%** | 0% | 0% | 3% | 6% |

The first four policy columns are nominal (d = 0); the last four are the same policies at d = 0.5. Bold marks a 12 V-policy result below 80 % at nominal. Where episodes fail, the failure is almost always "dropped".

Timing and accuracy for v12-base at nominal:
- Apex error is 0.7-1.3 cm on the starts it catches.
- The first bounce comes 0.15 s after a 1.5 m/s bounce-in, 0.27 s after a 10 cm drop, 0.43 s from a resting start, and 0.81 s near the rim.

## 2. Runner: `bounce_run.py` closed loop

Each row ran for 6-8 s. "Free" is when the hand lets go of the ball.

| Case | v12-base | v5-base |
|---|---|---|
| drop 15 cm, the hand hides the ball | on +0.08 s, 20 bounces, never fell | on +0.08 s, 21 bounces |
| held 1 s in view, then let go | **on 0.92 s before release** (paddle drifted 2.8 cm under the held ball); 20 bounces | same early start; fell after 1.6 s |
| set down resting, 3 cm off-centre | on +0.08 s, 21 bounces | fell after 1.7 s |
| **dropped 12 cm beside the paddle** | **never starts** (start zone is 8 cm); ball falls | same |
| depth noise 4 cm | on +0.07 s, 20 bounces, no false "lost" | fell after 1.3 s |
| **0.7 s detection gap in play** | **"lost" called while the ball was in play**: 0.4 s ramp back, re-armed when detections returned (+3.0 s), 21 bounces in total | fell before the gap |
| 15 fps camera | on +0.10 s, 20 bounces | fell after 2.3 s |
| true latency 100 ms (runner assumes 35 ms) | on +0.14 s, 17 bounces | fell after 3.2 s |
| hand false detections 10 % | 21 bounces | fell after 1.9 s |
| ball taken away, dropped again 0.3 s later | re-armed and started again at +0.08 s, 20 bounces | started again, fell |
| **static orange blob on the paddle, no ball** | **starts 80 ms after it appears and never stops** (the blob always counts as a ball on the paddle) | same |

## 3. What fails, ranked

Numbers are v12-base / v12-p32 at nominal unless stated.

1. **Running a 5 V policy on this 12 V arm.** v5-base survives 14-38 %, v5-p32 56-70 %, against 99-100 % for v12. This explains the real dry run, where v5-base dropped the simulated ball at 2 s. **Fix:** switch to a v12 (or v13) policy now.
2. **Starts beside the paddle, and tosses.** 42 % at 8 cm off-centre, 10 % at 12 cm, 0 % at 15 cm; 1 % with 0.5 m/s plus spin. The runner also never starts for a ball outside 8 cm.
   - Training fix: round 7 adds beside starts (15 %, 8-15 cm, up to 0.3 m/s).
   - Next round: tosses up to 0.5 m/s and spin up to 30 rad/s (needs a `flight_spin` option; the 20 rad/s is hard-coded today).
   - Runner fix: start zone 8 → 16 cm.
3. **Paddle off level.** 1°: 96/98 %. 2°: 60/75 %. 4°: 0/4 %.
   - Hand levelling has to be better than about 1.5°. Use a phone level app on the blade in `real/level.py`.
   - Round 7 trains up to 2°; the next round could go to 3° (`level_error` 0.05).
4. **Timestamp error (camera latency assumed wrong).** 50 ms late: 91/60 %. 100 ms late: 11/40 %.
   - The runner assumes 35 ms. Forum reports put the C920 at 150-180 ms in some pipelines.
   - Round 7 trains ±60 ms (`age_bias` 0.06).
   - **Measure the real latency** (runner fix 4).
5. **Camera position.** In front of the paddle: 25/22 %. Directly above it at 1.1 m: 72/76 %. Low behind the base: 95/92 %.
   - The current calibration says (0.17, 0.00, 0.82) m, but that fit is weakly determined (runner fix 5).
   - Round 7 trains the whole box from x −0.25 to 0.45 m and z 0.7 to 1.3 m.
6. **Hand taps at 1 Hz.** 19/18 %. Taps that often are not survivable today; training uses 0.4 Hz scaled by the task level. Next round: `push_rate_hz` 0.8.
7. **Drops from 40 cm.** 28/64 %. Round 7 trains drops up to 40 cm.
8. **15 fps in dim light.** 75/97 %. **USB stalls 3 %.** 85/99 %. **Joint calibration 2°.** 86/79 %. All are under 100 % but tolerable. Use good lighting.
9. **Compounding.** One realistic "bad day" (1.5° tilt, 120 ms latency, timestamps 50 ms late, camera above) gives 7/12 %. Each fault alone costs little; together they compound.
10. **Fine at nominal (≥ 95 %):** resting anywhere including the rim, drops of 10-20 cm, bounced in, held 0.5-1 s, latency up to 150 ms, depth noise 7 cm, depth scale 10 %, calibration error 3° / 2 cm, wind, 40 ms command delay, weaker servos, ball restitution 0.75, a 220 g paddle, dropout bursts, 2 % false detections.

**At d = 0.5, every policy drops to 50-70 % even on easy starts.** This is the general robustness ceiling, not one scenario.

## 4. Runner fixes (`real/bounce_run.py`, `real/rig.py`, `real/calibrate_camera.py`)

1. **Start zone.** In the armed check, change `np.hypot(r[0], r[1]) < 0.08` to `< 0.16`. This matches beside starts; the policy decides whether the ball is reachable. Keep `-0.05 < r[2] < 0.5` and the two-detection rule. Recommended zone: within 16 cm laterally, from 5 cm below to 50 cm above the paddle.
2. **Phantom ball (a static orange blob, red rubber or tape).**
   - Gate the finder by the calibrated ball size: `BallFinder(colour, min_r=0.5 * r0, max_r=2.5 * r0)`, where `r0 = median(camera_calib["rad"])` (7.5 px now). The blade is 30-45 px, so it gets rejected.
   - In the active state, call the ball lost when the detections have moved less than 1 cm over the last 1.5 s.
3. **Lost rule.**
   - `LOST_S` 0.6 → 1.0 s. A 0.7 s gap in play triggered the ramp back.
   - Require 2 consecutive detections outside the lateral or height limits before calling the ball lost; one noisy depth sample can trip the z test.
4. **Measure the camera latency.** It is the largest unmeasured error (item 4 above).
   - With the ball taped on, glide the paddle through a ±3 cm, 1 Hz sine for 5 s in `calibrate_camera.py`.
   - Cross-correlate the forward-kinematics ball position with the detections.
   - The lag plus 35 ms is the true capture-to-read latency. Write it to `real/config.yaml` as `camera_latency_s`; `bounce_run.py` already reads that key.
5. **Calibration fit.**
   - The current fit pinned the focal length near the bottom of its search range, with `radius_scale` 1.54 (about 1.0 expected).
   - The 12 calibration points span only about 4 cm in height, so focal length and camera distance trade off.
   - Either fix the focal length at the C920 nominal (a checkerboard is better), or add two poses with the paddle about 10 cm higher and lower.
6. **Held ball.** No change needed. The early start on a ball held in view costs nothing in either the simulator (100 %) or the closed-loop runner (20 bounces), and round 7 trains it.

## 5. Which policy to run

- **Now: `policies/v12-base.pt`.** It is at P16, the gain the 12 V servo model was fitted at.
- **v12-p32** is the more robust one in simulation: USB stalls 99 vs 85 %, 15 fps 97 vs 75 %, 2° tilt 75 vs 60 %, timestamps 100 ms late 40 vs 11 %, 40 cm drops 64 vs 28 %. Its P32 servo behaviour, though, is extrapolated from P16 data. Run a virtual-ball dry run first and look for oscillation.
- **Round-7 policies** (v13-base 13193, v13-p32 13194, warm-started from the v12 runs, due by 19:42) train every case above. Their scenario tests run automatically afterwards (13199/13200, results in `~/so101-bounce-results/scenario_tests/v13-*.txt`). If they hold the nominal rows and improve the bold ones, they replace v12.

## 6. Training changes

**Round 7, now running (variants `v13` / `v13_p32`):**
- `held_start_prob` 0.15
- `beside_start_prob` 0.15 (8-15 cm, up to 0.3 m/s)
- `flight_h` up to 0.40 m, with `ceiling_after_land`
- `overhead_cam_box` x −0.25…0.45, y ±0.2, z 0.7…1.3 m
- `level_error` 0.035
- `age_bias` 0.06
- `depth_scale` 0.05
- `cam_base_ms` up to 80, `cam_lat_max` 0.15
- `rest_spread` 0.9

**Proposed for round 8:**
- `beside_vlat` 0.5, plus a new `flight_spin` option up to 30 rad/s (tosses)
- `push_rate_hz` 0.8 (taps)
- `level_error` 0.05 (3°)
- `joint_offset` 0.05
- some 15 fps episodes at nominal light (dim rooms)
- set `overhead_cam` to the measured camera position once the calibration is trustworthy
