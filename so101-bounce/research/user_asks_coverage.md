# What the user asked for, and whether it is trained and tested

Every request for this project (2026-10-30 to 11-01), where it is in training (round 10, variants `v16*`, warm-started
from round 9 `v15*`), and scenario-test survival for the round-9 policy `v15-p32` and the round-10 pick `v16-p32-lo`.
Test setup for every number: servo set `12v_r10` (fitted to the real-ball runs of 2026-11-01: torque-capped lift and
elbow), 18 ms command delay, overhead camera only at the calibrated mount, nominal randomisation (d = 0), 512 episodes
of 8 s, `scripts/scenario_test.py` (`cluster/eval10.sbatch`). "+ launch" rows run the runner's scripted dip-and-strike
first. Visual requests (ball, rubber, wood and arm colours, textures, SimReady assets, lighting looks) change the
ball-detector renders, not the policy, which sees ball positions only.

| Ask | In training (v16) | v15-p32 | v16-p32-lo |
|---|---|---|---|
| Bounce zero-shot on the real arm, low enough to stay in view | target apex 7-10 cm (lo), view_top 10-14 cm | rest + launch 90 % | 99 % |
| Ball starts resting on the paddle, anywhere | rest_start_prob 0.5, rest_spread 0.95 | half-way + launch 85 %, rim 94 % | 98 %, 99 % |
| Start from a dead stop: dip / flick launch | launch replay (strike 3-8 steps), w_first, w_wait | rest centre 94 % | 100 % |
| Hard (dead) paddle | rubber e 0.55-0.75 (30 %) | e 0.65 + launch 76 % | 99 % |
| Drop it on / bounce it in, hand off immediately | flight starts 3-40 cm, bounced in | drop 20 cm 84 %, bounced in 75 % | 99 %, 98 % |
| Drop from high | flight_h to 40 cm | drop 40 cm 69 % | 79 % |
| Held above, then let go (visible / hidden) | held_start_prob 0.15 | 80 % / 97 % | 98 % / 100 % |
| Lands next to the paddle | beside starts 8-16 cm (feasible) | 8 cm from 20 cm 69 % | 79 % |
| Tossed in with spin | toss_p 0.35, curriculum | 0.3 m/s + spin 26 % | 27 % |
| Nudged / tapped by hand | push_rate_hz 1.0 | taps 1 Hz 19 % | 23 % |
| Paddle not quite level | level_error 3 deg | 1 deg + launch 77 % | 88 % |
| Start-up variation (arm unsettled, late switch-on, mid-bounce) | start_pose_p, switch_delay_p, midbounce_p, hist_noise_p | (in all rows) | (in all rows) |
| More reactive: camera timing | latency centred 62 ms (to 120), rolling shutter, timestamps | timestamps 50 ms late 67 % | 84 % |
| Camera noise in a dim room | depth error with ball speed, outliers, 15 fps | depth noise 7 cm 38 %, 15 fps 70 % | 80 %, 92 % |
| Camera anywhere / the real mount | overhead_cam_box + calibrated mount | latency 120 ms 94 % | 100 % |
| Grip length, paddle shape, mass | paddle_reach, blade shape, mass to 0.25 kg | (in all rows) | (in all rows) |
| Joint calibration error | joint_offset 0.05 rad | 2 deg 43 % | 47 % |
| Everything slightly off at once | - | "bad day" 2 % | 3 % |
| Robustness at difficulty 0.5 | ADR | rest + launch 43 %, drop 54 % | 46 %, 60 % |

## Not physically possible on this arm (and what is)

- Beside the paddle: from 20 cm up and 8 cm out, 79 %; from 10 cm up, 42 % (0.14 s of fall is shorter than the
  torque-capped arm needs to move the paddle 5 cm); 12 cm out or more, 0 % in every round.
- Tosses: 0.3 m/s with 15 rad/s spin, 25-29 %; 0.5 m/s with 30 rad/s, 0-2 % in every round and variant.
- Hand taps every second: about 20 % in every round. Occasional taps are fine.
- Paddle 2 deg off level: 40-47 %. Level it with `real/level_ball.py` (it reaches under 0.5 deg).
- Joint calibration off by 2 deg: about 45 %.

## Bounce height

Survival from rest with the launch, by target height (`--target-cm`), 7 / 10 / 13 / 15 cm: v15-p32 99 / 98 / 88 / 75;
v16-p32-lo 100 / 100 / 97 / 94. Higher bounces do not help on this arm; 7-10 cm is best, and the camera loses the ball
above about 12 cm.

## Round 11 (same test, every row at the deploy target 9 cm)

Four runs from v16-p32-lo's / v16-p32-wide's finals, 70 min each: lo continued, wide continued, `v17_p32_cal` (joint
calibration and level error up to 3 deg, taps up to 1 Hz, drawn full-range in 50-60 % of episodes: they had scaled with
a task difficulty level stuck near 0.1), `v17_p32_reach` (more beside starts, tosses 0.3 m/s / 15 rad/s full-range).

| Scenario (9 cm) | v16-p32-lo | lo cont. | wide cont. | cal | reach |
|---|---|---|---|---|---|
| rest + launch / near rim + launch | 100 / 100 | 99 / 99 | 100 / 98 | 99 / 98 | 100 / 100 |
| drop 20 cm 6 cm off / drop 40 cm | 100 / 87 | 100 / 83 | 99 / 94 | 100 / 90 | 100 / 86 |
| beside 8 cm from 20 cm / from 10 cm | 83 / 43 | 81 / 41 | 83 / 39 | 80 / 42 | 86 / 41 |
| tossed 0.3 m/s + spin 15 | 28 | 30 | 25 | 29 | 29 |
| paddle 2 deg off level (no launch / launch) | 56 / 51 | 55 / 53 | 61 / 54 | 61 / 54 | 54 / 47 |
| joint calibration 2 deg | 52 | 50 | 51 | 64 | 44 |
| hand taps 1 Hz | 29 | 28 | 22 | 27 | 26 |
| depth noise 7 cm / 15 fps dim | 88 / 95 | 85 / 95 | 82 / 61 | 82 / 94 | 90 / 95 |
| d = 0.5: rest + launch / drop 10 cm | 55 / 60 | 53 / 61 | 52 / 61 | 42 / 55 | 51 / 63 |

Round 11 moved little: cal bought joint-calibration robustness (52 -> 64 %) with general robustness (d = 0.5 rest +
launch 55 -> 42 %); reach a few points on beside starts. Taps at 1 Hz, 2 deg tilt and spinning tosses stay where they
were in every variant, so treat them as limits of this arm and camera, and level the paddle with `real/level_ball.py`.

## Round 12: a higher, steered launch and a high-then-low target

Ask: "get it bouncing from any angle; a much higher first toss and harder flicks (the paddle is not bouncy); start high,
then low". Built: launches with commands up to 1.6-1.9x the policy's range and strike steering from the ball's offset
and roll (`bounce/launch.py` NAMED `r12`, `r12_deep`: paddle ~1.0 m/s vs 0.67, first bounce 12.3 / 14.7 cm, but a
throw: the same height at every restitution, drifting 3.3 / 5.2 cm even from the centre); a target schedule (EnvCfg
`target_sched_p`, `target_start`, `target_high_s`, `target_ramp_s`). Four P32 runs from v17-lo-cont: schedule + r12,
schedule + r12 with strikes 3-8, steady 12 cm + r12, schedule + r12_deep.

Scenario test (12v_r10, steady 9 cm), survival %:

| Scenario | v16-p32-lo + r10 launch | v18 sched + r12 | v18 t12 + r12 | v18 deep + r12_deep | v18, no scripted launch |
|---|---|---|---|---|---|
| rest centre, launch | 100 | 47 | 65 | 0 | 99-100 |
| rest near rim, launch | 100 | 49 | 51 | 24 | 98-100 |
| rest rolling 0.2 m/s, launch | 78 | 47 | 52 | 12 | 98-99 |
| dead paddle e 0.55, launch | 100 (steady bounce 0.28 s) | 44 | 64 | 0 | 96-100 (0.84-0.92 s) |
| paddle 2 deg off level | 51 | 59 | 62 | 58 | 59-62 |
| joint calibration 2 deg | 50 | 56 | 52 | 49 | 51-56 |

With the schedule (16 cm for 1.5 s, 1 s ramp to 9 cm) survival is the same or a few points lower, and the first
steady bounce comes at ~2.65 s by design. On the torque-capped arm model the higher scripted launch makes the catch
harder (the ball drifts off centre while the paddle is still 8 cm up), and 70 min of training did not learn it
(29 % survival after the replayed launch in the training environment itself, 73 % when the policy launches). What
gets a dead paddle going is the round-10 launch (100 %, 0.28 s to the first 9 cm bounce at e 0.55), or the v18
policies' own launch (0.84-0.92 s, vs 1.40 s for v16-p32-lo without its launch).

## Which policy to run

`policies/v16-p32-lo.pt` with `--launch --target-cm 9` (unchanged after round 11). Rehearsal on the simulated
`12v_r10` arm at 9 cm with the launch: 7/8 runs the full 8 s (31-35 bounces); dropped 15 cm and dead paddle (e 0.65)
full 8 s; beside 10 cm dropped after 2.6 s. Alternatives: `v17-p32-reach.pt` (equal on the core starts, a little
better beside the paddle, worse with joint calibration error; rehearsal 4/4), `v16-p32-wide.pt` /
`v17-wide-cont.pt` (widest servo range, in case the real arm is faster than `12v_r10`), `v17-p32-cal.pt` (if the joint
calibration is known to be off by ~2 deg).

## Round 13 (2026-11-02): auto-level, feasibility, and a harder flick

**Auto-level** (the runner's `--auto-level`, ported exactly into the simulator as `EnvCfg.auto_level`): an average of
where on the blade the ball sits slowly trims the hold command's wrist against it. 12v_r10, 9 cm, 1024 episodes per
row, 95 % intervals about +-3 points. Survival %:

| Scenario | v16-p32-lo | + auto-level | v18-p32-t12 | + auto-level |
|---|---|---|---|---|
| paddle 0 / 1 deg off level | 100 / 95 | 100 / 96 | 100 / 96 | 100 / 96 |
| paddle 2 deg (no launch / launch) | 52 / 51 | 68 / 65 | 65 | 75 |
| paddle 3 deg (no launch / launch) | 21 / 23 | 33 / 30 | 22 | 37 |
| joint calibration 2 deg | 47 | 52 | 54 | 56 |
| rolling 0.2 m/s, dead paddle, drops, rest | unchanged | unchanged | unchanged | unchanged |

Gain (face-normal change per second per metre of drift): 0.25 and 0.5 are equal, 1.0 is a few points better at 2-3 deg
and a few worse on level and rolling starts, 2.0 is harmful (rest centre 58 %). Keep 0.5 (the runner's value).

Fine-tunes from v16-p32-lo (70 min, invisible tilt up to 3 deg and joint calibration error in half the episodes):
`v19-p32-tilt` (auto-level off in training) with auto-level at run time matches v16-p32-lo + auto-level on tilt (2 deg
69 / 65, 3 deg 34 / 25, joint calibration 55) and is better from 40 cm (99 vs 85 %); `v19-p32-al` (auto-level on in
training) is worse (2 deg 58 / 59, joint calibration 36). Training does not add much over the run-time trim.

**What is physically possible** (an oracle: cross-entropy search over open-loop arm commands from the exact start, with
the simulator's exact physics and no camera; 4096 x 20; success = alive 1.5 s, 3+ bounces, back within 4 cm):
- single hand taps (up to 0.3 m/s sideways or 0.9 m/s up or down, at the apex): recoverable; the policies already
  survive them (98-100 % over 1.5 s). Taps every second fail by accumulation (about 15 % per tap), not physics.
- tosses landing away from the centre at 0.3-0.5 m/s (from 20 cm, 3 cm off): never recoverable, at any spin.
- tosses up to 0.2-0.3 m/s toward the centre or sideways, and spin up to 30 rad/s on a centred drop: recoverable
  (policy 2-100 %: a learnable gap). 0.5 m/s toward the centre or sideways with spin is recoverable only with the
  paddle allowed 10 cm up (v18's box), not within v16's 6 cm. Training tosses are aimed at the paddle (within 4 cm of
  the centre), so they stay inside this envelope; the scenario test's random-direction toss rows include
  impossible cases.

**A harder flick** (`r13` in `bounce/launch.py`: r12 / r12_deep retuned with a return stroke that brings the paddle
back toward home before the hand-off): first bounce 11-13 cm, the ball reaches it 97-100 % of the time, but it drifts
4-8 cm, and the paddle tops out near 0.9-1 m/s on this torque-capped arm. `v20-p32-flick` trained on the catch after it
(70 % of launches, 60-105 % strength): with r13 it survives 0-35 % (dropped); with the round-10 launch 100 %, same as
v16-p32-lo (0.28-0.68 s to the first steady bounce). A harder flick does not beat the round-10 launch here.

**Deploy:** `policies/v16-p32-lo.pt --launch --target-cm 9 --auto-level` (gain 0.5). `policies/v19-p32-tilt.pt` with
the same flags is an alternative (equal on tilt with auto-level, better on 40 cm drops; not yet run through the full
suite).

The runner's `--auto-level` applied no trim in camera mode until 2026-11-02: frames arrive about 62 ms after capture
and the update required a detection under 50 ms old. It now updates once per new detection, as the simulator does
(fake-bus check: resting 3 cm off centre, trim +1.65 deg flex, 21 bounces).

## Round 14 (2026-11-02): starts the oracle can recover but the policy misses ("bouncing from any angle")

Recoverable (oracle) starts where v16-p32-lo + auto-level survives under 90 % (12v_r10, 9 cm, 1024 episodes, 8 s,
difficulty 0; toss offset 3 cm, direction relative to the offset):

| Start | Oracle | v16-p32-lo + AL |
|---|---|---|
| toss 0.3 m/s toward centre, from 20 cm / 30 cm | yes / yes | 61 / 52 % |
| toss 0.2 / 0.3 m/s sideways | yes / yes | 89 / 45 % |
| toss 0.2 m/s away; 0.1 m/s away + topspin 15 rad/s | yes / yes | 75 / 73 % |
| centred drop from 20 cm, topspin 30 rad/s | yes | 49 % |
| toss 0.5 m/s toward centre or sideways + spin 30 | only with a 10 cm box | 0 % |
| hand taps every second | each tap yes | 24 % |

(Toward centre 0.2 m/s, 94 %; from 10 cm, 91 %; sidespin centred drop, 100 %.)

Fine-tunes from v16-p32-lo (70 min) with those starts in training: aimed tosses up to 0.3 m/s (any6, 6 cm box) or
0.5 m/s (any10, 10 cm box), spin up to 30 rad/s, full range in 80 % of tosses (toss_full_p; in v16 the toss speed and
spin scaled with a task level stuck near 0.1), and hand taps 0-1 Hz in half the episodes. Change in survival vs
v16-p32-lo + auto-level (points, 95 % interval of the difference; * = beyond it):

| Row | d | any6 | any10 | v19-p32-tilt |
|---|---|---|---|---|
| toss 0.3 toward centre | 0 | +11 +-4 * | -1 | -16 * |
| toss 0.3 toward centre, 30 cm | 0 | -1 | +10 +-4 * | +8 * |
| toss 0.2 / 0.3 sideways | 0 | 0 / +5 * | +5 * / +2 | +7 * / -4 |
| topspin centred drop; 0.5 m/s spinning tosses; taps 1 Hz | 0 | +1; 0; +1 | +2; 0; +1 | -3; 0; -2 |
| drop 40 cm | 0 / 0.5 | -6 * / -1 | +9 * / -5 * | +16 * / +6 * |
| paddle 2 deg (no launch) / 3 deg | 0.5 | 0 / -3 * | -1 / -3 * | -3 / -2 |
| paddle 2 deg (no launch) | 0 | -5 * | 0 | +1 |
| 15 fps dim | 0 / 0.5 | -3 * / -5 * | -2 / -3 | -1 / -6 * |
| rest + launch, rolling + launch, dead paddle + launch | 0.5 | -1 to -4 | +6 to +8 * | -3 to +1 |
| camera latency 150 ms; depth scale +10 % | 0 | 0; -2 | +7 *; -5 * | -15 *; -5 * |

The gap to the oracle mostly stays: the largest gain on a recoverable start is +11 points, spinning 0.5 m/s tosses stay
at 0 even with the 10 cm box, topspin drops at ~50 % and taps at ~25 %. Every fine-tune also regresses some round-10
rows beyond the interval (any6: 40 cm drops, 2 deg tilt, 15 fps; any10: 40 cm drops and 3 deg tilt at difficulty 0.5;
v19-p32-tilt: camera latency, 15 fps).

**Deploy stays** `policies/v16-p32-lo.pt --launch --target-cm 9 --auto-level`. Runner-up: `policies/v21-p32-any10.pt`
(same flags): no regression on the round-10 rows at difficulty 0, +6 to +8 on launched / dead-paddle starts at 0.5, but
-5 on 40 cm drops and -3 at 3 deg tilt at 0.5.
