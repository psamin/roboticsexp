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
