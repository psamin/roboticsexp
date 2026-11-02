# Launching a resting ball (SO-101, table-tennis paddle)

2026-10-31. Rig: 12 V STS3215s, 50 Hz goals over USB, ~30 ms command delay, paddle level at ~28 cm, dead rubber
(e 0.6-0.8). Goal: a first bounce of 7-15 cm, then the policy takes over.

## Answer

- **Use a scripted "dip and flick", open loop, then hand over to the policy.** Swing shoulder lift, elbow flex and
  wrist flex together, first down and then up. The upswing passes level near the hold height and carries the ball,
  then lets it go.
  - [bounce/launch.py](../bounce/launch.py) (`DEFAULTS[16]`) already does this. My separate sim runs agree with it
    (numbers below).
  - In sim on the refit servo set it gives a 9.1-10.2 cm first bounce, with under 1 cm drift, at e 0.6-0.85.
- **What does not work on this arm:**
  - A level push. Keeping the face level caps the blade at 0.23-0.47 m/s at this pose.
  - A wrist-only flick, which reaches 0.5-0.7 m/s in sim.
  - The current policy's small-stroke pump-up, which stalls at 2-5 cm on a dead paddle.
  - A 10 cm bounce needs the ball to leave at 1.40 m/s.
- **The servo speed limit sets the height, not the stroke size.** The refit puts the firmware limit at ~2.3 rad/s,
  against 4.7 rad/s no-load.
  - With the speed limit at 1.8, 2.3 or 3.0 rad/s, the same command gives 5.7, 9.1 or 15.4 cm.
  - Scaling the joint amplitudes by 0.9-1.2x changes nothing.
  - Tune by stroke duration instead (see Tuning).
- **No source describes a "rolling trick" for starting from a resting ball.** Drills and coaching pages just say to
  pop the ball straight up. Rolling only adds sideways speed. The one useful related effect is that a ball farther
  out on the blade gets more speed from a flick (+1.7 cm of apex for 3 cm farther out, in sim).

## 1. Physics

**Separation.** The contact force is m(g + a_p), where a_p is the paddle's upward acceleration. The ball leaves the
face when a_p < -g, that is, when the paddle accelerates downward faster than gravity.
- **Push from rest.** The ball leaves at the paddle's speed at the moment its braking first exceeds g. The apex is
  then v²/2g above that point. A push must therefore reach 1.40 m/s and then brake at more than g. A slowly braking
  servo releases the ball after its peak speed.
- **Hit.** The paddle is heavy compared with the 2.7 g ball, so a hit gives v_out = (1 + e) v_p + e v_in. Here v_p is
  the paddle's upward speed and v_in the ball's downward speed at contact.

Apex 7 / 10 / 15 cm needs v_out 1.17 / 1.40 / 1.72 m/s. The paddle speed this takes, with f the free fall of the ball
before contact:

| apex | push (any e) | hit, e 0.6, f 0 / 1 / 2 cm | e 0.7 | e 0.8 | e 0.85 |
|---|---|---|---|---|---|
| 7 cm | 1.17 | 0.73 / 0.57 / 0.50 | 0.69 / 0.51 / 0.43 | 0.65 / 0.45 / 0.37 | 0.63 / 0.43 / 0.35 |
| 10 cm | 1.40 | 0.88 / 0.71 / 0.64 | 0.82 / 0.64 / 0.57 | 0.78 / 0.58 / 0.50 | 0.76 / 0.55 / 0.47 |
| 15 cm | 1.72 | 1.07 / 0.91 / 0.84 | 1.01 / 0.83 / 0.75 | 0.95 / 0.76 / 0.67 | 0.93 / 0.72 / 0.64 |

- **Why a hit beats a push.** A hit needs 40-60 % of the paddle speed a push needs. Restitution multiplies the
  paddle speed by (1 + e) and returns e of the ball's own fall speed. A push can never send the ball faster than the
  paddle.
- **Float time.** A ball dropping freely by f takes √(2f/g): 32 ms for 0.5 cm, 45 ms for 1 cm, 64 ms for 2 cm and
  78 ms for 3 cm. It arrives at 0.31, 0.44, 0.63 and 0.77 m/s.
  - That is 2-4 control steps, shorter than the camera latency (~60 ms) plus one frame at 30 fps.
  - So the dip and the hit have to be planned ahead, open loop. That works because a resting ball's state is known.
- **Stroke length.** Reaching v at a steady acceleration a needs s = v²/2a.
  - At 2g: 0.9 cm for 0.6 m/s, 2.5 cm for 1.0 m/s, 5.0 cm for 1.4 m/s.
  - A push to 1.4 m/s therefore needs about 5 cm of runway plus room to brake.
- **Straightness.** If the face is tilted θ at release, the ball lands 2h·sin 2θ away, where h is the apex.

  | apex | 1° | 2° | 3° | 5° | 10° |
  |---|---|---|---|---|---|
  | 10 cm | 0.7 cm | 1.4 cm | 2.1 cm | 3.5 cm | 6.8 cm |
- **Pumping from rest (what the policy does).** Hitting at the same v_p every time converges to
  v = (1 + e) v_p / (1 − e). Each bounce closes the gap by a factor e, so getting to 90 % of that speed takes
  ln 0.1 / ln e bounces: 4.5 at e 0.6, 6.5 at e 0.7, 14 at e 0.85.
- **Steady bouncing, after the launch.** People and stable open-loop jugglers hit the ball while the racket is still
  rising but slowing down (negative acceleration at impact). Sources: [Schaal, Sternad & Atkeson, J. Motor Behavior
  1996](https://is.mpg.de/publications/schaal_jmb_1996) and [Sternad et al., JEP:HPP 27(5) 2001](https://is.mpg.de/publications/sternad_jephpp_2001).
  This is about keeping a bounce going, not about starting one.

**This arm's geometry.** From the repo's MuJoCo model at the hold pose (lift -1.35 rad, elbow 0.60 rad, wrist
levelled; scratch script):

| joint | blade-centre rise per rad | horizontal distance, axis to blade centre | face pitch per rad |
|---|---|---|---|
| shoulder lift | 0.281 m | 0.281 m (axis 16 cm below the face) | 1 rad |
| elbow flex | 0.384 m | 0.384 m (axis 11 cm below) | 1 rad |
| wrist flex | 0.289 m | 0.289 m (axis 1.2 cm below) | 1 rad |

- **Wrist distance.** The blade centre is ~29 cm from the wrist-flex axis, not 13 cm. The 13 cm is the paddle's
  reach beyond the fingertips; the fingertips are another ~16 cm from the axis. Over the env's reach range
  (6-16 cm) this distance is 22-32 cm. Measure it on the real arm.
- **Level push.** All three joints pitch the face the same way, so keeping the face level means the joints must
  partly cancel. Only the differences between their lever arms then move the blade (elbow against wrist: 9.5 cm per
  rad).
  - Solving for a level vertical stroke with no forward drift gives joint rates of (6.4, -9.9, 3.5) rad/s per m/s.
  - The elbow limits it: 0.47 m/s at 4.7 rad/s, and 0.23 m/s at the refit 2.3 rad/s.
  - This matches the 2-4 cm the real arm reaches with a push.
- **Wrist flick.** At 29 cm, 4.7 rad/s would give 1.36 m/s, and 10 cm needs 4.8 rad/s from the wrist alone. Neither
  servo model gets close within the ±0.35 rad action range:
  - wrist-only pushes peak at ~1.9 rad/s on 12v_r8 (0.53 m/s at the blade) and ~2.4 rad/s on 12v (0.7 m/s);
  - the limits are the firmware speed cap and ~50-70 ms of torque build-up after each new goal.
  If the blade really were 13 cm out, 10 cm would need 10.8 rad/s.
- **Whole-arm flick.** Swinging all three joints the same way adds their speeds: 0.95 m per rad in total. 1.4 m/s
  then needs only ~1.5 rad/s per joint.
  - The face then turns at (blade speed) / ~0.33 m, about 3-4 rad/s.
  - The flick has to put the face level while the ball is on it.

## 2. How people start (sources)

| technique | what the source says | source |
|---|---|---|
| Balance, then pop it straight up | Balance the ball on the face, then bounce it; no separate start step | [Ping Pong For Good drills](https://www.pingpongforgood.org/blog/drills) |
| Pat it up a few inches | A footwork drill: pat the ball up after a table bounce, "Straight Up. Just a few inches, to start with." | [Veatch, Bounce with the ball](https://www.tomveatch.com/tt/bouncewiththeball.php) |
| Arm, not wrist | Keep the wrist steady; body and upper-arm swing control the contact | [Try Tennis: bouncing the ball](https://www.try-tennis.com/en/quick-start/pick-ball/) |
| Toss rather than flip | For the paddle-flip trick, "tossing it up rather than trying to flip the paddle" was easier | [PingSkills forum](https://www.pingskills.com/table-tennis-forum/flipping-the-ball-on-the-paddle) |
| Bigger, slower first stroke | Human teachers starting a resting paddle-ball moved "the paddle slower and further up than during the rhythmic behavior" | [Kober & Peters, ICRA 2009](https://www.ias.informatik.tu-darmstadt.de/publications/ICRA2009-Kober_5661%5B0%5D.pdf) |
| Off the floor | Tap with the foot, or press a sticky racket onto the ball and lift; nothing rolls it onto the bat | [PingSkills forum](https://www.pingskills.com/table-tennis-forum/picking-up-a-tt-ball) |

- I found nothing that spells out dip-then-hit or a rolling start for a ball resting on the paddle. Popping it up
  with a short upward stroke is what every source implies.
- A dip before the pop is how a hand gets runway without first lifting the ball. This is inference from the physics
  above, not something a source says.

## 3. How robots start

| system | start | source |
|---|---|---|
| Barrett WAM, two-ball toss juggling | A hand-set "initial stroke movement" is prepended to the learned cycle, because "applying the limit cycles PD-references from the start would result in a bad first throw". The second ball is released by a mechanism 3 m up. Via-points are set by hand: "most random initialization would not make any contact with the balls" | [Ploeger, Lutter & Peters, CoRL 2020](https://arxiv.org/abs/2010.13483) |
| Barrett WAM, paddle-ball on a string | "as we start with a still robot where the ball rests on the paddle, we require a start-up phase". This is a separate discrete primitive that blends into the rhythmic one | [Kober & Peters, ICRA 2009](https://www.ias.informatik.tu-darmstadt.de/publications/ICRA2009-Kober_5661%5B0%5D.pdf) |
| Franka, ball in a V-groove tool | The ball starts at rest. An offline optimal-control "swing-up" launches it; "each simulation run began with a standardized swing-up maneuver, at which point the closed-loop juggling controller took over" | [Ramadani et al., ICRA 2026](https://arxiv.org/abs/2606.06704) |
| Cassie with a paddle | "juggling was initiated by releasing the ball from rest" above the paddle | [Poggensee et al., ECC 2020](https://hybrid-robotics.berkeley.edu/publications/ECC2020_CassieJuggling.pdf) |
| Octo-Bouncer (stepper plate) | After balancing it alternates 5 small strokes (1 cm in 0.1 s) with 1 strong one (3 cm in 0.1 s), building up over a few bounces (`GetBallBouncing`, `Bouncing`/`BouncingStrong`) | [T-Kuhn/HighPrecisionStepperJuggler](https://github.com/T-Kuhn/HighPrecisionStepperJuggler) |
| Reist & D'Andrea blind juggler | Steady state only: a concave paddle and the paddle's acceleration at impact give open-loop stability. I found no start procedure | [Reist & D'Andrea, ICRA 2009](https://www.researchgate.net/publication/221074482_Bouncing_an_Unconstrained_Ball_in_Three_Dimensions_with_a_Blind_Juggling_Robot) |
| SO-100/SO-101 or other low-cost arm bouncing a ball | none found | n/a |

- **The pattern:** a separate scripted or optimised first stroke, or a ball dropped from above. A learned or feedback
  controller takes over once the ball is in flight.
- I found no robot that learns the start from rest end to end.

## 4. Sim checks on this arm

**Method.**
- Scratch scripts, not committed. They drive [BounceEnv](../bounce/env.py) with `ctrl_hz=500, substeps=1` and hold
  each goal for 10 steps: 50 Hz commands, logged every 2 ms.
- 30 ms command delay. Resting ball at the blade centre. Fixed e (`e_slope` 0). The 30 cm ceiling and the paddle box
  were relaxed so nothing was cut short.
- Each script ran in seconds on the Mac.
- Joint offsets are in rad from the hold command; + lowers the blade and tips it nose-down.

**Refit servo set `12v_r8`** ([bounce/servo.py](../bounce/servo.py); firmware speed limit 2.3 rad/s). Apex is height
above the hold surface; drift is at the apex, with − meaning toward the base:

| primitive | e 0.7 | e 0.6 | e 0.85 | vmax 1.8 | vmax 3.0 | inertia 0.09 | paddle travel |
|---|---|---|---|---|---|---|---|
| level push, 3 cm, 80 ms | 0 | 0 | 0 | 0 | 1.3 | 0 | -1.1..0 cm |
| wrist-only push, 0.35 rad, 120 ms | 5.3 [-0.8] | 5.4 | 5.3 | 4.1 | 7.4 | 5.2 | 0..5.3 cm |
| lift/elbow/wrist push (0.1, 0.25, 0.25), 80 ms, no dip | 11.5 [-3.5] | 11.7 | 11.6 | 8.6 | 16.6 | 9.3 | 0..8.1 cm |
| **[launch.py](../bounce/launch.py) P16 dip-and-flick** | **9.3 [-0.7]** | 9.2 | 10.2 | 5.7 | 15.0 | 5.8 | -2.7..5.1 cm |

- **Delay.** An open-loop launch from rest does not care about command delay. On the 12v set, 20-40 ms gave the same
  apex to 0.1 cm.
- **Restitution.** The default barely depends on e (9.2-10.1 cm for e 0.6-0.85). On these servos it is really a
  throw that follows a dip, not a hit. The timeline below shows why.

`DEFAULTS[16]` timeline in sim (e 0.7). Goals are in rad, degrees in brackets.

| phase | goal: lift / elbow / wrist flex | what happens |
|---|---|---|
| dip, 2 steps (0-40 ms) | +0.167 / +0.279 / +0.247 (+9.6° / +16.0° / +14.2°) | Nothing until ~50 ms (delay + servo ramp). From 60 ms the blade drops faster than g for ~40 ms and the ball floats up to 4.4 mm. The blade bottoms at -2.7 cm at ~120 ms and catches the ball softly |
| strike, 5 steps (40-140 ms) | -0.166 / -0.288 / -0.209 (-9.5° / -16.5° / -12.0°) | The upswing passes the hold height with the face level at ~175 ms and peaks at 1.1 m/s at ~200 ms. The ball is carried while the face swings from -5° to +9°, then released at ~225 ms at ~1.0 m/s |
| settle, 2 steps (140-180 ms) | 0 / 0 / 0 | The blade heads back. Apex 9.1 cm at ~320 ms; the ball is back down at ~450 ms |

**Peak joint speeds:**
- Default: ~1.2-1.3 rad/s on each joint.
- The 12v-set variant: 0.9-1.8 rad/s.

None comes near the 4.7 rad/s no-load speed. What limits the launch is how fast each servo builds speed, and the
speed cap, not top speed.

**Straightness.** The ball is released with the face tilted +8°, yet it drifts only 0.4-0.9 cm. While the ball is on
the face, the face swings from nose-down to nose-up through level, so the sideways pushes roughly cancel. Rubber
friction μ 0.1-0.6 changed the drift by under 1 cm (12v set, a primitive of the same shape). Longer strikes release later, at a larger tilt, and drift more
(next table).

**Tuning `DEFAULTS[16]` on `12v_r8`** (apex cm [drift cm]):

| change | e 0.7 | e 0.6 | e 0.85 | vmax 1.8 | inertia 0.09 |
|---|---|---|---|---|---|
| default (strike 5 steps) | 9.2 [-0.8] | 9.1 | 10.1 | 5.7 | 5.8 |
| strike 4 steps | 2.4 | 2.4 | 2.6 | 1.4 | 1.0 |
| strike 6 steps | 17.1 [-3.3] | 17.0 | 17.3 | 11.1 [-2.0] | 12.5 |
| 5 steps + a 6th at 0.5x the strike goal | 10.3 [-1.1] | 10.2 | 10.8 | 8.0 | 7.1 |
| 5 steps + a 6th at 0.75x | 14.4 [-2.2] | 14.4 | 14.8 | 10.5 | 10.4 |
| amplitudes x0.9-1.2 | 9.1-9.2 | 9.1-9.2 | 9.9-10.0 | n/a | n/a |
| ball 3 cm farther out / in along the handle | 10.8 / 7.7 | n/a | n/a | n/a | n/a |

**Other things tried:**
- **Bigger, longer dips on the faster `12v` set** (blade down 3.5 cm, contact while the blade is rising). These do get
  a restitution gain: the ball left at 1.5 m/s from a 1.3 m/s blade, which a push cannot do. But the apex was bimodal
  (11 or 15 cm from tiny changes), because the ball is caught and then pushed again.
- **Two open-loop strokes, a hop then a timed hit on the falling ball.** These were fragile. The hop height sets the
  timing of the second stroke, and it moves with servo strength (2nd apex 3.7-12.4 cm across e and servo changes).
  That second hit is better left to the closed-loop policy.

**What the trained policy does from rest** (`policies/v12-base.pt`, 12v set, nominal, 64 episodes):
- It pumps with small strokes: all three joints swing the same way by ±0.1 rad, the blade moves ±1 cm and the face
  ±2°.
- The first apex is 2.5-6.6 cm.

| restitution | episodes reaching 7 cm within 3 s | median time to 7 cm | notes |
|---|---|---|---|
| e 0.85 | 53/64 | 0.7 s | |
| e 0.7 | 36/64 | 1.3 s | Some episodes keep bouncing at 2-4 cm |

This matches the real-arm symptom, and the sim policy does it on servos stronger than the refit. A likely reason it
stalls on the real arm, not tested: the env's overhead-camera height error (`overhead_depth` 2-7 cm) is as large as
those hops, so the policy cannot time hits on them well.

**Caveats.**
- The sim's servo model sets the numbers. The same command spans 4.5-15 cm across the fitted ranges.
- The speed cap in particular is fitted, not read from the servos.

## 5. Recommendation

**Launch.**
- Keep the [launch.py](../bounce/launch.py) shape: lift, elbow and wrist flex all dip the same way for 40 ms, then all
  strike back for 100 ms, then hold.
- Do not use a level push or a wrist-only flick.
- Start from `DEFAULTS[16]` (P16) and tune on the real arm with the overhead camera, 5-10 trials per setting, ball
  centred:
  1. If the median first apex is under 7 cm, add a 6th strike step at 0.5x the strike goal, then 0.75x, then 1.0x
     (in sim +1, +5 and +8 cm).
  2. If it is over 13 cm, end the strike at 4 steps plus a 5th at 0.5-0.75x.
  3. Don't tune the amplitudes; they are past saturation.
  4. If the ball drifts toward the base by more than ~2 cm, the strike is too long. Prefer a shorter strike plus a
     partial step.
- **Before tuning:** read `Goal_Velocity`, `Maximum_Velocity_Limit`, `Maximum_Acceleration` and `Acceleration` on all
  five servos (register map in [tables.py](../third_party/lerobot/src/lerobot/motors/feetech/tables.py)). Then time
  one joint's step response.
  - A 2.3 rad/s cap is half the 12 V datasheet no-load speed (0.222 s/60°). If it is a register setting, raising it is
    the largest single gain: 3.0 rad/s gives 15 cm in sim.
  - The runner already forces `Acceleration` 0 and reads it back ([real/arm.py](../real/arm.py)).
- **Hand-off.**
  - The policy takes over 180 ms after the start, but in sim the ball leaves the face at ~225 ms. With 30 ms delay,
    the policy's first goals arrive before release and shape it.
  - Either train with exactly this hand-off (next section), or extend `settle_steps` to 3-4 so the ball is airborne
    when the policy starts.
  - The default keeps the blade within -2.7..+5.1 cm and under 10° of tilt. That is inside the env's paddle box
    (±6 cm).

**Training the policy around it.** Options in order:
1. **Scripted launch, then the policy** (the Ploeger, Kober and Ramadani pattern).
   - In training, make rest starts replay the same action sequence. Randomise the strike length by ±1 step and the
     partial step 0-1x, on `12v_r8` ranges (speed cap 1.8-4.0 rad/s).
   - Hand control to the policy at the same step as the runner.
   - The policy then learns the catch from exactly the states the runner gives it. It already handles 10-20 cm drops
     at 98-100 % ([research/scenarios_and_failures.md](scenarios_and_failures.md)).
2. **Backward hand-off curriculum**, if the policy should own the launch. Start with the full scripted launch, then
   hand over one step earlier each time the success rate holds, down to step 0.
   - Behaviour-clone the scripted actions for the steps the policy now covers. This is a warm start, which Ploeger
     found necessary, since random actions never launch the ball.
3. **Residual**, for camera-based correction. The policy adds at most 0.2 to the scripted actions during the
   launch.
4. **Pure RL from rest.** The current setup works against it:
   - train on `12v_r8` and dead rubber (e 0.6-0.85, `rubber_fullrange`);
   - turn on `w_first`/`w_wait` (both 0 by default);
   - the launch needs about -3..+5 cm of blade travel and ~10° of tilt for 0.3 s, which `w_home` (penalty past 3 cm)
     and `w_tilt` discourage;
   - a stronger launch leaves the ±6 cm `paddle_box`, which ends the episode: a 6-step strike reaches +8.8 cm and a
     no-dip push +8.1 cm.

   Without a warm start, expect the small-stroke pump-up it learned already.

Related: [scripts/tune_launch.py](../scripts/tune_launch.py) (sim tuning for launch.py),
[real/bounce_run.py](../real/bounce_run.py) `--launch` (runs it on the arm).
