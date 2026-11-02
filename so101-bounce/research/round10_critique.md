# Round 10 critique: why the real arm still cannot keep the ball bouncing

Evidence: today's real runs with v15-p32 at P32, the real ball and the overhead C920. 174611 was before levelling;
175420 and 175749 were after levelling. For comparison, 174006 is a rehearsal on the simulated 12v_r8 arm. The analysis
scripts are in the session scratchpad: critic_arm.py and critic_ball.py. Items marked **[R10]** are for the round-10
training agent; the others are runner, hardware or test changes.

## Ranked hypotheses

### 1. The real servos cannot accelerate the paddle as fast as the 12v_r8 model assumes. Confidence: high.

**For**
- The policy is not timid. It commands paddle upward speeds of 1.8-2.9 m/s (99th percentile), and no action component
  saturates (no |a| above 0.95). The real paddle achieves 0.29-0.35 m/s (99th percentile), with a maximum of 0.53 m/s.
- The scripted launch executes identical actions on both arms:
  - Real arm, 10 launches: the paddle peaks at 0.28-0.53 m/s.
  - Simulated 12v_r8 arm: 0.98 m/s, the value the primitive was tuned for.
- The elbow is the weakest joint, at 0.4-1.1 rad/s during real launches against 1.19 rad/s in simulation.
- Goal-to-measured lag is about 80 ms both real and simulated, so the gap is in acceleration and torque, not delay.
- The free wrist reaches 2.6-3.5 rad/s, so the speed limit itself is not what binds. The loaded joints simply do not
  build speed within a 60-100 ms stroke.

**Against:** none in the logs.

**Ruled out:** speed registers. Goal_Velocity and Goal_Time are 0, Acceleration 0 and P32 are written and read back
every run, and Torque_Limit is 1000.

**Candidate causes, cheapest test first**
- (a) The 12 V supply is current-limited, so the voltage sags when lift, elbow and wrist accelerate together.
- (b) Protection_Current 310 (about 2.0 A) clips the torque.
- (c) D_Coefficient 32 damps fast moves more than the model's kd_time does.
- (d) The 12v_r8 fit was dominated by slow motion, from 98 s of mostly gentle logs.

**Test (about 30 s, ball off)**
- Run the P32 launch primitive and ±10° elbow steps.
- Log Present_Position, Present_Voltage, Present_Current and Present_Load at the maximum bus rate.
- Repeat with D = 0.
- How to read the results:
  - Voltage dropping below about 10.5 V during strokes points at the supply.
  - Current flat at about 2 A points at the protection setting.
  - Faster strokes with D = 0 point at the damping.

**Change**
- **[R10]** Refit the servo model on the fast transients: the 10 real launch segments in 175420, 175749 and 174611,
  plus active play.
  - Add a supply or current torque cap, or lower strength and kt, until replaying the launch actions gives 0.3-0.5 m/s
    paddle speed and elbow 0.4-1.1 rad/s.
  - Randomise around that, not around 12v_r8.
- **Hardware:** if (a) is the cause, use a 12 V supply rated 5 A or more.

### 2. The scripted launch is tuned on the wrong dynamics, and the hand-off is mistimed. Confidence: high (a consequence of 1).

**For**
- On the real arm, the strike (3 steps, 60 ms) ends before the paddle has built speed.
- First bounces are 0-9 cm, with one attempt balanced 23 s and never launched.

**Change**
- Already in the runner: the stroke adapts by ±1 step from the first bounce, and the runner relaunches a resting ball.
- **[R10]** Retune the primitive on the refit model; expect 5-7 strike steps.
- **[R10]** In training, have rest starts replay the primitive with the strike length randomised over 3-8 steps, then
  hand off at the runner's step. This makes the policy robust to whatever length the runner's adaptation settles on.

### 3. Perception bias relative to the paddle after re-levelling. Confidence: medium.

**For**
- While active after levelling, the model's face tilt averages -2.0 to -2.3° in x (face leaning back), with a standard
  deviation of 2-3° and peaks of 9-11°.
- A ball on or near the paddle reads a median 2.6-4.1 cm off centre.
- The camera was calibrated at 18:53 yesterday with the old wrist offsets (flex 5.4, roll 95.5). level_ball.py changed
  them to 8.8 and 92.3, which moves the forward-kinematics paddle by about 1 cm in the frame the camera was fitted in.
  A policy that sees a centred ball as off-centre tilts to correct it and drives it off.

**Against:** part of the offset is real placement by the user's hand.

**Test:** rerun calibrate_camera.py (taped ball, 1.5 min) now that the offsets are fixed, and compare the camera position
and the off-centre median.

**Change**
- Always recalibrate the camera after level_ball.py; have the runner warn when the joint map is newer than
  camera_calib.json.
- **[R10]** Raise cam_bias from 0.008 to 0.025 m.

### 4. Camera exposure, blur, rolling shutter and latency at the edge of training. Confidence: medium.

Source: robust_setup.md.

**For**
- The C920 runs auto exposure at 8.3 ms with gain 248/255. In flight that is about 12 px of blur on a 30 px ball, which
  corrupts the radius, and so the depth from ball size.
- The C920 likely has a rolling shutter.
- The measured 62 ms latency sits at the top of the simulator's range.
- In-flight ballistic residuals (2 degrees of freedom, gravity removed):
  - z: 1.6 and 2.2 cm in two runs, which training covers; 12 cm in one run, an outlier segment.
  - xy: 1.1-4.2 cm.

**Against**
- Few flights happened at all, because the launches failed (item 1). Perception is not the first failure.
- At rest the error is 0.7 cm.

**Change**
- Lock the exposure at about 4 ms with uvc-util, plus a lamp.
- **[R10]** Centre the latency on 62 ms, range 45-120 ms.
- **[R10]** Add depth noise proportional to ball speed times exposure (σ_extra about 0.5 cm per m/s), plus rare
  in-flight radius outliers of 5-10 cm.

### 5. Sideways exits after weak or tilted hits. Confidence: medium (follows from 1 and 3).

**For:** attempts end 35-40 cm sideways; tilt peaks reach 9-11°.

**Change**
- **[R10]** Bias target apex sampling toward 10-13 cm. That gives 0.29-0.33 s of flight, more time for slow servos to
  re-centre than at 7 cm.
- **[R10]** Add a modest penalty on face tilt above 5° at contact.
- Keep the lateral reward.

### 6. The paddle is deader than trained. Confidence: low to unknown.

**For:** the user says it needs a lot of force.

**Against:** training already covers restitution 0.55-0.92, and sustaining needs only about 0.2-0.3 m/s of paddle speed
at impact even at e = 0.65.

**Test:** a camera drop test. Hold the paddle still, drop the ball from 20 cm, and take sqrt(rebound height / drop height)
from 3 drops; about 1 min. Then centre training on the measured value.

### 7. Reactiveness and state estimation. Confidence: low as a first-order cause.

**For:** the user says it needs to be "more reactive".

**Against:** the policy already commands fast moves (item 1). The loop runs at 20 ms with no overruns, so the latency is
camera plus servo, not compute.

**Change (later):** a ballistic Kalman filter giving position and velocity predicted to "now". It would have to go into
the simulator observation as well, or it shifts the input distribution. Not for round 10.

### Ruled out or minor

- Control rate: 50 Hz, steady.
- Action scale: commands are not saturated.
- Paddle level: fixed by level_ball.py to under 0.5°.
- Wrong camera: fixed by name lookup.
- Ball colour: white set.

## Round 10 summary

All items marked [R10] above:
1. Servo refit on fast real transients, with a current or supply cap.
2. Launch retuned on that model, with rest starts replaying it and the strike length randomised over 3-8 steps.
3. cam_bias up to 2.5 cm.
4. Latency centred on 62 ms, range 45-120 ms.
5. Speed-dependent depth noise and outliers.
6. Apex sampling biased toward 10-13 cm, plus a tilt-at-contact penalty.

Before round 10, if the arm is reconnected, run the 30 s launch-speed test from item 1: it decides between a supply fix
and a model change.
