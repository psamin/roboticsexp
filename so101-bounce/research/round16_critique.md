# Round 16 critique (2026-11-04/05)

Scope: the round-16 training jobs (v24_*), tonight's real runs, and the runner. Code is at 6ed1f83 plus the
uncommitted working tree as of about 00:15 Nov 5, which since the brief adds a std reset and ent_coef for reach, the
runner search window and the per-return re-trim.

Sources:
- Cluster: ~/so101-bounce-results/v24-p32-*/train.log and metrics.csv. drop16 and dense were read at 25 min (jobs
  13735/13736); reach and reach-sparse at 10 min (jobs 13739/13740).
- Real: real/logs/run-20261104-231308, -231959 and -234427. The last has 105 activations over 724 s.
- Scripts: the session scratchpad critique2/. Every local check ran in seconds. No serial port or camera was opened,
  and nothing under real/ was written.

The brief is out of date in one place. The reach job now running (resubmitted 23:45) has w_edge 1.0, with w_hit at its
0.3 default rather than 1.0. It also resets std to 0.1 and adds ent_coef 0.002, and v24_p32_reach_sparse is a new
ablation (configs/variants.yaml:504-537).

## Verdict

- **drop16 and dense cannot learn what they were set up for.**
  - Exploration has been frozen at the e^-5 floor since round 10 (item 1).
  - The height rewards give no gradient across a 7 cm gap (item 5).
  - After 25 min, the deploy-relevant numbers have not moved: apex error 8.0 → 8.0 cm, moderate survival 0.12 → 0.12.
- **The real failure is systematic, not noise.**
  - In run 234427, 70 % of 105 tries end with the ball last seen beyond the blade tip.
  - At contact the policy already hits with the face about 3° more tip-up than it does in sim.
  - So something tips the real face forward by about 3° relative to the model (item 2). Nothing in round 16 models
    it.
- **Two simulator inputs are now measured, and both are off.**
  - The new paddle's restitution is about 0.75-0.9, but training weights 0.55-0.75 (item 3).
  - The camera compresses flight heights to 0.79×, with a free-fit g of 4.9 m/s², even after the 23:31
    recalibration (item 4).
- **Reach is not safe to run as is.** Its goal box reaches below the base plane, and the runner has no workspace
  guard (item 6).

## Ranked issues

| # | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| 1 | blocker | **Exploration is frozen, and has been since round 10.** log_std sits just below the clamp (bounce/networks.py:16, :50), where its gradient is zero. ent_coef is 0 (bounce/ppo.py:19). Every fine-tune from v16 to v24 has explored with σ = 0.0067 in tanh pre-activation units. In joint space that is 0.08-0.13°, which is about the sim servo deadband (0.086°, bounce/servo.py:31). | Every policy from v16-p32-lo through v23-p32-learn16 has log_std −5.0001 to −5.0005, identical to 4 decimals. The clamp gradient at −5.0001 is 0.0 (checked). drop16 and dense report entropy H = −17.91 at every iteration, which is a mean log σ of exactly −5. Their evals at 0, 12.4 and 24.9 min: nominal survival 0.86 → 0.84/0.86 → 0.87/0.90; apex error 7.9 → 8.0/7.8 → 8.0/7.9 cm; moderate 0.12 every time. In reach, the new std reset is already collapsing: σ went 0.100 → 0.042 in 9.4 min, so mean log σ falls about 0.09/min and reaches the −5 floor around minute 30. After that the clamp freezes it again. | Stop 13735/13736 and resubmit with the std reset. Replace the hard clamp with a floor that keeps its gradient, e.g. `log_std = MIN + softplus(raw - MIN)`, or use a fixed annealed σ (0.1 → 0.03). Raise ent_coef about 5× (0.002 is too weak at this rate), or adapt it to a target entropy. Log action_std and stop the run if it is below 0.01. |
| 2 | blocker | **The real ball is lost forward because the face acts about 3° more tip-down than the model.** Neither the sim nor round 16 covers this. | **Outcomes:** last seen beyond the tip (x > +8 cm) in 8/16, 17/17 and 74/105 activations. Median flights per activation 4, 2 and 2; at least 5 flights in 17-25 %. **Face pitch at contact** (FK of the logged joints, + = tip down; 240 contacts): real −3.2°, still −2.2° if the contact estimate is shifted 40 ms earlier. The same policy in sim (d = 0, 20 cm drops): −0.26°, and pitch changes only about 0.5° per 20 ms around impact. **Drift:** a rigid face at −3.2° should send the ball about 4-5 cm per bounce toward the handle (e 0.85, v_in 1.53 m/s, T 0.31 s). Measured: +0.4 ± 0.6 cm per bounce (65 bounce pairs). **Hold:** the armed hold also sat 1.2-2.2° tip-up through the whole of run 234427, and the ball still went forward. The new paddle was levelled "by eye" (commit fa0922d), never with ball roll. **Two candidate causes:** (a) a static mis-level in the jaws or tape; (b) the face giving tip-down at impact through wrist slack (1.2-1.6°) and gear wind-up. The sim's ball impulse never acts on the arm (bounce/ball.py:71-77; env.py:879), so (b) is absent from the sim. | **Measure first** (both need your OK to move the arm). (a) level_ball.py, or a phone level on the held blade, at hold and after a try. (b) A 240 fps side video of 10 contacts, to see whether the tip dips. **Then:** if (a), re-level with level_ball.py and recalibrate the camera. If (b), add an impact tilt to the sim (tip-down, proportional to the hit's moment about the wrist, up to about 2°). In both cases, train invisible tilt over its full ±3° range in a share of episodes (table_tilt_full_p / level_error_full_p). |
| 3 | high | **Restitution prior is wrong for the new paddle.** Training draws 60 % of episodes from e 0.55-0.75 (configs/variants.yaml:474-475). | Inferred from tonight's bounce timing: v = g·T/2 from contact-to-contact times, with paddle vz from FK, over 65 pairs at v_in 1.53 m/s. That matches e0's 1.5 m/s reference (bounce/ball.py:21). Median e is 0.91 using paddle vz at the contact estimate and 0.75 using the peak vz within ±50 ms. Steady 12 cm bouncing at a paddle speed of only 0.07-0.15 m/s also needs e ≥ 0.8 (sim, same policy: 0.18 m/s). **learn16 in sim at fixed e** (d = 0, 384 episodes): 0.6 → 98 % survival but only 8.5 bounces in 8 s (dribbling); 0.75 → 97 %, 27 bounces; 0.88 → 78 %, 22 % dropped. | Confirm with an acoustic drop test: 10 drops on the held paddle, phone microphone, e = T(n+1)/T(n). Then centre `rubbers` on the measured value ±0.05, e.g. 70 % in 0.78-0.92. |
| 4 | high | **In-flight depth is still compressed after the 23:31 recalibration, and the sim does not model it.** The sim's depth error is zero-mean noise plus a scale on the full camera distance (env.py:1065-1075). That scale is almost constant over a 15 cm flight, so it cannot shrink height changes by 20 %. | Free-g parabola fits to flights with at least 6 detections: median g 3.9, 8.3 and 4.9 m/s². In 234427, 91 % of 138 fits are below 9.81. Camera apex ÷ apex from flight time: 0.78, 0.94, 0.79. So run_report's "10-12 cm" is about 12-15 cm true. Geometry at the blade (current calibration): 1 px of radius is 4.3 cm along the ray, split 0.82 vertical and 0.57 forward. A radius error therefore also moves x. | Rest the ball on a stand at 0, 5, 10, 15 and 20 cm above the blade centre and tip, and fit camera height against ruler height (15 min). Add a per-episode height gain k ∈ [0.7, 1.0] about the paddle depth in `_camera`. Until then, read apex from flight time, not from camera height. |
| 5 | high | **The height rewards give no gradient, and w_wait cancels the alive reward.** | The policy bounces about 9-10 cm against a 14-20 cm target (apex error 8.0 cm). Apex reward: σ 3 cm, so exp(−(7/3)²) ≈ 0.004 (env.py:948). Aim reward: σ 0.2 m/s against a gap of 0.46 m/s, exp(−5.3) ≈ 0.005 (env.py:936-939). Measured r_aim: +0.2 per episode in reach and +1.2 in dense, against r_hit +4.7 to +7.5. w_first / w_wait 1.0 are inherited from v16env (configs/variants.yaml:282-283). w_wait costs −1/s until a bounce lands within 3 cm of the target (env.py:986), which offsets the +1/s alive reward. r_launch went −2.2 → −2.95 per episode. | Ramp the target from 11 cm upward with the curriculum, or set target_apex to [0.11, 0.16]. Use apex_sigma 0.06 and aim_sigma 0.5. Set w_wait to 0 in drop-only variants. |
| 6 | high | **Reach's action range can drive the paddle into the table, and the runner has no guard.** | FK over the action-box corners from the hold command: at ±0.45 rad lift/elbow, the paddle centre goes down to z 0.005 m and the blade rim to −0.056 m. At ±0.30 rad the rim stays above +0.054 m. This assumes base z = 0 is the table (unverified). The runner only clamps to joint limits (real/bounce_run.py:373). The sim ends an episode outside paddle_box (env.py:1014), so the policy has never learned what to do outside it. Pan has never been step-tested: real play peaks at 50-53 °/s over a 12-16° goal range, and speed_test covered only lift, elbow and the launch stroke. 11 % of activations in 234427 are false starts while the hand is in view (item 10). | **Workspace guard in the runner:** before sending a goal, check its FK. Paddle centre within the variant's paddle_box of home, and rim z ≥ home z − 0.12. Clip the goal if it fails; go to "returning" if the measured paddle leaves the box. **Pan step test** at ±0.15, ±0.3 and ±0.45 rad before the first reach run. |
| 7 | medium | **drop16 and dense never see off-centre drops.** The variant comment says "landing up to 10 cm off centre", but the spread scales with a task level of 0.06-0.14. | env.py:560 gives spread 0.01 + 0.09·dt, at most ±2.3 cm per axis. The task level fell 0.14 → 0.06 (drop16) and 0.08 (dense) during the runs. Only reach sets drop_blade_p (configs/variants.yaml:521). | Add drop_blade_p 0.6 to v24env. |
| 8 | medium | **The eval cannot rank fine-tunes.** | user_hw and full are the same suite, because side_cam_prob is 0 (configs/variants.yaml:252; bounce/evaluation.py:9-11). They give identical results in every eval (0.00 survival, 1.4 bounces, 7.5 cm). The d = 1 suites score 0 survival, so half the score weight is constant. Moderate is 0.12 to two decimals at every eval. The score ignores apex error (evaluation.py:57-60). There is no initial eval after a resume, because eval_history is carried over (trainer.py:111); for reach (rescaled actor, std reset) there is no baseline at all. Evals take 145 s each, about 14.5 of the 65 min budget (trainer.py:41-42). | Use suites nominal, d 0.25, d 0.5 and a "measured rig" suite (e 0.85, hidden tilt 2-3°, depth gain 0.8, cam_view). Run an initial eval on resume. Add the share of apexes within ±3 cm of the target to the score. |
| 9 | medium | **The camera-moved check is off: real/camera_ref.png does not exist.** | `ls` finds no such file. camera_view_check returns None (real/rig.py:214-217), which only prints a warning (real/bounce_run.py:181). The 23:31 calibration ran before calibrate_camera.py saved the reference (now :154). | Now, if the camera has not been touched since 23:31: `uv run python real/camera_check.py --set-reference`. Make None a stop unless --camera-moved-ok. |
| 10 | medium | **Some tries start on a ball that is not yet dropped.** | In 234427, 12 of 105 activations (11 %) have at most 7 detections, start more than 22 cm up and 10-25 cm toward the arm, and are lost within about 1 s with "no detection". Likely the hand moving the ball fast enough to pass the vz < −0.5 test (real/bounce_run.py:305-309). Video not checked. | Before starting, require the ball to be falling for at least 3 frames, with |vz − g·t| consistent with free fall, and within START_LATERAL of the paddle. This matters more with reach's ±17 cm pan. |
| 11 | medium | **The frame's top edge drops toward the tip; drop16/dense model it as independent of x.** | From the current calibration, the frame top (ball-bottom apex) is 22 cm above the blade centre, 17.5 cm at +8 cm (the tip) and 15 cm at +12 cm. drop16/dense use view_top [0.20, 0.28] regardless of x (configs/variants.yaml:497). Only reach uses cam_view. With forward drift (item 2), targets of 17-20 cm put the apex out of view. | Use cam_view in every v24 variant. Keep the target at or below about 15 cm true until item 2 is fixed. |
| 12 | medium | **Reach's ADR levels collapsed, so the final policy may be trained for an easy world.** | Levels went from [0.78, 0.56, 0.62, 0.12] to [0.38, 0.18, 0.24, 0.00] in 2 min. At 9.4 min they were [0.42, 0.20, 0.44, 0.04]: sensing is at a third of where it was. The σ = 0.1 noise and the harder starts push boundary-episode survival below 25 %. | Hold ADR levels fixed for the first 10-15 min after a std reset. Judge the result at d 0.25/0.5, not only at the final levels. |
| 13 | low | **Hold pitch varies ±2° between runs, and the startup trim does not converge.** | Armed FK pitch: +0.45 / +1.16 / −1.39° (run medians). In 234427 the elbow goal sat 2.5° off the hold all session with the pitch at −2.2°. The per-return re-trim (real/bounce_run.py:380-392) addresses this. | After each trim, print and log the pitch, and refuse to arm above 1°. |
| 14 | low | The auto-level trim carries over between tries (real/bounce_run.py:231, never reset), while the sim resets it each episode. | Code. | Reset it at each activation, or carry it over in the sim as well. |
| 15 | low | Logs do not record the target or the flags, so tonight's targets cannot be recovered from the npz files. | real/bounce_run.py:440-441 | Save `vars(a)`, the config.yaml text and the calibration's `created` field. |
| 16 | low | The tanh rescale is not exact: tanh(r·u) ≠ r·tanh(u). | Real |a| p99 per joint is 0.47-0.62 (pan), 0.33-0.48 (lift) and up to 0.89 (wrist). At those values the rescaled goal is 12-17 % larger. | Acceptable; fine-tuning absorbs it. |

## Specific training questions (round 16)

| Question | Answer | Evidence |
|---|---|---|
| Resumed curriculum levels | Carried over correctly ([0.80, 0.58, 0.64, 0.14]). drop16/dense drift down slowly (task level 0.06-0.08); reach collapses (item 12) | train.log, metrics.csv |
| d = 1.0 suites scoring about 0 | Yes. They are dead weight, and user_hw is a duplicate of full (item 8) | evals at 12.4 and 24.9 min |
| PPO std nearly collapsed | Worse than nearly: frozen, with zero gradient, since v16 (item 1) | checkpoint log_std, clamp gradient check |
| New reward scale on a resumed critic | Not a problem. Reach's explained variance went 0.86 → 0.71 → 0.89 within 3 min; drop16/dense held 0.87-0.90 | train.log |
| tanh rescale of the actor | Small error (item 16). Note that it pushed log_std further below the clamp (pan −5.92), which is now superseded by the std reset | scratchpad v23-p32-learn16-reach.pt |
| Wider action range vs servo speed and torque | The sim caps torque per joint through 12v_r11, but with one range set for every joint (bounce/servo.py:107-118). Pan is unvalidated at large amplitude. The safety issue is item 6 | servo.py, real logs |
| w_back / handle keep-out vs chasing | Minor. The keep-out is tested in base-frame x (env.py:1009-1011), not along the handle, which is 29° off at a 0.5 rad pan. w_back only charges first apexes behind home, so a far beside drop toward the arm carries a penalty the policy cannot avoid | code |
| Is 65 min enough? | Not with frozen exploration: 25 min changed bounces (12 → 14) and nothing else. Reach cannot be judged yet, and its σ will be back at the floor around minute 30 | evals |

## Remaining sim-to-real gaps (beyond the table)

| Gap | State | Next step |
|---|---|---|
| Latency | Camera 62 ms + 0-20 ms waiting for the tick + 3-4 ms compute + about 100 ms goal-to-joint ≈ 175 ms. That is 57 % of the 0.31 s period at the true 12 cm apex; at 15 cm it is 51 %. A higher target helps here, but the frame top limits it (item 11) | No change; keep the target at about 13-15 cm true |
| Servo model | The held-out fit is fine on joint rms; validation on paddle vz and tilt at contact is still undone. Real wrist-flex goal minus joint at contact is 3° median | Use the 240 fps video from item 2 |
| Ball spin | Not measured. Hand drops can carry spin; sim drop spin scales with task level (env.py:660), so it stays near 0 | Low priority |
| Paddle centre vs model | Within about 1 cm at the calibrated camera (setup_state_audit.md:19-21) | None |
| Detection coverage in flight | 75-82 % of in-flight frames have a detection | Sim cam_dropout is about 1 % at d = 0; raise it toward 20 % in flight |

## Evaluation: what to measure

- **Measure bounces, not survival.** In sim at e = 0.6, learn16 "survives" 98 % with 8.5 bounces in 8 s.
  - On the real arm, report per try: number of flights, apex from flight time, and where the ball was last seen.
  - Tonight's baseline (234427): median 2 flights; at least 5 flights in 17 %; 70 % lost beyond the tip.
- **Next session, most information per minute first.** Items 2-6 need your OK to move the arm.

| # | Measurement | Time | Settles |
|---|---|---|---|
| 1 | `camera_check.py --set-reference` (camera untouched since 23:31) | 1 min | Item 9 |
| 2 | Ball-roll check or phone level on the held blade, at hold and after one try | 5 min | Item 2 (a), the static tilt |
| 3 | 10 drops onto the held paddle (policy off), phone microphone | 10 min | Item 3 (e) |
| 4 | 240 fps phone video from the side at blade height with a ruler, during 10 policy contacts | 20 min | Item 2 (b), the dip at impact; true apex; face pitch at contact |
| 5 | Ball on a stand at 5 heights over the centre and the tip, camera height against ruler | 15 min | Item 4 (height gain) |
| 6 | Pan step test at ±0.15 / 0.3 / 0.45 rad (only if reach will be run) | 10 min | Item 6 |

## Unverified

- Why the face acts tip-down: static mis-level or give at impact (item 2). The symptom is measured; the cause is not.
- The restitution estimate rests on 30 fps contact times (±17 ms) and 50 Hz FK paddle speed. Drag is ignored, and any
  blade flex is invisible to the joint encoders.
- That z = 0 in the base frame is the table surface (item 6).
- Whether the false starts in item 10 are the hand or the ball held over the arm (video not checked).
- The rate at which reach's σ collapses after minute 10 (extrapolated from 0-10 min).
