# Ball 3D position: measured error and how to improve it (Nov 4-5 2026)

## Answer

The overhead C920 tracks the ball's bearing well: pixel noise is about 0.5 px, and 89 % of frames in active play have a detection. The weak part is height. Depth comes from the ball's apparent radius, and that radius is accurate when the ball rests on the black blade (0.7 cm depth rms at the calibration poses). In flight it is biased:

- The ball reads about 1 px too small, so depth comes out +5.4 cm too far.
- In height that is about -3.5 cm on average, and it grows with height. Measured heights are compressed by about 30 %.
- The scatter around that bias is about 3 cm.
- An independent check, the contact timing between consecutive flights, puts the average in-flight height error at -2.2 cm (IQR -3.9 to -0.5).

The earlier "g = 4.6 m/s^2" finding is mostly an analysis artifact:

- About a quarter of the flights cut by the radius-height threshold are two bounces merged into one (fitted g 0.4).
- The rest are flattened by the compression above.
- With flights split at contacts found in the pixel track, the free-fit g is 9.2 (IQR 6.3 to 10.4).

**Top 3 steps** (details in the table at the end):

1. **Remove the in-flight radius bias at the source.** Put a dark matte background behind the bounce volume, add more light, lock a short exposure, and measure the radius from the ellipse minor axis.
2. **Fix the calibration.** Calibrate the intrinsics with a checkerboard and make the camera mount rigid.
3. **Add the OV9281 side camera and triangulate.**

A gravity-model filter on the single 30 fps camera helps less than expected (about 1 cm). It is worth doing after step 3, or with audio contact timing.

## Data and method

Logs:
- `run-20261104-231308`
- `run-20261104-231959`
- `run-20261104-234427` (newest, 23:32-23:44)

Videos were matched to these logs frame by frame by reading back the drawn time stamp. Scripts and outputs are in `/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/perception/`:

| Script | What it does |
|---|---|
| `common.py` | Loader: FK at capture time, paddle-frame position, inverse projection to (u, v, r) |
| `video_index.py` | Per-frame stamp and drawn circle |
| `rest.py`, `still.py` | Resting-ball noise |
| `flights2.py` | Flight fits and contact-timing check |
| `causal.py` | Real-time ballistic estimate |
| `gaps.py` | Dropouts |
| `gcheck.py` | Origin of the low g |
| `blur.py` | Blur measurement (inconclusive) |

Each script runs in about 1-2 s, except `video_index.py`, which takes about 45 s on the 12-min video.

**Calibration provenance (verified).** The 23:13 and 23:19 runs converted pixels with the committed Nov 2 20:50 calibration. With that calibration the drawn circles match the projected detections to 0.3 px rms; with the 23:31 one they are off by 7.7 / 20 px. The camera had moved since Nov 2:

| Comparison (static background, phase correlation) | Shift |
|---|---|
| Nov 2 videos vs 23:11 video | dx -2.2, dy +18.4 px |
| 23:15 vs 23:11 | 1.1 px |
| 23:32 (newest) vs 23:11 | dx +1.5, dy -4 to -6 px |

So the first two runs carried a stale calibration (about 2.4 cm at the paddle). Below they are re-projected with the 23:31 calibration, which leaves about 5 px of residual offset. Headline numbers come from the newest run, which used the correct calibration.

**Reference for in-flight height.** I fitted a pixel-only ballistic model to each flight: the 3D start point and velocity, g = 9.81, using u and v only, no radius. Its own bias was checked with the contact-timing test below.

**Contact-timing test.** For two consecutive flights with one bounce between them, each method's apex (time and height above the FK paddle face at the contact) predicts when the ball lands and when the next flight took off. These must coincide. The height offset that zeroes the mismatch is that method's height error, and it is independent of the radius.

## 1. Measured error

### Resting ball

These come from 5 still segments (72 detections) in run 231308 plus the calibration poses.

| Quantity | Value |
|---|---|
| Radius noise (frame to frame, detrended) | 0.05-0.09 px, which is 0.25-0.4 cm depth |
| u / v noise | 0.4-1.3 / 1.2-1.8 px (paddle moved 7-15 mm in these segments) |
| Height above face, stale Nov 2 calibration | +1.1 cm |
| Height above face, re-projected with 23:31 calibration | -0.3 cm |
| Depth from radius at calibration poses, 23:31 | 0.7 cm rms (15 poses) |
| Depth from radius at calibration poses, Nov 2 | 1.1 cm rms |
| Outlier pose 16 (repeat of pose 1) | radius 11.1 vs 13.9 px, +18 cm depth |

Both calibrations put the focal length at exactly 1.15x nominal (521.7 px). That is the upper edge of the search in [calibrate_camera.py](../real/calibrate_camera.py), so f is clamped. No lens distortion is modelled.

### In flight (newest run)

There are 153 flights split at contacts; 120 have a well-conditioned pixel fit, and 980 detections were compared.

| Quantity | Value |
|---|---|
| Free-fit g, run_report-style split (h > 1.5 cm) | 4.66 (IQR 0.98-7.48), 142 flights |
| ... of which 33 contain a contact (merged bounces) | g 0.37 |
| ... and the other 109 | g 6.12 (IQR 3.4-8.2) |
| Free-fit g, split at contacts in the pixel track | 9.15 (IQR 6.3-10.4) |
| z residual about a g = 9.81 parabola (radius positions) | 1.29 cm rms per flight (IQR 0.91-2.10) |
| Radius depth minus ballistic depth | +5.4 cm median, robust std 4.6 cm |
| Radius height minus ballistic height, all | -3.5 cm median, robust std 3.0 cm |
| ... at ballistic height < 4 cm (ball 1.5 m/s, blurred) | -0.9, robust std 4.6 cm |
| ... at 4-8 cm | -2.6, robust std 4.2 cm |
| ... at 8-12 cm | -3.9, robust std 3.1 cm |
| ... at >= 12 cm (ball 0.6 m/s) | -3.8, robust std 2.3 cm |
| Fit: height error vs height | -0.31 cm per cm, i.e. about 30 % compression |
| Apex above contact: radius / ballistic / timing (gT^2/8 from 340 ms apex spacing) | 12.3 / 15.6 / 14.2 cm |

**Contact-timing test** (68 bounce pairs; implied common height error):

| Method | Median | IQR |
|---|---|---|
| Radius (today's pipeline) | **-2.2 cm** | -3.9 to -0.5 |
| Pixel-only ballistic (no radius) | +1.4 cm | +0.3 to +2.2 |
| Pixels + radius at 0.5 px, full flight (batch EKF) | -1.2 cm | -2.7 to -0.2 |

All runs together (re-projected) give the same picture: radius -2.5, ballistic +1.4, fused -1.5 cm.

### Dropouts

| Quantity | Value |
|---|---|
| Frames in active play without a detection | 352 of 3221 (10.9 %) |
| Missed between first and last detection of a flight | 61 of 1041 (5.9 %) |
| Misses inside well-fitted flights at predicted height >= 10 cm | 55 of 62 |
| Gaps > 50 ms / > 117 ms between detections | 126 / 61 of 2798 |
| Highest point reached in the image (v min) | median 73 px; the top edge is not reached |

### Where on the blade

There is no meaningful dependence on position over the blade. Above 8 cm, the height bias is -3.3 to -4.3 cm across x bins from -20 to +20 cm and -3.5 to -4.1 cm across y. Height (and so speed) is what matters.

In the image, only 7 % of in-flight detections lie wholly inside the blade outline. The ball clears the black blade within 1-2 frames, so above that it is a white ball on the light-grey table (about 75-80 grey levels of contrast).

### Mechanism

From [real/rig.py](../real/rig.py) `BallFinder.refine`, frames, and the numbers above:
- **Different backgrounds at rest and in flight.** At rest the Otsu window contains the black blade, so the threshold is low and the whole disc is kept. In flight the window is ball on light grey, so the threshold is high and the shaded rim is dropped. The radius comes out about 1 px small, which gives the +5 cm depth bias. The bias is largest at the apex, where the ball is slow and sharp.
- **Motion blur near contact.** The ball is a vertical streak. One measured frame:
  - The finder radius was 12.1 px against 16.5 px typical.
  - The streak extended at least 12 px beyond the ball at about 850 px/s, which implies an exposure of at least 14 ms.
  - This is a single-frame estimate. The automated blur measurement failed because the recorded circle overlay covers the ball's edge.
- **Not testable on these recordings.** Every detected frame has the red circle drawn on the edge, so sub-pixel radius methods cannot be evaluated offline.

### Pixel-only fit has its own problem

The landing point (end of flight k) and take-off point (start of flight k+1) should coincide:

| Method | Take-off minus landing, x |
|---|---|
| Pixel-only ballistic | 7.2 cm (IQR 4.8-9.6) |
| Radius | 1.2 cm |

From one camera looking 35 deg off vertical, the velocity along the line of sight is weakly observable. The clamped focal length, the missing distortion model and the rolling shutter probably bias it (unverified which one).

### Real-time version

I anchored each new flight at the previous flight's predicted landing (prior 1.5 cm horizontal, 1 cm vertical) and fitted the first m detections. This was worse than the raw radius:

| Detections into the flight | Anchored fit, \|error\| median | Raw radius, \|error\| median |
|---|---|---|
| 2 | 8.0 cm | 2.4 cm |
| 4 | 7.0 cm | 3.5 cm |

At 30 fps the contact time is only known to about ±16 ms, which is about ±2.5 cm at 1.5 m/s.

## 2. Options, ranked by expected gain per effort

| # | Option | Expected gain (height) | Latency | Effort | Evidence |
|---|---|---|---|---|---|
| 1 | Dark matte cloth/board under and behind the bounce volume in the overhead view; more light; `uvc.lock: true` with `max_exposure` about 30-40 (3-4 ms; already supported by `lock_camera`); radius from the `cv2.fitEllipse` minor axis with a local-background threshold instead of Otsu | Bias -3.5 → about 0 cm; scatter 3 → about 1 cm, bounded by the at-rest 0.7 cm depth rms (target, unverified until one run) | 0; likely -5 to -10 ms from the shorter exposure (unverified) | About 1 h of setup + about 30 lines in `BallFinder`, then rerun `flights2.py` on one run | Mechanism above; rest vs flight numbers |
| 2 | Checkerboard intrinsics (`cv2.calibrateCamera`: f + distortion) fed to the PnP; rigid camera mount; keep the moved-camera check | Removes up to about 2.4 cm from stale extrinsics (18 px moves seen) and the clamped-f error (size unverified). Prerequisite for 3-5: with the stale calibration only 42/82 pixel fits were well-conditioned vs 76/82 | 0 | About 1 h | Provenance section; f = 1.15x bound |
| 3 | OV9281 side camera, triangulating with the overhead (timestamps, ballistic interpolation between unsynchronised frames) | Height from the side camera's image row: 0.5 px at f ≈ 900 px, 0.5 m is about 0.3 mm. Calibration-limited to about 0.2-0.5 cm (derived; the sim's `tri_noise` is 2 mm, and the policy already trains with a side camera in 50 % of episodes) | Frame period 33 → 8.3 ms (120 fps) or 4.8 ms (210 fps at 640x400). The sim's latency model gives about 32 vs 62 ms (unverified on hardware) | Purchase + 1-2 days (mount, extrinsics with the same arm-pose PnP, triangulation) | [UB0232 datasheet][ub0232], [B0332 datasheet][b0332] |
| 4 | Ballistic filter (EKF, g known, reset at bounces) on the C920 alone | About 1 cm (contact test: -2.2 → -1.2 cm, full-flight). Early-flight estimates are worse than raw at 30 fps | About 0.1 ms compute. It can also predict across the 62 ms latency, but that changes the policy's input distribution, so mirror it in `bounce/env.py` | 1 day | Contact test; causal test |
| 5 | Audio contact timing (mic, 10 kHz high-pass, energy onset) to anchor the filter at each bounce | Turns the ±16 ms contact time into about 1 ms. Mainly makes #4 work in real time (unverified on this rig; servo noise untested) | Audio buffer about 2.7 ms at 128 frames / 48 kHz (derived). The audio-to-camera clock offset must be calibrated | 1 day | [Gossard et al. 2024][gossard] |
| 6 | Wrist camera as a second view | Unverified: its view of the bounce volume was not checked (cameras were in use). 30 fps, rolling shutter, not synchronised. It moves fastest at the strike, exactly when the ball is near the paddle. Needs hand-eye calibration through FK. Worse geometry than a fixed side camera | Same as the C920 | 1-2 days | — |
| 7 | C920 at more than 30 fps | Not available: 30 fps at 1080p and 720p; 60 fps is a C922 feature | — | — | [spec listing][c920spec], [review][c920rev] |
| 8 | CNN detectors (TrackNetV3, YOLO) or HoughCircles | No gain. Detection is not the bottleneck: 89 % of frames are found, and misses cluster at the apex over the light background, which #1 fixes. These give centres/boxes, not sub-pixel radius. TrackNetV3 needs 8-frame windows (adds latency) and runs at 25 FPS on a GPU. HoughCircles radius is gradient-based with coarse steps | +1 frame or more | — | [TrackNetV3][tracknetv3], [TTNet][ttnet] |

**Published systems for context.** Every closed-loop system found uses two or more synchronised cameras; none uses depth from apparent size:

- **DeepMind table tennis:** 2x Ximea at 125 fps, global shutter, 4 ms exposure, triangulation + Kalman filter, about 40 ms observation latency ([paper][deepmind], [paper][deepmind2]).
- **MPI Tübingen:** 4 cameras, robust triangulation ([paper][mpi]).
- **Nakashima et al. 2008:** stereo at 120 Hz ([paper][nakashima]).
- **Omron FORPHEUS:** stereo at 80 fps ([Omron][omron]).
- **TT3D (monocular, broadcast video):** anchors the trajectory at the bounce by intersecting the image ray with the table plane and fits a flight model, rather than using the ball's diameter ([paper][tt3d]). Its simulated 3D error is 9-30 cm. This is the same reason #4 alone is weak here.
- **Blind juggler (Reist & D'Andrea):** uses no sensing at all ([press][reist]; primary paper not read).

## 3. Recommendation

| Step | Before (measured) | After (expected) | Latency | Effort |
|---|---|---|---|---|
| 1. Dark backdrop + 3-4 ms locked exposure + ellipse-minor-axis radius | In-flight height bias -3.5 cm (contact test -2.2), scatter 3.0 cm; 4.6 cm near the paddle; 10.9 % frames missed | Bias < 1 cm, scatter about 1-1.5 cm; fewer apex misses (unverified: check with one recorded run and `flights2.py`) | 0 to -10 ms | About 2 h |
| 2. Intrinsic checkerboard calibration + rigid mount | f clamped at the search bound; camera moved 18 px (about 2.4 cm) Nov 2 → Nov 4 and 4-6 px since | < 2 mm drift; f and distortion measured | 0 | About 1 h |
| 3. OV9281 side camera + triangulation (then #4/#5 on top) | As after step 1 | About 0.2-0.5 cm height; 120-210 fps; no blur with a global shutter and short exposure | About -30 ms (sim model; unverified) | 1-2 days + hardware |

Notes:

- **Record raw frames.** Before step 1, record frames without the overlay (or a parallel raw video) so detector changes can be scored offline. The circle drawn on the ball's edge blocked that test here.
- **The sim does not model the measured error shape.** In the sim, overhead depth error is the same at rest and in flight (`depth_scale`, drift, white noise, optional `depth_speed_k`). Reality is unbiased at rest and about 5 cm (7 %) too deep in flight. If step 1 does not remove the bias, model it in `bounce/env.py`.
- **Unverified items:**
  - That the C920 `exposure-time-abs` units are 100 µs (taken from the [real/rig.py](../real/rig.py) comment).
  - The OV9281 latency and price, and its behaviour on macOS.
  - The wrist camera's field of view.
  - The audio approach under servo noise.

## Sources

[ub0232]: https://www.uctronics.com/download/Amazon/UB0232_OV9281_Global_Shutter_UVC_Camera_Datasheet.pdf
[b0332]: https://www.uctronics.com/download/Amazon/B0332_OV9281_Global_Shutter_UVC_Camera_Datasheet.pdf
[c920spec]: https://www.comx-computers.co.za/LOG-960-001055-specifications-340180.htm
[c920rev]: https://www.pcgamesn.com/logitech-c920-hd-pro-webcam-review
[gossard]: https://arxiv.org/abs/2409.11760
[tt3d]: https://arxiv.org/abs/2504.10035
[tracknetv3]: https://github.com/qaz812345/TrackNetV3
[ttnet]: https://arxiv.org/abs/2004.09927
[deepmind]: https://arxiv.org/html/2309.03315
[deepmind2]: https://arxiv.org/html/2408.03906
[mpi]: https://arxiv.org/abs/1908.07332
[nakashima]: https://skoge.folk.ntnu.no/prost/proceedings/ifac2008/data/papers/3688.pdf
[omron]: https://www.omron.com/global/en/technology/information/forpheus/forpheus_technology.html
[reist]: https://spectrum.ieee.org/juggling-robots-get-fancier

**Verified by me:**
- UB0232: MJPG 120 fps at 1280x800, 210 fps at 640x400, 850 nm IR-pass filter, 100° diagonal FOV (read from the datasheet).
- C920: 30 fps cap (spec listings).
- Gossard (sound-based bounce detection, "millisecond-level" timing) and TT3D (bounce-anchored reprojection fit): from the arXiv abstracts.

**From a research subagent's reading, not rechecked:**
- B0332: 100-120 fps MJPG, visible-light.
- Gossard: 0.09-0.2 ms onset error, recall 0.95-1.0.
- TrackNetV3 numbers.
- DeepMind camera details.
- MPI, Nakashima and Omron details.
- TT3D: the 9-30 cm simulated error and the rejection of diameter-based depth.
