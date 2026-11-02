# Making the camera rig robust (SO-101 bounce)

2026-11-01. Rig: SO-101 with 12 V STS3215s at 50 Hz over USB from a MacBook (M3 Pro), paddle taped in the jaws,
overhead Logitech C920 at 640x480 @ 30 fps through OpenCV AVFoundation, white 40 mm ball. Each section covers what
people do (sources inline), what I checked on this Mac, and what to change here. "Verified" means I read the source code
or ran a command on this machine. "Unverified" means it is reported but I did not check it.

## Ranked recommendations

Ranked by impact over effort. "#" is the problem number from the brief.

| Rank | Change | # | Effort | Where |
|---|---|---|---|---|
| 1 | Lock C920 exposure, gain, white balance and focus with `uvc-util` when the camera opens, and read them back. Add a flicker-free lamp so exposure can drop to 3-4 ms. | 2, 3, 4 | 1 h + a lamp | [real/rig.py](../real/rig.py) `Camera.__init__` |
| 2 | Pre-flight check before arming: camera chosen by name in OpenCV's own order, a resting-ball reprojection check, control readback. Runtime watchdogs: frame rate and age, detection plausibility. A 12 V cut-off switch. | 1, 8 | 0.5 day | [real/rig.py](../real/rig.py), [real/bounce_run.py](../real/bounce_run.py) |
| 3 | Ball-roll levelling: let the ball roll on the paddle, fit its acceleration, correct the wrist offsets (resolution about 0.1 deg with a 1 s roll). Later, a rigid printed clamp instead of tape. | 6 | 0.5 day + a print | [real/level.py](../real/level.py) |
| 4 | More scene contrast (black rubber up, dark cloth, or the orange ball), plus geometric gating: a region of interest from the forward kinematics (FK) paddle position, a radius check, and a ballistic Kalman filter. | 3 | 0.5-1 day | `BallFinder` in [real/rig.py](../real/rig.py) |
| 5 | Checkerboard intrinsics with distortion (focus locked), more height spread in the calibration poses, `solvePnPGeneric` to list both planar solutions, an AprilTag on the base for a "camera moved" check. Later, the side camera for height. | 4 | 2 h (side camera: days) | [real/calibrate_camera.py](../real/calibrate_camera.py) |
| 6 | Repeat the latency measurement (5 runs, faster sway or a step) to get its spread, check row-dependent (rolling-shutter) delay, recentre the sim latency range on 62 ms. | 5 | 2 h | [real/calibrate_camera.py](../real/calibrate_camera.py), [bounce/env.py](../bounce/env.py) |
| 7 | Find out whether 2.3 rad/s is a firmware cap or a torque limit (one unloaded step test). Log voltage and current during launch strokes. | 7 | 2 h | [real/sysid.py](../real/sysid.py) |
| 8 | Start: arm only on a resting ball that passes the pre-flight check. Hand over from the scripted launch only if the first flight fits a ballistic arc. | 8 | 2 h | [real/bounce_run.py](../real/bounce_run.py) |

---

## 1. Camera indices change after replugging

**What people do**
- On Linux, people open cameras by stable paths (`/dev/v4l/by-id/...`); LeRobot users asked for the same
  ([lerobot#749](https://github.com/huggingface/lerobot/issues/749)). macOS has no such paths.
- LeRobot's docs only warn that indices "may change after rebooting your computer or re-plugging the camera" and say to
  rerun `lerobot-find-cameras opencv` ([LeRobot cameras](https://huggingface.co/docs/lerobot/cameras)).
- On macOS, people match the camera by name or `uniqueID` through AVFoundation:
  - [cv2-enumerate-cameras](https://github.com/lukehugh/cv2_enumerate_cameras) 1.4.0 returns name, `uniqueID` (as
    `path`) and VID/PID.
  - FFmpeg's avfoundation input accepts a device name, an index, `uid:<UID>` or `serial:<SERIAL>`
    ([ffmpeg-devices](https://ffmpeg.org/ffmpeg-devices.html)); `ffmpeg -f avfoundation -list_devices true -i ""` lists
    them. FFmpeg is not installed on this Mac.

**Verified here**
- **How OpenCV maps index to device.** Its macOS backend takes `devicesWithMediaType:Video` plus `Muxed`, sorts them by
  `uniqueID` string, and indexes into that sorted list
  ([cap_avfoundation_mac.mm](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_avfoundation_mac.mm),
  `startCaptureDevice`). An index is therefore a rank among whatever is connected right now.
- **The uniqueIDs on this Mac** (`system_profiler SPCameraDataType`):

  | Camera | uniqueID |
  |---|---|
  | C920 | `0x1121000046d08e5` = USB locationID `0x01121000` + VID `046d` + PID `08e5` |
  | Wrist camera | `0x11220001e458022` |
  | FaceTime | `1FD4B3A2-...` |
  | iPhone | `435E48BA-...` |

  - USB uniqueIDs start with `0x` and encode the port, so they change when the camera moves to another port.
    [Apple forum](https://developer.apple.com/forums/thread/836715) users also report swaps after a reboot.
  - When the C920 drops off the bus, every later device moves up one place. That is the failure seen: index 0 became
    another camera.
  - An iPhone or virtual camera whose UUID starts with `0` would sort ahead of every USB camera.
- **The new `rig.camera_index()`** looks the name up in the unsorted `devicesWithMediaType_(Video)` list.
  - On this Mac today, that order happens to match OpenCV's sorted order.
  - OpenCV sorts explicitly and also includes Muxed devices, so mirror it exactly:

```python
devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + \
       list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
devs.sort(key=lambda d: str(d.uniqueID()))          # OpenCV's order (cap_avfoundation_mac.mm)
names = [str(d.localizedName()) for d in devs]
```

  - cv2-enumerate-cameras sorts the same way but enumerates through a `DiscoverySession` with a fixed list of device
    types. A device type outside that list (Desk View, some virtual cameras) could shift its indices against OpenCV's.
    I have not seen this happen.

**Recommendation**
1. Keep name matching, with the sort above.
   - If two cameras ever share a name, match the uniqueID too (it is stable as long as the cable stays in the same
     port).
   - Refuse to run if the name matches more than one device.
2. Validate the view after opening, before arming:
   - Frames are 640x480.
   - About 30 fps over 1 s.
   - The image is not black or frozen.
   - The resting-ball check in section 8.
3. Optional: turn off Continuity Camera on the iPhone, or keep it away from the Mac, so it never appears in the list.
   The exact iOS settings path is unverified.

## 2. Auto exposure, white balance and focus

**What people do**
- OpenCV's `CAP_PROP_EXPOSURE` and similar calls are reported to have no effect on macOS:
  - `set()` returns False or nothing changes ([answers.opencv.org](https://answers.opencv.org/question/233090/set-is-not-working-for-brightness-and-exposure-on-macbook),
    [forum.opencv.org](https://forum.opencv.org/t/unable-to-set-most-camera-parameters/11844)).
  - Instead, people send UVC control requests directly:
    - [uvc-util](https://github.com/jtfrey/uvc-util): command line, IOKit.
    - [uvcc](https://github.com/joelpurra/uvcc): Node; tested on the C920; its README says to keep a preview app open.
    - [CameraController](https://github.com/itaybre/CameraController): GUI; users report greyed-out controls on
      Ventura.
    - [Webcam Settings](https://apps.apple.com/app/id533696630): paid, mixed reviews.
- The usual recipe: let auto exposure and white balance settle under the final lighting, read the values, switch auto
  off and write those values back.
- Google's table-tennis rig used a fixed 4 ms exposure and noted that fluorescent lights flicker below its 125 Hz frame
  rate, so frame brightness alternated ([arXiv 2309.03315](https://arxiv.org/abs/2309.03315)).
- With auto-exposure priority on, webcams drop the frame rate in dim light
  ([PCGamer](https://www.pcgamer.com/does-lighting-affect-fps-on-webcams/)).

**Verified here**
- **OpenCV ignores these controls on macOS.** `CvCaptureCAM::setProperty` / `getProperty` in `cap_avfoundation_mac.mm`
  handle only width, height, FPS and format. Exposure, gain, white balance and focus are silently ignored.
- **uvc-util works.** It is installed at `~/.local/bin/uvc-util` and reads the C920 on this macOS.
- **The C920's current state (2026-11-01):**

  | Control | Now | Range / note |
  |---|---|---|
  | `auto-exposure-mode` | 8 (aperture priority = auto) | supported modes bitmap 9 = manual (1) + aperture priority (8) |
  | `exposure-time-abs` | 83 (8.3 ms) | 3-2047, units of 100 µs |
  | `gain` | 248 | 0-255; near maximum, so the scene is dim and noisy |
  | `auto-white-balance-temp` | true (2922 K) | `white-balance-temp` 2000-6500 |
  | `auto-focus` | true (`focus-abs` 30) | 0-250, step 5 |
  | `power-line-frequency` | 2 (60 Hz) | |
  | `auto-exposure-priority` | 0 | frame rate held; keep it at 0 |

- **Why it matters here:**
  - **Motion blur.** At 8.3 ms and the 1.4 m/s take-off speed, the ball moves about 12 mm per exposure. At this range
    (f 476 px, 0.47 m) that is about 12 px, against a 30 px ball diameter.
  - **Depth error.** Autofocus hunting changes the ball's blur edge, and therefore its apparent radius, which the depth
    estimate uses.
  - **Colour drift.** Auto white balance shifts the white ball's saturation as the arm moves through the view.

**Recommendation**
1. Light the paddle area with a bright DC (flicker-free) LED lamp.
   - Moving to 4 ms halves the light per frame, and gain is already near maximum.
   - With mains-powered LED or fluorescent light, use exposures that are multiples of 8.33 ms, or expect banding.
2. In `Camera.__init__`, after the first frame arrives, call uvc-util for each control and read every value back with
   `-o`; stop if any value differs.
   - Settings live in camera RAM: they are lost on replug or power cycle.
   - One anecdotal report says another app reset them on open
     ([MacRumors](https://forums.macrumors.com/threads/microsoft-lifecam-hd-way-overexposed.1242582)).
   - Hence apply after open, every run. uvc-util accepts `0`/`1`/`true`/`false` for booleans (from its `UVCType.m`).

```sh
U="uvc-util -V 0x046d:0x08e5"          # or -L 0x01121000 if two C920s are ever connected
$U -s auto-exposure-priority=0 -s auto-exposure-mode=1 -s exposure-time-abs=40 -s gain=<tuned> \
   -s auto-white-balance-temp=0 -s white-balance-temp=<value read after auto settled> \
   -s auto-focus=0 -s focus-abs=<value read with the paddle at the hold pose> -s power-line-frequency=2
```

3. Tune `gain` once so the ball's brightest pixels sit around 230-245 and do not clip. Store the values in
   [real/config.yaml](../real/config.yaml).
4. With exposure fixed, the adaptive white threshold in `BallFinder` (90th percentile + 12) may no longer be needed. Keep
   it until the logs show it never moves.

## 3. Ball detection

**What people do**
- **Engineer the scene first.** Hobby trackers put a white ball on black card, or difference each frame against a
  stored background.
  - [PingPongTracker](https://github.com/wjasper/PingPongTracker) warns that any moving object shows up in the
    difference. Here the arm moves.
- **Google's table-tennis rig** found colour thresholds, shape fitting and background subtraction brittle against
  leftover balls, robots and people ([arXiv 2309.03315](https://arxiv.org/abs/2309.03315)). They trained a 27 k-parameter
  fully convolutional detector:
  - two temporal convolutions for motion;
  - raw Bayer input;
  - one new frame per step, at 125 fps;
  - an episode starts only when a ball comes from a defined region.
- **TrackNet** regresses a heatmap from several consecutive frames.
  - [TrackNetV3](https://github.com/qaz812345/TrackNetV3) lists about 25 FPS on a GPU, with the hardware not stated. That
    is too heavy for a 30 fps CPU loop.
  - The useful idea is the multi-frame input.
- **YOLO-nano.** I found no reliable Core ML latency figure. ONNX Runtime's CoreML provider ran YOLOv11n at 256x256 in
  7-8 ms on an M1 ([nativeORT](https://cran.r-project.org/web/packages/nativeORT/vignettes/BenchmarkPerformance.html)).
  It outputs boxes, not a subpixel centre, and needs labelled real frames.
- **Tracking and gating.** A Kalman filter per ball, with Mahalanobis gating, is the standard way to reject false
  detections ([juggling, arXiv 2607.15129](https://arxiv.org/abs/2607.15129);
  [CMU improbability filtering](https://www.ri.cmu.edu/publications/improbability-filtering-for-rejecting-false-positives)).

**Measured here**
- Setup: M3 Pro, one thread, synthetic 640x480 frames. Medians, with p95 where it differs much.

| Method | Time per frame |
|---|---|
| HSV mask + open + contours (current `BallFinder`), full frame | 0.51 ms |
| Same, 120x360 region of interest | 0.09 ms |
| Absolute difference against a stored background + threshold | 0.28 ms |
| HoughCircles | 2.1 ms (p95 4.5) |
| MOG2 background subtractor | 4.6 ms (p95 9.7) |
| Repo `BallNet` ([bounce/detector.py](../bounce/detector.py)), torch CPU, 320x240 | 36 ms |
| Same, 160x120 | 9 ms |

- `BallNet` has 384 k parameters and takes 36 ms at 320x240. Its docstring says about 0.2 M parameters and "a few ms".
  As it stands it would add more than a frame of latency.
- ONNX/CoreML was not measured: the `onnx` package is not installed.
- It is also trained only on synthetic data.

**Recommendation** (in order)
1. **Contrast.**
   - Put the black rubber side up.
   - Use a dark matte cloth on the table.
   - Cover any white printed arm parts that sit under the ball's path with matte black tape, or use the orange ball,
     whose hue separates it from white PLA.
   - This, plus the locked exposure from section 2, makes fixed thresholds hold.
2. **Geometric priors**, all cheap because the code already has the camera model and FK:
   - Search only a region around the Kalman prediction, or around the projected paddle centre when there is no track.
   - Reject blobs whose radius disagrees by more than about 20 % with the radius expected at the depth predicted from
     the track or paddle height.
   - Mask pixels covered by the projected arm and paddle rim.
3. **A ballistic Kalman filter** on the 3D detections:
   - Constant gravity, with a reset at each paddle contact.
   - Chi-square gate on the innovation (3 dof, 99 %: 11.3).
   - Drop several consecutive rejects to "lost".
   - The sim already trains with outliers (`outlier_p`, `hand_outlier_p` in [bounce/env.py](../bounce/env.py)). The
     filter keeps real outliers within that range.
4. **Motion cue.** AND the colour mask with a frame difference (0.3 ms). "White, round and moving, or resting on the
   paddle" excludes static white things.
5. **A learned detector only if 1-4 fail.** Run `BallNet` on a 96-128 px crop around the prediction, and fine-tune on
   real frames first.

## 4. Extrinsic calibration and depth

**What people do**
- **The mirrored pose is the planar PnP ambiguity.** For near-coplanar points, two poses fit equally well (a reflection
  about the plane).
  - The IPPE author: no PnP algorithm can resolve it when the two errors are close
    ([IPPE](https://www.github.com/tobycollins/IPPE)).
  - The fixes are non-coplanar points or a physical constraint.
  - OpenCV's `solvePnPGeneric` returns every candidate for IPPE, SQPNP and P3P (OpenCV
    [solvePnP docs source](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/doc/solvePnP.markdown)). It is
    present in the project's OpenCV 4.13 (verified).
- **Intrinsics.** People calibrate first with a checkerboard or ChArUco board (`cv2.calibrateCamera`, 15-25 views), so
  the extrinsic solve has 6 unknowns, not 7. Focus must be locked: intrinsics change with focus.
- **Fiducials.**
  - Single-tag accuracy is modest:
    - about 10 mm at 50 cm with 8 cm tags on a small camera
      ([OpenMV forum](https://forums.openmv.io/t/accuracy-of-april-tags-position/6572));
    - yaw is the main error source ([Abbas et al., Sensors 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960891/)).
  - Boards with many corners do better.
- **Hand-eye.** For a camera fixed off the arm, people use `cv2.calibrateHandEye` (gripper poses inverted) or
  `cv2.calibrateRobotWorldHandEye`.
  - An SO-101 user found it very sensitive to pose diversity, and the arm's few joints limit the rotations available
    ([OpenCV forum](https://forum.opencv.org/t/eye-to-hand-calibration-using-lerobot/24195)).
- **A second camera.**
  - Google triangulated two synchronised cameras placed on opposite sides of the table, which cut triangulation bias by
    about 10x. They refined with a Kalman filter, and a 4 cm observation bias cut reward by 36 %
    ([arXiv 2309.03315](https://arxiv.org/abs/2309.03315)).
  - Unsynchronised USB cameras are fused by timestamping each frame and running a Kalman filter that accepts
    asynchronous measurements
    ([Pustka et al.](https://campar.in.tum.de/pub/pustka2012ismar/pustka2012ismar.pdf)).
- **Depth cameras.** RealSense has had no official macOS support since High Sierra and is troublesome on Apple Silicon
  ([RealSense community](https://support.realsenseai.com/hc/en-us/community/posts/4548413451539-Activating-the-realsense-D435-Depth-Camera-in-MacOs)).
  I would not use one here.

**Verified here** (from [real/camera_calib.json](../real/camera_calib.json))
- 14 poses, 1.3 px reprojection, 1.2 cm rms depth-from-size error with the ball static, focal length 476 px.
- No distortion model: K comes from the nominal field of view plus a focal-length search.
- **The points are close to planar.** Their heights span only 0.21-0.30 m, so the mirrored solution is close in error.
  That explains the flip.
- **Depth from size is weak by geometry.** The ball radius is about 15 px.
  - Since dz/z = dr/r, a 0.3 px radius error is 2 %, about 1 cm at 0.47 m.
  - Blur and shading bias r during flight.
  - For an overhead camera, depth along the optical axis is the ball height, which is the axis bouncing needs most.

**Recommendation**
1. **Intrinsics now.**
   - Lock focus (section 2).
   - Calibrate with a printed checkerboard or ChArUco board (OpenCV's ChArUco calibration tutorial, `cv2.aruco` is in
     4.13).
   - Save K and distortion, and undistort detections before `ball_position`.
   - Remove the focal-length search from `fit()`.
2. **Better extrinsics.**
   - Spread the calibration poses in height (±8 cm, not ±5).
   - Run `solvePnPGeneric(..., SOLVEPNP_IPPE)` on the points to list both solutions, keep the one with the camera above
     the table, and log the error ratio between them. A ratio near 1 means the data cannot tell them apart.
3. **A "camera moved" check.**
   - Tape an AprilTag (tag36h11, about 6 cm) to the table beside the base. At calibration, store its corner pixels.
   - At startup, detect it with `cv2.aruco` and refuse to arm if the corners moved more than about 3 px.
   - This catches a bumped camera, which neither the name check nor the index can.
4. **The side camera, for height** (the planned OV9281).
   - The [Arducam B0332](https://core-electronics.com.au/120fps-global-shutter-usb-camera.html) is a UVC global-shutter
     camera: 120 fps only in MJPG; uncompressed YUY2 is 10 fps.
   - From the side, height is an image coordinate, about 1 mm per px at 0.5 m, not a size ratio.
   - Fuse it with the overhead camera through the Kalman filter in section 3, each camera with its own measured
     latency.
   - This is the largest accuracy gain available, and also the most work.

## 5. Camera latency

**What people do**
- **LED or clock in frame.** Blink an LED, or show a fast clock, in view, and time when it appears.
  - [LigerBots](https://ligerbots.org/docs/whitepapers/LigerBots_Camera_Latency_Whitepaper.pdf) used two photoresistors
    to get distributions. Logitech C930e through OpenCV at 30 fps: median 112 ms (97-129). PS Eye: 78 ms.
  - [makehardware](https://makehardware.com/2016/03/29/finding-a-low-latency-webcam/): ELP 2MP at 30 fps about 105 ms;
    PS3 Eye about 75 ms.
  - Both measure glass to glass, so the numbers include display latency.
- **No controlled C920 measurement found.** OBS users estimate 33-66 ms
  ([OBS forum](https://obsproject.com/forum/threads/webcam-latency-question-logitech-c910-c920.10517/)).
- **Timestamp instrumentation.** Google measured ball observation latency at 40 ms (σ 8.2) with 125 fps machine-vision
  cameras ([arXiv 2309.03315](https://arxiv.org/abs/2309.03315)).
  - Policies trained on the measured latency transferred best. 0 %, 20 % or 150 % of it transferred poorly.
  - i-Sim2Real failed completely without simulated sensor latency
    ([arXiv 2207.06572](https://arxiv.org/abs/2207.06572)).
- **Motion correlation.** Kalibr estimates the time offset between a camera and an IMU by fitting continuous
  trajectories ([kalibr](https://github.com/ethz-asl/kalibr)). The paddle sway uses the same idea.

**Assessment of the current method** (`latency()` in [real/calibrate_camera.py](../real/calibrate_camera.py))
- It measures camera delay relative to the servo position reads, on one clock. That is the quantity the policy needs.
- 62 ms is about 1.9 frame periods, consistent with the figures above once display latency is removed.
- Weaknesses:
  - **Low sensitivity.** One 5 s run at 1 Hz and ±3 cm: a 10 ms error moves the predicted ball only about 2 mm, about
    2 px.
  - **No spread estimate.**
  - **Rolling shutter.** The C920 is a rolling-shutter webcam, very likely, but its readout time is not published. Rows
    are captured at different times, so the delay depends on the ball's image row.

**Recommendation**
1. Repeat the sway 5 times at 2-3 Hz (still small, at a level pose) and record mean and spread. Or use a step:
   command a wrist-roll step and compare the first encoder change with the first frame that moves.
2. Run the fit once with the paddle near the top of the image and once near the bottom.
   - If the two differ by more than about 5 ms, add a per-row correction:
     `t = t_frame + (v/H - 0.5) * T_readout`, with `T_readout` taken from the difference.
3. Cross-check without extra hardware: point the C920 at the Mac screen and flip a full-screen window black to white at
   logged `perf_counter` times. The first bright frame gives an upper bound, including one display refresh.
4. Log frame inter-arrival times on every run. Bursty delivery means the latency varies per frame.
5. In the sim, `cam_frames` is (1.5, 1.0, 2.0) frame periods: 50 ms nominal, 33-67 ms range
   ([bounce/env.py](../bounce/env.py)). 62 ms sits at the top edge. Recentre the nominal on the measured value and set
   `age_bias` from the measured spread.

## 6. Paddle mounting and level

**What people do**
- I found no paddle-specific procedure.
  - Bouncing and juggling builds close the loop on the ball: tilting-platform bouncers adjust tilt from the ball
    position ([Hackster](https://www.hackster.io/news/this-teensy-controlled-robot-can-bounce-a-ball-endlessly-in-advanced-patterns-e6560ebd1959)).
  - General practice is a rigid, keyed mount with a datum face.
- **Inclinometers** are checked by reversal: read, turn 180 degrees, read again, and average to cancel the sensor's zero
  offset ([Jewell Instruments](https://jewellinstruments.com/wp-content/uploads/Procedure-For-Leveling.pdf)).
  - A DIY user found a digital inclinometer 0.45 deg off when flipped
    ([diynot](https://www.diynot.com/diy/threads/digital-inclinometers-how-accurate-is-yours.599673/post-5340059)).
  - Phone apps have no published accuracy.

**Physics (why the ball is the better level)**
- A thin-shelled ball rolling without slipping on a tilt θ accelerates at (3/5) g sin θ
  ([derivation](https://www.physicsforums.com/threads/what-is-the-acceleration-of-the-center-of-mass-of-the-rolling-hollow-sphere.147053/)).
  - That is 0.10 m/s² per degree: 1.3 cm in 0.5 s, 5 cm in 1 s. This matches the calibration script's note that a free
    ball rolls off at 1 deg within a second.
  - **Resolution (my estimate).** A quadratic fit gives these tilt resolutions at 1.5 mm noise per frame:

    | Track length | Frames | Tilt resolution |
    |---|---|---|
    | 0.5 s | 15 | about 0.4 deg |
    | 1 s | 30 | about 0.07 deg |

    The noise on a slowly rolling ball is likely well under 1.5 mm, which would improve both.
  - Rubber rolling resistance sets a floor, not measured.
- **The effect on bouncing (estimate).** A 1 deg tilt sends the ball off with about (1 + e) · v · sin θ ≈ 0.04 m/s of
  sideways speed at 1.4 m/s. Over a 0.29 s flight, that is about 1 cm of drift per bounce, which compounds unless the
  policy corrects.

**Recommendation**
1. **Ball-roll levelling**, as a mode of [real/level.py](../real/level.py):
   - Glide to the hold pose. Place the free ball at the centre. Track up to 1 s, or until the ball is 5 cm from the
     centre.
   - Fit a quadratic to x(t) and y(t) to get the acceleration a. Tilt = asin(|a| / 0.6 g), direction from a.
   - Map the tilt to wrist flex and roll corrections with the paddle Jacobian (the Newton step already in
     `fit_wrist_offsets`). Repeat 2-3 times until the tilt is under 0.2 deg.
   - This levels against gravity, so it also absorbs table tilt and joint sag at the hold pose, which is what bouncing
     needs.
2. **A rigid mount.** Replace the tape with a printed clamp that bolts to the fixed jaw, or replaces the gripper, with a
   slot keyed to the handle and a flat datum parallel to the blade.
   - Tape is likely to creep under repeated impacts. This is unverified; compare the levelling result before and after a
     session.
3. **If you use the phone level**, use the reversal method: two readings 180 deg apart, averaged.
4. **Keep the sim's tilt randomisation** (`table_tilt` ±1.5 deg, `mount_*`).
   - Check the run logs for steady sideways drift between bounces as a sign of residual tilt.

## 7. Servo limits and identification

**What people do**
- **Specs.** STS3215 12 V (C018): 0.222 s/60° no-load (4.7 rad/s, 45 rpm), 30 kg·cm stall, 2.7 A stall at 12 V
  ([RobotShop](https://robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215)).
- **Rhoban BAM** ([docs](https://bam.readthedocs.io/en/latest/usage/actuators.html), [repo](https://github.com/Rhoban/bam)):
  - It identifies servos on a pendulum bench under varied loads and P gains.
  - For the STS3215 it has only the 7.4 V m1 model.
  - Users identify their own units and retrain.
  - [bounce/servo.py](../bounce/servo.py) already follows BAM, and [real/sysid.py](../real/sysid.py) /
    `scripts/fit_servo.py` fit the real arm.
- **Protection.** The translated datasheet says:
  - Overload trips after more than 80 % of locked torque for about 2 s.
  - Overcurrent trips after more than 2 A for 2 s, and then the output turns off
    ([datasheet](https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf)).
  - Short strokes will not trip it. A stalled joint will, and torque then drops mid-run.
- **Gear variants.** The SO-101 leader uses 7.4 V parts at 1:345 (C001), 1:191 (C044) and 1:147 (C046)
  ([Seeed wiki](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_4/)).
  - A lower ratio is faster with less torque, but these are 7.4 V parts and would need their own supply.

**Verified here**
- The refit set `12v_r8` puts the speed limit at 2.3 rad/s (range 1.8-4.0), against 4.7 rad/s no-load.
- Present_Voltage is 11.5-11.8 V (comments in [bounce/servo.py](../bounce/servo.py)).
- The fit cannot tell a firmware speed cap from a torque-speed limit under this load.

**Recommendation**
1. **Settle cap versus torque with one test.** Run the same 20-30 deg fast step on wrist roll, which carries almost no
   gravity load, and on shoulder lift.
   - The same peak speed on both means a firmware or register cap. Read `Goal_Velocity` (46) and `Torque_Limit` (48) in
     `arm.registers()` to see whether one is set.
   - A lower peak on the loaded joint means a torque-speed line: model speed falling with load rather than a fixed cap.
2. **Log during real launch strokes:**
   - `Present_Voltage`, `Present_Current` and `Present_Load`.
   - Five servos at stall can draw about 13.5 A, so sag shows up there first. Check that the 12 V supply's rating covers
     it.
3. **Validate by replay.** Feed the logged goal sequences from `real/logs/run-*.npz` (launch amplitudes included) into
   the sim servo model and compare the joint traces. The small 3-8 deg sysid motions may not reach the speed limit.

## 8. Starting dynamic tasks, and safety

**What people do**
- **Gated starts.** Google starts an episode only when a ball is detected moving toward the robot from a defined region.
  A referee process filters spurious detections ([arXiv 2309.03315](https://arxiv.org/abs/2309.03315)).
- **Stopping an SO-101.** An SO-101 safety guide notes the arm has no hardware e-stop
  ([roboticscenter](https://roboticscenter.ai/hardware/so-101/safety)):
  - Ctrl-C in LeRobot disables torque.
  - Unplugging USB leaves the servos holding their last goal.
  - Only cutting 12 V de-energises the arm.
- **Per-step limits.** LeRobot limits per-step joint motion with `max_relative_target`
  ([lerobot#1837](https://github.com/huggingface/lerobot/pull/1837)).
- **Starting from rest** is covered in [launch_from_rest.md](launch_from_rest.md). The scripted dip-and-flick is the
  plan.

**Recommendation**
1. **Pre-flight check before arming** (refuse to arm on any failure):
   - The camera name is found in OpenCV's sorted order.
   - Frames are 640x480 at 30 fps or more.
   - The uvc-util readback matches.
   - The base AprilTag is where it was at calibration.
   - With the ball resting at the paddle centre at the hold pose, the detection is within about 8 px of the pixel
     predicted by FK and the camera model, and the radius is within ±15 %.
   - The last check alone catches the wrong camera, a moved camera, a stale calibration and a bad colour config. The
     code for it already exists in `CameraModel` and the FK.
2. **Runtime watchdogs.** Go back to holding when any of these fails:
   - The frame counter stalls more than 100 ms, or frame age exceeds about 120 ms (`cam_lat_max`).
   - The Kalman gate rejects several frames in a row.
   - A detection is outside the paddle cylinder (`START_LATERAL`) or the expected height range.
   - The implied speed exceeds 5 m/s.
   - `bounce_run.py` already covers lost, static and temperature cases.
3. **A per-step slew limit** on goals, like `max_relative_target`, sized from the largest real policy and launch steps in
   the logs. It goes on top of the existing joint clip.
4. **Hand-over check.** After the scripted launch, hand over to the policy only if the first 3-4 detections fit a
   ballistic arc with a 5-15 cm apex and less than 3 cm sideways. Otherwise glide back to holding.
5. **A switch on the 12 V line**, within reach. It is the only stop that does not depend on the Mac.
