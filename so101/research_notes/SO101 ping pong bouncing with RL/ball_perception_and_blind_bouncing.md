# Ball perception and blind (sensing-free) bouncing for a ping-pong ball on an SO-101 paddle

Scope: how to sense a 40 mm, 2.7 g ping-pong ball bouncing 5–30 cm above a paddle held by an SO-101 (LeRobot) well enough to close a loop at 30–60 Hz on a MacBook; when sensing can be avoided; camera placement; what a flat store-bought paddle changes; detecting the user's hand in the workspace.

Conventions used below:
- **Measured (this session)** = I ran it on the user's Mac (Apple M3 Pro, OpenCV 4.13.0, single thread, synthetic 640×480-class frames; script at /private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/bench.py).
- **Computed** = closed-form physics/geometry I worked out (script at /private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-roboticsexp/59747eac-fb48-4bb0-ae64-62061a139448/scratchpad/scripts/calc.py). These are listed under Inferences, not Cited Findings.
- "via search summary" = the number came from a search-engine digest of the source; I did not open the full text myself.

---

## 1. Is 30 fps enough? Frames per flight, motion blur, webcam latency, resulting accuracy, and what table-tennis robots use

### Takeaway
30 fps gives only 6–12 frames per flight for 5–20 cm bounces, and with the C920's auto exposure each frame smears the ball 3–7 cm at impact speed. End-to-end latency is probably 60–110 ms. That is a plausible starting point if the policy is trained with matching latency and noise, but marginal for timing impacts. 60–120 fps with a short global-shutter exposure is the clear upgrade. DeepMind's table-tennis system runs at 125 fps with 40 ms ball latency, and its performance fell off sharply at 50 fps, though its ball speeds are much higher than ours.

### Cited Findings
- **DeepMind table-tennis perception (primary source):** two hardware-synchronized Ximea MQ013CG-ON global-shutter cameras, up to 125 fps at 1280×1024, 4 ms exposure, 838 µs camera latency, mounted ~2 m above the table on opposite sides. The ball detector is a ~27k-parameter CNN on raw Bayer crops of 512×1024 (skipping Bayer→RGB saves ~1 ms per camera, "15% of the inter-frame time"), followed by DLT triangulation and a Kalman filter — [Robotic Table Tennis: A Case Study into a High Speed Learning System, arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- **DeepMind measured latency:** ball observation latency mean 40 ms (σ 8.2 ms); ABB robot observation 29 ms; ABB action latency 71 ms (σ 5.7). Perception runs at 125 Hz; the policy was explored at 20–100 Hz, 100 Hz for most experiments — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- **DeepMind sensitivity sweep:** performance was stable until a "heavy drop at 50 FPS", where the robot swung too late. Latency was tolerated up to a heavy drop at 150 ms. Zero-mean noise of ±4 cm (default) and even ±8 cm had only a minor effect, but a 4 cm bias reduced reward 36%. Non-zero-mean noise of [-1, 7] cm hurt about as much as doubling zero-mean noise — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- **DeepMind sim latency matters for transfer:** policies trained at 50% of the real latency transferred worse (1.33 vs 1.83 at baseline). 0%, 20% and 150% latency in sim performed poorly, and for catching, low simulated latencies transferred at 0–10% catch rate — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- 2024 coverage of DeepMind's amateur-level table-tennis robot attributed its losses largely to latency. It struggled with very fast balls, balls above its field of view (more than six feet above the table), and very low balls — [MIT Technology Review, Aug 2024](https://technologyreview.com/2024/08/09/1096102/google-deepmind-trained-a-robot-to-beat-humans-at-table-tennis)
- **Unverified (2026):** an article claims a newer DeepMind robot ("Ace") uses event-based vision sensors because standard cameras add too much latency. Not confirmed against a primary source — [startupfortune.com (secondary, unverified)](https://startupfortune.com/?p=8236)
- **C920 spec:** 78° diagonal FOV; max 30 fps at both 1080p and 720p — [Logitech C920 product page](https://logitech.com/en-za/shop/p/c920-pro-hd-webcam)
- **C920 latency is software/format dependent (anecdotal, no controlled measurement found):**
  - OBS users report 1–2 frames (33–66 ms) of extra webcam delay with "multithreaded optimization", and ~1 s delays in some OBS versions — [OBS forum: Webcam latency C910 & C920](https://obsproject.com/forum/threads/webcam-latency-question-logitech-c910-c920.10517/)
  - A vMix forum post warns that the C920's H.264 stream "will add considerable latency" — [vMix forum](https://forums.vmix.com/posts/m65054-Video-lag-with-Logitech-c920-camera)
- **Generic USB webcam latency measurements:**
  - An OpenCV clock-filming script measured one example at 108 ms glass-to-glass (camera unnamed; includes display latency) and warns "the latency can jump around quite a bit" — [perrytsao/Webcam-Latency-Measurement](https://github.com/perrytsao/Webcam-Latency-Measurement)
  - A Jetson developer reported ~58 ms average glass-to-glass at 120 fps MJPEG with GStreamer/V4L2, and another forum user saw 30–50 ms just to get a frame through OpenCV's capture classes — via search summary of [NVIDIA forum thread A](https://forums.developer.nvidia.com/t/minimizing-video-latency-between-camera-and-program-memory/245890) and [thread B](https://forums.developer.nvidia.com/t/proper-way-of-fast-access-to-external-camera-from-a-c-opencv-app/40721) (which number belongs to which thread was not verified)
- **The user's own repo:** on macOS the C920's manual exposure does not hold once a stream is open (locked at brightness 97, the recorded image came out at 203–217), so exposure is left on auto. Both cameras currently run 640×480 (wrist 480×640 rotated) at 30 fps and measured 30.5 fps — [camera_presets.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_presets.py), [env.sh](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/env.sh), [presets/latest.json](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/presets/latest.json)

### Inferences
- **Computed flight geometry** (apex h above the paddle; T = 2√(2h/g); impact speed v = √(2gh)):

  | Apex h | Flight T | Impact v | Frames @30 | @60 | @120 | @187 |
  |---|---|---|---|---|---|---|
  | 5 cm | 202 ms | 0.99 m/s | 6.1 | 12.1 | 24.2 | 37.8 |
  | 10 cm | 286 ms | 1.40 m/s | 8.6 | 17.1 | 34.3 | 53.4 |
  | 20 cm | 404 ms | 1.98 m/s | 12.1 | 24.2 | 48.5 | 75.5 |
  | 30 cm | 495 ms | 2.43 m/s | 14.8 | 29.7 | 59.4 | 92.5 |

- **Computed motion blur at impact speed** (blur = v × exposure):

  | Impact v | 33 ms exp | 16.7 ms | 8 ms | 4 ms | 2 ms | 1 ms |
  |---|---|---|---|---|---|---|
  | 1.0 m/s (5 cm apex) | 33 mm | 17 mm | 8 mm | 4 mm | 2 mm | 1 mm |
  | 1.4 m/s (10 cm) | 47 mm | 23 mm | 11 mm | 6 mm | 3 mm | 1.4 mm |
  | 2.0 m/s (20 cm) | 66 mm | 33 mm | 16 mm | 8 mm | 4 mm | 2 mm |
  | 2.4 m/s (30 cm) | 81 mm | 41 mm | 19 mm | 10 mm | 5 mm | 2.4 mm |

  A 40 mm ball smeared over 33–66 mm becomes an elongated, desaturated streak. Its centroid is still usable (the mid-exposure position), but diameter-based depth breaks, because the streak's long axis is no longer the diameter. Indoor auto exposure on a webcam is often 1/30–1/60 s, which is the left two columns. The macOS exposure-lock problem in the user's own repo means short exposure is not available on the C920 today.
- **Latency budget, 30 fps C920 on macOS:** the pipeline is mid-exposure (≈8–17 ms) + rolling readout and USB transfer (≈1 frame, 33 ms) + driver/AVFoundation/OpenCV buffering (0–2 frames, 0–66 ms) + MJPEG decode (0.6 ms measured) + HSV detection (1.2 ms measured) + sample-and-hold in a 30 Hz control loop (avg 17 ms). That totals roughly **60–130 ms** from photons to an action decision. The command, servo and motion delays come on top. The 108 ms and 58 ms anchors above are consistent with this, but **the C920 on this Mac has not been measured**.
- **Distance the ball moves during the latency:** at 1.4 m/s, 80 ms ≈ 11 cm, and at 2 m/s ≈ 16 cm. Latency must be compensated by forward prediction (ballistic propagation) or learned in sim with matched latency (DeepMind evidence above).
- **Accuracy implication:** apex error ≈ v·σ_v/g and impact-time error ≈ σ_v/g. With σ_v = 0.1 m/s, apex σ is 1.0–2.0 cm and impact-time σ is ~10 ms. With σ_v = 0.27 m/s (what a diameter-based height estimate at 30 fps gives, see §4), apex σ is 2.7–5.4 cm and impact-time σ ~28 ms. A 10 cm bounce has a 286 ms period, so a 28 ms timing error is ~10% of the cycle.
- **Comparison to DeepMind's 50 fps cliff:** the cliff happened with table-tennis rally speeds and a policy that must hit a precise swing window. The relevant quantity is ball displacement per frame relative to the tolerance. At our 1–2.4 m/s, 30 fps means 3–8 cm per frame near impact. That is similar to a ball diameter, so 30 fps is "coarse but not hopeless" for a paddle that only needs to be under the ball with roughly the right velocity. 60–120 fps would cut this to 1–4 cm.
- **Bottom line:** 30 fps can plausibly support low (5–15 cm) bounces if (a) exposure is short enough to avoid streaks, (b) height comes from a good viewing geometry (side view, not diameter), and (c) the sim reproduces the latency. Otherwise move to 60–120 fps global shutter.

### Gaps
- No published, controlled glass-to-glass or photon-to-app latency measurement for the C920 on macOS (AVFoundation/OpenCV) was found. It must be measured, e.g. with an LED blink triggered over the same serial link and timestamped in-frame, or a clock-filming test.
- Whether the C920 at 640×480 is a 4:3 crop or a scaled full sensor (this affects the focal length used in §4) is not verified. Calibrate intrinsics with a checkerboard.
- The C920's actual auto-exposure time under the user's lighting is unknown. It can be read via uvc-util while streaming.

---

## 2. Cheap high-frame-rate camera options (models, prices, fps, shutter, macOS support, latency)

### Takeaway
The best fit for a MacBook is a UVC global-shutter USB board, such as the Arducam OV9281 (mono, ~$60, 120 fps at 1280×800) or its colour OV9782 sibling (~AU$88). These work with OpenCV/AVFoundation without drivers. The ELP OV4689 reaches 260 fps cheaply, but it is rolling shutter and only at 640×360. The PS3 Eye (187 fps at 320×240, <$10 used) needs a non-UVC libusb driver on macOS. The Raspberry Pi GS camera needs a Pi. RealSense depth on a fast 40 mm ball is a poor fit. Event cameras are quote-only research kits.

### Cited Findings
- **Arducam B0332 (OV9281, mono, global shutter, UVC):** 1280×800, MJPG/YUY2, 70° low-distortion M12 lens, USB 2.0, no drivers needed. Advertised 120 fps; one Amazon-derived listing says 100 fps. Prices €59.90 incl. VAT, £46.15, US ~$60–63 incl. tariffs — [welectron](https://welectron.com/Arducam-B0332-120fps-Global-Shutter-USB-Camera-Board), [RobotShop UK](https://uk.robotshop.com/products/arducam-120fps-global-shutter-usb-camera-board-1mp-ov9281-uvc-m12-lens), [Core Electronics](https://core-electronics.com.au/120fps-global-shutter-usb-camera.html)
- **Arducam B0385 (OV9782, colour, global shutter, UVC):** ~AU$88 excl. GST — [Core Electronics](https://core-electronics.com.au/120fps-global-shutter-color-usb-camera-board.html)
- **ELP OV4689 "260 fps" USB camera:** 1920×1080 MJPEG@60, 1280×720@120, 640×360@260 fps. Rolling shutter per listing. ~$65–119 depending on seller/lens — [SVPRO/ELP listing](https://www.svpro.cc/product/elp-260fps-120fps-60fps-usb2-0-web-camera-driver-free-ov4689-cmos-sensor-1080p-hd-mini-high-frame-rate-module-camera-260fps-with-6mm-lens/), [Newegg](https://www.newegg.com/p/3C6-019J-02MT8)
- **PS3 Eye:**
  - 320×240 at 187 fps is the fastest valid QVGA mode; 205 fps gives partly corrupt video — [Linux gspca high-speed-mode patch](https://lkml.iu.edu/hypermail/linux/kernel/1412.3/01236.html)
  - On macOS it works via the PS3EYEDriver (libusb), with 320×240@187 Hz listed as supported — [PS Move API camera docs](https://psmoveapi.readthedocs.io/en/latest/camera.html)
  - A macOS Mojave OpenCV demo found 290 and 205 Hz corrupted and 187 Hz working — [PS3Eye-OpenCV-Demo README (mirror)](https://github.timelimit.workers.dev/ThomasDebrunner/PS3Eye-OpenCV-Demo/blob/master/README.md)
  - Commonly sold used for "less than $10" — [Processing forum](https://forum.processing.org/two/discussion/21366/)
- **Raspberry Pi Global Shutter Camera (Sony IMX296, 1.6 MP, 3.4 µm pixels):** 1456×1088 at 60 fps (60.3 fps all-pixel, 10-bit). $63.75 at SparkFun. A CSI camera, so it needs a Raspberry Pi, not a Mac — [SparkFun](https://www.sparkfun.com/raspberry-pi-global-shutter-camera.html), [Arducam IMX296 docs](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Global-Shutter/1.58MP-IMX296/), [MaxMax spec](https://maxmax.com/maincamerapage/monochrome-cameras/raspeberry-pi-cameras/raspberry-pi-global-shutter-1-6mp)
- **Intel RealSense D435:**
  - Depth up to 90 fps (up to 1280×720), global-shutter depth imagers, RGB rolling shutter at 30 fps — [xMachines spec](https://www.xmachines.ai/cameras/intel-realsense-d435)
  - Minimum depth ~0.28 m at max resolution per RS Online, with other sources claiming 0.105 m (conflict) — [RS Online D435f](https://sg.rs-online.com/web/p/depth-cameras/2650901)
- **Luxonis OAK-D Lite:**
  - Mono OV7251 global shutter, 99 fps at 640×480 and 117 fps at 640×400 per Luxonis docs; retailers claim 120/200 fps (conflict) — [Luxonis OV7251 docs](https://docs.luxonis.com/projects/hardware/en/latest/pages/articles/sensors/ov7251/)
  - Launched at $79 (crowdfunding, 2021); SparkFun $123.32 and marked retired — [Hackster](https://www.hackster.io/news/opencv-and-luxonis-launch-smaller-lighter-cheaper-oak-d-lite-all-in-one-computer-vision-gadget-fb5da778dfbd), [SparkFun](https://www.sparkfun.com/luxonis-oak-d-lite.html)
- **OV9282 (OAK-D / OAK-D Pro family):** 1280×800@60, 640×400@120 fps — [Arducam B0413 for OAK](https://welectron.com/Arducam-B0413-OV9282-Monochrome-Global-Shutter-Camera-Module-for-DepthAI-OAK)
- **Event cameras:** the Prophesee EVK4 HD is quote-only (no list price) — [Prophesee EVK4 HD brief 2026](https://www.prophesee.ai/wp-content/uploads/2026/03/EVK4-HD-Prophesee-Evaluation-Kit-Brief-2026.pdf). iniVation DVXplorer Micro is listed as a USB research device, with no price found — [RidgeRun event camera hardware overview](https://developer.ridgerun.com/wiki/index.php/Event_Cameras_Developer_Guide/Hardware_Overview)
- **Hobby precedent at 120 fps:** the Octo-Bouncer (a ping-pong ball on a flat tilting acrylic plate, bouncing "for hours") used an e-con See3CAM_CU135 at "120 FPS 640×480" with OpenCV Hough circles. The author "didn't notice a problematic delay" at 120 fps — [Electron Dust: The Octo-Bouncer](https://electrondust.com/2020/03/01/the-octo-bouncer/)
- **Hobby precedent at 90 fps:** Ping-Pong-Pi argued "we need 90 fps real time image processing" (Pi NoIR camera, green LED illumination) and juggled ~2 min before a misjudgment — [Hackaday.io Ping-Pong-Pi](https://hackaday.io/project/20402-ping-pong-pi/details)

### Inferences
| Option | fps @ res | Shutter | Mac path | Approx price | Fit |
|---|---|---|---|---|---|
| Logitech C920 (have) | 30 @ ≤1080p | rolling | UVC/AVFoundation | (have) | baseline; exposure not lockable on macOS (user's repo) |
| Arducam B0332 OV9281 mono | 120 @ 1280×800 (claimed) | global | UVC | ~$60 | **best value**; mono → use brightness contrast (black backdrop + white/orange ball) |
| Arducam B0385 OV9782 colour | 120 (claimed) | global | UVC | ~AU$88 | colour HSV tracking + hand detection |
| ELP OV4689 | 260 @ 640×360, 120 @ 720p | rolling | UVC | $65–119 | fast but rolling-shutter skew on a falling ball |
| PS3 Eye | 187 @ 320×240 | (unverified) | libusb PS3EYEDriver, not AVFoundation | <$10 used | cheapest high-fps; awkward driver; low resolution |
| RPi GS (IMX296) | 60 @ 1456×1088 | global | needs a Pi | ~$64 + Pi | not Mac-native |
| RealSense D435 | depth 90 | global (depth) | librealsense | not fetched | 40 mm fast ball near min-Z → poor depth; IR mono stream could be used as a GS camera |
| OAK-D Lite | ~99–117 mono | global | DepthAI (USB) | ~$80–150 | on-device detection possible; more software |
| Event camera | µs-scale | — | vendor SDK | quote-only | overkill / research |

- Global shutter matters. A falling ball under a rolling shutter is sheared by the row-readout time. At ~30 ms readout over 480 rows, the top and bottom rows of a 30-row ball blob are ~2 ms apart (computed), which is ≈4 mm of skew at 2 m/s. That is small for the ball itself, but the effective timestamp varies by up to a frame across the image. This is a minor effect compared with blur at long exposure; the bigger win from industrial/GS boards is controllable short exposure.
- A ~$60 UVC global-shutter camera at 120 fps would give ~34 frames per 10 cm bounce and allow 1–2 ms exposures if lit well. Expected photon-to-app latency is ~15–35 ms (½ exposure + ~1 frame transfer of 8.3 ms + 0–2 frames buffering + ~1 ms processing). This is an estimate, not measured.

### Gaps
- Whether macOS AVFoundation/OpenCV actually delivers 120 fps from the Arducam UVC boards (some UVC cameras only reach their top rate in MJPEG, and OpenCV's AVFoundation backend may not expose all formats) was not verified. Check with `ffmpeg -f avfoundation -list_options true -i "<n>"`.
- No measured latency found for any of these cameras on macOS.
- PS3 Eye shutter type, and the current price/availability of OAK-D Pro, RealSense D435 and event cameras, were not verified.
- **iPhone slow-motion streaming:** I found no source on whether an iPhone can stream 120/240 fps live to a Mac (Continuity Camera or otherwise) or what its latency is. Unanswered.

---

## 3. Ball detection, tracking and 3D estimation methods

### Takeaway
For a coloured ball against a controlled background, HSV thresholding plus connected components (or Hough circles) is sufficient and costs ~1 ms per 640×480 frame on this Mac (**measured**), or <0.1 ms inside a predicted ROI. CNN trackers like TrackNet exist for cluttered broadcast video but are unnecessary here. A Kalman filter with a ballistic (gravity) model and explicit bounce detection, plus forward prediction by the measured latency, is the standard back end. That is what DeepMind used (CNN → triangulation → Kalman), and the NII sim-to-real juggling work used a RealSense plus a Kalman filter.

### Cited Findings
- DeepMind: a tiny CNN (~27k params, 5 spatial + 2 temporal convs) outputs a ball score, a sub-pixel offset and pixel velocity per location. Stereo DLT triangulation runs when both cameras' scores exceed a threshold, then a recursive Kalman filter feeds the policy 3D position and velocity — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- **TrackNet (2019):**
  - A VGG-16 + upsampling heatmap network, 640×360 input, single or multiple frames, aimed at tiny fast balls in broadcast/smartphone video. Tennis: 99.7% precision, 97.3% recall single-match, but 95.3% precision / 75.7% recall in 10-fold CV. Badminton weaker (85.0% precision) — via search summary of [TrackNet, arXiv 1907.03698](https://arxiv.org/pdf/1907.03698)
  - TrackNetV2 improved speed from 2.6 to 31.84 fps (3-in/3-out) — via search summary of [TrackNetV2 record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........e9ad0ca9cbefaf3b71d58089ae71f5e5)
- **Robot-juggling sim-to-real with a 6-DoF arm (NII/SOKENDAI, 2026 poster):**
  - Policy observations were arm joint positions and velocities plus the "Cartesian position of the ball" from a "RealSense Camera with Kalman filter". The end effector was 3D-printed.
  - Training used PPO (RSL-RL) with a GRU, 4096 parallel envs, randomized ball/arm/actuator-gain parameters, and rewards for ball height/airtime, EE–ball XY alignment, action smoothness and an early-termination penalty.
  - It claims zero-shot transfer "with performance comparable to the simulation" but gives no numbers.
  - Source: [Castillo et al., NII Open House 2026 poster F06](https://www.nii.ac.jp/event/upload/oh2026/F06.pdf)
- Octo-Bouncer: OpenCV Hough circle detection at 120 fps 640×480, with a PID on ball position/velocity to tilt the plate — [Electron Dust](https://electrondust.com/2020/03/01/the-octo-bouncer/)
- Ping-Pong-Pi: brightness-clump centroid on a dark background for x,y, and "the z coordinate is extracted from the width of the diameter of the ball" with a single camera. The author says the single camera limits performance — [Hackaday.io](https://hackaday.io/project/20402-ping-pong-pi/details)
- **Measured (this session, M3 Pro, OpenCV 4.13, 1 thread, synthetic frames):**

  | Resolution | MJPEG decode | HSV + open + connected components (full frame) | HSV in 96 px ROI | MOG2 bg-sub | absdiff + threshold |
  |---|---|---|---|---|---|
  | 320×240 | 0.16 ms | 0.36 ms | 0.06 ms | 0.73 ms | 0.03 ms |
  | 640×480 | 0.57 ms | 1.17 ms | 0.08 ms | 2.90 ms | 0.12 ms |
  | 1280×720 | 1.86 ms | 3.54 ms | 0.08 ms | 9.46 ms | 0.36 ms |

  (Synthetic frames compress better than real ones, so real MJPEG decode is likely somewhat slower.)

### Inferences
- **Recommended pipeline (per camera):**
  1. Grab the newest frame, dropping stale buffered frames.
  2. HSV threshold in a ROI around the Kalman prediction, falling back to full frame on loss.
  3. Take the largest blob that passes area and circularity gates, and use its centroid. Use the min-enclosing-circle diameter only when there is no motion streak.
  4. Run an EKF with state (x, y, z, vx, vy, vz) under constant gravity and optional drag. Detect a bounce when the predicted z crosses the paddle surface (paddle pose from forward kinematics), or from a residual spike / vz sign flip. Re-initialize vz with restitution e and inflate the covariance.
  5. Propagate the state forward by the measured latency before handing it to the policy.
- **Drag is non-negligible for a ping-pong ball, but small at these speeds.** Computed: terminal velocity is roughly 8–9 m/s for a 2.7 g, 40 mm ball with Cd ≈ 0.5 (my estimate from standard drag; not sourced). At 2 m/s the drag deceleration is ~(2/8.5)²·g ≈ 0.055 g, which matters slightly for apex prediction over 20+ cm.
- A monocular camera with a tilted view can, in principle, recover 3D from a ballistic fit, because gravity's known magnitude and direction fixes scale. This is noisier than direct triangulation and depends on accurate extrinsics.
- DeepMind's bias-vs-noise result implies that systematic error (e.g. diameter-based depth bias from blur, or extrinsic miscalibration) is worse than random jitter. Calibrate carefully, and randomize observation bias in sim.
- TrackNet-class networks are unnecessary with a coloured ball and controlled background. A ~1 ms HSV detector keeps the whole perception budget tiny.

### Gaps
- No quantitative accuracy numbers for Hough vs HSV centroid on blurred ping-pong balls were found.
- The NII poster gives no quantitative results, camera fps or latency. A full paper was not found.
- A 2024 MDPI paper on a vision-based table-tennis juggling robot ([doi 10.3390/math12111634](https://doi.org/10.3390/math12111634)) returned HTTP 403, so its camera/latency/bounce-count details are unknown.

---

## 4. Camera placement: overhead (current), side view, overhead + side vs stereo, wrist/paddle camera — and a recommendation

### Takeaway
**Recommend: keep the overhead C920 for x/y, paddle-region monitoring and hand gating, and add one side camera at paddle height (ideally a ~$60 120 fps global-shutter UVC board against a dark backdrop) for z and vz.** Height from apparent diameter in the current overhead view changes only ~1.4–3 px per 5 cm and gives σ_z ≈ 1.3–2.7 cm per frame, which is not good enough for apex/impact timing at 5–20 cm bounces. A side view gives ~1 mm per pixel directly. A wrist/paddle camera looking up is not recommended as the primary sensor.

### Cited Findings
- The C920's diagonal FOV is 78° and its max frame rate is 30 fps — [Logitech](https://logitech.com/en-za/shop/p/c920-pro-hd-webcam)
- DeepMind placed its two cameras on opposite sides of the table, which reduced triangulation bias "by about an order of magnitude" vs same-side placement. It used hardware sync — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- Single-camera z from ball diameter was used by Ping-Pong-Pi, whose author says one camera limits performance and two would help — [Hackaday.io](https://hackaday.io/project/20402-ping-pong-pi/details)
- The Octo-Bouncer obtained a "3D position" from Hough circle results at 120 fps (the method is not described; a comment points to FOV-based pixel→world conversion). Camera placement is not stated on the page, and the plate is 8 mm acrylic — [Electron Dust](https://electrondust.com/2020/03/01/the-octo-bouncer/)
- The user's current layout runs the C920 ("front") at 640×480@30 and the wrist camera at 480×640@30 rotated 90° — [env.sh](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/env.sh)

### Inferences
**Geometry assumptions (computed):**
- C920 focal length from 78° dFOV at 16:9: f ≈ 1360 px at 1080p (HFOV 70.4°, VFOV 43.3°), ≈ 907 px at 1280×720, ≈ 453 px at 640×360. At 640×480, assuming it is a 4:3 crop of a 1440×1080 region, f ≈ 605 px; this is unverified, so calibrate.
- The camera is 0.9 m above the table, pitched 60° down. The slant range to the paddle is 0.6–0.9 m depending on paddle height and position.
- A vertical rise Δh moves the ball 0.866·Δh toward the camera (changes the diameter) and 0.5·Δh across the image (looks like moving away on the table).

**Overhead (current) — apparent-diameter height:**

| Mode | Range Z | Ball diameter | Δdiam per +5 cm height | Δdiam per +10 cm | Lateral mm/px | Image shift per +10 cm height |
|---|---|---|---|---|---|---|
| 640×480 | 0.6 m | 40.3 px | +3.1 px | +6.8 px | 1.0 | 50 px |
| 640×480 | 0.75 m | 32.2 px | +2.0 px | +4.2 px | 1.24 | 40 px |
| 640×480 | 0.9 m | 26.9 px | +1.4 px | +2.9 px | 1.49 | 34 px |
| 1280×720 | 0.75 m | 48.4 px | +3.0 px | +6.3 px | 0.83 | 60 px |
| 1280×720 | 0.9 m | 40.3 px | +2.0 px | +4.3 px | 0.99 | 50 px |

- With realistic diameter noise of 0.5–1 px (thresholding edges, lighting, partial blur), 640×480 at 0.75 m gives σ_height ≈ 1.3–2.7 cm per frame, and 1280×720 gives ≈ 0.9–1.8 cm.
- A least-squares velocity over the ~4 frames of a 10 cm bounce's rise at 30 fps then has σ_vz ≈ 0.27 m/s (for σ_z = 2 cm). That means apex σ ≈ 4 cm and impact-time σ ≈ 28 ms. **Not usable for 5–20 cm bounces on its own.**
- Motion blur (§1) also corrupts the diameter, and the error is biased (blur widens the minor axis little but elongates the major axis), which DeepMind found is worse than noise.
- The 0.5·Δh image shift (≈40 px per 10 cm) is large but confounded with forward motion on the table. Since a bouncing ball's horizontal velocity is small and nearly constant, a ballistic EKF can partially disentangle them, but this relies on good extrinsics.
- Overhead x/y accuracy is good (~1–1.5 mm/px). The view also covers the whole workspace, which makes it the right camera for hand gating (§8).
- At 5–20 cm apex the ball moves ≤80 px in the image, so it stays well inside the 480 px-high frame. The user's "keep it low" constraint is satisfied for the overhead view.

**Side camera at paddle height (looking horizontally across the paddle):**
- At 0.5 m distance with f ≈ 605 px, resolution is ≈ 0.83 mm/px in z and in one lateral axis. σ_z ≈ 1 mm gives σ_vz ≈ 1.3 cm/s at 30 fps (4 frames) and ≈ 0.6 cm/s at 120 fps. That means apex σ < 0.5 cm and impact-time σ ≈ 1–2 ms, roughly 20× better than diameter-based z.
- The camera must sit slightly above the paddle plane, or the blade edge (~1–2 cm thick with rubber) hides a resting ball. It also needs a non-moving region of interest, since the paddle moves through the view.
- A **dark backdrop** (black cloth or board) behind the bounce volume makes a white or orange ball trivial to threshold, even on a mono global-shutter sensor.
- Cons: a second camera to mount and calibrate, and the arm or the user's hand may enter its view.

**Two cameras (overhead + side) vs a stereo pair:**
- **Overhead + side (orthogonal):** each axis comes mostly from one camera (x,y overhead; z side), so moderate time offsets between unsynced USB cameras cost little in z. Calibration means intrinsics per camera (checkerboard, ~15 min each), extrinsics to the robot base (an AprilTag/ChArUco board placed at a known pose, or touching the paddle to known points via FK), and timestamp alignment. Tolerant of approximate sync.
- **Stereo pair (two identical cameras side by side):** depth precision degrades with baseline/range, and the cameras need tight time sync. Two UVC webcams are not hardware-synced. A 16 ms offset at 60 fps with the ball at 2 m/s is a 3.3 cm epipolar error, so stereo with unsynced webcams is worse than the orthogonal pair. DeepMind used hardware-synced cameras placed opposite each other for this reason.
- **Recommendation:** orthogonal overhead + side.

**Wrist- or paddle-mounted camera looking up:**
- **Pros:** the ball is always near, so it is large (assuming f ≈ 400 px: ~320 px diameter at 5 cm, 160 px at 10 cm, 80 px at 20 cm). Measurements are already in the paddle frame, which is what the controller needs.
- **Cons:**
  1. The paddle blade blocks the view of a ball on or just above the face unless the camera sits at the blade edge looking obliquely up, or looks through a hole.
  2. At 5–10 cm the ball overfills or leaves the FOV and is inside the near-focus limit of typical fixed-focus webcams (my assumption, not verified for the user's wrist camera).
  3. Ceiling lights saturate the image and swing auto-exposure.
  4. Pixel blur is magnified because the ball is close.
  5. The camera undergoes the stroke acceleration and impact shock. Converting to a world frame needs FK at the exact frame timestamp, but joint readings come over USB serial at 30–60 Hz with their own latency.
  6. The cable flexes with every stroke.
- The existing wrist camera points along the gripper, so with a paddle in the gripper it would mostly see the paddle face edge-on or be blocked. Not recommended as the primary sensor; possibly useful as a coarse "ball is on/near the paddle" detector.

**Expected end-to-end latency (estimated, not measured; photon → state available to the policy):**

| fps | ½ exposure (short exp.) | transfer ≈ 1 frame | buffering 0–2 frames | decode + HSV (measured, 640×480) | Estimated total |
|---|---|---|---|---|---|
| 30 (C920) | 1–17 ms (auto exposure is long) | 33 ms | 0–66 ms | ~1.7 ms | **~45–115 ms** |
| 60 | 1–8 ms | 17 ms | 0–33 ms | ~1.7 ms | **~25–60 ms** |
| 120 | 0.5–4 ms | 8 ms | 0–17 ms | ~1.7 ms (or ~0.7 ms with ROI) | **~12–30 ms** |

Add ½ control period for sample-and-hold (8–17 ms at 30–60 Hz), plus servo command and motion lag (not measured).

**Frames per bounce at apex 5 / 10 / 20 cm:** 6 / 9 / 12 frames at 30 fps; 12 / 17 / 24 at 60 fps; 24 / 34 / 48 at 120 fps (computed, §1).

**Ball colour / background:**
- An **orange ball on the white board** gives high saturation contrast, so HSV works.
- A **white ball on a white board** is nearly invisible to HSV and brightness thresholding. Avoid it, or switch to a dark background.
- For a mono side camera, use a dark backdrop with either ball. Orange in grayscale is mid-grey (~145/255 by the standard luma weights), which is low contrast against white but high contrast against black.

### Gaps
- The C920's real intrinsics at 640×480, and the actual camera-to-paddle range in the user's rig, were not measured. The tables use 0.6–0.9 m as stated by the user.
- The diameter-noise figure (0.5–1 px) is an assumption. It should be measured with the real ball under the real lighting.
- The wrist camera's model, FOV and minimum focus distance (USB ID 0x1e45:0x8022 per [camera_presets.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_presets.py)) are unknown.

---

## 5. Impact sensing without vision: microphone, piezo contact sensor, IMU, proprioception

### Takeaway
Acoustic/piezo impact detection is a well-trodden, cheap way to get impact timing (sub-millisecond onset) and, with 3–4 sensors, impact location on a flat plate. Hobby ping-pong bouncers have closed a lateral-correction loop on mic-based localization alone. I found no evidence that Reist & D'Andrea's blind juggler used a microphone. It is described as having no ball sensors at all. Proprioceptive detection of a 2.7 g ball impact on the SO-101 is very unlikely to work.

### Cited Findings
- **Kuhn's microphone plate:** four small (electret) microphones under the moving platform sensed the ball's impacts. The four intensities (or time differences) were compared to estimate where the bounce happened, and an Arduino PID tilted the plate to keep the ball on. It replaced an earlier LED/phototransistor design — [Laughing Squid](https://laughingsquid.com/pid-controlled-ball-bouncing-on-plate-machine/), [Gizmodo AU](https://gizmodo.com.au/?p=1120350)
- **Another hobby build:** four mics with threshold triggers, using left/right and front/back arrival-time differences for 2D impact position; "worked quite well" — [jpralves.net](https://jpralves.net/post/2017/01/26/ping-pong-ball-juggling-robot.html)
- A student ultrasonic-microphone array localized the impact point on a flat board in real time from inter-channel delays — via search summary of [Tel Aviv University project PDF](https://www.khina.sites.tau.ac.il/_files/ugd/88638f_0c5f33114fe7491696b30b3afff6e3c9.pdf)
- A low-cost microphone + neural classifier localized ping-pong table impacts with ~92% sensitivity — via search summary of [Modares Mechanical Engineering 2015](https://mme.modares.ac.ir/article_8678_en.html)
- **Bouncing-ball physics experiment:** the microphone peak marks each collision time (impact vibration is above the sampling rate, so timing comes from peak location) — via search summary of [Dynamics of the bouncing ball, arXiv 1405.3482](https://arxiv.org/pdf/1405.3482)
- **STReSSD (2020):** used audio with a robotic arm "to track and deflect a bouncing ball in real-time", inferring restitution/noise parameters from audio — [arXiv 2011.03136 abstract](https://arxiv.org/abs/2011.03136). Per a search summary of the full text, it used multiple microphones with time-delay estimation on short buffers, driving a UR5, and room reverberation sometimes caused wrong impact times.
- There is research on piezoelectric row-column sensing on table-tennis rackets for hit and rotation measurement (title only seen) — [PMC12539710](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12539710/)
- **Blind juggler sensing:** "has no sensors to detect the ball at all" — [Robohub, The Swinging Blind Juggler](https://robohub.org/the-swinging-blind-juggler/). In the T-RO 2012 paper, impact states were "measured experimentally" to estimate process-noise spectra (method not given in the abstract) — via search summary of [Reist & D'Andrea T-RO 2012 record](https://researchr.org/publication/ReistD12-0)

### Inferences
- **Practical design for SO-101:** tape 3–4 piezo discs (cheap contact sensors) to the underside of the paddle blade, wired to a microcontroller with interrupt-driven threshold timestamps. That gives (a) impact time with roughly 0.1 ms resolution and (b) a coarse impact position from time difference of arrival (TDOA). Wave speed in a wooden blade is ~km/s, so position resolution needs µs timing. Amplitude ratios are the cruder alternative used by hobbyists.
- **Latency:** the microcontroller sends over its own USB serial link (likely ~1–2 ms) vs ~50–100 ms for the 30 fps camera. I have not measured this. A Mac microphone via CoreAudio adds buffer latency (128–512 samples at 48 kHz ≈ 3–11 ms; standard values, not sourced here) and suffers from room noise and the servo whine of the arm.
- **Proprioception:** the ball's impulse is ≈ m(1+e)v ≈ 0.0027 kg × 1.85 × 1.4 m/s ≈ 0.007 N·s (computed). That is far too small to show up reliably in STS3215 servo current/position readings at 30–60 Hz. Treat "impact timing from proprioception" as infeasible without a dedicated sensor (inference; not tested).
- **IMU on the paddle:** a MEMS accelerometer would see the impact as a sharp transient on top of large stroke accelerations. That is feasible at ≥1 kHz sampling with high-pass filtering, but piezo discs are simpler and cheaper for pure impact detection (inference).
- **What it buys you:** impact timing and location once per bounce (3–5 Hz). That is enough for the period-to-period lateral and height corrections a flat paddle needs (§7), and it is a natural low-dimensional observation for an RL policy ("time since last impact, last impact x,y"). It cannot see a ball the user tosses until it lands.

### Gaps
- No measured latency or reliability numbers were found for piezo-on-paddle impact detection specifically. Hobby sources report qualitative success only.
- I could not confirm STReSSD's microphone hardware, sampling rate or timing accuracy from the full text.

---

## 6. Blind / open-loop bouncing: what the literature says, and whether a proprioception-only RL policy could do it

### Takeaway
Height (vertical) stability can be fully open-loop. Hitting the ball while the paddle is *decelerating* (negative acceleration at impact) makes period-one bouncing self-stabilizing; this is shown by Schaal/Sternad/Atkeson and engineered by Reist & D'Andrea. Lateral stability without sensing needs *geometry*: a concave/parabolic paddle (blind juggler) or a funnel end effector (Ploeger et al., 33 min of open-loop two-ball juggling). With such geometry, an RL policy using only proprioception is plausible, essentially learning a stable periodic stroke. Without it, it is not (§7).

### Cited Findings
- **Schaal, Sternad & Atkeson (1996):** stable periodic paddle juggling "is sufficiently characterized by a negative acceleration of the racket at the moment of impact". Human subjects varying height and gravity showed negative impact acceleration, with fluctuations matching a global stability analysis — [MPI-IS: One-handed juggling, J. Motor Behavior 28(2):165–183](https://is.mpg.de/publications/schaal_jmb_1996)
- **Ronsse et al.:** if the nominal paddle motion has negative acceleration at collision, the system is open-loop stable (small perturbations decay without feedback). Human paddle juggling nonetheless also uses visual and haptic feedback — via search summary of [Ronsse et al. (orbi ronsse08a)](https://orbi.uliege.be/bitstream/2268/25456/1/ronsse08a.pdf) and [Ankarali et al. 2014 (haptic feedback)](https://limbs.lcsr.jhu.edu/wp-content/uploads/2013/05/ankaralihaptic2014.pdf)
- **Ronsse, Lefèvre & Sepulchre:** in a sinusoidally vibrating 2D wedge billiard (a model of 2D impact juggling), some periodic orbits that are unstable in the fixed elastic wedge "become exponentially stable" in the vibrating wedge, i.e. open-loop stabilization by actuation rhythm plus geometry — [Open-loop stabilization of 2D impact juggling (orbi.uliege.be)](https://orbi.uliege.be/bitstream/2268/78758/1/41_openloop%20stabilization%20of%202D%20impact%20juggling.pdf) (abstract read directly)
- **Reist & D'Andrea, "Design and Analysis of a Blind Juggling Robot", IEEE T-RO 28(6):1228–1243, 2012:**
  - Juggles an unconstrained ball without feedback at heights up to 2 m, using a linear motor driving a parabolic aluminium paddle.
  - Open-loop stability comes from two design parameters: paddle curvature and paddle acceleration at impact.
  - The authors linearize the bounce-to-bounce map, analyse nine ball states including spin, and tune design parameters to minimize the effect of experimentally measured process noise.
  - Source: via search summaries of [researchr record](https://researchr.org/publication/ReistD12-0) (full text not read). The earlier version is "Bouncing an Unconstrained Ball in Three Dimensions with a Blind Juggling Robot", ICRA 2009 — [Crossref record](https://api.crossref.org/works/10.1109%2FROBOT.2009.5152616)
- **Blind juggler mechanism in plain terms:**
  - "Slightly concave" paddle: off-centre impacts are nudged back toward the centre on the next bounce.
  - Decelerating strike: a too-high ball arrives later, when the paddle is slower, so the next bounce is lower, and vice versa.
  - The swinging version uses a four-bar linkage to tilt the paddle, with a single linear motor for strike and pumping; "no sensors to detect the ball at all" — [Robohub](https://robohub.org/the-swinging-blind-juggler/)
  - Multi-ball sensorless variants also exist (four paddles juggling four balls) — [Gizmodo](https://gizmodo.com/watch-this-machine-juggle-four-balls-better-than-a-circ-1711247483)
- **Ploeger, Lutter & Peters (CoRL 2020):**
  - "an open-loop policy consisting of a desired position trajectory and a tracking controller" on a Barrett WAM juggled two balls for up to 33 min (~4500 catches) after 56 min of real-world learning with binary rewards.
  - "the funnel-shaped end-effector passively compensates for minor differences in ball trajectories".
  - A ball-observation-based task-space adaptation "even reduced system performance" and was dropped. OptiTrack was used only for the reward and episode start, not control.
  - In 30 trials capped at 120 s: 106.82 s average vs 66.52 s hand-tuned.
  - Sources: [ar5iv 2010.13483](https://ar5iv.labs.arxiv.org/html/2010.13483), [PMLR v155](https://proceedings.mlr.press/v155/ploeger21a.html)
- **Contrast (closed loop):**
  - The 2026 NII sim-to-real juggling RL used ball position from a RealSense + Kalman filter as an observation — [NII poster](https://www.nii.ac.jp/event/upload/oh2026/F06.pdf)
  - Ramadani et al. (ICRA 2026) do 7-DoF Franka non-prehensile tool juggling with offline-optimized trajectories organized for "real-time error correction", i.e. feedback, not open loop — [arXiv 2606.06704](https://arxiv.org/abs/2606.06704)

### Inferences
- **Vertical window (unverified formula):** the classic bouncing-ball stability window for period-one bouncing is −2g(1+e²)/(1+e)² < a_impact < 0. This is from memory of the Holmes/Schaal-Sternad analysis and was not verified from the sources this session. For e = 0.7–0.9 the window is roughly **−1.0 g < a_impact < 0**, so the paddle should be decelerating at contact, at less than ~1 g.
- **Reaching 5–20 cm requires small strokes.** A sinusoidal stroke at the bounce period (202–404 ms, i.e. 2.5–5 Hz) with acceleration under ~1 g at the impact phase needs only millimetres-to-centimetres of amplitude. Plausibly within the SO-101's reach, but servo bandwidth and backlash at 2.5–5 Hz are unverified.
- **Proprioception-only RL policy:**
  - On a **concave paddle** (or funnel insert), a policy with only joint states and a phase/clock input can in principle learn the blind juggler's solution: a periodic stroke with negative impact acceleration. Robustness would come from the physics, as Ploeger's open-loop success suggests.
  - Adding piezo impact timing as an observation would let it phase-lock its stroke to the actual impacts. That is the main residual error source for an open-loop rhythm, since restitution, ball imperfections and servo timing jitter shift the period.
  - The sim must model restitution, paddle curvature, rubber friction/spin, and servo dynamics/latency. Domain randomization over e and timing jitter is essential.
- **Expected bounces:** the blind juggler and Ploeger's funnel show sustained (minutes to hours) open-loop operation is achievable on purpose-built hardware. For an SO-101 with backlash and a 30–60 Hz loop, expect far fewer. That is unquantified.
- **Curvature choice (not sourced):** a parabolic or concave dish of radius R gives an effective tilt of ≈ x/R at impact offset x. That must over-correct the lateral random walk without causing oscillatory overshoot, a tuning problem Reist & D'Andrea solved via the bounce-map eigenvalues. Their actual curvature values were not retrieved.

### Gaps
- Reist & D'Andrea's actual paddle curvature, stroke frequency/amplitude, impact acceleration values, and bounce counts/durations were not retrievable (full text paywalled; the ICRA 2009 PDF mirror returned empty; blindjuggler.org now hosts unrelated gambling content).
- The exact stability-window formula and the Schaal/Sternad measured human impact accelerations were not verified from the papers.
- No published example of a learned (RL) proprioception-only paddle-bouncing policy on real hardware was found.

---

## 7. What a flat, store-bought table-tennis paddle changes

### Takeaway
A flat paddle removes the passive lateral restoring force. Vertical (height) stability via negative impact acceleration still works, but lateral drift is at best neutrally stable, so tilt and timing errors accumulate. My Monte Carlo estimate (assumptions below) says a blind flat-paddle bounce leaves the blade within ~3–16 bounces at 10 cm apex for 0.5–1° tilt error. No fixed stroke or tilt pattern can substitute, because correction must depend on where the ball is. **A flat paddle therefore requires closed-loop ball sensing**: camera, or once-per-bounce acoustic/piezo impact localization (proven on flat plates by hobbyists). Alternatively, glue a shallow concave insert to the face to recover passive stability. A black rubber face up with an orange ball is the best colour combination; the blade occludes the ball from the overhead camera only when the ball is below the blade plane.

### Cited Findings
- The blind juggler's lateral stability comes from its parabolic/concave paddle curvature, one of its two open-loop design parameters. The other, impact acceleration, stabilizes height — via search summary of [Reist & D'Andrea T-RO 2012](https://researchr.org/publication/ReistD12-0); [Robohub](https://robohub.org/the-swinging-blind-juggler/)
- Ploeger et al.'s open-loop juggling relied on a funnel-shaped end effector for "passive stability" — [ar5iv 2010.13483](https://ar5iv.labs.arxiv.org/html/2010.13483)
- Flat-plate ping-pong bouncers that work all close the loop on ball sensing:
  - The Octo-Bouncer (flat acrylic plate, 120 fps camera, PID tilt, "for hours") — [Electron Dust](https://electrondust.com/2020/03/01/the-octo-bouncer/), [Laughing Squid](https://laughingsquid.com/octo-bouncer-ball-juggling-robot/)
  - Kuhn's microphone plate (4 mics, PID tilt) — [Laughing Squid](https://laughingsquid.com/pid-controlled-ball-bouncing-on-plate-machine/)
  - Ping-Pong-Pi (camera, PID on racket tilt, ~2 min) — [Hackaday.io](https://hackaday.io/project/20402-ping-pong-pi/details)
- ITTF Law 2.4.6 requires a matt surface on each side: one side black, the other a bright colour clearly distinguishable from black and from the ball. Since 2021, colours other than red are permitted (lists differ by source: pink, violet/purple, green, blue) — [Megaspin rubber colour rule](https://megaspin.net/rules/rubber-color-rule.asp), [All About Table Tennis rules](https://allabouttabletennis.com/table-tennis-rules-and-regulations.html)

### Inferences
- **Why flat = no passive lateral stability (physics):** a flat face has the same normal everywhere. The post-impact horizontal velocity depends only on incoming velocity, paddle tilt and friction/spin, not on *where* the ball hit. So an off-centre ball is not pushed back, and the position error is a random walk driven by per-bounce tilt and timing errors.
- **Friction partly damps lateral velocity.** For a hollow sphere (I = ⅔mr²) that reaches rolling during contact, tangential velocity drops to 0.6× (computed from angular-momentum conservation about the contact point). Grippy rubber may approach this, but it is unverified for ping-pong rubber, and the spin it creates feeds back into the next bounce.
- **Monte Carlo estimate (computed; model assumptions are mine):**
  - Each bounce adds random lateral velocity ≈ (1+e)·v·sin(tilt error) with e = 0.85, a zero-mean random tilt error each bounce, lateral damping factor a = 1.0 (frictionless) or 0.6 (rolling), and a usable blade radius of 5.5 cm.
  - Median bounces before leaving the blade:

  | Apex | tilt σ 0.5° (a=1 / a=0.6) | 1.0° | 2.0° |
  |---|---|---|---|
  | 5 cm | 11 / 50 | 7 / 16 | 4 / 7 |
  | 10 cm | 7 / 17 | 4 / 7 | 3 / 3 |
  | 20 cm | 4 / 7 | 3 / 3 | 2 / 2 |

  - Any *systematic* tilt (paddle not level, backlash bias) makes this worse, because drift becomes ballistic rather than random-walk. Lower bounces survive longer because each flight is shorter, which supports the user's "keep it low" instinct.
- **Can tilt/stroke tricks substitute?** Not without knowing the ball's position. A deliberate tilt or lateral stroke produces the same correction regardless of impact location, so it can cancel a *known bias* but not random drift. A concave face is precisely a position-dependent tilt.
- **Cheapest ways to get back a passive restoring force:** a thin, shallow concave insert or a raised rim (e.g. a 3D-printed dish taped over the rubber), or a slightly cupped "funnel" ring. Either changes the restitution and spin properties, so they must be modelled in sim.
- **Minimum sensing for a flat paddle:** lateral position at or before each impact, at ≥1 sample per bounce. That can be the overhead camera's x,y (good accuracy, §4) or piezo-TDOA impact location. Height timing can be handled open-loop by negative-acceleration strokes, or closed-loop with a side camera or piezo impact timing.
- **Blade occlusion in the overhead view:**
  - The C920 looks down at ~60° from behind the base, so a ball *above* the paddle face is between camera and blade and is never hidden by the blade itself. The blade (~15 cm diameter) hides the ball only when the ball drops below the blade plane (i.e. falling off), or if the paddle tilts so its face turns away from the camera by more than ~30° (face edge-on to the line of sight).
  - The bigger occluder is the wrist/gripper/handle on the camera side of the blade. A part protruding h above the face casts a "view shadow" h/tan(60°) ≈ 0.58·h long on the face (computed; ≈3 cm for a 5 cm-tall wrist housing). Mount the paddle so the handle and gripper sit on the far side or below the blade plane.
  - From a side camera at paddle height, the blade edge plus rubber (~1–2 cm) hides the bottom of a resting ball unless the camera is slightly above the plane.
- **Rubber colour vs ball colour for HSV tracking:**
  - **Black face up + orange ball** gives maximal contrast in hue, saturation and value. Black face + white ball is also good (value contrast).
  - **Red face + orange ball is the risky pair.** Red (hue ≈ 0–10 or 170–180 on OpenCV's 0–180 scale) sits next to orange (≈ 8–25), so the threshold must be tight and blobs touching the paddle need a circularity check.
  - Red face + white ball is fine on value and saturation.
  - Orange is also close to skin hue (§8), so use saturation and shape gates. These hue ranges are standard OpenCV conventions, not from the sources above.

### Gaps
- No published analysis of lateral bounce-to-bounce dynamics on a flat *rubber* table-tennis face (restitution, friction and spin coupling) was found. The Monte Carlo numbers are a model, not data.
- Achievable paddle tilt repeatability of the SO-101 (STS3215 backlash, compliance at 2.5–5 Hz) is unknown, and it sets which row of the table applies.
- Restitution of a ping-pong ball on paddle rubber at 1–2.4 m/s was not sourced (e = 0.85 assumed).

---

## 8. Detecting the human hand in the workspace (gating the policy) and re-acquiring the ball

### Takeaway
Use the overhead C920 with a static (non-adaptive) background reference of the empty board. Take absdiff → threshold → minus a robot mask rendered from the current joint angles; this costs ~0.1 ms per 640×480 frame (**measured**). Debounce "hand gone" over ~3 frames. MediaPipe Hands is a viable cross-check, at ~9 ms detect+track on Apple Silicon per a third-party benchmark and 12–17 ms on a Pixel 6 per Google, but it misses partial hands at frame edges. Gating logic must let a ball-in-flight state override "hand present" so the arm reacts to a user toss immediately. Re-acquisition after occlusion is 1 frame for detection and ~2–3 frames for a usable velocity: ~70–100 ms at 30 fps, ~17–25 ms at 120 fps.

### Cited Findings
- MediaPipe HandLandmarker (full pipeline) averages 17.12 ms on CPU and 12.27 ms on GPU on a Pixel 6 (Google's benchmark) — [Google MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index)
- On an Apple M4 (third-party README, not independent), MediaPipe's XNNPACK CPU path takes ~3.25 ms tracking and ~8.77 ms detect+track for a 540×720 single-hand frame, vs 0.55/1.65 ms for the project's Neural-Engine port — [fasthands on PyPI](https://pypi.org/project/fasthands/)
- **Measured (this session, M3 Pro, 1 thread):** at 640×480, absdiff vs a reference + grayscale + threshold + countNonZero = **0.12 ms**; MOG2 adaptive background subtraction = **2.90 ms**; HSV + morphology + connected components = **1.17 ms** (table in §3). MediaPipe is not installed in the LeRobot venv and was not benchmarked.
- The user's repo already renders matching MuJoCo views of the scene from calibrated camera poses (`mujoco.Renderer` at 640×480 in [camera_check.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_check.py))

### Inferences
- **Presence detector (recommended):**
  1. Capture a reference frame of the empty workspace with the arm parked (or a per-pose reference).
  2. Each frame: compute `absdiff(frame, ref)`, threshold, and AND with NOT(robot_mask ∪ paddle_mask ∪ ball_blob). Optionally restrict to an "entry zone": the board area excluding the arm's footprint at the bottom centre, and the image borders.
  3. Hand present ⇔ foreground area > A_min for ≥1 frame. Hand gone ⇔ area < A_min for N consecutive frames (N ≈ 3 at 30 fps ≈ 100 ms).
- **Do not use adaptive MOG2 for presence.** It absorbs a stationary hand into the background after its history window (an inherent property of adaptive background models). It is also 25× slower than absdiff here.
- **Robot self-mask:** project the SO-101 + paddle model at the current joint state into the camera. The MuJoCo segmentation render the user already uses is one option, dilated by a few pixels for calibration and latency error. Use joint readings timestamped as close as possible to the frame. Cheaper alternative: a precomputed polygon covering the arm's swept volume in image space. Simple, but it blinds the detector near the paddle, exactly where the hand places the ball.
- **Telling hand from robot by placement:** with the camera behind the base (arm at bottom centre), ask the user to reach in from the left, right or far (top-of-image) side. A foreground blob connected to those image borders and not to the robot mask is a hand. A second cue is skin hue, but orange-ball hue overlaps skin, so do not rely on hue alone.
- **False-positive risks:**
  1. The arm's shadow on the white board. Mitigate with diffuse lighting, or threshold on chroma rather than luma.
  2. Auto-exposure/white-balance drift changing the whole frame. The user's repo locks white balance but not exposure on the C920, so normalize brightness or refresh the reference when no foreground is present.
  3. Cables moving.
  4. The ball itself, already excluded via the ball blob.
- **MediaPipe as a cross-check:** at ~9–17 ms per frame it fits in a 33 ms budget on a separate thread. It needs most of a hand visible, so a hand entering from a frame edge or holding the ball near the paddle may be missed. Use it only to veto, not as the sole gate (inference from the model's palm-detector design; not tested).
- **Gating logic for "user taps the ball up":**
  - If the ball EKF reports the ball airborne above the paddle footprint with a predicted landing inside the blade within the next ~0.5 s, *run the policy regardless of hand presence*.
  - If the ball is static (on the paddle or in the hand) and a hand is present, hold.
  - In sim, train with episodes that start from a resting ball and from small random tosses so the policy handles both.
- **Re-acquisition:** the HSV detector is stateless, so the ball is found in the first frame it is unoccluded, provided the ROI resets to full frame after loss. Reject blobs touching the hand mask, because partially occluded blobs bias the centroid. A Kalman re-init needs ≥2 frames for velocity, ~3 for a gated estimate: ≈ 67–100 ms at 30 fps, ≈ 17–25 ms at 120 fps. A user toss to 10 cm leaves 286 ms of flight, so at 30 fps re-acquisition (≈ 100 ms) plus perception latency (≈ 60–100 ms) consumes most of the flight before the first informed action. At 120 fps with ~20–30 ms latency, ~200 ms remains. This favours a higher-fps camera if user tosses are a real use case.

### Gaps
- No Apple M1/M3 MediaPipe Hands latency was measured, and no independent Apple Silicon benchmark was found (only the M4 numbers in the fasthands README).
- No source quantified MediaPipe's miss rate for partially visible or edge-entering hands.
- MuJoCo segmentation render cost at 640×480 on this Mac was not measured.

---

## 9. Practical: lighting, ball colour vs background, exposure, latency compensation

### Takeaway
Light the bounce volume brightly and diffusely so exposure can drop to ≤2–4 ms (blur ≤ 4–8 mm at 2 m/s). Use an orange ball on the white board with the black rubber face up, and a dark backdrop for any side camera. Measure end-to-end latency once and both compensate it (forward ballistic prediction) and reproduce it in sim with randomization. DeepMind showed mismatched sim latency hurts transfer.

### Cited Findings
- DeepMind used a 4 ms exposure on global-shutter cameras at 125 fps — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- DeepMind sim-to-real was best when sim latency matched reality; 0–50% or 150% of the real latency transferred worse — [arXiv 2309.03315](https://arxiv.org/html/2309.03315)
- Ping-Pong-Pi used active green LED illumination aimed at the ball and a green-channel threshold against a dark background. IR illumination failed because ambient IR overwhelmed it — [Hackaday.io](https://hackaday.io/project/20402-ping-pong-pi/details)
- On macOS, C920 manual exposure did not hold once streaming (user's repo), so short exposure needs either a different camera, a different control path, or so much light that auto-exposure chooses a short time on its own — [camera_presets.py](/Users/praneethsamineni/Documents/GitHub/roboticsexp/so101/camera_presets.py)
- Avoid the C920's H.264 stream (adds latency); use MJPEG/YUY2 — [vMix forum](https://forums.vmix.com/posts/m65054-Video-lag-with-Logitech-c920-camera)
- For measuring latency: film a displayed clock and average several snapshots — [perrytsao/Webcam-Latency-Measurement](https://github.com/perrytsao/Webcam-Latency-Measurement)

### Inferences
- **Lighting target:** auto-exposure will pick roughly 1/(scene brightness). Adding 1–2 bright LED panels close to the bounce volume should push the C920's auto exposure well below 1/30 s. Whether its firmware actually shortens exposure rather than lowering gain is unverified. A UVC global-shutter board with a working manual exposure control is the reliable route.
- **Avoid flicker:** at 60 Hz mains, cheap LED lighting flickers at 120 Hz. With exposures shorter than ~8 ms, frame-to-frame brightness can pulse, which breaks fixed HSV thresholds. Use DC/flicker-free LED panels (standard photography issue; not sourced here).
- **Latency compensation recipe:**
  1. Measure photon→state latency L_cam with an LED toggled by the same Mac over serial, recording the frame at which it first appears, repeated ~50 times for mean and σ.
  2. Measure command→motion latency L_act by commanding a step and watching the paddle on camera or encoder.
  3. In the controller, propagate the ball EKF forward by L_cam + ½ control period.
  4. In sim, delay ball observations by L_cam (randomized ± its measured σ and beyond) and delay actions by L_act.
- **Ball choice:** orange 40 mm ball. A white ball only works against dark backgrounds.
- **Keep the frame-grab thread separate and always use the newest frame.** Drop stale buffered frames to avoid adding 1–2 frame periods of hidden latency (the OBS reports of 33–66 ms buffering are consistent with this risk).

### Gaps
- No measurement of how the C920's auto exposure behaves (exposure vs gain) under bright light on macOS.
- The SO-101 actuation latency (command → paddle motion) was not found in this research. It is likely another tens of milliseconds and must be measured.
