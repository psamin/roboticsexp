# Dream-1 zero-shot on the single SO-101: debugging log

Goal: Dream Labs' `dreamlabs/dream-1-beta`, zero-shot, puts both objects (an orange-capped bottle and a clear bottle)
in the bowl with one SO-101 follower. Runs go through [dream_run.py](dream_run.py); every run's settings, frames and
trajectory are in `logs/dream-live-<time>/`. Regenerate the run table below with
`~/dream-env/bin/python so101/dream_runs.py`.

## How each piece is set up

| Piece | Setting now | Notes |
|---|---|---|
| Model | `dreamlabs/dream-1-beta` | The base model; on a non-YAM arm the client sends `other_embodiment`, so the model reads none of the arm's state, only the two cameras and the prompt |
| Arm model (IK) | Dream's built-in `so101_faas`, or `--arm-file ~/.dreamlabs/arms/so101-new-calib.json` | Both orient the same way (x forward, +pan turns right); built-in sits in Dream's station frame, the file in the URDF's own base frame |
| Tool point | Built-in: 9.7 cm out from the gripper body. File: 9.8 cm until 22:41, then moved to 7.8 cm | Backup of the 9.8 cm file: `~/.dreamlabs/arms/so101-new-calib.backup-20261021.json` |
| Which station arm | `--side right` (default) or `--side left` | Right: wrist cam is `wrist_r`, black `wrist_l`, left arm parked |
| Joint correction | On (built-in) unless `--no-correction`; always off with the arm file | Shoulder −6.2°, elbow +8.7°, wrist +0.8° against Dream's reference arm; base range 233° vs 191°, not matched |
| Servo calibration | LeRobot `my_follower.json`, Oct 8, unchanged | Backup: `my_follower.backup-20261021.json` beside it |
| Cameras | Index 0 top C920 at 1280×720, index 1 wrist (rotated upright), both cropped to 16:9 | Indices shift when devices (iPhone) connect; check with `camera_check.py` |
| Limits | 10° from each joint end; table floor `--floor-cm` 4 (was 2.5); `--no-limits` turns both off | The floor is Dream's `ik.region` |
| Motion | 30 rows/s, 35°/s cap per joint, gripper 0.05/row | Was 20 rows/s and 25°/s before Oct 20 23:20 |

## Changes in order

| When | Change | Why |
|---|---|---|
| Oct 20 22:4x | First live runs: built-in, 20 rows/s, 25°/s, 640×480 top camera, camera in front of the arm | Starting setup |
| Oct 20 23:2x | 30 rows/s, 35°/s, 1280×720 top camera, 90 s runs, two-object task; camera moved behind the arm | Match the YAM station's view |
| Oct 21 21:39 | Arm file from `dreamcli tcp` (tool point at the URDF gripper frame) | Namra: tool point is the usual failure |
| Oct 21 21:41 | Gripper command clamped to [0, 1] | The model's smoothed 0 came out −0.001 and the driver refused it |
| Oct 21 21:45 | New API key | Old key disabled mid-run |
| Oct 21 22:11 | 10° joint-end margin | Bus dropouts with a joint at its end |
| Oct 21 22:21 | 2.5 cm table floor (`ik.region`) | Gripper scraped the table |
| Oct 21 22:28 | `--no-correction` | Test the joint correction |
| Oct 21 22:31 | `--side left` | Test the right-arm habit |
| Oct 21 22:34 | `--no-limits` | Reproduce the 21:46 run that reached the clear bottle |
| Oct 21 22:41 | Tool point moved 2 cm toward the wrist (user, `dreamcli tcp`) | Fingertips close nearer the wrist |
| Oct 21 22:45 | Floor default 4 cm; runs record their tool point, limits and how they ended | No-floor run pressed into the table 24 s and the power cut out |
| Oct 21 22:55 | Defaults: left arm, 2.5 cm floor; tool point back at the fingertip (9.8 cm; the 7.8 cm one kept as `so101-new-calib.inward-7.8cm.json`); camera check shows the YAM view for the side | The user: moving the point inward made it worse; the best run was the left arm with a 2.5 cm floor |

## Findings so far

1. **The API key is not a factor.** The 21:46 run, right after the new key, went to the clear bottle; replies took
   the same 0.5 s before and after.
2. **Nothing is mirrored or flipped.** The camera is not mirrored (Photo Booth mirrors its preview, OpenCV does not);
   +pan turns the real arm right as the model expects; both arm models orient the same way.
3. **The arm tracks its commands** (mean gap about 0°, at most about 7°), so it goes where it is told.
4. **Servo power cuts out when the gripper presses into the table or a joint sits at its end** (21:49, 21:56,
   22:42). All six servos go silent at once and stay silent until the arm's power is switched off and on.
5. **Running it as the station's left arm gave the best run so far (22:34, `--side left`).** For 83 s it reached
   the middle of the table, closed on an object (the gripper stopped at 0.21 instead of 0, holding something for
   about 5 s around 70 s) and carried it toward the bowl; the clear bottle is missing from the table in those frames.
   The table layout changed in the same run (bowl moved to the left, orange bottle to the right), so the side and the
   layout are not yet separated. As the right arm with a floor, every run went right.
6. **The model's gripper height disagrees with reality by a few cm.** It reports the gripper 1 to 6 cm below the
   base plate in several runs, and at rest the three arm models disagree by 2.6 cm. `where_is_gripper.py` measures
   this against a ruler.

## Open questions

- Does the base sit directly on the table, or on a mount? (Sets where the floor really is.)
- With the closed fingertips touching the table, what does `where_is_gripper.py` print for z?
- As the left arm, does it pick up a bottle?
- For Namra: has their SO-101 zero-shot been one arm or two (XLeRobot), and which camera layout?

## Runs

| Run | Arm model | Tool point | Side | Correction | Floor | Joint margin | Steps/s | Cap | Ran | Calls | Base turned | Lowest gripper | Task | Ended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261020-225133 | built-in | built-in (9.7 cm) | right | on | none | 0 deg | 20 | 25 deg/s | 30 s | 13 | -88..-6 (left) | 4.2 cm | Pick up the medicine bottle with the right arm and place it in the bowl | ran to its 30 s limit |
| 20261020-225455 | built-in | built-in (9.7 cm) | right | on | none | 0 deg | 20 | 25 deg/s | 69 s | 34 | -34..-9 (left) | -2.0 cm | Pick up the medicine bottle with the right arm and place it in the bowl | not recorded |
| 20261020-233037 | built-in | built-in (9.7 cm) | right | on | none | 0 deg | 30 | 35 deg/s | 13 s | 10 | -13..58 (right) | 1.9 cm | Put both objects in the bowl with the right arm | stopped at 12.9 s, cause not recorded |
| 20261021-214105 | your file | 9.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 3 s | 1 | -9..-9 (center) | 11.7 cm | Put both objects in the bowl with the right arm | error: gripper command -0.001 refused (bug, fixed) |
| 20261021-214146 | your file | 9.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 20 s | 18 | -68..-11 (left) | 1.7 cm | Put both objects in the bowl with the right arm | API key disabled mid-run |
| 20261021-214339 | your file | 9.8 cm | | | | | | | 0 s | 0 | | | Put both objects in the bowl with the right arm | API key refused at start |
| 20261021-214640 | your file | 9.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 28 s | 29 | -79..5 (left) | -2.8 cm | Put both objects in the bowl with the right arm | stopped at 27.9 s, cause not recorded |
| 20261021-214921 | your file | 9.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 35 s | 31 | -55..36 (right) | -5.7 cm | Put both objects in the bowl with the right arm | arm stopped answering (servo bus) |
| 20261021-215646 | built-in | built-in (9.7 cm) | right | on | none | 0 deg | 30 | 35 deg/s | 75 s | 77 | -44..76 (right) | -1.5 cm | Put both objects in the bowl with the right arm | arm stopped answering (servo bus) |
| 20261021-221225 | built-in | built-in (9.7 cm) | right | on | none | 10 deg | 30 | 35 deg/s | 35 s | 32 | 5..76 (right) | -1.4 cm | Put both objects in the bowl with the right arm | top camera index showed a dark image |
| 20261021-221722 | built-in | built-in (9.7 cm) | right | on | none | 10 deg | 30 | 35 deg/s | 25 s | 23 | -39..22 (right) | -1.0 cm | Put both objects in the bowl with the right arm | not recorded |
| 20261021-222300 | built-in | built-in (9.7 cm) | right | on | 2.5 cm | 10 deg | 30 | 35 deg/s | 42 s | 41 | -7..76 (right) | 3.4 cm | Put both objects in the bowl with the right arm | not recorded |
| 20261021-222637 | built-in | built-in (9.7 cm) | right | on | 2.5 cm | 10 deg | 30 | 35 deg/s | 18 s | 15 | -3..76 (right) | 3.6 cm | Pick up the orange bottle with the right arm | not recorded |
| 20261021-222912 | built-in | built-in (9.7 cm) | right | off | 2.5 cm | 10 deg | 30 | 35 deg/s | 16 s | 16 | 1..76 (right) | 4.0 cm | Put both objects in the bowl with the right arm | not recorded |
| 20261021-223159 | built-in | built-in (9.7 cm) | right | on | 2.5 cm | 10 deg | 30 | 35 deg/s | 28 s | 28 | -29..76 (right) | 3.9 cm | Put both objects in the bowl with the right arm | not recorded |
| 20261021-223439 | built-in | built-in (9.7 cm) | left | on | 2.5 cm | 10 deg | 30 | 35 deg/s | 83 s | 83 | -67..18 (left) | -2.1 cm | Put both objects in the bowl with the left arm | not recorded |
| 20261021-223652 | your file | 9.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 44 s | 45 | -30..6 (left) | -1.9 cm | Put both objects in the bowl with the right arm | not recorded |
| 20261021-224217 | your file | 7.8 cm | right | off | none | 0 deg | 30 | 35 deg/s | 34 s | 35 | -24..23 (right) | -5.1 cm | Put both objects in the bowl with the right arm | arm stopped answering; gripper pressed into the table 24 s (no floor) |
