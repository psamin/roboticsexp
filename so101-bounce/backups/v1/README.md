# v1 rig snapshot (2026-11-02)

Real-arm setup at the end of the 2026-11-02 session. Code version: git tag `so101-bounce-v1`.

| File | What |
|---|---|
| `config.yaml` | ports, joint map (wrist offsets flex 10.8, roll 92.5 deg), camera latency 48 ms, overhead camera on auto exposure / focus locked after autofocus |
| `hold_pose.json` | bounce pose for the new (second) paddle, levelled with `real/level_ball.py` and confirmed by eye |
| `camera_calib.json` | overhead C920 at (-0.10, 0.01, 0.80) m; reprojection 1.4 px, depth rms 1.1 cm |
| `calibration/` | LeRobot servo calibration (follower and leader) |
| `v16-p32-lo.pt` | deploy policy |

Restore (from `so101-bounce/`):

    cp backups/v1/{config.yaml,hold_pose.json,camera_calib.json} real/
    cp -R backups/v1/calibration/. real/calibration/
    cp backups/v1/v16-p32-lo.pt policies/

Run:

    uv run python real/bounce_run.py --checkpoint policies/v16-p32-lo.pt --launch --launch-lean 6 --launch-steps 6 --target-cm 9 --record

Known state: the scripted flick throws the ball back toward the handle on the real arm (lean 6 is the sim's best fix);
the camera calibration predates the last re-level by 2.4 deg of wrist flex (under 1 cm of ball-position error).
Re-level if the paddle is re-taped; recalibrate the camera if it moves.
