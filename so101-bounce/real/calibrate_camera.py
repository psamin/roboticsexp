"""Where is the overhead camera? The arm finds out: with the ball resting at the paddle centre, it glides through a
dozen level-paddle poses; at each one the camera's view of the ball is paired with where forward kinematics says the
ball is. A PnP fit (with the focal length searched around the C920's nominal value) gives the camera pose in the
robot base frame; the ball's apparent size gives the scale for depth from radius.

  uv run python real/calibrate_camera.py            # needs real/hold_pose.json (sysid.py set-hold) and the ball

Start with the arm folded at rest. The arm glides to the hold pose and waits; stick the ball at the paddle centre
(a small loop of tape: a free ball rolls off at 1 deg of tilt within a second), take your hand away, press Enter. Each move is slow (2 s) and keeps the paddle level so the ball stays put; if it rolls
off, press Ctrl-C (the arm glides back and turns off). Writes real/camera_calib.json and real/camera_ref.png (a frame
the runner compares its view with, to catch a moved camera).
"""
import json
import re
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import arm as A                                                    # noqa: E402
from rig import BALL_RADIUS, CAMERA_CALIB, CAMERA_REF, CONFIG, BallFinder, Camera, calib_inputs, camera_index, lock_camera, default_K, deg_to_sim, shift_paddle, sim_to_deg  # noqa
from bounce.robot import ArmKinematics, build_model                # noqa: E402

OFFSETS = [(0, 0, 0), (0.25, 0, 0), (-0.25, 0, 0), (0, 0.15, 0.25), (0, -0.15, -0.25), (0, 0.15, -0.1), (0, -0.15, 0.1),
           (0.2, 0.12, 0.2), (-0.2, -0.12, -0.2), (0.2, -0.12, -0.2), (-0.2, 0.12, 0.2),
           (0, 0, 0.45), (0, 0.3, -0.45), (0.2, 0.3, -0.45), (-0.2, 0, 0.45), (0, 0, 0)]   # pan, lift, elbow (rad)
# the four before the last move the paddle ~5 cm down / ~4 cm up, so focal length and camera distance separate


def level(kin, q, iters=10):
    """Adjust wrist flex and roll so the paddle face is level (same Newton step as the simulator)."""
    q = torch.tensor(q, dtype=torch.float32)[None].clone()
    for _ in range(iters):
        def f(qq):
            n = kin.forward(qq)["paddle_R"][..., 2]
            return torch.stack([n[:, 0], n[:, 1]], -1)
        f0 = f(q)
        cols = []
        for j in (3, 4):
            dq = torch.zeros_like(q)
            dq[:, j] = 1e-3
            cols.append((f(q + dq) - f0) / 1e-3)
        q[:, 3:5] -= torch.linalg.solve(torch.stack(cols, -1), f0.unsqueeze(-1)).squeeze(-1)
    return q[0].numpy()


def main():
    hold = json.loads(A.HOLD.read_text())
    kin = shift_paddle(ArmKinematics(build_model(), 1))         # the ball sits at the real blade centre, as in the runner
    cam_cfg = CONFIG["cameras"]["overhead"]
    cam = Camera(camera_index(cam_cfg), cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], latency_s=0.0)   # raw read times
    lock_camera(cam, cam_cfg)
    finder = BallFinder(CONFIG.get("ball_colour", "orange"))
    b = A.open_bus()
    rest = A.hold_here(b)
    obj, img, rad = [], [], []
    sway_fk, sway_det = [], []
    ref_frame = None
    try:
        A.glide(b, hold, 3.0)
        input("Tape the ball to the centre of the paddle (small loop of tape under it), take your hand away, press Enter...")
        q_hold = deg_to_sim(hold)
        for k, (dp, dl, de) in enumerate(OFFSETS):
            q = q_hold.copy()
            q[0] += dp
            q[1] += dl
            q[2] += de
            q = level(kin, q)
            jump = float(np.abs(np.degrees(q - q_hold)).max())
            if jump > 30:                                         # the levelling solve diverged: do not swing there
                print(f"pose {k + 1}: skipped (levelling asked for a {jump:.0f} deg move)")
                continue
            A.glide(b, sim_to_deg(q), 2.0)
            time.sleep(1.0)
            dets = []
            n0 = cam.get()[2]
            t_end = time.time() + 5.0
            while len(dets) < 15 and time.time() < t_end:
                frame, _, n = cam.get()
                if frame is not None and n != n0:
                    n0 = n
                    d = finder.find(frame)
                    if d is not None:
                        dets.append(d)
                time.sleep(0.005)
            if len(dets) < 15:
                cv2.imwrite(str(A.HERE / "logs" / "calib_no_ball.jpg"), cam.get()[0])
                print(f"pose {k + 1}: ball not found by camera {cam_cfg['index']} ({len(dets)} detections in 5 s); "
                      f"frame saved to real/logs/calib_no_ball.jpg")
                continue
            u, v, r = np.median(np.array(dets), 0)
            if rad and not 0.6 < r / rad[0] < 1.6:                # a different blob, or the ball fell off
                print(f"pose {k + 1}: skipped (ball looks {r:.1f} px, first pose {rad[0]:.1f} px)")
                continue
            qm = deg_to_sim(A.read_pos(b))
            fk = kin.forward(torch.tensor(qm, dtype=torch.float32)[None])
            p = (fk["paddle_o"][0] + BALL_RADIUS * fk["paddle_R"][0, :, 2]).numpy()
            obj.append(p)
            img.append([u, v])
            rad.append(r)
            ref_frame = cam.get()[0].copy()                     # hands away, camera where it is calibrated
            print(f"pose {k + 1}/{len(OFFSETS)}: ball at {np.round(p, 3)} m -> pixel ({u:.1f}, {v:.1f}), radius {r:.1f} px")
        if len(obj) >= 6:                                         # camera latency: sway the paddle, match timing later
            A.glide(b, hold, 2.0)
            print("latency test: the paddle sways side to side for 5 s")
            q0, n0, t0 = deg_to_sim(hold), cam.get()[2], time.perf_counter()
            while (t := time.perf_counter()) - t0 < 5.0:
                q = q0.copy()
                q[0] += 0.085 * np.sin(2 * np.pi * (t - t0))      # +-3 cm sideways at 1 Hz; pan keeps the paddle level
                b.sync_write("Goal_Position", sim_to_deg(q))
                qm = deg_to_sim(A.read_pos(b))
                tr = time.perf_counter()
                fk = kin.forward(torch.tensor(qm, dtype=torch.float32)[None])
                sway_fk.append((tr, *(fk["paddle_o"][0] + BALL_RADIUS * fk["paddle_R"][0, :, 2]).numpy()))
                frame, tf, n = cam.get()
                if frame is not None and n != n0:
                    n0 = n
                    d = finder.find(frame)
                    if d is not None:
                        sway_det.append((tf, d[0], d[1]))
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - t)))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        print(f"back to the hold pose ({A.glide(b, hold, 2.0):.1f} deg error)")
        A.pause("Press Enter (the arm folds back and turns off; then untape the ball)...")
        A.glide(b, {n: rest[n] for n in A.ARM}, 3.0)
        A.park(b)
        cam.close()
    W, H = cam_cfg["width"], cam_cfg["height"]
    obs = {"obj": np.asarray(obj).tolist(), "img": np.asarray(img).tolist(), "rad": np.asarray(rad).tolist(), "image_size": [W, H]}
    (A.HERE / "logs").mkdir(exist_ok=True)
    (A.HERE / "logs" / f"calib_obs-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(obs))
    if len(obj) < 6:
        raise SystemExit("too few poses for a calibration")
    out = fit(np.array(obj), np.array(img), np.array(rad), W, H)
    lat = latency(out, sway_fk, sway_det)
    if lat is not None:
        out["camera_latency_s"], out["latency_fit_px"] = lat
        cfg_file = A.HERE / "config.yaml"
        txt = cfg_file.read_text()
        line = f"camera_latency_s: {lat[0]:.3f}     # measured by calibrate_camera.py (frame read time minus capture)"
        txt = re.sub(r"^camera_latency_s:.*$", line, txt, flags=re.M) if re.search(r"^camera_latency_s:", txt, re.M) \
            else txt.rstrip("\n") + "\n" + line + "\n"
        cfg_file.write_text(txt)
        print(f"camera latency {1000 * lat[0]:.0f} ms (fit {lat[1]:.1f} px) -> real/config.yaml")
    else:
        print("camera latency not measured (too few detections while swaying, or no clear minimum); keeping the config value")
    out.update(calib_inputs())                                  # paddle geometry it was fitted with
    CAMERA_CALIB.write_text(json.dumps({**out, **obs}, indent=1))
    cv2.imwrite(str(CAMERA_REF), ref_frame)
    print(f"camera at {np.round(out['camera_position_m'], 3)} m (base frame); reprojection {out['reprojection_px']:.2f} px; "
          f"depth from ball size: rms error {out['depth_err_cm_rms']:.1f} cm; focal {out['K'][0][0]:.0f} px; "
          f"radius scale {out['radius_scale']:.2f} -> {CAMERA_CALIB}, {CAMERA_REF.name}")


def latency(out, fk, det):
    """Delay (s) between a frame's capture and its read time: the shift that best lines up the ball's forward-
    kinematics track (projected with the fitted camera) with its detections. None if unclear."""
    if len(det) < 30:
        return None
    f, d = np.array(fk), np.array(det)
    K = np.array(out["K"])
    R_cb = np.array(out["R_base_from_cam"]).T
    rvec, tvec = cv2.Rodrigues(R_cb)[0], (-R_cb @ np.array(out["t_base_from_cam"])).reshape(3, 1)
    errs = []
    lags = np.arange(0.0, 0.25, 0.0025)
    for L in lags:
        ts = d[:, 0] - L
        ok = (ts > f[0, 0]) & (ts < f[-1, 0])
        if ok.sum() < 20:
            errs.append(np.inf)
            continue
        P = np.stack([np.interp(ts[ok], f[:, 0], f[:, k]) for k in (1, 2, 3)], 1)
        uv = cv2.projectPoints(P, rvec, tvec, K, None)[0][:, 0]
        errs.append(float(np.sqrt(((uv - d[ok, 1:]) ** 2).sum(-1).mean())))
    i = int(np.argmin(errs))
    if i in (0, len(lags) - 1) or not np.isfinite(errs[i]):
        return None
    return float(lags[i]), errs[i]


def looking_down(c, yaw):
    """(rvec, tvec) of a camera at c (base frame) pointing straight down, image x axis at `yaw` around vertical."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    R_bc = np.stack([x, np.cross(z, x), z], 1)
    R_cb = R_bc.T
    return cv2.Rodrigues(R_cb)[0], (-R_cb @ np.asarray(c, float)).reshape(3, 1)


def fit(obj, img, rad, W, H):
    """Camera pose by PnP over a focal-length search. The level-paddle points lie close to one plane, so PnP also
    admits a mirrored pose (camera under the table); solves start from overhead guesses and any pose that puts the
    camera below the points or the points behind the camera is rejected."""
    best = None
    guesses = [looking_down(c, yaw) for c in [(0.35, 0, 0.95), (0.1, 0, 1.1), (-0.17, 0, 0.94), (0.35, 0, 0.7)]
               for yaw in np.radians([0, 90, 180, 270])]
    for s in np.linspace(0.9, 1.15, 11):                 # around the C920's nominal focal length
        K = default_K(W, H)
        K[0, 0] *= s
        K[1, 1] *= s
        for rv0, tv0 in guesses:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rv0.copy(), tv0.copy(), useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            R_cb = cv2.Rodrigues(rvec)[0]
            z_cam = (obj @ R_cb.T + tvec.ravel())[:, 2]
            cam_z = (-R_cb.T @ tvec.ravel())[2]
            if (z_cam < 0.05).any() or cam_z < obj[:, 2].max() + 0.1:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.sqrt(((proj[:, 0] - img) ** 2).sum(-1).mean()))
            if best is None or err < best[0]:
                best = (err, K, rvec, tvec)
    if best is None:
        raise SystemExit("no camera pose above the paddle fits the detections; nothing written")
    err, K, rvec, tvec = best
    R_cb = cv2.Rodrigues(rvec)[0]                       # base -> camera
    z_cam = (obj @ R_cb.T + tvec.ravel())[:, 2]
    f = 0.5 * (K[0, 0] + K[1, 1])
    radius_scale = float(np.median(f * BALL_RADIUS / (rad * z_cam)))
    if not 0.5 < radius_scale < 2.0 or err > 8:
        raise SystemExit(f"calibration looks wrong (reprojection {err:.1f} px, radius scale {radius_scale:.2f}); "
                         "nothing written")
    depth_err = f * BALL_RADIUS / (rad * radius_scale) - z_cam
    t_bc = -R_cb.T @ tvec.ravel()
    return {"K": K.tolist(), "R_base_from_cam": R_cb.T.tolist(), "t_base_from_cam": t_bc.tolist(),
            "radius_scale": radius_scale, "reprojection_px": err,
            "depth_err_cm_rms": float(100 * np.sqrt((depth_err ** 2).mean())), "camera_position_m": t_bc.tolist(),
            "poses": len(obj), "image_size": [W, H], "created": time.strftime("%Y-%m-%d %H:%M")}

if __name__ == "__main__":
    main()
