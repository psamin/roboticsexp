"""Synthetic training images for the ball detector, with everything that changes the look randomised.

  MUJOCO_GL=egl python scripts/make_detector_data.py --out data/detector --images 50000 --shard 2000

Each image: the SO-101 near its bouncing pose with a procedural paddle (shape, rubber colours, wood, handle), a ball
(orange / white / yellow shades) above, on, partly behind or off the paddle, or absent; skin-toned "hand" capsules and
other coloured objects as distractors; random floor colour, lights, camera pose around the overhead (C920 behind the
arm) and side placements, field of view; motion blur from the ball's speed over a random exposure; sensor noise,
white balance, gamma and JPEG compression. Real assets (assets/README.md): SimReady paddle meshes for a share of the
looks, tabletop props (mugs, tape, markers, keyboard, notebook, lamp) on the table or held, photo table surfaces. Labels: ball centre (px), radius (px), visible fraction.
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.set_num_threads(1)                         # many workers share the node's CPUs

from bounce.robot import GRIPPER_CLOSED_ON_HANDLE, build_spec
from bounce.visual import add_paddle, add_prop, asset_catalog, color_arm, sample_look, surfaces, _tint

W, H = 320, 240
N_DISTRACT = 6
N_PROPS = 4                                     # real tabletop objects (assets/simready), on the table or held


def build(look, rng):
    import mujoco
    spec = build_spec(with_ball=True)
    for name in ("blade", "handle"):
        spec.geom(name).rgba = [0, 0, 0, 0]
    add_paddle(spec, spec.worldbody.add_body(name="paddle_vis", mocap=True), look)
    spec.geom("ball").rgba = list(look["ball_rgb"]) + [1]
    color_arm(spec, look["arm_rgb"])
    # lighting like a room: a main light from above (dim .. bright), ambient fill, 0-3 accent spots;
    # 2700-6500 K colours, hard or soft shadows (the arm and paddle shadow the ball)
    level = float(np.exp(rng.uniform(np.log(0.15), np.log(2.2))))

    def colour(strength):
        warm = rng.uniform(0, 1)                       # 0 = 6500 K daylight, 1 = 2700 K tungsten
        return list(np.clip(np.array([1.0, 0.92 - 0.2 * warm, 0.85 - 0.4 * warm]) * strength, 0, 3))

    tilt = np.deg2rad(rng.uniform(0, 45))
    az = rng.uniform(0, 2 * np.pi)
    key_dir = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    key = spec.worldbody.add_light(pos=list(np.array([0.3, 0.0, 0.3]) - 1.5 * key_dir), dir=list(key_dir),
                                   diffuse=colour(level))
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.castshadow = bool(rng.random() < 0.85)
    key.ambient = list(np.full(3, level * rng.uniform(0.1, 0.45)))
    key.specular = list(np.full(3, rng.uniform(0.1, 0.6)))
    for i in range(int(rng.integers(0, 4))):
        pos = rng.uniform([-0.8, -0.9, 0.3], [1.0, 0.9, 1.9])
        aim = np.array([0.3, 0.0, 0.3]) + rng.normal(0, 0.15, 3)
        spot = spec.worldbody.add_light(pos=list(pos), dir=list(aim - pos), diffuse=colour(level * rng.uniform(0.3, 1.2)))
        spot.type = mujoco.mjtLightType.mjLIGHT_SPOT
        spot.cutoff, spot.exponent = float(rng.uniform(50, 90)), float(rng.uniform(0.5, 4))
        spot.castshadow = bool(rng.random() < 0.6)
        spot.specular = list(np.full(3, rng.uniform(0.1, 0.6)))

    surf = surfaces()

    def material(name, base_rgb, photo_p=0.0):
        """A random floor/wall surface: plain, gradient, checker, speckled (procedural texture) or a photo of a table
        top (wood, laminate, plastic, cloth)."""
        if surf and rng.random() < photo_p:
            tex = spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(rng.choice(surf)))
            mat = spec.add_material(name=name, rgba=list(np.clip(_tint(rng), 0, 1)) + [1])
            mat.specular, mat.shininess = float(rng.uniform(0, 0.4)), float(rng.uniform(0, 0.5))
            texs = list(mat.textures)
            texs[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mat.textures = texs
            mat.texuniform = True
            mat.texrepeat = [float(1 / rng.uniform(0.3, 1.5))] * 2        # one photo tile covers 0.3-1.5 m
            return name
        kind = rng.choice(["flat", "gradient", "checker", "speckle"], p=[0.4, 0.2, 0.2, 0.2])
        mat = spec.add_material(name=name, rgba=[1, 1, 1, 1] if kind != "flat" else list(base_rgb) + [1])
        mat.specular, mat.shininess = float(rng.uniform(0, 0.5)), float(rng.uniform(0, 0.6))
        if kind != "flat":
            tex = spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=256, height=256)
            tex.builtin = {"gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           "speckle": mujoco.mjtBuiltin.mjBUILTIN_FLAT}[kind]
            tex.rgb1 = list(base_rgb)
            tex.rgb2 = list(np.clip(np.array(base_rgb) * rng.uniform(0.5, 1.5) + rng.normal(0, 0.1, 3), 0, 1))
            if kind == "speckle":
                tex.mark, tex.random = mujoco.mjtMark.mjMARK_RANDOM, float(rng.uniform(0.01, 0.2))
                tex.markrgb = list(rng.uniform(0, 1, 3))
            texs = list(mat.textures)
            texs[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mat.textures = texs
            mat.texrepeat = list(rng.uniform(1, 12, 2))
        return name

    fl = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01],
                                 material=material("floor_mat", look["floor_rgb"], photo_p=0.4))
    fl.contype = fl.conaffinity = 0
    # walls / backdrop: random colours and textures, sometimes the dark backdrop planned behind the side view
    for k, (pos, quat) in enumerate((([-0.45, 0, 0.5], [0.7071, 0, 0.7071, 0]), ([1.1, 0, 0.5], [0.7071, 0, -0.7071, 0]),
                                     ([0.3, 0.9, 0.5], [0.7071, 0.7071, 0, 0]), ([0.3, -0.9, 0.5], [0.7071, -0.7071, 0, 0]))):
        dark = rng.random() < 0.25
        rgb = list(rng.uniform(0.0, 0.12, 3)) if dark else list(rng.uniform(0.3, 1.0, 3))
        wall = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01],
                                       pos=list(np.array(pos) + rng.normal(0, 0.1, 3)), quat=quat,
                                       material=material(f"wall{k}_mat", rgb, photo_p=0.1))
        wall.contype = wall.conaffinity = 0
    # distractors: skin-toned hand-like capsules and assorted objects (mocap so they can be moved per frame)
    for i in range(N_DISTRACT):
        b = spec.worldbody.add_body(name=f"distract{i}", mocap=True)
        if i < 2:
            skin = np.array([0.95, 0.75, 0.62]) * rng.uniform(0.35, 1.0)
            g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rng.uniform(0.008, 0.012), 0.035, 0],
                           rgba=list(np.clip(skin, 0, 1)) + [1])
        else:
            kind = rng.choice([mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                               mujoco.mjtGeom.mjGEOM_ELLIPSOID])
            size = rng.uniform(0.01, 0.05, 3)
            if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:           # never ball-like: one axis >= 1.8x another
                size[rng.integers(3)] = min(size) * rng.uniform(1.8, 3.0)
            g = b.add_geom(type=kind, size=list(size), rgba=list(rng.uniform(0, 1, 3)) + [1])
        g.contype = g.conaffinity = 0
    props = [k for k, v in asset_catalog().items() if v["kind"] == "prop"]
    for i in range(N_PROPS if props else 0):
        add_prop(spec, spec.worldbody.add_body(name=f"prop{i}", mocap=True), str(rng.choice(props)), _tint(rng))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.quality.offsamples = 0          # no multisampling: segmentation ids must not be blended (EGL)
    spec.visual.headlight.active = 0            # no camera headlight: dim scenes stay dim, highlights vary
    spec.visual.headlight.ambient = list(np.full(3, rng.uniform(0.0, 0.08)))
    return spec.compile()


def camera(rng, target):
    """A free camera aimed near the paddle: mostly the overhead C920 placement, sometimes the side camera."""
    kind = rng.choice(["overhead", "side", "front"], p=[0.55, 0.35, 0.10])
    if kind == "overhead":    # behind the base, ~0.9 m up, looking down ~60 deg
        az, el, dist = rng.uniform(150, 210), rng.uniform(-75, -45), rng.uniform(0.75, 1.1)
    elif kind == "side":      # roughly at paddle height, looking across
        az, el, dist = rng.choice([-90, 90]) + rng.uniform(-20, 20), rng.uniform(-20, 5), rng.uniform(0.5, 0.9)
    else:
        az, el, dist = rng.uniform(-30, 30), rng.uniform(-45, -10), rng.uniform(0.6, 1.0)
    return kind, dict(lookat=target + rng.normal(0, [0.09, 0.09, 0.06]), azimuth=az, elevation=el, distance=dist)


def project(m, d, r, cam_kw, p, fovy):
    """Pixel position of world point p for the free camera (same math MuJoCo's renderer uses)."""
    import mujoco
    cam = mujoco.MjvCamera()
    for k, v in cam_kw.items():
        setattr(cam, k, v)
    az, el = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = p - eye
    z = rel @ fwd
    f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
    return np.array([W / 2 + f * (rel @ right) / z, H / 2 - f * (rel @ up) / z]), z, f


def distort_and_roll(img, uv, rng):
    """Radial lens distortion (k1) and camera roll; returns the warped image and the moved label point."""
    k1 = rng.uniform(-0.12, 0.05) if rng.random() < 0.7 else 0.0       # webcams: mild barrel distortion
    roll = np.deg2rad(rng.uniform(-8, 8)) if rng.random() < 0.6 else 0.0
    cx, cy, f = W / 2, H / 2, 0.5 * W
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    # inverse map: output pixel -> undo roll -> undo distortion (fixed-point) -> source pixel
    c, s_ = np.cos(-roll), np.sin(-roll)
    xr, yr = (xs - cx) * c - (ys - cy) * s_, (xs - cx) * s_ + (ys - cy) * c
    xn, yn = xr / f, yr / f
    xu, yu = xn.copy(), yn.copy()
    for _ in range(4):
        r2 = xu * xu + yu * yu
        xu, yu = xn / (1 + k1 * r2), yn / (1 + k1 * r2)
    sx, sy = np.clip(xu * f + cx, 0, W - 1), np.clip(yu * f + cy, 0, H - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
    im = img.astype(np.float32)
    out = (im[y0, x0] * (1 - wx) * (1 - wy) + im[y0, x1] * wx * (1 - wy) + im[y1, x0] * (1 - wx) * wy +
           im[y1, x1] * wx * wy)
    # forward map for the label: distort, then roll
    un, vn = (uv[0] - cx) / f, (uv[1] - cy) / f
    r2 = un * un + vn * vn
    dn, dv = un * (1 + k1 * r2), vn * (1 + k1 * r2)
    c, s_ = np.cos(roll), np.sin(roll)
    u2, v2 = (dn * c - dv * s_) * f + cx, (dn * s_ + dv * c) * f + cy
    return out.astype(np.uint8), np.array([u2, v2]), 1 + k1 * r2


def postprocess(img, rng):
    from PIL import Image
    x = img.astype(np.float32) / 255
    x = x * rng.uniform(0.8, 1.2, 3)                                   # white balance mismatch
    if rng.random() < 0.85:          # webcam auto-exposure: toward a mid-grey target, gain capped (dim rooms stay dark)
        gain = float(np.clip(rng.uniform(0.28, 0.6) / max(float(x.mean()), 1e-3), 0.3, rng.uniform(3.0, 10.0)))
    else:                                                              # fixed exposure: under- to over-exposed
        gain = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
    x = np.clip(x * gain, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.2:                                             # LED flicker banding (rolling shutter)
        period = rng.uniform(10, 60)
        x = x * (1 - rng.uniform(0.03, 0.15) * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(H) / period +
                                                                     rng.uniform(0, 6.3))))[:, None, None]
    sigma = rng.uniform(0.003, 0.02) * (1.0 + 0.6 * max(gain - 1.0, 0.0))   # sensor gain noise grows with gain
    x = x + rng.normal(0, sigma, x.shape)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.uniform(40, 95)))
    return np.asarray(Image.open(buf).convert("RGB"))


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=20000)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--per_scene", type=int, default=20, help="images per sampled look (scene rebuild is slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first_shard", type=int, default=0, help="shard numbering offset (one process per shard)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    import torch
    from bounce.env import BounceEnv, EnvCfg
    env = BounceEnv(EnvCfg(num_envs=a.per_scene, device="cpu", seed=a.seed))
    imgs, labels, k_shard, done = [], [], a.first_shard, 0
    gl_reported = False
    while done < a.images:
        env.forced = {"difficulty": 1.0, "rest_start": None}
        env.reset_all()
        look = sample_look(rng, blade=env.blade[0].numpy())
        m = build(look, rng)
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, H, W)              # one GL context per process (two at once fail under EGL)
        if not gl_reported:
            try:
                from OpenGL import GL
                print("GL renderer:", GL.glGetString(GL.GL_RENDERER).decode(), flush=True)
            except Exception as e:
                print("GL renderer: unknown", e, flush=True)
            gl_reported = True
        bid = m.body("ball").id
        qa = m.jnt_qposadr[m.body_jntadr[bid]]
        ball_geom = m.geom("ball").id
        pv = m.body_mocapid[m.body("paddle_vis").id]
        prop_mocap = [m.body_mocapid[i] for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"prop{j}")
                                                  for j in range(N_PROPS)) if i >= 0]
        # arm poses near the hold pose: random actions held for a few steps
        for _ in range(int(rng.integers(0, 15))):
            env.step(torch.tensor(rng.uniform(-0.6, 0.6, (a.per_scene, 5)), dtype=torch.float32))
        q, o, R = env.q.numpy(), env.o_prev.numpy(), env.R_prev.numpy()
        for i in range(a.per_scene):
            d.qpos[:5], d.qpos[5] = q[i], GRIPPER_CLOSED_ON_HANDLE
            n = R[i][:, 2]
            mode = rng.choice(["air", "rest", "edge", "absent", "far"], p=[0.62, 0.12, 0.08, 0.10, 0.08])
            if mode == "air":
                p = o[i] + n * rng.uniform(0.025, 0.30) + R[i][:, 0] * rng.uniform(-0.07, 0.07) + \
                    R[i][:, 1] * rng.uniform(-0.07, 0.07)
            elif mode == "rest":
                p = o[i] + n * 0.02 + R[i][:, 0] * rng.uniform(-0.05, 0.05) + R[i][:, 1] * rng.uniform(-0.05, 0.05)
            elif mode == "edge":      # beside or just below the blade rim: partly hidden
                ang = rng.uniform(0, 2 * np.pi)
                p = o[i] + 0.085 * (np.cos(ang) * R[i][:, 0] + np.sin(ang) * R[i][:, 1]) + n * rng.uniform(-0.03, 0.02)
            elif mode == "far":
                p = o[i] + rng.uniform([-0.3, -0.3, -0.1], [0.3, 0.3, 0.4])
            else:
                p = np.array([5.0, 5.0, -5.0])                              # out of the scene
            vel = rng.normal(0, [0.4, 0.4, 1.5])
            d.mocap_pos[pv] = o[i]
            qv = np.zeros(4)
            mujoco.mju_mat2Quat(qv, R[i].astype(np.float64).flatten())
            d.mocap_quat[pv] = qv
            for j in range(N_DISTRACT):
                mid = m.body_mocapid[m.body(f"distract{j}").id]
                near = rng.random() < (0.5 if j < 2 else 0.3)
                d.mocap_pos[mid] = (o[i] + rng.uniform([-0.15, -0.15, 0.0], [0.15, 0.15, 0.25])) if near else \
                    np.array([0.3, 0, 0]) + rng.uniform([-0.6, -0.6, 0], [0.6, 0.6, 0.02])
                qd = rng.normal(size=4)
                d.mocap_quat[mid] = qd / np.linalg.norm(qd)
            for mid in prop_mocap:
                u = rng.random()
                if u < 0.6:                                            # standing on the table around the arm
                    xy = rng.uniform([-0.35, -0.55], [0.75, 0.55])
                    if np.hypot(*xy) < 0.12:
                        xy = xy / max(np.hypot(*xy), 1e-6) * 0.12
                    d.mocap_pos[mid] = [xy[0], xy[1], 0.0]
                    yaw = rng.uniform(0, 2 * np.pi)
                    d.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                elif u < 0.8:                                          # held up near the paddle
                    d.mocap_pos[mid] = o[i] + rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 0.3])
                    qd = rng.normal(size=4)
                    d.mocap_quat[mid] = qd / np.linalg.norm(qd)
                else:                                                  # not in the scene
                    d.mocap_pos[mid] = [5.0, 5.0, -5.0]
            kind, cam_kw = camera(rng, o[i] + n * 0.08)
            fovy = rng.uniform(42, 62)
            m.vis.global_.fovy = fovy
            cam = mujoco.MjvCamera()
            for key, val in cam_kw.items():
                setattr(cam, key, val)
            # motion blur: average sub-frames over the exposure
            exposure = rng.choice([0.002, 0.005, 0.01, 0.02, 0.033])
            acc = np.zeros((H, W, 3), np.float32)
            subs = 1 if exposure < 0.004 else 3
            for s in range(subs):
                ps = p + vel * exposure * (s / max(subs - 1, 1) - 0.5)
                d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = ps, [1, 0, 0, 0]
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                acc += r.render()
            img = (acc / subs).astype(np.uint8)
            d.qpos[qa:qa + 3] = p
            mujoco.mj_forward(m, d)
            try:
                r.enable_segmentation_rendering()
                r.update_scene(d, cam)
                visible_px = int((r.render()[..., 0] == ball_geom).sum())
            except IndexError:                      # garbled ids: find the ball's pixels by rendering without it
                r.disable_segmentation_rendering()
                r.update_scene(d, cam)
                with_ball = r.render().astype(np.int16)
                d.qpos[qa:qa + 3] = [5.0, 5.0, -5.0]
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                visible_px = int((np.abs(with_ball - r.render()).sum(-1) > 12).sum())
                d.qpos[qa:qa + 3] = p
                mujoco.mj_forward(m, d)
            r.disable_segmentation_rendering()
            uv, z, f = project(m, d, r, cam_kw, p, fovy)
            rad = f * 0.02 / z if z > 0 else 0.0
            area = np.pi * rad * rad
            area0 = np.pi * rad * rad                     # visibility is measured on the undistorted render
            img, uv, rscale = distort_and_roll(img, uv, rng)
            rad = rad * rscale
            img = postprocess(img, rng)
            in_frame = mode != "absent" and z > 0 and -rad < uv[0] < W + rad and -rad < uv[1] < H + rad
            vis = min(1.0, visible_px / area0) if in_frame and area0 > 0 else 0.0
            imgs.append(img)
            labels.append([uv[0], uv[1], rad, vis, float(in_frame), {"overhead": 0, "side": 1, "front": 2}[kind],
                           float(img.mean())])
            done += 1
            if len(imgs) == a.shard or done >= a.images:
                np.savez_compressed(os.path.join(a.out, f"shard{k_shard:04d}.npz"), images=np.stack(imgs),
                                    labels=np.array(labels, np.float32),
                                    columns=np.array(["u", "v", "radius", "visible", "in_frame", "camera", "brightness"]))
                print(f"shard {k_shard}: {len(imgs)} images, {done}/{a.images}", flush=True)
                imgs, labels, k_shard = [], [], k_shard + 1
            if done >= a.images:
                break
        r.close()


if __name__ == "__main__":
    main()
