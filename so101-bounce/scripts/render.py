"""Render a saved rollout (scripts/inspect_policy.py --save traj.npz) with the MuJoCo SO-101 + paddle + ball model.

  python scripts/render.py traj.npz --env 0 --out bounce.mp4 [--view side|overhead|both]

Kinematic replay only: the joint angles and ball positions come from the training simulator.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bounce.robot import GRIPPER_CLOSED_ON_HANDLE, build_spec


def scene(look=None):
    """MuJoCo SO-101 + ball for rendering. With `look` (bounce.visual.sample_look), the paddle is drawn as a
    procedural mesh on a mocap body posed from the simulator's paddle frame, and the physics paddle boxes are hidden."""
    import mujoco
    from bounce.visual import add_paddle, color_arm
    spec = build_spec(with_ball=True)
    light = look["light"] if look else 0.7
    spec.worldbody.add_light(pos=[0.3, -0.6, 1.4], dir=[-0.1, 0.5, -1], diffuse=[light] * 3)
    spec.worldbody.add_light(pos=[-0.4, 0.6, 1.2], dir=[0.4, -0.4, -1], diffuse=[light / 2] * 3)
    floor_rgb = look["floor_rgb"] if look else [0.93, 0.93, 0.93]
    floor = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=list(floor_rgb) + [1])
    floor.contype = floor.conaffinity = 0
    if look:
        for name in ("blade", "handle"):
            spec.geom(name).rgba = [0, 0, 0, 0]
        add_paddle(spec, spec.worldbody.add_body(name="paddle_vis", mocap=True), look)
        spec.geom("ball").rgba = list(look["ball_rgb"]) + [1]
        color_arm(spec, look["arm_rgb"])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    return spec.compile()


def set_paddle(m, d, o, R):
    """Pose the visual paddle (if any) at paddle origin o (3,) and rotation R (3, 3)."""
    import mujoco
    bid = m.body("paddle_vis").id if "paddle_vis" in [m.body(i).name for i in range(m.nbody)] else -1
    if bid < 0:
        return
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).flatten())
    mid = m.body_mocapid[bid]
    d.mocap_pos[mid], d.mocap_quat[mid] = o, q


def main():
    import imageio.v2 as iio
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--out", default="bounce.mp4")
    ap.add_argument("--view", choices=("side", "overhead", "both"), default="side")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    T = np.load(a.traj)
    q, ball, dt = T["q"][:, a.env], T["ball"][:, a.env], float(T["dt"])
    m = scene()
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.height, a.width)
    cams = {"side": dict(lookat=[0.25, 0, 0.32], distance=0.95, azimuth=-90, elevation=-8),
            "overhead": dict(lookat=[0.3, 0, 0.3], distance=1.0, azimuth=180, elevation=-58)}
    views = ["side", "overhead"] if a.view == "both" else [a.view]
    writer = iio.get_writer(a.out, fps=round(1 / dt), codec="libx264", quality=8, macro_block_size=8,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])
    bid = m.body("ball").id
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]
    for k in range(len(q)):
        d.qpos[:5] = q[k]
        d.qpos[5] = GRIPPER_CLOSED_ON_HANDLE
        d.qpos[qadr:qadr + 3] = ball[k]
        d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        frames = []
        for v in views:
            cam = mujoco.MjvCamera()
            for key, val in cams[v].items():
                setattr(cam, key, val)
            r.update_scene(d, cam)
            frames.append(r.render())
        writer.append_data(np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0])
    writer.close()
    print("wrote", a.out, len(q), "frames")


if __name__ == "__main__":
    main()
