#!/bin/bash
# Can this node run the two simulators? RoboTwin 2.0 needs SAPIEN 3.0.0b1 (Vulkan, x86_64 only);
# ASPIRE's LIBERO-Pro needs MuJoCo with EGL. Prints PASS/FAIL per check.
#SBATCH -J aa-probe
#SBATCH -p general
#SBATCH -t 0:20:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/probe-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh

echo "== system"
grep PRETTY_NAME /etc/os-release; (ldd --version 2>&1 || true) | sed -n 1p; nproc; df -h /dev/shm | sed -n '$p'
for t in conda micromamba nvcc gcc g++ ffmpeg vulkaninfo; do printf '%s: %s\n' "$t" "$(command -v $t || echo missing)"; done
vulkan_check

"$UV" venv -q --python 3.10 "$WORK/venv"
VIRTUAL_ENV=$WORK/venv "$UV" pip install -q setuptools==69.5.1 sapien==3.0.0b1 numpy==1.26.4 mujoco==3.3.0
PY=$WORK/venv/bin/python

echo "== sapien render"
$PY - <<'PY' && echo "SAPIEN: PASS" || echo "SAPIEN: FAIL"
import math, sapien, numpy as np
print("device", getattr(sapien.render, "get_device_summary", lambda: "n/a")())
scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, 1, -1], [0.7, 0.7, 0.7])
scene.add_ground(0)
b = scene.create_actor_builder()
b.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[1.0, 0.0, 0.0])
box = b.build_kinematic(name="box")
box.set_pose(sapien.Pose([0, 0, 0.1]))
th = math.atan2(0.5, 1.0)                                   # camera +x forward; pitch down at the box
cam = scene.add_camera("cam", 320, 240, 1.0, 0.01, 10.0)
cam.entity.set_pose(sapien.Pose([-1.0, 0, 0.6], [math.cos(th / 2), 0, math.sin(th / 2), 0]))
def shot():
    scene.update_render(); cam.take_picture(); return cam.get_picture("Color")[..., :3]
a = shot(); box.set_pose(sapien.Pose([0, 0, -5])); c = shot()
print("std", float(a.std()), "red-ish px", int(((a[..., 0] > 0.4) & (a[..., 1] < 0.2)).sum()),
      "box diff", float(np.abs(a - c).mean()))
assert a.std() > 0.01 and np.abs(a - c).mean() > 1e-3, "blank or box not rendered"
PY

echo "== mujoco egl render"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY - <<'PY' && echo "MUJOCO_EGL: PASS" || echo "MUJOCO_EGL: FAIL"
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/>'
    '<geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 240, 320); r.update_scene(d); img = r.render()
print("shape", img.shape, "mean", float(img.mean()))
assert img.std() > 0, "blank frame"
PY
echo "== done $(date)"
