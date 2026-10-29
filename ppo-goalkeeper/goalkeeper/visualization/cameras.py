"""Camera presets, USD camera authoring, and pinhole projection for checking what a render shows."""
import math

import numpy as np
from pxr import Gf, UsdGeom

H_APERTURE = 20.955   # mm, USD default

PRESETS = {
    # behind the penalty mark, ball in the foreground, keeper centred in front of the goal
    "hero": {"eye": (9.8, 0.55, 1.05), "target": (0.6, 0.0, 0.6), "focal": 25.0},
    # elevated, diagonal
    "angle": {"eye": (8.2, -5.0, 2.6), "target": (2.8, 0.2, 0.4), "focal": 22.0},
    # playback views
    "front": {"eye": (9.4, 0.0, 1.35), "target": (1.0, 0.0, 0.55), "focal": 20.0},
    "side": {"eye": (3.2, -7.4, 1.6), "target": (2.4, 0.2, 0.4), "focal": 18.0},
}


def define_camera(stage, path, eye, target, focal, width, height, origin=(0.0, 0.0, 0.0)):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(H_APERTURE * height / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    e = Gf.Vec3d(*(np.asarray(eye) + origin)); t = Gf.Vec3d(*(np.asarray(target) + origin))
    view = Gf.Matrix4d().SetLookAt(e, t, Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return cam, view


def project(view: Gf.Matrix4d, focal, width, height, p_world):
    """Pixel (u, v) of a world point, and whether it is in front of the camera and inside the frame."""
    pc = view.Transform(Gf.Vec3d(*p_world))
    if pc[2] >= -1e-6:
        return None, False
    f_px = width * focal / H_APERTURE
    u = width / 2 + f_px * pc[0] / -pc[2]
    v = height / 2 - f_px * pc[1] / -pc[2]
    return (u, v), (0 <= u < width and 0 <= v < height)


def angular_size_px(view, focal, width, p_world, radius):
    pc = view.Transform(Gf.Vec3d(*p_world))
    return width * focal / H_APERTURE * radius / max(-pc[2], 1e-6)


def hfov_deg(focal):
    return 2 * math.degrees(math.atan(H_APERTURE / 2 / focal))
