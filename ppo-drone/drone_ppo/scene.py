"""Render-only dressing and capture helpers. Physics is identical with or without it.

The scene is deliberately minimal: a flat light-grey floor, a neutral grey background, one green target sphere
(visual only, no collider), a soft dome light and one directional light.
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

H_APERTURE = 20.955
CAMERAS = {
    # setup shot: drone near the floor, target sphere at 1 m
    "setup": {"eye": (1.55, -1.25, 0.95), "target": (0.0, 0.0, 0.55), "focal": 30.0},
    # playback / trained shot: around the hover point
    "hover": {"eye": (1.35, -1.1, 1.2), "target": (0.05, 0.0, 0.9), "focal": 30.0},
    "wide": {"eye": (3.2, -2.6, 1.9), "target": (0.0, 0.0, 0.8), "focal": 24.0},
}


def dress_scene(origin, floor_grey: float = 0.62):
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, "/World/Looks/Floor")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Floor/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(floor_grey, floor_grey, floor_grey))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    s, (ox, oy, oz) = 200.0, origin
    floor.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(ox - s, oy - s, oz), Gf.Vec3f(ox + s, oy - s, oz),
                                          Gf.Vec3f(ox + s, oy + s, oz), Gf.Vec3f(ox - s, oy + s, oz)]))
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    floor.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat)

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")          # neutral grey background + soft fill
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.86, 0.88))
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 0.0, 140.0))

    marker = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/Target",
        markers={"target": sim_utils.SphereCfg(
            radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.25),
                                                                      roughness=0.35))}))
    return marker


def make_renderable(stage, path: str) -> int:
    """Instanced meshes under physics-driven links do not render on DGX Spark (Isaac Sim 5.1); de-instance them."""
    from pxr import Usd
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.IsInstanceable():
            p.SetInstanceable(False)
            n += 1
    return n


def define_camera(stage, path, eye, target, focal, width, height, origin=(0.0, 0.0, 0.0)):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(H_APERTURE * height / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    e, t = np.asarray(eye) + origin, np.asarray(target) + origin
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*e), Gf.Vec3d(*t), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return view


def project(view, focal, width, height, p):
    pc = view.Transform(Gf.Vec3d(*p))
    if pc[2] >= -1e-6:
        return None, False
    f = width * focal / H_APERTURE
    u, v = width / 2 + f * pc[0] / -pc[2], height / 2 - f * pc[1] / -pc[2]
    return (u, v), (0 <= u < width and 0 <= v < height)


def set_quality(mode="pt", spp_total=256, spp_frame=16):
    import carb
    s = carb.settings.get_settings()
    if mode == "pt":
        s.set("/rtx/rendermode", "PathTracing")
        s.set("/rtx/pathtracing/spp", spp_frame)
        s.set("/rtx/pathtracing/totalSpp", spp_total)
        s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        s.set("/rtx/pathtracing/adaptiveSampling/enabled", False)
    else:
        s.set("/rtx/rendermode", "RaytracedLighting")
    s.set("/rtx/post/histogram/enabled", True)


class Grabber:
    def __init__(self, cams: dict, width: int, height: int):
        import omni.replicator.core as rep
        self.annot = {}
        for name, path in cams.items():
            rp = rep.create.render_product(path, (width, height))
            a = rep.AnnotatorRegistry.get_annotator("rgb")
            a.attach([rp])
            self.annot[name] = a

    def get(self, name):
        d = self.annot[name].get_data()
        if d is None or getattr(d, "size", 0) == 0:
            return None
        return np.asarray(d)[..., :3].copy()
