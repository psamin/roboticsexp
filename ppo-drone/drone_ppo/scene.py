"""Render-only dressing and capture helpers. Physics is identical with or without it.

The scene is deliberately minimal: a flat light-grey floor, a neutral grey background, one green target sphere
(visual only, no collider), a soft dome light and one directional light.
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

H_APERTURE = 20.955
CAMERAS = {
    # setup shot: drone near the floor, target sphere at 1 m
    "setup": {"eye": (1.7, -1.35, 0.66), "target": (0.0, 0.02, 0.63), "focal": 22.0},
    # playback / trained shot: around the hover point
    "hover": {"eye": (1.2, -0.62, 1.32), "target": (0.22, 0.14, 1.08), "focal": 35.0},
    "wide": {"eye": (3.2, -2.6, 1.9), "target": (0.0, 0.0, 0.8), "focal": 24.0},
}


def dress_scene(origin, style: str = "isaac", floor_grey: float = 0.8):
    """style: "isaac" (Isaac Lab's stock grid floor + neutral dome, as in the reference quadcopter demo),
    "white" (white floor and backdrop), or "studio" (lit grey floor, darker grey backdrop)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if style in ("isaac", "white"):
        if style == "white":
            _floor(stage, origin, 0.95)
        dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
        dome.CreateIntensityAttr(2000.0 if style == "isaac" else 2600.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75) if style == "isaac" else Gf.Vec3f(1.0, 1.0, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        key.CreateIntensityAttr(1500.0)
        key.CreateAngleAttr(3.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(30.0, 0.0, 140.0))
        return TargetSphere(stage)
    _floor(stage, origin, floor_grey)
    return _studio_lights(stage)


def _floor(stage, origin, grey):
    mat = UsdShade.Material.Define(stage, "/World/Looks/Floor")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Floor/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(grey, grey, grey))
    # roughness 0 with no specular: RTX's rough-diffuse model darkens the far floor at grazing angles (measured:
    # roughness 0.9 -> far floor at 39 of 255 vs background 101; roughness 0 -> 102, a seamless horizon)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
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


def _studio_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")          # mid-grey backdrop, darker than the lit floor
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.6, 0.61, 0.64))
    key = UsdLux.DistantLight.Define(stage, "/World/Key")         # near-overhead: a crisp shadow under the drone
    key.CreateIntensityAttr(2800.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(15.0, 0.0, 140.0))
    fill = UsdLux.DistantLight.Define(stage, "/World/Fill")     # soft fill from the camera side (+x, -y, above)
    fill.CreateIntensityAttr(900.0)
    fill.CreateAngleAttr(8.0)
    UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(52.5, 0.0, 39.8))

    return TargetSphere(stage)


class TargetSphere:
    """One green sphere at environment 0's target. A plain prim (no instancer, no collider): visual only."""

    def __init__(self, stage, radius: float = 0.015):
        sphere = UsdGeom.Sphere.Define(stage, "/World/Target")
        sphere.CreateRadiusAttr(radius)
        mat = UsdShade.Material.Define(stage, "/World/Looks/Target")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/Target/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.72, 0.22))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)
        UsdGeom.PrimvarsAPI(sphere).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        self.op = UsdGeom.Xformable(sphere).AddTranslateOp()

    def visualize(self, translations):
        self.op.Set(Gf.Vec3d(*[float(x) for x in translations[0]]))


def make_renderable(stage, path: str, dark: bool = True) -> int:
    """Instanced meshes under physics-driven links do not render on DGX Spark (Isaac Sim 5.1); de-instance them.
    dark: matte black plastic (as on a real Crazyflie 2.x) so the drone stands out against light backgrounds."""
    from pxr import Usd
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.IsInstanceable():
            p.SetInstanceable(False)
            n += 1
    if dark:
        mat = UsdShade.Material.Define(stage, "/World/Looks/DroneBlack")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/DroneBlack/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.025, 0.025, 0.028))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
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


def set_quality(mode="pt", spp_total=256, spp_frame=16, auto_exposure=True):
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
    s.set("/rtx/post/histogram/enabled", auto_exposure)


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
