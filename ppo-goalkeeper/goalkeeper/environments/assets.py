"""Fetch the open-licence assets and turn them into USD files for the environment.

Downloaded (pinned URLs, licences in assets/ATTRIBUTION.md):
  goal   "Porteria Futbol Sala 2" futsal goal, Sketchfab CC-BY 4.0, via the Objaverse mirror on Hugging Face
  ball   Poly Haven "football", CC0, native USD + textures
  turf   ambientCG Grass005, CC0
  sky    Poly Haven "suburban_football_field" HDRI, CC0
The goalkeeper robot is NVIDIA's Clearpath Jackal from the Isaac Sim asset library (loaded at runtime).

Written:
  goal_visual.usdc    goal frame + net meshes, moved into the field frame (goal line at x = 0, opening toward +x)
  goal_physics.usda   static colliders: posts, crossbar, and soft net panels that catch the ball
  ball.usda           rigid-body ball: sphere collider, hollow-shell inertia, rolling damping, the football mesh
"""
import os
import urllib.request
from pathlib import Path

import numpy as np

from ..physics import field as F

GOAL_GLB = ("https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-026/"
            "2539585ea28a4b6e9d0c14d13c593c11.glb")
PH_MODEL = "https://dl.polyhaven.org/file/ph-assets/Models"
BALL_FILES = {
    "football_2k.usdc": f"{PH_MODEL}/usd/2k/football/football_2k.usdc",
    "textures/football_diff_2k.jpg": f"{PH_MODEL}/jpg/2k/football/football_diff_2k.jpg",
    "textures/football_rough_2k.exr": f"{PH_MODEL}/exr/2k/football/football_rough_2k.exr",
    "textures/football_nor_gl_2k.exr": f"{PH_MODEL}/exr/2k/football/football_nor_gl_2k.exr",
}
GRASS_ZIP = "https://ambientcg.com/get?file=Grass005_2K-JPG.zip"
HDRI = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/8k/suburban_football_field_8k.hdr"

# Goal GLB (glTF, y-up) -> field frame. Measured from the mesh: post back faces at x=-0.045, ground at y=-0.04,
# the posts' inner edges at z=-1.626 and 1.306 (2.932 m apart). Scale so the inside width is exactly 3.0 m.
GLB_ORIGIN = np.array([-0.045, -0.04, -0.16])
GLB_SCALE = 2 * F.GOAL_HALF_WIDTH / 2.932


def _get(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ppo-goalkeeper/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(str(dest) + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(str(dest) + ".part", dest)
    return dest


def download(cache: Path, with_render_assets: bool = True):
    cache = Path(cache)
    _get(GOAL_GLB, cache / "goal" / "futsal_goal.glb")
    for rel, url in BALL_FILES.items():
        _get(url, cache / "football" / rel)
    if with_render_assets:
        z = _get(GRASS_ZIP, cache / "grass" / "Grass005_2K-JPG.zip")
        import zipfile
        with zipfile.ZipFile(z) as zf:
            for name in ("Color", "NormalGL", "Roughness"):
                member = f"Grass005_2K-JPG_{name}.jpg"
                if not (cache / "grass" / member).exists():
                    zf.extract(member, cache / "grass")
        _get(HDRI, cache / "sky" / Path(HDRI).name)
    return cache


def _preview_material(stage, path, color, roughness, metallic=0.0):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def build_goal_visual(glb: Path, out: Path):
    """Convert the futsal goal GLB into a z-up USD in the field frame, all white (frame paint + net)."""
    import trimesh
    from pxr import Usd, UsdGeom, UsdShade, Vt
    scene = trimesh.load(str(glb))
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Goal")
    stage.SetDefaultPrim(root.GetPrim())
    paint = _preview_material(stage, "/Goal/Looks/WhitePaint", (0.86, 0.86, 0.85), 0.32)
    net_mat = _preview_material(stage, "/Goal/Looks/Net", (0.80, 0.80, 0.78), 0.85)
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        parts.append(m)
    net_idx = int(np.argmax([len(m.faces) for m in parts]))   # the net is by far the densest mesh
    for i, m in enumerate(parts):
        d = m.vertices - GLB_ORIGIN
        pts = GLB_SCALE * np.stack([d[:, 0], -d[:, 2], d[:, 1]], axis=1)
        nrm = np.asarray(m.vertex_normals)
        nrm = np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], axis=1)
        name = "Net" if i == net_idx else f"Frame_{i}"
        mesh = UsdGeom.Mesh.Define(stage, f"/Goal/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m.faces.astype(np.int32).ravel()))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(net_mat if i == net_idx else paint)
    stage.GetRootLayer().Save()
    return out


def _box_collider(stage, path, center, size, material=None):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)        # invisible in renders
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _physics_material(stage, path, static_f, dynamic_f, restitution):
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static_f)
    api.CreateDynamicFrictionAttr(dynamic_f)
    api.CreateRestitutionAttr(restitution)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateRestitutionCombineModeAttr("max" if restitution > 0.3 else "min")
    return mat


def build_goal_physics(out: Path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/GoalPhysics")
    stage.SetDefaultPrim(root.GetPrim())
    metal = _physics_material(stage, "/GoalPhysics/Metal", 0.4, 0.3, 0.5)
    net = _physics_material(stage, "/GoalPhysics/Netting", 0.9, 0.9, 0.02)
    hw, pd, pw, h = F.GOAL_HALF_WIDTH, F.POST_DEPTH, F.POST_WIDTH, F.GOAL_HEIGHT
    top = 2.128
    for s, name in ((1, "PostL"), (-1, "PostR")):
        _box_collider(stage, f"/GoalPhysics/{name}", (pd / 2, s * (hw + pw / 2), top / 2), (pd, pw, top), metal)
    _box_collider(stage, "/GoalPhysics/Crossbar", (pd / 2, 0.0, (h + top) / 2), (pd, 2 * (hw + pw), top - h), metal)
    nd = F.NET_DEPTH
    _box_collider(stage, "/GoalPhysics/NetBack", (-nd - 0.02, 0.0, 1.0), (0.04, 2 * hw + 0.3, 2.0), net)
    for s, name in ((1, "NetSideL"), (-1, "NetSideR")):
        _box_collider(stage, f"/GoalPhysics/{name}", (-nd / 2, s * (hw + pw + 0.02), 1.0), (nd, 0.04, 2.0), net)
    _box_collider(stage, "/GoalPhysics/NetTop", (-nd / 2, 0.0, top + 0.02), (nd, 2 * hw + 0.3, 0.04), net)
    stage.GetRootLayer().Save()
    return out


def build_ball(football_usdc: Path, out: Path, lin_damping=0.08, ang_damping=0.3):
    """Rigid-body ball: hollow-shell inertia (2/3 m r^2), sphere collider, the Poly Haven football as the visual."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    ball = UsdGeom.Xform.Define(stage, "/Ball")
    stage.SetDefaultPrim(ball.GetPrim())
    p = ball.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    mass = UsdPhysics.MassAPI.Apply(p)
    mass.CreateMassAttr(F.BALL_MASS)
    i = 2.0 / 3.0 * F.BALL_MASS * F.BALL_RADIUS ** 2
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(i, i, i))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
    rb.CreateLinearDampingAttr(lin_damping)
    rb.CreateAngularDampingAttr(ang_damping)
    rb.CreateMaxAngularVelocityAttr(20000.0)          # deg/s; the default would cap a fast-rolling ball's spin
    rb.CreateMaxDepenetrationVelocityAttr(3.0)
    rb.CreateEnableCCDAttr(True)
    mat = _physics_material(stage, "/Ball/Leather", 0.6, 0.5, 0.6)
    col = UsdGeom.Sphere.Define(stage, "/Ball/Collision")
    col.CreateRadiusAttr(F.BALL_RADIUS)
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim()).CreateContactOffsetAttr(0.02)
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    vis = UsdGeom.Xform.Define(stage, "/Ball/Visual")
    vis.GetPrim().GetReferences().AddReference(os.path.relpath(football_usdc, out.parent), Sdf.Path("/football"))
    # the inflated mesh is centred at (0.15, 0, 0.111) in the source file; the deflated twin is not used
    s = F.BALL_RADIUS / 0.11097
    vis.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    vis.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, -0.11097))
    stage.OverridePrim("/Ball/Visual/football_deflated").SetActive(False)
    stage.GetRootLayer().Save()
    return out


def build_all(cache: Path, out_dir: Path, with_render_assets: bool = True):
    cache, out_dir = Path(cache), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    download(cache, with_render_assets)
    paths = {
        "goal_visual": build_goal_visual(cache / "goal" / "futsal_goal.glb", out_dir / "goal_visual.usdc"),
        "goal_physics": build_goal_physics(out_dir / "goal_physics.usda"),
        "ball": build_ball(cache / "football" / "football_2k.usdc", out_dir / "ball.usda"),
    }
    if with_render_assets:
        paths["grass_dir"] = cache / "grass"
        paths["hdri"] = cache / "sky" / Path(HDRI).name
    return {k: str(v) for k, v in paths.items()}
