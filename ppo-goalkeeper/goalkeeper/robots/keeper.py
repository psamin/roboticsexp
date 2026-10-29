"""The goalkeeper: NVIDIA's Clearpath Jackal asset with a blocking panel.

The Jackal faces along +y, so driving forward/backward moves it across the goal mouth. Its four wheels are
velocity-driven and move the robot through tyre-ground friction (no teleporting, no rail). The panel is a separate
rigid link on a fixed joint to base_link, on the robot's right side (world +x, facing the kicker); keeping it a
separate link means its collider does not change the Jackal's own mass distribution.
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

from ..physics import field as F

JACKAL_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd"
WHEEL_RADIUS = 0.098           # Jackal tyre radius (m)
TRACK_WIDTH = 0.37             # wheel centre-to-centre across the robot (m)
SPAWN_Z = 0.064                # base_link height when standing on its wheels (measured: 0.0635 m)
YAW_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))   # (w, x, y, z): forward axis along world +y

PANEL_SIZE = (0.70, 0.012, 0.34)   # along robot x (= world y), thickness, height
PANEL_BOTTOM = 0.035               # gap under the panel, well below the ball radius
PANEL_MASS = 1.6


def keeper_cfg(prim_path: str = "/World/envs/env_.*/Keeper") -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=JACKAL_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(F.KEEPER_X, 0.0, SPAWN_Z), rot=YAW_90),
        actuators={
            "wheels": ImplicitActuatorCfg(joint_names_expr=[".*wheel.*"], stiffness=0.0, damping=20.0,
                                          effort_limit_sim=20.0, velocity_limit_sim=30.0),
        },
    )


def make_visuals_renderable(stage, keeper_path: str):
    """Instanced meshes under physics-driven links do not render on DGX Spark (Isaac Sim 5.1); de-instance them."""
    from pxr import Usd
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(keeper_path)):
        if p.IsInstanceable():
            p.SetInstanceable(False)
            n += 1
    return n


def set_tyre_friction(stage, keeper_path: str, static: float, dynamic: float):
    """Rubber-on-turf friction for the wheel material shipped with the Jackal asset. Returns the original values."""
    from pxr import Usd
    before = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(keeper_path)):
        if p.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(p)
            px = PhysxSchema.PhysxMaterialAPI(p) if p.HasAPI(PhysxSchema.PhysxMaterialAPI) else None
            before[str(p.GetPath())] = (m.GetStaticFrictionAttr().Get(), m.GetDynamicFrictionAttr().Get(),
                                        px.GetFrictionCombineModeAttr().Get() if px else None)
            m.CreateStaticFrictionAttr().Set(static)
            m.CreateDynamicFrictionAttr().Set(dynamic)
            PhysxSchema.PhysxMaterialAPI.Apply(p).CreateFrictionCombineModeAttr().Set("average")
    return before


def _material(stage, path, color, roughness, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _box(stage, path, center, size, material=None, purpose=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if purpose:
        cube.CreatePurposeAttr(purpose)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def add_panel(stage, keeper_path: str, base_body: str = "base_link"):
    """Author the panel link + fixed joint under a spawned Jackal (before environments are cloned).

    Positions are in base_link's frame: x forward (world +y), y left (world -x), z up. The panel's front face sits
    PANEL_OFFSET_X in front of base_link along world +x, i.e. at robot y = -PANEL_OFFSET_X.
    """
    base = f"{keeper_path}/{base_body}"
    t, h = PANEL_SIZE[1], PANEL_SIZE[2]
    z_c = PANEL_BOTTOM + h / 2 - SPAWN_Z                 # panel centre height relative to base_link
    y_c = -(F.PANEL_OFFSET_X - t / 2)
    xf_cache = UsdGeom.XformCache()
    base_local = xf_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(base)) * \
        xf_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(keeper_path)).GetInverse()

    panel = UsdGeom.Xform.Define(stage, f"{keeper_path}/goal_panel")
    offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, y_c, z_c))
    panel.AddTransformOp().Set(offset * base_local)
    p = panel.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(PANEL_MASS)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

    sheet_path = f"{keeper_path}/goal_panel/Looks/Polycarbonate"
    glass = sim_utils.GlassMdlCfg(glass_color=(0.86, 0.9, 0.94), frosting_roughness=0.03, thin_walled=True,
                                  glass_ior=1.58)
    glass.func(sheet_path, glass)
    alu = _material(stage, f"{keeper_path}/goal_panel/Looks/Aluminium", (0.62, 0.63, 0.65), 0.35, 0.9)
    w = PANEL_SIZE[0]
    _box(stage, f"{keeper_path}/goal_panel/Sheet", (0, 0, 0), (w, t, h))
    sim_utils.bind_visual_material(f"{keeper_path}/goal_panel/Sheet", sheet_path)
    rail = 0.022
    for s in (-1, 1):   # aluminium extrusion frame around the sheet
        _box(stage, f"{keeper_path}/goal_panel/Upright{'L' if s > 0 else 'R'}", (s * (w / 2 + rail / 2), 0, 0),
             (rail, rail, h + rail), alu)
    _box(stage, f"{keeper_path}/goal_panel/TopRail", (0, 0, h / 2 + rail / 2), (w + 2 * rail, rail, rail), alu)
    # brackets from the panel back to the top plate (visual only)
    for s in (-1, 1):
        _box(stage, f"{keeper_path}/goal_panel/Bracket{'L' if s > 0 else 'R'}",
             (s * 0.16, 0.10, 0.27 - h / 2), (0.03, 0.20, 0.012), alu)
    # 4 cm thick collider whose front face matches the sheet's front face (panel-frame +y points back to the robot)
    col = _box(stage, f"{keeper_path}/goal_panel/Collision", (0, 0.02 - t / 2, 0), (w + 2 * rail, 0.04, h),
               purpose=UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim()).CreateContactOffsetAttr(0.02)

    j = UsdPhysics.FixedJoint.Define(stage, f"{keeper_path}/goal_panel_joint")
    j.CreateBody0Rel().SetTargets([Sdf.Path(base)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"{keeper_path}/goal_panel")])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, y_c, z_c))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return f"{keeper_path}/goal_panel"
