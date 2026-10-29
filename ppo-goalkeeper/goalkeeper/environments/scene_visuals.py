"""Render-only dressing for the field: mown turf, futsal markings, sky and sun. No physics here.

Markings follow the futsal Laws of the Game: 8 cm lines; the penalty area is two 6 m quarter circles centred on
the outside of each post, joined by a 3.16 m line; penalty mark 6 m and second penalty mark 10 m from the goal.
"""
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from ..physics import field as F

LINE_W = 0.08
TILE = 1.6                 # turf texture repeat, m
STRIPE = 2.5               # mowing stripe width, m


def _textured_material(stage, path, grass_dir: Path, tint=(1, 1, 1), bias=(0, 0, 0), roughness_scale=1.0):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    st = UsdShade.Shader.Define(stage, f"{path}/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def tex(name, file, cs, scale, b):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(grass_dir / file))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*b))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        return t

    col = tex("Color", "Grass005_2K-JPG_Color.jpg", "sRGB", (*tint, 1), (*bias, 0))
    nrm = tex("Normal", "Grass005_2K-JPG_NormalGL.jpg", "raw", (2, 2, 2, 1), (-1, -1, -1, 0))
    rgh = tex("Rough", "Grass005_2K-JPG_Roughness.jpg", "raw", (roughness_scale,) * 4, (0, 0, 0, 0))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(col.ConnectableAPI(), "rgb")
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm.ConnectableAPI(), "rgb")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rgh.ConnectableAPI(), "r")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.25)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _mesh(stage, path, verts, faces, origin, material, z=0.0):
    v = np.asarray(verts, dtype=np.float64)
    pts = np.column_stack([v[:, 0] + origin[0], v[:, 1] + origin[1], np.full(len(v), z + origin[2])])
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    f = np.asarray(faces, dtype=np.int32)
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), f.shape[1], dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(v)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy((v[:, :2] / TILE).astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def _rect(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], [(0, 1, 2, 3)]


def _ribbon(points, width):
    """Quad strip of constant width along a polyline."""
    p = np.asarray(points, dtype=np.float64)
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.column_stack([-d[:, 1], d[:, 0]]) * width / 2
    verts = np.concatenate([p - n, p + n])
    k = len(p)
    faces = [(i, i + 1, k + i + 1, k + i) for i in range(k - 1)]
    return verts, faces


def _disc(cx, cy, r, seg=32):
    verts = [(cx, cy)] + [(cx + r * math.cos(2 * math.pi * i / seg), cy + r * math.sin(2 * math.pi * i / seg))
                          for i in range(seg)]
    faces = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
    return verts, faces


def _merge(parts):
    verts, faces, off = [], [], 0
    for v, f in parts:
        verts.extend(list(map(tuple, np.asarray(v)[:, :2])))
        faces.extend([tuple(i + off for i in face) for face in f])
        off += len(v)
    return verts, faces


def field_markings():
    hw_out = F.GOAL_HALF_WIDTH + F.POST_WIDTH          # outside of each post
    r = 6.0
    tri, quad = [], []
    quad.append(_ribbon([(LINE_W / 2, -10.0), (LINE_W / 2, 10.0)], LINE_W))           # goal line
    for s in (-1, 1):
        arc = [(r * math.cos(t), s * (hw_out + r * math.sin(t))) for t in np.linspace(0, math.pi / 2, 64)]
        quad.append(_ribbon(arc, LINE_W))
        quad.append(_ribbon([(0.0, s * 10.0), (40.0, s * 10.0)], LINE_W))             # touchlines
        corner = [(0.25 * math.sin(t), s * (10.0 - 0.25 * math.cos(t))) for t in np.linspace(0, math.pi / 2, 16)]
        quad.append(_ribbon(corner, LINE_W))
    quad.append(_ribbon([(r, -hw_out), (r, hw_out)], LINE_W))                         # 3.16 m line
    quad.append(_ribbon([(20.0, -10.0), (20.0, 10.0)], LINE_W))                       # halfway line
    centre = [(20.0 + 3.0 * math.cos(t), 3.0 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 128)]
    quad.append(_ribbon(centre, LINE_W))
    tri.append(_disc(F.PENALTY_X, 0.0, 0.10))
    tri.append(_disc(10.0, 0.0, 0.10))
    return _merge(quad), _merge(tri)


def build_field(stage, asset_paths: dict, origin=(0.0, 0.0, 0.0), sky: str = "puresky"):
    """sky: "puresky" (sky-only HDRI, field to the horizon), "studio" (plain grey dome + sun),
    or "photo" (HDRI photographed on a real training pitch)."""
    grass = Path(asset_paths["grass_dir"])
    root = "/World/Field"
    UsdGeom.Xform.Define(stage, root)
    light = _textured_material(stage, f"{root}/Looks/TurfLight", grass, tint=(0.95, 1.0, 0.9))
    dark = _textured_material(stage, f"{root}/Looks/TurfDark", grass, tint=(0.72, 0.82, 0.68))
    paint = _textured_material(stage, f"{root}/Looks/LinePaint", grass, tint=(0.25, 0.25, 0.25),
                               bias=(0.68, 0.68, 0.66), roughness_scale=1.0)
    x0, x1, y0, y1 = -14.0, 46.0, -24.0, 24.0
    xs = np.arange(x0, x1 + 1e-6, STRIPE)
    for i in range(len(xs) - 1):
        v, f = _rect(xs[i], xs[i + 1], y0, y1)
        _mesh(stage, f"{root}/Stripe_{i:02d}", v, f, origin, light if i % 2 else dark)
    big = 3000.0
    for name, (a, b, c, d) in {"ApronN": (x0, x1, y1, big), "ApronS": (x0, x1, -big, y0),
                               "ApronE": (x1, big, -big, big), "ApronW": (-big, x0, -big, big)}.items():
        v, f = _rect(a, b, c, d)
        _mesh(stage, f"{root}/{name}", v, f, origin, dark)
    (qv, qf), (tv, tf) = field_markings()
    _mesh(stage, f"{root}/Lines", qv, qf, origin, paint, z=0.003)
    _mesh(stage, f"{root}/Marks", tv, tf, origin, paint, z=0.003)

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    if sky == "studio":
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.86, 0.88, 0.9))
    else:
        dome.CreateIntensityAttr(1000.0)
        dome.CreateTextureFileAttr(str(asset_paths["hdri_puresky" if sky == "puresky" else "hdri"]))
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    if sky != "puresky":          # the pure-sky HDRI carries its own sun
        s = UsdLux.DistantLight.Define(stage, "/World/Sun")
        s.CreateIntensityAttr(1800.0)
        s.CreateAngleAttr(2.5)
        s.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
        # from behind the kicker's left shoulder, 40 degrees up: shadows fall toward the goal
        UsdGeom.Xformable(s).AddRotateXYZOp().Set(Gf.Vec3f(45.0, 0.0, 115.0))
    return root
