"""How things look: procedural table-tennis paddles, ball and table colours, lighting, for renders and for the
ball detector's synthetic training images. Physics never reads any of this.

Blade: a superellipse outline (the same one the impact model uses) narrowing slightly toward the handle, a wood core
(5-7 mm) with a rubber+sponge layer on each face (1.5-2.2 mm sponge plus ~0.5 mm topsheet). Handle: straight, flared
or anatomic, 90-110 mm long, 22-30 mm wide. Colours follow what is sold: black and red rubber (ITTF since forever),
plus blue, green, pink and violet (allowed on one side since 2021); orange or white balls; light to dark wood.
"""
import colorsys
import json
from pathlib import Path

import numpy as np

RUBBER = {"black": ((0.0, 1.0), (0.0, 0.15), (0.03, 0.14)), "red": ((0.97, 1.03), (0.75, 1.0), (0.45, 0.9)),
          "blue": ((0.58, 0.64), (0.6, 1.0), (0.4, 0.85)), "green": ((0.30, 0.40), (0.6, 1.0), (0.4, 0.85)),
          "pink": ((0.88, 0.95), (0.45, 0.8), (0.75, 1.0)), "violet": ((0.74, 0.80), (0.5, 0.9), (0.4, 0.8)),
          "orange": ((0.03, 0.08), (0.8, 1.0), (0.7, 1.0))}
RUBBER_P = {"black": 0.4, "red": 0.35, "blue": 0.07, "green": 0.05, "pink": 0.05, "violet": 0.04, "orange": 0.04}
BALL = {"orange": ((0.04, 0.10), (0.75, 1.0), (0.8, 1.0)), "white": ((0.0, 1.0), (0.0, 0.08), (0.85, 1.0)),
        "yellow": ((0.12, 0.16), (0.6, 0.9), (0.85, 1.0))}
BALL_P = {"orange": 0.48, "white": 0.48, "yellow": 0.04}
WOOD = ((0.06, 0.10), (0.35, 0.65), (0.35, 0.9))
# 3D-printed arm parts: mostly white (the user's arm), plus other common filament colours; servos stay black
ARM = {"white": ((0.0, 1.0), (0.0, 0.08), (0.82, 0.98)), "yellow": ((0.12, 0.16), (0.7, 0.95), (0.85, 1.0)),
       "black": ((0.0, 1.0), (0.0, 0.1), (0.08, 0.2)), "grey": ((0.0, 1.0), (0.0, 0.05), (0.4, 0.7)),
       "orange": ((0.05, 0.09), (0.7, 1.0), (0.8, 1.0)), "blue": ((0.55, 0.65), (0.5, 0.9), (0.4, 0.8)),
       "red": ((0.98, 1.02), (0.7, 0.95), (0.5, 0.85))}
ARM_P = {"white": 0.7, "yellow": 0.06, "black": 0.08, "grey": 0.06, "orange": 0.04, "blue": 0.03, "red": 0.03}
# Real 3D assets (assets/README.md: SimReady paddles and tabletop props, CC-BY-4.0; ambientCG surfaces, CC0)
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "simready"
MESH_PADDLE_P = 0.35                 # share of looks that use a real paddle mesh instead of the procedural one


def _hsv(rng, rng_hsv):
    (h0, h1), (s0, s1), (v0, v1) = rng_hsv
    return list(colorsys.hsv_to_rgb(rng.uniform(h0, h1) % 1.0, rng.uniform(s0, s1), rng.uniform(v0, v1)))


def _pick(rng, probs):
    keys = list(probs)
    return keys[rng.choice(len(keys), p=np.array([probs[k] for k in keys]) / sum(probs.values()))]


def sample_look(rng, blade=(0.0785, 0.075, 2.2), ball=None, arm=None):
    """A random appearance. blade = (half-length, half-width, superellipse exponent) from the physics episode."""
    top, bottom = _pick(rng, RUBBER_P), _pick(rng, RUBBER_P)
    wood = _hsv(rng, WOOD)
    handle_kind = rng.choice(["straight", "flared", "anatomic"])
    painted = rng.random() < 0.25
    ball_kind = ball or _pick(rng, BALL_P)
    arm_kind = arm or _pick(rng, ARM_P)
    look_out = {"blade": tuple(float(x) for x in blade), "throat": float(rng.uniform(0.0, 0.12)),
            "core_mm": float(rng.uniform(5.0, 7.0)), "rubber_mm": float(rng.uniform(2.0, 2.7)),
            "top": top, "bottom": bottom, "top_rgb": _hsv(rng, RUBBER[top]), "bottom_rgb": _hsv(rng, RUBBER[bottom]),
            "wood_rgb": wood, "handle": str(handle_kind), "handle_len": float(rng.uniform(0.09, 0.11)),
            "handle_w": float(rng.uniform(0.022, 0.030)), "handle_t": float(rng.uniform(0.020, 0.028)),
            "handle_rgb": _hsv(rng, RUBBER[_pick(rng, RUBBER_P)]) if painted else wood,
            "ball": ball_kind, "ball_rgb": _hsv(rng, BALL[ball_kind]),
            # rubber surface: glossy (fresh, tacky) to matte (worn, dusty); visible pimples or speckle; worn patches;
            # brand logos near the handle; edge tape around the blade
            "gloss": [float(rng.uniform(0.0, 0.8)), float(rng.uniform(0.0, 0.8))],
            "pattern": [str(rng.choice(["plain", "pimples", "speckle", "worn"], p=[0.4, 0.25, 0.2, 0.15]))
                        for _ in range(2)],
            "logos": [(float(rng.uniform(0.012, 0.04)), float(rng.uniform(0.005, 0.012)),
                       float(rng.uniform(-0.06, -0.02)), float(rng.uniform(-0.02, 0.02)),
                       list(rng.choice([[0.95, 0.95, 0.95], [0.85, 0.7, 0.2], [0.05, 0.05, 0.05], list(rng.uniform(0, 1, 3))])))
                      for _ in range(int(rng.integers(0, 3)))],
            "tape_rgb": (list(rng.choice([[0.05, 0.05, 0.05], [0.75, 0.75, 0.78], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7],
                                          [0.85, 0.7, 0.2]])) if rng.random() < 0.7 else None),
            "arm": arm_kind, "arm_rgb": _hsv(rng, ARM[arm_kind]),
            "floor_rgb": list(np.clip(rng.normal([0.92, 0.92, 0.92], 0.04), 0, 1)) if rng.random() < 0.6
            else _hsv(rng, ((0.05, 0.12), (0.2, 0.5), (0.4, 0.8))),
            "light": float(rng.uniform(0.5, 0.9))}
    paddles = [k for k, v in asset_catalog().items() if v["kind"] == "paddle"]
    use_mesh = bool(paddles) and rng.random() < MESH_PADDLE_P
    look_mesh = {"paddle_mesh": str(rng.choice(paddles)) if use_mesh else None,
                 "paddle_tint": _tint(rng), "paddle_flip": bool(rng.random() < 0.5)}
    return {**look_out, **look_mesh}


def _tint(rng):
    """Colour multiplier for a textured asset: mostly its own colours (brightness jitter), sometimes recoloured."""
    if rng.random() < 0.7:
        return list(np.clip(rng.uniform(0.75, 1.2) * (1 + rng.normal(0, 0.06, 3)), 0, 1.5))
    return list(0.35 + 0.65 * rng.uniform(0, 1, 3))


_CATALOG, _OBJ, _FRAME = None, {}, {}


def asset_catalog():
    """assets/simready/catalog.json ({} when the assets are not present)."""
    global _CATALOG
    if _CATALOG is None:
        f = ASSETS / "catalog.json"
        _CATALOG = json.loads(f.read_text()) if f.exists() else {}
    return _CATALOG


def surfaces():
    """Photo textures of table tops (wood, laminate, plastic, cloth)."""
    return sorted(str(p) for p in (ASSETS / "surfaces").glob("*.png"))


def _load_obj(path):
    """(vertices (n, 3), triangles (m, 3), per-corner uv (m, 3, 2) or None) of an OBJ written by the converter."""
    if path not in _OBJ:
        v, vt, f, ft = [], [], [], []
        for line in open(path):
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("f "):
                c = [x.split("/") for x in line.split()[1:4]]
                f.append([int(x[0]) - 1 for x in c])
                if len(c[0]) > 1 and c[0][1]:
                    ft.append([int(x[1]) - 1 for x in c])
        uv = np.array(vt, float)[np.array(ft, int)] if ft else None
        _OBJ[path] = (np.array(v, float), np.array(f, int), uv)
    return _OBJ[path]


def _parts(name):
    out = []
    for p in asset_catalog()[name]["parts"]:
        v, f, uv = _load_obj(str(ASSETS / name / p["obj"]))
        out.append((v, f, uv, str(ASSETS / name / p["png"]) if "png" in p else None, p["rgb"]))
    return out


def paddle_frame(name):
    """Blade frame of a paddle mesh: rows of R map asset coordinates to the paddle frame (x toward the tip, handle at
    -x, z the face normal), c is the centre of the top face, b the blade half-width (m)."""
    if name not in _FRAME:
        v = np.concatenate([p[0] for p in _parts(name)])
        mu = v.mean(0)
        _, U = np.linalg.eigh(np.cov((v - mu).T))
        n, y, x = U[:, 0], U[:, 1], U[:, 2]
        s, t = (v - mu) @ x, (v - mu) @ y
        k = np.clip(((s - s.min()) / np.ptp(s) * 40).astype(int), 0, 39)
        width = np.array([np.ptp(t[k == i]) if (k == i).sum() > 3 else 0.0 for i in range(40)])
        if width[:20].mean() > width[20:].mean():                  # the wide end is the blade: make it +x
            x, s = -x, -s
        b = 0.5 * width.max()
        centre_s = s.max() - 1.047 * b                             # standard blades: half-length ~ 1.047 x half-width
        blade = s > centre_s - 0.5 * b
        h = (v - mu) @ n
        tb = t[blade]
        c = mu + centre_s * x + 0.5 * (tb.max() + tb.min()) * y + h[blade].max() * n
        R = np.stack([x, np.cross(n, x), n])
        _FRAME[name] = (R, c, b)
    return _FRAME[name]


def _add_mesh_geoms(spec, body, prefix, parts, verts, tint):
    """Visual-only textured mesh geoms on `body` from converted asset parts (vertices already in the body frame)."""
    import mujoco
    for k, ((_, f, uv, png, rgb), v) in enumerate(zip(parts, verts)):
        mesh = spec.add_mesh(name=f"{prefix}_m{k}")
        mesh.uservert = v.flatten().tolist()
        mesh.userface = f.flatten().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL     # scans are not watertight
        if png is not None and uv is not None:
            uvf = uv.reshape(-1, 2).copy()
            uvf[:, 1] = 1.0 - uvf[:, 1]                               # USD/OBJ v points up; MuJoCo's points down
            mesh.usertexcoord = uvf.flatten().tolist()
            mesh.userfacetexcoord = np.arange(3 * len(f)).tolist()
            tex = spec.add_texture(name=f"{prefix}_t{k}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
            mat = spec.add_material(name=f"{prefix}_mat{k}", rgba=list(np.clip(tint, 0, 1)) + [1])
            texs = list(mat.textures)
            texs[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mat.textures = texs
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, material=mat.name)
        else:
            base = np.array(rgb, float) * np.array(tint, float)
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, rgba=list(np.clip(base, 0, 1)) + [1])
        g.contype = g.conaffinity = 0
        g.density = 0


def add_mesh_paddle(spec, body, look):
    """A real paddle mesh in the paddle frame, scaled to the episode's blade width; either face up."""
    name = look["paddle_mesh"]
    R, c, b = paddle_frame(name)
    scale = look["blade"][1] / b
    flip = np.diag([1.0, -1.0, -1.0]) if look.get("paddle_flip") else np.eye(3)
    parts = _parts(name)
    verts = []
    for v, *_ in parts:
        local = (v - c) @ R.T
        if look.get("paddle_flip"):                       # the other face up: rotate 180 deg about x, keep it on top
            local = local @ flip.T
            local[:, 2] -= local[:, 2].max()
        verts.append(scale * local)
    _add_mesh_geoms(spec, body, f"{body.name}_asset", parts, verts, look.get("paddle_tint", [1, 1, 1]))


def add_prop(spec, body, name, tint):
    """A tabletop object standing on its base at the body origin."""
    a = asset_catalog()[name]
    lo, hi = np.array(a["bbox_min"]), np.array(a["bbox_max"])
    origin = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
    parts = _parts(name)
    _add_mesh_geoms(spec, body, f"{body.name}_{name}", parts, [p[0] - origin for p in parts], tint)


def blade_outline(a, b, shape, throat, n=72):
    """Points (n, 2) of the blade outline in the paddle frame (x toward the tip, handle at -x)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = a * np.sign(c) * np.abs(c) ** (2 / shape)
    y = b * np.sign(s) * np.abs(s) ** (2 / shape)
    y = y * (1 - throat * np.clip(-x / a, 0, 1) ** 2)
    return np.stack([x, y], -1)


def handle_outline(a, length, width, kind, n=24):
    """Top-view outline of the handle (x from the blade's handle edge back to the butt)."""
    x = np.linspace(-a + 0.012, -a - length, n)
    u = (x - x[0]) / (x[-1] - x[0])                         # 0 at the blade, 1 at the butt
    if kind == "flared":
        w = width * (0.85 + 0.35 * u ** 2)
    elif kind == "anatomic":
        w = width * (0.9 + 0.2 * np.sin(np.pi * u))
    else:
        w = width * np.ones_like(u)
    return np.concatenate([np.stack([x, w / 2], -1), np.stack([x[::-1], -w[::-1] / 2], -1)])


def extrude(outline, z0, z1):
    """Prism mesh from a polygon (n, 2): vertices (2n + 2, 3) and triangles."""
    n = len(outline)
    c = outline.mean(0)
    v = np.concatenate([np.c_[outline, np.full(n, z0)], np.c_[outline, np.full(n, z1)],
                        [[c[0], c[1], z0], [c[0], c[1], z1]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return v, np.array(f)


def _rubber_material(spec, name, rgb, gloss, pattern, rng_seed):
    """Rubber look: glossy to matte, plain / pimple grid / speckle / worn blotches (procedural texture)."""
    import mujoco
    rng = np.random.default_rng(rng_seed)
    mat = spec.add_material(name=name, rgba=[1, 1, 1, 1] if pattern != "plain" else list(rgb) + [1])
    mat.specular, mat.shininess, mat.reflectance = float(0.1 + 0.6 * gloss), float(0.1 + 0.8 * gloss), 0.0
    if pattern != "plain":
        tex = spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=128, height=128)
        dark = list(np.clip(np.array(rgb) * 0.75, 0, 1))
        light = list(np.clip(np.array(rgb) * 1.25 + 0.05, 0, 1))
        if pattern == "pimples":
            tex.builtin, tex.rgb1 = mujoco.mjtBuiltin.mjBUILTIN_CHECKER, list(rgb)
            tex.rgb2 = list(np.clip(np.array(rgb) * 0.88, 0, 1))                   # subtle pimple grid
        else:
            tex.builtin, tex.rgb1, tex.rgb2 = mujoco.mjtBuiltin.mjBUILTIN_FLAT, list(rgb), list(rgb)
            tex.mark = mujoco.mjtMark.mjMARK_RANDOM
            tex.random = float(rng.uniform(0.02, 0.1) if pattern == "speckle" else rng.uniform(0.15, 0.35))
            tex.markrgb = dark if pattern == "speckle" else light
        texs = list(mat.textures)
        texs[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.textures = texs
        mat.texrepeat = [float(rng.uniform(20, 40))] * 2 if pattern == "pimples" else [1.0, 1.0]
    return name


def add_paddle(spec, body, look):
    """Add visual-only paddle meshes to `body` (origin = centre of the hitting surface, z = face normal)."""
    import mujoco
    if look.get("paddle_mesh") and look["paddle_mesh"] in asset_catalog():
        return add_mesh_paddle(spec, body, look)
    a, b, shape = look["blade"]
    blade = blade_outline(a, b, shape, look["throat"])
    r, core = look["rubber_mm"] / 1000, look["core_mm"] / 1000
    seed = int(abs(hash((a, b, shape, look["throat"]))) % (2 ** 31))
    mats = {"rubber_top": _rubber_material(spec, f"{body.name}_rt", look["top_rgb"], look.get("gloss", [0.3, 0.3])[0],
                                           look.get("pattern", ["plain", "plain"])[0], seed),
            "rubber_bottom": _rubber_material(spec, f"{body.name}_rb", look["bottom_rgb"],
                                              look.get("gloss", [0.3, 0.3])[1], look.get("pattern", ["plain"] * 2)[1],
                                              seed + 1)}
    parts = [("rubber_top", blade, -r, 0.0, look["top_rgb"]),
             ("core", blade * 0.997, -r - core, -r, look["wood_rgb"]),
             ("rubber_bottom", blade, -2 * r - core, -r - core, look["bottom_rgb"]),
             ("handle", handle_outline(a, look["handle_len"], look["handle_w"], look["handle"]),
              -r - core / 2 - look["handle_t"] / 2, -r - core / 2 + look["handle_t"] / 2, look["handle_rgb"])]
    if look.get("tape_rgb") is not None:                      # edge tape: a thin band just outside the blade edge
        parts.append(("tape", blade * 1.012, -2 * r - core + 0.0004, -0.0006, look["tape_rgb"]))   # top hidden
                                                                                                # under the rubber
    for i, (w, h, x, y, rgb) in enumerate(look.get("logos", [])):    # logos printed on the forehand rubber
        rect = np.array([[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]])
        parts.append((f"logo{i}", rect, 0.0, 0.0004, rgb))
    for name, poly, z0, z1, rgb in parts:
        v, f = extrude(poly, z0, z1)
        mesh = spec.add_mesh(name=f"{body.name}_{name}")
        mesh.uservert = v.flatten().tolist()
        mesh.userface = f.flatten().tolist()
        if name in mats:                                      # planar texture coordinates for the rubber pattern
            span = np.ptp(v[:, :2], axis=0) + 1e-9
            uv = (v[:, :2] - v[:, :2].min(0)) / span
            mesh.usertexcoord = uv.flatten().tolist()
            mesh.userfacetexcoord = f.flatten().tolist()
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, material=mats[name])
        else:
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, rgba=list(rgb) + [1.0])
        g.contype = g.conaffinity = 0
        g.density = 0


def color_arm(spec, rgb):
    """Recolour the SO-101's printed parts (every material except the black STS3215 servos)."""
    for m in spec.materials:
        if not m.name.startswith("sts3215"):
            m.rgba = list(rgb) + [1.0]
