"""SimReady USD -> OBJ (+ albedo PNG) for MuJoCo, plus the ambientCG table-surface photos.

  uv run --with usd-core --with pillow python -I assets/convert_simready.py assets/raw/simready assets/simready

Reads the downloads in assets/raw (fetch_simready.py) and writes assets/simready: one folder per asset with OBJ parts
(decimated by vertex clustering to <= 30k triangles) and albedo PNGs (<= 1024 px), catalog.json, and surfaces/.
"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdShade

Image.MAX_IMAGE_PIXELS = None
SRC, DST = sys.argv[1], sys.argv[2]
PKGS = {  # name: (dataset dir, package, kind)
    "paddle_red": ("nvidia_simready-assets", "misc_toy_pongPaddle_bat_a01", "paddle"),
    "paddle_black": ("nvidia_simready-assets", "misc_toy_pongPaddle_bat_a02", "paddle"),
    "paddle_scan_orange": ("ALLSIDES_simready-scanned-objects", "table_tennis_racket_3BBQBK5EWT", "paddle"),
    "mug_black": ("nvidia_simready-assets", "kitchenware_mug_a01", "prop"),
    "mug_red": ("nvidia_simready-assets", "kitchenware_mug_f01", "prop"),
    "mug_scan": ("ALLSIDES_simready-scanned-objects", "mug_A07RDPW843", "prop"),
    "mouse_scan": ("ALLSIDES_simready-scanned-objects", "computer_mouse_62RQ5CXV0E", "prop"),
    "notebook": ("nvidia_simready-assets", "misc_notebook_a01", "prop"),
    "tape_dispenser": ("nvidia_simready-assets", "misc_tape_dispenser_a01", "prop"),
    "tape_roll": ("nvidia_simready-assets", "misc_tape_refill_a01", "prop"),
    "marker_orange": ("nvidia_simready-assets", "stationery_marker_alcohol_chromeOrange_a01", "prop"),
    "keyboard": ("nvidia_simready-assets", "computer_keyboard_a01", "prop"),
    "cup_white": ("nvidia_simready-assets", "dishware_plastic_cup_a01", "prop"),
    "lamp_white": ("nvidia_simready-assets", "lamp_table_bedside_a01", "prop"),
}
MAX_FACES = 30000


def root_stage(pkgdir):
    lods = sorted(glob.glob(os.path.join(pkgdir, "simready_usd", "payloads", "*_meshes_body_*_lod01.usd")))
    if lods:                                   # ALLSIDES scans: open the mesh LOD layer directly
        return Usd.Stage.Open(lods[0]), lods[0]
    r = [p for p in glob.glob(os.path.join(pkgdir, "simready_usd", "*.usd")) if "/payloads/" not in p][0]
    return Usd.Stage.Open(r), r


def albedo(prim, pkgdir):
    """(texture path or None, constant rgb) of the bound material's base colour."""
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    cands, const = [], None
    if mat:
        for d in Usd.PrimRange(mat.GetPrim()):
            if not d.IsA(UsdShade.Shader):
                continue
            for i in UsdShade.Shader(d).GetInputs():
                v, n = i.Get(), i.GetBaseName().lower()
                if hasattr(v, "resolvedPath") and v.path:
                    cands.append((n, v.resolvedPath or v.path))
                elif ("diffuse" in n or "base_color" in n or "basecolor" in n) and isinstance(v, Gf.Vec3f):
                    const = [float(x) for x in v]
    for n, p in cands:
        base = os.path.basename(p).lower()
        if any(k in base for k in ("alb", "albedo", "basecolor", "diffuse")):
            if not os.path.exists(p):       # texture referenced by a different LOD: use the downloaded albedo
                alt = glob.glob(os.path.join(pkgdir, "**", "*lod01_albedo.png"), recursive=True)
                p = alt[0] if alt else p
            if os.path.exists(p):
                return p, const
    alt = glob.glob(os.path.join(pkgdir, "**", "*lod01_albedo.png"), recursive=True)
    return (alt[0] if alt else None), (const or [0.8, 0.8, 0.8])


def mesh_arrays(prim):
    m = UsdGeom.Mesh(prim)
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
    pts = np.array(m.GetPointsAttr().Get(), float)
    pts = np.c_[pts, np.ones(len(pts))] @ xf
    pts = pts[:, :3]
    counts = np.array(m.GetFaceVertexCountsAttr().Get(), int)
    idx = np.array(m.GetFaceVertexIndicesAttr().Get(), int)
    pv = None
    api = UsdGeom.PrimvarsAPI(prim)
    for name in ("st", "st0", "UVMap", "uv", "map1"):
        if api.HasPrimvar(name):
            pv = api.GetPrimvar(name)
            break
    uv, uv_interp = None, None
    if pv is not None:
        uv = np.array(pv.ComputeFlattened(), float)
        uv_interp = pv.GetInterpolation()
    # triangulate (fan); keep the face-vertex corner index for face-varying UVs
    tris, corners, c0 = [], [], 0
    for n in counts:
        for k in range(1, n - 1):
            tris.append([idx[c0], idx[c0 + k], idx[c0 + k + 1]])
            corners.append([c0, c0 + k, c0 + k + 1])
        c0 += n
    tris, corners = np.array(tris), np.array(corners)
    if uv is None:
        tuv = None
    elif uv_interp == UsdGeom.Tokens.faceVarying:
        tuv = uv[corners]
    else:
        tuv = uv[tris]
    return pts, tris, tuv


def decimate(pts, tris, tuv, max_faces):
    """Vertex clustering on a grid that grows until the mesh has <= max_faces triangles (keeps the surface closed;
    texture coordinates stay per triangle corner)."""
    if len(tris) <= max_faces:
        return pts, tris, tuv
    g = float(np.ptp(pts, 0).max()) / 400
    n0 = len(tris)
    while True:
        key = np.floor(pts / g).astype(np.int64)
        _, cid, inv_count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        cid = cid.ravel()
        cent = np.zeros((cid.max() + 1, 3))
        np.add.at(cent, cid, pts)
        cent /= inv_count[:, None]
        t = cid[tris]
        ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
        if ok.sum() <= max_faces:
            print(f"    decimated {n0} -> {ok.sum()} faces (grid {1000 * g:.2f} mm)")
            return cent, t[ok], (tuv[ok] if tuv is not None else None)
        g *= 1.25


def write_obj(path, pts, tris, tuv):
    used = np.unique(tris)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    with open(path, "w") as f:
        for p in pts[used]:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        if tuv is not None:
            for t in tuv.reshape(-1, 2):
                f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
            for i, t in enumerate(tris):
                a, b, c = remap[t] + 1
                f.write(f"f {a}/{3 * i + 1} {b}/{3 * i + 2} {c}/{3 * i + 3}\n")
        else:
            for t in tris:
                a, b, c = remap[t] + 1
                f.write(f"f {a} {b} {c}\n")


def main():
    os.makedirs(DST, exist_ok=True)
    catalog = {}
    for name, (ds, pkg, kind) in PKGS.items():
        pkgdir = os.path.join(SRC, ds, pkg)
        st, rootfile = root_stage(pkgdir)
        meshes = [p for p in st.Traverse(Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        out = os.path.join(DST, name)
        os.makedirs(out, exist_ok=True)
        parts, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
        for k, prim in enumerate(meshes):
            pts, tris, tuv = mesh_arrays(prim)
            pts, tris, tuv = decimate(pts, tris, tuv, MAX_FACES)
            lo, hi = np.minimum(lo, pts[np.unique(tris)].min(0)), np.maximum(hi, pts[np.unique(tris)].max(0))
            tex, const = albedo(prim, pkgdir)
            write_obj(os.path.join(out, f"part{k}.obj"), pts, tris, tuv if tex else None)
            part = {"obj": f"part{k}.obj", "faces": int(len(tris)), "rgb": const}
            if tex:
                im = Image.open(tex).convert("RGB")
                im.thumbnail((1024, 1024) if kind == "paddle" else (512, 512))
                im.save(os.path.join(out, f"part{k}.png"))
                part["png"] = f"part{k}.png"
            parts.append(part)
        catalog[name] = {"kind": kind, "source": f"{ds}/{pkg}", "license": "CC-BY-4.0", "parts": parts,
                         "bbox_min": lo.round(4).tolist(), "bbox_max": hi.round(4).tolist()}
        print(name, kind, len(parts), "parts", sum(p["faces"] for p in parts), "faces", (hi - lo).round(3).tolist(),
              ["png" in p for p in parts], flush=True)
    json.dump(catalog, open(os.path.join(DST, "catalog.json"), "w"), indent=1)
    import zipfile                                         # ambientCG surfaces: the colour map, 512 px
    os.makedirs(os.path.join(DST, "surfaces"), exist_ok=True)
    for z in sorted(glob.glob(os.path.join(os.path.dirname(SRC.rstrip("/")), "ambientcg", "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            colour = [n for n in zf.namelist() if n.endswith("_Color.jpg")][0]
            im = Image.open(zf.open(colour)).convert("RGB")
        im.thumbnail((512, 512))
        im.save(os.path.join(DST, "surfaces", os.path.basename(z)[:-4] + ".png"))


main()
