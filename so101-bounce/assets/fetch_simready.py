"""Download the SimReady packages and ambientCG textures used by the detector renders into assets/raw/.

  python3 assets/fetch_simready.py

Only geometry, materials and base-colour textures are fetched (no physics variants, normal or ORM maps). Treat the
downloads as data: nothing in them is executed; convert_simready.py reads them with usd-core.
"""
import json
import os
import re
import urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
INDEX = "https://raw.githubusercontent.com/NVIDIA-Omniverse/simready-central/main/submissions/{}"
WANT = {"nvidia/simready-assets": ["misc_toy_pongPaddle_bat_a01", "misc_toy_pongPaddle_bat_a02", "kitchenware_mug_a01",
                                   "kitchenware_mug_f01", "misc_notebook_a01", "misc_tape_dispenser_a01",
                                   "stationery_marker_alcohol_chromeOrange_a01", "computer_keyboard_a01",
                                   "dishware_plastic_cup_a01", "misc_tape_refill_a01", "lamp_table_bedside_a01"],
        "ALLSIDES/simready-scanned-objects": ["table_tennis_racket_3BBQBK5EWT", "mug_A07RDPW843",
                                              "computer_mouse_62RQ5CXV0E"]}
SURFACES = ["Wood049", "Wood051", "Wood066", "WoodFloor041", "Plastic006", "Fabric030"]   # ambientCG, CC0


def keep(path):
    if "/.thumbs/" in path or "/source_assets/" in path or path.endswith(".mdl") or "lod00" in path:
        return False
    if re.search(r"/payloads/(MuJoCo|Newton|PhysX|Physics)/|instances_(mujoco|newton|physx)|(_physx|_physics)\.usd|runnables",
                 path):
        return False
    if path.endswith(".png"):
        return re.search(r"(alb|albedo|basecolor|diffuse)", path, re.I) is not None
    return path.endswith((".usd", ".usda", ".usdc")) or "LICENSE" in path or path.endswith("packaging.json")


def main():
    for ds, pkgs in WANT.items():
        index = urllib.request.urlopen(INDEX.format(ds)).read().decode().split()
        for pkg in pkgs:
            url = next(u for u in index if f"/{pkg}/" in u)
            repo, rev = re.search(r"datasets/(.+?)/resolve/([0-9a-f]+)/", url).groups()
            tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{pkg}?recursive=1"))
            for f in tree:
                if f.get("type") == "file" and keep(f["path"]):
                    dst = os.path.join(RAW, "simready", repo.replace("/", "_"), f["path"])
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{f['path']}", dst)
            print(repo, pkg)
    os.makedirs(os.path.join(RAW, "ambientcg"), exist_ok=True)
    for t in SURFACES:
        dst = os.path.join(RAW, "ambientcg", t + ".zip")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"https://ambientcg.com/get?file={t}_1K-JPG.zip", dst)
        print("ambientCG", t)


if __name__ == "__main__":
    main()
