"""Convert raw expert episodes (run_expert.py --mode collect) into a LeRobot v3 dataset (LeRobot 0.6.1).

  python to_lerobot.py --raw RAW/train --out DS/flick_train --repo_id local/flick_train
Features: observation.images.camera1 (close-up), observation.images.camera2 (side), observation.state (4 finger
joints), action (4 joint targets). Task string on every frame. timestamp/index are written by LeRobot itself.
"""
import argparse
import glob
import json
import os
import shutil

import imageio.v2 as iio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repo_id", required=True)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

eps = sorted(glob.glob(os.path.join(a.raw, "ep_*")))
assert eps, f"no episodes in {a.raw}"
first = np.load(os.path.join(eps[0], "lowdim.npz"))
h, w = iio.get_reader(os.path.join(eps[0], "camera1.mp4")).get_data(0).shape[:2]
names = [str(n) for n in first["joint_names"]] if "joint_names" in first.files else ["slide_x", "mcp", "pip", "dip"]
features = {
    "observation.images.camera1": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
    "observation.images.camera2": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
    "action": {"dtype": "float32", "shape": (len(names),), "names": names},
}
if os.path.exists(a.out):
    shutil.rmtree(a.out)
robot_type = "franka_panda" if names[0].startswith("panda") else "flick_finger"
ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=a.out, robot_type=robot_type,
                           use_videos=True, image_writer_threads=8)
index = []
for e in eps:
    low = np.load(os.path.join(e, "lowdim.npz"))
    meta = json.load(open(os.path.join(e, "meta.json")))
    c1 = [f[:, :, :3] for f in iio.get_reader(os.path.join(e, "camera1.mp4"))]
    c2 = [f[:, :, :3] for f in iio.get_reader(os.path.join(e, "camera2.mp4"))]
    T = min(len(c1), len(c2), len(low["action"]))
    # Frame 0 of every raw episode is stale: right after a reset the rgb annotator returned nothing and the
    # recorder reused the previous episode's last image. Start each episode at frame 1 (one control step later).
    for t in range(1, T):
        ds.add_frame({
            "observation.images.camera1": c1[t],
            "observation.images.camera2": c2[t],
            "observation.state": low["state"][t].astype(np.float32),
            "action": low["action"][t].astype(np.float32),
            "task": meta["task"],
        })
    ds.save_episode()
    index.append(dict(episode=len(index), seed=meta["seed"], frames=T - 1, flicks=meta.get("n_flicks"),
                      success=meta["success"]))
    print(f"episode {len(index)-1}: seed {meta['seed']} frames {T}", flush=True)
ds.finalize()
json.dump(index, open(os.path.join(a.out, "episode_seeds.json"), "w"), indent=1)
print(f"wrote {len(index)} episodes, {sum(i['frames'] for i in index)} frames -> {a.out}")
