"""Convert a recorded LeRobot SO-101 dataset into robotfpga `armlab` demo shards for TinyPolicy.

    python lerobot_to_armlab.py --repo-id flyingturtleboop/so101_medicine_bottle_pickplace \
        --out <dir> --exclude 11 14 --val 2

Writes <dir>/shard_000.npz with the keys armlab.policy.data.DemoSet reads (image, state, instr,
action, lengths, meta) and <dir>/norm.json, which a controller needs to map policy outputs back to
LeRobot joint values.

- image: the `front` (C920, top) camera, full frame resized to 96x96 RGB uint8 (spec §5.1).
- state / action: LeRobot degrees mapped to [-1, 1] by each joint's calibrated half-range,
  deg / half_span, since LeRobot degrees are measured from the middle of the calibrated range.
  Gripper 0..100 -> g / 50 - 1. Clipped to [-1, 1]. This is the real arm's own normalization; the
  sim's radians mapping (QUESTIONS.md Q5) isn't involved because training uses real data only.
- instr: 0 for every frame (one task).
Episodes are ordered with `--val` held-out ones first, because train.py uses the first
`--val-episodes` episodes for validation.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, HF_LEROBOT_HOME

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SIZE = 96


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--exclude", type=int, nargs="*", default=[])
    p.add_argument("--val", type=int, default=2, help="episodes held out for validation (placed first)")
    p.add_argument("--camera", default="front")
    p.add_argument("--follower-id", default="my_follower")
    args = p.parse_args()

    root = HF_LEROBOT_HOME / args.repo_id
    info = json.loads((root / "meta/info.json").read_text())
    task = json.loads((root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"] if (root / "meta/tasks.jsonl").exists() else None
    cal = json.loads((HF_LEROBOT_CALIBRATION / f"robots/so_follower/{args.follower_id}.json").read_text())
    half_span = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / 4095 for j in JOINTS[:5]])

    def norm(x):
        x = np.asarray(x, np.float64)
        out = np.empty_like(x)
        out[:, :5] = x[:, :5] / half_span
        out[:, 5] = x[:, 5] / 50 - 1
        return np.clip(out, -1, 1).astype(np.float32)

    t = pq.read_table(sorted((root / "data").rglob("*.parquet")),
                      columns=["index", "episode_index", "frame_index", "observation.state", "action"])
    index, ep, fr = (t[c].to_numpy() for c in ("index", "episode_index", "frame_index"))
    state, action = np.array(t["observation.state"].to_pylist()), np.array(t["action"].to_pylist())

    # Decode the camera video once, in global-index order (one video file per camera here).
    videos = sorted((root / "videos" / f"observation.images.{args.camera}").rglob("*.mp4"))
    assert len(videos) == 1, f"expected one {args.camera} video file, found {len(videos)}"
    cap, frames = cv2.VideoCapture(str(videos[0])), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    frames = np.stack(frames)
    assert len(frames) == info["total_frames"], f"decoded {len(frames)} frames, dataset has {info['total_frames']}"

    keep = [e for e in sorted(set(ep.tolist())) if e not in args.exclude]
    order = keep[-args.val:] + keep[:-args.val] if args.val else keep  # held-out episodes first
    rows = [np.where(ep == e)[0][np.argsort(fr[ep == e])] for e in order]
    sel = np.concatenate(rows)
    out = dict(
        image=frames[index[sel]],
        state=norm(state[sel]),
        instr=np.zeros(len(sel), np.int64),
        action=norm(action[sel]),
        lengths=np.array([len(r) for r in rows], np.int32),
        meta=np.array(json.dumps([{"source": args.repo_id, "episode": int(e), "task": task, "success": True}
                                  for e in order])),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "shard_000.npz", **out)
    (args.out / "norm.json").write_text(json.dumps({
        "joints": JOINTS, "half_span_deg": dict(zip(JOINTS[:5], half_span.round(4).tolist())),
        "gripper": "norm = g / 50 - 1 (g in LeRobot 0..100)",
        "inverse": "deg = norm * half_span_deg; gripper = (norm + 1) * 50",
        "camera": args.camera, "image": f"{SIZE}x{SIZE} RGB, full frame resized (no crop)",
        "episodes_val_first": [int(e) for e in order], "val": args.val, "excluded": args.exclude,
        "calibration": args.follower_id, "repo_id": args.repo_id}, indent=2))
    print(f"wrote {args.out}/shard_000.npz: {len(order)} episodes ({args.val} val first), {len(sel)} frames, "
          f"image {out['image'].shape}, state range [{out['state'].min():.2f}, {out['state'].max():.2f}], "
          f"action range [{out['action'].min():.2f}, {out['action'].max():.2f}]")


if __name__ == "__main__":
    main()
