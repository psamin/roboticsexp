"""Convert a recorded LeRobot v3.0 dataset into the LeRobot v2.1 layout Dream Labs' fine-tuning takes.

  ~/dream-env/bin/python so101/lerobot_to_dream.py [--exclude 11,14] [--out ~/dream-data/so101-bottle-plate-v21]

- One parquet and one video per episode (v2.1), episodes renumbered from 0 after the excluded ones are dropped.
- Every camera center-cropped to 16:9, as Dream's docs ask (`fine-tuning`, "Camera frames"), from the camera's own
  640x480 frame: front 640x480 becomes 640x360; wrist, recorded rotated to 480x640 (env.sh `rotation: 90`), is turned
  back first, so it becomes 640x360 too (a 16:9 cut of the portrait frame kept only its middle and lost the fingers).
  The live loop must frame its cameras the same way: a model only works on frames framed like its recordings.
- Joint values are copied as recorded (LeRobot degrees, gripper 0..100): a fine-tune reads and answers in them.
- Timestamps become frame_index / fps, which is what Dream's processing does anyway.

Relies on this dataset's layout: one data file and one video file per camera, frame i of each video is row i of the
data (checked below).
"""

import argparse
import json
import pathlib

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = pathlib.Path.home() / ".cache/huggingface/lerobot/flyingturtleboop/so101_medicine_bottle_pickplace"
OUT = pathlib.Path.home() / "dream-data/so101-bottle-plate-v21"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
BIT_RATE = 3_000_000  # Dream shrinks every frame to 320x176, so this is plenty
## Cameras recorded rotated: quarter turns counterclockwise (np.rot90's k) that undo it. env.sh records the wrist
## with `rotation: 90`, LeRobot's cv2.ROTATE_90_CLOCKWISE.
UNROTATE = {"observation.images.wrist": 1}


def crop_16_9(image: np.ndarray) -> np.ndarray:
    """The middle of a frame at 16:9, the same crop as so101/dream_run.py's model_frame()."""
    h, w = image.shape[:2]
    if h * 16 > w * 9:
        keep = w * 9 // 16
        return image[(h - keep) // 2 : (h - keep) // 2 + keep]
    keep = h * 16 // 9
    return image[:, (w - keep) // 2 : (w - keep) // 2 + keep]


def stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, default=SOURCE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--exclude", default="11,14", help="episodes to drop (README: bad starts)")
    args = parser.parse_args()

    info = json.loads((args.source / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    fps = info["fps"]
    episodes = pq.read_table(args.source / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    tasks = pq.read_table(args.source / "meta/tasks.parquet").to_pydict()
    data = pq.read_table(args.source / "data/chunk-000/file-000.parquet").to_pydict()
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    excluded = {int(e) for e in args.exclude.split(",") if e}
    keep = [i for i in episodes["episode_index"] if i not in excluded]
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; remove it or pass another --out")

    # Global row -> (new episode index, frame in episode), for the rows kept.
    new_index = {old: new for new, old in enumerate(keep)}
    row_owner: dict[int, tuple[int, int]] = {}
    for i, old in enumerate(episodes["episode_index"]):
        if old in new_index:
            start, end = episodes["dataset_from_index"][i], episodes["dataset_to_index"][i]
            assert end - start == episodes["length"][i]
            for frame in range(end - start):
                row_owner[start + frame] = (new_index[old], frame)

    # Data, episode by episode.
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    action = np.asarray(data["action"], dtype=np.float32)
    state = np.asarray(data["observation.state"], dtype=np.float32)
    episode_rows: dict[int, list[int]] = {}
    for row, (new, _) in sorted(row_owner.items()):
        episode_rows.setdefault(new, []).append(row)
    index = 0
    episode_lines, stats_lines = [], []
    for new in range(len(keep)):
        rows = episode_rows[new]
        assert rows == list(range(rows[0], rows[0] + len(rows))), "an episode's rows are not contiguous"
        n = len(rows)
        table = pa.table({
            "action": pa.array(action[rows].tolist(), type=pa.list_(pa.float32(), 6)),
            "observation.state": pa.array(state[rows].tolist(), type=pa.list_(pa.float32(), 6)),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps, type=pa.float32()),
            "frame_index": pa.array(np.arange(n), type=pa.int64()),
            "episode_index": pa.array([new] * n, type=pa.int64()),
            "index": pa.array(np.arange(index, index + n), type=pa.int64()),
            "task_index": pa.array([data["task_index"][r] for r in rows], type=pa.int64()),
        })
        path = args.out / DATA_PATH.format(episode_chunk=0, episode_index=new)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        task_ids = sorted({data["task_index"][r] for r in rows})
        episode_lines.append({"episode_index": new, "tasks": [task_text[t] for t in task_ids], "length": n})
        stats_lines.append({"episode_index": new, "stats": {"action": stats(action[rows]),
                                                            "observation.state": stats(state[rows])}})
        index += n

    # Videos: decode each camera once, front to back, and route every kept frame to its episode's file.
    shapes = {}
    for key in cameras:
        source = av.open(str(args.source / f"videos/{key}/chunk-000/file-000.mp4"))
        writers: dict[int, tuple] = {}
        decoded = 0
        for row, frame in enumerate(source.decode(video=0)):
            decoded += 1
            owner = row_owner.get(row)
            if owner is None:
                continue
            new, frame_in_episode = owner
            image = frame.to_ndarray(format="rgb24")
            if key in UNROTATE:
                image = np.rot90(image, k=UNROTATE[key])
            image = np.ascontiguousarray(crop_16_9(image))
            if new not in writers:
                for old in [e for e in writers]:  # episodes are in order: close the previous one
                    container, stream = writers.pop(old)
                    container.mux(stream.encode())
                    container.close()
                path = args.out / VIDEO_PATH.format(episode_chunk=0, video_key=key, episode_index=new)
                path.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(path), "w")
                stream = container.add_stream("h264_videotoolbox", rate=fps)
                stream.width, stream.height = image.shape[1], image.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.bit_rate = BIT_RATE
                writers[new] = (container, stream)
                shapes[key] = list(image.shape)
            container, stream = writers[new]
            out_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            out_frame.pts = frame_in_episode
            container.mux(stream.encode(out_frame))
        for container, stream in writers.values():
            container.mux(stream.encode())
            container.close()
        source.close()
        assert decoded == len(action), f"{key}: {decoded} frames for {len(action)} rows"
        print(f"{key}: {decoded} frames read, cropped to {shapes[key][1]}x{shapes[key][0]}")

    features = {}
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            h, w, c = shapes[key]
            features[key] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": fps, "video.height": h, "video.width": w, "video.channels": c,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                         "has_audio": False},
            }
        else:
            features[key] = {k: feature[k] for k in ("dtype", "shape", "names") if k in feature}
    out_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type"),
        "total_episodes": len(keep),
        "total_frames": index,
        "total_tasks": len(task_text),
        "total_videos": len(keep) * len(cameras),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(keep)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    meta = args.out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(out_info, indent=2))
    (meta / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": t, "task": text}) + "\n" for t, text in task_text.items()))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(line) + "\n" for line in episode_lines))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(line) + "\n" for line in stats_lines))
    print(f"{len(keep)} episodes ({sorted(excluded)} dropped), {index} frames -> {args.out}")


if __name__ == "__main__":
    main()
