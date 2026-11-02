"""Serve a SmolVLA policy (base, LoRA adapter, or full fine-tune) to the Isaac Sim eval over a localhost socket.

The two sides run different Pythons (Isaac 3.10/3.11 vs LeRobot 3.12) and different numpy majors, so the wire format
is raw bytes + a JSON header, never pickle.
  python policy_server.py --ckpt OUT/pretrained_model --dataset_root DS --port 5555
  python policy_server.py --base --dataset_root DS --port 5555          # unadapted lerobot/smolvla_base
"""
import argparse
import json
import socket
import struct
import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--base", action="store_true")
ap.add_argument("--dataset_root", required=True)
ap.add_argument("--repo_id", default="local/flick_train")
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--task", default="Flick the loosened nut until it spins off the bolt")
ap.add_argument("--n_action_steps", type=int, default=10,
                help="actions executed per predicted 50-step chunk before re-planning (same for every policy)")
a = ap.parse_args()

meta = LeRobotDatasetMetadata(a.repo_id, root=a.dataset_root)
if a.base:
    path = "lerobot/smolvla_base"
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.input_features, cfg.output_features = None, None      # derive from our dataset, as training does
    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path, dataset_stats=meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"features": {**policy.config.input_features, **policy.config.output_features},
                                     "norm_map": policy.config.normalization_mapping, "stats": meta.stats},
            "rename_observations_processor": {"rename_map": {}}},
        postprocessor_overrides={"unnormalizer_processor": {"features": policy.config.output_features,
                                                            "norm_map": policy.config.normalization_mapping,
                                                            "stats": meta.stats}})
else:
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    cfg.pretrained_path = a.ckpt                                # from_pretrained does not set it
    policy = make_policy(cfg, ds_meta=meta)                     # loads base + LoRA adapter when cfg.use_peft
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
policy.eval()
policy.config.n_action_steps = a.n_action_steps     # the action queue is rebuilt from this on every reset()
policy.reset()
print(f"policy ready: {type(policy).__name__} use_peft={getattr(policy.config, 'use_peft', False)} "
      f"chunk={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps}", flush=True)


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def send_json(conn, obj):
    b = json.dumps(obj).encode()
    conn.sendall(struct.pack("!I", len(b)) + b)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", a.port))
srv.listen(1)
print(f"SERVER_READY port {a.port}", flush=True)
while True:
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            hdr = json.loads(recv_exact(conn, struct.unpack("!I", recv_exact(conn, 4))[0]))
            if hdr["cmd"] == "reset":
                policy.reset()
                send_json(conn, {"ok": True})
                continue
            if hdr["cmd"] == "bye":
                send_json(conn, {"ok": True})
                raise SystemExit(0)
            obs = {}
            for key, shape, dtype in hdr["arrays"]:
                n = int(np.prod(shape)) * np.dtype(dtype).itemsize
                arr = np.frombuffer(recv_exact(conn, n), dtype=dtype).reshape(shape)
                if dtype == "uint8":   # HWC uint8 -> 1,C,H,W float in [0,1]
                    obs[key] = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
                else:
                    obs[key] = torch.from_numpy(arr.copy()).float().unsqueeze(0)
            obs["task"] = [a.task]
            t0 = time.perf_counter()
            with torch.inference_mode():
                act = post(policy.select_action(pre(obs)))
            torch.cuda.synchronize()
            send_json(conn, {"action": act.squeeze(0).float().cpu().numpy().tolist(),
                             "infer_ms": (time.perf_counter() - t0) * 1000})
    except ConnectionError:
        conn.close()
