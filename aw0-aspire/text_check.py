"""CPU vs GPU Cosmos-Reason1 instruction embeddings: the only thing AW0_TEXT_DEVICE=cpu changes.
usage: text_check.py <awam> <robotwin> <text_encoder_dir> <task>..."""
import importlib.util
import json
import os
import sys
import time

import torch

awam, robotwin, text_path, tasks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
spec = importlib.util.spec_from_file_location("deploy_policy", f"{awam}/robotwin/deploy/deploy_policy.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)
prompts = [p for t in tasks for p in json.load(open(f"{robotwin}/description/task_instruction/{t}.json"))["seen"][:3]]

os.environ.pop("AW0_TEXT_DEVICE", None)
gpu = dp._CosmosTextEncoder(text_path, "cuda")                      # released behaviour
ref = {p: gpu.encode(p).float().cpu() for p in prompts}
del gpu
torch.cuda.empty_cache()

os.environ["AW0_TEXT_DEVICE"] = "cpu"
cpu = dp._CosmosTextEncoder(text_path, "cuda")                      # patched: computed on CPU, returned on cuda
worst = 1.0
for p in prompts:
    t = time.time()
    e = cpu.encode(p)
    dt = time.time() - t
    assert e.device.type == "cuda"
    a, b = ref[p].flatten(), e.float().cpu().flatten()              # [L*100352]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    worst = min(worst, cos)
    print(f"cos={cos:.6f} max_abs={(a - b).abs().max():.4f} mean_abs={(a - b).abs().mean():.5f} {dt:.1f}s | {p[:70]}")
print(f"TEXT_CHECK worst_cos={worst:.6f} over {len(prompts)} prompts")
