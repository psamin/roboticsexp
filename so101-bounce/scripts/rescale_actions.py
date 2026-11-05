"""Warm start for a wider action range: scale the actor's mean outputs (and std) per joint by old/new action_scale,
so goal = q_ref + scale * tanh(u) moves about the same as before (exact near zero). Resets best score and wall time.
An optional std resets the exploration noise (the old fine-tunes sat at the e^-5 floor and explored nothing).

  uv run python scripts/rescale_actions.py in.pt out.pt 0.20,0.30,0.30,0.35,0.30 0.40,0.40,0.40,0.35,0.30 [0.1]
"""
import math
import sys

import torch

src, dst, old, new = sys.argv[1], sys.argv[2], *(torch.tensor([float(x) for x in s.split(",")]) for s in sys.argv[3:5])
c = torch.load(src, map_location="cpu", weights_only=False)
r = old / new
last = max(k for k in c["actor"] if k.startswith("mean.") and k.endswith(".weight"))
c["actor"][last] = c["actor"][last] * r[:, None]
c["actor"][last.replace("weight", "bias")] = c["actor"][last.replace("weight", "bias")] * r
c["actor"]["log_std"] = c["actor"]["log_std"] + torch.log(r)
if len(sys.argv) > 5:
    c["actor"]["log_std"] = torch.full_like(c["actor"]["log_std"], math.log(float(sys.argv[5])))
    c.pop("opt", None)                                          # Adam moments belong to the old std
c["extra"]["best"], c["extra"]["wall_s"] = -1e9, 0.0
torch.save(c, dst)
print(f"{last}: rows scaled by {[round(float(x), 3) for x in r]}; log_std {c['actor']['log_std'].tolist()}")
