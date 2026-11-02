"""Load lerobot/smolvla_base, wrap it with LeRobot's own PEFT path at each rank, and log exactly which modules get
LoRA adapters and how many parameters train. No data needed. Writes peft_modules.json.

  python peft_inspect.py --ranks 8 32 128 --out <dir>
"""
import argparse
import json
import os
import re

import torch
from lerobot.configs.default import PeftConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", type=int, nargs="+", default=[8, 32, 128])
ap.add_argument("--target_modules", default=None, help="override; default = SmolVLA's built-in PEFT targets")
ap.add_argument("--out", required=True)
a = ap.parse_args()

report = {}
base = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
n_total = sum(p.numel() for p in base.parameters())
report["base_total_params"] = n_total
report["default_targets"] = base._get_default_peft_targets()
del base
for r in a.ranks:
    pol = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    pol.config.pretrained_path = "lerobot/smolvla_base"   # lerobot-train sets this; PEFT validation requires it
    cfg = PeftConfig(r=r, lora_alpha=r)
    if a.target_modules:
        cfg.target_modules = a.target_modules
    import dataclasses
    peft_model = pol.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg))
    trainable = {n: p.numel() for n, p in peft_model.named_parameters() if p.requires_grad}
    adapted = sorted({re.sub(r"\.lora_[AB].*$", "", n) for n in trainable if ".lora_" in n})
    other = sorted(n for n in trainable if ".lora_" not in n)
    report[f"r{r}"] = dict(
        r=r, lora_alpha=r, trainable_params=sum(trainable.values()),
        trainable_pct=100 * sum(trainable.values()) / n_total,
        n_adapted_modules=len(adapted), adapted_modules=adapted, other_trainable=other,
        peft_config=str(peft_model.peft_config["default"]),
    )
    print(f"r={r}: {sum(trainable.values()):,} trainable ({report[f'r{r}']['trainable_pct']:.3f}%), "
          f"{len(adapted)} adapted modules", flush=True)
    del pol, peft_model
    torch.cuda.empty_cache()
print("module types adapted:", sorted({m.split('.')[-1] for m in report[f'r{a.ranks[0]}']['adapted_modules']}))
os.makedirs(a.out, exist_ok=True)
json.dump(report, open(os.path.join(a.out, "peft_modules.json"), "w"), indent=1)
print("wrote peft_modules.json")
