"""Can this node run SmolVLA inference on its GPU? Loads lerobot/smolvla_base and times select_action on dummy inputs."""
import time

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("arch list", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").cuda().eval()
batch = {}
for k, f in p.config.input_features.items():
    batch[k] = torch.rand(1, *f.shape, device="cuda")
batch["task"] = ["Flick the loosened nut until it spins off the bolt"]
tok = p.language_tokenizer if hasattr(p, "language_tokenizer") else None
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(p.config.vlm_model_name)
enc = tk(batch["task"], padding="max_length", max_length=p.config.tokenizer_max_length, return_tensors="pt")
batch["observation.language.tokens"] = enc["input_ids"].cuda()
batch["observation.language.attention_mask"] = enc["attention_mask"].bool().cuda()
with torch.inference_mode():
    for i in range(3):
        p.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = p.select_action(batch)
        torch.cuda.synchronize()
        print(f"select_action {i}: {(time.perf_counter()-t0)*1000:.0f} ms, action {tuple(a.shape)}")
print("INFER_OK")
