"""Train the ball detector on synthetic shards (scripts/make_detector_data.py) and report held-out accuracy.

  python scripts/train_detector.py --data data/detector --out outputs/detector --epochs 20

Held-out: the last shard(s). Metrics: detection rate (ball >= 50 % visible, found within 4 px), position error,
false positives on images with no visible ball, and per-camera breakdown. Exports ONNX for the Mac.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from bounce.detector import BallNet, decode, loss_fn


def load(files):
    X, L = [], []
    for f in files:
        z = np.load(f)
        X.append(z["images"])
        L.append(z["labels"])
    return np.concatenate(X), np.concatenate(L)


def to_tensor(x, device):
    return torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.25)


@torch.no_grad()
def evaluate(model, X, L, device, bs=512):
    model.eval()
    dets, oks = [], []
    for i in range(0, len(X), bs):
        d, ok = decode(model(to_tensor(X[i:i + bs], device)))
        dets.append(d.cpu())
        oks.append(ok.cpu())
    model.train()
    d, ok = torch.cat(dets).numpy(), torch.cat(oks).numpy()
    vis = (L[:, 4] > 0) & (L[:, 3] >= 0.5)
    absent = (L[:, 4] == 0) | (L[:, 3] < 0.05)
    err = np.hypot(d[:, 0] - L[:, 0], d[:, 1] - L[:, 1])
    hit = ok & (err < 4.0)
    rad_err = np.abs(d[:, 2] - L[:, 2])
    res = {"radius_err_px_median": float(np.median(rad_err[vis & ok])) if (vis & ok).any() else float("nan"),
           "radius_err_px_p90": float(np.percentile(rad_err[vis & ok], 90)) if (vis & ok).any() else float("nan"),
           "detected@4px": float(hit[vis].mean()), "median_err_px": float(np.median(err[vis & ok])) if
           (vis & ok).any() else float("nan"), "false_positive": float(ok[absent].mean()) if absent.any() else
           float("nan"), "n_visible": int(vis.sum()), "n_absent": int(absent.sum())}
    for c, name in enumerate(("overhead", "side", "front")):
        m = vis & (L[:, 5] == c)
        if m.any():
            res[f"detected@4px_{name}"] = float(hit[m].mean())
    if L.shape[1] > 6:                       # by image brightness: dark / medium / bright
        for name, lo, hi in (("dark", 0, 70), ("medium", 70, 170), ("bright", 170, 256)):
            m = vis & (L[:, 6] >= lo) & (L[:, 6] < hi)
            if m.any():
                res[f"detected@4px_{name}"] = float(hit[m].mean())
                res[f"n_{name}"] = int(m.sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val_shards", type=int, default=1)
    ap.add_argument("--width", type=int, default=24)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(a.data, "shard*.npz")))
    Xtr, Ltr = load(files[:-a.val_shards])
    Xva, Lva = load(files[-a.val_shards:])
    print(f"train {len(Xtr)} images, held-out {len(Xva)}, device {dev}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model = BallNet(a.width).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    rng = np.random.default_rng(0)
    Ltr_t = torch.from_numpy(Ltr).to(dev)
    best, hist, t0 = -1, [], time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(Xtr))
        for k in range(len(Xtr) // a.batch):
            idx = perm[k * a.batch:(k + 1) * a.batch]
            x = to_tensor(Xtr[idx], dev)
            lab = Ltr_t[idx].clone()
            if rng.random() < 0.5:                                   # horizontal flip
                x = x.flip(-1)
                lab[:, 0] = x.shape[-1] - 1 - lab[:, 0]
            loss, parts = loss_fn(model(x), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        res = evaluate(model, Xva, Lva, dev)
        res.update(epoch=ep + 1, loss=float(loss), minutes=(time.time() - t0) / 60)
        hist.append(res)
        print(json.dumps(res), flush=True)
        if res["detected@4px"] > best:
            best = res["detected@4px"]
            torch.save({"model": model.state_dict(), "width": a.width, "eval": res}, os.path.join(a.out, "best.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    sd = torch.load(os.path.join(a.out, "best.pt"), map_location="cpu")
    model = BallNet(sd["width"])
    model.load_state_dict(sd["model"])
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, 240, 320), os.path.join(a.out, "ballnet.onnx"), input_names=["image"],
                      output_names=["out"], dynamic_axes={"image": {0: "batch"}, "out": {0: "batch"}}, opset_version=17)
    print("best held-out", json.dumps(sd["eval"]), "-> ballnet.onnx", flush=True)


if __name__ == "__main__":
    main()
