"""Aggregate the ablation: eval/<tag>/eval.json + train-<tag>-<job>/{train_stats,peft_modules}.json -> table + plots.

  python analyze.py --results <flicklora-results dir> --out <dir>
Writes comparison.csv, comparison.md, and plots (success, turns/flick, peak/sustained speed, axial travel,
flicks, completion time, train cost). Mean ± std over training seeds where several exist.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

train = {}
# failed attempts leave same-named dirs without stats: keep finished runs only, newest job id per tag
for d in sorted(glob.glob(os.path.join(a.results, "train-*")), key=lambda p: int(p.rsplit("-", 1)[-1])):
    m = re.match(r"train-(.+)-(\d+)$", os.path.basename(d))
    if not m or "smoke" in m.group(1) or "integ" in m.group(1):
        continue
    if not os.path.exists(os.path.join(d, "train_stats.json")):
        continue
    st = json.load(open(os.path.join(d, "train_stats.json"))) if os.path.exists(os.path.join(d, "train_stats.json")) else {}
    pm = os.path.join(d, "peft_modules.json")
    if os.path.exists(pm):
        p = json.load(open(pm))
        k = [k for k in p if k.startswith("r")][0]
        st["trainable_params"] = p[k]["trainable_params"]
        st["n_adapted_modules"] = p[k]["n_adapted_modules"]
    log = os.path.join(d, "train.log")
    if os.path.exists(log):
        txt = open(log).read()
        mm = re.search(r"num_learnable_params=(\d+)", txt)
        if mm:
            st.setdefault("trainable_params", int(mm.group(1)))
        losses = re.findall(r"step:(\S+) .*? loss:([\d.]+)", txt)
        st["final_loss"] = float(losses[-1][1]) if losses else None
    train[m.group(1)] = st


def group_of(tag):   # lora-r32-s1000 -> lora-r32 ; full-s1000 -> full
    return re.sub(r"-s\d+$", "", tag)


rows = defaultdict(list)
for f in glob.glob(os.path.join(a.results, "eval", "*", "eval.json")):
    tag = os.path.basename(os.path.dirname(f))
    if "integ" in tag or "smoke" in tag:
        continue
    s = json.load(open(f))["summary"]
    s.update({f"train_{k}": v for k, v in train.get(tag, {}).items()})
    rows[group_of(tag)].append(s)

order = ["base", "lora-r8", "lora-r32", "lora-r128", "full", "expert"]
metrics = [("success_rate", "success rate", 100), ("policy_failures", "policy fails", 1), ("sim_failures", "sim fails", 1),
           ("mean_turns_per_flick", "turns / flick", 1), ("mean_peak_w", "peak ω (rad/s)", 1),
           ("mean_sustained_w", "sustained ω (rad/s)", 1), ("mean_axial_mm", "axial travel (mm)", 1),
           ("mean_flicks_success", "flicks (successes)", 1), ("mean_completion_s", "completion (s)", 1),
           ("train_train_seconds", "train time (min)", 1 / 60), ("train_peak_gpu_mem_mib", "peak GPU mem (GiB)", 1 / 1024),
           ("train_trainable_params", "trainable params (M)", 1e-6), ("train_final_loss", "final train loss", 1)]


def agg(vals, scale):
    v = [x * scale for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v)) if len(v) > 1 else None


table = []
for g in order + sorted(set(rows) - set(order)):
    if g not in rows:
        continue
    r = {"policy": g, "n_seeds": len(rows[g])}
    for k, name, sc in metrics:
        r[name] = agg([s.get(k) for s in rows[g]], sc)
    table.append(r)

with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "n_seeds"] + [m[1] for m in metrics])
    for r in table:
        w.writerow([r["policy"], r["n_seeds"]] + ["" if r[m[1]][0] is None else round(r[m[1]][0], 4) for m in metrics])


def fmt(mu_sd):
    mu, sd = mu_sd
    if mu is None:
        return "–"
    return f"{mu:.3g}" + (f" ± {sd:.2g}" if sd is not None else "")


with open(os.path.join(a.out, "comparison.md"), "w") as f:
    f.write("| policy | seeds | " + " | ".join(m[1] for m in metrics) + " |\n")
    f.write("|---|---|" + "---|" * len(metrics) + "\n")
    for r in table:
        f.write(f"| {r['policy']} | {r['n_seeds']} | " + " | ".join(fmt(r[m[1]]) for m in metrics) + " |\n")
print(open(os.path.join(a.out, "comparison.md")).read())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["success rate", "turns / flick", "peak ω (rad/s)", "sustained ω (rad/s)", "axial travel (mm)",
            "completion (s)", "train time (min)", "peak GPU mem (GiB)", "trainable params (M)"]
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    names = [r["policy"] for r in table]
    for ax, key in zip(axs.flat, show):
        mu = [r[key][0] or 0 for r in table]
        sd = [r[key][1] or 0 for r in table]
        ax.bar(names, mu, yerr=sd, color=["#888" if n in ("base", "expert") else "#3a6ea5" for n in names], capsize=3)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=30)
        if key == "trainable params (M)":
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparison.png"), dpi=130)
    print("wrote comparison.png")
except ImportError:
    print("matplotlib missing: skipped plots")
