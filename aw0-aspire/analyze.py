"""Paired analysis of AW-0 WAM (A1) vs the ASPIRE-style frozen programs (A2) on identical RoboTwin scenes.
Written before any A1/A2 result existed. Rules:
- Pairing: an episode pair counts only if the WAM shard's realized seed equals the seed scan's seed for that
  (task, protocol, shard, episode) and the code arm's first frame matches the scan's first frame
  (mean abs diff < FRAME_MAD_MAX on 0-255 RGB).
- Gate A0 (tasks GATE): per cell, Fisher exact test of our 50 episodes vs the paper's Table 11 rate on 50 rollouts;
  pass if at most 1 of the 8 cells has p < 0.05 and the pooled difference is within +-10 points.
- Comparison: per (task, protocol) exact McNemar on discordant pairs, Holm-corrected; pooled mean of per-task
  differences (code - WAM, equal weight per task as RoboTwin reports) with a stratified paired bootstrap 95% CI.
usage: analyze.py <results_root> > summary.json   (results_root holds seedscan-*, wam-eval-*, code-eval-*)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import binomtest, fisher_exact

TASKS = Path(__file__).with_name("tasks.txt").read_text().split()
GATE = TASKS[:4]
PAPER = {  # AW-0 paper Table 11 (p.45), success % over 50 rollouts: (WAM clean, WAM rand, hybrid clean, hybrid rand)
    "place_cans_plasticbox": (98, 12, 80, 34), "place_dual_shoes": (64, 12, 2, 2),
    "blocks_ranking_rgb": (96, 32, 94, 82), "open_laptop": (84, 38, 72, 48),
    "press_stapler": (86, 8, 100, 68), "place_shoe": (78, 22, 60, 30),
    "place_container_plate": (94, 26, 86, 22), "pick_dual_bottles": (80, 28, 96, 28),
}
PROTOS = ("clean", "randomized")
FRAME_MAD_MAX = 8.0
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def scan_seeds(root):
    seeds = {}
    for f in glob.glob(f"{root}/seedscan-*/seeds.tsv"):
        for line in open(f):
            t, p, k, ep, s = line.split()
            seeds[(t, p, int(k), int(ep))] = int(s)
    return seeds


def wam_episodes(root):
    """(task, proto, shard, ep) -> (seed, success) from eval_policy's per-episode 'Success rate: a/b ... seed: N'."""
    best = {}   # a retried shard leaves several logs: keep the one with the most episodes
    for f in glob.glob(f"{root}/wam-eval-*/logs/**/eval_*.log", recursive=True):
        m = re.search(r"_(clean|randomized)_(\w+?)_g0_s(\d+)/", f)
        if not m:
            continue
        shard = (m.group(2), m.group(1), int(m.group(3)))
        eps, prev = [], 0
        for line in open(f, errors="ignore"):
            r = re.search(r"Success rate: (\d+)/(\d+) .*current seed: (\d+)", ANSI.sub("", line))
            if r:
                suc, n, seed = map(int, r.groups())
                eps.append((seed, suc - prev))
                prev = suc
        if len(eps) > len(best.get(shard, [])):
            best[shard] = eps
    return {(*shard, i): e for shard, eps in best.items() for i, e in enumerate(eps)}


def code_episodes(root):
    out = {}
    for f in glob.glob(f"{root}/code-eval-*/results.tsv"):
        for line in open(f):
            t, p, k, ep, s, rew, _, raised = line.rstrip("\n").split("\t")
            out[(t, p, int(k), int(ep))] = (int(s), rew == "1.000", int(raised), str(Path(f).parent))
    return out


def frame_mad(root, key, code_dir, scan_ep=None):
    """Mean abs RGB difference between the scan's and the code arm's first frame; scan_ep picks another
    scene for the mismatched reference distribution."""
    t, p, k, ep = key
    a = glob.glob(f"{root}/seedscan-*/frames/{t}/{p}/s{k}/ep{(ep if scan_ep is None else scan_ep):03d}.png")
    b = f"{code_dir}/frames/{t}/{p}/s{k}/ep{ep:03d}.png"
    if not a or not Path(b).exists():
        return None
    x = np.asarray(Image.open(a[0]).convert("RGB"), dtype=np.float32)
    y = np.asarray(Image.open(b).convert("RGB").resize(x.shape[1::-1]), dtype=np.float32)
    return float(np.abs(x - y).mean())


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def main(root):
    seeds, wam, code = scan_seeds(root), wam_episodes(root), code_episodes(root)
    pairs, dropped = {}, {"no_wam": 0, "no_code": 0, "seed_mismatch": 0, "frame_mismatch": 0, "no_frame": 0}
    mads, ref_mads = [], []
    for key, s in seeds.items():
        if key not in wam:
            dropped["no_wam"] += 1; continue
        if key not in code:
            dropped["no_code"] += 1; continue
        if wam[key][0] != s or code[key][0] != s:
            dropped["seed_mismatch"] += 1; continue
        mad = frame_mad(root, key, code[key][3])
        if mad is None:
            dropped["no_frame"] += 1; continue
        mads.append(mad)
        ref = frame_mad(root, key, code[key][3], scan_ep=(key[3] + 1) % 10)
        if ref is not None:
            ref_mads.append(ref)
        if mad >= FRAME_MAD_MAX:
            dropped["frame_mismatch"] += 1; continue
        pairs[key] = (wam[key][1], int(code[key][1]))

    cells, rng = {}, np.random.default_rng(0)
    for t in TASKS:
        for p in PROTOS:
            w = [wam[k][1] for k in wam if k[0] == t and k[1] == p]
            c = [int(code[k][1]) for k in code if k[0] == t and k[1] == p]
            pr = np.array([v for k, v in pairs.items() if k[0] == t and k[1] == p])   # [n, 2] (wam, code)
            n10 = int(((pr[:, 0] == 1) & (pr[:, 1] == 0)).sum()) if len(pr) else 0
            n01 = int(((pr[:, 0] == 0) & (pr[:, 1] == 1)).sum()) if len(pr) else 0
            cells[f"{t}/{p}"] = {
                "wam_rate": float(np.mean(w)) if w else None, "wam_n": len(w),
                "code_rate": float(np.mean(c)) if c else None, "code_n": len(c),
                "paired_n": len(pr), "wam_only": n10, "code_only": n01,
                "mcnemar_p": binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0,
                "paper_wam": PAPER[t][PROTOS.index(p)] / 100, "paper_hybrid": PAPER[t][2 + PROTOS.index(p)] / 100,
                "code_check_success_raised": sum(code[k][2] for k in code if k[0] == t and k[1] == p),
            }
    keys = [k for k in cells if cells[k]["paired_n"]]
    for k, a in zip(keys, holm(np.array([cells[k]["mcnemar_p"] for k in keys]))):
        cells[k]["mcnemar_p_holm"] = float(a)

    gate = []
    for t in GATE:
        for p in PROTOS:
            c = cells[f"{t}/{p}"]
            if c["wam_n"]:
                ours = round(c["wam_rate"] * c["wam_n"])
                paper = round(c["paper_wam"] * 50)
                pv = fisher_exact([[ours, c["wam_n"] - ours], [paper, 50 - paper]]).pvalue
                gate.append({"cell": f"{t}/{p}", "ours": c["wam_rate"], "paper": c["paper_wam"], "p": pv})
    pooled_gate = float(np.mean([g["ours"] - g["paper"] for g in gate])) if gate else None
    gate_pass = bool(gate) and bool(sum(g["p"] < 0.05 for g in gate) <= 1 and abs(pooled_gate) <= 0.10)

    pooled = {}
    for p in PROTOS + ("both",):
        sel = [(t, q) for t in TASKS for q in PROTOS if (p == "both" or q == p)
               and cells[f"{t}/{q}"]["paired_n"]]
        if not sel:
            continue
        arrays = [np.array([v for k, v in pairs.items() if (k[0], k[1]) == s]) for s in sel]   # each [n, 2]
        diff = float(np.mean([a[:, 1].mean() - a[:, 0].mean() for a in arrays]))
        boots = [np.mean([b[:, 1].mean() - b[:, 0].mean() for b in (a[rng.integers(0, len(a), len(a))] for a in arrays)])
                 for _ in range(5000)]
        pooled[p] = {"cells": len(sel), "pairs": int(sum(len(a) for a in arrays)), "code_minus_wam": diff,
                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     "wam": float(np.mean([a[:, 0].mean() for a in arrays])),
                     "code": float(np.mean([a[:, 1].mean() for a in arrays]))}

    json.dump({"pairing": {"scanned": len(seeds), "paired": len(pairs), "dropped": dropped,
                           "frame_mad_median": float(np.median(mads)) if mads else None,
                           "frame_mad_max": float(np.max(mads)) if mads else None,
                           "frame_mad_other_scene_min": float(np.min(ref_mads)) if ref_mads else None,
                           "frame_mad_other_scene_median": float(np.median(ref_mads)) if ref_mads else None},
               "gate": {"cells": gate, "pooled_diff": pooled_gate, "pass": gate_pass},
               "cells": cells, "pooled": pooled}, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
