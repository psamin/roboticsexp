"""Paired analysis of AW-0 WAM (A1) vs the ASPIRE-style frozen programs (A2) on identical RoboTwin scenes.
Written before any A1/A2 result existed. Rules:
- Pairing: an episode pair counts only if the WAM shard's realized seed equals the seed scan's seed for that
  (task, protocol, shard, episode) and the code arm's first frame matches the scan's first frame
  (mean abs diff < FRAME_MAD_MAX on 0-255 RGB).
- WAM runs come in two configurations, told apart by the run name: A1 "wam-bf16-*" (L4: bf16, text encoder on CPU,
  all tasks; the comparison arm) and A0-ref "wam-fp32-*" (L40/L40S: the released fp32 precision, GATE tasks; its
  text encoder also runs on CPU, because moving the 7B encoder onto a 44 GB card for each instruction peaks at
  44.6 GB and OOMs, so the pair differs only in precision).
- Gate A0 (tasks GATE), decided on A0-ref and also reported for A1: per cell, Fisher exact test of our 50 episodes
  vs the paper's Table 11 rate on 50 rollouts; pass if at most 1 of the 8 cells has p < 0.05 and the pooled
  difference is within +-10 points.
- Deviation check: A1 vs A0-ref paired on the same scenes (exact McNemar pooled over the GATE cells).
- Comparison: per (task, protocol) exact McNemar on discordant pairs, Holm-corrected; pooled mean of per-task
  differences (code - WAM, equal weight per task as RoboTwin reports) with a stratified paired bootstrap 95% CI.
- Arm F (fresh fix-loop program, fixloop-eval-*): paired exact McNemar vs A2 and vs A1 on the same scanned scenes;
  its locked Stage-2 seeds (shard "L") are reported apart.
usage: analyze.py <results_root> > summary.json   (holds seedscan-*, wam-eval-*, code-eval-*, fixloop-eval-*)
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


def wam_episodes(root, dtype):
    """(task, proto, shard, ep) -> (seed, success) from eval_policy's per-episode 'Success rate: a/b ... seed: N'."""
    best = {}   # a retried shard leaves several logs: keep the one with the most episodes
    for f in glob.glob(f"{root}/wam-eval-*/logs/**/wam-{dtype}-*/eval_*.log", recursive=True):
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


def code_episodes(root, prefix="code-eval"):
    """A2 (code-eval-*) or arm F (fixloop-eval-*). Rows with shard "L" are the locked Stage-2 seeds, returned apart."""
    out, locked = {}, []
    for f in glob.glob(f"{root}/{prefix}-*/results.tsv"):
        for line in open(f):
            t, p, k, ep, s, rew, _, raised = line.rstrip("\n").split("\t")
            if k == "L":
                locked.append((t, p, int(s), rew == "1.000"))
                continue
            out[(t, p, int(k), int(ep))] = (int(s), rew == "1.000", int(raised), str(Path(f).parent))
    return out, locked


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
    seeds, wam, ref = scan_seeds(root), wam_episodes(root, "bf16"), wam_episodes(root, "fp32")
    code, _ = code_episodes(root)
    fresh, fresh_locked = code_episodes(root, "fixloop-eval")
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
        other = frame_mad(root, key, code[key][3], scan_ep=(key[3] + 1) % 10)
        if other is not None:
            ref_mads.append(other)
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

    def gate_of(arm):
        cells_, n_sig = [], 0
        for t in GATE:
            for p in PROTOS:
                w = [v[1] for k, v in arm.items() if k[0] == t and k[1] == p]
                if not w:
                    continue
                ours, paper = sum(w), round(PAPER[t][PROTOS.index(p)] / 100 * 50)
                pv = fisher_exact([[ours, len(w) - ours], [paper, 50 - paper]]).pvalue
                cells_.append({"cell": f"{t}/{p}", "n": len(w), "ours": ours / len(w), "paper": paper / 50, "p": pv})
        diff = float(np.mean([c["ours"] - c["paper"] for c in cells_])) if cells_ else None
        ok = len(cells_) == 2 * len(GATE) and bool(sum(c["p"] < 0.05 for c in cells_) <= 1 and abs(diff) <= 0.10)
        return {"cells": cells_, "pooled_diff": diff, "pass": ok}

    dev = [(v, ref[k][1]) for k, v in wam.items() if k in ref and ref[k][0] == v[0] and k[0] in GATE]
    n10 = sum(1 for (a, _), b in dev if a == 1 and b == 0)   # bf16 succeeded, fp32 failed
    n01 = sum(1 for (a, _), b in dev if a == 0 and b == 1)
    deviation = {"paired": len(dev), "bf16_rate": float(np.mean([a[1] for a, _ in dev])) if dev else None,
                 "fp32_rate": float(np.mean([b for _, b in dev])) if dev else None, "bf16_only": n10, "fp32_only": n01,
                 "mcnemar_p": binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0}

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

    # Arm F: the fresh fix-loop program vs A2 (shipped program) and A1 (WAM) on the same scanned scenes
    arm_f = {}
    for p in PROTOS:
        ks = [k for k in fresh if k[1] == p and k in code and fresh[k][0] == code[k][0] == seeds.get(k)]
        f_ok = [int(fresh[k][1]) for k in ks]
        vs = {}
        for name, other in (("shipped_program", {k: int(code[k][1]) for k in ks}),
                            ("wam_bf16", {k: wam[k][1] for k in ks if k in wam and wam[k][0] == seeds[k]})):
            kk = list(other)
            n01 = sum(1 for k in kk if other[k] == 0 and fresh[k][1])
            n10 = sum(1 for k in kk if other[k] == 1 and not fresh[k][1])
            vs[name] = {"paired": len(kk), "other_rate": float(np.mean([other[k] for k in kk])) if kk else None,
                        "fresh_only": n01, "other_only": n10,
                        "mcnemar_p": binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0}
        arm_f[p] = {"n": len(ks), "rate": float(np.mean(f_ok)) if ks else None, "vs": vs}
    arm_f["locked_stage2"] = {"success": sum(r for *_, r in fresh_locked), "n": len(fresh_locked)}

    json.dump({"arm_f_fresh_fixloop": arm_f, "pairing": {"scanned": len(seeds), "paired": len(pairs), "dropped": dropped,
                           "frame_mad_median": float(np.median(mads)) if mads else None,
                           "frame_mad_max": float(np.max(mads)) if mads else None,
                           "frame_mad_other_scene_min": float(np.min(ref_mads)) if ref_mads else None,
                           "frame_mad_other_scene_median": float(np.median(ref_mads)) if ref_mads else None},
               "gate_released_fp32": gate_of(ref), "gate_a1_bf16": gate_of(wam), "deviation_bf16_vs_fp32": deviation,
               "cells": cells, "pooled": pooled}, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
