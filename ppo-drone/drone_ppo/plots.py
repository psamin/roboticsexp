"""Plots from logged metrics only (metrics.csv, eval_history.json, eval_results.json)."""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax, title, xlabel="training time (min)"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def _csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                out.append(math.nan)
        return out
    return col


def training_curves(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    col = _csv(os.path.join(run_dir, "metrics.csv"))
    ev = json.load(open(os.path.join(run_dir, "eval_history.json")))
    t, et = col("wall_min"), [e["wall_min"] for e in ev]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, facecolor=SURFACE)
    _style(axes[0], "Average episode reward (training episodes)")
    axes[0].plot(t, col("ep_return"), color=BLUE, linewidth=1.6)
    _style(axes[1], "Distance to target at episode end (m)")
    axes[1].plot(t, col("final_dist_m"), color=BLUE, linewidth=1.4, label="training episodes")
    _style(axes[2], "Hover success rate")
    axes[2].plot(t, col("success_rate"), color=BLUE, linewidth=1.4, label="training episodes")
    for suite, c in (("hover", ORANGE), ("robust", AQUA)):
        axes[1].plot(et, [e.get(f"{suite}_final_dist_m", math.nan) for e in ev], color=c, linewidth=2, marker="o",
                     markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"held-out: {suite}")
        axes[2].plot(et, [e.get(f"{suite}_hover_success_rate", math.nan) for e in ev], color=c, linewidth=2,
                     marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"held-out: {suite}")
    axes[1].set_yscale("log")
    axes[2].set_ylim(0, 1.04)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes[1:]:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "training_curves.png")
    fig.savefig(p1, facecolor=SURFACE)
    plt.close(fig)

    panels = [("Actor (clipped surrogate) loss", "actor_loss", None), ("Critic loss", "critic_loss", "log"),
              ("Policy entropy (pre-squash Gaussian)", "entropy", None), ("Approximate KL", "approx_kl", None),
              ("Samples per second", "sps", None), ("Crash / out-of-bounds rate", "crash_rate", None),
              ("Episode length (s)", "ep_length_s", None), ("Curriculum stage (0 hover, 1 reach, 2 robust)", "stage", None)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=140, facecolor=SURFACE)
    for ax, (title, key, scale) in zip(axes.flat, panels):
        _style(ax, title)
        if key == "stage":
            ax.step(t, col(key), where="post", color=BLUE, linewidth=2)
            ax.set_yticks([0, 1, 2])
        else:
            ax.plot(t, col(key), color=BLUE, linewidth=1.4)
        if scale:
            ax.set_yscale(scale)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "training_diagnostics.png")
    fig.savefig(p2, facecolor=SURFACE)
    plt.close(fig)
    return [p1, p2]


def eval_comparison(results: dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pols = list(results["policies"])
    suites = list(results["policies"][pols[0]])
    colors = {"ppo": BLUE, "pd": ORANGE, "random": AQUA}
    names = {"ppo": "PPO (trained)", "pd": "PD controller", "random": "random actions"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150, facecolor=SURFACE)
    for ax, key, title, pct in ((axes[0], "hover_success_rate", "Hover success rate, held-out conditions", True),
                                (axes[1], "final_dist_m", "Distance to target at episode end (m)", False)):
        _style(ax, title, xlabel="")
        w = 0.8 / len(pols)
        for i, pol in enumerate(pols):
            xs = [j + (i - (len(pols) - 1) / 2) * w for j in range(len(suites))]
            ys = [results["policies"][pol][s][key] for s in suites]
            ax.bar(xs, ys, width=w - 0.03, color=colors.get(pol, INK2), label=names.get(pol, pol), edgecolor=SURFACE,
                   linewidth=2)
            for x, y in zip(xs, ys):
                ax.annotate(f"{100*y:.0f}%" if pct else f"{y:.2f}", xy=(x, y), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(suites)), suites)
        if pct:
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(out_dir, "eval_comparison.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p
