"""Learning curves from a training run's metrics.csv and eval_history.json (real logged values only)."""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"beginner": "#2a78d6", "intermediate": "#eb6834", "advanced": "#1baf7a"}
ONE = "#2a78d6"


def _style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("training time (min)", color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (KeyError, ValueError, TypeError):
                out.append(math.nan)
        return out
    return col


def _ref(ax, y, label, style):
    ax.axhline(y, color=INK2, linewidth=1.2, linestyle=style)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK2)


def learning_curves(run_dir, out_dir, baselines: dict | None = None):
    """baselines: {"analytical": {"advanced": {"save_rate": ..}, ...}, "random": {...}}"""
    os.makedirs(out_dir, exist_ok=True)
    col = _read_csv(os.path.join(run_dir, "metrics.csv"))
    with open(os.path.join(run_dir, "eval_history.json")) as f:
        ev = json.load(f)
    t = col("wall_min")
    et = [e["wall_min"] for e in ev]
    files = []

    # 1. headline: held-out save rate per level, with baselines on the advanced level
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160, facecolor=SURFACE)
    _style(ax, "Held-out save rate during PPO training (on-target shots)", "save rate")
    for lv, c in SERIES.items():
        ys = [e.get(f"{lv}_save_rate", math.nan) for e in ev]
        ax.plot(et, ys, color=c, linewidth=2, marker="o", markersize=5, label=lv,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if ys:
            ax.annotate(f"{lv} {100*ys[-1]:.0f}%", xy=(et[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
    if baselines:
        if "analytical" in baselines:
            _ref(ax, baselines["analytical"]["advanced"]["save_rate"],
                 f"analytical keeper, advanced: {100*baselines['analytical']['advanced']['save_rate']:.0f}%", "--")
        if "random" in baselines:
            _ref(ax, baselines["random"]["advanced"]["save_rate"],
                 f"random actions, advanced: {100*baselines['random']['advanced']['save_rate']:.0f}%", ":")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2, bbox_to_anchor=(1.0, 0.08))
    if et:
        ax.set_xlim(0, max(et) * 1.12)
    fig.tight_layout()
    p = os.path.join(out_dir, "save_rate.png"); fig.savefig(p, facecolor=SURFACE); plt.close(fig); files.append(p)

    # 2. training diagnostics as small multiples (one measure per panel, one y-scale each)
    panels = [
        ("Training episodes: save share", "frac_save", None),
        ("Curriculum level (0 beginner, 2 advanced)", "curriculum_level", None),
        ("Policy (clipped surrogate) loss", "policy_loss", None),
        ("Value loss", "value_loss", "log"),
        ("Policy entropy", "entropy", None),
        ("Approximate KL per update", "approx_kl", None),
        ("Mean episode return", None, None),
        ("Samples per second (collection + update)", "sps_total", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=140, facecolor=SURFACE)
    for ax, (title, key, scale) in zip(axes.flat, panels):
        _style(ax, title)
        if key is None:
            terms = ["rew_save", "rew_goal", "rew_progress", "rew_action", "rew_jerk"]
            cols = [col(k) for k in terms]
            y = [sum(v for v in vals if not math.isnan(v)) for vals in zip(*cols)]
        else:
            y = col(key)
        if key == "curriculum_level":
            ax.step(t, y, where="post", color=ONE, linewidth=2)
            ax.set_yticks([0, 1, 2])
        else:
            ax.plot(t, y, color=ONE, linewidth=1.6)
        if scale:
            ax.set_yscale(scale)
    fig.tight_layout()
    p = os.path.join(out_dir, "training_diagnostics.png"); fig.savefig(p, facecolor=SURFACE); plt.close(fig)
    files.append(p)
    return files


def eval_bars(results: dict, out_path):
    """Grouped bars: policy x shot level, from evaluate.py's eval_results.json."""
    pols = list(results["policies"])
    levels = list(results["policies"][pols[0]])
    colors = {"ppo": "#2a78d6", "analytical": "#eb6834", "random": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=SURFACE)
    _style(ax, "Save rate on held-out test shots (on target)", "save rate")
    ax.set_xlabel("")
    w = 0.8 / len(pols)
    for i, pol in enumerate(pols):
        xs = [j + (i - (len(pols) - 1) / 2) * w for j in range(len(levels))]
        ys = [results["policies"][pol][lv]["save_rate"] for lv in levels]
        ax.bar(xs, ys, width=w - 0.03, color=colors.get(pol, INK2), label=pol, edgecolor=SURFACE, linewidth=2)
        for x, y in zip(xs, ys):
            ax.annotate(f"{100*y:.0f}%", xy=(x, y), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=len(pols), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
