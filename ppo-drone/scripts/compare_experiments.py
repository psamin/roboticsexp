"""Compare reward-shaping / curriculum variants from their logged results (no Isaac needed).

  python scripts/compare_experiments.py --exp outputs/exp --out outputs/exp/report
Reads <exp>/<variant>/eval.json and <exp>/<variant>/train/eval_history.json; writes a markdown table and plots.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_ppo.plots import INK, INK2, SURFACE, _style

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
ORDER = ["baseline", "stability", "shaped_nospin", "l2f", "l2f_full", "l2f_full_nospin", "l2f_motor_lag"]
SUITES = ["hover", "reach", "robust", "recovery"]
COLS = [("hover_success_rate", "success", lambda v: f"{100*v:.1f}%"),
        ("crash_rate", "crash", lambda v: f"{100*v:.1f}%"),
        ("final_dist_m", "final dist", lambda v: f"{100*v:.1f} cm"),
        ("time_to_hover_s", "time to hover", lambda v: f"{v:.2f} s"),
        ("ang_speed_last_2s", "body rate, last 2 s", lambda v: f"{v:.2f} rad/s"),
        ("yaw_rate_last_2s", "yaw rate, last 2 s", lambda v: f"{v:.2f} rad/s"),
        ("ang_acc_last_2s", "wobble, last 2 s", lambda v: f"{v:.1f} rad/s²"),
        ("action_rate_last_2s", "action jitter, last 2 s", lambda v: f"{v:.4f}"),
        ("max_tilt_deg", "max tilt", lambda v: f"{v:.1f}°")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="outputs/exp")
    p.add_argument("--out", default="outputs/exp/report")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    found = [v for v in ORDER if os.path.exists(os.path.join(a.exp, v, "eval.json"))]
    res = {v: json.load(open(os.path.join(a.exp, v, "eval.json"))) for v in found}
    hist = {}
    for v in found:
        h = os.path.join(a.exp, v, "train", "eval_history.json")
        if os.path.exists(h):
            hist[v] = json.load(open(h))
    fmt = lambda f, x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f(x)

    lines = []
    for suite in SUITES:
        lines += [f"\n### {suite}\n", "| policy | " + " | ".join(c[1] for c in COLS) + " |",
                  "|---|" + "---|" * len(COLS)]
        rows = [(v, res[v]["policies"]["ppo"][suite]) for v in found]
        if found and "pd" in res[found[0]]["policies"]:
            rows += [("PD controller", res[found[0]]["policies"]["pd"][suite]),
                     ("random actions", res[found[0]]["policies"]["random"][suite])]
        for name, r in rows:
            lines.append(f"| {name} | " + " | ".join(fmt(f, r.get(k)) for k, _, f in COLS) + " |")
    with open(os.path.join(a.out, "comparison.md"), "w") as f:
        f.write("Held-out evaluation, 1,024 episodes per suite (test seed 2000). Stability columns average the last 2 s "
                "of episodes that did not fail.\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # held-out learning curves per variant (robust suite): success and final distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=150, facecolor=SURFACE)
    _style(axes[0], "Held-out robust-suite hover success")
    _style(axes[1], "Held-out robust-suite distance at episode end (m)")
    for i, v in enumerate(found):
        if v not in hist:
            continue
        t = [e["wall_min"] for e in hist[v]]
        for ax, key in ((axes[0], "robust_hover_success_rate"), (axes[1], "robust_final_dist_m")):
            y = [e.get(key, math.nan) for e in hist[v]]
            ax.plot(t, y, color=COLORS[i], linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, label=v)
            ax.annotate(v, xy=(t[-1], y[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=INK,
                        va="center")
    axes[0].set_ylim(0, 1.04)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "learning_curves.png"), facecolor=SURFACE)
    plt.close(fig)

    # stability metrics, robust and recovery suites
    metrics = [("hover_success_rate", "hover success"), ("final_dist_m", "final distance (m)"),
               ("ang_speed_last_2s", "body rate, last 2 s (rad/s)"), ("action_rate_last_2s", "action jitter, last 2 s")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.4), dpi=140, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, metrics):
        _style(ax, title, xlabel="")
        w = 0.8 / max(len(found), 1)
        for i, v in enumerate(found):
            xs = [j + (i - (len(found) - 1) / 2) * w for j in range(2)]
            ys = [res[v]["policies"]["ppo"][s].get(key, math.nan) for s in ("robust", "recovery")]
            ax.bar(xs, ys, width=w - 0.02, color=COLORS[i], label=v, edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks([0, 1], ["robust", "recovery"])
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "stability.png"), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
