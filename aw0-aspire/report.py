"""Results page for the AW-0 vs ASPIRE comparison: inline-SVG charts and tables from analyze.py's summary.json,
filled into a hand-written page template at its {{name}} placeholders.
usage: report.py <summary.json> <costs.json> <template.html> <out.html>"""
import html
import json
import math
import sys

S = json.load(open(sys.argv[1]))
COST = json.load(open(sys.argv[2]))
TASKS = [c.split("/")[0] for c in S["cells"]][::2]
PROTOS = ("clean", "randomized")
COL = {"wam": "var(--s1)", "code": "var(--s2)", "fresh": "var(--s3)", "fp32": "var(--s1d)"}


def wilson(p, n):
    if not n:
        return (0, 0)
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def dot_chart(rows, series, title, ref_label=None, height_per=34):
    """rows: [(label, {series: (rate, n)}, ref_rate|None)]; horizontal dot plot 0-100 % with Wilson whiskers."""
    W, L, R, T = 640, 190, 24, 30
    H = T + height_per * len(rows) + 30
    x = lambda v: L + v * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for g in range(0, 101, 25):
        out.append(f'<line x1="{x(g/100):.1f}" x2="{x(g/100):.1f}" y1="{T-8}" y2="{H-26}" class="grid"/>'
                   f'<text x="{x(g/100):.1f}" y="{H-8}" class="tick" text-anchor="middle">{g}%</text>')
    off = {s: (i - (len(series) - 1) / 2) * 9 for i, s in enumerate(series)}
    for i, (label, vals, ref) in enumerate(rows):
        y = T + height_per * i + height_per / 2
        out.append(f'<text x="{L-10}" y="{y+4:.1f}" class="lab" text-anchor="end">{html.escape(label)}</text>')
        if ref is not None:
            out.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="{y-13:.1f}" y2="{y+13:.1f}" class="ref" '
                       f'tabindex="0" data-tip="{html.escape(f"{pct(ref)} · {ref_label} · {label}")}"/>')
        for s in series:
            if s not in vals or vals[s][1] == 0:
                continue
            p, n = vals[s]
            lo, hi = wilson(p, n)
            yy = y + off[s]
            tip = f"{pct(p)} ({round(p*n)}/{n}, 95% CI {pct(lo)}-{pct(hi)}) · {SERIES[s]} · {label}"
            out.append(f'<g class="mark" tabindex="0" data-tip="{html.escape(tip)}">'
                       f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{COL[s]}" class="whisk"/>'
                       f'<circle cx="{x(p):.1f}" cy="{yy:.1f}" r="5" fill="{COL[s]}" class="dot"/>'
                       f'<rect x="{x(lo)-6:.1f}" y="{yy-8:.1f}" width="{x(hi)-x(lo)+12:.1f}" height="16" class="hit"/></g>')
    out.append("</svg>")
    return "".join(out)


def legend(series, ref_label=None):
    items = [f'<span><svg width="14" height="10"><circle cx="5" cy="5" r="4.5" fill="{COL[s]}"/></svg>{SERIES[s]}</span>'
             for s in series]
    if ref_label:
        items.append(f'<span><svg width="14" height="12"><line x1="5" x2="5" y1="0" y2="12" class="ref"/></svg>{ref_label}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


SERIES = {"wam": "AW-0 WAM (bf16, A1)", "code": "Shipped ASPIRE-style program (A2)",
          "fresh": "Fresh ASPIRE fix loop, Claude (F)", "fp32": "AW-0 WAM (fp32, A0-ref)"}
C = S["cells"]

# 1. per task, both arms
per_task = {}
for p in PROTOS:
    rows = [(t.replace("_", " "), {"wam": (C[f"{t}/{p}"]["wam_rate"] or 0, C[f"{t}/{p}"]["wam_n"]),
                                    "code": (C[f"{t}/{p}"]["code_rate"] or 0, C[f"{t}/{p}"]["code_n"])},
             C[f"{t}/{p}"]["paper_wam"]) for t in TASKS]
    per_task[p] = dot_chart(rows, ["wam", "code"], f"Success per task, {p}", "AW-0 paper, WAM")

# 2. gate: released-precision WAM vs paper
grows = [(f'{g["cell"].split("/")[0].replace("_", " ")} · {g["cell"].split("/")[1][:5]}',
          {"fp32": (g["ours"], g["n"])}, g["paper"]) for g in S["gate_released_fp32"]["cells"]]
gate_chart = dot_chart(grows, ["fp32"], "Gate: our fp32 WAM vs the paper", "AW-0 paper")

# 3. pooled difference forest plot (code - WAM)
P = S["pooled"]


def forest():
    W, L, R, T, rh = 640, 190, 24, 20, 40
    keys = [k for k in ("clean", "randomized", "both") if k in P]
    H = T + rh * len(keys) + 30
    x = lambda v: L + (v + 0.2) / 1.2 * (W - L - R)   # -20 .. +100 points
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Pooled difference, code minus WAM">']
    for g in (-0.2, 0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{T-6}" y2="{H-26}" class="{"zero" if g == 0 else "grid"}"/>'
                   f'<text x="{x(g):.1f}" y="{H-8}" class="tick" text-anchor="middle">{g*100:+.0f}</text>')
    for i, k in enumerate(keys):
        v = P[k]
        y = T + rh * i + rh / 2
        lo, hi = v["ci95"]
        tip = (f'{v["code_minus_wam"]*100:+.1f} points (95% CI {lo*100:+.1f} to {hi*100:+.1f}) · '
               f'code {pct(v["code"])} vs WAM {pct(v["wam"])} · {v["pairs"]} paired scenes · {k}')
        out.append(f'<text x="{L-10}" y="{y+4:.1f}" class="lab" text-anchor="end">{k} ({v["cells"]} cells)</text>'
                   f'<g class="mark" tabindex="0" data-tip="{html.escape(tip)}">'
                   f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--s2)" class="whisk"/>'
                   f'<circle cx="{x(v["code_minus_wam"]):.1f}" cy="{y:.1f}" r="5" fill="var(--s2)" class="dot"/>'
                   f'<rect x="{x(lo)-6:.1f}" y="{y-9:.1f}" width="{x(hi)-x(lo)+12:.1f}" height="18" class="hit"/></g>')
    out.append("</svg>")
    return "".join(out)


# 4. arm F on its task
F = S["arm_f_fresh_fixloop"]
ft = "blocks_ranking_rgb"
frows = []
for p in PROTOS:
    c = C[f"{ft}/{p}"]
    frows.append((p, {"wam": (c["wam_rate"] or 0, c["wam_n"]), "code": (c["code_rate"] or 0, c["code_n"]),
                      "fresh": (F[p]["rate"] or 0, F[p]["n"])}, c["paper_wam"]))
f_chart = dot_chart(frows, ["wam", "code", "fresh"], "blocks ranking rgb, three arms", "AW-0 paper, WAM", 44)


def cost_bars():
    items = COST["items"]
    W, L, R, T, rh = 640, 230, 70, 10, 30
    H = T + rh * len(items) + 24
    mx = max(i["usd"] for i in items)
    x = lambda v: L + v / mx * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Token cost by part, API-equivalent dollars">']
    for i, it in enumerate(items):
        y = T + rh * i
        tip = f'${it["usd"]:.2f} · {it["tokens"]/1e6:.1f}M tokens · {it["label"]}'
        out.append(f'<text x="{L-10}" y="{y+rh/2+4:.1f}" class="lab" text-anchor="end">{html.escape(it["label"])}</text>'
                   f'<g class="mark" tabindex="0" data-tip="{html.escape(tip)}">'
                   f'<rect x="{L}" y="{y+6}" width="{max(2, x(it["usd"])-L):.1f}" height="{rh-12}" rx="3" fill="var(--s1)" class="bar"/>'
                   f'<rect x="{L}" y="{y}" width="{W-L-R}" height="{rh}" class="hit"/></g>'
                   f'<text x="{x(it["usd"])+6:.1f}" y="{y+rh/2+4:.1f}" class="val">${it["usd"]:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def table():
    head = ("<tr><th>Task</th><th>Setting</th><th>WAM bf16</th><th>Program</th><th>Paired</th><th>WAM only</th>"
            "<th>Program only</th><th>McNemar p (Holm)</th><th>Paper WAM</th><th>Paper hybrid</th></tr>")
    rows = []
    for k, c in C.items():
        t, p = k.split("/")
        rows.append(f'<tr><td>{t}</td><td>{p}</td><td class="num">{pct(c["wam_rate"])} ({c["wam_n"]})</td>'
                    f'<td class="num">{pct(c["code_rate"])} ({c["code_n"]})</td><td class="num">{c["paired_n"]}</td>'
                    f'<td class="num">{c["wam_only"]}</td><td class="num">{c["code_only"]}</td>'
                    f'<td class="num">{c.get("mcnemar_p_holm", 1):.3g}</td><td class="num">{pct(c["paper_wam"])}</td>'
                    f'<td class="num">{pct(c["paper_hybrid"])}</td></tr>')
    return f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>'


def gate_table():
    rows = "".join(f'<tr><td>{g["cell"]}</td><td class="num">{pct(g["ours"])} ({g["n"]})</td>'
                   f'<td class="num">{pct(g["paper"])}</td><td class="num">{g["p"]:.2f}</td></tr>'
                   for g in S["gate_released_fp32"]["cells"])
    return (f'<div class="wrap"><table><tr><th>Cell</th><th>Ours, fp32</th><th>Paper</th><th>Fisher p</th></tr>'
            f'{rows}</table></div>')


frag = {"per_task_clean": per_task["clean"], "per_task_randomized": per_task["randomized"],
        "legend_arms": legend(["wam", "code"], "AW-0 paper, WAM"), "gate_chart": gate_chart,
        "legend_gate": legend(["fp32"], "AW-0 paper"), "forest": forest(), "f_chart": f_chart,
        "legend_f": legend(["wam", "code", "fresh"], "AW-0 paper, WAM"), "cost_chart": cost_bars(),
        "table": table(), "gate_table": gate_table()}
page = open(sys.argv[3]).read()
for k, v in frag.items():
    page = page.replace("{{" + k + "}}", v)
assert "{{" not in page, page[page.index("{{"):page.index("{{") + 40]
open(sys.argv[4], "w").write(page)
