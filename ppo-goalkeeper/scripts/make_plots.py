"""PLOTS: learning curves and evaluation bars from a finished (or running) training run. No Isaac needed."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from goalkeeper.visualization.plots import eval_bars, learning_curves

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/run")
p.add_argument("--eval", default="outputs/logs/eval_results.json")
p.add_argument("--out", default="outputs/plots")
a = p.parse_args()

baselines = None
if os.path.exists(a.eval):
    with open(a.eval) as f:
        res = json.load(f)
    baselines = {k: v for k, v in res["policies"].items() if k != "ppo"}
    print(eval_bars(res, os.path.join(a.out, "eval_save_rates.png")))
for f in learning_curves(a.run, a.out, baselines):
    print(f)
