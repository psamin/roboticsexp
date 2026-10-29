"""PLOTS from a training run and an evaluation file (no Isaac needed).

  python scripts/make_plots.py --run outputs/run --eval outputs/logs/eval_results.json --out outputs/plots
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_ppo.plots import eval_comparison, training_curves

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/run")
p.add_argument("--eval", default="outputs/logs/eval_results.json")
p.add_argument("--out", default="outputs/plots")
a = p.parse_args()
for f in training_curves(a.run, a.out):
    print(f)
if os.path.exists(a.eval):
    print(eval_comparison(json.load(open(a.eval)), a.out))
