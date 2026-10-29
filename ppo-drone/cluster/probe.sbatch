#!/bin/bash
#SBATCH -J drone-probe
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:25:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source /nethome-instruction/pi34/roboticsexp/ppo-drone/cluster/common.sh
OUT=$HOME_DIR/ppo-drone-results/preview
mkdir -p "$OUT"
isaac_env
cd "$PROJ"
set +e
run_py scripts/render_probe.py --out "$OUT/probe" 2>&1 | grep -E "\[light\]|\[ground\]|\[setting\]|\[probe\]|PROBE|Traceback|Error|  File"
