#!/bin/bash
#SBATCH -J drone-shot
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
run_py scripts/screenshot.py --width 640 --height 360 --spp 16 --dump_settings --out "$OUT/settings_probe.png" 2>&1 | grep -E "settings|SCREENSHOT|Traceback|Error|  File"
