#!/bin/bash
# Waypoint-tour video.  CKPT=<best.pt> VIDEO=<out.mp4> [STYLE=isaac]
#SBATCH -J drone-tour
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
isaac_env
cd "$PROJ"
set +e
run_py scripts/tour.py --checkpoint "$CKPT" --video "$VIDEO" --style ${STYLE:-isaac} 2>&1 | grep -E "\[tour\]|TOUR|Traceback|Error|  File"
