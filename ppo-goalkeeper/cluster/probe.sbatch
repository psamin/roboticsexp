#!/bin/bash
#SBATCH -J gk-probe
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-goalkeeper}/cluster/common.sh"
isaac_env
"$UV" pip list 2>/dev/null | grep -iE "^(isaacsim |isaaclab|torch|warp|numpy|gymnasium)" || true
set +e
run_py "$PROJ/scripts/probe_isaaclab.py" 2>&1 | grep -vE "^\s*$|\[Warning\].*(deprecated|carb.audio)" | tail -150
echo PROBE_EXIT=$?
