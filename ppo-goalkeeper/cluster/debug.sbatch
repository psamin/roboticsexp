#!/bin/bash
#SBATCH -J gk-debug
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:20:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-goalkeeper}/cluster/common.sh"
export GK_ASSET_CACHE=$HOME_DIR/ppo-goalkeeper-cache
isaac_env
cd "$PROJ"
set +e
"$PY" -c "import numpy; print('numpy', numpy.__version__)"
run_py scripts/${SCRIPT:-debug_drive.py} 2>&1 | grep --line-buffered -E "^k=|^env_|DEBUG|Traceback|Error|error|  File|numpy"
