#!/bin/bash
# Does LeRobot 0.6.1 + SmolVLA run on a DGX Spark (aarch64, GB10)? Decides whether eval can co-locate policy + sim.
#SBATCH -J flick-sparklr
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:40:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
set +e +o pipefail
t0=$(date +%s); lerobot_env; echo "lerobot install took $(( $(date +%s) - t0 ))s"
"$LR_PY" "$PROJ/train/infer_check.py" 2>&1 | grep -vE "Warning|warn\(" | tail -15
echo SPARK_LR_DONE
