#!/bin/bash
# Render one screenshot.  OUT_PNG=<path> [CKPT=<best.pt>] [CAMERA=closeup|setup|hover|wide] [SPP=256]
#SBATCH -J drone-shot
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:25:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
isaac_env
cd "$PROJ"
set +e
run_py scripts/screenshot.py --out "$OUT_PNG" --camera ${CAMERA:-closeup} --spp ${SPP:-256} ${CKPT:+--checkpoint $CKPT} 2>&1 \
  | grep -E "\[screenshot\]|SCREENSHOT|Traceback|Error|  File"
