#!/bin/bash
# Ball placement test.   NAME=<run> [CKPT=best.pt] sbatch --export=ALL .../cluster/position.sbatch
#SBATCH -J bounce-position
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 0:20:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/${NAME:?set NAME}
$PY scripts/position_test.py --checkpoint "$OUT/checkpoints/${CKPT:-best.pt}" --envs 1024 --device cuda 2>&1 | tee "$OUT/position_test.txt"
echo POSITION_DONE
