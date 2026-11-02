#!/bin/bash
# Failure analysis of a run's checkpoint.   NAME=<run> [CKPT=best.pt] sbatch --export=ALL .../cluster/analysis.sbatch
#SBATCH -J bounce-analysis
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
$PY scripts/failure_analysis.py --checkpoint "$OUT/checkpoints/${CKPT:-best.pt}" --envs 8192 --device cuda 2>&1 | tee "$OUT/failures.txt"
echo ANALYSIS_DONE
