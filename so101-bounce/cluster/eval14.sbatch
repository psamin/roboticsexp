#!/bin/bash
# Round-14 evaluation of one checkpoint: the full scenario suite (round-10 rows plus the oracle's recoverable envelope)
# with the runner's --auto-level, servo set 12v_r10, steady target 9 cm, 1024 episodes per row, 95 % intervals,
# difficulty 0 and 0.5.
#   CKPT=v21-p32-any6/checkpoints/best.pt NAME=v21-any6 sbatch --export=ALL .../cluster/eval14.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r14/$NAME.txt
#SBATCH -J bounce-eval
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/scenario_tests_r14
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
$PY scripts/scenario_test.py --checkpoint "$c" --servo 12v_r10 --targets 9 --per ${PER:-1024} --difficulties 0,0.5 \
  --auto_level --ci --device cuda 2>&1 | grep -v -i "$F" | tee "$OUT/${NAME:?set NAME}.txt"
echo EVAL_DONE
