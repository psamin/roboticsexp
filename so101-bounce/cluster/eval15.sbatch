#!/bin/bash
# Round-15 evaluation of one checkpoint on the servo set fitted to the real speed test (12v_r11): the full scenario
# suite with the runner's --auto-level, steady target 9 cm, 1024 episodes per row, 95 % intervals, difficulty 0 and
# 0.5, the same 10 cm paddle limit for every checkpoint (the real stroke rises past 6 cm).
#   CKPT=policies/v16-p32-lo.pt NAME=v16-p32-lo-r11 [SERVO=12v_r11] sbatch --export=ALL .../cluster/eval15.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r15/$NAME.txt
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
OUT=$RESULTS/scenario_tests_r15
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
$PY scripts/scenario_test.py --checkpoint "$c" --servo ${SERVO:-12v_r11} --paddle_box ${BOX:-0.12,0.10} --targets 9 \
  --per ${PER:-1024} --difficulties 0,0.5 --auto_level --ci --device cuda 2>&1 | grep -v -i "$F" | tee "$OUT/${NAME:?set NAME}.txt"
echo EVAL_DONE
