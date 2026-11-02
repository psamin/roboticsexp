#!/bin/bash
# Round-11 evaluation of one checkpoint (full test at the deploy target 9 cm) on servo set $SERVO (default 12v_r10, fitted to the real-ball runs): target-height sweep
# first, then the full scenario test, each with and without the scripted launch on resting starts.
#   CKPT=v16-p32/checkpoints/best.pt sbatch --export=ALL .../cluster/eval11.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r11/<name>-sweep.txt and <name>.txt
#SBATCH -J bounce-eval
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:60:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/scenario_tests_r11
mkdir -p "$OUT"
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
name=$(echo "${c#$RESULTS/}" | sed 's#/checkpoints/#-#; s#/#-#g; s#\.pt$##')
ROWS="rest centre;rest centre + launch;rest half-way + launch;drop 10 cm;drop 20 cm, 6 cm off;held 1 s hidden;\
dead paddle e 0.65, resting + launch;dead paddle e 0.65, dropped 15 cm;paddle 1 deg off level + launch;\
camera latency 120 ms;timestamps 50 ms late;joint calibration 2 deg"
[ "${SWEEP:-1}" = 1 ] && $PY scripts/scenario_test.py --checkpoint "$c" --servo ${SERVO:-12v_r10} --per 256 --difficulties 0 \
  --targets 7,9,11,13 --only "$ROWS" --device cuda 2>&1 | grep -v -i "warn\|cuda capab\|Minimum and\|(8.0)" | tee "$OUT/$name-sweep.txt"
[ "${FULL:-1}" = 1 ] && $PY scripts/scenario_test.py --checkpoint "$c" --servo ${SERVO:-12v_r10} --targets ${TARGET:-9} --per 512 --device cuda 2>&1 \
  | grep -v -i "warn\|cuda capab\|Minimum and\|(8.0)" | tee "$OUT/$name.txt"
echo EVAL_DONE
