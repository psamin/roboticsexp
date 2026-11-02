#!/bin/bash
# Round-12 evaluation of one checkpoint on servo set 12v_r10, steady target 9 cm, with its launch (LNAME, a name in
# bounce/launch.py NAMED; empty = the round-10 default): the full scenario test with the high-then-low target schedule
# (SCHED start_cm,high_s,ramp_s) and without it.
#   CKPT=v18-p32-sched/checkpoints/best.pt LNAME=r12 sbatch --export=ALL .../cluster/eval12.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r12/<name>-sched.txt and <name>.txt
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
OUT=$RESULTS/scenario_tests_r12
mkdir -p "$OUT"
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
name=$(echo "${c#$RESULTS/}" | sed 's#/checkpoints/#-#; s#/#-#g; s#\.pt$##')
F='warn\|cuda capab\|Minimum and\|(8.0)'
$PY scripts/scenario_test.py --checkpoint "$c" --servo 12v_r10 --targets ${TARGET:-9} --sched ${SCHED:-16,1.5,1.0} \
  ${LNAME:+--launch_name $LNAME} --per 512 --device cuda 2>&1 | grep -v -i "$F" | tee "$OUT/$name-sched.txt"
$PY scripts/scenario_test.py --checkpoint "$c" --servo 12v_r10 --targets ${TARGET:-9} \
  ${LNAME:+--launch_name $LNAME} --per 512 --device cuda 2>&1 | grep -v -i "$F" | tee "$OUT/$name.txt"
echo EVAL_DONE
