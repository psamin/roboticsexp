#!/bin/bash
# Round-13b evaluation: does a harder flick now beat the round-10 launch? Resting starts (centre, half-way, rim,
# rolling, dead paddle) with the launch, servo set 12v_r10, steady target 9 cm, 1024 episodes per row with 95 %
# intervals and the time to the first steady bounce. v20-p32-flick with the hard launch r13 (and with r10), vs
# v16-p32-lo with r10 and with r13.
#   sbatch --dependency=afterany:<train job> .../cluster/eval13c.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r13/flick-*.txt
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
OUT=$RESULTS/scenario_tests_r13
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
ROWS="rest centre + launch;rest half-way + launch;rest near rim + launch;rest rolling 0.2 m/s + launch;rest near rim, rolling 0.15 m/s + launch;dead paddle e 0.55, resting + launch;dead paddle e 0.65, resting + launch;paddle 1 deg off level + launch;drop 20 cm, 6 cm off;bounced in (up 1.5 m/s)"
run() {   # name ckpt launch_name
  $PY scripts/scenario_test.py --checkpoint "$RESULTS/$2" --servo 12v_r10 --targets 9 --per ${PER:-1024} \
    --difficulties 0,0.5 --launch on --ci --device cuda --only "$ROWS" ${3:+--launch_name $3} 2>&1 | grep -v -i "$F" \
    | tee "$OUT/flick-$1.txt"
}
run v20-r13 v20-p32-flick/checkpoints/best.pt r13
run v20-r10 v20-p32-flick/checkpoints/best.pt ""
run v16-r10 v16-p32-lo/checkpoints/best.pt ""
run v16-r13 v16-p32-lo/checkpoints/best.pt r13
echo EVAL_DONE
