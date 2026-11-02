#!/bin/bash
# Round-13 fine-tune evaluation (after jobs v19-p32-al / v19-p32-tilt): the runner's --auto-level (EnvCfg.auto_level) on servo set 12v_r10, steady target 9 cm, 1024
# episodes per row with 95 % intervals: v16-p32-lo with its round-10 launch and v18-p32-t12 without a scripted launch,
# each with and without auto-level, vs v16-p32-lo, on the tilt rows and a few general rows (regressions).
#   sbatch --dependency=afterany:<train jobs> .../cluster/eval13b.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r13/
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
ROWS="hand taps 1 Hz;drop 10 cm;bounced in (up 1.5 m/s);held 0.5 s visible;drop 40 cm;paddle 0 deg off level;paddle 1 deg off level;paddle 2 deg off level;paddle 3 deg off level;joint calibration 2 deg;rest rolling 0.2 m/s;dead paddle e 0.55, resting;dead paddle e 0.65, dropped 15 cm;rest centre;drop 20 cm, 6 cm off"
LROWS="$ROWS;paddle 0 deg off level + launch;paddle 1 deg off level + launch;paddle 2 deg off level + launch;paddle 3 deg off level + launch;rest rolling 0.2 m/s + launch;dead paddle e 0.55, resting + launch;rest centre + launch"
run() {   # name ckpt launch extra...
  local name=$1 ck=$2 la=$3; shift 3
  $PY scripts/scenario_test.py --checkpoint "$RESULTS/$ck" --servo 12v_r10 --targets 9 --per ${PER:-1024} \
    --difficulties 0 --launch $la --ci --device cuda --only "$([ $la = off ] && echo "$ROWS" || echo "$LROWS")" "$@" \
    2>&1 | grep -v -i "$F" | tee "$OUT/$name.txt"
}
run v19-p32-al-al v19-p32-al/checkpoints/best.pt both --auto_level
run v19-p32-al-off v19-p32-al/checkpoints/best.pt both
run v19-p32-tilt-off v19-p32-tilt/checkpoints/best.pt both
run v19-p32-tilt-al v19-p32-tilt/checkpoints/best.pt both --auto_level
run v16-p32-lo-off-b v16-p32-lo/checkpoints/best.pt both
run v16-p32-lo-al-b v16-p32-lo/checkpoints/best.pt both --auto_level
echo EVAL_DONE
