#!/bin/bash
# Round-15b evaluation (learned launch vs scripted launch) on the fitted servo set: resting starts with and without
# the scripted launch, drops, dead paddles, tilt and camera rows, with the runner's --auto-level, the handle keep-out
# (a ball coming down > 6 cm behind the paddle centre is lost), a 10 cm paddle limit, target 10 cm, 1024 episodes per
# row, 95 % intervals, difficulty 0 and 0.5.
#   CKPT=policies/v18-p32-t12.pt NAME=v18-p32-t12 LAUNCH=off sbatch --export=ALL .../cluster/eval15b.sbatch
#   CKPT=policies/v16-p32-lo.pt NAME=v16-p32-lo-lean6 LAUNCH=on LEAN=6 sbatch --export=ALL .../cluster/eval15b.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r15b/$NAME.txt
#SBATCH -J bounce-eval
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:25:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/scenario_tests_r15b
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
ROWS="rest centre;rest centre + launch;rest half-way;rest half-way + launch;rest near rim;rest near rim + launch;\
rest rolling 0.2 m/s;rest rolling 0.2 m/s + launch;rest near rim, rolling 0.15 m/s;rest near rim, rolling 0.15 m/s + launch;\
drop 10 cm;drop 20 cm, 6 cm off;drop 40 cm;beside 8 cm, 20 cm up (reachable);held 1 s hidden;toss 0.2 m/s toward centre;\
paddle 2 deg off level;paddle 2 deg off level + launch;paddle 3 deg off level;paddle 3 deg off level + launch;\
camera latency 150 ms;depth scale +10 %;15 fps, dim;hand taps 1 Hz;joint calibration 2 deg;\
dead paddle e 0.65, resting;dead paddle e 0.65, resting + launch;dead paddle e 0.65, dropped 15 cm;\
dead paddle e 0.55, resting;dead paddle e 0.55, resting + launch"
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
$PY scripts/scenario_test.py --checkpoint "$c" --servo ${SERVO:-12v_r11} --paddle_box 0.12,0.10 --keepout 0.06 \
  --launch ${LAUNCH:-off} --launch_lean ${LEAN:-0} --targets ${TARGET:-10} --per ${PER:-1024} --difficulties 0,0.5 \
  --auto_level --ci --only "$ROWS" --device cuda 2>&1 | grep -v -i "$F" | tee "$OUT/${NAME:?set NAME}.txt"
echo EVAL_DONE
