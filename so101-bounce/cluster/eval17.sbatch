#!/bin/bash
# Round-17 evaluation: the round-17 picks against the deploy pick v23-p32-learn16 on how the user starts the ball
# (drops), beside drops, tosses and rig faults. No scripted launch, handle keep-out 6 cm, each checkpoint's own paddle
# box, targets 10 and 14 cm, 1024 episodes per row, 95 % intervals, difficulty 0 and 0.5.
#   CKPT=policies/v23-p32-learn16.pt NAME=v23-p32-learn16 sbatch --export=ALL .../cluster/eval17.sbatch
#   CKPT=v25-reach-3/checkpoints/best.pt NAME=v25-reach-3 sbatch --export=ALL .../cluster/eval17.sbatch
# Results: ~/so101-bounce-results/scenario_tests_r17/$NAME.txt
#SBATCH -J bounce-eval
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/scenario_tests_r17
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
ROWS="drop 10 cm;drop 20 cm, 6 cm off;drop 40 cm;beside 8 cm, 20 cm up (reachable);beside 12 cm, 30 cm up (reachable);\
beside 15 cm, 40 cm up (reachable);toss 0.2 m/s toward centre;toss 0.3 m/s sideways;held 1 s hidden;\
camera latency 150 ms;depth scale +10 %;joint calibration 2 deg;servos 15 % weaker (hot);ball restitution 0.75;\
dead paddle e 0.65, dropped 15 cm"
c=${CKPT:?set CKPT}
[ -f "$c" ] || c="$RESULTS/$c"
$PY scripts/scenario_test.py --checkpoint "$c" --servo ${SERVO:-12v_r11} --keepout 0.06 --launch off \
  --targets ${TARGET:-10,14} --per ${PER:-1024} --difficulties 0,0.5 --ci --only "$ROWS" --device cuda 2>&1 \
  | grep -v -i "$F" | tee "$OUT/${NAME:?set NAME}.txt"
echo EVAL_DONE
