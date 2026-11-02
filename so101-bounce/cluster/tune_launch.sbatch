#!/bin/bash
# Launch-primitive tuning (scripts/tune_launch.py, CPU).   SERVO=12v_r10 [HANDOFF="ckpt ..."] sbatch --export=ALL ...
#SBATCH -J bounce-tune
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
mkdir -p "$RESULTS/launch"
for G in ${GAINS:-32 16}; do
  $PY scripts/${TUNER:-tune_launch}.py --p_gain $G --servo ${SERVO:-12v_r10} --out "$RESULTS/launch/${TUNER:-tune_launch}${TAG:-}_p${G}_${SERVO:-12v_r10}.json" \
    ${HANDOFF:+--handoff $HANDOFF} ${TARGS:-} 2>&1 | tee "$RESULTS/launch/${TUNER:-tune_launch}${TAG:-}_p${G}_${SERVO:-12v_r10}.txt"
done
echo TUNE_DONE
