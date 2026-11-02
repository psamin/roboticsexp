#!/bin/bash
# Servo model fit to real-arm logs on a GPU.   LOGS="real/logs/run-a.npz ..." [ARGS="--n 4096"] sbatch --export=ALL ...
#SBATCH -J bounce-fit
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:40:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
mkdir -p "$RESULTS/servo_fit"
$PY scripts/fit_servo.py ${LOGS:?set LOGS} --base 12v --device cuda ${ARGS:-} 2>&1 | tee "$RESULTS/servo_fit/fit-$SLURM_JOB_ID.txt"
echo FIT_DONE
