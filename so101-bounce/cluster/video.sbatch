#!/bin/bash
# Video of a finished run's best checkpoint.   NAME=<run name> sbatch --export=ALL .../cluster/video.sbatch
#SBATCH -J bounce-video
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/${NAME:?set NAME}
set +e
MUJOCO_GL=egl $PY scripts/make_video.py --checkpoint "$OUT/checkpoints/${CKPT:-best.pt}" --out "$OUT/${VIDEO:-video.mp4}" 2>&1 | grep -v pix_fmt | tail -8
echo VIDEO_DONE
