#!/bin/bash
# PLAYBACK: rendered, policy-driven penalty shots recorded to MP4.  CKPT=<best.pt> POLICIES=ppo,analytical,random
#SBATCH -J gk-play
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:40:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-goalkeeper}/cluster/common.sh"
export GK_ASSET_CACHE=$HOME_DIR/ppo-goalkeeper-cache
isaac_env
cd "$PROJ"
set +e
run_py scripts/play.py --checkpoint "$CKPT" --policies ${POLICIES:-ppo} --episodes ${EPISODES:-12} --level ${LEVEL:-2} \
    --video_dir "${VIDEO_DIR:-$HOME_DIR/ppo-goalkeeper-results/play-$SLURM_JOB_ID}" 2>&1 \
  | grep --line-buffered -vE '^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|h5py|freedreno'
echo PLAY_EXIT=${PIPESTATUS[0]}
