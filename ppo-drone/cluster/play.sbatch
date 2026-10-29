#!/bin/bash
# Playback video.  CKPT=<best.pt> VIDEO=<out.mp4> [EPISODES=3] [EPISODE_S=6]
#SBATCH -J drone-play
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
isaac_env
cd "$PROJ"
set +e
run_py scripts/play.py --checkpoint "$CKPT" --video "$VIDEO" --episodes ${EPISODES:-3} --episode_s ${EPISODE_S:-6} 2>&1 \
  | grep -E "\[play\]|PLAY|Traceback|Error|  File"
