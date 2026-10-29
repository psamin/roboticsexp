#!/bin/bash
# Final media for a trained run: setup close-up (no policy), trained close-up, and the policy-driven video.
#   RUN=<results dir with train/checkpoints/best.pt> [STYLE=isaac]
#SBATCH -J drone-media
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
isaac_env
cd "$PROJ"
set +e
ST=${STYLE:-isaac}
CK=$RUN/train/checkpoints/best.pt
mkdir -p "$RUN/media"
F='\[screenshot\]|SCREENSHOT|\[play\]|PLAY|Traceback|Error|  File'
run_py scripts/screenshot.py --style $ST --spp 384 --out "$RUN/media/drone_ppo_setup_$ST.png" 2>&1 | grep -E "$F"
run_py scripts/screenshot.py --style $ST --spp 384 --checkpoint "$CK" --out "$RUN/media/drone_ppo_trained_$ST.png" 2>&1 | grep -E "$F"
run_py scripts/play.py --style $ST --checkpoint "$CK" --episodes 3 --episode_s 6 --video "$RUN/media/drone_ppo_demo_$ST.mp4" 2>&1 | grep -E "$F"
echo MEDIA_DONE
