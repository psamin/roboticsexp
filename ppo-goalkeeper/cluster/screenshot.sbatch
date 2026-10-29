#!/bin/bash
# SCREENSHOT: RTX render of the finished scene (no policy loaded). Preview: WIDTH=960 HEIGHT=540 SPP=64 SUFFIX=_preview
#SBATCH -J gk-shot
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
run_py scripts/capture_screenshot.py --width ${WIDTH:-2560} --height ${HEIGHT:-1440} --spp ${SPP:-512} \
    --mode ${MODE:-pt} --views ${VIEWS:-hero,angle} --suffix "${SUFFIX:-}" --out_dir "${OUT_DIR:-outputs/screenshots}" 2>&1 \
  | grep --line-buffered -vE "^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)"
echo SHOT_EXIT=${PIPESTATUS[0]}
