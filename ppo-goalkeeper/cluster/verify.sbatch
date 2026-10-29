#!/bin/bash
# BUILD + VERIFY: unit tests, physics checks, random/still/analytical baselines.
#SBATCH -J gk-verify
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
"$PY" -m pytest -q tests 2>&1 | tail -5
run_py scripts/verify_env.py --num_envs ${NUM_ENVS:-1024} 2>&1 | grep --line-buffered -vE "^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)"
echo VERIFY_EXIT=${PIPESTATUS[0]}
