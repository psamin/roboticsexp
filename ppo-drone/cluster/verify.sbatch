#!/bin/bash
# Unit tests, environment/physics checks, baselines, and a preview of the setup screenshot.
#SBATCH -J drone-verify
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:40:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
OUT=$HOME_DIR/ppo-drone-results/verify-$SLURM_JOB_ID
mkdir -p "$OUT"
isaac_env
cd "$PROJ"
set +e
FILTER='^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|freedreno|_internal.py|simulation_app.py|app_launcher.py|PythonExtension'
"$PY" -m pytest -q tests 2>&1 | tail -3
run_py scripts/verify_env.py --num_envs ${NUM_ENVS:-1024} --out "$OUT/verify.json" 2>&1 | grep --line-buffered -vE "$FILTER"
run_py scripts/screenshot.py --width 1280 --height 720 --spp 64 --out "$OUT/drone_ppo_setup_preview.png" 2>&1 | grep --line-buffered -vE "$FILTER"
echo VERIFY_JOB_DONE
