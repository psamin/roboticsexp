#!/bin/bash
# PPO smoke test: short headless training run, then evaluation and plots from its outputs.
#SBATCH -J gk-smoke
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-goalkeeper}/cluster/common.sh"
export GK_ASSET_CACHE=$HOME_DIR/ppo-goalkeeper-cache
OUT=$HOME_DIR/ppo-goalkeeper-results/smoke-$SLURM_JOB_ID
isaac_env
cd "$PROJ"
set +e
FILTER='^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|h5py|freedreno|_internal.py|simulation_app.py|app_launcher.py|PythonExtension'
run_py scripts/train.py --num_envs ${NUM_ENVS:-256} --budget_min ${BUDGET_MIN:-3} --eval_every_min 1 --out "$OUT/train" 2>&1 | grep --line-buffered -vE "$FILTER"
run_py scripts/evaluate.py --checkpoint "$OUT/train/checkpoints/best.pt" --num_envs 256 --out "$OUT/eval_results.json" 2>&1 | grep --line-buffered -vE "$FILTER"
"$PY" scripts/make_plots.py --run "$OUT/train" --eval "$OUT/eval_results.json" --out "$OUT/plots"
ls -la "$OUT" "$OUT/train" "$OUT/plots"
echo SMOKE_EXIT
