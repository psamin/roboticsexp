#!/bin/bash
# One reward-shaping / curriculum experiment: train a variant from configs/variants.yaml, then evaluate it on all
# held-out suites (hover, reach, robust, and the never-trained recovery suite) with stability metrics.
#   VARIANT=<name> [BUDGET_MIN=20] [NUM_ENVS=4096] [BASELINES=1 to also evaluate PD and random actions]
#SBATCH -J drone-exp
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
OUT=$HOME_DIR/ppo-drone-results/exp/$VARIANT
mkdir -p "$OUT"
isaac_env
cd "$PROJ"
set +e
F='^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|freedreno|_internal.py|simulation_app.py|app_launcher.py|PythonExtension'
echo "== variant $VARIANT"
run_py scripts/train.py --variant "$VARIANT" --budget_min ${BUDGET_MIN:-20} --num_envs ${NUM_ENVS:-4096} \
  --out "$OUT/train" 2>&1 | grep --line-buffered -vE "$F"
POL=ppo; [ "${BASELINES:-0}" = 1 ] && POL=ppo,pd,random
run_py scripts/evaluate.py --checkpoint "$OUT/train/checkpoints/best.pt" --policies $POL --out "$OUT/eval.json" 2>&1 \
  | grep --line-buffered -vE "$F"
echo EXP_DONE
