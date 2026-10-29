#!/bin/bash
# Full run, in order: BUILD -> VERIFY -> SCREENSHOT -> PPO smoke test -> TRAIN (60 min budget) -> EVALUATE -> PLOTS
# -> PLAYBACK videos. Each stage writes straight to $OUT in home, so a stage that fails or a walltime kill keeps
# everything produced before it. Training stops itself at the budget and keeps latest.pt / best.pt.
#SBATCH -J gk-pipeline
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 12
#SBATCH --mem=110G
#SBATCH -t 2:15:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-goalkeeper}/cluster/common.sh"
export GK_ASSET_CACHE=$HOME_DIR/ppo-goalkeeper-cache
OUT=${OUT:-$HOME_DIR/ppo-goalkeeper-results/run-${SLURM_JOB_ID}}
NUM_ENVS=${NUM_ENVS:-1024}
BUDGET_MIN=${BUDGET_MIN:-60}
mkdir -p "$OUT"/{logs,screenshots,plots,videos}
isaac_env
cd "$PROJ"
set +e
FILTER='^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|h5py|freedreno|_internal.py|simulation_app.py|app_launcher.py|PythonExtension'
stage() { echo; echo "===== $1  $(date +%T)"; }

stage "1/8 unit tests"
"$PY" -m pytest -q tests 2>&1 | tail -3 | tee "$OUT/logs/pytest.txt"

stage "2/8 build + verify physics and baselines"
run_py scripts/verify_env.py --num_envs "$NUM_ENVS" --out "$OUT/logs/verify.json" 2>&1 | grep --line-buffered -vE "$FILTER"

stage "3/8 screenshot (before any training)"
run_py scripts/capture_screenshot.py --out_dir "$OUT/screenshots" --spp ${SPP:-512} 2>&1 | grep --line-buffered -vE "$FILTER"

stage "4/8 PPO smoke test (2 min, 256 envs)"
run_py scripts/train.py --num_envs 256 --budget_min 2 --eval_every_min 1 --out "$OUT/smoke" 2>&1 \
  | grep --line-buffered -vE "$FILTER"

stage "5/8 PPO training (${BUDGET_MIN} min budget, ${NUM_ENVS} envs)"
run_py scripts/train.py --num_envs "$NUM_ENVS" --budget_min "$BUDGET_MIN" --out "$OUT/train" 2>&1 \
  | grep --line-buffered -vE "$FILTER"

stage "6/8 evaluate best checkpoint vs baselines (held-out test shots)"
run_py scripts/evaluate.py --checkpoint "$OUT/train/checkpoints/best.pt" --out "$OUT/logs/eval_results.json" 2>&1 \
  | grep --line-buffered -vE "$FILTER"

stage "7/8 plots"
"$PY" scripts/make_plots.py --run "$OUT/train" --eval "$OUT/logs/eval_results.json" --out "$OUT/plots"

stage "8/8 playback videos (policy-driven, rendered)"
run_py scripts/play.py --checkpoint "$OUT/train/checkpoints/best.pt" --policies ppo,analytical,random \
  --episodes ${EPISODES:-12} --video_dir "$OUT/videos" 2>&1 | grep --line-buffered -vE "$FILTER"

stage "done"
ls -laR "$OUT" | head -80
