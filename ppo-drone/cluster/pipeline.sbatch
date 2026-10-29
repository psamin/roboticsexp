#!/bin/bash
# Full run in the required order: tests -> verify physics + baselines -> setup screenshot -> PPO smoke test ->
# PPO training (60 min budget) -> evaluation vs PD/random -> plots -> playback video -> trained screenshot.
# Every stage writes straight to $OUT in home, so anything finished survives a later failure.
#SBATCH -J drone-pipeline
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 12
#SBATCH --mem=110G
#SBATCH -t 2:15:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/ppo-drone}/cluster/common.sh"
OUT=${OUT:-$HOME_DIR/ppo-drone-results/run-${SLURM_JOB_ID}}
NUM_ENVS=${NUM_ENVS:-4096}
BUDGET_MIN=${BUDGET_MIN:-60}
mkdir -p "$OUT"/{logs,screenshots,plots,videos}
isaac_env
cd "$PROJ"
set +e
FILTER='^\s*$|\[Warning\]|\[Info\]|deprecated|UserWarning|warnings.warn|cuda capability|Minimum and Maximum|\(8.0\) - \(12.0\)|^\||isaaclab_tasks|freedreno|_internal.py|simulation_app.py|app_launcher.py|PythonExtension'
stage() { echo; echo "===== $1  $(date +%T)"; }
stage "1/9 unit tests"
"$PY" -m pytest -q tests 2>&1 | tail -3 | tee "$OUT/logs/pytest.txt"
stage "2/9 environment + physics checks, PD / random / untrained baselines"
run_py scripts/verify_env.py --num_envs 1024 --out "$OUT/logs/verify.json" 2>&1 | grep --line-buffered -vE "$FILTER"
stage "3/9 setup screenshot (before training)"
run_py scripts/screenshot.py --out "$OUT/screenshots/drone_ppo_setup.png" --spp ${SPP:-256} 2>&1 | grep --line-buffered -vE "$FILTER"
stage "4/9 PPO smoke test (2 min, 256 envs)"
run_py scripts/train.py --num_envs 256 --budget_min 2 --eval_every_min 1 --out "$OUT/smoke" 2>&1 | grep --line-buffered -vE "$FILTER"
stage "5/9 PPO training (${BUDGET_MIN} min, ${NUM_ENVS} envs)"
run_py scripts/train.py --num_envs "$NUM_ENVS" --budget_min "$BUDGET_MIN" --out "$OUT/train" 2>&1 | grep --line-buffered -vE "$FILTER"
stage "6/9 evaluation: PPO vs PD vs random (held-out test conditions)"
run_py scripts/evaluate.py --checkpoint "$OUT/train/checkpoints/best.pt" --out "$OUT/logs/eval_results.json" 2>&1 | grep --line-buffered -vE "$FILTER"
stage "7/9 plots"
"$PY" scripts/make_plots.py --run "$OUT/train" --eval "$OUT/logs/eval_results.json" --out "$OUT/plots"
stage "8/9 playback video (policy-driven)"
run_py scripts/play.py --checkpoint "$OUT/train/checkpoints/best.pt" --video "$OUT/videos/drone_ppo_demo.mp4" 2>&1 | grep --line-buffered -vE "$FILTER"
stage "9/9 trained screenshot"
run_py scripts/screenshot.py --checkpoint "$OUT/train/checkpoints/best.pt" --out "$OUT/screenshots/drone_ppo_trained.png" --spp ${SPP:-256} 2>&1 | grep --line-buffered -vE "$FILTER"
stage "done"
ls -laR "$OUT" | head -60
