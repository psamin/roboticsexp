#!/bin/bash
# One PPO training run.   NAME=<run name> [VARIANT=p32] [BUDGET=60] [RESUME=ckpt.pt] [SETS="env.num_envs=8192 ..."]
#   NAME=base BUDGET=60 sbatch --export=ALL $HOME/roboticsexp/so101-bounce/cluster/train.sbatch
# Results: ~/so101-bounce-results/$NAME (checkpoints, metrics.csv, eval_history.json, train.log)
#SBATCH -J bounce-train
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/${NAME:?set NAME}
mkdir -p "$OUT"
$PY -m pytest -q tests 2>&1 | tail -3
ARGS=(--out "$OUT" --set train.budget_min=${BUDGET:-60})
[ -n "${VARIANT:-}" ] && ARGS+=(--variant "$VARIANT")
[ -n "${RESUME:-}" ] && ARGS+=(--resume "$RESUME")       # warm start from a checkpoint
for s in ${SETS:-}; do ARGS+=(--set "$s"); done
set +e
$PY scripts/train.py "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"
echo "TRAIN_EXIT ${PIPESTATUS[0]}"
# which randomised conditions the best checkpoint still fails in
$PY scripts/failure_analysis.py --checkpoint "$OUT/checkpoints/best.pt" --envs 8192 --device cuda 2>&1 | tee "$OUT/failures.txt"
# video of the best checkpoint (MuJoCo EGL rendering on the GPU node)
MUJOCO_GL=egl $PY scripts/make_video.py --checkpoint "$OUT/checkpoints/best.pt" --out "$OUT/video.mp4" 2>&1 | grep -v pix_fmt | tail -6
echo VIDEO_DONE
