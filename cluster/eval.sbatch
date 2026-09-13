#!/bin/bash
#SBATCH --job-name=rx-eval
#SBATCH --partition=general
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out
# Evaluate a checkpoint with delayexp.evaluate on a GPU node.
# Submit from the repo root on the login node:
#   sbatch cluster/eval.sbatch <checkpoint_dir> <name> [delayexp.evaluate flags]
#   sbatch cluster/eval.sbatch runs/baseline-seed1-<job>/train/PandaPickCube-*/checkpoints baseline --delays none

REPO="$SLURM_SUBMIT_DIR"
[ -r "$REPO/pyproject.toml" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
source "$REPO/cluster/setup_env.sh"

CKPT="${1:?usage: sbatch cluster/eval.sbatch <checkpoint_dir> <name> [flags]}"
NAME="${2:?usage: sbatch cluster/eval.sbatch <checkpoint_dir> <name> [flags]}"
shift 2
OUT="runs/eval-$NAME-$SLURM_JOB_ID"
mkdir -p "$OUT"
CMD=(python -m delayexp.evaluate --checkpoint "$CKPT" --out "$OUT" "$@")
echo "${CMD[*]}" > "$OUT/command.txt"
"${CMD[@]}"
