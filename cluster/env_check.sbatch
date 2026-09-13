#!/bin/bash
#SBATCH --job-name=rx-envcheck
#SBATCH --partition=general
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out
# Milestone 1 on a GPU node: the env loads, JAX sees the GPU, headless EGL rendering works.
# Submit from the repo root on the login node:  sbatch cluster/env_check.sbatch

# /nethome/<user> has been a self-referencing symlink on some compute nodes;
# /nethome-instruction/<user> serves the same files.
REPO="$SLURM_SUBMIT_DIR"
[ -r "$REPO/pyproject.toml" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
source "$REPO/cluster/setup_env.sh"

python -m delayexp.env_check --out "runs/env_check-$SLURM_JOB_ID"
