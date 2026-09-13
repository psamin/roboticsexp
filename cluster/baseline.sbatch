#!/bin/bash
#SBATCH --job-name=rx-baseline
#SBATCH --partition=general
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Smoke job 7441 on 1x L4: 1,638,400 steps in 124 s incl. one eval (~13k steps/s), JIT 42 s.
# The tuned config trains 24,576,000 steps (20M rounded up to whole epochs), about 31 min.
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm/%x-%j.out
# Milestone 3: baseline PPO on PandaPickCube with upstream's tuned config
# (mujoco_playground/config/manipulation_params.py: brax_ppo_config), unchanged.
# Submit from the repo root on the login node:  sbatch cluster/baseline.sbatch [seed]

REPO="$SLURM_SUBMIT_DIR"
[ -r "$REPO/pyproject.toml" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
source "$REPO/cluster/setup_env.sh"

SEED="${1:-1}"
RUN_DIR="runs/baseline-seed$SEED-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
python -m delayexp.provenance > "$RUN_DIR/provenance.json"
tar czf "$RUN_DIR/source.tar.gz" pyproject.toml uv.lock delayexp cluster scripts

# No PPO overrides: num_timesteps=20M and num_evals=4 come from the tuned config.
# brax rounds the step count up to whole epochs, so it reports the actual total.
TRAIN=(train-jax-ppo --env_name=PandaPickCube --impl=jax --seed="$SEED"
       --num_videos=2 --logdir="$RUN_DIR/train")
echo "${TRAIN[*]}" > "$RUN_DIR/commands.txt"
"${TRAIN[@]}"
