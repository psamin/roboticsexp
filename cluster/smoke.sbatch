#!/bin/bash
#SBATCH --job-name=rx-smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/%x-%j.out
# Milestone 2 end-to-end smoke test with upstream train-jax-ppo:
#   1. train one epoch at the baseline's real batch shapes -> checkpoint -> rollout video
#   2. reload that checkpoint with --play_only -> the same rollout video again
#   3. compare the two videos
# Submit from the repo root on the login node:  sbatch cluster/smoke.sbatch

REPO="$SLURM_SUBMIT_DIR"
[ -r "$REPO/pyproject.toml" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
source "$REPO/cluster/setup_env.sh"

RUN_DIR="runs/smoke-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
python -m delayexp.provenance > "$RUN_DIR/provenance.json"
tar czf "$RUN_DIR/source.tar.gz" pyproject.toml uv.lock delayexp cluster scripts

# brax runs at least one epoch: batch_size * unroll_length * num_minibatches *
# num_resets_per_eval = 512 * 10 * 32 * 10 = 1,638,400 env steps with the tuned config.
TRAIN=(train-jax-ppo --env_name=PandaPickCube --impl=jax --seed=1
       --num_timesteps=1638400 --num_evals=2 --num_videos=1 --logdir="$RUN_DIR/train")
echo "${TRAIN[*]}" > "$RUN_DIR/commands.txt"
"${TRAIN[@]}"

CKPT_DIR=$(ls -d "$RUN_DIR"/train/PandaPickCube-*/checkpoints)
RELOAD=(train-jax-ppo --env_name=PandaPickCube --impl=jax --seed=1 --play_only
        --load_checkpoint_path="$CKPT_DIR" --num_videos=1 --logdir="$RUN_DIR/reload")
echo "${RELOAD[*]}" >> "$RUN_DIR/commands.txt"
"${RELOAD[@]}"

# Both runs roll out the deterministic policy from PRNGKey(seed), so identical
# params should give the same frames (GPU nondeterminism could still add small diffs).
python - "$RUN_DIR" <<'EOF'
import glob, sys
import mediapy, numpy as np
run = sys.argv[1]
a = mediapy.read_video(glob.glob(f"{run}/train/*/rollout0.mp4")[0]).astype(int)
b = mediapy.read_video(glob.glob(f"{run}/reload/*/rollout0.mp4")[0]).astype(int)
if a.shape != b.shape:
    sys.exit(f"video shapes differ: {a.shape} vs {b.shape}")
diff = np.abs(a - b)
print(f"train vs reload rollout video: {a.shape}, max pixel diff {diff.max()}, mean {diff.mean():.4f}")
EOF
