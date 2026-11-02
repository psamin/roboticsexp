#!/bin/bash
# SmolVLA fine-tune on one GPU: LoRA at a given rank, or full fine-tuning. Same optimizer/steps/seed across runs.
#   Smoke test (rank 32, public SO-100 data, validates the PEFT pipeline before our dataset exists):
#     MODE=lora RANK=32 STEPS=300 DATASET=lerobot/svla_so100_pickplace sbatch flicklora/cluster/train_lora.sbatch
#   Flick ablation (one job per rank, see run_ablation.sh):
#     MODE=lora RANK=8 DATASET_DIR=<local lerobot dataset> sbatch ...
#SBATCH -J flick-lora
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
MODE=${MODE:-lora}            # lora | full
RANK=${RANK:-32}
ALPHA=${ALPHA:-$RANK}         # alpha == r for every rank (scaling 1.0)
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
SEED=${SEED:-1000}
TAG=${TAG:-$MODE-r$RANK}
DATASET=${DATASET:-}          # hub repo id, or
DATASET_DIR=${DATASET_DIR:-}  # a LeRobot dataset in home, or
RAW_DIR=${RAW_DIR:-}          # raw expert episodes (run_expert.py collect); converted here, deterministically
# LoRA: 10x the full-FT LR, per the LeRobot PEFT guide. Full FT uses SmolVLA's defaults.
if [ "$MODE" = lora ]; then LR=${LR:-1e-3}; LR_END=${LR_END:-1e-4}; else LR=${LR:-1e-4}; LR_END=${LR_END:-2.5e-6}; fi
OUT=$RESULTS/train-$TAG-$SLURM_JOB_ID
mkdir -p "$OUT"
lerobot_env

if [ -n "$RAW_DIR" ]; then
  "$LR_PY" "$PROJ/train/to_lerobot.py" --raw "$RAW_DIR" --out "$WORK/ds" --repo_id local/flick_train 2>&1 | tail -3
  DS_ARGS=(--dataset.repo_id=local/flick_train --dataset.root="$WORK/ds")
  mkdir -p "$OUT/dataset" && cp -r "$WORK/ds/meta" "$OUT/dataset/" && cp "$WORK/ds/episode_seeds.json" "$OUT/dataset/"
elif [ -n "$DATASET_DIR" ]; then
  cp -r "$DATASET_DIR" "$WORK/ds"          # read the dataset from RAM, not NFS
  DS_ARGS=(--dataset.repo_id="local/$(basename "$DATASET_DIR")" --dataset.root="$WORK/ds")
else
  DS_ARGS=(--dataset.repo_id="${DATASET:?set DATASET or DATASET_DIR}")
fi
PEFT_ARGS=()
[ "$MODE" = lora ] && PEFT_ARGS=(--peft.method_type=LORA --peft.r="$RANK" --peft.lora_alpha="$ALPHA" ${TARGETS:+--peft.target_modules="$TARGETS"})

# which modules get adapters, trainable parameter count (cheap, no data)
if [ "$MODE" = lora ]; then
  "$LR_PY" "$PROJ/train/peft_inspect.py" --ranks "$RANK" --out "$OUT" ${TARGETS:+--target_modules "$TARGETS"} 2>&1 | grep -v Warning | tail -8 || echo "peft_inspect failed (non-fatal)"
fi

# peak GPU memory: sample nvidia-smi every second
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu_mem.txt" &
SMI=$!
save() { kill $SMI 2>/dev/null || true
  local last=$WORK/out/checkpoints/last/pretrained_model
  [ -d "$last" ] && cp -rL "$last" "$OUT/pretrained_model" && echo "saved $last -> $OUT"
  echo "{\"peak_gpu_mem_mib\": $(sort -n "$OUT/gpu_mem.txt" | tail -1), \"train_seconds\": $(( $(date +%s) - T0 ))," \
       "\"mode\": \"$MODE\", \"rank\": $RANK, \"alpha\": $ALPHA, \"steps\": $STEPS, \"batch\": $BATCH, \"seed\": $SEED," \
       "\"lr\": $LR, \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"}" > "$OUT/train_stats.json"
  cat "$OUT/train_stats.json"; }
trap 'save; rm -rf "$WORK"' EXIT
trap 'echo "walltime close: saving"; exit 1' USR1 TERM INT
T0=$(date +%s)
"$LR_BIN/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  "${DS_ARGS[@]}" \
  --dataset.video_backend=pyav \
  --policy.input_features=null --policy.output_features=null \
  --policy.optimizer_lr="$LR" --policy.scheduler_decay_lr="$LR_END" --policy.scheduler_decay_steps="$STEPS" \
  "${PEFT_ARGS[@]}" \
  --batch_size="$BATCH" --steps="$STEPS" --seed="$SEED" \
  --save_freq="$STEPS" --log_freq=50 --num_workers=10 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir="$WORK/out" --job_name="flick-$TAG" 2>&1 | tee "$OUT/train.log" | grep -vE "Warning|warn\(" &
wait $!
echo TRAIN_DONE
