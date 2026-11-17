#!/bin/bash
# Build the RoboTwin + AW-0 env from scratch, then run one released-WAM episode (click_alarmclock, clean)
# in bf16 and in fp32, logging GPU memory. Proves the pipeline end to end before any scale-up.
#SBATCH -J aa-smoke-wam
#SBATCH -p general
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/smoke-wam-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
build_all

OUT=$RESULTS/smoke-wam-$SLURM_JOB_ID; mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$OUT/gpu_mem.csv" &
cd "$AWAM"
for dtype in bf16 fp32; do
  step "smoke $dtype"
  AGENTICWAM_DEPLOY_DTYPE=$dtype AGENTICWAM_TEXT_OFFLOAD=1 $PY scripts/eval_robotwin_release.py wam \
      --tasks click_alarmclock --plan smoke --protocols clean --gpus 0 --name "smoke-$dtype" --render \
      && echo "SMOKE $dtype: exit 0" || echo "SMOKE $dtype: FAILED"
  cp -r evaluate_results "$OUT/evaluate_results-$dtype" 2>/dev/null || true
done
kill %1
du -sh "$OUT"; step "done"
