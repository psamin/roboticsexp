#!/bin/bash
# Arms A0/A1: released AW-0 wam.pt through RoboTwin's eval_policy.py (AW-0 launcher), clean and randomized,
# EPS episodes in SHARDS shards (shard k = RoboTwin seed k, the stream seed_scan.sbatch reproduces).
# Keeps the per-shard eval logs (per-episode seed and outcome) in home.
# usage: TASKS="a b" DTYPE=bf16 EPS=50 SHARDS=5 [AW0_TEXT_DEVICE=cpu] sbatch --export=ALL wam_eval.sbatch
#SBATCH -J aa-wam-eval
#SBATCH -p general
#SBATCH -t 6:00:00
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/wam-eval-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
echo "TASKS=$TASKS DTYPE=$DTYPE EPS=$EPS SHARDS=$SHARDS AW0_TEXT_DEVICE=${AW0_TEXT_DEVICE:-unset}"; nvidia-smi -L
build_all
OUT=$RESULTS/wam-eval-$SLURM_JOB_ID; mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 30 > "$OUT/gpu_mem.csv" &
cd "$AWAM"
name=wam-$DTYPE-$SLURM_JOB_ID
AGENTICWAM_DEPLOY_DTYPE=$DTYPE AGENTICWAM_TEXT_OFFLOAD=1 $PY scripts/eval_robotwin_release.py wam \
    --tasks "$(echo $TASKS | tr ' ' ',')" --episodes "$EPS" --shards "$SHARDS" --gpus 0 --name "$name" \
    || echo "WAM_EVAL exited $?"
$PY robotwin/eval/report.py "$name" || true
mkdir -p "$OUT/logs"
find evaluate_results -name '*.log' -exec cp --parents {} "$OUT/logs/" \;
find evaluate_results \( -name 'run_meta.json' -o -name 'report.*' -o -name 'summary.json' -o -name 'manifest.json' \) \
    -exec cp --parents {} "$OUT/logs/" \;
cat evaluate_results/robotwin_benchmark/$name/report.md 2>/dev/null
kill %1; step "done"
