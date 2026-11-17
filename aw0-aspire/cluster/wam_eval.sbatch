#!/bin/bash
# Arms A0/A1: released AW-0 wam.pt through RoboTwin's eval_policy.py (AW-0 launcher), clean and randomized,
# EPS episodes in SHARDS shards (shard k = RoboTwin seed k, the stream seed_scan.sbatch reproduces).
# Keeps the per-shard eval logs (per-episode seed and outcome) in home.
# usage: TASKS="a b" DTYPE=bf16 EPS=50 SHARDS=5 [AW0_TEXT_DEVICE=cpu] [RESUME=<job id>] sbatch --export=ALL wam_eval.sbatch
# RESUME reuses that job's finished shards (from its logs in home) and runs only the rest.
#SBATCH -J aa-wam-eval
#SBATCH -p general
#SBATCH -t 12:00:00
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
if [ -n "${RESUME:-}" ]; then   # finished shards carry run_meta.json with the episode count; the benchmark skips them
  name=wam-$DTYPE-$RESUME
  mkdir -p evaluate_results && cp -r "$RESULTS/wam-eval-$RESUME/logs/evaluate_results/robotwin" evaluate_results/
  echo "resuming $name: $(find evaluate_results -name run_meta.json | wc -l) shard records restored"
fi
save_logs() {   # results to home as they come (a walltime kill loses at most the shard in progress)
  [ -d evaluate_results ] || return 0
  mkdir -p "$OUT/logs"
  find evaluate_results \( -name '*.log' -o -name 'run_meta.json' -o -name 'report.*' -o -name 'summary.json' \
      -o -name 'manifest.json' \) -exec cp --parents {} "$OUT/logs/" \; || true
}
( while sleep 600; do save_logs; done ) &
# Moving the 7B text encoder on and off the GPU per instruction fragments the cache (fp32 OOM after 5 episodes on 44 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AGENTICWAM_DEPLOY_DTYPE=$DTYPE AGENTICWAM_TEXT_OFFLOAD=1 $PY scripts/eval_robotwin_release.py wam \
    --tasks "$(echo $TASKS | tr ' ' ',')" --episodes "$EPS" --shards "$SHARDS" --gpus 0 --name "$name" \
    || echo "WAM_EVAL exited $?"
$PY robotwin/eval/report.py "$name" || true
save_logs
cat evaluate_results/robotwin_benchmark/$name/report.md 2>/dev/null
kill %1 %2; step "done"
