#!/bin/bash
# Arm A2: replay each task's shipped ASPIRE-style frozen program (privileged tier) on exactly the scenes the
# evaluator realizes (seeds.tsv from seed_scan.sbatch). Writes results.tsv
# (task, protocol, shard, episode, seed, reward, seconds, check_success exceptions) and frame 0 of each trial.
# usage: SCAN=<dir with seeds.tsv> TASKS="a b" [PROG_DIR=<dir>/<task>.py programs] [OUT_PREFIX=fixloop-eval] \
#        sbatch --export=ALL code_eval.sbatch
#SBATCH -J aa-code-eval
#SBATCH -p general
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/code-eval-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
build_sim
OUT=$RESULTS/${OUT_PREFIX:-code-eval}-$SLURM_JOB_ID; mkdir -p "$OUT/logs"
cd "$SIM"; set +u; source env.sh; set -u
for task in $TASKS; do
  prog=$AWAM/configs/robotwin/tasks_indep/$task/frozen_program.py
  [ -n "${PROG_DIR:-}" ] && prog=$PROG_DIR/$task.py
  sha256sum "$prog" >> "$OUT/programs.sha256"
  awk -v t="$task" '$1 == t' "$SCAN/seeds.tsv" | while IFS=$'\t' read -r _ proto k ep seed; do
    name=${task}_${proto}_s${k}_ep${ep}; t=$SECONDS
    timeout 900 $PY iterate/scripts/replay_trial.py --code "$prog" --task "$task" --seed "$seed" \
        --config "iterate/configs/priv_$proto.yaml" --output-dir "$WORK/replay/$name" > "$OUT/logs/$name.log" 2>&1 \
        || echo "replay exited $?" >> "$OUT/logs/$name.log"
    trial=$(find "$WORK/replay" -path "*oracle/$name/trial_*" -maxdepth 3 -type d 2>/dev/null | head -1 || true)
    reward=$(basename "${trial:-none}" | sed -n 's/.*_reward_\([0-9.]*\)_.*/\1/p')
    vid=$(find "${trial:-/nonexistent}" -name 'video_turn_00.mp4' 2>/dev/null | head -1 || true)
    if [ -n "$vid" ]; then
      mkdir -p "$OUT/frames/$task/$proto/s$k"
      ffmpeg -v error -i "$vid" -frames:v 1 "$OUT/frames/$task/$proto/s$k/ep$(printf %03d "$ep").png" -y || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$proto" "$k" "$ep" "$seed" "${reward:-NA}" \
        "$((SECONDS - t))" "$(grep -c 'check_success() raised' "$OUT/logs/$name.log")" >> "$OUT/results.tsv"
    rm -rf "$WORK/replay/$name"
  done
  echo "CODE_EVAL $task: $(awk -v t="$task" '$1 == t && $6 == "1.000"' "$OUT/results.tsv" | wc -l)/$(awk -v t="$task" '$1 == t' "$OUT/results.tsv" | wc -l) success"
done
step "done"
