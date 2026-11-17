#!/bin/bash
# Replays programs for a fix-loop agent that runs elsewhere. A request is a directory in $Q/inbox holding
# program.py and req.env (TASK, SEEDS). Seeds < 150 use priv_clean, others priv_randomized (the fix-loop bands).
# Results land in $Q/out/<request>/ (trial logs, summaries, 4 frames per seed, result.tsv, DONE).
# Exits after IDLE_MIN minutes without requests or when $Q/STOP exists.
#SBATCH -J aa-trials
#SBATCH -p general
#SBATCH -t 8:00:00
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/trials-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
Q=$HOME_DIR/aw0-aspire-queue; IDLE_MIN=${IDLE_MIN:-45}
mkdir -p "$Q/inbox" "$Q/out" "$Q/done"; rm -f "$Q/STOP"
build_sim
cd "$SIM"; set +u; source env.sh; set -u
echo "$SLURM_JOB_ID" > "$Q/SERVER_READY"
last=$SECONDS
while [ ! -e "$Q/STOP" ] && [ $(( (SECONDS - last) / 60 )) -lt "$IDLE_MIN" ]; do
  req=$(ls -1 "$Q/inbox" 2>/dev/null | head -1)
  [ -z "$req" ] && { sleep 10; continue; }
  [ -f "$Q/inbox/$req/req.env" ] && [ -f "$Q/inbox/$req/program.py" ] || { sleep 5; continue; }
  TASK=; SEEDS=; source "$Q/inbox/$req/req.env"
  out=$Q/out/$req; mkdir -p "$out"; cp "$Q/inbox/$req/program.py" "$out/"
  step "request $req: $TASK seeds $SEEDS"
  for s in $SEEDS; do
    cfg=priv_clean; [ "$s" -ge 150 ] && cfg=priv_randomized
    t=$SECONDS
    timeout 900 $PY iterate/scripts/replay_trial.py --code "$out/program.py" --task "$TASK" --seed "$s" \
        --config "iterate/configs/$cfg.yaml" --output-dir "$WORK/replay/$req/s$s" > "$out/seed$s.log" 2>&1 \
        || echo "replay exited $?" >> "$out/seed$s.log"
    trial=$(find "$WORK/replay/$req" -type d -name "trial_${s}_*" | head -1)
    mkdir -p "$out/seed$s"
    if [ -n "$trial" ]; then
      find "$trial" -maxdepth 1 -type f ! -name '*.mp4' ! -name '*.pkl' -size -5M -exec cp {} "$out/seed$s/" \;
      vid=$(find "$trial" -name '*.mp4' | head -1)
      [ -n "$vid" ] && for f in 0.1 0.4 0.7 0.98; do
        d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid")
        ffmpeg -v error -ss "$(awk -v d="$d" -v f="$f" 'BEGIN{print d*f}')" -i "$vid" -frames:v 1 -vf scale=640:-1 -q:v 4 "$out/seed$s/frame_$f.jpg" -y || true
      done
    fi
    printf '%s\t%s\t%s\t%ss\t%s\n' "$s" "$cfg" "$(basename "${trial:-none}")" "$((SECONDS - t))" \
        "$(grep -c 'check_success() raised' "$out/seed$s.log")" >> "$out/result.tsv"
  done
  touch "$out/DONE"; mv "$Q/inbox/$req" "$Q/done/"; rm -rf "$WORK/replay/$req"; last=$SECONDS
done
rm -f "$Q/SERVER_READY"; step "server exit"
