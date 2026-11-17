#!/bin/bash
# Realized evaluation scenes per (task, protocol, shard): RoboTwin's own eval_policy.py (through AW-0's
# robotwin_entry.py, EPS episodes per shard) with a policy that stops at once. Writes seeds.tsv
# (task, protocol, shard, episode, seed) and the first head frame of every episode.
# usage: TASKS="a b" SHARDS="0 1 2 3 4" EPS=10 sbatch --export=ALL seed_scan.sbatch
#SBATCH -J aa-seedscan
#SBATCH -p general
#SBATCH -t 8:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/seedscan-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
build_sim
cp -r "$HOME_DIR/aw0-aspire/seedscan_policy" "$ROBOTWIN/policy/aa_seedscan"
echo "policy_name: aa_seedscan" > "$ROBOTWIN/policy/aa_seedscan/deploy_policy.yml"
OUT=$RESULTS/seedscan-$SLURM_JOB_ID; mkdir -p "$OUT"
cd "$ROBOTWIN"
for task in $TASKS; do for proto in clean randomized; do for k in $SHARDS; do
  step "$task $proto shard $k"
  export SEEDSCAN_FRAMES=$OUT/frames/$task/$proto/s$k; mkdir -p "$SEEDSCAN_FRAMES"
  log=$OUT/logs/${task}_${proto}_s$k.log; mkdir -p "$OUT/logs"
  $PY "$AWAM/robotwin/eval/robotwin_entry.py" --config policy/aa_seedscan/deploy_policy.yml --overrides \
      --task_name "$task" --task_config "demo_$proto" --ckpt_setting scan --seed "$k" \
      --policy_name aa_seedscan --instruction_type seen --eval_num_episodes "$EPS" > "$log" 2>&1 \
      || echo "scan $task $proto $k exited $?"
  sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -o "current seed: [0-9]*" | awk -v t="$task" -v p="$proto" -v k="$k" \
      '{print t "\t" p "\t" k "\t" NR-1 "\t" $3}' >> "$OUT/seeds.tsv"
  echo "SCAN $task $proto s$k: $(grep -c "current seed" "$log") episodes, $(grep -c "error occurs" "$log") expert errors"
done; done; done
rm -rf "$ROBOTWIN/eval_result"; step "done"
