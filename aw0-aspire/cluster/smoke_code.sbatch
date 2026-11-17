#!/bin/bash
# Code-as-policy arm smoke (privileged tier, AW-0's codegen/sim replay path). Positive: the shipped ASPIRE-style
# frozen program for click_alarmclock on seed 101 (clean) and 151 (randomized). Negative controls: the setup-only
# program from SETUP.md and a do-nothing program must score 0. Any "check_success() raised" fails the smoke.
#SBATCH -J aa-smoke-code
#SBATCH -p general
#SBATCH -t 3:00:00
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/smoke-code-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
build_sim

OUT=$RESULTS/smoke-code-$SLURM_JOB_ID; mkdir -p "$OUT/programs"
cat > "$OUT/programs/setup_only.py" <<'PY'
print('observation:', type(get_observation()).__name__)
print('home motion:', goto_home_joint_position('left'))
PY
echo "print('noop')" > "$OUT/programs/noop.py"
FROZEN=$AWAM/configs/robotwin/tasks_indep

cd "$SIM"; set +u; source env.sh; set -u   # codegen/sim's env.sh reads optional unset vars
replay() {   # name task seed config program
  local t=$SECONDS
  $PY iterate/scripts/replay_trial.py --code "$5" --task "$2" --seed "$3" \
      --config "iterate/configs/$4.yaml" --output-dir "$OUT/replay/$1" > "$OUT/$1.log" 2>&1 || echo "replay $1 exited $?"
  local trial; trial=$(find "$OUT/replay/oracle/$1" -maxdepth 1 -type d -name 'trial_*' 2>/dev/null | head -1 || true)
  echo "CODE $1: $(basename "${trial:-none}") | $((SECONDS - t)) s | check_success raised: $(grep -c 'check_success() raised' "$OUT/$1.log")"
}
replay setup_only   stack_blocks_two 101 priv_clean      "$OUT/programs/setup_only.py"
replay noop         click_alarmclock 101 priv_clean      "$OUT/programs/noop.py"
replay frozen_clean click_alarmclock 101 priv_clean      "$FROZEN/click_alarmclock/frozen_program.py"
replay frozen_rand  click_alarmclock 151 priv_randomized "$FROZEN/click_alarmclock/frozen_program.py"
find "$OUT/replay" -name '*.mp4' -size +20M -delete   # keep home small
du -sh "$OUT"; step "done"
