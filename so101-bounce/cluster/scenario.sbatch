#!/bin/bash
# Scenario test (research/scenarios_and_failures.md).   CKPTS="policies/v12-base.pt v13-base/checkpoints/best.pt ..."
#   sbatch --export=ALL .../cluster/scenario.sbatch      Results: ~/so101-bounce-results/scenario_tests/<name>.txt
#SBATCH -J bounce-scen
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:60:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
mkdir -p "$RESULTS/scenario_tests"
for c in ${CKPTS:?set CKPTS}; do
  [ -f "$c" ] || c="$RESULTS/$c"
  name=$(echo "${c#$RESULTS/}" | sed 's#/checkpoints/#-#; s#/#-#g; s#\.pt$##')
  $PY scripts/scenario_test.py --checkpoint "$c" --per ${PER:-512} --device cuda 2>&1 | tee "$RESULTS/scenario_tests/$name.txt"
done
echo SCEN_DONE
