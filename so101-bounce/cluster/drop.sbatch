#!/bin/bash
# Drop-start test.   CKPTS="<v5-base/checkpoints/best.pt or path under the project> ..." sbatch --export=ALL .../cluster/drop.sbatch
#SBATCH -J bounce-drop
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
mkdir -p "$RESULTS/drop_tests"
for c in ${CKPTS:?set CKPTS}; do
  [ -f "$c" ] || c="$RESULTS/$c"
  name=$(echo "${c#$RESULTS/}" | sed 's#/checkpoints/#-#; s#/#-#g; s#\.pt$##')
  $PY scripts/drop_test.py --checkpoint "$c" --envs 1024 --device cuda 2>&1 | tee "$RESULTS/drop_tests/$name.txt"
done
echo DROP_DONE
