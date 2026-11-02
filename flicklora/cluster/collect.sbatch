#!/bin/bash
# Phase B check + Phase C demos on a DGX Spark (Isaac Sim 5.1): 4 scripted-expert test episodes with videos (also
# auto-detects which sweep direction unscrews), then 100 successful train demos and 10 val demos.
#   sbatch flicklora/cluster/collect.sbatch            (N_TRAIN=100 N_VAL=10 by default)
#SBATCH -J flick-collect
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
N_TRAIN=${N_TRAIN:-100}; N_VAL=${N_VAL:-10}; HZ=${HZ:-1200}; ONLY_TEST=${ONLY_TEST:-}; ROBOT=${ROBOT:-finger}
RAW=$RESULTS/raw-$ROBOT-$SLURM_JOB_ID
PB=$RESULTS/phaseB-$ROBOT-$SLURM_JOB_ID
mkdir -p "$RAW" "$PB"
fetch_ref
isaac_env
cp "$PROJ"/sim/*.py "$PB/"     # exact code used for this data
if [ -z "${SIGN:-}" ]; then   # set SIGN=-1 to skip the test once the direction is known
  echo "================= Phase B: expert test episodes"
  isaac_run "$PROJ/sim/run_expert.py" --mode test --robot "$ROBOT" --ref "$WORK/ref" --out "$PB" --physics_hz "$HZ" 2>&1 \
    | grep -E "run_expert|franka_env|Traceback|Error|error:|File \"" | grep -v deprecated
  SIGN=$(isaac_run -c "import json,glob;print(json.load(open(glob.glob('$PB/phaseB_*hz.json')[0]))['flick_sign'])" 2>/dev/null | tail -1)
fi
echo "flick sign: ${SIGN:-unknown}"
[ -n "$SIGN" ] || { echo "Phase B failed; not collecting"; exit 1; }
[ -n "$ONLY_TEST" ] && { echo COLLECT_DONE; exit 0; }
for split in train val; do
  n=$N_TRAIN; s0=0; [ $split = val ] && { n=$N_VAL; s0=50000; }
  echo "================= Phase C: $split ($n successes)"
  isaac_run "$PROJ/sim/run_expert.py" --mode collect --robot "$ROBOT" --split $split --n $n --seed0 $s0 --flick_sign "$SIGN" \
     --ref "$WORK/ref" --out "$RAW" --physics_hz "$HZ" --hero "" 2>&1 | grep -E "run_expert|Traceback|Error|File \"" | grep -v deprecated
done
du -sh "$RAW"/*
echo "RAW=$RAW"
echo COLLECT_DONE
