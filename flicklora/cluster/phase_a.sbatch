#!/bin/bash
# Phase A (+ first Phase B check) on one RTX GPU: install Isaac Sim 4.5, measure CAD, record the original
# static_bolt.usda, record our restyled scene driven and contact-driven, build the side-by-side video.
#   sbatch flicklora/cluster/phase_a.sbatch
#SBATCH -J flick-phaseA
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
OUT=$RESULTS/phaseA-$SLURM_JOB_ID
mkdir -p "$OUT"
fetch_ref
t0=$(date +%s); isaac_env; echo "isaac install took $(( $(date +%s) - t0 ))s"

set +e   # run every stage even if one fails; each writes its own json
echo "================= boot test"
HOME=$WORK/kithome XDG_CACHE_HOME=$WORK/kithome/.cache timeout 900 $ISAAC_PY $PROJ/sim/boot_test.py 2>&1 | grep -E "BOOT_OK|RENDER_OK|Segmentation|Fatal|GLIBC|Driver Version" | head
for mode in inspect original ours_driven ours_passive; do
  echo "================= $mode"
  timeout 1500 bash -c "HOME=$WORK/kithome XDG_CACHE_HOME=$WORK/kithome/.cache $ISAAC_PY $PROJ/sim/phase_a.py --mode $mode --ref $WORK/ref --out $OUT" 2>&1 \
    | grep -vE "^\s*$|Warning: Possibly invalid|omni.kit.registry" | tail -400
  echo "exit[$mode]=${PIPESTATUS[0]}"
done
$ISAAC_PY "$PROJ/sim/compare_video.py" --out "$OUT" 2>&1 | tail -20
ls -la "$OUT"
echo PHASE_A_DONE
