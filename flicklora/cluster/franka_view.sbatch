#!/bin/bash
#SBATCH -J franka-view
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 0:40:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
set +e +o pipefail
isaac_env
OUT=$RESULTS/franka-view-$SLURM_JOB_ID
isaac_run "$PROJ/sim/franka_view.py" "$OUT" 2>&1 | grep -vE "deprecated|^\s*$|ext: |carb.audio|requests|asset.browser" | grep -iE "variant|meshes|geometry prims|wrote|NO IMAGE|instanceable|/World/Franka|Could not|Error|error|Warning.*(asset|open|resolve|Props|Material)" | cut -c1-300 | head -60
ls -la "$OUT"
echo VIEW_DONE
