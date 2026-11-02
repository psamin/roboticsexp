#!/bin/bash
# Feasibility oracle (scripts/oracle_feasibility.py): can ANY open-loop command sequence recover a tapped or tossed
# ball on servo set 12v_r10? Run with each deploy candidate's control interface, plus its own policy for comparison.
#   sbatch .../cluster/oracle.sbatch
# Results: ~/so101-bounce-results/oracle/
#SBATCH -J bounce-oracle
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:60:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
cd "$PROJ"
OUT=$RESULTS/oracle
mkdir -p "$OUT"
F='warn\|cuda capab\|Minimum and\|(8.0)'
for ck in ${CKS:-v16-p32-lo v18-p32-t12}; do
  $PY scripts/oracle_feasibility.py --checkpoint "$RESULTS/$ck/checkpoints/best.pt" --servo 12v_r10 --device cuda --cases ${CASES:-main} \
    --pop ${POP:-4096} --iters ${ITERS:-20} --horizon ${HORIZON:-1.5} 2>&1 | grep -v -i "$F" | tee "$OUT/$ck${CASES:+-$CASES}.txt"
done
echo ORACLE_DONE
