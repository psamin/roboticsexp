#!/bin/bash
# Does running the Cosmos-Reason1 text encoder on CPU (AW0_TEXT_DEVICE=cpu, needed to fit AW-0 on a 24 GB L4)
# change the instruction embeddings? Compares against the released GPU bf16 path on the comparison tasks.
#SBATCH -J aa-textcheck
#SBATCH -p general
#SBATCH -t 1:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -o /nethome-instruction/pi34/aw0-aspire-results/textcheck-%j.txt
source /nethome-instruction/$USER/aw0-aspire/cluster/common.sh
source /nethome-instruction/$USER/aw0-aspire/cluster/env.sh
sources; py_env; weights
PYTHONPATH=$AWAM/src $PY $HOME_DIR/aw0-aspire/text_check.py "$AWAM" "$ROBOTWIN" "$COSMOS" \
    $(cat $HOME_DIR/aw0-aspire/tasks.txt)
step "done"
