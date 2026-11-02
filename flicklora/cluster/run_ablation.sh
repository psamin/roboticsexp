#!/bin/bash
# Phase E in one command (run on a Futurama login node; it only calls sbatch):
#   bash flicklora/cluster/run_ablation.sh /nethome-instruction/pi34/flicklora-results/raw-<collect jobid>
# Submits, per seed: LoRA r=8/32/128 (alpha=r) + full fine-tune (seed 1000 only), each on one L40/L40S, identical
# data/steps/optimizer/seed; then one Spark eval per policy (starts when its training finishes), plus the unadapted
# base model and the scripted expert on the same 30 test configs. Writes a manifest of job ids.
set -euo pipefail
RAW=${1:?raw dir from collect.sbatch (contains train/)}
STEPS=${STEPS:-10000}
SEEDS=${SEEDS:-"1000 2000"}
RANKS=${RANKS:-"8 32 128"}
FULL=${FULL:-1}
GPU=${GPU:-l40}
ROBOT=${ROBOT:-finger}       # finger | franka (evals run the matching sim)
# This script runs on the login node, where home is $HOME (/nethome/<user>); jobs on compute nodes must use
# /nethome-instruction/<user> (the /nethome symlink is broken there). So: submit from $HOME paths, pass runtime paths.
C=$HOME/roboticsexp/flicklora/cluster
R=/nethome-instruction/$USER/flicklora-results
MAN=$HOME/flicklora-results/ablation-$(date +%m%d-%H%M).txt
cd "$HOME/roboticsexp"
echo "raw=$RAW steps=$STEPS seeds=$SEEDS ranks=$RANKS full=$FULL" | tee "$MAN"
first=""
sub_train() {  # mode rank seed
  local tag=$ROBOT-$1-r$2-s$3; [ "$1" = full ] && tag=$ROBOT-full-s$3; [ "$ROBOT" = finger ] && tag=${tag#finger-}
  local jid=$(MODE=$1 RANK=$2 SEED=$3 STEPS=$STEPS RAW_DIR=$RAW/train TAG=$tag \
     sbatch --parsable -J "fl-$tag" --gres=gpu:$GPU:1 -t 4:00:00 ${DEP:+--dependency=afterok:$DEP} --export=ALL $C/train_lora.sbatch)
  # Spark nodes cannot rebuild a login environment, and --export=ALL,VAR=... makes Slurm try ("user env retrieval
  # failed"). So evals get their variables from the environment with a plain --export=ALL.
  local eid=$(ROBOT=$ROBOT TAG=$tag TRAIN_DIR=$R/train-$tag-$jid sbatch --parsable -J "ev-$tag" --dependency=afterok:$jid \
     --export=ALL $C/eval.sbatch)
  echo "train $tag $jid  eval $eid" | tee -a "$MAN"
  [ -z "$first" ] && first="$jid:$tag"
  return 0
}
for s in $SEEDS; do
  for r in $RANKS; do sub_train lora "$r" "$s"; done
done
[ "$FULL" = 1 ] && sub_train full 0 "${SEEDS%% *}"
[ -n "${NO_REF:-}" ] && { echo "manifest: $MAN (no base/expert evals: NO_REF set)"; exit 0; }
fj=${first%%:*}; ft=${first#*:}
eb=$(ROBOT=$ROBOT TAG=base BASE=1 META=$R/train-$ft-$fj/dataset sbatch --parsable -J ev-base --dependency=afterok:$fj --export=ALL $C/eval.sbatch)
ee=$(ROBOT=$ROBOT TAG=expert EXPERT=1 SIGN=${SIGN:--1} sbatch --parsable -J ev-expert ${DEP:+--dependency=afterok:$DEP} --export=ALL $C/eval.sbatch)
echo "eval base $eb  eval expert $ee" | tee -a "$MAN"
echo "manifest: $MAN"
