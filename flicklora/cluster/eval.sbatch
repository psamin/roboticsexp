#!/bin/bash
# Phase E: evaluate one policy on the 30 held-out test configs (same seeds for every rank) on a DGX Spark.
# Policy server (LeRobot) and Isaac Sim run side by side on the node, talking over localhost.
#   TAG=lora-r32-s1000 TRAIN_DIR=<train dir> sbatch eval.sbatch     (uses TRAIN_DIR/pretrained_model + /dataset)
#   TAG=base BASE=1 META=<train dir>/dataset sbatch eval.sbatch     (unadapted lerobot/smolvla_base)
#   TAG=expert EXPERT=1 SIGN=-1 sbatch eval.sbatch                  (scripted expert = upper reference)
#SBATCH -J flick-eval
#SBATCH -p general
#SBATCH --gres=gpu:gb10:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
TAG=${TAG:?}; N=${N:-30}; SEED0=${SEED0:-90000}; SIGN=${SIGN:--1}; HZ=${HZ:-1200}; ROBOT=${ROBOT:-finger}
OUT=$RESULTS/eval; [ "$ROBOT" = finger ] || OUT=$RESULTS/eval-$ROBOT
mkdir -p "$OUT"
fetch_ref
if [ -z "${EXPERT:-}" ]; then
  if [ -n "${TRAIN_DIR:-}" ]; then
    [ -d "$TRAIN_DIR" ] || TRAIN_DIR=$(ls -d ${TRAIN_DIR}* 2>/dev/null | head -1)   # allow a prefix (job id unknown)
    CKPT=${CKPT:-$TRAIN_DIR/pretrained_model}; META=${META:-$TRAIN_DIR/dataset}
  fi
  lerobot_env
  if [ -n "${BASE:-}" ]; then SARGS=(--base); else SARGS=(--ckpt "$CKPT"); fi
  "$LR_PY" "$PROJ/train/policy_server.py" "${SARGS[@]}" --dataset_root "$META" --port 5555 > "$WORK/server.log" 2>&1 &
  SRV=$!
  for i in $(seq 900); do grep -q SERVER_READY "$WORK/server.log" && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  grep -E "policy ready|SERVER_READY|Error|Traceback" "$WORK/server.log" | head
  grep -q SERVER_READY "$WORK/server.log" || { tail -40 "$WORK/server.log"; exit 1; }
  EARGS=(--port 5555)
else
  EARGS=(--expert --flick_sign "$SIGN")
fi
isaac_env
isaac_run "$PROJ/sim/eval_policy.py" --robot "$ROBOT" --ref "$WORK/ref" --out "$OUT" --tag "$TAG" --n "$N" --seed0 "$SEED0" \
   --physics_hz "$HZ" "${EARGS[@]}" 2>&1 | grep -E "^\[$TAG\]|SUMMARY|Traceback|Error|File \"" | grep -v deprecated
[ -n "${SRV:-}" ] && kill $SRV 2>/dev/null
echo EVAL_DONE
