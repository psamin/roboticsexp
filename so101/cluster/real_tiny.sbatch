#!/bin/bash
# TinyPolicy on REAL SO-101 demos: float BC -> QAT -> int8 export -> golden vectors -> HLS C model.
# Like robotfpga's cluster/final.sbatch but without the MuJoCo evals (meaningless for a real-data
# policy; judge it on the arm instead). Data: shards from so101/lerobot_to_armlab.py.
#   NAME=real-bottle-v1 DATA=/nethome-instruction/pi34/robotfpga-work/datasets/real_bottle_v1 sbatch real_tiny.sbatch
#SBATCH -J tiny-real
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
set -euo pipefail

NAME=${NAME:?}
DATA=${DATA:?}
STEPS=${STEPS:-10000}         # float BC steps (13 episodes / 6.4k frames: ~400 epochs at bs 256)
QAT_STEPS=${QAT_STEPS:-5000}
VAL=${VAL:-2}                 # held-out episodes (placed first by the converter)
REV=${REV:-main}
HOME_DIR=/nethome-instruction/$USER   # /nethome is a broken symlink on compute nodes
OUT=$HOME_DIR/robotfpga-work/runs/$NAME
WORK=/dev/shm/$USER-$SLURM_JOB_ID
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK" "$OUT"
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv \
       XDG_CACHE_HOME=$WORK/cache TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton
UV=$HOME_DIR/.local/bin/uv

git clone -q https://github.com/psamin/robotfpga.git "$WORK/robotfpga"
cd "$WORK/robotfpga" && git checkout -q "$REV"
echo "job $SLURM_JOB_ID on $(hostname) robotfpga $(git rev-parse --short HEAD) data $DATA -> $OUT"
"$UV" sync -q
cp "$DATA/norm.json" "$OUT/norm.json"
run() { echo "+ $*"; "$UV" run python -m "$@"; }

run armlab.policy.train --data "$DATA" --out "$OUT/float" --steps "$STEPS" --val-episodes "$VAL" --ckpt-every "$STEPS"
FLOAT=$(ls -t "$OUT"/float/*.pt | head -1)
run armlab.policy.qat_train --data "$DATA" --float-ckpt "$FLOAT" --out "$OUT/qat" --steps "$QAT_STEPS" --val-episodes "$VAL"

# Golden vectors from real frames, then the HLS C model must match them bit for bit.
"$UV" run python scripts/pack_frames.py --data "$DATA" --out "$OUT/packets.npy" --n 80
rm -rf "$OUT/handoff" && cp -r "$OUT/qat/export" "$OUT/handoff"
"$UV" run python ref/vectors.py --model "$OUT/handoff" --packets "$OUT/packets.npy"
make -s -C hls csim ART="$OUT/handoff" | tee "$OUT/hls_csim.txt"
echo "DONE $OUT"
