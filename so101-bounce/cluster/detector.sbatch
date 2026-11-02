#!/bin/bash
# Ball detector: render synthetic images in parallel into job scratch (never home: 15 GB quota), then train.
#   NAME=v4 [IMAGES=60000] [WIDTH=16] sbatch --export=ALL $HOME/roboticsexp/so101-bounce/cluster/detector.sbatch
# Results: ~/so101-bounce-results/detector-$NAME (best.pt, ballnet.onnx, history.json, sample grid)
#SBATCH -J bounce-detector
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/so101-bounce}/cluster/common.sh"
bounce_env
"$UV" pip install -q onnx onnxscript
cd "$PROJ"
OUT=$RESULTS/detector-${NAME:?set NAME}
mkdir -p "$OUT"
DATA=$WORK/detdata
N=${IMAGES:-60000}
t0=$SECONDS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# One render process at a time: the H200's graphics engine thrashes between EGL contexts (two processes render
# >40x slower than one; v3's 12 stalled and aborted). Each process renders CHUNK images (shards of 800), then exits,
# which bounds the number of GL contexts it creates.
CHUNK=${CHUNK:-4000}
SHARD=${SHARD:-800}
for k in $(seq 0 $(( (N + CHUNK - 1) / CHUNK - 1 ))); do
  MUJOCO_GL=egl $PY scripts/make_detector_data.py --out "$DATA/p0" --images $CHUNK --shard $SHARD \
    --seed $((1000 + k)) --first_shard $((k * CHUNK / SHARD)) >> "$WORK/gen0.log" 2>&1 || echo "chunk $k failed" >> "$WORK/gen0.log"
  echo "chunk $k done at $((SECONDS - t0)) s"
done
grep -h "GL renderer" "$WORK"/gen*.log | sort | uniq -c || true
mkdir -p "$OUT/gen_logs" "$DATA/all"
cp "$WORK"/gen*.log "$OUT/gen_logs/"
shopt -s nullglob
k=0
for f in "$DATA"/p*/shard*.npz; do ln -s "$f" "$DATA/all/shard$(printf %05d $k).npz"; k=$((k + 1)); done
echo "rendered $k shards in $((SECONDS - t0)) s"
if [ "$k" -lt 8 ]; then tail -5 "$WORK"/gen*.log; echo "DETECTOR_EXIT too few shards"; exit 1; fi
set +e
$PY scripts/train_detector.py --data "$DATA/all" --out "$OUT" --epochs ${EPOCHS:-25} --batch 256 --val_shards 3 \
  --width ${WIDTH:-16} 2>&1 | tee "$OUT/train.log" | grep -v "^\[torch.onnx\]"
echo "DETECTOR_EXIT ${PIPESTATUS[0]}"
