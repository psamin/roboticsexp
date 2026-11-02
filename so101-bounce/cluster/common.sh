# Sourced by every so101-bounce job on the GT Futurama cluster. The simulator is plain PyTorch (MuJoCo is only used
# to read the robot model), so it runs on any CUDA node: the H200 (x86) or the DGX Sparks / GH200 (aarch64).
# Everything is installed fresh into job-scoped /dev/shm scratch; results go straight to home.
set -euo pipefail
HOME_DIR=/nethome-instruction/$USER          # /nethome/<user> is a self-symlink on compute nodes
PROJ=${PROJ:-$HOME_DIR/roboticsexp/so101-bounce}
[ -d "$PROJ" ] || PROJ="${PROJ/#\/nethome\//\/nethome-instruction\/}"
RESULTS=${RESULTS:-$HOME_DIR/so101-bounce-results}
WORK=/dev/shm/$USER-bounce-${SLURM_JOB_ID:-local}
mkdir -p "$WORK/bin" "$RESULTS"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
trap 'echo "signal: cleaning up"; exit 1' TERM INT USR1
export UV_CACHE_DIR=$WORK/uv-cache UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$WORK/uv-python PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton

echo "== $(hostname) job ${SLURM_JOB_ID:-} $(date)"
uname -m; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

if [ "$(uname -m)" = "aarch64" ]; then
  # the home uv is an x86 binary
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
  UV=$WORK/bin/uv
  TORCH_INDEX=https://download.pytorch.org/whl/cu130
else
  UV=$HOME_DIR/.local/bin/uv
  TORCH_INDEX=https://download.pytorch.org/whl/cu128
fi

bounce_env() {
  local t0=$SECONDS
  "$UV" venv -q --python 3.11 "$WORK/venv"
  export VIRTUAL_ENV=$WORK/venv
  "$UV" pip install -q "torch==2.9.0" --index-url "$TORCH_INDEX"
  "$UV" pip install -q numpy pyyaml "mujoco==3.13.0" pytest imageio imageio-ffmpeg pillow
  PY=$WORK/venv/bin/python
  echo "env ready in $((SECONDS - t0)) s: $($PY -c 'import torch,mujoco;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"mujoco",mujoco.__version__)')"
}
