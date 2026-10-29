# Sourced by every ppo-goalkeeper job on the GT Futurama cluster.
# Isaac Sim only runs on the DGX Spark nodes there (aarch64, GB10, Ubuntu); the x86 RTX nodes have glibc 2.34,
# which the Isaac Sim 5.x wheels do not support. Everything is installed fresh into job-scoped /dev/shm scratch.
set -euo pipefail
HOME_DIR=/nethome-instruction/$USER          # /nethome/<user> is a self-symlink on compute nodes
PROJ=${PROJ:-$HOME_DIR/roboticsexp/ppo-goalkeeper}
[ -d "$PROJ" ] || PROJ="${PROJ/#\/nethome\//\/nethome-instruction\/}"
WORK=/dev/shm/$USER-gk-${SLURM_JOB_ID:-local}
ISAAC_VERSION=5.1.0
ISAACLAB_TAG=v2.3.2
TORCH_SPEC="torch==2.9.0 torchvision==0.24.0"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
mkdir -p "$WORK/bin"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
trap 'echo "signal: cleaning up"; exit 1' TERM INT USR1
export UV_CACHE_DIR=$WORK/uv-cache UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$WORK/uv-python PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton HF_HOME=$WORK/hf

echo "== $(hostname) job ${SLURM_JOB_ID:-} $(date)"
uname -m; (. /etc/os-release && echo "$PRETTY_NAME"); ldd --version 2>&1 | sed -n 1p
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# The home uv is an x86 binary; fetch an arm64 one into scratch.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
UV=$WORK/bin/uv

isaac_env() {
  local t0=$SECONDS
  "$UV" venv -q --python 3.11 "$WORK/venv"
  export VIRTUAL_ENV=$WORK/venv
  "$UV" pip install -q "isaacsim[all,extscache]==$ISAAC_VERSION" \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  "$UV" pip install -q -U $TORCH_SPEC --index-url "$TORCH_INDEX"
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$WORK/IsaacLab"
  # flatdict 4.0.1 (an isaaclab dep) imports pkg_resources in its setup.py; newer setuptools dropped it
  echo "setuptools<70" > "$WORK/build-constraints.txt"
  "$UV" pip install -q -e "$WORK/IsaacLab/source/isaaclab" --build-constraint "$WORK/build-constraints.txt" \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  # isaaclab's deps can pull a different torch build; put back the aarch64 cu130 one Isaac Lab documents
  "$UV" pip install -q -U $TORCH_SPEC --index-url "$TORCH_INDEX"
  "$UV" pip install -q imageio imageio-ffmpeg matplotlib tensorboard pyyaml pytest
  PY=$WORK/venv/bin/python
  # Kit on aarch64 needs libgomp preloaded (it prints this exact fix otherwise)
  local gomp_torch
  gomp_torch=$(find "$WORK/venv" -path "*torch.libs/libgomp*.so*" | head -1)
  export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${gomp_torch:+:$gomp_torch}"
  # Kit writes caches under $HOME; the real home is unreadable from compute nodes
  mkdir -p "$WORK/kithome"
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  echo "isaac env ready in $((SECONDS - t0)) s: $($PY -c 'import torch,sys;print("python",sys.version.split()[0],"torch",torch.__version__,"cuda",torch.cuda.is_available())')"
}
# Run a Python entry point with Kit's caches pointed at scratch.
run_py() { HOME=$WORK/kithome XDG_CACHE_HOME=$WORK/kithome/.cache "$PY" "$@"; }
