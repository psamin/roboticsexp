# Sourced by every FlickLoRA job. Sets up job-scoped scratch in /dev/shm and two separate environments:
#   isaac_env  -> Python 3.10 + Isaac Sim 4.5.0 (newest wheel that installs on glibc 2.34; 5.x/6.x need 2.35)
#   lerobot_env-> Python 3.12 + LeRobot 0.6.1 [smolvla,peft]
# Results go straight to home (scratch is wiped when the job ends).
set -euo pipefail
HOME_DIR=/nethome-instruction/$USER          # /nethome/<user> is a self-symlink on compute nodes
PROJ=${PROJ:-$HOME_DIR/roboticsexp/flicklora}
[ -d "$PROJ" ] || PROJ="${PROJ/#\/nethome\//\/nethome-instruction\/}"
RESULTS=$HOME_DIR/flicklora-results
WORK=/dev/shm/$USER-flick-${SLURM_JOB_ID:-local}
UV=$HOME_DIR/.local/bin/uv
ARCH=$(uname -m)
if [ "$ARCH" = aarch64 ]; then
  # DGX Spark (GB10, Ubuntu): the home uv is an x86 binary; fetch an arm64 uv into scratch. Isaac Sim 5.1 is the
  # first release with official DGX Spark support (4.5 segfaults at boot on the x86 nodes' 610 driver).
  export UV_PYTHON_INSTALL_DIR=$WORK/uv-python
  mkdir -p "$WORK/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
  UV=$WORK/bin/uv
  ISAAC_VERSION=${ISAAC_VERSION:-5.1.0}; ISAAC_PYVER=${ISAAC_PYVER:-3.11}
else
  ISAAC_VERSION=${ISAAC_VERSION:-4.5.0}; ISAAC_PYVER=${ISAAC_PYVER:-3.10}
fi
REF_COMMIT=3089cc9e5d65ec476e364f71346cb3f02cf48db7     # Tech-Multiverse/omniverse-nut-and-bolt-digital-twin
LEROBOT_VERSION=0.6.1
mkdir -p "$WORK" "$RESULTS"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
trap 'echo "signal: cleaning up"; exit 1' TERM INT USR1
export UV_CACHE_DIR=$WORK/uv-cache HF_HOME=$WORK/hf TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1

echo "== $(hostname) job ${SLURM_JOB_ID:-} $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

fetch_ref() {   # reference CAD + USD, pinned
  [ -d "$WORK/ref" ] && return
  git clone -q https://github.com/Tech-Multiverse/omniverse-nut-and-bolt-digital-twin.git "$WORK/ref"
  git -C "$WORK/ref" checkout -q "$REF_COMMIT"
}

vulkan_check() {
  echo "== vulkan"
  ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true
  ldconfig -p | grep -E "libvulkan.so|libGLX_nvidia.so|libEGL_nvidia.so" || true
  if ! compgen -G "/usr/share/vulkan/icd.d/nvidia_icd*.json" >/dev/null && ! compgen -G "/etc/vulkan/icd.d/nvidia_icd*.json" >/dev/null; then
    lib=$(ldconfig -p | awk '/libGLX_nvidia.so.0/{print $NF; exit}')
    if [ -n "$lib" ]; then   # driver has the Vulkan ICD library but no manifest: write one
      mkdir -p "$WORK/icd"
      printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.3.0"}}\n' "$lib" > "$WORK/icd/nvidia_icd.json"
      export VK_ICD_FILENAMES=$WORK/icd/nvidia_icd.json
      echo "wrote ICD manifest -> $lib"
    else
      echo "WARNING: no NVIDIA Vulkan ICD found; RTX rendering will fail (physics still works)"
    fi
  fi
}

isaac_env() {
  vulkan_check
  "$UV" venv -q --python "$ISAAC_PYVER" "$WORK/isaac"
  VIRTUAL_ENV=$WORK/isaac "$UV" pip install -q "isaacsim[all,extscache]==$ISAAC_VERSION" \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  VIRTUAL_ENV=$WORK/isaac "$UV" pip install -q imageio imageio-ffmpeg
  ISAAC_PY=$WORK/isaac/bin/python
  if [ "$ARCH" = aarch64 ]; then   # Kit on aarch64 needs libgomp preloaded (it prints this exact fix)
    gomp_torch=$(find "$WORK/isaac" -path "*torch.libs/libgomp*.so*" | head -1)
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${gomp_torch:+:$gomp_torch}"
    echo "LD_PRELOAD=$LD_PRELOAD"
  fi
  # Kit writes caches under $HOME; real home is unreadable from compute nodes, so give it a scratch home
  mkdir -p "$WORK/kithome"
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  echo "isaac env ready: isaacsim $ISAAC_VERSION on $ARCH, python $($ISAAC_PY -c 'import sys;print(sys.version.split()[0])')"
}
isaac_run() { HOME=$WORK/kithome XDG_CACHE_HOME=$WORK/kithome/.cache "$ISAAC_PY" "$@"; }

lerobot_env() {
  "$UV" venv -q --python 3.12 "$WORK/lerobot"
  VIRTUAL_ENV=$WORK/lerobot "$UV" pip install -q "lerobot[smolvla,peft,training,av-dep]==$LEROBOT_VERSION" \
      imageio imageio-ffmpeg matplotlib   # to_lerobot.py decodes the raw mp4s with imageio; analyze.py plots
  LR_PY=$WORK/lerobot/bin/python
  LR_BIN=$WORK/lerobot/bin
  echo "lerobot env ready: $($LR_PY -c 'import lerobot,peft,torch;print(lerobot.__version__, "peft", peft.__version__, "torch", torch.__version__)')"
}
