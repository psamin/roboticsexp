# Sourced by aw0-aspire jobs: job-scoped scratch in /dev/shm (home is nearly full), results straight to home.
set -euo pipefail
HOME_DIR=/nethome-instruction/$USER          # /nethome/<user> is a self-symlink on compute nodes
RESULTS=$HOME_DIR/aw0-aspire-results
WORK=/dev/shm/$USER-aa-${SLURM_JOB_ID:-local}
UV=$HOME_DIR/.local/bin/uv
mkdir -p "$WORK" "$RESULTS"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
trap 'echo "signal: cleaning up"; exit 1' TERM INT USR1
export UV_CACHE_DIR=$WORK/uv-cache UV_PYTHON_INSTALL_DIR=$WORK/uv-python HF_HOME=$WORK/hf XDG_CACHE_HOME=$WORK/xdg
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1

echo "== $(hostname) job ${SLURM_JOB_ID:-} $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader

vulkan_check() {   # SAPIEN (RoboTwin) renders through Vulkan; write an ICD manifest if the driver lacks one
  echo "== vulkan"
  ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true
  ldconfig -p | grep -E "libvulkan.so|libGLX_nvidia.so|libEGL_nvidia.so|libEGL.so.1" || true
  if ! compgen -G "/usr/share/vulkan/icd.d/nvidia_icd*.json" >/dev/null && ! compgen -G "/etc/vulkan/icd.d/nvidia_icd*.json" >/dev/null; then
    lib=$(ldconfig -p | awk '/libGLX_nvidia.so.0/{print $NF; exit}')
    if [ -n "$lib" ]; then
      mkdir -p "$WORK/icd"
      printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.3.0"}}\n' "$lib" > "$WORK/icd/nvidia_icd.json"
      export VK_ICD_FILENAMES=$WORK/icd/nvidia_icd.json
      echo "wrote ICD manifest -> $lib"
    else
      echo "no NVIDIA Vulkan ICD library on this node"
    fi
  fi
}
