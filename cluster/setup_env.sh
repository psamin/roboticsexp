# Sourced by cluster/*.sbatch after they set REPO. Builds a job-scoped Python env
# in /dev/shm from uv.lock and points every cache there, so nothing large lands in
# the 15 GB home quota. /dev/shm is RAM: it counts against the job's --mem.

set -euo pipefail
cd "$REPO"

UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || UV="${UV/#\/nethome\//\/nethome-instruction\/}"

JOB_SCRATCH="/dev/shm/$USER-$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"
cleanup() { rm -rf "$JOB_SCRATCH"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export UV_CACHE_DIR="$JOB_SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$JOB_SCRATCH/uv-python"
export UV_PROJECT_ENVIRONMENT="$JOB_SCRATCH/venv"
export XDG_CACHE_HOME="$JOB_SCRATCH/xdg-cache"
export MUJOCO_GL=egl
# Upstream README: full-precision matmul avoids TF32-related RL instability on Ampere+ GPUs.
export JAX_DEFAULT_MATMUL_PRECISION=highest
# Upstream README: a node-level LD_LIBRARY_PATH can shadow the pip-installed CUDA 12 libraries.
unset LD_LIBRARY_PATH

echo "node=$(hostname) job=$SLURM_JOB_ID repo=$REPO scratch=$JOB_SCRATCH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$UV" sync --locked --extra cuda --python 3.12
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
bash scripts/fetch_menagerie.sh

# Upstream train-jax-ppo writes videos with mediapy, which looks for `ffmpeg` on PATH.
mkdir -p "$JOB_SCRATCH/bin"
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$JOB_SCRATCH/bin/ffmpeg"
export PATH="$JOB_SCRATCH/bin:$PATH"
