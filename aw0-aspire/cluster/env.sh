# Sourced after common.sh. Builds the one Python env that runs RoboTwin 2.0 and the AW-0 policy in the same process
# (RoboTwin's eval_policy.py imports the policy), plus a CUDA 12.8 toolkit for the cuRobo / PyTorch3D extensions.
# Built extension wheels are cached in home (small); everything else lives in job scratch.
AWAM_COMMIT=275818e5787d59012eba4c43504342f1201ad25b      # ZhuoyangLiu2005/agentic-wam
ROBOTWIN_COMMIT=bf44be51cf5717a5595ce59447f2cf5263d2aa95  # pinned by awam codegen/sim/scripts/setup_external.sh
CUROBO_REF=v0.7.8
PYTORCH3D_REF=75ebeeaea0908c5527e7b1e305fbc7681382db47   # tag V0.7.8 (capital V; awam setup_external.sh pins a nonexistent v0.7.8)
TORCH_SPEC="torch==2.7.1+cu128 torchvision==0.22.1+cu128"
WHEELS=$HOME_DIR/aw0-aspire-cache/wheels-torch2.7.1-cu128-sm89
AWAM=$WORK/awam
SIM=$AWAM/codegen/sim
ROBOTWIN=$SIM/capsim/third_party/RoboTwin
CKPT=$AWAM/checkpoints
PY=$WORK/venv/bin/python
export PATH=$WORK/venv/bin:$PATH   # ninja for torch extension builds; `python` for scripts that shell out
export MAMBA_ROOT_PREFIX=$WORK/mamba TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}

t0=$SECONDS
step() { echo "== [$(( (SECONDS - t0) / 60 )) min] $*"; }

cuda_toolkit() {   # nvcc + headers + FFmpeg 7 (torchcodec) from conda-forge via a scratch micromamba
  step "cuda toolkit"
  mkdir -p "$WORK/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORK" bin/micromamba
  "$WORK/bin/micromamba" create -q -y -p "$WORK/cuda" -c conda-forge \
      cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-libraries-dev=12.8 "ffmpeg=7" >/dev/null
  export CUDA_HOME=$WORK/cuda PATH=$WORK/cuda/bin:$PATH
  export CPATH=$CUDA_HOME/targets/x86_64-linux/include LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib
  nvcc --version | tail -1
}

sources() {
  step "sources"
  git clone -q https://github.com/ZhuoyangLiu2005/agentic-wam.git "$AWAM" && git -C "$AWAM" checkout -q "$AWAM_COMMIT"
  git clone -q https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN" && git -C "$ROBOTWIN" checkout -q "$ROBOTWIN_COMMIT"
  git clone -q --depth 1 --branch "$CUROBO_REF" https://github.com/NVlabs/curobo.git "$ROBOTWIN/envs/curobo"
}

py_env() {
  step "python env"
  "$UV" venv -q --python 3.10 "$WORK/venv"
  VIRTUAL_ENV=$WORK/venv "$UV" pip install -q pip setuptools==69.5.1 wheel ninja $TORCH_SPEC \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  VIRTUAL_ENV=$WORK/venv "$UV" pip install -q -e "$AWAM" \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  # RoboTwin runtime from codegen/sim/requirements.txt, without its torch 2.4.1 / hub / imageio pins that clash with AW-0
  VIRTUAL_ENV=$WORK/venv "$UV" pip install -q sapien==3.0.0b1 mplib==0.2.1 numpy==1.26.4 opencv-python==4.10.0.84 \
      transforms3d==0.4.2 scipy==1.10.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 "zarr<3" h5py "pyglet<2" \
      "moviepy<2" av tyro==1.0.6 pydantic fastapi uvicorn cloudpickle scikit-learn warp-lang==1.12.0 pyyaml \
      viser openai termcolor matplotlib requests \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
}

extensions() {   # cuRobo and PyTorch3D compiled once against this torch, wheels kept in home
  step "extensions"
  mkdir -p "$WHEELS"
  compgen -G "$WHEELS/nvidia_curobo-*.whl" >/dev/null || \
      $PY -m pip wheel -q --no-build-isolation --no-deps -w "$WHEELS" "$ROBOTWIN/envs/curobo"
  compgen -G "$WHEELS/pytorch3d-*.whl" >/dev/null || \
      FORCE_CUDA=1 $PY -m pip wheel -q --no-build-isolation --no-deps -w "$WHEELS" \
          "git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_REF"
  ls -la "$WHEELS"
  VIRTUAL_ENV=$WORK/venv "$UV" pip install -q "$WHEELS"/*.whl numpy==1.26.4 $TORCH_SPEC \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
}

import_check() {   # RoboTwin hides a failed cuRobo import behind try/except; import it directly so it fails here
  step "import check"
  (cd "$ROBOTWIN" && $PY -c "
import torch, sapien, mplib, pytorch3d.ops, curobo.wrap.reacher.motion_gen
from envs.robot.planner import CuroboPlanner
print('imports ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())")
  VIRTUAL_ENV=$WORK/venv "$UV" pip check || echo "pip check reported conflicts (above)"
}

robotwin_assets() {
  step "robotwin assets"
  (cd "$ROBOTWIN/assets" && $PY _download.py >/dev/null && \
   for n in objects embodiments background_texture; do unzip -oq "$n.zip" && rm "$n.zip"; done)
  (cd "$ROBOTWIN" && $PY script/update_embodiment_config_path.py)
  du -sh "$ROBOTWIN/assets"
}

weights() {   # the release into checkpoints/; public components into the HF cache the launcher reads offline
  step "weights"
  local hf=$WORK/venv/bin/hf
  export HF_HUB_DISABLE_PROGRESS_BARS=1    # `hf download` prints the local path as its last stdout line
  $hf download miniFranka/agentic-wam-posttrain-robotwin-suit --local-dir "$CKPT/robotwin_release" >/dev/null
  SDAR=$($hf download JetLM/SDAR-1.7B-Chat | tail -1)
  QWEN=$($hf download Qwen/Qwen3-VL-2B-Instruct | tail -1)
  COSMOS=$($hf download nvidia/Cosmos-Reason1-7B | tail -1)
  WAN=$($hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth | tail -1)
  for d in "$SDAR" "$QWEN" "$COSMOS" "$WAN"; do [ -e "$d" ] || { echo "weights: no such path '$d'" >&2; exit 1; }; done
  du -sh "$CKPT/robotwin_release" "$HF_HOME"
}

site_config() {   # configs/robotwin/site.yaml and configs/paths/default.yaml for this scratch layout
  step "site config"
  local vk=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
  sed -e "s#^python_bin: .*#python_bin: $WORK/venv/bin#" -e "s#^vulkan_icd: .*#vulkan_icd: $vk#" \
      -e "s#^hf_cache_policy: .*#hf_cache_policy: $HF_HOME#" -e "s#^hf_cache_vision: .*#hf_cache_vision: $HF_HOME#" \
      -e "s#^code_model: .*#code_model: $SDAR#" \
      -e "s#  - \${repo}/third_party/ffmpeg7-lib#  - $WORK/cuda/lib#" \
      "$AWAM/configs/robotwin/site.example.yaml" > "$AWAM/configs/robotwin/site.yaml"
  $PY - "$AWAM/configs/paths/default.yaml" "$WAN" "$COSMOS" "$SDAR" "$QWEN" <<'PY'
import sys, yaml
path, wan, cosmos, sdar, qwen = sys.argv[1:]
cfg = yaml.safe_load(open(path))
cfg.update(wan_vae=wan, text_encoder=cosmos, code_model=sdar, code_vision=qwen)
yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
print({k: cfg[k] for k in ("wan_vae", "text_encoder", "code_model", "code_vision")})
PY
}

sim_config() {   # codegen/sim site.yaml (robotwin_py, robotwin_root) for the code-as-policy harness
  step "codegen/sim config"
  (cd "$SIM" && SETUP_PYTHON=$PY bash scripts/setup_external.sh configure)
}

build_sim() { cuda_toolkit; sources; py_env; extensions; robotwin_assets; sim_config; import_check; step "sim env ready"; }
build_all() { build_sim; weights; site_config; step "env ready"; }
