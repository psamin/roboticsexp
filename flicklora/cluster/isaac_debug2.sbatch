#!/bin/bash
# Isaac Sim 4.5 segfaults at boot on driver 610 (all variants). Try 5.1 (py3.11) and 6.2 (py3.12), whose wheels are
# tagged manylinux_2_35 (node has glibc 2.34): force-install, list any GLIBC_2.35 symbol needs, then boot+render.
#SBATCH -J flick-isaacdbg2
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:50:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
set +e +o pipefail
# libGLU for iray (harmless if unused)
mkdir -p "$WORK/syslibs" && (cd "$WORK/syslibs" && curl -sfLO https://dl.rockylinux.org/pub/rocky/9/AppStream/x86_64/os/Packages/m/mesa-libGLU-9.0.1-6.el9.x86_64.rpm && rpm2cpio mesa-libGLU-9.0.1-6.el9.x86_64.rpm | cpio -idm --quiet)
export LD_LIBRARY_PATH=$WORK/syslibs/usr/lib64:${LD_LIBRARY_PATH:-}
export OMNI_KIT_ACCEPT_EULA=YES
for spec in "5.1.0 3.11" "6.2.0 3.12"; do
  set -- $spec; ver=$1; py=$2; env=$WORK/isaac-$ver
  echo "=================== ISAAC $ver (python $py)"
  t0=$(date +%s)
  "$UV" venv -q --python "$py" "$env"
  VIRTUAL_ENV=$env "$UV" pip install -q --python-platform x86_64-manylinux_2_35 "isaacsim[all,extscache]==$ver" \
      --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match 2>&1 | tail -5
  echo "install exit=$? took $(( $(date +%s) - t0 ))s"
  site=$($env/bin/python -c 'import isaacsim,os;print(os.path.dirname(isaacsim.__file__))' 2>&1); echo "site=$site"
  echo "-- .so files needing GLIBC_2.35+ (sample):"
  find "$env/lib" -name "*.so*" -type f 2>/dev/null | head -3000 | while read f; do
    objdump -T "$f" 2>/dev/null | grep -qE "GLIBC_2\.(3[5-9]|4[0-9])" && echo "  $f"; done | head -15
  timeout 600 env HOME=$WORK/kh-$ver XDG_CACHE_HOME=$WORK/kh-$ver/.cache "$env/bin/python" "$PROJ/sim/boot_test.py" 2>&1 \
    | grep -E "BOOT_OK|RENDER_OK|Segmentation|Fatal|GLIBC|Driver Version|Error\]" | grep -vE "deprecated|requests|asset.browser|carb.audio|py stderr" | head -25
  echo "exit[$ver]=${PIPESTATUS[0]}"
  rm -rf "$env" "$WORK/uv-cache"   # keep /dev/shm small between versions
done
echo DEBUG2_DONE
