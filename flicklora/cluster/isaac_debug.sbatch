#!/bin/bash
# Why does Isaac Sim 4.5 segfault at boot? Scan for missing system libs, then try boot+render variants.
#SBATCH -J flick-isaacdbg
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:45:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
source "${PROJ:-/nethome-instruction/pi34/roboticsexp/flicklora}/cluster/common.sh"
isaac_env
SITE=$($ISAAC_PY -c 'import isaacsim,os;print(os.path.dirname(isaacsim.__file__))')
echo "== apps:"; ls "$SITE/apps" | head -30
set +e +o pipefail   # diagnostics: keep going whatever fails
echo "== system libs Kit/RTX commonly need:"
for l in libGLU.so.1 libXt.so.6 libSM.so.6 libICE.so.6 libXrandr.so.2 libXi.so.6 libXinerama.so.1 libXcursor.so.1 libX11.so.6 libxkbcommon.so.0 libnvidia-gl libGLX_nvidia.so.0 libnvidia-rtcore libnvoptix.so.1; do
  printf "%-22s %s\n" "$l" "$(ldconfig -p | grep -m1 "$l" | awk '{print $NF}')"
done

# user-space copies of X/GL libs the nodes lack (from Rocky 9 RPMs, no root needed)
mkdir -p "$WORK/syslibs" && cd "$WORK/syslibs"
for rpm in m/mesa-libGLU-9.0.1-6.el9 l/libXt-1.2.0-6.el9 l/libSM-1.2.3-10.el9 l/libICE-1.0.10-8.el9; do
  curl -sfLO "https://dl.rockylinux.org/pub/rocky/9/AppStream/x86_64/os/Packages/$rpm.x86_64.rpm" && \
    rpm2cpio "$(basename $rpm).x86_64.rpm" | cpio -idm --quiet 2>/dev/null
done
ls "$WORK/syslibs/usr/lib64" 2>/dev/null | head
cd "$WORK"

run() {  # name, env..., -- experience
  local name=$1; shift
  echo "=================== VARIANT $name"
  timeout 600 env HOME=$WORK/kithome-$name XDG_CACHE_HOME=$WORK/kithome-$name/.cache "$@" 2>&1 \
    | grep -E "BOOT_OK|RENDER_OK|Segmentation|Fatal|Error\]|GPU|Vulkan|vk|driver|Traceback|Error:" | grep -v "py stderr\|deprecated\|requests\|asset.browser\|carb.audio" | head -40
  echo "exit[$name]=${PIPESTATUS[0]}"
  local klog=$(find "$WORK/kithome-$name" -name "kit_*.log" 2>/dev/null | head -1)
  [ -n "$klog" ] && { echo "--- kit log tail ($name)"; grep -vE "deprecated|ext: " "$klog" | tail -25; }
}
B=$PROJ/sim/boot_test.py
VARS=(OMNI_KIT_ACCEPT_EULA=YES)
run A_default  "${VARS[@]}" "$ISAAC_PY" "$B"
run B_libs     "${VARS[@]}" LD_LIBRARY_PATH="$WORK/syslibs/usr/lib64:${LD_LIBRARY_PATH:-}" "$ISAAC_PY" "$B"
for exp in isaacsim.exp.base.python.kit isaacsim.exp.base.zero_delay.kit; do
  [ -f "$SITE/apps/$exp" ] && run "C_${exp%.kit}" "${VARS[@]}" LD_LIBRARY_PATH="$WORK/syslibs/usr/lib64:${LD_LIBRARY_PATH:-}" "$ISAAC_PY" "$B" "$SITE/apps/$exp"
done
echo DEBUG_DONE
