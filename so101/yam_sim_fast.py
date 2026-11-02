"""Dream YAM Sim (`dreamcli example yam-mujoco`) with the cameras rendered only when the model gets their frames.

  ~/dream-env/bin/python so101/yam_sim_fast.py [--verbose] [--show-side-panel]

Dream's example renders all three cameras on every 50 ms control step to refresh its camera panel, but sends the
model a frame only every 4th step (dream_control.FRAME_EVERY_STEPS, 0.2 s apart). On this Mac the three renders take
about 76 ms, so every step overran its 50 ms and the sim ran at about 60% speed. Here a step the model gets no frame
from reuses the last rendered images: the model sees exactly the frames it did, and the camera panel updates 5 times
a second instead of 20. Nothing in the installed example is changed; everything else is Dream's code as shipped.

The terminal half runs the example's own entry (yam_mujoco.main) and starts its viewer process through this file
(`--worker`), which patches the render and then runs the example's viewer_workspace.py as it ships.
"""

import pathlib
import runpy
import subprocess
import sys

import dream_examples.yam_mujoco as example

EXAMPLE = pathlib.Path(next(iter(example.__path__)))  # a namespace package: no __file__
sys.path.insert(0, str(EXAMPLE))  # the example's modules import each other by bare name


def patch_capture() -> None:
    """Render the cameras only on the steps whose frames join the model's windows."""
    import dream_control

    render = dream_control.capture
    last: dict = {}

    def capture(env, windows, keep=True):
        if keep or "images" not in last:
            last["images"] = render(env, windows, keep=keep)
        return last["images"]

    dream_control.capture = capture


class _Subprocess:
    """subprocess, with Popen starting this file as the viewer process in place of viewer_workspace.py."""

    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(args, *rest, **kwargs):
        if len(args) > 1 and pathlib.Path(args[1]).name == "viewer_workspace.py":
            args = [args[0], str(pathlib.Path(__file__).resolve()), "--worker", *args[2:]]
        return subprocess.Popen(args, *rest, **kwargs)


def main() -> None:
    if "--worker" in sys.argv:
        sys.argv = [str(EXAMPLE / "viewer_workspace.py"), *(a for a in sys.argv[1:] if a != "--worker")]
        patch_capture()
        runpy.run_path(str(EXAMPLE / "viewer_workspace.py"), run_name="__main__")
        return
    import yam_mujoco

    yam_mujoco.subprocess = _Subprocess()
    yam_mujoco.main()


if __name__ == "__main__":
    main()
