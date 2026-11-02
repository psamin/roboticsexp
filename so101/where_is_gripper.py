"""Where dream's model of the arm puts the gripper point, live, while you move the arm by hand (motors stay off).

  ~/dream-env/bin/python so101/where_is_gripper.py [--arm-file ~/.dreamlabs/arms/so101-new-calib.json]

Prints x (forward), y (left) and z (up) in cm from the arm's base plate, five times a second, with the joint
correction on and off. Touch the closed fingertips to the table: z should read the table's height below the base
plate (0 if the base sits on the table). Read only: it never switches a servo on. Ctrl-C ends it.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yourdfpy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import dream_run  # noqa: E402
from dream_ik import specs, solver  # noqa: E402
from robots import feetech, so101  # noqa: E402


def built_in_point(side: str):
    spec = specs.FAAS_ARMS["so101_faas"][side]
    urdf = solver._urdf_with_tool(spec, spec.tools["tcp"])

    def point(q: np.ndarray) -> np.ndarray:
        urdf.update_cfg({n: float(v) for n, v in zip(spec.joints, list(q) + [0.0])})
        return urdf.get_transform(solver.TOOL_LINK, "arm_root")[:3, 3]

    return point


def arm_file_point(path: pathlib.Path):
    arm = json.loads(path.read_text())
    urdf = yourdfpy.URDF.load(str(path.parent / arm["urdf"]), load_meshes=False)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def point(q: np.ndarray) -> np.ndarray:
        urdf.update_cfg(dict(zip(names, map(float, q))))
        T = urdf.get_transform(arm["flange"], arm["base"])
        return T[:3, 3] + T[:3, :3] @ np.asarray(arm["tcp"], dtype=float)

    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-file", type=pathlib.Path, help="also show the point by your dreamcli tcp arm file")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--port", default=dream_run.PORT)
    args = parser.parse_args()

    calibration = feetech.read_calibration(str(dream_run.CALIBRATION), so101.JOINTS)
    off = np.radians([dream_run.zero_offsets_deg(calibration, args.side)[n] for n in dream_run.NAMES[:-1]])
    built_in = built_in_point(args.side)
    from_file = arm_file_point(args.arm_file.expanduser()) if args.arm_file else None
    arm = dream_run.ReadOnlyArm(args.port, calibration)
    print("cm from the base plate: x forward, y left, z up. Ctrl-C ends.")
    try:
        while True:
            joints, grip = arm.read()
            line = (f"built-in, correction on {np.round(100 * built_in(joints - off), 1)}   "
                    f"off {np.round(100 * built_in(joints), 1)}")
            if from_file is not None:
                line += f"   your file {np.round(100 * from_file(joints), 1)}"
            print(line + f"   gripper {grip:.2f}", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


if __name__ == "__main__":
    main()
