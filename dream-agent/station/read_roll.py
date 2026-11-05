"""Prints the arm's wrist roll, read-only (no motor switched on), to check my_robot.ROLL_SHIFT_DEG.

Hold the hand upright: pointing forward, wrist camera on top, jaws closing left to right. so101_faas calls that -90 deg.
If the model-unit roll reads r instead, set ROLL_SHIFT_DEG to ROLL_SHIFT_DEG + (r + 90).

  ~/dream-env/bin/python ~/Documents/GitHub/roboticsexp/dream-agent/station/read_roll.py
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import my_robot  # noqa: E402
from robots import config  # noqa: E402

arm = my_robot.ReadOnlyArm(config.load(HERE / "my_robot.yaml").follower_r, my_robot.REAL_SIDE)
try:
    for _ in range(40):
        roll = np.degrees(arm.read().positions_rad[4])
        print(f"wrist roll {roll:+6.1f} deg model units (upright = -90) | {roll + my_robot.ROLL_SHIFT_DEG:+6.1f} arm units")
        time.sleep(0.5)
finally:
    arm.close()
