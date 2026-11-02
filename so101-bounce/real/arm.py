"""The SO-101 follower's motor bus, opened directly (not through SOFollower.connect(), which turns torque off while it
reconfigures the motors and would drop an arm holding a paddle). Shared by the real-arm scripts.

Positions are LeRobot's calibrated degrees (the same numbers teleop and datasets use). Run these scripts with the
project environment: `uv run python real/<script>.py` from so101-bounce (LeRobot is vendored in third_party/).
Nothing else may have the serial port open. Ports and calibration: real/config.yaml.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

import yaml

HERE_ = Path(__file__).parent
CONFIG = yaml.safe_load((HERE_ / "config.yaml").read_text())
PORT = CONFIG["follower_port"]
CALIBRATION = HERE_ / CONFIG["follower_calibration"]          # this project's copy
LEROBOT_CALIBRATION = HF_LEROBOT_CALIBRATION / "robots/so_follower/my_follower.json"
USE_GRIPPER = bool(CONFIG.get("use_gripper", True))
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"] + (["gripper"] if USE_GRIPPER else [])
ARM = NAMES[:5]
HERE = Path(__file__).parent
HOLD = HERE / "hold_pose.json"
LEROBOT_DEFAULTS = {"P_Coefficient": 16, "D_Coefficient": 32, "Acceleration": 254}
# What the bouncing policy is trained for: Acceleration 0 (no ramp; 254 may cap the paddle at ~39 rad/s^2, too slow
# to launch the ball). The runner writes these and reads them back before any motion.
BOUNCE_REGISTERS = {"Acceleration": 0}


def port_free():
    out = subprocess.run(["lsof", PORT], capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit(f"{PORT} is open by another program (a live run?):\n{out}\nStop it first.")


def open_bus():
    port_free()
    cal = json.loads(CALIBRATION.read_text())
    if LEROBOT_CALIBRATION.exists() and json.loads(LEROBOT_CALIBRATION.read_text()) != cal:
        print(f"warning: {CALIBRATION} differs from LeRobot's {LEROBOT_CALIBRATION} (recalibrated since the copy?); "
              f"using this project's copy", file=sys.stderr)
    b = FeetechMotorsBus(PORT, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper"
                                         else MotorNormMode.DEGREES) for i, n in enumerate(NAMES)},
                         calibration={n: MotorCalibration(**cal[n]) for n in NAMES})
    try:
        b.connect()
    except RuntimeError as e:
        if "Missing motor IDs" in str(e):
            sys.exit(f"{e}\n\nA servo is not answering. If it is ID 6 (the gripper): (1) unplug the arm's power supply "
                     "for 5 s and plug it back in (clears a tripped overload), then run `uv run python real/paddle.py grip`; "
                     "(2) if it is still missing, reseat the gripper servo's 3-wire cable at both ends; (3) or zip-tie the "
                     "paddle handle into the jaws and set `use_gripper: false` in real/config.yaml.")
        raise
    return b


GRIP_OPEN = 90.0                            # LeRobot gripper units: 0 = closed (range_min), 100 = open
GRIP_FILE = Path(__file__).parent / "grip.json"   # the gentle clamp goal found by paddle.py grip


def grip_goal(present):
    """Gripper goal that keeps the handle clamped: the saved gentle-clamp goal, else where it is now."""
    try:
        return float(json.loads(GRIP_FILE.read_text())["goal"])
    except (OSError, KeyError, ValueError):
        return float(present)


def clamp_paddle(b, squeeze: float = 4.0):
    """Close the jaws slowly until they stop on the handle, then hold just past contact (squeeze, in gripper units)
    with the gripper's torque limited, so it never stalls at full force (which trips its overload protection)."""
    b.write("Torque_Limit", "gripper", 400, normalize=False)   # RAM limit (Max_Torque_Limit is only the power-on default)
    target = read_pos(b)["gripper"]
    for _ in range(80):
        target = max(target - 1.5, 0.0)
        b.sync_write("Goal_Position", {"gripper": target})
        time.sleep(0.04)
        if read_pos(b)["gripper"] - target > 3.0:          # the jaws stopped: they are on the handle
            break
    contact = read_pos(b)["gripper"]
    goal = max(contact - squeeze, 0.0)
    b.sync_write("Goal_Position", {"gripper": goal})
    GRIP_FILE.write_text(json.dumps({"contact": round(contact, 1), "goal": round(goal, 1)}))
    return contact, goal


def pause(msg):
    """input() where the next step folds the arm back and turns it off: Ctrl-C counts as Enter."""
    try:
        input(msg)
    except (KeyboardInterrupt, EOFError):                          # no keyboard (scripted run) counts as Enter too
        print()


def read_pos(b):
    return b.sync_read("Present_Position")


def hold_here(b):
    """Turn torque on without moving the arm: goal = present position; the gripper keeps squeezing the handle."""
    pos = read_pos(b)
    goal = {**pos, "gripper": grip_goal(pos["gripper"])} if USE_GRIPPER else pos
    b.sync_write("Goal_Position", goal)
    b.enable_torque()
    return pos


def park(b):
    """End of a script: arm joints off (the arm should be folded at rest), gripper left on so the paddle stays held."""
    b.disable_torque(ARM)
    b.disconnect(disable_torque=False)


def glide(b, target: dict, seconds: float = 3.0, hz: int = 50):
    start = read_pos(b)
    names = [n for n in target if n in start]
    a, z = np.array([start[n] for n in names]), np.array([target[n] for n in names])
    steps = max(1, int(seconds * hz))
    for i in range(1, steps + 1):
        s = i / steps
        s = s * s * (3 - 2 * s)
        b.sync_write("Goal_Position", {n: float(v) for n, v in zip(names, a + s * (z - a))})
        time.sleep(1 / hz)
    time.sleep(0.5)
    end = read_pos(b)
    return max(abs(end[n] - target[n]) for n in names)


def registers(b, names=("P_Coefficient", "D_Coefficient", "I_Coefficient", "Acceleration", "Max_Torque_Limit",
                        "Protection_Current", "Overload_Torque", "Protection_Time", "Present_Voltage",
                        "Present_Temperature", "Return_Delay_Time")):
    out = {}
    for r in names:
        try:
            out[r] = {m: int(b.read(r, m, normalize=False)) for m in ARM}
        except Exception as e:                                    # register missing on this firmware
            out[r] = f"unreadable: {e}"
    return out


def set_registers(b, values: dict):
    """Write RAM copies of tuning registers with torque on, then read them back (the servos keep the EEPROM value
    after a power cycle, since torque-on writes go to RAM only)."""
    for r, v in values.items():
        for m in ARM:
            b.write(r, m, int(v), normalize=False)
    return {r: {m: int(b.read(r, m, normalize=False)) for m in ARM} for r in values}
