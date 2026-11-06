"""The SO-101 follower's safe joint range (robot_limits.yaml), for every experiment in this repo to load and narrow.

  ticks(calibration)        {joint: (lo, hi)} servo ticks
  lerobot_deg(calibration)  {joint: (lo, hi)} LeRobot degrees (0 at the middle of the joint's recorded range)
"""
import json
import pathlib

import yaml

FILE = pathlib.Path(__file__).resolve().parent / "robot_limits.yaml"
MAX_RES = 4095                                   # STS3215 ticks per turn less one, as LeRobot's degree conversion


def _load(calibration=None):
    """calibration: a LeRobot calibration file, {joint: {"range_min", "range_max"}}, or None for the file's own."""
    cfg = yaml.safe_load(FILE.read_text())
    if isinstance(calibration, dict):
        return cfg, calibration
    return cfg, json.loads(pathlib.Path(calibration or cfg["calibration"]).expanduser().read_text())


def ticks(calibration=None):
    """Per joint, (lo, hi) servo ticks: the recorded range less margin_deg, narrowed by the file's bounds.
    calibration: the LeRobot calibration file the caller drives the arm with (default: the file's)."""
    cfg, cal = _load(calibration)
    margin = cfg["margin_deg"] * MAX_RES / 360
    out = {}
    for name, (lo_deg, hi_deg) in cfg["joints"].items():
        c = cal[name]
        mid = (c["range_min"] + c["range_max"]) / 2
        lo, hi = c["range_min"] + margin, c["range_max"] - margin
        if lo_deg is not None:
            lo = max(lo, mid + lo_deg * MAX_RES / 360)
        if hi_deg is not None:
            hi = min(hi, mid + hi_deg * MAX_RES / 360)
        out[name] = (lo, hi)
    return out


def lerobot_deg(calibration=None):
    """Per joint, (lo, hi) in LeRobot degrees."""
    _, cal = _load(calibration)
    out = {}
    for name, (lo, hi) in ticks(calibration).items():
        mid = (cal[name]["range_min"] + cal[name]["range_max"]) / 2
        out[name] = ((lo - mid) * 360 / MAX_RES, (hi - mid) * 360 / MAX_RES)
    return out
