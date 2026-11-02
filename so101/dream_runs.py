"""Every live Dream-1 run on the SO-101 as one markdown table, from so101/logs/dream-live-*/.

  ~/dream-env/bin/python so101/dream_runs.py            # prints the table (paste into DREAM_DEBUG_LOG.md)

Per run: the settings dream_run.py recorded, which way the base turned (left is negative shoulder_pan), how low the
gripper point went (dream's so101_faas model of the arm, so approximate for arm-file runs), and how the run ended.
Runs before Oct 21 22:45 recorded neither their ending nor their tool point; KNOWN fills those in from the session.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import dream_run  # noqa: E402
from dream_ik import specs, solver  # noqa: E402
from robots import feetech, so101  # noqa: E402

LOGS = pathlib.Path(__file__).parent / "logs"
## Fake-servo and fake-model tests, not runs on the arm.
FAKE = {"20261020-224815", "20261020-224938", "20261020-231428", "20261020-231554"}
## What the logs of the early runs do not say, from the session notes: (ending, tool point).
KNOWN = {
    "20261020-225133": ("ran to its 30 s limit", None),
    "20261020-233037": ("stopped at 12.9 s, cause not recorded", None),
    "20261021-214105": ("error: gripper command -0.001 refused (bug, fixed)", "9.8 cm"),
    "20261021-214146": ("API key disabled mid-run", "9.8 cm"),
    "20261021-214339": ("API key refused at start", "9.8 cm"),
    "20261021-214640": ("stopped at 27.9 s, cause not recorded", "9.8 cm"),
    "20261021-214921": ("arm stopped answering (servo bus)", "9.8 cm"),
    "20261021-215646": ("arm stopped answering (servo bus)", None),
    "20261021-221225": ("top camera index showed a dark image", None),
    "20261021-223652": ("not recorded", "9.8 cm"),
    "20261021-224217": ("arm stopped answering; gripper pressed into the table 24 s (no floor)", "7.8 cm"),
}


def gripper_height_cm(joints: np.ndarray, off: np.ndarray, side: str) -> float:
    """The lowest the tool point went above the base plate, by dream's so101_faas model of that side's arm."""
    spec = specs.FAAS_ARMS["so101_faas"][side]
    urdf = solver._urdf_with_tool(spec, spec.tools["tcp"])
    lowest = np.inf
    for q in joints[::10] - off:
        urdf.update_cfg({n: float(v) for n, v in zip(spec.joints, list(q) + [0.0])})
        lowest = min(lowest, urdf.get_transform(solver.TOOL_LINK, "arm_root")[2, 3])
    return 100.0 * lowest


def tool_point(summary: dict, known: str | None) -> str:
    tool = summary.get("tool")
    if tool is None:
        if not str(summary.get("arm", "")).endswith(".json"):
            return "built-in (9.7 cm)"
        return known or "?"
    if tool.get("tcp") == "so101_faas built-in":
        return "built-in (9.7 cm)"
    # Arm file on gripper_frame_link: its origin is 9.8 cm out from the gripper body, the tool point along it.
    return f"{9.8 + 100.0 * float(tool['tcp'][2]):.1f} cm"


def main() -> None:
    calibration = feetech.read_calibration(str(dream_run.CALIBRATION), so101.JOINTS)
    print("| Run | Arm model | Tool point | Side | Correction | Floor | Joint margin | Steps/s | Cap | Ran | Calls "
          "| Base turned | Lowest gripper | Task | Ended |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in sorted(LOGS.glob("dream-live-*")):
        tag = run.name.removeprefix("dream-live-")
        if tag in FAKE:
            continue
        ended, known_tool = KNOWN.get(tag, ("not recorded", None))
        try:
            s = json.loads((run / "summary.json").read_text())
            t = np.load(run / "trajectory.npz")
        except (OSError, ValueError):
            print(f"| {tag} | | | | | | | | | | | | | | no log |")
            continue
        joints = np.asarray(t["joints"])
        arm = "your file" if str(s.get("arm", "")).endswith(".json") else "built-in"
        offsets = s.get("offsets_deg") or {}
        correction = "on" if any(abs(v) > 0 for v in offsets.values()) else "off"
        floor = f"{100 * s['floor_m']:.1f} cm" if s.get("floor_m") else "none"
        # Not recorded before 22:45: the margin came in at 22:11, and --no-limits (no floor, no margin) at 22:34.
        unrecorded = 0.0 if tag < "20261021-2211" or (tag >= "20261021-2234" and not s.get("floor_m")) else 10.0
        margin = s.get("joint_margin_deg", unrecorded)
        if joints.ndim != 2 or len(joints) == 0:
            print(f"| {tag} | {arm} | {tool_point(s, known_tool)} | | | | | | | 0 s | {len(s['chunks'])} | | | "
                  f"{s['task']} | {s.get('ended', ended)} |")
            continue
        tt = np.asarray(t["t"])
        pan = np.degrees(joints[:, 0])
        moving = tt > 3.0
        lo, hi = (pan[moving].min(), pan[moving].max()) if moving.any() else (pan[0], pan[0])
        turned = f"{lo:.0f}..{hi:.0f} ({'right' if hi > 20 else 'left' if lo < -20 else 'center'})"
        off = np.radians([offsets.get(n, 0.0) for n in dream_run.NAMES[:-1]])
        print(f"| {tag} | {arm} | {tool_point(s, known_tool)} | {s.get('side', 'right')} | {correction} | {floor} "
              f"| {margin:.0f} deg | {s.get('row_hz', 20)} | {s['max_speed_deg_s']:.0f} deg/s | {tt[-1]:.0f} s "
              f"| {len(s['chunks'])} | {turned} | {gripper_height_cm(joints, off, s.get('side', 'right')):.1f} cm "
              f"| {s['task']} "
              f"| {s.get('ended', ended)} |")
    del calibration  # read only to check the calibration file loads


if __name__ == "__main__":
    main()
