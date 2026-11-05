"""rig.camera_view_check on saved overhead frames (real/logs, local only): the camera bumped between the 2026-11-02 runs
and 2026-11-04 is flagged; frames from one session are not."""
import sys
from pathlib import Path

import cv2
import pytest

REAL = Path(__file__).resolve().parents[1] / "real"
sys.path.insert(0, str(REAL))
import rig  # noqa: E402

LOGS = REAL / "logs"


def frame(name, i=None):
    if not (LOGS / name).exists():
        pytest.skip(f"real/logs/{name} not here")
    if i is None:
        return cv2.imread(str(LOGS / name))
    c = cv2.VideoCapture(str(LOGS / name))
    c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read()
    assert ok
    return f


def check(tmp_path, ref, frames):
    cv2.imwrite(str(tmp_path / "ref.png"), ref)
    return rig.camera_view_check(frames, tmp_path / "ref.png")


NOV2 = ("run-20261102-212201.mp4", 100)                  # arm holding, no hands in view


def test_nov2_view_against_nov4_frames_flags(tmp_path):
    ref = frame(*NOV2)
    for later in [frame("camera-check-20261104-231018.png"), frame("camera-check-20261104-232305.png"),
                  frame("run-20261104-231109.mp4", 1505), frame("run-20261104-231521.mp4", 3790),
                  frame("run-20261104-233217.mp4", 30)]:
        ok, msg = check(tmp_path, ref, [later])
        assert ok is False and "CAMERA MOVED" in msg, msg
    dx, dy, _ = rig.image_shift(ref, frame("camera-check-20261104-231018.png"))
    assert abs(dx + 9.5) < 1.5 and abs(dy + 19.3) < 1.5          # the scene sits up and left of the Nov 2 view


def test_hand_move_at_2323_flags(tmp_path):
    ok, msg = check(tmp_path, frame("camera-check-20261104-231018.png"), [frame("camera-check-20261104-232305.png")])
    assert ok is False and "CAMERA MOVED" in msg, msg                # ~6 px, the smallest move in the logs


def test_same_session_frames_pass(tmp_path):
    for name, ref_i, others in [("run-20261102-212201.mp4", 100, (730, 1200)),            # 730: a hand in the box
                                ("run-20261102-214053.mp4", 13010, (500, 25000)),
                                ("run-20261104-231109.mp4", 1505, (200, 2800)),
                                ("run-20261104-231521.mp4", 3790, (500, 7000)),
                                ("run-20261104-233217.mp4", 30, (3600, 18000))]:     # ~2 px step mid-run
        for i in others:
            ok, msg = check(tmp_path, frame(name, ref_i), [frame(name, i)])
            assert ok is True, f"{name} {ref_i} vs {i}: {msg}"
    ok, msg = check(tmp_path, frame("camera-check-20261104-231018.png"), [frame("run-20261104-231521.mp4", 3790)])
    assert ok is True, msg                                         # 23:10 check vs a 23:15 run (~1.4 px)


def test_dark_frame_is_not_a_pass(tmp_path):
    dark = frame("run-20261102-211230.mp4", 1183)                    # mean brightness ~30
    ref = frame("run-20261102-211230.mp4", 640)
    ok, msg = check(tmp_path, ref, [dark])
    assert ok is False and "DOES NOT MATCH" in msg, msg
    ok, msg = check(tmp_path, ref, [dark, frame("run-20261102-211230.mp4", 200)])
    assert ok is True, msg                                         # the best-matching frame decides


def test_no_reference(tmp_path):
    ok, msg = rig.camera_view_check([], tmp_path / "missing.png")
    assert ok is None and "WARNING" in msg
