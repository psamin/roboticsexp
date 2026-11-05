"""Offline checks: no microphone, no network, no arm, no credits.  uv run python -m pytest -q test_offline.py"""
import io
import json
import wave

import numpy as np
import pytest

import robot
import voice


def test_silence_and_short_clips_are_never_sent():
    assert not voice.worth_sending(np.zeros(voice.RATE * 3, np.int16))                   # silence
    assert not voice.worth_sending((3000 * np.ones(int(0.2 * voice.RATE))).astype(np.int16))   # too short
    t = np.arange(voice.RATE) / voice.RATE
    assert voice.worth_sending((3000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16))   # one second of tone


def test_wav_is_mono_int16_at_the_mic_rate():
    audio = (1000 * np.ones(1600)).astype(np.int16)
    with wave.open(io.BytesIO(voice.to_wav(audio))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (voice.RATE, 1, 2, 1600)


def test_transcribe_sends_one_request_with_the_key(monkeypatch):
    sent = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout):
        sent.append(req)
        return Reply(json.dumps({"results": {"channels": [{"alternatives": [{"transcript": " hand me the box "}]}]}}).encode())

    monkeypatch.setattr(voice.urllib.request, "urlopen", fake_urlopen)
    assert voice.transcribe(np.ones(16000, np.int16), key="k") == "hand me the box"
    assert len(sent) == 1 and sent[0].get_header("Authorization") == "Token k"
    assert sent[0].full_url.startswith("https://api.deepgram.com/v1/listen")


def test_mock_robot_logs_steps_and_returns_a_photo():
    if not robot.MOCK_SCENE.exists():
        pytest.skip("mock scene image is local only")
    r = robot.MockRobot()
    assert len(r.look()) > 1000
    assert "done (mock)" in r.run_step("Pick up the box and place it on the plate", 30)
    assert r.steps == ["Pick up the box and place it on the plate"]


def test_live_robot_needs_a_typed_y(monkeypatch):
    ran = []
    monkeypatch.setattr(robot.subprocess, "run", lambda *a, **k: ran.append(a) or type("R", (), {"returncode": 0})())
    assert robot.LiveRobot(confirm=lambda _: "").run_step("Pick up the box", 30).startswith("skipped")
    assert robot.LiveRobot(confirm=lambda _: "yes").run_step("Pick up the box", 30).startswith("skipped")
    assert not ran
    out = robot.LiveRobot(confirm=lambda _: "y").run_step("Pick up the box", 30)
    assert out.startswith("finished") and ran and "--live" in ran[0][0]
