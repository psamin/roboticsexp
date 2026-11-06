"""Hands-free speech in, macOS speech out.

Listener.listen() reads the microphone locally and keeps only speech: a stretch starts when the level stays above
the room's background (tracked continuously) for 150 ms and ends after 0.8 s of quiet. Only that stretch goes to
Deepgram, one request per utterance; silence and clips under MIN_S are never sent. A request has to contain the wake
word (default "robot"; loose match, since speech recognition mangles it) unless it comes within FOLLOW_UP_S of the
last answer. say() uses macOS `say` (free, on-device).
"""
import io
import json
import queue
import re
import subprocess
import time
import urllib.request
import wave
from collections import deque
from pathlib import Path

import numpy as np

RATE = 48000        # the MacBook microphone's own rate (Deepgram bills by duration, not size)
FRAME_S = 0.03
MIN_S = 0.4         # shorter clips are dropped without a request
MIN_RMS = 150       # int16 RMS a voice must exceed whatever the background
START_S, END_S, MAX_S = 0.15, 0.8, 15.0
FOLLOW_UP_S = 20.0  # after an answer, the next utterance needs no wake word
DEEPGRAM = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en"
MIC_DENIED = ("the microphone gives pure silence: macOS has not allowed this app to use it. System Settings -> "
              "Privacy & Security -> Microphone: turn on your terminal app (Terminal, iTerm or Visual Studio Code), "
              "then quit and reopen it.")


def deepgram_key(env=Path(__file__).resolve().parent / ".env"):
    for line in env.read_text().splitlines():
        if line.startswith("DEEPGRAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"DEEPGRAM_API_KEY not in {env}")


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def worth_sending(audio, rate=RATE):
    return len(audio) >= MIN_S * rate and rms(audio) >= MIN_RMS


def to_wav(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(audio, key=None):
    """One Deepgram request for one clip. Returns the transcript ('' if nobody spoke)."""
    return transcribe_wav(to_wav(audio), key)


def transcribe_wav(wav_bytes, key=None):
    req = urllib.request.Request(DEEPGRAM, wav_bytes, {"Authorization": f"Token {key or deepgram_key()}",
                                                      "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    return res["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


class Segmenter:
    """Frame-by-frame speech detector: feed() returns the utterance (int16 samples) when one ends, else None."""

    def __init__(self):
        self.floor, self.pre = None, deque(maxlen=int(0.4 / FRAME_S))
        self.seg, self.voiced, self.quiet = None, 0, 0

    def feed(self, f):
        level = rms(f)
        if self.seg is None:
            self.floor = level if self.floor is None else 0.98 * self.floor + 0.02 * level   # room background
            self.pre.append(f)
            self.voiced = self.voiced + 1 if level > max(MIN_RMS, 3 * self.floor) else 0
            if self.voiced * FRAME_S >= START_S:
                self.seg, self.quiet = list(self.pre), 0
            return None
        self.seg.append(f)
        self.quiet = self.quiet + 1 if level < max(MIN_RMS, 3 * self.floor) else 0
        if self.quiet * FRAME_S < END_S and len(self.seg) * FRAME_S < MAX_S:
            return None
        out, self.seg, self.voiced = np.concatenate(self.seg), None, 0
        self.pre.clear()
        return out


def heard_wake(text, wake):
    """Loose match: Deepgram may write 'robot' as 'robo', 'rob it', 'row bot'. 'hey' must be the first word ('the
    yellow' holds 'hey' once spaces are dropped)."""
    if not wake:
        return True
    w = re.sub(r"[^a-z]", "", wake.lower())
    if w == "hey":
        words = re.findall(r"[a-z]+", text.lower())
        return bool(words) and words[0] in ("hey", "hay", "hei")
    t = re.sub(r"[^a-z]", "", text.lower())
    return w in t or (w == "robot" and any(v in t for v in ("robo", "robit", "rowbot", "roabot", "robbot")))


class Listener:
    def __init__(self, device=None, wake="hey"):
        self.device, self.wake, self.last_answer = device, wake, -1e9

    def answered(self):
        self.last_answer = time.time()

    def utterance(self):
        import sounddevice as sd
        q, seg, zeros = queue.Queue(), Segmenter(), 0
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=int(RATE * FRAME_S),
                            device=self.device, callback=lambda d, *_: q.put(d[:, 0].copy())):
            while True:
                f = q.get()
                zeros = zeros + 1 if not f.any() else 0
                if zeros * FRAME_S > 2.0:
                    raise SystemExit(MIC_DENIED)
                if (out := seg.feed(f)) is not None:
                    return out

    def listen(self):
        """Blocks until someone says a request; returns its transcript."""
        while True:
            audio = self.utterance()
            if not worth_sending(audio):
                continue
            text = transcribe(audio)
            if not text:
                continue
            follow_up = time.time() - self.last_answer < FOLLOW_UP_S
            if follow_up or heard_wake(text, self.wake):
                return text
            print(f"  (heard {text!r}: no '{self.wake}', ignored)")


def say(text):
    subprocess.run(["say", text], check=False)
