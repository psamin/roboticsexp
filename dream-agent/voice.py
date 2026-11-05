"""Push-to-talk speech in, macOS speech out.

listen() records only between two Enter presses, then sends that one clip to Deepgram. Nothing is streamed or sent
while you are not talking; clips under MIN_S or near-silent are dropped without a request. say() uses macOS `say`
(free, on-device).
"""
import io
import json
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np

RATE = 48000        # the MacBook microphone's own rate (Deepgram bills by duration, not size)
MIN_S = 0.4          # shorter clips are dropped without a request
MIN_RMS = 200        # int16 RMS: below this the clip is silence
DEEPGRAM = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=en"


def deepgram_key(env=Path(__file__).resolve().parent / ".env"):
    for line in env.read_text().splitlines():
        if line.startswith("DEEPGRAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"DEEPGRAM_API_KEY not in {env}")


def record():
    """Int16 mono samples between two Enter presses."""
    import sounddevice as sd
    chunks = []
    input("  [Enter] to talk ")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=lambda d, *_: chunks.append(d.copy())):
        input("  listening... [Enter] when done ")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.int16)


def worth_sending(audio):
    return len(audio) >= MIN_S * RATE and float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) >= MIN_RMS


def to_wav(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(audio, key=None):
    """One Deepgram request for one clip. Returns the transcript ('' if nobody spoke)."""
    req = urllib.request.Request(DEEPGRAM, to_wav(audio), {"Authorization": f"Token {key or deepgram_key()}",
                                                          "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    return res["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


def listen():
    audio = record()
    if not worth_sending(audio):
        print("  (too short or silent: nothing sent)")
        return ""
    return transcribe(audio)


def say(text):
    subprocess.run(["say", text], check=False)
