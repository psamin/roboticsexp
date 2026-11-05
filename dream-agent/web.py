"""The fetch agent in a browser page: the page hears you (it asks for the microphone itself), shows what the agent
sees and does, speaks its answers, and has Run / Skip buttons for real arm steps.

  uv run python web.py                  # live overhead camera, plan only (free); open http://127.0.0.1:8000
  uv run python web.py --robot dry      # + Dream-1 inference per step, the arm does not move (billed)
  uv run python web.py --robot live     # + Dream-1 moves the arm (billed; each step needs Run or a spoken "yes")
  uv run python web.py --robot mock     # a saved photo instead of the camera

The page detects speech itself and sends only finished utterances (WAV) here; this server sends them to Deepgram
(the key stays on this Mac) and the transcript to the planner. Served on 127.0.0.1 only.
"""
import argparse
import asyncio
import io
import itertools
import json
import re
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web
from claude_agent_sdk import ClaudeSDKClient

import voice
from agent import options, turn
from robot import FRONT_CAMERA, MOCK_SCENE, WRIST_CAMERA, CameraHub, LiveRobot, MockRobot

HERE = Path(__file__).resolve().parent
YES = re.compile(r"^\W*(yes|yeah|yep|go|go ahead|do it|run it|okay|ok|sure)\b", re.I)
NO = re.compile(r"^\W*(no|nope|stop|skip|cancel|don't|do not)\b", re.I)
STOP = re.compile(r"\b(stop|halt|freeze|abort)\b", re.I)


class Session:
    def __init__(self, robot, args):
        self.robot, self.args, self.sockets = robot, args, set()
        self.busy, self.cost, self.wake, self.last_answer = False, 0.0, args.wake, -1e9
        self.pending, self.ids, self.loop, self.client = {}, itertools.count(1), None, None

    async def emit(self, ev):
        if ev["type"] == "done":
            self.cost += ev["cost"]
            ev["total_cost"] = self.cost
        for ws in list(self.sockets):
            try:
                await ws.send_json(ev)
            except ConnectionError:
                self.sockets.discard(ws)

    async def status(self):
        await self.emit({"type": "status", "busy": self.busy, "robot": getattr(self.robot, "mode", "mock"),
                         "calls_left": self.robot.budget.left, "total_cost": self.cost, "wake": self.wake})

    def confirm(self, question):
        """LiveRobot calls this from its worker thread; the answer comes from a button or a spoken yes / no."""
        fut = asyncio.run_coroutine_threadsafe(self._ask(question), self.loop)
        return "y" if fut.result() else "n"

    async def _ask(self, question):
        cid = next(self.ids)
        self.pending[cid] = self.loop.create_future()
        await self.emit({"type": "confirm", "id": cid, "text": question})
        try:
            return await self.pending[cid]
        finally:
            del self.pending[cid]
            await self.emit({"type": "confirmed", "id": cid})

    def answer(self, ok, cid=None):
        for k, fut in self.pending.items():
            if (cid is None or k == cid) and not fut.done():
                fut.set_result(ok)
                return True
        return False

    def stop_arm(self):
        stop = getattr(self.robot, "stop", None)
        if stop and stop():
            asyncio.create_task(self.emit({"type": "note", "text": "stopping: the arm glides back"}))
            return True
        return False

    async def request(self, text):
        if self.busy:
            await self.emit({"type": "note", "text": f"still working, ignored: {text}"})
            return
        self.busy = True
        await self.status()
        await self.emit({"type": "user", "text": text})
        try:
            await turn(self.client, text, self.emit)
        except Exception as e:                                # keep the page alive; show what broke
            await self.emit({"type": "note", "text": f"error: {e}"})
        finally:
            self.busy, self.last_answer = False, time.time()
            await self.status()

    async def heard(self, wav_bytes):
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate, audio = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16)
        if not voice.worth_sending(audio, rate):              # never pay for silence
            return
        text = await asyncio.to_thread(voice.transcribe_wav, wav_bytes)
        if not text:
            return
        if STOP.search(text) and self.stop_arm():
            await self.emit({"type": "user", "text": text})
            return
        if self.pending and (YES.match(text) or NO.match(text)):
            await self.emit({"type": "user", "text": text})
            self.answer(bool(YES.match(text)))
            return
        follow_up = time.time() - self.last_answer < voice.FOLLOW_UP_S
        if follow_up or voice.heard_wake(text, self.wake):
            asyncio.create_task(self.request(text))
        else:
            await self.emit({"type": "ignored", "text": text, "wake": self.wake})


def frame_jpeg(session):
    hub = getattr(session.robot, "hub", None)
    if hub is None:
        img = cv2.imread(str(session.robot.scene))
    elif hub.paused or hub.frame is None:
        img = np.full((360, 640, 3), 40, np.uint8)
        cv2.putText(img, "camera in use by Dream-1" if hub.paused else "starting camera...", (150, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    else:
        img = hub.frame
    img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


async def video(request):
    """MJPEG stream of the overhead camera (~10 fps) for the page."""
    resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    try:
        while True:
            jpg = await asyncio.to_thread(frame_jpeg, request.app["session"])
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                             + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def frame(request):
    """Newest frame of one camera for so101/dream_run.py --frames (X-Frame-Age: seconds since capture)."""
    hub = request.app["hubs"].get(request.match_info["name"])
    if hub is None or hub.frame is None:
        raise web.HTTPServiceUnavailable()
    img, at = hub.frame, hub.at
    jpg = await asyncio.to_thread(lambda: cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    return web.Response(body=jpg, content_type="image/jpeg", headers={"X-Frame-Age": f"{time.time() - at:.4f}"})


async def index(request):
    return web.FileResponse(HERE / "static/index.html")


async def socket(request):
    s = request.app["session"]
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    s.sockets.add(ws)
    await s.status()
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            asyncio.create_task(s.heard(msg.data))
        elif msg.type == WSMsgType.TEXT:
            m = json.loads(msg.data)
            if m["type"] == "text" and m["text"].strip():
                asyncio.create_task(s.request(m["text"].strip()))
            elif m["type"] == "stop":
                s.stop_arm()
            elif m["type"] == "confirm":
                s.answer(bool(m["ok"]), m.get("id"))
            elif m["type"] == "mode" and isinstance(s.robot, LiveRobot) and m["mode"] in ("camera", "dry", "live"):
                s.robot.mode = m["mode"]                # each dry / live step still needs Run or a spoken yes
                await s.emit({"type": "note", "text": f"mode: {m['mode']}"})
                await s.status()
            elif m["type"] == "wake":
                s.wake = m["word"]
                await s.status()
    s.sockets.discard(ws)
    return ws


async def start(app):
    s = app["session"]
    s.loop = asyncio.get_running_loop()
    s.client = ClaudeSDKClient(options=options(s.robot, s.emit, s.args.handoff, s.args.model))
    await s.client.connect()


async def stop(app):
    await app["session"].client.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=("camera", "dry", "live", "mock"), default="camera")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wake", default="robot", help="word a request must contain ('' for none)")
    ap.add_argument("--scene", default=str(MOCK_SCENE), help="mock: the table image look() returns")
    ap.add_argument("--dream-args", default="", help="live: extra so101/dream_run.py flags")
    ap.add_argument("--handoff", default="the plate", help="where 'give it to me' puts things")
    ap.add_argument("--max-calls", type=int, default=100, help="cap on estimated billed Dream calls this session")
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    session = Session(None, args)
    hubs = {}
    if args.robot != "mock":                    # one owner per camera, at the size the Dream model was set up for
        fine_tune = "dream-1-ft" in args.dream_args
        hubs = {"front": CameraHub(FRONT_CAMERA, (640, 480) if fine_tune else (1280, 720)),
                "wrist": CameraHub(WRIST_CAMERA, (640, 480))}
    session.robot = (MockRobot(args.scene, max_calls=args.max_calls) if args.robot == "mock"
                     else LiveRobot(args.dream_args, confirm=session.confirm, max_calls=args.max_calls,
                                    mode=args.robot, hub=hubs["front"],
                                    frames_url=f"http://127.0.0.1:{args.port}"))
    app = web.Application()
    app["session"], app["hubs"] = session, hubs
    app.router.add_get("/", index)
    app.router.add_get("/ws", socket)
    app.router.add_get("/video.mjpg", video)
    app.router.add_get("/frame/{name}", frame)
    app.router.add_static("/static/", HERE / "static")
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    print(f"dream-agent ({args.robot} robot): open http://127.0.0.1:{args.port}")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
