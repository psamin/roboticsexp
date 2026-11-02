"""Servo identification on the real arm with the paddle mounted. You run it; Claude never opens the port.

  cd so101-bounce            # everything runs from this project's environment (uv sync once)
  uv run python real/sysid.py set-hold   # torque off on the arm: hold the arm in the bouncing pose with the paddle
                                            # level (phone level app on the blade), press Enter; saved to hold_pose.json
  uv run python real/sysid.py check      # reads only: registers, voltage, temperature, bus timing (300 reads)
  uv run python real/sysid.py run        # ~9 min of small joint motions around the hold pose, logged

Start `run` with the arm resting folded on the table: it glides to the hold pose, tests each joint, glides back,
and turns the motors off where it started. Ctrl-C at any time does the same. It stops early if a servo passes
60 C or stays above 90 % load. Every joint test is small: 3 deg sines, 4-8 deg steps.

For each tuning (P 16 or 32, Acceleration 254 or 0) and each joint, the log records goals and measured positions at
the loop's full rate: steps, sine dwells at 1.5-4.5 Hz (the bounce rates), and a 0.5-6 Hz chirp.
Output: so101-bounce/real/logs/sysid-<time>.npz, fitted later by scripts/fit_servo.py.
"""
import json
import sys
import time

import numpy as np

import arm as A

TUNINGS = [{"P_Coefficient": 16, "Acceleration": 254}, {"P_Coefficient": 32, "Acceleration": 254},
           {"P_Coefficient": 16, "Acceleration": 0}, {"P_Coefficient": 32, "Acceleration": 0}]
ORDER = ["wrist_flex", "wrist_roll", "elbow_flex", "shoulder_lift", "shoulder_pan"]
TEMP_LIMIT = 60
LOAD_LIMIT = 900            # Present_Load is 0-1000 (per mille of max)


def segments():
    """(name, duration s, offset(t) in degrees)."""
    seg = []
    for amp in (4, -4, 8, -8):
        seg.append((f"step{amp:+d}", 0.8, lambda t, a=amp: a))
        seg.append(("rest", 0.5, lambda t: 0.0))
    for f in (1.5, 2.5, 3.5, 4.5):
        seg.append((f"sine{f}", 2.5, lambda t, f=f: 3.0 * np.sin(2 * np.pi * f * t)))
        seg.append(("rest", 0.5, lambda t: 0.0))
    f0, f1, T = 0.5, 6.0, 6.0
    seg.append(("chirp", T, lambda t: 3.0 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T)))))
    seg.append(("rest", 0.5, lambda t: 0.0))
    return seg


def set_hold():
    b = A.open_bus()
    try:
        b.disable_torque(A.ARM)
        input("Torque is OFF on the arm joints (support it!). Put it in the bouncing pose: forearm up, paddle level "
              "about 35 cm in front of the base. Press Enter to save...")
        pos = A.read_pos(b)
        A.HOLD.write_text(json.dumps({n: round(float(pos[n]), 2) for n in A.ARM}, indent=1))
        print("saved", A.HOLD, json.loads(A.HOLD.read_text()))
        input("Lower the arm back to its folded rest pose on the table and press Enter (arm motors stay off)...")
    finally:
        A.park(b)


def check():
    b = A.open_bus()
    try:
        regs = A.registers(b)
        print(json.dumps(regs, indent=1))
        ts, errors = [], 0
        for _ in range(300):
            t0 = time.perf_counter()
            try:
                A.read_pos(b)
            except Exception:
                errors += 1
            ts.append(time.perf_counter() - t0)
        ts = 1000 * np.array(ts)
        print(f"sync_read of 6 positions: p50 {np.percentile(ts, 50):.2f} ms, p95 {np.percentile(ts, 95):.2f}, "
              f"p99 {np.percentile(ts, 99):.2f}, max {ts.max():.1f}; errors {errors}/300")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (A.HERE / "logs").mkdir(exist_ok=True)
        (A.HERE / "logs" / f"check-{stamp}.json").write_text(json.dumps(
            {"registers": regs, "read_ms": ts.tolist(), "errors": errors}, indent=1))
    finally:
        b.disconnect(disable_torque=False)


def run():
    hold = json.loads(A.HOLD.read_text())
    b = A.open_bus()
    rest = A.hold_here(b)
    log = {k: [] for k in ("t", "goal", "pos", "tuning", "joint", "segment")}
    slow = {k: [] for k in ("t", "temp", "load", "volt")}
    t_start = time.perf_counter()
    try:
        print(f"gliding to the hold pose (error {A.glide(b, hold, 3.0):.1f} deg)")
        regs0 = A.registers(b)
        it = 0
        for ti, tuning in enumerate(TUNINGS):
            got = A.set_registers(b, tuning)
            print(f"tuning {tuning}: read back {got}")
            for ji, joint in enumerate(ORDER):
                for name, dur, fn in segments():
                    t0 = time.perf_counter()
                    while (t := time.perf_counter() - t0) < dur:
                        goal = dict(hold)
                        goal[joint] = hold[joint] + float(fn(t))
                        b.sync_write("Goal_Position", goal)
                        pos = A.read_pos(b)
                        now = time.perf_counter() - t_start
                        log["t"].append(now)
                        log["goal"].append([goal[n] for n in A.ARM])
                        log["pos"].append([pos[n] for n in A.ARM])
                        log["tuning"].append(ti)
                        log["joint"].append(A.ARM.index(joint))
                        log["segment"].append(name)
                        it += 1
                        if it % 25 == 0:
                            temp = b.sync_read("Present_Temperature", A.ARM, normalize=False)
                            load = b.sync_read("Present_Load", A.ARM, normalize=False)
                            volt = b.sync_read("Present_Voltage", A.ARM, normalize=False)
                            slow["t"].append(now)
                            slow["temp"].append([temp[n] for n in A.ARM])
                            slow["load"].append([load[n] % 1024 for n in A.ARM])     # bit 10 is the direction
                            slow["volt"].append([volt[n] for n in A.ARM])
                            if max(temp.values()) > TEMP_LIMIT:
                                raise RuntimeError(f"servo temperature {temp}")
                            if len(slow["load"]) > 3 and min(max(x) for x in slow["load"][-3:]) > LOAD_LIMIT:
                                raise RuntimeError(f"sustained load {slow['load'][-1]}")
                print(f"  {joint} done ({len(log['t']) / (time.perf_counter() - t_start):.0f} Hz average loop)")
    except (KeyboardInterrupt, RuntimeError) as e:
        print("stopping:", e)
    finally:
        try:
            A.set_registers(b, A.LEROBOT_DEFAULTS)
            print(f"back to the hold pose, then rest (errors {A.glide(b, hold, 1.5):.1f}, "
                  f"{A.glide(b, {n: rest[n] for n in A.ARM}, 3.0):.1f} deg); motors off")
        finally:
            A.park(b)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (A.HERE / "logs").mkdir(exist_ok=True)
        out = A.HERE / "logs" / f"sysid-{stamp}.npz"
        np.savez_compressed(out, t=np.array(log["t"]), goal=np.array(log["goal"]), pos=np.array(log["pos"]),
                            tuning=np.array(log["tuning"]), joint=np.array(log["joint"]),
                            segment=np.array(log["segment"]), slow_t=np.array(slow["t"]),
                            temp=np.array(slow["temp"]), load=np.array(slow["load"]), volt=np.array(slow["volt"]),
                            hold=json.dumps(hold), tunings=json.dumps(TUNINGS),
                            registers=json.dumps(regs0 if "regs0" in dir() else {}))
        print("log:", out, f"({len(log['t'])} samples)")


if __name__ == "__main__":
    {"set-hold": set_hold, "check": check, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
