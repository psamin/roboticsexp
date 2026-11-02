"""Nut-motion metrics from the physics-rate trace (t, q, q_rev, qd, w). Pure numpy, shared by expert tests and eval.

A flick = a burst where the nut's angular speed rises faster than friction could ever explain (> accel_thresh rad/s^2
sustained for >= 1 ms). Angular displacement per flick = rotation from that flick's start until the next flick (or
the end). Sustained speed = mean |w| over the time the nut is actually spinning (|w| > 5 rad/s).
"""
import numpy as np


def nut_metrics(trace, accel_thresh=3000.0, detach_t=None, pitch=None):
    T = np.asarray(trace, dtype=np.float64)
    if len(T) < 3:
        return dict(n_flicks=0)
    t, q, qr, qd, w = T.T
    dt = np.median(np.diff(t))
    acc = np.gradient(np.abs(w), dt)
    hot = acc > accel_thresh
    # group samples into events; merge gaps < 15 ms
    starts = []
    last = -1e9
    for i in np.flatnonzero(hot):
        if t[i] - last > 0.015:
            starts.append(i)
        last = t[i]
    min_len = max(1, int(round(0.001 / dt)))
    starts = [s for s in starts if hot[s:s + min_len].all() or min_len == 1]
    flicks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(t) - 1
        seg = slice(s, e + 1)
        flicks.append(dict(t=float(t[s]), peak_w=float(np.abs(w[seg]).max()),
                           dtheta_rad=float(qr[e] - qr[s]), dturns=float((qr[e] - qr[s]) / (2 * np.pi)),
                           axial_mm=float((q[e] - q[s]) * 1000)))
    spinning = np.abs(w) > 5.0
    k = max(1, int(round(0.010 / dt)))              # 10 ms moving average: ignores single-step contact spikes
    w10 = np.convolve(np.abs(w), np.ones(k) / k, mode="same")
    out = dict(
        n_flicks=len(flicks),
        flicks=flicks,
        peak_w=float(np.abs(w).max()),
        peak_w_10ms=float(w10.max()),
        sustained_w=float(np.abs(w[spinning]).mean()) if spinning.any() else 0.0,
        spin_time_s=float(spinning.sum() * dt),
        total_turns=float((qr[-1] - qr[0]) / (2 * np.pi)),
        axial_mm=float((q[-1] - q[0]) * 1000),
        mean_dturns_per_flick=float(np.mean([f["dturns"] for f in flicks])) if flicks else 0.0,
        mean_peak_w_per_flick=float(np.mean([f["peak_w"] for f in flicks])) if flicks else 0.0,
    )
    if pitch:
        # thread coupling check: travel vs rotation (should equal the pitch)
        if abs(qr[-1] - qr[0]) > 0.5:
            out["measured_pitch_mm"] = float((q[-1] - q[0]) / ((qr[-1] - qr[0]) / (2 * np.pi)) * 1000)
            out["pitch_mm"] = pitch * 1000
    return out
