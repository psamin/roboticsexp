"""Scripted Franka flick expert (privileged nut state; outputs only the 7 arm joint targets).

Cycle: APPROACH (to a raised pose to one side of the nut) -> LOWER -> FLICK (one joint-6 target jump; the wrist's
torque/speed limits make the swing, ~0.6 m/s at the fingertip) -> LIFT -> BACK (raised swing back, re-aimed at the
nut's new position) -> LOWER -> FLICK ... until the nut leaves the bolt.
Aim: the fingertip's lowest point passes at the hex flat radius (always catches a corner); deeper after a miss.
"""
from __future__ import annotations

import math

import numpy as np

import franka as FR


class FrankaFlickExpert:
    def __init__(self, env, rng, flick_sign=1, swing_deg=30.0, depth=(-0.15, 0.15), lift=0.035):
        self.env, self.rng = env, rng
        self.sign, self.swing = flick_sign, math.radians(swing_deg)
        self.depth, self.lift = depth, lift
        self.bias, self.flick_peak, self.misses, self.flicks = 0.0, 0.0, 0, 0
        self.q_warm = FR.Q_READY.copy()
        self.phase, self.path = "APPROACH", None
        self.ik_err = []

    def _solve(self, tip):
        best = None
        for seed in (self.q_warm, FR.Q_READY, FR.Q_READY + np.array([0, 0, 0, 0, 0, 0, -math.pi / 2])):
            q, e = FR.flick_config(seed, tip, np.eye(4), self.env.tip_z)
            if best is None or e < best[1]:
                best = (q, e)
            if e < 1e-4:
                break
        self.ik_err.append(best[1])
        self.q_warm = best[0]
        return best[0]

    def _aim(self, info):
        r_flat, r_corner = self.env.r_band
        d = self.rng.uniform(*self.depth) + self.bias
        c = info["nut_center"]
        z_tip = c[2] + r_flat + d * (r_corner - r_flat)
        qb = self._solve(c + np.array([0, 0, z_tip - c[2]]))
        qr = self._solve(c + np.array([0, 0, z_tip - c[2] + self.lift]))
        off = np.zeros(7)
        off[5] = self.sign * self.swing
        self.q_start, self.q_end = qb - off, qb + off
        self.q_rstart, self.q_rend = qr - off, qr + off

    def _plan(self, q_from, q_to, steps):
        s = np.linspace(0, 1, steps + 1)[1:]
        s = s * s * (3 - 2 * s)
        self.path = [np.asarray(q_from) + (np.asarray(q_to) - np.asarray(q_from)) * v for v in s]

    def act(self, state, info):
        q = np.asarray(state, float)
        if self.phase == "FLICK":
            self.flick_peak = max(self.flick_peak, abs(info["w"]))
        if info["detached"]:
            if self.phase != "DONE":
                self.phase = "DONE"
                self._plan(q, q + np.array([0, -0.15, 0, 0.15, 0, 0, 0]), 15)    # back off and up
            return self.path.pop(0) if self.path else q
        if self.path:
            return self.path.pop(0)
        if self.phase == "APPROACH":
            self._aim(info)
            self.phase = "APPROACH2"
            self._plan(q, self.q_rstart, int(self.rng.integers(36, 48)))
        elif self.phase in ("APPROACH2", "BACK"):
            self.phase = "LOWER"
            self._plan(q, self.q_start, int(self.rng.integers(6, 9)))
        elif self.phase == "LOWER":
            self.phase = "FLICK"
            self.flicks += 1
            self.flick_peak = 0.0
            self.path = [self.q_end.copy() for _ in range(int(self.rng.integers(17, 21)))]
        elif self.phase == "FLICK":
            if self.flick_peak < 5.0:
                self.misses += 1
                self.bias = max(self.bias - 0.25, -0.75)
            self.phase = "LIFT"
            self._plan(q, self.q_rend, 6)
        elif self.phase == "LIFT":
            self.phase = "BACK"
            self._aim(info)                 # re-aim at the nut's new position, new depth draw
            self._plan(q, self.q_rstart, int(self.rng.integers(12, 16)))
        return self.path.pop(0)
