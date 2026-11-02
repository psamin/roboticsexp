"""Scripted flick expert (uses privileged nut state; outputs only finger joint targets).

Cycle: APPROACH (raised to the +y side, curled, slide over the nut) -> AIM (uncurl until the fingertip's lowest
point will pass through the hex-corner band) -> FLICK (one big MCP target jump; the torque/speed-limited drive makes
the swing) -> RELOAD (curl to lift the tip clear of the corners, swing back) -> repeat until the nut detaches.
Geometry: the knuckle is h_k above the bolt axis; at the bottom of the swing the tip passes r = h_k - R - r_tip
from the axis, where R is the knuckle->tip reach set by PIP/DIP curl. Corners stick out between r_flat and r_corner.
"""
from __future__ import annotations

import math

import numpy as np

import scene as S


def _solve_pip_for_reach(fs, R_target):
    """Curl (pip, dip = 0.7*pip) that gives knuckle->tip reach R_target. Returns (pip, dip, psi)."""
    lo, hi = 0.0, math.radians(95)
    for _ in range(40):
        mid = (lo + hi) / 2
        R, _ = S.finger_reach(fs, mid, 0.7 * mid)
        lo, hi = (mid, hi) if R > R_target else (lo, mid)
    pip = (lo + hi) / 2
    R, psi = S.finger_reach(fs, pip, 0.7 * pip)
    return pip, 0.7 * pip, psi


class FlickExpert:
    def __init__(self, env, rng: np.random.Generator, flick_sign=-1, theta0=math.radians(42), depth=(-0.15, 0.15)):
        """flick_sign=-1: the tip sweeps +y -> -y over the top of the nut (the unscrewing direction for a right-hand
        thread on a +x bolt). depth: where the tip's lowest point passes, as a fraction of the corner band
        (0 = flat radius, 1 = corner radius). A tip passing at radius r only catches a corner lying within
        acos(r / r_corner) of vertical (+-22 deg at 0.45 of the band), so with a flat on top a shallow sweep misses
        every time. Aiming at or just below the flat radius always catches a corner; after a miss, aim deeper."""
        self.env, self.fs, self.rng = env, env.fs, rng
        self.sign = flick_sign
        self.theta0 = theta0
        self.depth = depth
        self.bias = 0.0                 # extra depth after misses (band fractions, negative = deeper)
        self.flick_peak = 0.0
        self.misses = 0
        self.phase, self.n, self.flicks = "APPROACH", 0, 0
        self.path = None
        self._new_aim()

    def _new_aim(self):
        env = self.env
        r_flat, r_corner = env.r_band
        d = self.rng.uniform(*self.depth) + self.bias
        r_t = r_flat + d * (r_corner - r_flat)
        self.aim_pip, self.aim_dip, psi = _solve_pip_for_reach(self.fs, env.h_k - self.fs.r3 - r_t)
        # MCP angle where the knuckle->tip vector points straight down is -psi*sign_of_curl; pip curls toward +y
        self.mcp_bottom = -psi
        self.mcp_start = self.mcp_bottom - self.sign * self.theta0
        self.mcp_end = self.mcp_bottom + self.sign * self.theta0
        self.reload_pip = min(self.aim_pip + math.radians(35), math.radians(95))
        self.reload_dip = 0.7 * self.reload_pip

    def _plan(self, q_from, q_to, steps):
        s = np.linspace(0, 1, steps + 1)[1:]
        s = s * s * (3 - 2 * s)                      # smoothstep
        self.path = [np.asarray(q_from) + (np.asarray(q_to) - np.asarray(q_from)) * v for v in s]

    def act(self, state, info):
        x = info["nut_cx"]
        q = np.asarray(state, dtype=np.float64)
        if info["detached"]:
            if self.phase != "DONE":
                self.phase = "DONE"
                self._plan(q, [x, self.mcp_start, self.reload_pip, self.reload_dip], 10)
            return self.path.pop(0) if self.path else np.array([x, self.mcp_start, self.reload_pip, self.reload_dip])
        if self.phase == "FLICK":
            self.flick_peak = max(self.flick_peak, abs(info["w"]))
        if self.path:
            target = self.path.pop(0)
            target[0] = x if self.phase in ("AIM", "RELOAD_BACK") else target[0]
            return target
        if self.phase == "APPROACH":         # go high and to the start side, slide over the nut
            self.phase = "APPROACH2"
            self._plan(q, [x, self.mcp_start, self.reload_pip, self.reload_dip], int(self.rng.integers(10, 16)))
        elif self.phase in ("APPROACH2", "RELOAD_BACK"):
            self.phase = "AIM"
            self._plan(q, [x, self.mcp_start, self.aim_pip, self.aim_dip], int(self.rng.integers(5, 8)))
        elif self.phase == "AIM":
            self.phase = "FLICK"
            self.flicks += 1
            self.flick_peak = 0.0
            hold = int(self.rng.integers(4, 6))
            self.path = [np.array([x, self.mcp_end, self.aim_pip, self.aim_dip])] * hold
        elif self.phase == "FLICK":           # lift the tip clear of the corners, then swing back
            if self.flick_peak < 5.0:         # the sweep missed every corner: go deeper next time
                self.misses += 1
                self.bias = max(self.bias - 0.25, -0.75)
            self.phase = "RELOAD_UP"
            self._plan(q, [x, self.mcp_end, self.reload_pip, self.reload_dip], 4)
        elif self.phase == "RELOAD_UP":
            self.phase = "RELOAD_BACK"
            self._new_aim()
            self._plan(q, [x, self.mcp_start, self.reload_pip, self.reload_dip], int(self.rng.integers(7, 10)))
        return self.path.pop(0)
