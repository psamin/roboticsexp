"""Feetech STS3215 position servo, after Rhoban BAM's identified model of this servo (github.com/Rhoban/bam).

BAM measured the firmware with an oscilloscope and a pendulum test bench:
  - the firmware moves an internal target toward the goal at most `max_velocity` (~5.1-5.3 rad/s);
  - duty = clip((target_internal - q) * P * 0.166 * error_gain_ratio, +-0.97);
  - the DC motor gives tau = kt / R * (vin * duty - kt * dq);
  - friction is Coulomb + viscous; the reflected rotor inertia (armature, ~0.022-0.026 kg m^2) dominates.
Additions here, all randomised because nobody has measured them on an SO-101: an acceleration cap on the internal
target (LeRobot writes Acceleration=254, whose meaning is unconfirmed), a derivative term (LeRobot uses D=32, BAM
identified with D=0), and the supply voltage (the stock kit runs 7.4 V servos from 5 V).
BAM 7.4 V model m1: kt 1.1776, R 2.4787, error_gain_ratio 1.005, armature 0.02608, max_velocity 5.288,
friction_base 0.0515, friction_viscous 0.0599, command_delay 1.05 ms.
"""
import torch

ERROR_GAIN = 0.166
MAX_PWM = 0.97

# name: (nominal, low, high). Values are sampled as nominal + difficulty * (U(low, high) - nominal).
SERVO_RANGES = {
    "kt": (1.1776, 1.0, 1.35),
    "R": (2.4787, 2.1, 2.9),
    "error_gain_ratio": (1.005, 0.85, 1.15),
    "armature": (0.0261, 0.019, 0.032),
    "fric_c": (0.0515, 0.03, 0.08),
    "fric_v": (0.0599, 0.025, 0.09),
    "max_velocity": (5.288, 4.6, 5.6),
    "log_max_accel": (3.5, 2.2, 3.9),        # log10 rad/s^2: 160 (2.2) to 8000 (3.9); 38.6 would be "Acceleration=254"
    "kd_time": (0.004, 0.0, 0.02),           # s, derivative action relative to the P term
    "vin": (5.0, 4.7, 5.4),                  # V at the servo: the stock kit's 5 V brick (retrain if you change it)
    "deadband": (0.0015, 0.0, 0.005),        # rad, position errors the firmware ignores (dead zone, 0-3 steps)
    "supply_r": (0.0, 0.0, 0.3),             # ohm, supply + wiring resistance shared by all servos (voltage sag)
    "strength": (0.8, 0.55, 1.05),           # torque scale: BAM's fit is ~1.8x the datasheet stall torque
    "backlash": (0.007, 0.0035, 0.0175),     # rad, total gear play (0.2-1.0 deg)
    "gear_k": (150.0, 80.0, 300.0),          # N m / rad, gear train + horn stiffness once the play is taken up
                                             # (~0.4-1.4 deg of wind-up at stall; unmeasured)
    "gear_c": (0.2, 0.08, 0.4),              # N m s / rad, damping in the engaged gears
    "load_fric": (0.08, 0.0, 0.25),          # Coulomb friction per N m transmitted (BAM m4/m6: ~0.2)
}

# The user's arm: 12 V STS3215s (Present_Voltage 11.5-11.8 V). Fitted 2026-10-31 by scripts/fit_servo.py to 2 s of the
# real arm tracking policy goals at P16 (rms 4.98 deg with the 5 V set -> 1.45 deg). kt is pinned to the 12 V
# datasheet no-load speed (0.222 s/60 deg); strength is 1 (R absorbs it). Stall ~6.3 N m (5 V set: ~1.8 N m).
# 2 s of data cannot separate rotor inertia from command delay (armature 0.026 + 35 ms delay fits almost as well as
# 0.068 + 30 ms), so those ranges span both; the delay itself is EnvCfg.act_delay_ms (variants v12*).
SERVO_RANGES_12V = {
    **SERVO_RANGES,
    "kt": (2.41, 2.1, 3.0),
    "R": (4.35, 3.6, 5.4),
    "armature": (0.065, 0.024, 0.08),
    "vin": (11.7, 11.0, 12.4),
    "strength": (1.0, 0.85, 1.15),
}
# Refit (round 8) on ~4 min of real-arm logs at P16 and P32 (scripts/fit_servo.py, job 13215): rms 0.81 deg (12 V set:
# 0.94); the firmware speed limit comes out near 2.3 rad/s, not 5.3, which is why real launch strokes were weaker
# than the 12 V model predicted. Ranges cover the best 40 candidates' spread.
SERVO_RANGES_12V_R8 = {
    **SERVO_RANGES_12V,
    "kt": (2.34, 2.0, 3.0),
    "R": (3.54, 2.7, 5.0),
    "armature": (0.046, 0.025, 0.09),
    "max_velocity": (2.3, 1.8, 4.0),
    "strength": (1.0, 0.8, 1.15),
}
# Round 10: on the real ball runs (2026-11-01, P32) the launch moved the paddle up at 0.45 m/s where 12v_r8 predicts
# 1.03: the loaded joints (lift, elbow) cannot build speed within a 60-100 ms stroke while the free wrist still reaches
# 2.6 rad/s in play. A per-joint torque cap reproduces all of it (launch 0.46 m/s, elbow 0.57 vs 0.52 rad/s, wrist
# 1.24 vs 1.24 in launches); a lower speed cap or gear play do not. "i_max" is that cap in model amps (torque
# kt * i_max ~ 1.75 N m against 0.93 N m of gravity at the elbow); the top of its range reaches the 12v_r8 behaviour.
SERVO_RANGES_12V_R10 = {
    **SERVO_RANGES_12V_R8,
    "max_velocity": (4.0, 2.3, 5.0),
    "i_max": (0.75, 0.55, 1.4),
    "supply_r": (0.1, 0.0, 0.4),
}
# Wide: everything from the round-8 behaviour (no torque cap, 5 rad/s) down to the round-10 cap, for robustness.
SERVO_RANGES_12V_R10W = {**SERVO_RANGES_12V_R10, "max_velocity": (4.0, 2.0, 5.3), "i_max": (0.75, 0.5, 3.0)}
# Round 15: fitted (scripts/fit_servo.py, jobs 13465-13467) to the 2026-11-02 speed test (real/speed_test.py: launch
# strokes and elbow steps at P32) plus the 2026-11-01 real-ball logs, with i_max free: rms 0.55 deg (12v_r10 nominal
# 0.75; elbow on fast strokes 1.03 vs 2.03). The torque cap is ~3.3 N m (kt * i_max), not 1.75, with more rotor
# inertia: the round-10 launch stroke lifts the paddle 7.7 cm in the model (real 10.5 cm, 12v_r10 2.7 cm). The command
# delay fits at 14-35 ms, inside act_delay_ms; the ~100 ms goal-to-joint lag is inertia and the current limit.
# Ranges span the three fits (all logs / speed test only / real-ball logs only).
SERVO_RANGES_12V_R11 = {
    **SERVO_RANGES_12V_R10,
    "kt": (3.26, 2.6, 3.6),
    "R": (3.93, 2.3, 4.6),
    "armature": (0.100, 0.07, 0.13),
    "max_velocity": (2.6, 2.2, 3.8),
    "i_max": (1.0, 0.8, 1.6),
}
# Wide: the r11 nominal, with ranges reaching down to the round-10 torque cap and up to the speed-test-only fit.
SERVO_RANGES_12V_R11W = {**SERVO_RANGES_12V_R11, "kt": (3.26, 2.0, 3.6), "R": (3.93, 2.3, 5.0),
                         "armature": (0.100, 0.04, 0.13), "max_velocity": (2.6, 2.2, 4.5), "i_max": (1.0, 0.65, 1.6)}
SERVO_VARIANTS = {"5v": SERVO_RANGES, "12v": SERVO_RANGES_12V, "12v_r8": SERVO_RANGES_12V_R8,
                  "12v_r10": SERVO_RANGES_12V_R10, "12v_r10w": SERVO_RANGES_12V_R10W, "12v_r11": SERVO_RANGES_12V_R11,
                  "12v_r11w": SERVO_RANGES_12V_R11W}


class ServoBank:
    """N environments x J joints of STS3215 servos, all tensors (N, J)."""

    def __init__(self, num_envs: int, num_joints: int, p_gain: float, device, ranges=SERVO_RANGES):
        self.N, self.J, self.device = num_envs, num_joints, device
        self.p_gain, self.ranges = p_gain, ranges
        self.p = {k: torch.full((num_envs, num_joints), v[0], device=device) for k, v in ranges.items()}
        self.target = torch.zeros(num_envs, num_joints, device=device)       # firmware's internal target
        self.target_vel = torch.zeros(num_envs, num_joints, device=device)
        self.current = torch.zeros(num_envs, num_joints, device=device)        # A, last step, for supply sag

    def sample(self, ids, difficulty: torch.Tensor, gen: torch.Generator):
        """Draw new parameters for environments `ids`; difficulty (len(ids),) in [0, 1] scales the spread."""
        n = ids.numel()
        d = difficulty[:, None]
        for k, (nom, lo, hi) in self.ranges.items():
            if k in ("vin", "supply_r"):       # one supply per arm
                u = torch.rand(n, 1, generator=gen, device=self.device).expand(n, self.J)
            else:
                u = torch.rand(n, self.J, generator=gen, device=self.device)
            self.p[k][ids] = nom + d * (lo + (hi - lo) * u - nom)

    def stiffness(self):
        """Linear-region stiffness d(tau)/d(error) in N m / rad, used to find the sag-compensated hold command."""
        p = self.p
        return self.p_gain * ERROR_GAIN * p["error_gain_ratio"] * p["vin"] * p["strength"] * p["kt"] / p["R"]

    def reset(self, ids, q):
        """q: (len(ids), J) positions the firmware's internal target starts from."""
        self.target[ids] = q
        self.current[ids] = 0.0
        self.target_vel[ids] = 0.0

    def torque(self, goal, q, dq, dt, dq_motor=None):
        """Motor torque (N, J) and duty cycle for goal positions `goal`. The firmware's P (and D) act on the output
        shaft encoder (q, dq); back-EMF follows the motor side (dq_motor, on the far side of the gear play)."""
        p = self.p
        # internal target: velocity-limited, acceleration-limited, braking in time to stop at the goal
        err = goal - self.target
        amax = torch.pow(10.0, p["log_max_accel"])
        v_stop = torch.sqrt(2 * amax * err.abs())
        v_des = torch.sign(err) * torch.minimum(torch.minimum(err.abs() / dt, v_stop), p["max_velocity"])
        dv = (v_des - self.target_vel).clamp(-amax * dt, amax * dt)
        self.target_vel = self.target_vel + dv
        step = self.target_vel * dt
        # never step past the goal
        step = torch.where((step * err > 0) & (step.abs() > err.abs()), err, step)
        self.target = self.target + step
        k = self.p_gain * ERROR_GAIN * p["error_gain_ratio"]
        e = self.target - q
        e = torch.sign(e) * (e.abs() - p["deadband"]).clamp(min=0)
        duty = (e * k - k * p["kd_time"] * dq).clamp(-MAX_PWM, MAX_PWM)
        vin = (p["vin"] - p["supply_r"] * self.current.abs().sum(-1, keepdim=True)).clamp(min=0.5 * p["vin"])
        self.current = (vin * duty - p["kt"] * (dq if dq_motor is None else dq_motor)) / p["R"]
        if "i_max" in p:                    # firmware current limit (Protection_Current), sets only when fitted
            self.current = self.current.clamp(-p["i_max"], p["i_max"])
        tau = p["strength"] * p["kt"] * self.current
        return tau, duty

    def friction(self, dq, tau_transmitted=None):
        """Motor-side friction: Coulomb (plus a part that grows with the torque the gears carry) and viscous."""
        coulomb = self.p["fric_c"] if tau_transmitted is None else \
            self.p["fric_c"] + self.p["load_fric"] * tau_transmitted.abs()
        return -coulomb * torch.tanh(dq / 0.02) - self.p["fric_v"] * dq

    def rated_stall(self):
        """Datasheet-style stall torque at the supply voltage (1.91 N m at 7.4 V), for the overload model."""
        return 0.258 * self.p["vin"]
