"""One SO-101 (LeRobot's arm: five joints and a gripper on Feetech STS3215 servos, one USB serial bus) on dream's
control loop as a robot of your own (`body: custom`), driven from a Mac.

dream's loop always runs two arms (`follower_l`, `follower_r`). The real arm plays the side whose `can_channel` is its
serial port; the side whose `can_channel` is `parked` is an arm in memory at its home, gripper open, that never
touches hardware (the docs' "On another arm": the other arm parked at the model's rest pose).

Units are those of dream's `so101_faas` model (the config's `custom.ik`): six joints (the arm's five and the held
joint6), radians from the calibrated middle pose, each joint's zero moved to the reference arm's by dream's own rule
(`robots.so101_calibration`: the setup wizard's calibration writes the same shift into the servos; here it is applied
in software, so the LeRobot calibration and the datasets recorded with it stay as they are). The gripper is 0 closed
to 1 open across its calibrated range.

Both drivers are dream's own SO-101 driver (`robots.motor_arm` over `robots.feetech`), opened as dream opens one:
  ReadOnlyArm  the dry run: its read-only mode (the thread only reads; no motor switched on); refuses write().
  Arm          switches the motors on holding where the arm is. Yours to run, with a hand on the power switch.

close() and every disconnect path leave the arm where it is; only dream's driver's torque_off() parks it first.
"""
import dataclasses
import logging
import os
import pathlib
import time

import numpy as np
import numpy.typing as npt

from robots import config, feetech, followers, motor_arm, so101, so101_calibration

## The arm's LeRobot calibration file (`lerobot-calibrate`, arm id my_follower); SO101_CALIBRATION overrides it.
CALIBRATION = pathlib.Path(os.environ.get(
    "SO101_CALIBRATION",
    pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"))
## The `can_channel` of the side with no arm.
PARKED = "parked"
## The side the real arm plays (the config's follower_r).
REAL_SIDE = "right"
## The servos but the gripper, base to wrist.
NAMES = [j.name for j in so101.JOINTS][:-1]
## so101_faas has a sixth, held joint (joint6) after the arm's five; it always reads 0.
DOF = 6
## Targets stay this far inside each joint's real calibrated range: so101_faas's URDF lets the shoulder pan and the
## wrist flex 5-6 deg past this arm's stops, and dream's solve trusts the URDF (docs, "On another arm": Safety).
JOINT_MARGIN_DEG = 8.0
## Fastest a joint target may move, whatever the model asks: above the 90th percentile of what dream's IK asked on
## 2026-11-05 (75 deg/s), so the arm keeps up with the plan; a cap below it (60) held 11-17 % of rows back, the arm
## lagged, and the model, which anchors each chunk on where the hand is, drifted further each chunk.
MAX_JOINT_SPEED_DEG_S = 90.0
## The joint ranges this arm's own demonstrations used (flyingturtleboop/so101_medicine_bottle_pickplace, 7824
## frames, p0.5-p99.5, model units, degrees), widened by DEMO_MARGIN_DEG. dream's IK reached far or awkward hand
## targets by flipping to an elbow-backwards pose the demos never use (elbow down to -108 deg in 32 of 93 replies on
## 2026-11-05 00:45; the demos' elbow stays above -16): targets outside these ranges are held at their edge.
DEMO_RANGE_DEG = ((-105, 40), (-97, 50), (-16, 88), (-71, 86), (-19, 56))
DEMO_MARGIN_DEG = 10.0


def reference_shift_rad(calibration: dict[str, feetech.Calibration], side: str) -> np.ndarray:
    """Per joint, how far this arm's calibrated middle sits from the reference arm's (that side's), radians, by
    dream's rule (`so101_calibration.ArmCalibration`): the middle of the recorded range against the reference's;
    0 for the wrist roll, and for a joint whose range is longer than the reference's (not the same stops).
    Model units = arm units - shift."""
    deg_per_tick = 360.0 / (feetech.TICKS_PER_TURN - 1)
    shift = np.zeros(DOF)
    for name, ref in so101_calibration.reference(side).items():
        entry = calibration[name]
        span = (entry.range_max - entry.range_min) * deg_per_tick
        off = ((entry.range_min + entry.range_max) / 2 - ref.middle) * deg_per_tick
        if span > ref.span_deg + so101_calibration.LONG_RANGE_DEG:
            continue
        if abs(off) > so101_calibration.MAX_SHIFT_DEG:
            raise RuntimeError(f"{name}'s calibrated middle is {off:.0f} deg from the reference arm's: recalibrate")
        shift[NAMES.index(name)] = np.radians(off)
    return shift


class _Parked(followers.Driver):
    """The side with no arm: a fixed parked pose (its home) with grip 1, never moved. Its targets are dropped, so the
    model never sees it move (docs, "On another arm": the left as a fixed parked pose, never play q_l_chunk)."""

    def __init__(self, cfg: config.ArmConfig) -> None:
        self._q = np.array(cfg.home, dtype=np.float64)

    def read(self) -> followers.Reading:
        return followers.Reading(self._q.copy(), np.zeros_like(self._q), 1.0)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        pass

    def close(self) -> None:
        pass

    def torque_off(self) -> None:
        pass


def real_limits_rad(calibration: dict[str, feetech.Calibration], shift: np.ndarray,
                    home: tuple[float, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Where each joint's target may go, model units: the real calibrated range less JOINT_MARGIN_DEG, narrowed to the
    demonstrations' range (DEMO_RANGE_DEG +- DEMO_MARGIN_DEG), widened to take in the home pose; joint6 pinned at 0."""
    m = np.radians(JOINT_MARGIN_DEG)
    lo = np.array([np.radians(feetech.ticks_to_deg(calibration[n].range_min)) for n in NAMES] + [0.0]) - shift
    hi = np.array([np.radians(feetech.ticks_to_deg(calibration[n].range_max)) for n in NAMES] + [0.0]) - shift
    lo[:-1] += m
    hi[:-1] -= m
    demo = np.radians(np.array(DEMO_RANGE_DEG, dtype=float) + [-DEMO_MARGIN_DEG, DEMO_MARGIN_DEG])
    lo[:5], hi[:5] = np.maximum(lo[:5], demo[:, 0]), np.minimum(hi[:5], demo[:, 1])
    if len(home):
        lo, hi = np.minimum(lo, np.asarray(home, dtype=float)), np.maximum(hi, np.asarray(home, dtype=float))
    return lo, hi


def narrow_ik_limits() -> None:
    """Gives dream's IK this arm's usable joint ranges (real_limits_rad) in place of so101_faas's URDF limits.

    The docs ("On another arm", Safety): the solve trusts your URDF's limits; check them against the real arm. With
    the URDF's (elbow down to -110 deg) the solve reached far or awkward hand targets by flipping to an
    elbow-backwards pose (replaying the 2026-11-05 00:45 run's targets through dream's own IK: 35 of 92 chunks), and
    the arm lurched over the top. The client builds its solver in this process with dream_ik.solver.build_arm (one
    call site, functools.cache'd), so this wraps that function for the real arm's side before the solver exists;
    nothing in the dream library is edited. The solver still widens a range to take in where the arm is."""
    from dream_ik import solver
    if getattr(solver.build_arm, "so101_usable", False):
        return
    cal = feetech.read_calibration(str(CALIBRATION), so101.JOINTS)
    lo_spec, hi_spec = real_limits_rad(cal, reference_shift_rad(cal, REAL_SIDE))
    original = solver.build_arm
    if getattr(original, "cache_info", None) and original.cache_info().currsize:
        logging.getLogger(__name__).warning("dream's IK was built before my_robot.py loaded: its joint limits are the "
                                            "URDF's. Keep the wrist_l camera's source my_robot.py:Black, which "
                                            "loads this file before dreamcli makes its client")

    def build_arm(embodiment: str, side: str, tool: str):
        arm = original(embodiment, side, tool)
        if embodiment != "so101_faas" or side != REAL_SIDE:
            return arm
        order = list(arm.to_pyroki)                            # spec order -> the solver's (pyroki) order
        lo = np.maximum(np.asarray(arm.lo), lo_spec[order])
        hi = np.minimum(np.asarray(arm.hi), hi_spec[order])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)        # the held joint6 stays 0..0
        logging.getLogger(__name__).info("dream IK limits for %s %s narrowed to this arm's usable range (deg): %s",
                                         embodiment, side, [f"{a:.0f}..{b:.0f}" for a, b in
                                                            zip(np.degrees(lo), np.degrees(hi))])
        import jax.numpy as jnp
        return dataclasses.replace(arm, lo=jnp.asarray(lo), hi=jnp.asarray(hi))

    build_arm.so101_usable = True
    solver.build_arm = build_arm


def _dream_so101(cfg: config.ArmConfig, side: str, energize: bool) -> followers.Driver:
    """dream's own SO-101 driver (`robots.motor_arm.MotorArm`) on this arm, opened as dream opens one (`open_arm`:
    the servos' calibration checked against the file, a warm motor refused, the pose read before it returns).
    energize=False is its read-only mode: the thread only reads the pose and no motor is switched on."""
    open_deg, closed_deg = so101.gripper_ends(str(CALIBRATION))
    arm_cfg = config.ArmConfig(can_channel=cfg.can_channel, gripper_open_deg=open_deg, gripper_closed_deg=closed_deg,
                               calibration=str(CALIBRATION), home=(0.0,) * DOF)
    return motor_arm.open_arm(so101.ARM, arm_cfg, side, opener=feetech.open_bus, energize=energize)._driver


class _Shifted(followers.Driver):
    """dream's SO-101 driver with the joints moved to the reference arm's zeros on the way in and out."""

    def __init__(self, cfg: config.ArmConfig, side: str, live: bool) -> None:
        cal = feetech.read_calibration(str(CALIBRATION), so101.JOINTS)
        self._shift = reference_shift_rad(cal, side)
        self._lo, self._hi = real_limits_rad(cal, self._shift, cfg.home)
        self._last_target, self._last_t = None, 0.0
        self._live = live
        self._arm = _dream_so101(cfg, side, energize=live)
        self.drops_when_limp = getattr(self._arm, "drops_when_limp", True)

    def read(self) -> followers.Reading:
        r = self._arm.read()
        return followers.Reading(np.asarray(r.positions_rad, dtype=np.float64) - self._shift, r.velocities_rad_s,
                                 r.gripper, r.gripper_effort, r.motor_temps_c)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        if not self._live:
            raise RuntimeError("ReadOnlyArm is the dry run: it never writes a target")
        self._arm.write(self.guard(np.asarray(positions_rad, dtype=np.float64)) + self._shift, gripper)

    def guard(self, q: np.ndarray) -> np.ndarray:
        """The target kept inside the real range (widened to where the joint is now, so a joint resting near a limit
        is never jerked inward) and moved at most MAX_JOINT_SPEED_DEG_S from the last target. Model units."""
        now = np.asarray(self.read().positions_rad, dtype=np.float64)
        q = np.clip(q, np.minimum(self._lo, now), np.maximum(self._hi, now))
        t = time.monotonic()
        prev = now if self._last_target is None else self._last_target
        step = np.radians(MAX_JOINT_SPEED_DEG_S) * min(t - self._last_t, 0.1) if self._last_target is not None else 0.0
        if self._last_target is not None:
            q = prev + np.clip(q - prev, -step, step)
        self._last_target, self._last_t = q, t
        return q

    def close(self) -> None:
        self._arm.close()                                      # dream's driver: lets go without moving

    def torque_off(self) -> None:
        if self._live:
            self._arm.torque_off()                             # dream's driver parks where it opened, then off
        else:
            self._arm.close()                                  # never switched on: nothing to park

    def disengage(self) -> bool:
        return self._live and self._arm.disengage()

    def is_alive(self) -> bool:
        return self._arm.is_alive()

    def why_stopped(self) -> str:
        return self._arm.why_stopped()

    def alarm(self) -> str | None:
        return self._arm.alarm()


def ReadOnlyArm(cfg: config.ArmConfig, side: str) -> followers.Driver:  # noqa: N802 (the config names it)
    """The dry run's driver: the real arm read-only, the parked side in memory."""
    return _Parked(cfg) if cfg.can_channel == PARKED else _Shifted(cfg, side, live=False)


def Arm(cfg: config.ArmConfig, side: str) -> followers.Driver:  # noqa: N802
    """The run's driver: the real arm under dream's SO-101 driver (motors on), the parked side in memory."""
    return _Parked(cfg) if cfg.can_channel == PARKED else _Shifted(cfg, side, live=True)


class Black:
    """A camera source for the parked side's wrist: black frames, as the docs' "On another arm" sends."""

    def __init__(self, cfg=None) -> None:
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        return True, self._frame

    def release(self) -> None:
        pass


# dreamcli opens the cameras, then makes the dream client (which builds the IK), then opens the arms: the wrist_l camera
# (my_robot.py:Black) imports this file first, so this runs before the IK exists.
narrow_ik_limits()
