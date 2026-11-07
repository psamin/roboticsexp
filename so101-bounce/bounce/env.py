"""SO-101 ping-pong bouncing: a vectorised PyTorch environment (thousands of arms on one GPU).

One control step (50 Hz) = `substeps` physics steps (500 Hz). Each physics step:
  delayed joint goal -> STS3215 servo model -> joint dynamics with gravity and inertia from the MuJoCo model ->
  forward kinematics -> ball flight -> ball/paddle impact -> camera capture (rate, latency, noise, bias, dropout).

The policy sees what the real system will see: measured joint angles (with calibration error), its own recent
actions, the paddle pose from forward kinematics of the measured angles, and the last few camera detections of the
ball with their ages. The critic also gets the true ball state and the episode's physical parameters.

Actions are joint goals relative to the episode's hold command q_ref (the command that holds the paddle level
against gravity sag, which the real supervisor finds the same way): goal = q_ref + action_scale * a, a in [-1, 1].
"""
from dataclasses import dataclass, field
import math

import torch

from . import ball as B
from .robot import ArmKinematics, PaddleSpec, axis_angle, build_model
from .servo import SERVO_VARIANTS, ServoBank

J = 5
CAM_RING = 32           # camera captures kept (detections in flight + the ones in the observation)
CMD_RING = 24
FAIL_REASONS = ("dropped", "out_of_view", "paddle_limits", "overload", "resting", "none")


@dataclass
class EnvCfg:
    num_envs: int = 4096
    device: str = "cuda"
    seed: int = 0
    ctrl_hz: float = 50.0
    substeps: int = 10
    history: int = 4                            # control steps of joint angles and actions in the observation
    cam_k: int = 4                              # most recent ball detections in the observation
    episode_s: float = 8.0
    p_gain: float = 16.0                        # STS3215 P register; LeRobot writes 16 (factory default 32)
    servo_variant: str = "5v"                   # servo parameter set in bounce/servo.py: "5v" or "12v" (the user's arm)
    action_scale: tuple = (0.20, 0.30, 0.30, 0.35, 0.30)   # rad: pan, lift, elbow, wrist flex, wrist roll
    nominal_lift: float = -1.35                 # hold pose found by a gravity-load search (forearm up, paddle level)
    nominal_elbow: float = 0.60
    paddle: PaddleSpec = field(default_factory=PaddleSpec)
    # (nominal, low, high), sampled per episode as nominal + difficulty * (U(low, high) - nominal)
    act_delay_ms: tuple = (8.0, 4.0, 40.0)
    # cameras: overhead C920 alone (30 fps, height from apparent size: large error along its line of sight), or with
    # a fast side camera (60-120 fps, triangulated: small error in every direction); the policy is told which
    side_cam_prob: float = 0.5
    overhead_cam: tuple = (-0.17, 0.0, 0.94)    # m, C920 behind the base (the Dream-1 layout)
    side_cam: tuple = (0.35, -0.75, 0.30)       # m, beside the paddle at about its height
    overhead_fps: tuple = (30.0, 25.0, 30.0)      # dim rooms may also halve it to 15 (auto exposure)
    side_fps: tuple = (120.0, 60.0, 120.0)
    cam_frames: tuple = (1.5, 1.0, 2.0)         # latency in frame periods (exposure, readout, USB transfer)
    cam_base_ms: tuple = (20.0, 8.0, 50.0)      # + decode, detection, threading
    cam_noise_px: tuple = (0.0015, 0.0005, 0.004)    # m, error across each camera's line of sight
    overhead_depth: tuple = (0.035, 0.02, 0.07)      # m, overhead-only error along its line of sight (ball is
                                                     # 5-7 px in radius at 320x240: 0.25-0.5 px of size error)
    tri_noise: tuple = (0.002, 0.0005, 0.005)        # m, two-camera error along each line of sight
    depth_scale: float = 0.03                   # overhead-only depth scale error (ball size, intrinsics)
    side_scale: float = 0.01
    light: tuple = (0.6, 0.0, 1.0)              # 0 dim (exposure = frame period, blur, noise) .. 1 bright (short)
    blur_k: tuple = (0.0, -0.5, 0.5)            # detected centre shifted along the velocity, x exposure
    cam_dropout: tuple = (0.01, 0.0, 0.10)
    burst_in: tuple = (0.0, 0.0, 0.03)          # per frame: start a run of missed detections
    burst_out: tuple = (0.4, 0.6, 0.2)          # per frame: end it (lower = longer bursts)
    occlusion_p: tuple = (0.0, 0.0, 0.5)        # missed when the ball is within 1.5 cm of the paddle face
    blur_miss_p: tuple = (0.0, 0.0, 0.3)        # missed when the motion streak exceeds 3 cm
    age_bias: float = 0.025                     # s, per-session error in the estimated capture time
    hand_outlier_p: float = 0.15                # extra false detections per frame while a hand is in view
    paddle_mass: tuple = (0.16, 0.08, 0.22)    # kg, light fun paddles to heavy pro ones
    paddle_reach: tuple = (0.13, 0.06, 0.16)   # m, blade centre beyond the fingertips: where the jaws hold the handle
    blade_a: tuple = (0.0785, 0.070, 0.087)    # m, blade half-length (along the handle)
    blade_b: tuple = (0.075, 0.068, 0.082)     # m, blade half-width
    blade_shape: tuple = (2.2, 2.0, 2.8)       # superellipse exponent: 2 = oval, higher = squarer
    face_lift: tuple = (0.0, -0.005, 0.008)    # m, hitting surface above the jaw centre (rubber + blade thickness)
    mount_roll: float = 0.08                   # rad, handle rolled in the jaws (levelled out by the hold pose)
    mount_pitch: float = 0.05                  # rad, handle tilted up/down in the jaws (levelled out too)
    mount_yaw: float = 0.10                    # rad, handle turned sideways in the jaws (not levelled out)
    mount_side: float = 0.01                   # m, handle off-centre between the jaws
    paddle_com_shift: float = 0.02             # m, centre of mass along the handle
    # room, paddle wobble, bus and camera imperfections
    table_tilt: float = 0.026                  # rad, table not level: gravity tilted relative to the robot base
    wind: float = 0.3                          # m/s, horizontal air current (fan, AC vent)
    paddle_jitter: tuple = (0.0, 0.0, 0.002)   # rad, std of paddle-face vibration (blade flex, grip play)
    jitter_hz: tuple = (20.0, 8.0, 40.0)       # its natural frequency
    jitter_zeta: tuple = (0.15, 0.05, 0.3)     # and damping ratio
    glitch_p: tuple = (0.0, 0.0, 0.03)         # per control step: a USB hiccup freezes commands and readings 1-4 steps
    ctrl_jitter_ms: float = 4.0                # extra random command delay each step at full difficulty
    cam_rot_err: float = 0.026                 # rad, camera calibration rotation error (position-dependent bias)
    outlier_p: tuple = (0.0, 0.0, 0.02)        # per frame: a wild detection 5-20 cm off (hand, glare)
    stamp_noise: tuple = (0.0, 0.0, 0.003)     # s, frame timestamp error
    rest_spread: float = 0.75                  # resting starts out to this fraction of the blade (full difficulty)
    flight_spread: float = 0.07                # m, dropped/tossed starts off centre (full difficulty)
    drop_blade_p: float = 0.0                  # share of drops over anywhere on the blade (uniform over its area out
    drop_blade_r: float = 1.0                  # to this fraction of the rim), whatever the task level
    mass_scale: tuple = (1.0, 0.9, 1.1)
    cam_bias: float = 0.008                     # m per axis at full difficulty
    joint_offset: float = 0.04                  # rad, calibration error at full difficulty
    joint_noise: float = 0.0015                 # rad
    level_error: float = 0.015                  # rad, how level the hold pose really is
    hold_jitter: float = 0.15                   # rad on shoulder lift and elbow: the real hold pose will differ
    # episode starts and disturbances
    rest_start_prob: float = 0.35
    push_rate_hz: float = 0.4                   # hand taps per second at full difficulty
    push_std: tuple = (0.15, 0.15, 0.45)        # m/s
    target_apex: tuple = (0.07, 0.15)           # m, apex above the point where the ball last left the paddle
    # more start types and wider ranges (defaults: as in rounds 1-6)
    flight_h: tuple = (0.03, 0.18)              # m, dropped/tossed starts: ball bottom above the paddle
    held_start_prob: float = 0.0                # held still above the paddle for 0..held_max_s, then let go
    held_max_s: float = 1.0
    held_hidden_prob: float = 0.5               # the hand hides the held ball from the camera
    held_outlier_p: float = 0.3                 # per frame while held and visible: the hand detected 3-8 cm away
    beside_start_prob: float = 0.0              # dropped or tossed beside the paddle: the arm has to move to it
    beside_offset: tuple = (0.08, 0.15)         # m from the paddle centre
    beside_vlat: float = 0.3                    # m/s, sideways speed up to this (random direction)
    ceiling_after_land: bool = False            # out-of-view ceiling applies once the ball has touched the paddle
    overhead_cam_box: tuple = ()                # ((x0, y0, z0), (x1, y1, z1)): overhead camera anywhere in it
    cam_lat_max: float = 0.12                   # s, camera latency cap
    overhead_cam_box_p: float = 1.0             # with overhead_cam_box: share of episodes placed in the box (the
                                                # rest use overhead_cam +- 10 cm)
    # start-up variation (round 8): the arm and ball as the real runner finds them when the policy switches on
    start_pose_p: float = 0.0                   # arm displaced from the hold command and still settling
    start_pose_dq: tuple = (0.086, 0.06, 0.06, 0.052, 0.052)   # rad per joint: paddle ~+-3 cm, +-3 deg tilt
    start_qd: float = 0.5                       # rad/s, joint velocity left over from the ramp back
    switch_delay_p: float = 0.0                 # the policy's first action comes up to switch_delay_s after the
    switch_delay_s: float = 0.25                # ball starts; the arm holds still meanwhile
    midbounce_p: float = 0.0                    # share of plain flight starts begun mid-bounce
    midbounce_apex: tuple = (0.05, 0.15)        # m, apex of that bounce above the paddle (random phase)
    hist_noise_p: float = 0.0                   # non-zero action / joint history at the start (as after a re-arm)
    rest_roll_p: float = 0.0                    # resting starts already rolling (paddle off level, a nudge)
    rest_roll_v: float = 0.3                    # m/s, up to, any direction along the face
    toss_p: float = 0.0                         # share of plain flight starts tossed in: sideways and spinning, or
    toss_vlat: float = 0.5                      # thrown down; aimed to land within 4 cm of the centre (m/s)
    toss_vz: tuple = (-1.5, 0.5)                # m/s, vertical speed at release
    toss_spin: float = 30.0                     # rad/s, up to, random axis
    sidein_p: float = 0.0                       # share bounced in from beside the paddle at about its height (or just
    sidein_offset: tuple = (0.10, 0.16)         # below its edge), up over the edge and down onto the blade
    sidein_h: tuple = (-0.03, 0.03)             # m, start height of the ball's bottom relative to the face
    sidein_apex: tuple = (0.05, 0.20)           # m, apex above the face
    rubber_fullrange: bool = False              # rubber restitution and grip from the type's full range at any
                                                # difficulty (the user's paddle is a hard, dead one)
    # limits
    ceiling: float = 0.30                       # ball above home + ceiling leaves the camera view
    max_lateral: float = 0.18
    paddle_box: tuple = (0.12, 0.06)            # paddle centre within +-xy, +-z of home
    max_tilt: float = 0.6
    max_rest_s: float = 2.0
    overload_frac: float = 0.8
    overload_s: tuple = (2.0, 1.2, 2.5)
    # reward
    w_alive: float = 1.0
    w_apex: float = 1.0
    apex_sigma: float = 0.03
    w_hit: float = 0.3
    w_lateral: float = 1.0
    w_rate: float = 0.02
    w_qd: float = 0.01
    w_load: float = 2.0
    w_tilt: float = 0.2
    w_home: float = 0.5                         # keep the paddle near its home height (no dribbling from up high)
    w_fail: float = 3.0
    w_first: float = 0.0                        # bonus for the first bounce within 3 cm of the target height
    w_wait: float = 0.0                         # per second until that first on-target bounce (launch quickly)
    w_back: float = 0.0                         # per m of first-apex drift toward -x (the paddle handle and the base)
    w_aim: float = 0.0                          # per hit: ball leaves at the velocity that reaches the target apex
    aim_sigma: float = 0.2                      # (vz = sqrt(2 g h)) and lands back on the centre; m/s
    w_track: float = 0.0                        # per s in flight: paddle centre under the ball's predicted landing point
    w_edge: float = 0.0                         # per hit: (contact offset / blade semi-axis)^2, 0 at the centre, 1 at the rim
    handle_keepout: float = 0.0                 # m: a ball off the paddle this far behind its centre (-x) within 4 cm
                                                # of paddle height is lost (it comes down on the handle); 0 = off
    # round 9
    launch_replay_p: float = 0.0                # resting starts that first run the runner's scripted launch
                                                # (bounce/launch.py, its exact actions), then hand over to the policy
    launch_jitter: float = 0.15                 # +- scale of the replayed launch; 30 % also one strike step more/less
    view_top: tuple = ()                        # (lo, hi) m above home: the real camera's top frame edge (the ball
    view_top_p: float = 0.0                     # leaves the C920's view ~12 cm up): no detections above it
    cam_view: tuple = ()                        # the real overhead view (R_base_from_cam 9 values, t 3, focal px, width,
    cam_view_p: float = 0.0                     # height; real/camera_calib.json): no detection outside the frame or the
    cam_view_half: float = 0.14                 # runner's search window (rig.paddle_roi: home +-half sideways and
    cam_view_back: float = 0.14                 # forward, -back toward the arm, z range above), drawn from the
    cam_view_z: tuple = (-0.05, 0.30)           # calibration while the true camera is moved by
    cam_view_jit: tuple = (0.02, 0.035)         # up to (m, rad) per episode (the mount moved ~2 cm / 2 deg in a day)
    cam_height_gain: tuple = ()                 # (lo, hi): measured height above home scaled by a per-episode gain
                                                # while the ball flies (real C920: ~0.79, radius reads small in flight)
    beside_feasible: bool = False               # beside starts: offset grows with task difficulty, dropped from high
                                                # enough for the arm to get there (fall time vs pan reach)
    toss_curriculum: bool = False               # toss sideways speed and spin grow with task difficulty
    joint_offset_full_p: float = 0.0            # share of episodes with full-range joint calibration error
    # round 10 (real-ball runs 2026-11-01)
    launch_strike_range: tuple = ()             # (lo, hi): replayed launches use a strike of lo..hi steps, uniform
                                                # (the runner adapts its strike length from the first bounce)
    depth_speed_k: float = 0.0                  # m of extra overhead depth error per m/s of ball speed (motion blur
                                                # corrupts the ball's radius at ~8 ms exposure)
    radius_outlier_p: float = 0.0               # per in-flight detection: a 5-10 cm error along the line of sight
    rolling_shutter: float = 0.0                # s per m: capture-time error that grows with the ball's forward
                                                # offset (image rows), +- this per session
    w_tilt_contact: float = 0.0                 # penalty per hit for face tilt beyond tilt_contact_deg at contact
    tilt_contact_deg: float = 5.0
    # round 11: these scale with the task difficulty level, which stayed near 0.1, so the policy hardly trained them.
    # A share of episodes draws each over its full range instead (still uniform from zero: a curriculum by mixture).
    level_error_full_p: float = 0.0             # hold-pose level error
    push_full_p: float = 0.0                    # hand taps: rate uniform 0..push_rate_hz
    beside_full_p: float = 0.0                  # beside starts: offset and sideways speed (heights stay reachable)
    toss_full_p: float = 0.0                    # tosses: sideways speed and spin (with toss_curriculum)
    # round 12: a higher, steered launch and a high-then-low target (start bouncing high, then settle low)
    launch_max_action: float = 1.0              # replayed launch actions may reach this (bounce/launch.py Launch.gain)
    launch_steer: bool = False                  # replayed strikes tilt the face from the ball's offset and roll
    launch_name: str = ""                       # a named launch in bounce/launch.py NAMED (default: DEFAULTS[p_gain])
    target_sched_p: float = 0.0                 # episodes whose target starts high, then ramps to the steady target
    target_start: tuple = (0.15, 0.18)          # m, the high starting target
    target_high_s: tuple = (1.0, 2.0)           # s held at the high target from the start of the episode
    target_ramp_s: float = 1.0                  # s to ramp from the high to the steady target
    # round 13: the runner's --auto-level (real/bounce_run.py): an EMA of where on the blade the ball is, from the
    # camera's detections, slowly trims the hold command's wrist flex / roll against it (a tilted face sends the ball
    # downhill), and the policy's joint reference moves with it
    auto_level: bool = False
    al_alpha: float = 0.03                      # EMA weight per new detection within al_radius of the paddle centre
    al_radius: float = 0.08
    al_gain: float = 0.5                        # face-normal change per second per metre of drift
    al_period_s: float = 0.1
    al_deadband: float = 0.004                  # m of drift below which nothing is trimmed
    al_clip: float = 0.052                      # rad, at most this trim per wrist joint
    launch_mix: tuple = ()                      # several replayed launches: names in bounce/launch.py ("" = DEFAULTS)
    launch_mix_p: tuple = ()                    # ... and the share of replayed launches each gets
    launch_scale_range: tuple = ()              # (lo, hi): named launches replayed at this strength (a curriculum by
                                                # mixture from gentler to the full hard flick); "" keeps launch_jitter
    table_tilt_full_p: float = 0.0              # share of episodes with full-range table tilt (an invisible paddle tilt:
                                                # the model believes the face is level, gravity says otherwise)
    # rubber types (prob at full difficulty, restitution range, friction range): restitution and grip go together
    rubbers: tuple = ((0.55, (0.84, 0.92), (0.30, 0.60)),   # fresh inverted
                      (0.20, (0.78, 0.86), (0.12, 0.30)),   # worn / dusty inverted
                      (0.15, (0.80, 0.88), (0.15, 0.35)),   # short pimples
                      (0.10, (0.72, 0.82), (0.08, 0.20)))   # long pimples
    # hits that do not average out (2026-11-05 real arm: the ball walked off the tip and left centred hits sideways,
    # while every sim imperfection was zero-mean, so off-centre bounces drifted back to the middle)
    impact_tip: tuple = (0.0, 0.0)              # rad, per episode: the face tips forward at each hit (the taped paddle
                                                # pivots in the jaws): a push along the handle of tip * (1 + e0) * |v_n|
    kick_range: tuple = ()                      # (lo, hi) m/s, per episode: std of the per-hit random push (rubber
                                                # texture); () keeps BALL_RANGES["kick"] scaled by the physics level
    strike_push: tuple = (0.0, 0.0)             # (lo, hi) per episode: push toward the tip = k * the paddle's upward
                                                # speed at the hit (measured k 0.45-0.8 on rally hits, ~0 on a still
                                                # paddle's first catch; impact_tip's |v_n| scaling had it backwards)
    tilt_authority: tuple = (1.0, 1.0)          # (lo, hi) per episode: share of the face's tilt away from the hold pose
                                                # that reaches the ball at a hit (the re-taped paddle barely steered it)
    adr_floor: tuple = ()                       # per group (actuation, sensing, physics, task): the curriculum's lowest
                                                # level (the task level sat at 0.06, so off-centre states were rare)
    critic_rig: bool = False                    # the critic also sees each episode's strike push, tilt authority,
                                                # texture push and impact tip (hidden from the actor, as on the arm)
    dive_p: float = 0.0                         # chance per hit of a big horizontal push in a random direction
    dive_kick: tuple = (0.15, 0.6)              # m/s: lands the ball ~5-19 cm away at a 12 cm apex (0.31 s flight)
    dive_fwd: float = 1.0                       # scale on a big push's forward/back part (forward reach is capped by
                                                # geometry and servo speed at ~5-7 cm per flight; sideways is not)
    # curriculum: one randomisation level per group (actuation, sensing, physics, task); each new episode draws
    # difficulty = level * U(0,1)^0.5 per group; a group's level rises while its hardest episodes mostly survive
    difficulty: float = 0.0
    adr_step: float = 0.02
    adr_up: float = 0.6
    adr_down: float = 0.25


def _lerp(r, d, u):
    nom, lo, hi = r
    return nom + d * (lo + (hi - lo) * u - nom)


def _lerp_log(r, d, u):
    """Like _lerp but log-uniform for ranges spanning a large ratio; a zero lower bound gives exact zeros 20% of the
    time and log-uniform values from hi/30 up to hi otherwise."""
    nom, lo, hi = r
    if lo > 0:
        x = lo * (hi / lo) ** u
    else:
        x = torch.where(u < 0.2, torch.zeros_like(u), (hi / 30) * 30 ** ((u - 0.2) / 0.8))
    return nom + d * (x - nom)


class BounceEnv:
    def __init__(self, cfg: EnvCfg):
        self.cfg = c = cfg
        self.N, self.device = c.num_envs, torch.device(c.device)
        dev, N = self.device, c.num_envs
        self.gen = torch.Generator(device=dev).manual_seed(c.seed)
        self.dt = 1.0 / c.ctrl_hz
        self.sub_dt = self.dt / c.substeps
        self.max_steps = int(round(c.episode_s * c.ctrl_hz))
        self.model = build_model(c.paddle)
        self.kin = ArmKinematics(self.model, N, dev)
        self.kin_nominal = ArmKinematics(self.model, N, dev)      # what the real controller's FK knows
        self.servo = ServoBank(N, J, c.p_gain, dev, ranges=SERVO_VARIANTS[c.servo_variant])
        self.scale = torch.tensor(c.action_scale, device=dev)
        self.g = torch.tensor([0.0, 0.0, -9.81], device=dev)
        self.push_std = torch.tensor(c.push_std, device=dev)
        self.g_env = self.g.expand(N, 3).clone()          # per-episode gravity in the robot frame (table tilt)
        self.levels = [max(float(c.difficulty), c.adr_floor[g] if c.adr_floor else 0.0) for g in range(4)]   # actuation, sensing, physics, task
        self.paddle_rot0 = self.kin.paddle_rot[0].clone()
        self.paddle_pos0 = self.kin.paddle_pos[0].clone()
        self.ipos_paddle0 = self.kin.ipos[0, self.kin.paddle].clone()
        self.mass0 = self.kin.mass[0].clone()
        # nominal level pose and the home point (paddle surface centre there)
        q0 = torch.tensor([[0.0, c.nominal_lift, c.nominal_elbow, 0.75, 0.0]], device=dev)
        self.q_nom = self._solve_level(q0.expand(1, J).clone(), torch.zeros(1, dtype=torch.long, device=dev))[0]
        fk = self.kin.forward(self.q_nom[None].expand(N, J))
        self.home = fk["paddle_o"][0].clone()
        self.home_env = self.home.expand(N, 3).clone()     # per episode: paddle centre at the hold pose, as measured
        z = lambda *s: torch.zeros(*s, device=dev)
        # state
        self.q, self.qd = self.q_nom.expand(N, J).clone(), z(N, J)
        self.qm, self.qdm = self.q.clone(), z(N, J)          # motor side of the gear play
        self.q_ref = self.q.clone()
        self.cmd_ring = self.q.expand(CMD_RING, N, J).clone()
        self.ring_i = 0
        self.p_ball, self.v_ball, self.w_ball = z(N, 3), z(N, 3), z(N, 3)
        self.o_prev, self.R_prev = fk["paddle_o"].clone(), fk["paddle_R"].clone()
        self.t = z(N)
        self.steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.rest_time, self.since_impact, self.overload_time = z(N), z(N), z(N)
        self.prev_vz = z(N)
        self.launch_z = z(N)                     # height of the ball's bottom when it last left the paddle
        self.cam_pos, self.cam_t = z(N, CAM_RING, 3), torch.full((N, CAM_RING), -1e9, device=dev)
        self.cam_ptr = torch.zeros(N, dtype=torch.long, device=dev)
        self.next_cap = z(N)
        self.q_hist, self.a_hist = z(N, c.history, J), z(N, c.history, J)
        # per-episode parameters
        self.diff, self.diff_g = z(N), z(N, 4)
        self.ball_prm = {k: torch.full((N,), v[0], device=dev) for k, v in B.BALL_RANGES.items()}
        self.delay_sub = torch.zeros(N, dtype=torch.long, device=dev)
        self.cam_period, self.cam_lat, self.cam_drop = z(N), z(N), z(N)
        self.cam_side = torch.zeros(N, dtype=torch.bool, device=dev)
        self.sig_perp, self.sig_ray, self.depth_err, self.exposure = z(N), z(N), z(N), z(N)
        self.cam_o, self.cam_s = z(N, 3), z(N, 3)
        self.burst_in, self.burst_out, self.occl_p, self.blurmiss_p = z(N), z(N), z(N), z(N)
        self.cam_bad = torch.zeros(N, dtype=torch.bool, device=dev)
        self.age_bias_env, self.hand_t = z(N), z(N)
        self.depth_drift = z(N)                            # slowly drifting part of the overhead depth error
        self.reach_env, self.lift_env = z(N), z(N)
        self.tip_env, self.strike_k, self.tilt_g = z(N), z(N), torch.ones(N, device=dev)
        self.home_R = torch.eye(3, device=dev).expand(N, 3, 3).clone()
        self.boundary_g = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.adr_buf = [[] for _ in range(4)]
        self.cam_bias, self.q_off = z(N, 3), z(N, J)
        self.overload_limit, self.push_p = z(N), z(N)
        self.target_apex = z(N)
        self.wind, self.pj, self.pjd, self.pj_std, self.pj_w, self.pj_z = z(N, 3), z(N, 2), z(N, 2), z(N), z(N), z(N)
        self.glitch_p_env, self.glitch_left = z(N), torch.zeros(N, dtype=torch.long, device=dev)
        self.ctrl_jit = torch.zeros(N, dtype=torch.long, device=dev)
        self.cam_R = torch.eye(3, device=dev).expand(N, 3, 3).clone()
        self.view_R, self.view_t = torch.eye(3, device=dev).expand(N, 3, 3).clone(), z(N, 3)
        self.view_roi, self.view_on = z(N, 4), torch.zeros(N, dtype=torch.bool, device=dev)
        self.h_gain = torch.ones(N, device=dev)
        self.blur, self.outlier_p_env, self.stamp_sd = z(N), z(N), z(N)
        self.rs_env = z(N)
        self.blade = torch.tensor([c.paddle.blade_a, c.paddle.blade_b, 2.2], device=dev).expand(N, 3).clone()
        self.rest_start = torch.zeros(N, dtype=torch.bool, device=dev)
        self.hold_left, self.held_hidden = z(N), torch.zeros(N, dtype=torch.bool, device=dev)
        self.p_hold, self.landed = z(N, 3), torch.ones(N, dtype=torch.bool, device=dev)
        self.held_on = c.held_start_prob > 0          # held starts in use (training, or forced in evaluation)
        if c.auto_level:
            self.al_drift, self.al_last, self.al_qref0 = z(N, 2), torch.full((N,), -1e9, device=dev), z(N, J)
            self.al_period = max(1, int(round(c.al_period_s * c.ctrl_hz)))
        self.wait_left = torch.zeros(N, dtype=torch.long, device=dev)   # control steps before the policy acts
        self.launch_step = torch.full((N,), -1, dtype=torch.long, device=dev)   # replayed launch: next step, -1 off
        self.launch_var = torch.ones(N, dtype=torch.long, device=dev)
        self.launch_scale = torch.ones(N, device=dev)
        self.view_top_env = torch.full((N,), float("inf"), device=dev)
        if c.launch_replay_p > 0:                    # the tuned primitive with one strike step fewer / as tuned / more
            from bounce import launch as LAUNCH
            base = LAUNCH.get(c.p_gain, c.launch_name)
            lengths = (range(c.launch_strike_range[0], c.launch_strike_range[1] + 1) if c.launch_strike_range else
                       [base.strike_steps + k for k in (-1, 0, 1)])
            seqs = [LAUNCH.actions(LAUNCH.Launch(**{**base.as_dict(), "strike_steps": L}), 1, dev)[:, 0] for L in lengths]
            strike_of = [(base.dip_steps, L) for L in lengths]
            if c.launch_mix:                         # several launches: group g holds launch g's strike-length variants
                seqs, strike_of, sp, sv, grp = [], [], [], [], []
                for g, nm in enumerate(c.launch_mix):
                    lb = LAUNCH.get(c.p_gain, nm)
                    lens = lengths if nm == "" else [lb.strike_steps + k for k in (-1, 0, 1)]
                    for L in lens:
                        seqs.append(LAUNCH.actions(LAUNCH.Launch(**{**lb.as_dict(), "strike_steps": L}), 1, dev)[:, 0])
                        strike_of.append((lb.dip_steps, L))
                        sp.append(float(getattr(lb, "steer_p", 0.0)))
                        sv.append(float(getattr(lb, "steer_v", 0.0)))
                        grp.append(g)
                grp_t = torch.tensor(grp, device=dev)
                self.mix_start = torch.tensor([grp.index(g) for g in range(len(c.launch_mix))], device=dev)
                self.mix_size = torch.tensor([grp.count(g) for g in range(len(c.launch_mix))], device=dev)
                self.mix_cum = torch.tensor(c.launch_mix_p, dtype=torch.float32, device=dev).cumsum(0)
                self.mix_cum = self.mix_cum / self.mix_cum[-1]
                self.mix_named = torch.tensor([nm != "" for nm in c.launch_mix], device=dev)
                self.launch_sp = torch.tensor(sp, device=dev)
                self.launch_sv = torch.tensor(sv, device=dev)
                self.mix_grp = grp_t
            T = max(x.shape[0] for x in seqs)
            self.launch_seq = torch.zeros(len(seqs), T, J, device=dev)
            for i, x in enumerate(seqs):
                self.launch_seq[i, :x.shape[0]] = x
            self.launch_len = torch.tensor([x.shape[0] for x in seqs], device=dev)
            self.launch_base = base
            self.launch_strike = torch.zeros(len(seqs), T, dtype=torch.float32, device=dev)   # 1 during the strike
            for i, (d0, L) in enumerate(strike_of):
                self.launch_strike[i, d0:d0 + L] = 1.0
        self.on_target = torch.zeros(N, dtype=torch.bool, device=dev)   # a bounce within 3 cm of the target yet
        # episode statistics
        self.ep = {k: z(N) for k in ("ret", "bounces", "apex_err", "lateral", "launch_t", "peak_load", "steady_t",
                                     "first_apex", "first_dx")}
        self.tgt_steady, self.tgt_start, self.t_high = z(N), z(N), z(N)
        self.sched = torch.zeros(N, dtype=torch.bool, device=dev)
        self.sched_any = c.target_sched_p > 0
        self.launch_steer_a, self._steer_J = z(N, 2), None
        self.completed = []
        self.forced = None              # evaluation: fixed difficulty and start type
        self.final_obs = self.final_critic_obs = None
        self.goal_prev = self.q.clone()
        self.oeff_prev = self.o_prev.clone()
        self.Reff_prev = self.R_prev.clone()
        self.kin.g = self.g_env
        self.reset_all()

    # ------------------------------------------------------------------ setup helpers
    def _solve_level(self, q, ids, iters: int = 8, up=None):
        """Adjust wrist flex and wrist roll of q (n, J) so the paddle face is level, i.e. perpendicular to `up`
        (n, 3; default the robot's z): the user levels it with a phone level (Newton, numeric Jacobian)."""
        eps = 1e-3
        if up is None:
            up = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand_as(up)
        e1 = ex - (ex * up).sum(-1, keepdim=True) * up
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = torch.cross(up, e1, dim=-1)
        for _ in range(iters):
            def tilt(qq):
                n = self._fk_ids(qq, ids)["paddle_R"][..., 2]
                return torch.stack([(n * e1).sum(-1), (n * e2).sum(-1)], -1)
            f = tilt(q)
            cols = []
            for j in (3, 4):
                dq = torch.zeros_like(q)
                dq[:, j] = eps
                cols.append((tilt(q + dq) - f) / eps)
            Jm = torch.stack(cols, -1)                                       # (n, 2, 2)
            step = torch.linalg.solve(Jm + 1e-6 * torch.eye(2, device=q.device), f.unsqueeze(-1)).squeeze(-1)
            q = q.clone()
            q[:, 3:5] -= step
        return q

    def _fk_ids(self, q, ids):
        """Forward kinematics for a subset of environments (their own paddle mounts)."""
        kin = self.kin
        saved = kin.paddle_pos, kin.paddle_rot, kin.ipos, kin.mass
        kin.paddle_pos, kin.paddle_rot = kin.paddle_pos[ids], kin.paddle_rot[ids]
        kin.ipos, kin.mass = kin.ipos[ids], kin.mass[ids]
        try:
            return kin.forward(q)
        finally:
            kin.paddle_pos, kin.paddle_rot, kin.ipos, kin.mass = saved

    def _u(self, n, *shape):
        return torch.rand(n, *shape, generator=self.gen, device=self.device)

    def _n(self, n, *shape):
        return torch.randn(n, *shape, generator=self.gen, device=self.device)

    # ------------------------------------------------------------------ reset
    def reset_all(self):
        self._reset(torch.arange(self.N, device=self.device))
        self.completed = []
        return self._obs(), {}

    def reset(self):
        return self.reset_all()

    def _reset(self, ids):
        c, n, dev = self.cfg, ids.numel(), self.device
        if n == 0:
            return
        if self.forced is not None:
            dg = torch.full((n, 4), float(self.forced["difficulty"]), device=dev)
            if self.forced.get("groups") is not None:                  # evaluation: per-group levels
                dg = torch.tensor(self.forced["groups"], dtype=torch.float32, device=dev).expand(n, 4).clone()
            rest = torch.full((n,), bool(self.forced["rest_start"]), device=dev, dtype=torch.bool) \
                if self.forced.get("rest_start") is not None else self._u(n) < c.rest_start_prob
        else:
            dg = torch.tensor(self.levels, device=dev) * self._u(n, 4).sqrt()
            rest = self._u(n) < c.rest_start_prob
        # ADR boundary sampling (OpenAI 2019): a quarter of episodes put one group exactly at its current level
        bnd = torch.where(self._u(n) < 0.25, torch.randint(0, 4, (n,), generator=self.gen, device=dev),
                          torch.full((n,), -1, dtype=torch.long, device=dev))
        if self.forced is None:
            lv = torch.tensor(self.levels, device=dev)
            for g in range(4):
                dg[:, g] = torch.where(bnd == g, lv[g].expand(n), dg[:, g])
        else:
            bnd[:] = -1
        self.boundary_g[ids] = bnd
        da, ds, dp, dt_ = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 3]          # actuation, sensing, physics, task
        self.diff_g[ids], self.diff[ids], self.rest_start[ids] = dg, dg.mean(-1), rest
        L = lambda r, d: _lerp(r, d, self._u(n))
        LL = lambda r, d: _lerp_log(r, d, self._u(n))
        # ---- robot and paddle (physics group; servo parameters in the actuation group)
        self.servo.sample(ids, da, self.gen)
        ms = _lerp(c.mass_scale, dp[:, None], self._u(n, self.mass0.numel()))
        self.kin.mass[ids] = self.mass0 * ms
        self.kin.mass[ids, self.kin.paddle] = L(c.paddle_mass, dp)
        reach, lift = L(c.paddle_reach, dp), L(c.face_lift, dp)
        self.reach_env[ids], self.lift_env[ids] = reach, lift
        side = c.mount_side * dp * (2 * self._u(n) - 1)
        ax = self.paddle_rot0                                 # paddle axes (columns) in the gripper frame
        self.kin.paddle_pos[ids] = self.paddle_pos0 + (reach - c.paddle.reach)[:, None] * ax[:, 0] + \
            lift[:, None] * ax[:, 2] + side[:, None] * ax[:, 1]
        ang = dp[:, None] * (2 * self._u(n, 3) - 1) * torch.tensor([c.mount_roll, c.mount_pitch, c.mount_yaw],
                                                                    device=dev)
        e = torch.eye(3, device=dev)
        self.kin.paddle_rot[ids] = ax @ axis_angle(e[0], ang[:, 0]) @ axis_angle(e[1], ang[:, 1]) @ \
            axis_angle(e[2], ang[:, 2])
        self.kin.ipos[ids, self.kin.paddle] = self.ipos_paddle0 + (c.paddle_com_shift * dp * (2 * self._u(n) - 1))[
            :, None] * e[0]
        self.blade[ids] = torch.stack([L(c.blade_a, dp), L(c.blade_b, dp), L(c.blade_shape, dp)], -1)
        # ---- hold pose (task group): level against gravity, then the command that holds it there
        q_start = self.q_nom.expand(n, J).clone()
        q_start[:, 1:3] += c.hold_jitter * dt_[:, None] * (2 * self._u(n, 2) - 1)
        dp_t = dp if c.table_tilt_full_p <= 0 else torch.where(self._u(n) < c.table_tilt_full_p, 1.0, dp)
        tilt_ab = c.table_tilt * dp_t[:, None] * (2 * self._u(n, 2) - 1)
        g_dir = torch.cat([tilt_ab, -torch.ones(n, 1, device=dev)], -1)
        self.g_env[ids] = 9.81 * g_dir / g_dir.norm(dim=-1, keepdim=True)
        q_level = self._solve_level(q_start, ids, up=-self.g_env[ids] / 9.81)
        dt_lv = dt_ if c.level_error_full_p <= 0 else torch.where(self._u(n) < c.level_error_full_p, 1.0, dt_)
        q_level[:, 3:5] += c.level_error * dt_lv[:, None] * (2 * self._u(n, 2) - 1)
        fk = self._fk_ids(q_level, ids)
        tau_g, _ = self._gravity_ids(fk, ids)
        sp = self.servo.p
        hold_current = (tau_g / (sp["strength"][ids] * sp["kt"][ids])).abs().sum(-1, keepdim=True)
        sag = (1 - sp["supply_r"][ids] * hold_current / sp["vin"][ids]).clamp(min=0.5)     # supply voltage sag
        k = self.servo.stiffness()[ids] * sag
        q_ref = q_level + tau_g / k
        q_init, qd_init = q_level, torch.zeros_like(q_level)
        if c.start_pose_p > 0:                       # arm not yet settled at the hold pose when the policy starts
            sel = (self._u(n) < c.start_pose_p)[:, None]
            dq = torch.tensor(c.start_pose_dq, device=dev) * (2 * self._u(n, J) - 1)
            qd0 = c.start_qd * (2 * self._u(n, J) - 1)
            q_init = torch.where(sel, (q_level + dq).clamp(self.kin.lower + 0.02, self.kin.upper - 0.02), q_level)
            qd_init = torch.where(sel, qd0, qd_init)
            fk = self._fk_ids(q_init, ids)           # where the paddle really is (ball start, previous pose)
        self.q[ids], self.qd[ids], self.q_ref[ids] = q_init, qd_init, q_ref
        if c.auto_level:
            self.al_drift[ids], self.al_last[ids], self.al_qref0[ids] = 0.0, -1e9, q_ref
        # motor side of the gears: wound up against the gravity load, play taken up on the loaded side
        self.qm[ids] = q_init + tau_g / sp["gear_k"][ids] + torch.sign(tau_g) * 0.5 * sp["backlash"][ids]
        self.qdm[ids] = qd_init
        self.servo.reset(ids, q_ref)
        self.cmd_ring[:, ids] = q_ref
        self.o_prev[ids], self.R_prev[ids], self.Reff_prev[ids] = fk["paddle_o"], fk["paddle_R"], fk["paddle_R"]
        self.oeff_prev[ids] = fk["paddle_o"]
        self.goal_prev[ids] = q_ref
        # ---- ball and rubber (physics group)
        for key, r in B.BALL_RANGES.items():
            self.ball_prm[key][ids] = L(r, dp)
        probs = torch.tensor([rb[0] for rb in c.rubbers], device=dev)
        kind = torch.multinomial(probs.expand(n, -1), 1, generator=self.gen).squeeze(-1)
        lo_e = torch.tensor([rb[1][0] for rb in c.rubbers], device=dev)[kind]
        hi_e = torch.tensor([rb[1][1] for rb in c.rubbers], device=dev)[kind]
        lo_m = torch.tensor([rb[2][0] for rb in c.rubbers], device=dev)[kind]
        hi_m = torch.tensor([rb[2][1] for rb in c.rubbers], device=dev)[kind]
        nom_e, nom_m = B.BALL_RANGES["e0"][0], B.BALL_RANGES["mu"][0]
        if c.rubber_fullrange:
            self.ball_prm["e0"][ids] = lo_e + (hi_e - lo_e) * self._u(n)
            self.ball_prm["mu"][ids] = lo_m + (hi_m - lo_m) * self._u(n)
        else:
            self.ball_prm["e0"][ids] = nom_e + dp * (lo_e + (hi_e - lo_e) * self._u(n) - nom_e)
            self.ball_prm["mu"][ids] = nom_m + dp * (lo_m + (hi_m - lo_m) * self._u(n) - nom_m)
        self.tip_env[ids] = c.impact_tip[0] + (c.impact_tip[1] - c.impact_tip[0]) * self._u(n)
        self.strike_k[ids] = c.strike_push[0] + (c.strike_push[1] - c.strike_push[0]) * self._u(n)
        self.tilt_g[ids] = c.tilt_authority[0] + (c.tilt_authority[1] - c.tilt_authority[0]) * self._u(n)
        if c.kick_range:
            self.ball_prm["kick"][ids] = c.kick_range[0] + (c.kick_range[1] - c.kick_range[0]) * self._u(n)
        self.wind[ids] = torch.cat([c.wind * dp[:, None] * (2 * self._u(n, 2) - 1), torch.zeros(n, 1, device=dev)], -1)
        self.pj_std[ids], self.pj[ids], self.pjd[ids] = LL(c.paddle_jitter, dp), 0.0, 0.0
        self.pj_w[ids] = 2 * math.pi * L(c.jitter_hz, dp)
        self.pj_z[ids] = L(c.jitter_zeta, dp)
        # ---- ball start (task group): resting anywhere on the blade, or dropped / tossed
        o, R = fk["paddle_o"], fk["paddle_R"]
        if self.forced is not None and self.forced.get("rest_radius") is not None:
            rad = self.forced["rest_radius"] + 0.05 * (2 * self._u(n) - 1)
        else:
            rad = (0.2 + (c.rest_spread - 0.2) * dt_) * self._u(n).sqrt()    # uniform over the area
        ang = 2 * math.pi * self._u(n)
        bl = self.blade[ids]
        du0, dv0 = bl[:, 0] * rad * torch.cos(ang), bl[:, 1] * rad * torch.sin(ang)
        p_rest = o + B.RADIUS * R[..., 2] + du0[:, None] * R[..., 0] + dv0[:, None] * R[..., 1]
        lat = (0.01 + (c.flight_spread - 0.01) * dt_)[:, None] * (2 * self._u(n, 2) - 1)
        if c.drop_blade_p > 0:
            rb, ab = c.drop_blade_r * self._u(n).sqrt(), 2 * math.pi * self._u(n)
            lat_b = torch.stack([bl[:, 0] * rb * torch.cos(ab), bl[:, 1] * rb * torch.sin(ab)], -1)
            lat = torch.where((self._u(n) < c.drop_blade_p)[:, None], lat_b, lat)
        h0 = c.flight_h[0] + (c.flight_h[1] - c.flight_h[0]) * self._u(n)
        drop = self.forced is not None and self.forced.get("drop_height") is not None
        if drop:                                          # evaluation: released at rest from a fixed height and offset
            ang_d = 2 * math.pi * self._u(n)
            lat = float(self.forced.get("drop_offset", 0.0)) * torch.stack([torch.cos(ang_d), torch.sin(ang_d)], -1)
            h0 = torch.full((n,), float(self.forced["drop_height"]), device=dev)
        held = torch.zeros(n, dtype=torch.bool, device=dev)
        beside = torch.zeros(n, dtype=torch.bool, device=dev)
        hold_s, hidden = torch.zeros(n, device=dev), torch.zeros(n, dtype=torch.bool, device=dev)
        if self.forced is None and (c.held_start_prob > 0 or c.beside_start_prob > 0):
            u2 = self._u(n) * (1 - c.rest_start_prob)                       # among the flight starts
            held = ~rest & (u2 < c.held_start_prob)
            beside = ~rest & ~held & (u2 < c.held_start_prob + c.beside_start_prob)
            ang_b = 2 * math.pi * self._u(n)
            r_b = c.beside_offset[0] + (c.beside_offset[1] - c.beside_offset[0]) * self._u(n)
            h_b = 0.05 + (c.flight_h[1] - 0.05) * self._u(n)
            dt_b = dt_ if c.beside_full_p <= 0 else torch.where(self._u(n) < c.beside_full_p, 1.0, dt_)
            if c.beside_feasible:                  # near to far with task difficulty; high enough to reach in time:
                r_b = c.beside_offset[0] + (c.beside_offset[1] - c.beside_offset[0]) * self._u(n) * dt_b
                t_need = 0.12 + (r_b - 0.07).clamp(min=0) / 0.8      # ~0.12 s to see and start, then ~0.8 m/s of pan
                h_min = (0.5 * 9.81 * t_need ** 2).clamp(max=c.flight_h[1])
                h_b = h_min + (c.flight_h[1] - h_min) * self._u(n)
            lat = torch.where(beside[:, None], r_b[:, None] * torch.stack([torch.cos(ang_b), torch.sin(ang_b)], -1), lat)
            h0 = torch.where(beside, h_b, h0)
            hold_s = c.held_max_s * self._u(n)
            hidden = self._u(n) < c.held_hidden_prob
        if self.forced is not None and self.forced.get("held_s") is not None:
            held = ~rest
            hold_s = torch.full((n,), float(self.forced["held_s"]), device=dev)
            hidden = torch.full((n,), bool(self.forced.get("held_hidden", False)), dtype=torch.bool, device=dev)
            self.held_on = True
        if self.held_on:
            self.hold_left[ids] = torch.where(held, hold_s, torch.zeros_like(hold_s))
            self.held_hidden[ids] = held & hidden
        mid = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.forced is None and c.midbounce_p > 0:   # ball already bouncing when the policy switches on
            mid = ~rest & ~held & ~beside & (self._u(n) < c.midbounce_p)
            apex = c.midbounce_apex[0] + (c.midbounce_apex[1] - c.midbounce_apex[0]) * self._u(n)
            h_mid = apex * self._u(n)
            vz_mid = torch.sqrt(2 * 9.81 * (apex - h_mid)) * torch.where(self._u(n) < 0.5, -1.0, 1.0)
            h0 = torch.where(mid, h_mid, h0)
        toss = torch.zeros(n, dtype=torch.bool, device=dev)
        sidein = torch.zeros_like(toss)
        tossing = self.forced is None and (c.toss_p > 0 or c.sidein_p > 0)
        if tossing:                                    # thrown or bounced in from the side, aimed near the centre
            plain = ~rest & ~held & ~beside & ~mid
            u4 = self._u(n)
            toss = plain & (u4 < c.toss_p)
            sidein = plain & ~toss & (u4 < c.toss_p + c.sidein_p)
            ang_t = 2 * math.pi * self._u(n)
            dir_t = torch.stack([torch.cos(ang_t), torch.sin(ang_t)], -1)
            aim = 0.04 * (2 * self._u(n, 2) - 1)
            dt_t = dt_ if c.toss_full_p <= 0 else torch.where(self._u(n) < c.toss_full_p, 1.0, dt_)
            vlat_t = c.toss_vlat * self._u(n) * ((0.3 + 0.7 * dt_t) if c.toss_curriculum else 1.0)
            vz_t = c.toss_vz[0] + (c.toss_vz[1] - c.toss_vz[0]) * self._u(n)
            t_t = (vz_t + torch.sqrt(vz_t ** 2 + 2 * 9.81 * h0)) / 9.81           # time to reach the face
            start_t = aim - (vlat_t * t_t)[:, None] * dir_t
            k_t = (0.16 / start_t.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0)    # start inside the view limit
            start_t, vlat_t = aim + (start_t - aim) * k_t[:, None], vlat_t * k_t
            r_s = c.sidein_offset[0] + (c.sidein_offset[1] - c.sidein_offset[0]) * self._u(n)
            h_s = c.sidein_h[0] + (c.sidein_h[1] - c.sidein_h[0]) * self._u(n)
            apex_s = c.sidein_apex[0] + (c.sidein_apex[1] - c.sidein_apex[0]) * self._u(n)
            vz_s = torch.sqrt(2 * 9.81 * (apex_s - h_s))
            t_s = (vz_s + torch.sqrt(2 * 9.81 * apex_s)) / 9.81
            start_s = r_s[:, None] * dir_t
            v_s = (aim - start_s) / t_s[:, None]
            lat = torch.where(toss[:, None], start_t, torch.where(sidein[:, None], start_s, lat))
            h0 = torch.where(sidein, h_s, h0)
        ds_q = ds if c.joint_offset_full_p <= 0 else torch.where(self._u(n) < c.joint_offset_full_p, 1.0, ds)
        self.q_off[ids] = c.joint_offset * ds_q[:, None] * (2 * self._u(n, J) - 1)
        self.home_env[ids] = self.kin_nominal.forward(q_level + self.q_off[ids])["paddle_o"]
        if c.tilt_authority[0] < 1:
            self.home_R[ids] = self._fk_ids(q_level, ids)["paddle_R"]         # the face at the hold pose
        p_fly = self.home_env[ids] + torch.cat([lat, (h0 + B.RADIUS)[:, None]], -1)
        v_fly = torch.cat([(0.02 + 0.13 * dt_)[:, None] * (2 * self._u(n, 2) - 1), (2 * self._u(n, 1) - 1) * 1.0], -1)
        if drop:
            v_fly = torch.zeros_like(v_fly)
            if self.forced.get("drop_vlat"):                          # evaluation: tossed in sideways
                ang_v = 2 * math.pi * self._u(n)
                v_fly[:, :2] = float(self.forced["drop_vlat"]) * torch.stack([torch.cos(ang_v), torch.sin(ang_v)], -1)
            if self.forced.get("drop_vz"):
                v_fly[:, 2] = float(self.forced["drop_vz"])
        if self.forced is None and c.beside_start_prob > 0:
            ang_v = 2 * math.pi * self._u(n)
            v_b = c.beside_vlat * self._u(n)[:, None] * torch.stack([torch.cos(ang_v), torch.sin(ang_v)], -1)
            if c.beside_feasible:
                v_b = v_b * dt_b[:, None]
            v_fly[:, :2] = torch.where(beside[:, None], v_b, v_fly[:, :2])
        if self.forced is None and c.midbounce_p > 0:
            v_fly[:, 2] = torch.where(mid, vz_mid, v_fly[:, 2])
        if tossing:
            v_fly[:, :2] = torch.where(toss[:, None], vlat_t[:, None] * dir_t,
                                       torch.where(sidein[:, None], v_s, v_fly[:, :2]))
            v_fly[:, 2] = torch.where(toss, vz_t, torch.where(sidein, vz_s, v_fly[:, 2]))
        v_fly = torch.where(held[:, None], torch.zeros_like(v_fly), v_fly)
        self.p_ball[ids] = torch.where(rest[:, None], p_rest, p_fly)
        self.v_ball[ids] = torch.where(rest[:, None], torch.zeros_like(v_fly), v_fly)
        self.w_ball[ids] = 20.0 * dt_[:, None] * (2 * self._u(n, 3) - 1) * (~rest)[:, None] * (0.0 if drop else 1.0)
        if drop and self.forced.get("drop_spin"):
            w_dir = self._n(n, 3)
            self.w_ball[ids] = float(self.forced["drop_spin"]) * w_dir / w_dir.norm(dim=-1, keepdim=True)
        if tossing and c.toss_p > 0:
            w_dir = self._n(n, 3)
            w_dir = w_dir / w_dir.norm(dim=-1, keepdim=True)
            spin = c.toss_spin * self._u(n) * ((0.2 + 0.8 * dt_t) if c.toss_curriculum else 1.0)
            self.w_ball[ids] = torch.where(toss[:, None], spin[:, None] * w_dir, self.w_ball[ids])
        if self.forced is None and c.rest_roll_p > 0:     # dead stop that is not quite dead: rolling slowly
            roll = rest & (self._u(n) < c.rest_roll_p)
            ang_r = 2 * math.pi * self._u(n)
            v_r = (c.rest_roll_v * self._u(n))[:, None] * (torch.cos(ang_r)[:, None] * R[..., 0] +
                                                           torch.sin(ang_r)[:, None] * R[..., 1])
            self.v_ball[ids] = torch.where(roll[:, None], v_r, self.v_ball[ids])
            self.w_ball[ids] = torch.where(roll[:, None], torch.cross(R[..., 2], v_r, dim=-1) / B.RADIUS,
                                           self.w_ball[ids])
        self.w_ball[ids] = torch.where(held[:, None], torch.zeros_like(self.w_ball[ids]), self.w_ball[ids])
        self.p_hold[ids] = self.p_ball[ids]
        self.landed[ids] = rest
        self.prev_vz[ids] = self.v_ball[ids, 2]
        self.launch_z[ids] = torch.where(rest, self.p_ball[ids, 2] - B.RADIUS, self.home_env[ids, 2])
        dt_p = dt_ if c.push_full_p <= 0 else torch.where(self._u(n) < c.push_full_p, self._u(n), dt_)
        self.push_p[ids] = c.push_rate_hz * dt_p * self.sub_dt
        self.target_apex[ids] = c.target_apex[0] + (c.target_apex[1] - c.target_apex[0]) * self._u(n)
        self.tgt_steady[ids] = self.target_apex[ids]
        if c.target_sched_p > 0:                     # start high (more momentum, easier to get going), then low
            sel = self._u(n) < c.target_sched_p
            self.sched[ids] = sel
            self.tgt_start[ids] = c.target_start[0] + (c.target_start[1] - c.target_start[0]) * self._u(n)
            self.t_high[ids] = c.target_high_s[0] + (c.target_high_s[1] - c.target_high_s[0]) * self._u(n)
            self.target_apex[ids] = torch.where(sel, self.tgt_start[ids], self.target_apex[ids])
        # ---- actuation timing
        self.delay_sub[ids] = (LL(c.act_delay_ms, da) / 1000 / self.sub_dt).round().long().clamp(0, CMD_RING - 2)
        self.glitch_p_env[ids], self.glitch_left[ids] = LL(c.glitch_p, da), 0
        self.ctrl_jit[ids] = (c.ctrl_jitter_ms * da / 1000 / self.sub_dt).round().long()
        self.overload_limit[ids] = L(c.overload_s, da)
        # ---- cameras and lighting (sensing group)
        if self.forced is not None and self.forced.get("side_cam") is not None:
            side_cam = torch.full((n,), bool(self.forced["side_cam"]), dtype=torch.bool, device=dev)
        else:
            side_cam = self._u(n) < c.side_cam_prob
        light = L(c.light, ds)
        over_fps = torch.where((light < 0.3) & (self._u(n) < 0.5), torch.full_like(light, 15.0), L(c.overhead_fps, ds))
        fps = torch.where(side_cam, L(c.side_fps, ds), over_fps)
        self.cam_side[ids], self.cam_period[ids] = side_cam, 1.0 / fps
        exposure = self.cam_period[ids] * (0.1 + 0.9 * (1 - light))           # dim light: exposure -> frame period
        self.cam_lat[ids] = (L(c.cam_frames, ds) * self.cam_period[ids] + LL(c.cam_base_ms, ds) / 1000).clamp(
            max=c.cam_lat_max)                                                # beyond ~120 ms nothing can bounce
        dim = 1 + 1.5 * (1 - light)                                            # dim light: noisier detections
        self.sig_perp[ids] = LL(c.cam_noise_px, ds) * dim
        self.sig_ray[ids] = torch.where(side_cam, LL(c.tri_noise, ds) * dim, LL(c.overhead_depth, ds))
        self.depth_drift[ids] = 0.0
        self.depth_err[ids] = torch.where(side_cam, c.side_scale, c.depth_scale) * ds * (2 * self._u(n) - 1)
        jit = 0.1 * ds[:, None] * (2 * self._u(n, 3) - 1)
        self.cam_o[ids] = torch.tensor(c.overhead_cam, device=dev) + jit
        if c.overhead_cam_box:                                               # camera mount unknown: anywhere in a box
            lo = torch.tensor(c.overhead_cam_box[0], dtype=torch.float32, device=dev)
            hi = torch.tensor(c.overhead_cam_box[1], dtype=torch.float32, device=dev)
            box = lo + (hi - lo) * self._u(n, 3)
            if c.overhead_cam_box_p < 1.0:                                   # the rest near the measured mount
                box = torch.where((self._u(n) < c.overhead_cam_box_p)[:, None], box, self.cam_o[ids])
            self.cam_o[ids] = box
        if self.forced is not None and self.forced.get("cam_pos") is not None:
            self.cam_o[ids] = torch.tensor(self.forced["cam_pos"], dtype=torch.float32, device=dev).expand(n, 3)
        self.cam_s[ids] = torch.tensor(c.side_cam, device=dev) + jit
        self.exposure[ids], self.blur[ids] = exposure, L(c.blur_k, ds) * exposure
        self.cam_drop[ids] = LL(c.cam_dropout, ds) * dim
        self.burst_in[ids], self.burst_out[ids] = LL(c.burst_in, ds), L(c.burst_out, ds)
        self.occl_p[ids], self.blurmiss_p[ids] = L(c.occlusion_p, ds), L(c.blur_miss_p, ds)
        self.cam_bad[ids] = False
        self.cam_bias[ids] = c.cam_bias * ds[:, None] * (2 * self._u(n, 3) - 1)
        ca = c.cam_rot_err * ds[:, None] * (2 * self._u(n, 3) - 1)
        self.cam_R[ids] = axis_angle(e[0], ca[:, 0]) @ axis_angle(e[1], ca[:, 1]) @ axis_angle(e[2], ca[:, 2])
        if c.cam_view_p > 0:
            R0 = torch.tensor(c.cam_view[:9], dtype=torch.float32, device=dev).reshape(3, 3)
            t0 = torch.tensor(c.cam_view[9:12], dtype=torch.float32, device=dev)
            f, W, H = c.cam_view[12:15]
            h, b = c.cam_view_half, c.cam_view_back
            d = torch.tensor([[dx, dy, dz] for dx in (-b, h) for dy in (-h, h) for dz in c.cam_view_z], device=dev)
            pc = (self.home_env[ids][:, None] + d - t0) @ R0                      # R0^T (p - t0), (n, 8, 3)
            u, v = f * pc[..., 0] / pc[..., 2] + W / 2, f * pc[..., 1] / pc[..., 2] + H / 2
            self.view_roi[ids] = torch.stack([u.min(1).values.clamp(min=0), v.min(1).values.clamp(min=0),
                                              u.max(1).values.clamp(max=W), v.max(1).values.clamp(max=H)], -1)
            jr = c.cam_view_jit[1] * (2 * self._u(n, 3) - 1)
            self.view_R[ids] = axis_angle(e[0], jr[:, 0]) @ axis_angle(e[1], jr[:, 1]) @ axis_angle(e[2], jr[:, 2]) @ R0
            self.view_t[ids] = t0 + c.cam_view_jit[0] * (2 * self._u(n, 3) - 1)
            self.view_on[ids] = self._u(n) < c.cam_view_p
        if c.cam_height_gain:
            self.h_gain[ids] = c.cam_height_gain[0] + (c.cam_height_gain[1] - c.cam_height_gain[0]) * self._u(n)
        self.outlier_p_env[ids] = LL(c.outlier_p, ds)
        self.stamp_sd[ids] = LL(c.stamp_noise, ds)
        self.age_bias_env[ids] = c.age_bias * ds * (2 * self._u(n) - 1)
        if c.rolling_shutter:
            self.rs_env[ids] = c.rolling_shutter * (2 * self._u(n) - 1)
        self.hand_t[ids] = 0.0
        # ---- clocks and buffers
        self.t[ids], self.steps[ids] = 0.0, 0
        self.on_target[ids] = False
        self.rest_time[ids], self.since_impact[ids], self.overload_time[ids] = 0.0, 0.0, 0.0
        self.cam_t[ids], self.cam_ptr[ids] = -1e9, 0
        self.next_cap[ids] = self.cam_period[ids] * self._u(n)
        qm = self._measure_q(ids)
        self.q_hist[ids] = (qm - self.q_ref[ids])[:, None].expand(n, c.history, J)
        self.a_hist[ids] = 0.0
        if c.hist_noise_p > 0:                       # history not reset (e.g. after a re-arm)
            sel = (self._u(n) < c.hist_noise_p)[:, None, None]
            self.a_hist[ids] = torch.where(sel, 0.5 * (2 * self._u(n, c.history, J) - 1), self.a_hist[ids])
            self.q_hist[ids] = torch.where(sel, self.q_hist[ids] + 0.02 * (2 * self._u(n, c.history, J) - 1),
                                           self.q_hist[ids])
        if c.switch_delay_p > 0:                     # the runner switches the policy on after the ball is seen
            d = (self._u(n) < c.switch_delay_p).float() * c.switch_delay_s * self._u(n)
            self.wait_left[ids] = (d * c.ctrl_hz).round().long()
        if c.launch_replay_p > 0:                    # resting ball: the runner's --launch runs first
            sel = rest & (self._u(n) < c.launch_replay_p) if self.forced is None else torch.zeros_like(rest)
            self.launch_step[ids] = torch.where(sel, 0, -1)
            self.launch_scale[ids] = 1 + c.launch_jitter * (2 * self._u(n) - 1)
            if c.launch_mix:                         # which launch, then which of its strike lengths
                g = torch.searchsorted(self.mix_cum, self._u(n).contiguous()).clamp(max=len(c.launch_mix) - 1)
                k = (self._u(n) * self.mix_size[g]).long().clamp(max=self.mix_size[g] - 1)
                self.launch_var[ids] = self.mix_start[g] + k
                if c.launch_scale_range:
                    lo, hi = c.launch_scale_range
                    self.launch_scale[ids] = torch.where(self.mix_named[g], lo + (hi - lo) * self._u(n),
                                                         self.launch_scale[ids])
            elif c.launch_strike_range:
                self.launch_var[ids] = (self._u(n) * self.launch_seq.shape[0]).long().clamp(max=self.launch_seq.shape[0] - 1)
            else:
                odd = self._u(n) < 0.3
                self.launch_var[ids] = torch.where(odd, 2 * (self._u(n) < 0.5).long(), torch.ones_like(self.launch_var[ids]))
        if c.view_top_p > 0 and c.view_top:
            top = c.view_top[0] + (c.view_top[1] - c.view_top[0]) * self._u(n)
            self.view_top_env[ids] = torch.where(self._u(n) < c.view_top_p, top, torch.full_like(top, float("inf")))
        for v in self.ep.values():
            v[ids] = 0.0
        self.ep["launch_t"][ids] = float("nan")
        self.ep["steady_t"][ids] = float("nan")
        self.ep["first_apex"][ids] = float("nan")
        self.ep["first_dx"][ids] = float("nan")
        if self.forced is not None and self.forced.get("post_reset") is not None:
            self.forced["post_reset"](self, ids)                  # evaluation: set any per-episode state directly

    def _gravity_ids(self, fk, ids):
        kin = self.kin
        saved = kin.mass
        saved_g = kin.g
        kin.mass, kin.g = kin.mass[ids], kin.g[ids]
        try:
            return kin.gravity_and_inertia(fk)
        finally:
            kin.mass, kin.g = saved, saved_g

    def _measure_q(self, ids=None):
        q, off = (self.q, self.q_off) if ids is None else (self.q[ids], self.q_off[ids])
        noisy = q + off + self.cfg.joint_noise * self._n(q.shape[0], J)
        step = 2 * math.pi / 4096
        return torch.round(noisy / step) * step

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        c = self.cfg
        a = action.clamp(-c.launch_max_action, c.launch_max_action)   # 1 unless a larger scripted launch is allowed
        if c.launch_replay_p > 0:
            waiting = self.wait_left > 0
        if c.switch_delay_p > 0:                     # policy not switched on yet: the arm holds the hold command
            a = torch.where((self.wait_left > 0)[:, None], torch.zeros_like(a), a)
            self.wait_left = (self.wait_left - 1).clamp(min=0)
        if self.sched_any:                           # high-then-low target: the policy observes the current one
            frac = ((self.t - self.t_high) / c.target_ramp_s).clamp(0, 1)
            cur = self.tgt_start + (self.tgt_steady - self.tgt_start) * frac
            self.target_apex = torch.where(self.sched, cur, self.target_apex)
        if c.launch_replay_p > 0:                    # replayed scripted launch overrides the policy's action
            on = (self.launch_step >= 0) & ~waiting
            k = self.launch_step.clamp(0, self.launch_seq.shape[1] - 1)
            la = self.launch_seq[self.launch_var, k] * self.launch_scale[:, None]
            if c.launch_steer:                       # tilt the strike so the ball flies back toward the centre
                start = on & (self.launch_step == 0)
                if bool(start.any()):
                    off = self.p_ball[:, :2] - self.home_env[:, :2] + 0.01 * self._n(self.N, 2)   # as the camera sees it
                    vel = self.v_ball[:, :2] + 0.05 * self._n(self.N, 2)
                    self.launch_steer_a = torch.where(start[:, None], self._launch_steer(off, vel), self.launch_steer_a)
                la = la.clone()
                la[:, 3:5] = la[:, 3:5] + self.launch_strike[self.launch_var, k][:, None] * self.launch_steer_a
            la = la.clamp(-c.launch_max_action, c.launch_max_action)
            a = torch.where(on[:, None], la, a)
            nxt = self.launch_step + 1
            self.launch_step = torch.where(on, torch.where(nxt >= self.launch_len[self.launch_var], -1, nxt),
                                           self.launch_step)
        if c.auto_level:
            self._auto_level()
        goal = (self.q_ref + self.scale * a).clamp(self.kin.lower + 0.02, self.kin.upper - 0.02)
        # USB hiccup: the command does not reach the servos and the joint reading is stale for 1-4 steps
        start = (self.glitch_left == 0) & (self._u(self.N) < self.glitch_p_env)
        self.glitch_left = torch.where(start, torch.randint(1, 5, (self.N,), generator=self.gen, device=self.device),
                                       self.glitch_left)
        glitch = self.glitch_left > 0
        goal = torch.where(glitch[:, None], self.goal_prev, goal)
        self.goal_prev = goal
        delay = (self.delay_sub + (self._u(self.N) * (self.ctrl_jit + 1).float()).floor().long()).clamp(0, CMD_RING - 2)
        ev = {k: torch.zeros(self.N, device=self.device) for k in ("apex", "hit", "n_hit", "n_apex", "first", "tilt_hit",
                                                                    "back", "aim", "edge")}
        load_pen = torch.zeros(self.N, device=self.device)
        ar = torch.arange(self.N, device=self.device)
        fk = self.kin.forward(self.q)
        for _ in range(c.substeps):
            self.ring_i = (self.ring_i + 1) % CMD_RING
            self.cmd_ring[self.ring_i] = goal
            delayed = self.cmd_ring[(self.ring_i - delay) % CMD_RING, ar]
            tau_g, M = self.kin.gravity_and_inertia(fk)          # at the current pose (fk from the last substep)
            sp = self.servo.p
            tau, duty = self.servo.torque(delayed, self.q, self.qd, self.sub_dt, dq_motor=self.qdm)
            # gear play between the motor side (qm, carries the armature) and the output shaft (q, the encoder)
            gap = self.qm - self.q
            half = 0.5 * sp["backlash"]
            engaged = gap.abs() > half
            tau_c = sp["gear_k"] * torch.sign(gap) * (gap.abs() - half).clamp(min=0) + \
                sp["gear_c"] * (self.qdm - self.qd) * engaged
            # teeth in contact can only push (no pulling while they separate)
            tau_c = torch.where(gap > half, tau_c.clamp(min=0), torch.where(gap < -half, tau_c.clamp(max=0),
                                                                             torch.zeros_like(tau_c)))
            qddm = (tau + self.servo.friction(self.qdm, tau_c) - tau_c) / sp["armature"]
            qdd = (tau_c - tau_g) / (M + 1e-3)                    # + output-side gear inertia
            self.qdm = self.qdm + qddm * self.sub_dt
            self.qm = self.qm + self.qdm * self.sub_dt
            self.qd = self.qd + qdd * self.sub_dt
            q = self.q + self.qd * self.sub_dt
            hit_lim = (q < self.kin.lower) | (q > self.kin.upper)
            self.q = q.clamp(self.kin.lower, self.kin.upper)
            self.qd = torch.where(hit_lim, torch.zeros_like(self.qd), self.qd)
            load = tau.abs() / self.servo.rated_stall()
            over = (load > c.overload_frac).any(-1)
            self.overload_time = torch.where(over, self.overload_time + self.sub_dt, torch.zeros_like(self.t))
            load_pen += ((load - 0.6).clamp(min=0) ** 2).sum(-1) * self.sub_dt
            self.ep["peak_load"] = torch.maximum(self.ep["peak_load"], load.max(-1).values)
            # paddle and ball
            fk = self.kin.forward(self.q)
            o, R = fk["paddle_o"], fk["paddle_R"]
            # paddle-face wobble: a slowly varying small tilt (blade flex, grip play)
            w, zt = self.pj_w[:, None], self.pj_z[:, None]
            drive = (4 * zt * w ** 3 * self.pj_std[:, None] ** 2 * self.sub_dt).sqrt()
            self.pjd = self.pjd + (-w * w * self.pj - 2 * zt * w * self.pjd) * self.sub_dt + drive * self._n(self.N, 2)
            self.pj = self.pj + self.pjd * self.sub_dt
            e3 = torch.eye(3, device=self.device)
            Reff = R @ axis_angle(e3[0], self.pj[:, 0]) @ axis_angle(e3[1], self.pj[:, 1])
            pivot = o - self.reach_env[:, None] * R[..., 0] - self.lift_env[:, None] * R[..., 2]   # the jaws
            oeff = pivot + ((Reff @ R.transpose(1, 2)) @ (o - pivot).unsqueeze(-1)).squeeze(-1)
            push = self._u(self.N) < self.push_p                     # a hand tap, in flight or on a resting ball
            dv = self._n(self.N, 3) * self.push_std
            dv[:, 2] = torch.where(self.rest_time > 0, dv[:, 2].abs(), dv[:, 2])
            self.v_ball = self.v_ball + dv * push[:, None]
            self.hand_t = torch.where(push, torch.full_like(self.hand_t, 0.4), (self.hand_t - self.sub_dt).clamp(min=0))
            self.p_ball, self.v_ball = B.flight(self.p_ball, self.v_ball, self.w_ball, self.ball_prm["drag"],
                                                self.ball_prm["magnus"], self.sub_dt, self.g_env, self.wind)
            if c.tilt_authority[0] < 1:                  # only tilt_g of the face's tilt from the hold pose reaches the ball
                n0, nh = Reff[..., 2], self.home_R[..., 2]
                ng = nh + self.tilt_g[:, None] * (n0 - nh)
                ng = ng / ng.norm(dim=-1, keepdim=True)
                ax = torch.cross(n0, ng, dim=-1)
                s = ax.norm(dim=-1)
                Reff = axis_angle(ax / s.clamp(min=1e-9)[:, None], torch.asin(s.clamp(max=1.0))) @ Reff
            kick = self.ball_prm["kick"][:, None] * self._n(self.N, 3)       # applied only on a hit, along the face
            if c.strike_push[1]:                         # the face gives toward the tip under a fast upward strike
                vp = (((oeff - self.oeff_prev) / self.sub_dt) * Reff[..., 2]).sum(-1).clamp(min=0)
                kick = kick + (self.strike_k * vp)[:, None] * Reff[..., 0]
            dive_p = 0.0 if self.forced is not None and self.forced.get("no_dive") else c.dive_p
            if c.impact_tip[1] or dive_p:
                vn = (self.v_ball * Reff[..., 2]).sum(-1).abs()
                kick = kick + (self.tip_env * (1 + self.ball_prm["e0"]) * vn)[:, None] * Reff[..., 0]   # toward the tip
                if dive_p:
                    big = (self._u(self.N) < dive_p).float()
                    ang = 2 * math.pi * self._u(self.N)
                    mag = (c.dive_kick[0] + (c.dive_kick[1] - c.dive_kick[0]) * self._u(self.N)) * big
                    kick = kick + torch.stack([c.dive_fwd * mag * torch.cos(ang), mag * torch.sin(ang),
                                               torch.zeros_like(mag)], -1)
            self.p_ball, self.v_ball, self.w_ball, touch, impact, speed, du, dv_ = B.paddle_impact(
                self.p_ball, self.v_ball, self.w_ball, oeff, Reff, self.oeff_prev, self.Reff_prev, self.ball_prm,
                self.blade[:, 0], self.blade[:, 1], self.blade[:, 2], self.sub_dt,
                e_noise=self.ball_prm["e_noise"] * self._n(self.N), kick=kick)
            if self.held_on:                                         # held starts: the hand keeps the ball still
                hold = self.hold_left > 0
                self.p_ball = torch.where(hold[:, None], self.p_hold, self.p_ball)
                self.v_ball = torch.where(hold[:, None], torch.zeros_like(self.v_ball), self.v_ball)
                self.w_ball = torch.where(hold[:, None], torch.zeros_like(self.w_ball), self.w_ball)
                touch, impact = touch & ~hold, impact & ~hold
                self.hand_t = torch.where(hold, torch.full_like(self.hand_t, 0.3), self.hand_t)
                self.hold_left = (self.hold_left - self.sub_dt).clamp(min=0)
            self.landed = self.landed | touch
            self.o_prev, self.R_prev, self.Reff_prev, self.oeff_prev = o, R, Reff, oeff
            self.t = self.t + self.sub_dt
            self.rest_time = torch.where(touch, self.rest_time + self.sub_dt, torch.zeros_like(self.t))
            real_hit = impact & (speed > 0.25)
            self.since_impact = torch.where(real_hit, torch.zeros_like(self.t), self.since_impact + self.sub_dt)
            self.launch_z = torch.where(touch, self.p_ball[:, 2] - B.RADIUS, self.launch_z)
            ev["hit"] += real_hit * torch.exp(-(du ** 2 + dv_ ** 2) / 0.03 ** 2)
            ev["n_hit"] += real_hit.float()
            if c.w_edge:
                ev["edge"] += real_hit * ((du / self.blade[:, 0]) ** 2 + (dv_ / self.blade[:, 1]) ** 2)
            if c.w_aim:
                vz_des = (2 * 9.81 * self.target_apex).sqrt()
                v_xy_des = (-(self.p_ball[:, :2] - self.home_env[:, :2]) * 9.81 / (2 * vz_des)[:, None]).clamp(-0.5, 0.5)
                err2 = (self.v_ball[:, :2] - v_xy_des).pow(2).sum(-1) + (self.v_ball[:, 2] - vz_des).pow(2)
                ev["aim"] += real_hit * torch.exp(-err2 / c.aim_sigma ** 2)
            if c.w_tilt_contact:
                tc = torch.rad2deg(torch.acos(Reff[:, 2, 2].clamp(-1, 1)))
                ev["tilt_hit"] += real_hit * ((tc - c.tilt_contact_deg) / c.tilt_contact_deg).clamp(min=0)
            # apex: vertical velocity turns from up to down in flight
            vz = self.v_ball[:, 2]
            apex = (self.prev_vz > 0) & (vz <= 0) & ~touch
            h = self.p_ball[:, 2] - B.RADIUS - self.launch_z           # bounce height above where it left the paddle
            good = apex & (h > 0.03)
            ev["apex"] += good * torch.exp(-((h - self.target_apex) / c.apex_sigma) ** 2)
            ev["n_apex"] += good.float()
            self.ep["apex_err"] += good * (h - self.target_apex).abs()
            if c.w_first or c.w_wait:
                hit_t = good & ((h - self.target_apex).abs() < 0.03)
                ev["first"] += (hit_t & ~self.on_target).float()
                self.on_target = self.on_target | hit_t
            steady = torch.where(self.sched, self.tgt_steady, self.target_apex)
            settled = ~self.sched | (self.t >= self.t_high + c.target_ramp_s)
            st_hit = good & ((h - steady).abs() < 0.03) & settled & torch.isnan(self.ep["steady_t"])
            self.ep["steady_t"] = torch.where(st_hit, self.t, self.ep["steady_t"])
            first = good & torch.isnan(self.ep["launch_t"])
            self.ep["launch_t"] = torch.where(first, self.t, self.ep["launch_t"])
            dxf = self.p_ball[:, 0] - self.home_env[:, 0]                # + = away from the base
            self.ep["first_apex"] = torch.where(first, self.p_ball[:, 2] - B.RADIUS - self.home_env[:, 2],
                                                self.ep["first_apex"])                # above the paddle's home
            self.ep["first_dx"] = torch.where(first, dxf, self.ep["first_dx"])
            ev["back"] += first * (-dxf).clamp(min=0)
            self.prev_vz = vz
            self._camera()
        self.steps += 1
        # reward
        lateral = (self.p_ball[:, :2] - self.home_env[:, :2]).norm(dim=-1)
        bouncing = (self.rest_time < 0.1).float()
        rate = (a - self.a_hist[:, 0]).pow(2).sum(-1)
        tilt = torch.acos(fk["paddle_R"][:, 2, 2].clamp(-1, 1))
        r = {"alive": c.w_alive * self.dt * bouncing,
             "apex": c.w_apex * ev["apex"],
             "hit": c.w_hit * ev["hit"],
             "lateral": -c.w_lateral * self.dt * (lateral / 0.05).pow(2).clamp(max=4.0),
             "rate": -c.w_rate * rate,
             "qd": -c.w_qd * self.dt * self.qd.pow(2).sum(-1),
             "load": -c.w_load * load_pen,
             "tilt": -c.w_tilt * self.dt * (tilt / 0.3).pow(2),
             "home": -c.w_home * self.dt * ((fk["paddle_o"] - self.home_env).norm(dim=-1) / 0.03).pow(2).clamp(max=4.0)}
        if c.w_tilt_contact:                         # hit with the face near level (sideways exits after tilted hits)
            r["tilt_contact"] = -c.w_tilt_contact * ev["tilt_hit"]
        if c.w_first or c.w_wait:                    # get to the target height quickly, from a dead stop too
            r["launch"] = c.w_first * ev["first"] - c.w_wait * self.dt * (~self.on_target).float()
        if c.w_back:                                 # launch forward or straight up, not toward the handle
            r["back"] = -c.w_back * ev["back"]
        if c.w_aim:
            r["aim"] = c.w_aim * ev["aim"]
        if c.w_edge:
            r["edge"] = -c.w_edge * ev["edge"]
        if c.w_track:                                # ballistic landing point at the paddle's height
            vz = self.v_ball[:, 2]
            dz = (self.p_ball[:, 2] - B.RADIUS - fk["paddle_o"][:, 2]).clamp(min=0)
            t_l = (vz + (vz ** 2 + 2 * 9.81 * dz).sqrt()) / 9.81
            land = self.p_ball[:, :2] + self.v_ball[:, :2] * t_l[:, None]
            d = (fk["paddle_o"][:, :2] - land).norm(dim=-1)
            r["track"] = -c.w_track * self.dt * (self.rest_time <= 0) * (d / 0.05).pow(2).clamp(max=4.0)
        # termination
        rel = self.p_ball - self.home_env
        po = fk["paddle_o"] - self.home_env
        reason = torch.full((self.N,), len(FAIL_REASONS) - 1, dtype=torch.long, device=self.device)
        above = rel[:, 2] > (c.ceiling if self.forced is None or self.forced.get("ceiling") is None
                             else float(self.forced["ceiling"]))
        if c.ceiling_after_land:                                  # high drops start above the ceiling
            above = above & self.landed
        lost = self.p_ball[:, 2] < fk["paddle_o"][:, 2] - 0.04
        if c.handle_keepout > 0:
            lost = lost | ((self.p_ball[:, 0] - fk["paddle_o"][:, 0] < -c.handle_keepout) & (self.rest_time <= 0)
                           & (self.p_ball[:, 2] < fk["paddle_o"][:, 2] + 0.04))
        conds = [lost,
                 above | (lateral > c.max_lateral),
                 (po[:, :2].abs() > c.paddle_box[0]).any(-1) | (po[:, 2].abs() > c.paddle_box[1]) | (tilt > c.max_tilt),
                 self.overload_time > self.overload_limit,
                 self.rest_time > c.max_rest_s]
        term = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        for i, cond in enumerate(conds):
            reason = torch.where(cond & ~term, torch.full_like(reason, i), reason)
            term = term | cond
        trunc = (self.steps >= self.max_steps) & ~term
        r["fail"] = -c.w_fail * term.float()
        reward = sum(r.values())
        # bookkeeping
        self.a_hist = torch.cat([a[:, None], self.a_hist[:, :-1]], 1)
        qm = self._measure_q()
        qm = torch.where(glitch[:, None], self.q_hist[:, 0] + self.q_ref, qm)      # stale reading during a hiccup
        self.q_hist = torch.cat([(qm - self.q_ref)[:, None], self.q_hist[:, :-1]], 1)
        self.glitch_left = (self.glitch_left - 1).clamp(min=0)
        self.ep["ret"] += reward
        self.ep["bounces"] += ev["n_apex"]
        self.ep["lateral"] += lateral
        for k, v in r.items():
            self.ep.setdefault("r_" + k, torch.zeros(self.N, device=self.device))
            self.ep["r_" + k] += v
        obs = self._obs()
        done = term | trunc
        self.final_obs, self.final_critic_obs = obs["policy"].clone(), obs["critic"].clone()
        if bool(done.any()):
            ids = done.nonzero().squeeze(-1)
            self._record(ids, term, reason)
            self._reset(ids)
            obs = self._obs()
        return obs, reward, term, trunc, {}

    def _camera(self):
        """One possible frame per env: detection error across/along each camera's line of sight, calibration
        rotation and bias, motion-blur shift, misses (random, bursts, near the paddle, long streaks), false detections
        (more while a hand is in view), and a per-session error in the estimated capture time."""
        due = self.t >= self.next_cap
        n, p = self.N, self.p_ball
        side = self.cam_side[:, None]
        ray_o = p - self.cam_o
        dist_o = ray_o.norm(dim=-1, keepdim=True)
        ray_o = ray_o / dist_o
        ray_s = p - self.cam_s
        dist_s = ray_s.norm(dim=-1, keepdim=True)
        ray_s = ray_s / dist_s
        err = self._n(n, 3) * self.sig_perp[:, None]
        rho = torch.exp(-self.cam_period / 0.3)
        self.depth_drift = torch.where(due, rho * self.depth_drift + (1 - rho * rho).sqrt() * 0.7 * self.sig_ray *
                                       self._n(n), self.depth_drift)
        white = torch.where(side, 1.0, 0.7)[:, 0] * self.sig_ray
        drift = torch.where(side[:, 0], torch.zeros_like(self.depth_drift), self.depth_drift)
        err = err + (self._n(n, 1) * white[:, None] + drift[:, None] +
                     torch.where(side, torch.zeros_like(dist_o), self.depth_err[:, None] * dist_o)) * ray_o
        err = err + torch.where(side, (self._n(n, 1) * self.sig_ray[:, None] + self.depth_err[:, None] * dist_s) *
                                ray_s, torch.zeros_like(err))
        if self.cfg.depth_speed_k or self.cfg.radius_outlier_p:   # overhead only: blur and bad radii in flight
            speed = self.v_ball.norm(dim=-1, keepdim=True)
            extra = self.cfg.depth_speed_k * speed * self._n(n, 1)
            flying = (p[:, 2:3] - B.RADIUS - self.o_prev[:, 2:3]) > 0.015
            bad = flying & (self._u(n, 1) < self.cfg.radius_outlier_p)
            extra = extra + bad * torch.sign(self._n(n, 1)) * (0.05 + 0.05 * self._u(n, 1))
            err = err + torch.where(side, torch.zeros_like(extra), extra) * ray_o
        rel = (self.cam_R @ (p - self.home_env).unsqueeze(-1)).squeeze(-1)
        meas = self.home_env + rel + self.cam_bias + err + self.blur[:, None] * self.v_ball
        if self.cfg.cam_height_gain:
            flying = (p[:, 2] - B.RADIUS - self.o_prev[:, 2]) > 0.015
            hz = self.home_env[:, 2] + self.h_gain * (meas[:, 2] - self.home_env[:, 2])
            meas = torch.cat([meas[:, :2], torch.where(flying, hz, meas[:, 2])[:, None]], -1)
        p_out = self.outlier_p_env + self.cfg.hand_outlier_p * self.diff_g[:, 1] * (self.hand_t > 0).float()
        wild = self._u(n) < p_out
        dirn = self._n(n, 3)
        dirn = dirn / dirn.norm(dim=-1, keepdim=True)
        meas = meas + wild[:, None] * dirn * (0.05 + 0.15 * self._u(n))[:, None]
        # misses
        flip_in = due & ~self.cam_bad & (self._u(n) < self.burst_in)
        flip_out = due & self.cam_bad & (self._u(n) < self.burst_out)
        self.cam_bad = (self.cam_bad | flip_in) & ~flip_out
        near = (p[:, 2] - B.RADIUS - self.o_prev[:, 2]) < 0.015
        streak = self.v_ball.norm(dim=-1) * self.exposure
        miss = (self._u(n) < self.cam_drop) | (self.cam_bad & (self._u(n) < 0.9)) | \
            (near & (self._u(n) < self.occl_p)) | ((streak > 0.03) & (self._u(n) < self.blurmiss_p))
        if self.cfg.view_top_p > 0:                  # above the camera's top frame edge
            miss = miss | ((p[:, 2] - self.home_env[:, 2]) > self.view_top_env)
        if self.cfg.cam_view_p > 0:                  # outside the real frame or the runner's search window
            f, W, H = self.cfg.cam_view[12:15]
            pc = ((p - self.view_t)[:, None] @ self.view_R)[:, 0]
            u, v = f * pc[:, 0] / pc[:, 2] + W / 2, f * pc[:, 1] / pc[:, 2] + H / 2
            r = self.view_roi
            miss = miss | (self.view_on & ((u < r[:, 0]) | (u > r[:, 2]) | (v < r[:, 1]) | (v > r[:, 3])))
        seen = due & ~miss
        if self.held_on:
            hold = self.hold_left > 0
            seen = seen & ~(hold & self.held_hidden)
            fake = hold & ~self.held_hidden & (self._u(n) < self.cfg.held_outlier_p)
            d = self._n(n, 3)
            d[:, 2] = d[:, 2].abs()
            d = d / d.norm(dim=-1, keepdim=True)
            meas = torch.where(fake[:, None], p + d * (0.03 + 0.05 * self._u(n))[:, None], meas)
        stamp = self.t + self.age_bias_env + self.stamp_sd * self._n(n)
        if self.cfg.rolling_shutter:
            stamp = stamp + self.rs_env * (p[:, 0] - self.home_env[:, 0])
        ar = torch.arange(n, device=self.device)
        ptr = self.cam_ptr
        self.cam_pos[ar, ptr] = torch.where(seen[:, None], meas, self.cam_pos[ar, ptr])
        self.cam_t[ar, ptr] = torch.where(seen, stamp, self.cam_t[ar, ptr])
        self.cam_ptr = torch.where(seen, (ptr + 1) % CAM_RING, ptr)
        jitter = 1 + 0.03 * self._n(n)
        self.next_cap = torch.where(due, self.next_cap + self.cam_period * jitter, self.next_cap)

    def _auto_level(self):
        """The runner's --auto-level, per env: the newest detection the controller has (latency applied) updates an
        EMA of the ball's offset from the paddle centre (nominal FK of the measured joints); every al_period steps,
        if the drift exceeds the deadband, solve J dq = -al_gain * drift * period for wrist flex / roll (J: face normal
        xy per joint, nominal model) and move the hold command (and so the policy's joint reference) by dq, within
        +-al_clip of the hold command the episode started with."""
        c, N = self.cfg, self.N
        avail = self.cam_t <= (self.t - self.cam_lat)[:, None]
        top, idx = torch.where(avail, self.cam_t, torch.full_like(self.cam_t, -1e9)).max(dim=1)
        new = (top > -1e8) & (top > self.al_last)
        pos = self.cam_pos[torch.arange(N, device=self.device), idx]
        po = self.kin_nominal.forward(self.q_hist[:, 0] + self.q_ref)["paddle_o"]
        off = pos[:, :2] - po[:, :2]
        upd = (new & (off.norm(dim=-1) < c.al_radius))[:, None]
        self.al_drift = torch.where(upd, (1 - c.al_alpha) * self.al_drift + c.al_alpha * off, self.al_drift)
        self.al_last = torch.where(new, top, self.al_last)
        due = ((self.steps % self.al_period) == 0) & (self.al_drift.norm(dim=-1) > c.al_deadband)
        if not bool(due.any()):
            return
        qe = self.q_ref
        n0 = self.kin_nominal.forward(qe)["paddle_R"][:, :2, 2]
        cols = []
        for j in (3, 4):
            dq = torch.zeros_like(qe)
            dq[:, j] = 1e-3
            cols.append((self.kin_nominal.forward(qe + dq)["paddle_R"][:, :2, 2] - n0) / 1e-3)
        Jm = torch.stack(cols, -1)
        step = torch.linalg.solve(Jm, (-c.al_gain * self.al_period / c.ctrl_hz * self.al_drift)[..., None])[..., 0]
        trim = (qe[:, 3:5] - self.al_qref0[:, 3:5] + step).clamp(-c.al_clip, c.al_clip)
        self.q_ref[:, 3:5] = torch.where(due[:, None], self.al_qref0[:, 3:5] + trim, qe[:, 3:5])

    def _launch_steer(self, off, vel):
        """(N, 2) wrist flex / roll actions for the strike: bounce/launch.py steer() at the nominal hold pose."""
        from bounce import launch as LAUNCH
        if self._steer_J is None:
            kin1 = ArmKinematics(self.model, 1, self.device)
            self._steer_J = LAUNCH.steer_jacobian(kin1, self.q_nom.reshape(1, J)).to(self.device)
        p = self.launch_base
        if self.cfg.launch_mix:                      # each replayed launch steers with its own gains
            want = -(self.launch_sp[self.launch_var][:, None] * off + self.launch_sv[self.launch_var][:, None] * vel)
        else:
            want = -(p.steer_p * off + p.steer_v * vel)
        dq = torch.linalg.solve(self._steer_J, want.T).T
        return (dq / self.scale[3:5]).clamp(-self.cfg.launch_max_action, self.cfg.launch_max_action)

    # ------------------------------------------------------------------ observations
    def _obs(self):
        N = self.N
        qm = self.q_hist[:, 0] + self.q_ref
        fk = self.kin_nominal.forward(qm)        # FK of measured angles with the nominal paddle mount, as on the arm
        po, n = fk["paddle_o"] - self.home_env, fk["paddle_R"][..., 2]
        avail = self.cam_t <= (self.t - self.cam_lat)[:, None]
        score = torch.where(avail, self.cam_t, torch.full_like(self.cam_t, -1e9))
        K = self.cfg.cam_k
        top, idx = score.topk(K, dim=1)
        valid = top > -1e8
        pos = torch.gather(self.cam_pos, 1, idx[..., None].expand(N, K, 3)) - self.home_env[:, None]
        age = (self.t[:, None] - top).clamp(0, 0.5)
        pos = torch.where(valid[..., None], pos, torch.zeros_like(pos))
        age = torch.where(valid, age, torch.full_like(age, 0.5))
        latest_rel = torch.where(valid[:, :1], pos[:, 0] - po, torch.zeros_like(po))
        policy = torch.cat([2.0 * self.q_hist.reshape(N, -1), self.a_hist.reshape(N, -1), 10 * po, n,
                            10 * pos.reshape(N, -1), 10 * age, valid.float(), 10 * latest_rel,
                            10 * self.target_apex[:, None], self.cam_side.float()[:, None]], -1)
        p = self.servo.p
        priv = torch.cat([10 * (self.p_ball - self.home_env), self.v_ball, 0.05 * self.w_ball, 0.2 * self.qd,
                          10 * (self.q - self.q_ref),
                          torch.stack([p["vin"][:, 0] / 7.4, self.delay_sub * self.sub_dt * 50, self.cam_lat * 20,
                                       self.cam_period * 30, self.ball_prm["e0"], self.ball_prm["mu"] * 3,
                                       self.kin.mass[:, self.kin.paddle] * 5, self.diff,
                                       (self.rest_time > 0.1).float(), self.since_impact.clamp(max=1.0),
                                       self.overload_time], -1),
                          p["kt"] / 1.18, p["armature"] * 40, p["log_max_accel"] / 3.0,
                          torch.stack([10 * self.blade[:, 0], 10 * self.blade[:, 1], self.blade[:, 2] / 2.5], -1),
                          3 * self.wind[:, :2], 3 * self.g_env[:, :2],
                          torch.stack([30 * self.exposure, 100 * p["backlash"].mean(1), p["strength"].mean(1)], -1)],
                         -1)
        if self.cfg.critic_rig:
            priv = torch.cat([priv, torch.stack([self.strike_k, self.tilt_g, 10 * self.ball_prm["kick"],
                                                 20 * self.tip_env], -1)], -1)
        return {"policy": policy, "critic": torch.cat([policy, priv], -1)}

    @property
    def level(self):
        return sum(self.levels) / len(self.levels)

    @level.setter
    def level(self, v):
        self.levels = [float(v)] * 4

    @property
    def obs_dim(self):
        return 2 * self.cfg.history * J + 3 + 3 + 5 * self.cfg.cam_k + 3 + 1 + 1

    @property
    def critic_dim(self):
        return self.obs_dim + 3 + 3 + 3 + J + J + 11 + 3 * J + 3 + 4 + 3 + (4 if self.cfg.critic_rig else 0)

    # ------------------------------------------------------------------ statistics and curriculum
    def _record(self, ids, term, reason):
        steps = self.steps[ids].float().clamp(min=1)
        rec = {"length_s": steps * self.dt, "survived": (~term[ids]).float(), "reason": reason[ids].float(),
               "rest_start": self.rest_start[ids].float(), "difficulty": self.diff[ids], "env_id": ids.float(),
               "diff_g": self.diff_g[ids].clone(), "boundary_g": self.boundary_g[ids].float()}
        for k, v in self.ep.items():
            val = v[ids].clone()
            if k == "apex_err":
                val = val / self.ep["bounces"][ids].clamp(min=1)
            elif k == "lateral":
                val = val / steps
            rec[k] = val
        rec["launched"] = (~torch.isnan(self.ep["launch_t"][ids])).float()
        self.completed.append(rec)

    def pop_completed(self):
        if not self.completed:
            return None
        out = {k: torch.cat([r[k] for r in self.completed]) for k in self.completed[0]}
        self.completed = []
        return out

    def update_curriculum(self, stats, buffer: int = 256):
        """Per-group ADR with boundary sampling (OpenAI 2019): episodes that put one group exactly at its level are
        collected per group; once a group has `buffer` of them, its level moves up if they mostly survive and down
        if they mostly fail, and the buffer is cleared."""
        if stats is None or self.forced is not None:
            return
        for g in range(4):
            m = stats["boundary_g"] == g
            self.adr_buf[g].extend(stats["survived"][m].tolist())
            if len(self.adr_buf[g]) >= buffer:
                perf = sum(self.adr_buf[g]) / len(self.adr_buf[g])
                if perf > self.cfg.adr_up:
                    self.levels[g] = min(1.0, self.levels[g] + self.cfg.adr_step)
                elif perf < self.cfg.adr_down:
                    self.levels[g] = max(self.cfg.adr_floor[g] if self.cfg.adr_floor else 0.0, self.levels[g] - self.cfg.adr_step)
                self.adr_buf[g] = []
