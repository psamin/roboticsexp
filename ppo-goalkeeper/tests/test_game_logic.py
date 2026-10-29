"""Shot sampling, outcome rules, the bounded drive, the curriculum, and the analytical keeper on an idealised
point-mass model. CPU only. The same checks against real Isaac physics live in scripts/verify_env.py."""
import torch

from goalkeeper.controllers.baselines import AnalyticalKeeper
from goalkeeper.physics import field as F
from goalkeeper.physics import observation, outcomes
from goalkeeper.physics.drive import DriveConfig, step_reference
from goalkeeper.physics.shots import LEVELS, rolling_spin, sample_shots
from goalkeeper.rl.curriculum import Curriculum


def test_shots_hit_their_targets_and_respect_ranges():
    g = torch.Generator().manual_seed(0)
    for i, lv in enumerate(LEVELS):
        s = sample_shots(torch.full((5000,), i), g)
        speed = s["vel"].norm(dim=-1)
        assert speed.min() >= lv.speed[0] - 1e-5 and speed.max() <= lv.speed[1] + 1e-5
        # straight line from the start point through the aim point on the goal line (x = 0)
        t = -s["start"][:, 0] / s["vel"][:, 0]
        y_at_line = s["start"][:, 1] + s["vel"][:, 1] * t
        assert torch.allclose(y_at_line, s["target_y"], atol=1e-4)
        on = s["on_target"]
        assert (s["target_y"][on].abs() <= lv.target_half_width + 1e-6).all()
        assert (s["target_y"][~on].abs() >= 1.75).all()
        assert (s["keeper_y"].abs() <= lv.keeper_start + 1e-6).all()
    assert sample_shots(torch.zeros(1000, dtype=torch.long), g)["on_target"].all()


def test_rolling_spin_has_zero_contact_slip():
    v = torch.tensor([[-4.0, 0.7]])
    w = rolling_spin(v)
    contact_vel = v + torch.linalg.cross(w, torch.tensor([[0.0, 0.0, -F.BALL_RADIUS]]))[:, :2]
    assert torch.allclose(contact_vel, torch.zeros(1, 2), atol=1e-6)


def test_outcome_rules():
    r = F.BALL_RADIUS
    pos = torch.tensor([
        [-r - 0.01, 0.5, r],   # in the net                       -> goal
        [-r - 0.01, 0.5, r],   # in the net, keeper touched first -> goal (deflected in)
        [-r - 0.01, 2.0, r],   # past the line, wide, untouched   -> miss
        [-r - 0.01, 2.0, r],   # past the line, wide, touched     -> save
        [0.9, 0.3, r],         # rebounding off the keeper        -> save
        [0.3, 1.4, r],         # rebounding off the post          -> miss
        [3.0, 0.0, r],         # still travelling                 -> none
        [-r + 0.02, 0.0, r],   # on the line, not wholly over     -> none
    ])
    vel = torch.tensor([[-3.0, 0], [-3.0, 0], [-3.0, 0], [-3.0, 0], [2.0, 0], [1.0, 0.2], [-4.0, 0], [-4.0, 0]])
    touched = torch.tensor([False, True, False, True, True, False, False, False])
    got = outcomes.classify(pos, torch.cat([vel, torch.zeros(8, 1)], 1), touched)
    G, S, M, N = outcomes.GOAL, outcomes.SAVE, outcomes.MISS, outcomes.NONE
    assert got.tolist() == [G, G, M, S, S, M, N, N]


def test_drive_respects_speed_accel_and_travel_limits():
    cfg = DriveConfig()
    dt = 1 / 120
    y, v = torch.zeros(3), torch.zeros(3)
    a = torch.tensor([1.0, -1.0, 0.3])
    ys, vs = [], []
    for _ in range(600):
        v_new = step_reference(a, y, v, dt, cfg)
        assert ((v_new - v).abs() <= cfg.amax * dt + 1e-6).all()
        v = v_new
        y = y + v * dt
        ys.append(y.clone()); vs.append(v.clone())
    ys, vs = torch.stack(ys), torch.stack(vs)
    assert vs.abs().max() <= cfg.vmax + 1e-6
    assert ys.abs().max() <= cfg.y_limit + 0.02          # brakes in time at full command
    assert abs(ys[-1, 0] - cfg.y_limit) < 0.05 and abs(ys[-1, 1] + cfg.y_limit) < 0.05


def test_curriculum_promotes_on_save_rate():
    cur = Curriculum(3, thresholds=(0.8, 0.8), window=100)
    lv = torch.zeros(100, dtype=torch.long)
    on = torch.ones(100, dtype=torch.bool)
    assert cur.record(lv, torch.full((100,), outcomes.GOAL), on) is None and cur.level == 0
    assert cur.record(lv, torch.full((100,), outcomes.SAVE), on) == 1
    mix = cur.sample_levels(10000, "cpu")
    assert set(mix.unique().tolist()) == {0, 1} and (mix == 1).float().mean() > 0.7


def kinematic_episode(level: int, n: int, policy, seed=0):
    """Point-mass ball (constant velocity) and point-mass keeper behind the same bounded drive."""
    g = torch.Generator().manual_seed(seed)
    s = sample_shots(torch.full((n,), level), g)
    cfg, dt, decim = DriveConfig(), 1 / 120, 4
    b = s["start"].clone(); bv = s["vel"].clone()
    ky, v = s["keeper_y"].clone(), torch.zeros(n)
    saved, done = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    prev_a = torch.zeros(n)
    for _ in range(200):
        obs = observation.build(b, bv, ky, v, prev_a)
        a = policy(obs).squeeze(-1)
        for _ in range(decim):
            v = step_reference(a, ky, v, dt, cfg)
            ky = ky + v * dt
            b = b + bv * dt
            hit = (~done) & (b[:, 0] <= F.CONTACT_X) & ((b[:, 1] - ky).abs() <= F.PANEL_HALF_WIDTH + F.BALL_RADIUS)
            saved |= hit
            done |= b[:, 0] <= F.CONTACT_X
        prev_a = a
    return saved[s["on_target"]].float().mean().item()


def test_analytical_keeper_saves_beginner_shots_in_kinematic_model():
    assert kinematic_episode(0, 2000, AnalyticalKeeper()) >= 0.98
    assert kinematic_episode(1, 2000, AnalyticalKeeper()) >= 0.95
