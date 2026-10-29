"""Isaac Lab direct-workflow environment: a Jackal goalkeeper against rolling penalty shots.

Physics runs at 120 Hz; the policy acts at 30 Hz (decimation 4). Each episode is one shot.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ..physics import field as F
from ..physics import observation, outcomes, rewards
from ..physics.drive import DriveConfig, step_reference
from ..physics.shots import LEVELS, reachable, rolling_spin, sample_shots
from ..rl.curriculum import Curriculum
from ..robots import keeper as K


@configclass
class GoalkeeperEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 4.0
    action_space = 1
    observation_space = observation.OBS_DIM
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=4,
        physx=PhysxCfg(bounce_threshold_velocity=0.3, gpu_max_rigid_patch_count=2 ** 20),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=10.0, replicate_physics=True)

    asset_paths: dict = {}          # from environments.assets.build_all()
    render_scene: bool = False      # turf, field lines, sky, goal mesh (rendering only; physics is identical)
    sky: str = "puresky"            # "puresky", "studio" or "photo" (see scene_visuals.build_field)

    # drive (DriveConfig) and rewards (RewardConfig)
    vmax: float = 2.0
    amax: float = 6.0
    heading_gain: float = 4.0       # 1/s, yaw hold
    lateral_gain: float = 1.0       # rad/m, steer back onto the keeper line while moving
    reward_save: float = 10.0
    reward_goal: float = -10.0
    reward_progress: float = 1.0
    reward_action: float = 0.01
    reward_jerk: float = 0.02
    gamma: float = 0.99

    # shots
    level: int = -1                 # fixed shot level; -1 = curriculum
    curriculum_thresholds: tuple = (0.85, 0.85)
    curriculum_window: int = 2048
    kick_delay_s: float = 0.0       # ball rests on the mark this long before the kick (playback/screenshots)
    hold_after_outcome_s: float = 0.0   # keep simulating after the result (playback), no reward
    contact_force_threshold: float = 0.5
    tyre_friction: tuple = (0.9, 0.85)     # static, dynamic: rubber on dry turf


class GoalkeeperEnv(DirectRLEnv):
    cfg: GoalkeeperEnvCfg

    def __init__(self, cfg: GoalkeeperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.drive = DriveConfig(cfg.vmax, cfg.amax, F.KEEPER_Y_LIMIT)
        self.rcfg = rewards.RewardConfig(cfg.reward_save, cfg.reward_goal, cfg.reward_progress, cfg.reward_action,
                                         cfg.reward_jerk, cfg.gamma)
        names = self.robot.joint_names
        self.wheel_ids = [i for i, j in enumerate(names) if "wheel" in j]
        self.wheel_side = torch.tensor([1.0 if "left" in names[i] else -1.0 for i in self.wheel_ids], device=dev)
        self.wheel_targets = torch.zeros(n, self.robot.num_joints, device=dev)
        self.v_ref = torch.zeros(n, device=dev)
        self.action = torch.zeros(n, 1, device=dev)
        self.last_action = torch.zeros(n, 1, device=dev)
        self.prev_policy_action = torch.zeros(n, 1, device=dev)
        self.applied = torch.zeros(n, 1, device=dev)
        self.prev_applied = torch.zeros(n, 1, device=dev)
        self.touched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.touched_geom = torch.zeros(n, dtype=torch.bool, device=dev)
        self.outcome = torch.zeros(n, dtype=torch.long, device=dev)
        self.decided_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.just_decided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.kicked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.y_star = torch.zeros(n, device=dev)
        self.phi = torch.zeros(n, device=dev)
        self.shot = {k: v for k, v in sample_shots(torch.zeros(n, dtype=torch.long, device=dev)).items()}
        self.shot["reachable"] = torch.ones(n, dtype=torch.bool, device=dev)
        self.term_names = ("save", "goal", "progress", "action", "jerk")
        self.ep_sums = {k: torch.zeros(n, device=dev) for k in self.term_names}
        self.kick_steps = int(round(cfg.kick_delay_s / self.step_dt))
        self.hold_steps = int(round(cfg.hold_after_outcome_s / self.step_dt))
        self.curriculum = Curriculum(len(LEVELS), cfg.curriculum_thresholds, cfg.curriculum_window) \
            if cfg.level < 0 else None
        self.forced_shots: dict | None = None
        self.record_episodes = True
        self._completed: list[dict] = []
        self.ball_speed_at_line = torch.zeros(n, device=dev)

    # ----------------------------------------------------------------- scene
    def _setup_scene(self):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.robot = Articulation(K.keeper_cfg())
        K.add_panel(stage, "/World/envs/env_0/Keeper")
        before = K.set_tyre_friction(stage, "/World/envs/env_0/Keeper", self.cfg.tyre_friction[0], self.cfg.tyre_friction[1])
        print(f"[env] tyre friction set to {self.cfg.tyre_friction} (asset values were {before})", flush=True)
        if self.cfg.render_scene:
            K.make_visuals_renderable(stage, "/World/envs/env_0/Keeper")
        ball_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Ball",
            spawn=sim_utils.UsdFileCfg(usd_path=self.cfg.asset_paths["ball"], activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(F.PENALTY_X, 0.0, F.BALL_RADIUS)),
        )
        self.ball = RigidObject(ball_cfg)
        goal = sim_utils.UsdFileCfg(usd_path=self.cfg.asset_paths["goal_physics"])
        goal.func("/World/envs/env_0/GoalPhysics", goal)
        if self.cfg.render_scene:
            vis = sim_utils.UsdFileCfg(usd_path=self.cfg.asset_paths["goal_visual"])
            vis.func("/World/envs/env_0/GoalVisual", vis)
        # infinite plane collider (wheel cylinders lose contact against a huge box collider); turf friction
        ground = sim_utils.GroundPlaneCfg(
            visible=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.85,
                                                            restitution=0.05, restitution_combine_mode="min"),
        )
        ground.func("/World/ground", ground)
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Ball",
            filter_prim_paths_expr=["/World/envs/env_.*/Keeper/goal_panel", "/World/envs/env_.*/Keeper/base_link"],
        ))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["keeper"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.sensors["contact"] = self.contact
        if self.cfg.render_scene:
            from .scene_visuals import build_field
            build_field(stage, self.cfg.asset_paths, self.scene.env_origins[0].tolist(), self.cfg.sky)
        else:
            light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
            light.func("/World/Light", light)

    # ----------------------------------------------------------------- control
    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        lat = self.shot["latency"].bool().unsqueeze(-1)
        self.prev_applied = self.applied.clone()
        self.applied = torch.where(lat, self.prev_policy_action, a)
        self.prev_policy_action = a.clone()
        self.last_action = self.action.clone()
        self.action = a
        # pre-kick pause (playback only): keep the ball still, then launch it
        if self.kick_steps > 0:
            kick = (~self.kicked) & (self.episode_length_buf >= self.kick_steps)
            if kick.any():
                self._launch(kick.nonzero().squeeze(-1))

    def _keeper_state(self):
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        q = self.robot.data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return pos, yaw, self.robot.data.root_lin_vel_w

    def _apply_action(self):
        pos, yaw, _ = self._keeper_state()
        self._accumulate_touch()
        self.v_ref = step_reference(self.applied.squeeze(-1), pos[:, 1], self.v_ref, self.physics_dt, self.drive)
        # heading hold, with a small steer back onto the keeper line while moving
        steer = (self.cfg.lateral_gain * (pos[:, 0] - F.KEEPER_X) * torch.sign(self.v_ref)).clamp(-0.15, 0.15)
        yaw_err = torch.atan2(torch.sin(yaw - math.pi / 2 - steer), torch.cos(yaw - math.pi / 2 - steer))
        wz = -self.cfg.heading_gain * yaw_err
        v_side = self.v_ref.unsqueeze(-1) - self.wheel_side * wz.unsqueeze(-1) * K.TRACK_WIDTH / 2
        self.wheel_targets[:, self.wheel_ids] = v_side / K.WHEEL_RADIUS
        self.robot.set_joint_velocity_target(self.wheel_targets)

    def _accumulate_touch(self):
        f = self.contact.data.force_matrix_w            # (N, 1, M, 3)
        if f is not None:
            self.touched |= f.norm(dim=-1).sum(dim=(1, 2)) > self.cfg.contact_force_threshold
        # geometric cross-check: ball surface within 2 cm of the panel's front face, inside its width
        b = self.ball.data.root_pos_w - self.scene.env_origins
        k = self.robot.data.root_pos_w - self.scene.env_origins
        near = ((b[:, 0] - F.BALL_RADIUS) - F.PANEL_FRONT_X - (k[:, 0] - F.KEEPER_X)).abs() < 0.03
        within = (b[:, 1] - k[:, 1]).abs() < F.PANEL_HALF_WIDTH + 0.03
        self.touched_geom |= near & within & self.kicked

    # ----------------------------------------------------------------- MDP
    def _ball_state(self):
        return self.ball.data.root_pos_w - self.scene.env_origins, self.ball.data.root_lin_vel_w

    def _get_observations(self) -> dict:
        bp, bv = self._ball_state()
        kp, _, kv = self._keeper_state()
        noise_p = self.shot["obs_noise_pos"].unsqueeze(-1)
        noise_v = self.shot["obs_noise_vel"].unsqueeze(-1)
        bxy = bp[:, :2] + noise_p * torch.randn_like(bp[:, :2])
        bvxy = bv[:, :2] + noise_v * torch.randn_like(bv[:, :2])
        obs = observation.build(bxy, bvxy, kp[:, 1], kv[:, 1], self.action.squeeze(-1))
        return {"policy": obs}

    def _get_dones(self):
        self._accumulate_touch()
        bp, bv = self._ball_state()
        live = (self.outcome == outcomes.NONE) & self.kicked
        new = torch.where(live, outcomes.classify(bp, bv, self.touched), torch.zeros_like(self.outcome))
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        new = torch.where((self.outcome == outcomes.NONE) & (new == outcomes.NONE) & timeout,
                          torch.full_like(new, outcomes.TIMEOUT), new)
        self.just_decided = new != outcomes.NONE
        self.outcome = torch.where(self.just_decided, new, self.outcome)
        self.decided_step = torch.where(self.just_decided, self.episode_length_buf, self.decided_step)
        crossing = self.just_decided & (bp[:, 0] < 0)
        self.ball_speed_at_line = torch.where(crossing, bv[:, :2].norm(dim=-1), self.ball_speed_at_line)
        held = (self.episode_length_buf - self.decided_step) >= self.hold_steps
        decided = self.outcome != outcomes.NONE
        terminated = decided & (self.outcome != outcomes.TIMEOUT) & held
        time_out = (self.outcome == outcomes.TIMEOUT) & held
        if self.hold_steps > 0:
            time_out |= self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        bp, bv = self._ball_state()
        kp, _, _ = self._keeper_state()
        y_new, approaching = rewards.intercept_y(bp[:, :2], bv[:, :2])
        self.y_star = torch.where(approaching, y_new, self.y_star)
        active = (self.outcome == outcomes.NONE) | self.just_decided
        phi_now = torch.where(self.just_decided | ~self.kicked, torch.zeros_like(self.phi),
                              rewards.potential(kp[:, 1], self.y_star, torch.ones_like(approaching)))
        terms = rewards.compute(self.rcfg, torch.where(self.just_decided, self.outcome, torch.zeros_like(self.outcome)),
                                self.phi, phi_now, self.action, self.last_action)
        self.phi = phi_now
        total = torch.zeros(self.num_envs, device=self.device)
        for k in self.term_names:
            v = torch.where(active & self.kicked, terms[k], torch.zeros_like(terms[k]))
            self.ep_sums[k] += v
            total += v
        return total

    # ----------------------------------------------------------------- resets
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._record(env_ids)
        super()._reset_idx(env_ids)
        n = env_ids.numel()
        if self.forced_shots is not None:
            shots = {k: v[env_ids] for k, v in self.forced_shots.items()}
        else:
            levels = self.curriculum.sample_levels(n, self.device) if self.curriculum is not None \
                else torch.full((n,), self.cfg.level, dtype=torch.long, device=self.device)
            shots = sample_shots(levels)
            shots["reachable"] = reachable(shots, self.cfg.vmax, self.cfg.amax, self.step_dt, self.step_dt)
        for k, v in shots.items():
            self.shot[k][env_ids] = v.to(self.shot[k].dtype)

        origins = self.scene.env_origins[env_ids]
        # keeper: upright, facing +y, at its start position, wheels still
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0] = origins[:, 0] + F.KEEPER_X
        root[:, 1] = origins[:, 1] + shots["keeper_y"]
        root[:, 2] = origins[:, 2] + K.SPAWN_Z
        root[:, 3:7] = torch.tensor(K.YAW_90, device=self.device)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)
        self.wheel_targets[env_ids] = 0.0
        # ball: resting on the mark, random spin orientation, launched now unless there is a kick delay
        ball = self.ball.data.default_root_state[env_ids].clone()
        ball[:, 0] = origins[:, 0] + shots["start"][:, 0]
        ball[:, 1] = origins[:, 1] + shots["start"][:, 1]
        ball[:, 2] = origins[:, 2] + F.BALL_RADIUS + 0.002
        yaw = torch.rand(n, device=self.device) * 2 * math.pi
        ball[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        ball[:, 7:] = 0.0
        self.ball.write_root_pose_to_sim(ball[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball[:, 7:], env_ids)
        self.kicked[env_ids] = False
        if self.kick_steps == 0:
            self._launch(env_ids)

        for buf in (self.v_ref, self.y_star, self.phi, self.ball_speed_at_line):
            buf[env_ids] = 0.0
        for buf in (self.action, self.last_action, self.prev_policy_action, self.applied, self.prev_applied):
            buf[env_ids] = 0.0
        self.touched[env_ids] = False
        self.touched_geom[env_ids] = False
        self.outcome[env_ids] = outcomes.NONE
        self.decided_step[env_ids] = 0
        for k in self.term_names:
            self.ep_sums[k][env_ids] = 0.0
        self.started[env_ids] = True
        sx, sy = shots["start"][:, 0], shots["start"][:, 1]
        self.y_star[env_ids] = (sy + shots["vel"][:, 1] * (sx - F.CONTACT_X) / -shots["vel"][:, 0]).clamp(
            -F.KEEPER_Y_LIMIT, F.KEEPER_Y_LIMIT)
        self.phi[env_ids] = torch.where(self.kicked[env_ids], -(shots["keeper_y"] - self.y_star[env_ids]).abs(),
                                        torch.zeros(n, device=self.device))

    def _launch(self, env_ids):
        v = torch.zeros(env_ids.numel(), 6, device=self.device)
        v[:, :2] = self.shot["vel"][env_ids]
        v[:, 3:6] = rolling_spin(v[:, :2])
        self.ball.write_root_velocity_to_sim(v, env_ids)
        self.kicked[env_ids] = True

    def _record(self, env_ids):
        """Store finished episodes (outcome, shot, reward sums) for the training/eval scripts."""
        mask = self.started[env_ids] & (self.outcome[env_ids] != outcomes.NONE)
        if not self.record_episodes or not mask.any():
            return
        ids = env_ids[mask]
        rec = {
            "env_id": ids.clone(),
            "outcome": self.outcome[ids].clone(),
            "level": self.shot["level"][ids].clone(),
            "on_target": self.shot["on_target"][ids].clone(),
            "reachable": self.shot["reachable"][ids].clone(),
            "touched_contact": self.touched[ids].clone(),
            "touched_geom": self.touched_geom[ids].clone(),
            "length": self.episode_length_buf[ids].clone(),
            "speed": self.shot["speed"][ids].clone(),
            "ball_speed_at_line": self.ball_speed_at_line[ids].clone(),
        }
        for k in self.term_names:
            rec[f"r_{k}"] = self.ep_sums[k][ids].clone()
        self._completed.append(rec)
        if self.curriculum is not None and self.forced_shots is None:
            self.curriculum.record(rec["level"], rec["outcome"], rec["on_target"], int(self.common_step_counter))

    def pop_completed(self) -> dict | None:
        if not self._completed:
            return None
        out = {k: torch.cat([r[k] for r in self._completed]) for k in self._completed[0]}
        self._completed = []
        return out
