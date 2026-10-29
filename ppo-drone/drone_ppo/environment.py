"""Crazyflie hover / target-reaching environment (Isaac Lab direct workflow).

Adapted from Isaac Lab's Isaac-Quadcopter-Direct-v0 (BSD-3, isaaclab_tasks/direct/quadcopter): same Crazyflie
asset, same 100 Hz physics / 50 Hz control, same collective-thrust + body-torque interface applied as an external
wrench on the drone body. The drone only moves through those forces; its pose is never set during an episode.

Changes for this project: hover-centred thrust mapping, body-frame observations with an action history, a
three-stage task curriculum, two reward modes (the shaped reward in task.reward_terms, or the "Learning to Fly in
Seconds" squared-cost reward with its penalty curriculum), unstable-flight penalties, optional first-order motor lag,
random force/torque disturbances and privileged critic inputs, crash/out-of-bounds failures kept separate from time
limits, the pre-reset observation kept for bootstrapping truncated episodes, and stability metrics per episode.
"""
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older Isaac Lab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from . import task as T
from .curriculum import Curriculum

FLIGHT_MATERIAL = sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply", restitution_combine_mode="multiply",
                                                 static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
SHAPED_TERMS = ("distance", "precision", "lin_vel", "ang_vel", "upright", "effort", "action_rate", "ang_acc",
                "lin_acc", "success", "crash")
L2F_TERMS = ("l2f_constant", "l2f_position", "l2f_orientation", "l2f_lin_vel", "l2f_ang_vel", "l2f_action",
             "l2f_clip", "ang_acc", "lin_acc", "success", "crash")
PRIVILEGED_DIM = 12


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    action_space = T.ACT_DIM
    observation_space = T.OBS_DIM
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=2, physics_material=FLIGHT_MATERIAL)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True,
                                                     clone_in_fabric=True)
    robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight: float = T.THRUST_TO_WEIGHT
    moment_scale: float = T.MOMENT_SCALE

    reward_mode: str = "shaped"      # "shaped" (task.RewardWeights) or "l2f" (task.L2FWeights)
    rewards: dict = {}               # overrides for task.RewardWeights fields
    l2f: dict = {}                   # overrides for task.L2FWeights fields
    penalty_curriculum: dict = {}    # {"every": iters, "gate": rate or None, "factors": {...}, "caps": {...}}
    action_history: int = 1          # number of past actions in the observation (most recent first)
    motor_tau: float = 0.0           # s, first-order lag on thrust and torques (0 = ideal actuators)
    disturbance_force: float = 0.0   # std of a per-episode constant force, as a fraction of the weight
    disturbance_torque: float = 0.0  # std of a per-episode constant torque, N*m
    asymmetric_critic: bool = False  # critic also sees disturbances, actuator state, height and distance
    success_radius: float = 0.10     # m
    success_hold_s: float = 1.0      # continuous time inside the radius that counts as a hover success
    stage: int = -1                  # fixed curriculum stage; -1 = automatic curriculum
    curriculum_thresholds: tuple = (0.8, 0.8)
    curriculum_window: int = 2048
    render_scene: bool = False       # floor, green target sphere, lights (rendering only)
    scene_style: str = "isaac"       # "isaac" (stock grid floor), "white", or "studio"; see scene.dress_scene


class DroneHoverEnv(DirectRLEnv):
    cfg: DroneHoverEnvCfg

    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.body_id = self.robot.find_bodies("body")[0]
        self.mass = float(self.robot.root_physx_view.get_masses()[0].sum())
        self.weight = self.mass * T.GRAVITY
        inertia = self.robot.root_physx_view.get_inertias()[0, self.body_id[0]].reshape(3, 3)
        self.inertia_diag = [float(inertia[i, i]) for i in range(3)]
        if cfg.reward_mode == "l2f":
            self.w, self.term_names = T.L2FWeights(**cfg.l2f), L2F_TERMS
        else:
            self.w, self.term_names = T.RewardWeights(**cfg.rewards), SHAPED_TERMS
        pc = cfg.penalty_curriculum
        self.penalty_curriculum = T.PenaltyCurriculum(self.w, pc["factors"], pc["caps"], pc["every"], pc.get("gate")) \
            if pc else None
        self.obs_dim = T.obs_dim(cfg.action_history)
        self.critic_dim = self.obs_dim + (PRIVILEGED_DIM if cfg.asymmetric_critic else 0)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.actions, self.prev_actions = z(n, T.ACT_DIM), z(n, T.ACT_DIM)
        self.action_hist = z(n, cfg.action_history, T.ACT_DIM)
        self.cmd = z(n, 4)                       # commanded [thrust, tau_x, tau_y, tau_z]
        self.applied = z(n, 4)                   # after the motor lag
        self.force_b, self.torque_b = z(n, 1, 3), z(n, 1, 3)
        self.target_w, self.wind_w, self.dist_force_w, self.dist_torque_b = z(n, 3), z(n, 3), z(n, 3), z(n, 3)
        self.prev_lin_vel, self.prev_ang_vel = z(n, 3), z(n, 3)
        self.stage_of = torch.zeros(n, dtype=torch.long, device=dev)
        self.hover_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stats = {k: z(n) for k in ("dist", "tail_dist", "tail_n", "tail_ang_speed", "tail_ang_acc",
                                        "tail_action_rate", "tail_tilt", "max_tilt")}
        self.last_dist = z(n)
        self.final_obs, self.final_critic_obs = z(n, self.obs_dim), z(n, self.critic_dim)
        self.ep_sums = {k: z(n) for k in self.term_names}
        self.hold_steps = int(round(cfg.success_hold_s / self.step_dt))
        self.tail_steps = int(round(2.0 / self.step_dt))
        self.curriculum = Curriculum(T.N_TRAIN_STAGES, cfg.curriculum_thresholds, cfg.curriculum_window) \
            if cfg.stage < 0 else None
        self.forced: dict | None = None      # fixed held-out conditions for evaluation (indexed by env id)
        self._completed: list[dict] = []

    # ------------------------------------------------------------------------------------------- scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        if self.cfg.render_scene:
            import omni.usd
            from .scene import make_renderable
            make_renderable(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
        grid_visible = (not self.cfg.render_scene) or self.cfg.scene_style == "isaac"
        ground = sim_utils.GroundPlaneCfg(physics_material=FLIGHT_MATERIAL, visible=grid_visible)
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        if self.cfg.render_scene:
            from .scene import dress_scene
            self.target_marker = dress_scene(self.scene.env_origins[0].tolist(), self.cfg.scene_style)
        else:
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light.func("/World/Light", light)

    # ------------------------------------------------------------------------------------------- control
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.action_hist = torch.cat([self.actions.unsqueeze(1), self.action_hist[:, :-1]], dim=1)
        self.cmd[:, 0] = T.thrust_from_action(self.actions[:, 0], self.weight, self.cfg.thrust_to_weight)
        self.cmd[:, 1:] = self.cfg.moment_scale * self.actions[:, 1:]
        self.thrust = self.cmd[:, 0]          # kept for the verification script

    def _apply_action(self):
        if self.cfg.motor_tau > 0:            # first-order actuator lag, per physics step
            alpha = self.physics_dt / (self.cfg.motor_tau + self.physics_dt)
            self.applied += alpha * (self.cmd - self.applied)
        else:
            self.applied = self.cmd.clone()
        q = self.robot.data.root_quat_w
        self.force_b[:, 0, :] = quat_apply_inverse(q, self.wind_w + self.dist_force_w)
        self.force_b[:, 0, 2] += self.applied[:, 0]
        self.torque_b[:, 0, :] = self.applied[:, 1:] + self.dist_torque_b
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self.body_id, forces=self.force_b, torques=self.torque_b)

    # ------------------------------------------------------------------------------------------- MDP
    def _rel_target_b(self):
        rel_b, _ = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.target_w)
        return rel_b

    def _get_observations(self) -> dict:
        d = self.robot.data
        obs = T.build_obs(self._rel_target_b(), d.projected_gravity_b, d.root_lin_vel_b, d.root_ang_vel_b,
                          self.action_hist.reshape(self.num_envs, -1))
        out = {"policy": obs}
        if self.cfg.asymmetric_critic:
            q = d.root_quat_w
            priv = torch.cat([
                quat_apply_inverse(q, self.wind_w + self.dist_force_w) / (0.1 * self.weight),
                self.dist_torque_b / 1e-4,
                (self.applied[:, :1] / self.weight - 1.0), self.applied[:, 1:] / self.cfg.moment_scale,
                (d.root_pos_w[:, 2:3] - self.scene.env_origins[:, 2:3]),
                (self.target_w - d.root_pos_w).norm(dim=-1, keepdim=True),
            ], dim=-1).clamp(-T.OBS_CLIP, T.OBS_CLIP)
            out["critic"] = torch.cat([obs, priv], dim=-1)
        return out

    def _get_dones(self):
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        self.crashed, self.out_of_bounds = T.failure(pos, self.robot.data.projected_gravity_b)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.crashed | self.out_of_bounds, time_out

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        err = self.target_w - d.root_pos_w
        dist = err.norm(dim=-1)
        inside = dist < self.cfg.success_radius
        self.hover_count = torch.where(inside, self.hover_count + 1, torch.zeros_like(self.hover_count))
        success_now = (self.hover_count >= self.hold_steps) & ~self.succeeded
        self.success_step = torch.where(success_now, self.episode_length_buf, self.success_step)
        self.succeeded |= success_now
        lin_vel, ang_vel = d.root_lin_vel_b, d.root_ang_vel_b
        ang_acc = (ang_vel - self.prev_ang_vel) / self.step_dt
        lin_acc = (lin_vel - self.prev_lin_vel) / self.step_dt
        first = self.episode_length_buf <= 1                  # no acceleration estimate on the first step
        ang_acc = torch.where(first.unsqueeze(-1), torch.zeros_like(ang_acc), ang_acc)
        lin_acc = torch.where(first.unsqueeze(-1), torch.zeros_like(lin_acc), lin_acc)
        self.prev_ang_vel, self.prev_lin_vel = ang_vel.clone(), lin_vel.clone()
        self._update_stats(dist, ang_vel, ang_acc, d.projected_gravity_b)
        if self.cfg.reward_mode == "l2f":
            terms = T.l2f_reward_terms(self.w, err, lin_vel, ang_vel, d.projected_gravity_b, self.actions,
                                       success_now, self.reset_terminated, self.step_dt, ang_acc, lin_acc)
        else:
            terms = T.reward_terms(self.w, dist, lin_vel, ang_vel, d.projected_gravity_b, self.actions,
                                   self.prev_actions, success_now, self.reset_terminated, self.step_dt, ang_acc, lin_acc)
        total = torch.zeros(self.num_envs, device=self.device)
        for k in self.term_names:
            self.ep_sums[k] += terms[k]
            total += terms[k]
        return total

    def _update_stats(self, dist, ang_vel, ang_acc, grav_b):
        s = self.stats
        tilt = torch.rad2deg(torch.acos((-grav_b[:, 2]).clamp(-1.0, 1.0)))
        s["dist"] += dist
        s["max_tilt"] = torch.maximum(s["max_tilt"], tilt)
        tail = (self.episode_length_buf >= self.max_episode_length - self.tail_steps).float()
        s["tail_n"] += tail
        s["tail_dist"] += tail * dist
        s["tail_ang_speed"] += tail * ang_vel.norm(dim=-1)
        s["tail_ang_acc"] += tail * ang_acc.norm(dim=-1)
        s["tail_action_rate"] += tail * (self.actions - self.prev_actions).norm(dim=-1)
        s["tail_tilt"] += tail * tilt
        self.last_dist = dist

    def update_penalty_curriculum(self, iteration: int, success_rate: float) -> bool:
        return self.penalty_curriculum.step(iteration, success_rate) if self.penalty_curriculum else False

    # ------------------------------------------------------------------------------------------- resets
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # the observation of the state the episode ended in (used to bootstrap time-limit truncations)
        o = self._get_observations()
        self.final_obs[env_ids] = o["policy"][env_ids]
        self.final_critic_obs[env_ids] = o.get("critic", o["policy"])[env_ids]
        self._record(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = env_ids.numel()
        if self.forced is not None:
            c = {k: v[env_ids] for k, v in self.forced.items()}
        else:
            stages = self.curriculum.sample(n, self.device) if self.curriculum is not None \
                else torch.full((n,), self.cfg.stage, dtype=torch.long, device=self.device)
            c = T.sample_conditions(stages)
            if n == self.num_envs:   # spread out time limits after a full reset (as the reference does)
                self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        origins = self.scene.env_origins[env_ids]
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] = origins + c["start"]
        root[:, 3:7] = T.quat_from_rpy(c["rpy"])
        root[:, 7:10] = c["start_vel"]
        root[:, 10:] = c["start_ang_vel"] if "start_ang_vel" in c else 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                            self.robot.data.default_joint_vel[env_ids], None, env_ids)
        self.target_w[env_ids] = origins + c["target"]
        self.wind_w[env_ids] = c["wind_frac"] * self.weight
        # random disturbances are a training-time randomisation; held-out conditions carry their own (none by default)
        train = self.forced is None
        self.dist_force_w[env_ids] = torch.randn(n, 3, device=self.device) * self.cfg.disturbance_force * self.weight \
            if train else c.get("dist_force_w", torch.zeros(n, 3, device=self.device))
        self.dist_torque_b[env_ids] = torch.randn(n, 3, device=self.device) * self.cfg.disturbance_torque \
            if train else c.get("dist_torque_b", torch.zeros(n, 3, device=self.device))
        self.stage_of[env_ids] = c["stage"]
        hover = torch.zeros(n, 4, device=self.device)
        hover[:, 0] = self.weight
        self.cmd[env_ids], self.applied[env_ids] = hover, hover
        for buf in (self.actions, self.prev_actions, self.action_hist, self.prev_lin_vel, self.prev_ang_vel):
            buf[env_ids] = 0.0
        for buf in self.stats.values():
            buf[env_ids] = 0.0
        self.hover_count[env_ids] = 0
        self.succeeded[env_ids] = False
        self.success_step[env_ids] = -1
        for k in self.term_names:
            self.ep_sums[k][env_ids] = 0.0
        self.started[env_ids] = True
        if self.cfg.render_scene:
            self.target_marker.visualize(translations=self.target_w)

    def _record(self, env_ids):
        mask = self.started[env_ids] & (self.reset_terminated[env_ids] | self.reset_time_outs[env_ids])
        if not mask.any():
            return
        ids = env_ids[mask]
        length = self.episode_length_buf[ids].float().clamp(min=1)
        s = self.stats
        tn = s["tail_n"][ids]
        nan = torch.full_like(length, float("nan"))
        tail = lambda k: torch.where(tn > 0, s[k][ids] / tn.clamp(min=1), nan)
        rec = {
            "env_id": ids.clone(), "stage": self.stage_of[ids].clone(), "success": self.succeeded[ids].clone(),
            "crashed": (self.reset_terminated[ids] & self.crashed[ids]).clone(),
            "out_of_bounds": (self.reset_terminated[ids] & self.out_of_bounds[ids]).clone(),
            "timed_out": (self.reset_time_outs[ids] & ~self.reset_terminated[ids]).clone(),
            "length_s": length * self.step_dt,
            "final_dist": self.last_dist[ids].clone(),
            "mean_dist": s["dist"][ids] / length,
            "tail_dist": tail("tail_dist"),
            "tail_ang_speed": tail("tail_ang_speed"),         # rad/s, last 2 s
            "tail_ang_acc": tail("tail_ang_acc"),             # rad/s^2, last 2 s (wobble)
            "tail_action_rate": tail("tail_action_rate"),     # |a_t - a_t-1|, last 2 s (control jitter)
            "tail_tilt_deg": tail("tail_tilt"),
            "max_tilt_deg": s["max_tilt"][ids].clone(),
            "time_to_hover_s": torch.where(self.success_step[ids] >= 0,
                                           (self.success_step[ids] - self.hold_steps + 1).float() * self.step_dt, nan),
        }
        for k in self.term_names:
            rec[f"r_{k}"] = self.ep_sums[k][ids].clone()
        rec["return"] = sum(rec[f"r_{k}"] for k in self.term_names)
        self._completed.append(rec)
        if self.curriculum is not None and self.forced is None:
            self.curriculum.record(rec["stage"], rec["success"], int(self.common_step_counter))

    def pop_completed(self) -> dict | None:
        if not self._completed:
            return None
        out = {k: torch.cat([r[k] for r in self._completed]) for k in self._completed[0]}
        self._completed = []
        return out
