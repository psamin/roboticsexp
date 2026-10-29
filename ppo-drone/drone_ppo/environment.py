"""Crazyflie hover / target-reaching environment (Isaac Lab direct workflow).

Adapted from Isaac Lab's Isaac-Quadcopter-Direct-v0 (BSD-3, isaaclab_tasks/direct/quadcopter): same Crazyflie
asset, same 100 Hz physics / 50 Hz control, same collective-thrust + body-torque interface applied as an external
wrench on the drone body. The drone only moves through those forces; its pose is never set during an episode.

Changes for this project: hover-centred thrust mapping, body-frame observations with the previous action, a
three-stage curriculum, the reward terms in task.py, crash/out-of-bounds failures kept separate from time limits,
and the pre-reset observation kept for bootstrapping truncated episodes.
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

    rewards: dict = {}               # overrides for task.RewardWeights fields
    success_radius: float = 0.10     # m
    success_hold_s: float = 1.0      # continuous time inside the radius that counts as a hover success
    stage: int = -1                  # fixed curriculum stage; -1 = automatic curriculum
    curriculum_thresholds: tuple = (0.8, 0.8)
    curriculum_window: int = 2048
    render_scene: bool = False       # grey floor, green target sphere, lights (rendering only)


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
        self.w = T.RewardWeights(**cfg.rewards)
        self.actions = torch.zeros(n, T.ACT_DIM, device=dev)
        self.prev_actions = torch.zeros(n, T.ACT_DIM, device=dev)
        self.force_b = torch.zeros(n, 1, 3, device=dev)
        self.torque_b = torch.zeros(n, 1, 3, device=dev)
        self.target_w = torch.zeros(n, 3, device=dev)
        self.wind_w = torch.zeros(n, 3, device=dev)
        self.stage_of = torch.zeros(n, dtype=torch.long, device=dev)
        self.hover_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.dist_sum = torch.zeros(n, device=dev)
        self.tail_sum = torch.zeros(n, device=dev)
        self.tail_n = torch.zeros(n, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.final_obs = torch.zeros(n, T.OBS_DIM, device=dev)
        self.term_names = ("distance", "precision", "lin_vel", "ang_vel", "upright", "effort", "action_rate",
                           "success", "crash")
        self.ep_sums = {k: torch.zeros(n, device=dev) for k in self.term_names}
        self.hold_steps = int(round(cfg.success_hold_s / self.step_dt))
        self.tail_steps = int(round(2.0 / self.step_dt))
        self.curriculum = Curriculum(len(T.STAGES), cfg.curriculum_thresholds, cfg.curriculum_window) \
            if cfg.stage < 0 else None
        self.forced: dict | None = None      # fixed held-out conditions for evaluation (indexed by env id)
        self._completed: list[dict] = []

    # ------------------------------------------------------------------------------------------- scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        ground = sim_utils.GroundPlaneCfg(physics_material=FLIGHT_MATERIAL, visible=not self.cfg.render_scene)
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        if self.cfg.render_scene:
            from .scene import dress_scene
            self.target_marker = dress_scene(self.scene.env_origins[0].tolist())
        else:
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light.func("/World/Light", light)

    # ------------------------------------------------------------------------------------------- control
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.thrust = T.thrust_from_action(self.actions[:, 0], self.weight, self.cfg.thrust_to_weight)
        self.torque_b[:, 0, :] = self.cfg.moment_scale * self.actions[:, 1:]

    def _apply_action(self):
        wind_b = quat_apply_inverse(self.robot.data.root_quat_w, self.wind_w)
        self.force_b[:, 0, :] = wind_b
        self.force_b[:, 0, 2] += self.thrust
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self.body_id, forces=self.force_b, torques=self.torque_b)

    # ------------------------------------------------------------------------------------------- MDP
    def _rel_target_b(self):
        rel_b, _ = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.target_w)
        return rel_b

    def _get_observations(self) -> dict:
        d = self.robot.data
        obs = T.build_obs(self._rel_target_b(), d.projected_gravity_b, d.root_lin_vel_b, d.root_ang_vel_b, self.actions)
        return {"policy": obs}

    def _get_dones(self):
        pos = self.robot.data.root_pos_w - self.scene.env_origins
        self.crashed, self.out_of_bounds = T.failure(pos, self.robot.data.projected_gravity_b)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.crashed | self.out_of_bounds, time_out

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        dist = (self.target_w - d.root_pos_w).norm(dim=-1)
        inside = dist < self.cfg.success_radius
        self.hover_count = torch.where(inside, self.hover_count + 1, torch.zeros_like(self.hover_count))
        success_now = (self.hover_count >= self.hold_steps) & ~self.succeeded
        self.success_step = torch.where(success_now, self.episode_length_buf, self.success_step)
        self.succeeded |= success_now
        self.dist_sum += dist
        tail = self.episode_length_buf >= self.max_episode_length - self.tail_steps
        self.tail_sum += torch.where(tail, dist, torch.zeros_like(dist))
        self.tail_n += tail.float()
        self.last_dist = dist
        terms = T.reward_terms(self.w, dist, d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b,
                               self.actions, self.prev_actions, success_now, self.reset_terminated, self.step_dt)
        total = torch.zeros(self.num_envs, device=self.device)
        for k in self.term_names:
            self.ep_sums[k] += terms[k]
            total += terms[k]
        return total

    # ------------------------------------------------------------------------------------------- resets
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # the observation of the state the episode ended in (used to bootstrap time-limit truncations)
        self.final_obs[env_ids] = self._get_observations()["policy"][env_ids]
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
        root[:, 10:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                            self.robot.data.default_joint_vel[env_ids], None, env_ids)
        self.target_w[env_ids] = origins + c["target"]
        self.wind_w[env_ids] = c["wind_frac"] * self.weight
        self.stage_of[env_ids] = c["stage"]
        for buf in (self.actions, self.prev_actions, self.dist_sum, self.tail_sum, self.tail_n):
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
        rec = {
            "env_id": ids.clone(), "stage": self.stage_of[ids].clone(), "success": self.succeeded[ids].clone(),
            "crashed": (self.reset_terminated[ids] & self.crashed[ids]).clone(),
            "out_of_bounds": (self.reset_terminated[ids] & self.out_of_bounds[ids]).clone(),
            "timed_out": (self.reset_time_outs[ids] & ~self.reset_terminated[ids]).clone(),
            "length_s": length * self.step_dt,
            "final_dist": self.last_dist[ids].clone() if hasattr(self, "last_dist") else torch.zeros(len(ids)),
            "mean_dist": self.dist_sum[ids] / length,
            "tail_dist": torch.where(self.tail_n[ids] > 0, self.tail_sum[ids] / self.tail_n[ids].clamp(min=1),
                                     torch.full_like(length, float("nan"))),
            "time_to_hover_s": torch.where(self.success_step[ids] >= 0,
                                           (self.success_step[ids] - self.hold_steps + 1).float() * self.step_dt,
                                           torch.full_like(length, float("nan"))),
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
