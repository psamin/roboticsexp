"""The policy's observation, built the same way on the real arm as in simulation (bounce/env.py: BounceEnv._obs).

Per control tick on the arm:
    obs = RealObs(...); each tick: obs.joints(q_measured) -> obs.ball(detections) -> x = obs.vector(t)
    a = policy(x); send goals q_ref + action_scale * a; obs.acted(a)
The order matters: in simulation the observation after step(a_k) holds a_k as the newest action and the joint angles
measured after it, which is exactly what the arm has at the next tick before choosing a_{k+1}.

Units: joint angles in radians in the simulator's convention, positions in metres in the robot base frame, time in
seconds. Only torch is needed (no MuJoCo): kinematics come from ArmKinematics.save().
"""
import torch

from .robot import ArmKinematics

J = 5


class RealObs:
    def __init__(self, kin: ArmKinematics, q_ref, home, target_apex: float = 0.10, history: int = 4, cam_k: int = 4,
                 side_camera: bool = False):
        self.kin = kin
        self.q_ref = torch.as_tensor(q_ref, dtype=torch.float32).reshape(1, J)
        self.home = torch.as_tensor(home, dtype=torch.float32).reshape(1, 3)
        self.target = float(target_apex)
        self.side = 1.0 if side_camera else 0.0       # is a second (side) camera fused into the ball estimate
        self.H, self.K = history, cam_k
        self.q_hist = torch.zeros(1, history, J)
        self.a_hist = torch.zeros(1, history, J)
        self.det = []                       # (capture time, position (3,)) newest last
        self.started = False

    def joints(self, q_meas):
        d = torch.as_tensor(q_meas, dtype=torch.float32).reshape(1, 1, J) - self.q_ref[:, None]
        self.q_hist = d.expand(1, self.H, J).clone() if not self.started else torch.cat([d, self.q_hist[:, :-1]], 1)
        self.started = True

    def ball(self, t_capture: float, pos):
        """A camera detection: capture time (s, same clock as vector()) and ball centre (m, base frame)."""
        self.det.append((float(t_capture), torch.as_tensor(pos, dtype=torch.float32).reshape(3)))
        self.det = self.det[-32:]

    def acted(self, a):
        a = torch.as_tensor(a, dtype=torch.float32).reshape(1, 1, J).clamp(-1, 1)
        self.a_hist = torch.cat([a, self.a_hist[:, :-1]], 1)

    def vector(self, t_now: float):
        qm = self.q_hist[:, 0] + self.q_ref
        fk = self.kin.forward(qm)
        po, n = fk["paddle_o"] - self.home, fk["paddle_R"][..., 2]
        K = self.K
        recent = sorted(self.det, key=lambda x: -x[0])[:K]
        pos, age, valid = torch.zeros(1, K, 3), torch.full((1, K), 0.5), torch.zeros(1, K)
        for i, (tc, p) in enumerate(recent):
            pos[0, i], age[0, i], valid[0, i] = p - self.home[0], min(max(t_now - tc, 0.0), 0.5), 1.0
        latest_rel = pos[:, 0] - po if valid[0, 0] > 0 else torch.zeros_like(po)
        return torch.cat([2.0 * self.q_hist.reshape(1, -1), self.a_hist.reshape(1, -1), 10 * po, n,
                          10 * pos.reshape(1, -1), 10 * age, valid, 10 * latest_rel,
                          torch.tensor([[10 * self.target, self.side]])], -1)
