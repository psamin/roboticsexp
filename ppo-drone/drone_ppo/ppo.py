"""Proximal Policy Optimization (clipped objective), written directly in PyTorch."""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .networks import Critic, SquashedGaussianActor, squash_log_prob
from .rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    num_minibatches: int = 4
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    adaptive_lr: bool = True          # adjust lr per minibatch toward a target KL (as Isaac Lab's reference runner)
    desired_kl: float = 0.01
    lr_bounds: tuple = (1e-5, 1e-3)
    hidden: tuple = field(default_factory=lambda: (128, 128))
    init_std: float = 0.5


def clipped_surrogate(new_logp, old_logp, adv, clip: float):
    """L = -E[min(r A, clip(r, 1-eps, 1+eps) A)] with r = pi_new / pi_old; plus KL and clip-fraction diagnostics.

    approx_kl is the low-variance estimator E[(r - 1) - log r] of KL(pi_old || pi_new).
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return loss, approx_kl, clip_frac


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden, cfg.init_std).to(self.device)
        self.critic = Critic(obs_dim, cfg.hidden).to(self.device)
        self.lr = cfg.lr
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs):
        u, a, logp = self.actor.sample(obs)
        return u, a, logp, self.critic(obs)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor.deterministic(obs)

    @torch.no_grad()
    def value(self, obs):
        return self.critic(obs)

    def _set_lr(self, lr):
        self.lr = float(min(max(lr, self.cfg.lr_bounds[0]), self.cfg.lr_bounds[1]))
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def update(self, buf: RolloutBuffer) -> dict:
        c = self.cfg
        sums, n = {}, 0
        for _ in range(c.epochs):
            for mb in buf.minibatches(c.num_minibatches):
                adv = mb["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)                 # advantage normalisation
                dist = self.actor.dist(mb["obs"])
                new_logp = squash_log_prob(dist, mb["raw_actions"])
                entropy = dist.entropy().sum(-1).mean()                      # of the pre-squash Gaussian
                pg_loss, kl, clip_frac = clipped_surrogate(new_logp, mb["log_probs"], adv, c.clip)
                v = self.critic(mb["obs"])
                v_loss = 0.5 * (v - mb["returns"]).pow(2).mean()
                loss = pg_loss + c.vf_coef * v_loss - c.ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.opt.step()
                if c.adaptive_lr:
                    if kl > 2.0 * c.desired_kl:
                        self._set_lr(self.lr / 1.5)
                    elif kl < 0.5 * c.desired_kl:
                        self._set_lr(self.lr * 1.5)
                stats = {"actor_loss": pg_loss, "critic_loss": v_loss, "entropy": entropy, "approx_kl": kl,
                         "clip_frac": clip_frac, "grad_norm_actor": gn_a, "grad_norm_critic": gn_c}
                for k, val in stats.items():
                    sums[k] = sums.get(k, 0.0) + float(val.detach())
                n += 1
        out = {k: v / max(n, 1) for k, v in sums.items()}
        with torch.no_grad():
            y, v = buf.returns.flatten(), buf.values.flatten()
            out["explained_variance"] = float(1 - (y - v).var() / (y.var() + 1e-8))
        out["action_std"] = float(self.actor.log_std.detach().exp().mean())
        out["lr"] = self.lr
        return out

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "opt": self.opt.state_dict(),
                "lr": self.lr, "cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "act_dim": self.act_dim}

    def load_state_dict(self, sd, load_optimizer: bool = True):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if load_optimizer and "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
            self._set_lr(sd.get("lr", self.cfg.lr))

    def save(self, path, **extra):
        torch.save({**self.state_dict(), "extra": extra}, path)

    @classmethod
    def load(cls, path, device="cpu", load_optimizer: bool = True):
        sd = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(sd["cfg"])
        cfg["hidden"], cfg["lr_bounds"] = tuple(cfg["hidden"]), tuple(cfg["lr_bounds"])
        agent = cls(sd["obs_dim"], sd["act_dim"], PPOConfig(**cfg), device)
        agent.load_state_dict(sd, load_optimizer)
        return agent, sd.get("extra", {})
