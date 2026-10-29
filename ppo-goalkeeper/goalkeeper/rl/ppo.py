"""Proximal Policy Optimization, written directly in PyTorch (no RL library)."""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .buffer import RolloutBuffer
from .networks import Critic, GaussianActor


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    num_minibatches: int = 4
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    clip_value: bool = False           # returns span about +-10, so clipping value moves to +-clip would stall the critic
    target_kl: float | None = 0.03      # stop the epoch loop early when the policy moved too far
    hidden: tuple = field(default_factory=lambda: (128, 128))
    init_std: float = 0.6


def clipped_surrogate(new_logp, old_logp, adv, clip: float):
    """PPO-clip policy loss, plus diagnostics.

    L = -E[min(r * A, clip(r, 1 - eps, 1 + eps) * A)],  r = pi_new(a|s) / pi_old(a|s).
    approx_kl uses the low-variance estimator E[(r - 1) - log r].
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return loss, approx_kl, clip_frac


def value_loss(v, old_v, returns, clip: float | None):
    if clip is None:
        return 0.5 * (v - returns).pow(2).mean()
    v_clipped = old_v + (v - old_v).clamp(-clip, clip)
    return 0.5 * torch.max((v - returns).pow(2), (v_clipped - returns).pow(2)).mean()


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.actor = GaussianActor(obs_dim, act_dim, cfg.hidden, cfg.init_std).to(self.device)
        self.critic = Critic(obs_dim, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs):
        dist = self.actor(obs)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), self.critic(obs)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor.mean(obs)

    @torch.no_grad()
    def value(self, obs):
        return self.critic(obs)

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        c = self.cfg
        sums, n, stopped_early = {}, 0, False
        for _ in range(c.epochs):
            epoch_kl = []
            for mb in buf.minibatches(c.num_minibatches):
                adv = mb["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                dist = self.actor(mb["obs"])
                new_logp = dist.log_prob(mb["actions"]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                pg_loss, kl, clip_frac = clipped_surrogate(new_logp, mb["log_probs"], adv, c.clip)
                v = self.critic(mb["obs"])
                v_loss = value_loss(v, mb["values"], mb["returns"], c.clip if c.clip_value else None)
                loss = pg_loss - c.ent_coef * entropy + c.vf_coef * v_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.opt.step()
                stats = {"policy_loss": pg_loss, "value_loss": v_loss, "entropy": entropy, "approx_kl": kl,
                         "clip_frac": clip_frac, "grad_norm_actor": gn_a, "grad_norm_critic": gn_c}
                for k, val in stats.items():
                    sums[k] = sums.get(k, 0.0) + float(val.detach() if torch.is_tensor(val) else val)
                n += 1
                epoch_kl.append(float(kl))
            if c.target_kl is not None and sum(epoch_kl) / len(epoch_kl) > 1.5 * c.target_kl:
                stopped_early = True
                break
        out = {k: v / max(n, 1) for k, v in sums.items()}
        with torch.no_grad():
            y, v = buf.returns.flatten(), buf.values.flatten()
            out["explained_variance"] = float(1 - (y - v).var() / (y.var() + 1e-8))
        out["action_std"] = float(self.actor.log_std.exp().mean())
        out["minibatch_updates"] = n
        out["kl_early_stop"] = float(stopped_early)
        return out

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "opt": self.opt.state_dict(),
                "cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "act_dim": self.act_dim}

    def load_state_dict(self, sd, load_optimizer: bool = True):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if load_optimizer and "opt" in sd:
            self.opt.load_state_dict(sd["opt"])

    def save(self, path, **extra):
        torch.save({**self.state_dict(), "extra": extra}, path)

    @classmethod
    def load(cls, path, device="cpu", load_optimizer: bool = True):
        sd = torch.load(path, map_location=device, weights_only=False)
        cfg = PPOConfig(**{**sd["cfg"], "hidden": tuple(sd["cfg"]["hidden"])})
        agent = cls(sd["obs_dim"], sd["act_dim"], cfg, device)
        agent.load_state_dict(sd, load_optimizer)
        return agent, sd.get("extra", {})
