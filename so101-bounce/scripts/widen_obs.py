"""Warm start a policy with a longer memory (EnvCfg history / cam_k): the actor's and the critic's first layers keep
the old weights at the new positions of the same inputs and get zero weights for the older entries, so the policy
acts exactly as before and can learn to use the extra history. Resets the best score, wall time and optimizer.
  uv run python scripts/widen_obs.py in.pt out.pt NEW_HISTORY NEW_CAM_K
"""
import sys

import torch

J = 5


def blocks(h, k):
    """The policy observation's blocks in env._obs order; within each, the newest entry comes first."""
    return [h * J, h * J, 3, 3, 3 * k, k, k, 3, 1, 1]


def index_map(h0, k0, h1, k1):
    """For each old policy input, its column in the new observation; and the new observation's size."""
    m, start = [], 0
    for s0, s1 in zip(blocks(h0, k0), blocks(h1, k1)):
        m += [start + i for i in range(s0)]
        start += s1
    return m, start


def main():
    src, dst, h1, k1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    c = torch.load(src, map_location="cpu", weights_only=False)
    env = c["extra"].setdefault("config", {}).setdefault("env", {})
    h0, k0 = env.get("history", 4), env.get("cam_k", 4)
    m, new_obs = index_map(h0, k0, h1, k1)
    old_obs = c["obs_dim"]
    assert len(m) == old_obs, (len(m), old_obs)

    def widen(w, priv=0):
        out = torch.zeros(w.shape[0], new_obs + priv)
        out[:, m] = w[:, :old_obs]
        out[:, new_obs:] = w[:, old_obs:]
        return out

    c["actor"]["mean.0.weight"] = widen(c["actor"]["mean.0.weight"])
    priv = c["critic_dim"] - old_obs
    c["critic"]["v.0.weight"] = widen(c["critic"]["v.0.weight"], priv)
    c["obs_dim"], c["critic_dim"] = new_obs, new_obs + priv
    c.pop("opt", None)
    env["history"], env["cam_k"] = h1, k1
    c["extra"].update(best=-1e9, wall_s=0.0)
    torch.save(c, dst)
    print(f"history {h0} -> {h1}, detections {k0} -> {k1}: obs {old_obs} -> {new_obs}, critic {old_obs + priv} -> "
          f"{new_obs + priv}")


if __name__ == "__main__":
    main()
