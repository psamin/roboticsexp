"""Advance through the task stages when the measured hover-success rate supports it."""
import torch


class Curriculum:
    """Promote once the success rate over the last `window` episodes of the current stage reaches its threshold.

    A share `current_frac` of new episodes use the current stage; the rest replay earlier stages.
    """

    def __init__(self, n_stages: int, thresholds=(0.8, 0.8), window: int = 2048, current_frac: float = 0.8,
                 start: int = 0):
        self.n, self.thresholds, self.window, self.current_frac = n_stages, tuple(thresholds), window, current_frac
        self.stage = start
        self._res: list[torch.Tensor] = []
        self.history: list[dict] = []

    def sample(self, n: int, device) -> torch.Tensor:
        cur = torch.full((n,), self.stage, dtype=torch.long, device=device)
        if self.stage == 0:
            return cur
        lower = torch.randint(0, self.stage, (n,), device=device)
        return torch.where(torch.rand(n, device=device) >= self.current_frac, lower, cur)

    def rate(self):
        if not self._res:
            return 0.0, 0
        r = torch.cat(self._res)[-self.window:]
        return float(r.float().mean()), int(r.numel())

    def record(self, stages: torch.Tensor, success: torch.Tensor, step: int = 0):
        m = stages == self.stage
        if m.any():
            self._res.append(success[m].detach().cpu())
            if sum(t.numel() for t in self._res) > 4 * self.window:
                self._res = [torch.cat(self._res)[-self.window:]]
        if self.stage >= self.n - 1:
            return None
        rate, count = self.rate()
        if count >= self.window and rate >= self.thresholds[self.stage]:
            self.history.append({"from": self.stage, "to": self.stage + 1, "rate": rate, "step": step})
            self.stage += 1
            self._res = []
            return self.stage
        return None

    def state_dict(self):
        return {"stage": self.stage, "history": self.history}

    def load_state_dict(self, sd):
        self.stage, self.history, self._res = sd["stage"], list(sd["history"]), []
