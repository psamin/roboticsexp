"""Save-rate-driven curriculum over the shot levels in physics/shots.py."""
import torch

from ..physics.outcomes import SAVE


class Curriculum:
    """Promote to the next level once the save rate on on-target shots of the current level reaches a threshold.

    While at level k, a share `current_frac` of new shots come from level k and the rest from lower levels, so
    earlier skills keep being exercised.
    """

    def __init__(self, n_levels: int, thresholds=(0.85, 0.85), window: int = 2048, current_frac: float = 0.75,
                 start_level: int = 0):
        self.n_levels, self.thresholds, self.window = n_levels, tuple(thresholds), window
        self.current_frac = current_frac
        self.level = start_level
        self._results: list[torch.Tensor] = []
        self.history: list[dict] = []

    def sample_levels(self, n: int, device, generator=None) -> torch.Tensor:
        lv = torch.full((n,), self.level, dtype=torch.long, device=device)
        if self.level == 0:
            return lv
        lower = torch.randint(0, self.level, (n,), device=device, generator=generator)
        use_lower = torch.rand(n, device=device, generator=generator) >= self.current_frac
        return torch.where(use_lower, lower, lv)

    def rate(self) -> tuple[float, int]:
        if not self._results:
            return 0.0, 0
        r = torch.cat(self._results)[-self.window:]
        return float(r.float().mean()), int(r.numel())

    def record(self, levels: torch.Tensor, outcomes: torch.Tensor, on_target: torch.Tensor, step: int = 0):
        """Feed finished episodes. Returns the new level name index if a promotion happened, else None."""
        mask = (levels == self.level) & on_target
        if mask.any():
            self._results.append((outcomes[mask] == SAVE).cpu())
            if sum(t.numel() for t in self._results) > 4 * self.window:
                self._results = [torch.cat(self._results)[-self.window:]]
        if self.level >= self.n_levels - 1:
            return None
        rate, count = self.rate()
        if count >= self.window and rate >= self.thresholds[self.level]:
            self.history.append({"from": self.level, "to": self.level + 1, "rate": rate, "step": step})
            self.level += 1
            self._results = []
            return self.level
        return None

    def state_dict(self):
        return {"level": self.level, "history": self.history}

    def load_state_dict(self, sd):
        self.level, self.history = sd["level"], list(sd["history"])
        self._results = []
