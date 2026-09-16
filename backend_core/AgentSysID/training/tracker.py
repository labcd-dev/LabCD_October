"""Best-config tracker used by the Critic / main loop."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class BestConfigTracker:
    def __init__(self, run_mode: str = "regular", memory: int = 5):
        self.run_mode = run_mode
        self.best_mse = float("inf")
        self.best_config: Dict[str, Any] = {}
        self.best_rmse: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._recent_failures: Deque[str] = deque(maxlen=memory)

    def update(self, mse: float, config: Dict[str, Any], current_rmse: Optional[float] = None) -> bool:
        """Return True if this is a new best."""
        entry = {"mse": mse, "config": dict(config), "rmse": current_rmse}
        self.history.append(entry)
        if mse < self.best_mse:
            self.best_mse = mse
            self.best_config = dict(config)
            self.best_rmse = current_rmse
            return True
        return False

    def add_reasoning_to_memory(self, reasoning: str) -> None:
        if reasoning:
            self._recent_failures.append(reasoning)

    def get_recent_failures_str(self) -> str:
        if not self._recent_failures:
            return "(none)"
        return "\n".join(f"- {r}" for r in self._recent_failures)
