"""
Best-config tracker used by the Critic / Actor / Explorer loop.

Ported from ``_legacy/framework.BestConfigTracker``: it keeps the global best
configuration, the Critic's reasoning that produced it, and a run-mode-sized
memory of failed configurations that is fed back into every agent prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BestConfigTracker:
    #: How many failures each run mode remembers (0 = unlimited).
    MEMORY_BY_MODE: Dict[str, int] = {"fast": 5, "regular": 10, "heavy": 0, "expert": 0}

    def __init__(self, run_mode: str = "regular", memory: Optional[int] = None):
        self.best_mse: float = float("inf")
        self.best_rmse: Optional[float] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_reasoning: str = "N/A"
        self.recent_failures: List[Dict[str, Any]] = []
        self.last_was_best: bool = False
        self.run_mode: str = (run_mode or "regular").lower()
        self.history: List[Dict[str, Any]] = []

        if memory is not None:
            self.memory_limit = int(memory)
        else:
            self.memory_limit = self.MEMORY_BY_MODE.get(self.run_mode, 10)

    # ------------------------------------------------------------------
    def update(
        self,
        mse: float,
        config: Dict[str, Any],
        current_rmse: Optional[float] = None,
    ) -> bool:
        """Record a cycle result. Returns True when it is a new global best."""
        self.last_was_best = False
        self.history.append(
            {"mse": mse, "rmse": current_rmse, "config": dict(config)}
        )

        if mse < self.best_mse:
            self.best_mse = mse
            self.best_rmse = current_rmse
            self.best_config = dict(config)
            self.last_was_best = True
            return True

        self.recent_failures.append(
            {"config": dict(config), "mse": mse, "reasoning": "Pending..."}
        )

        # Dynamic memory capacity: fast=5, regular=10, heavy/expert=unlimited
        if self.memory_limit and len(self.recent_failures) > self.memory_limit:
            self.recent_failures.pop(0)

        return False

    # ------------------------------------------------------------------
    def add_reasoning_to_memory(self, reasoning: str) -> None:
        """Attach the Critic's text reasoning to the correct saved cycle."""
        if not reasoning:
            return
        if self.last_was_best:
            self.best_reasoning = reasoning
        elif self.recent_failures:
            self.recent_failures[-1]["reasoning"] = reasoning

    def get_recent_failures_str(self) -> str:
        """Render the failure memory for injection into agent prompts."""
        if not self.recent_failures:
            return "None"
        res = ""
        for fail in self.recent_failures:
            res += (
                f"- Config: {fail['config']} | MSE: {fail['mse']:.6f}\n"
                f"  Past Critic Logic: {fail.get('reasoning', 'N/A')}\n"
            )
        return res
