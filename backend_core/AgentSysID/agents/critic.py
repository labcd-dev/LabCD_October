"""
Critic Agent – evaluates training / validation metrics and decides whether
the current architecture is acceptable or needs adjustment.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.training.tracker import BestConfigTracker


class CriticAgent:
    def __init__(self, tracker: BestConfigTracker, run_mode: str = "regular", log_filename: Optional[str] = None):
        self.tracker = tracker
        self.run_mode = run_mode
        self.log_filename = log_filename

    def evaluate(
        self,
        train_mse: float,
        val_mse: float,
        current_config: Dict[str, Any],
        activation: str,
        measured_latency: float,
        max_latency: float,
        cycle_number: int,
    ) -> Dict[str, Any]:
        """
        Returns a decision dict:
          {
            "status": "accept" | "reject" | "retry",
            "reason": str,
            "suggestions": str,
          }
        """
        overfit_ratio = val_mse / max(train_mse, 1e-12)
        system = (
            "You are a strict technical Critic for neural system identification. "
            "Evaluate the latest training cycle. Reply with JSON containing: "
            "status (accept|reject|retry), reason (short), suggestions (short)."
        )
        user = (
            f"Cycle: {cycle_number}\n"
            f"Train MSE: {train_mse:.6e}\n"
            f"Val MSE: {val_mse:.6e}\n"
            f"Overfit ratio: {overfit_ratio:.2f} (limit={cfg.OVERFIT_RATIO_LIMIT})\n"
            f"Latency ms: {measured_latency:.3f} (max={max_latency})\n"
            f"Config: {json.dumps(current_config)}\n"
            f"Activation: {activation}\n"
            f"Best val MSE so far: {self.tracker.best_mse:.6e}\n"
            f"Run mode: {self.run_mode}"
        )
        raw = invoke_llm(
            system, user, agent_name="Critic", cycle=cycle_number, log_filename=self.log_filename
        )
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                status = str(data.get("status", "retry")).lower()
                if status not in ("accept", "reject", "retry"):
                    status = "retry"
                return {
                    "status": status,
                    "reason": data.get("reason", ""),
                    "suggestions": data.get("suggestions", ""),
                }
        except Exception:
            pass

        # Heuristic fallback
        if overfit_ratio > cfg.OVERFIT_RATIO_LIMIT:
            return {"status": "reject", "reason": "Severe overfitting", "suggestions": "Increase regularization or reduce capacity"}
        if val_mse < cfg.MSE_TARGET and measured_latency <= max_latency:
            return {"status": "accept", "reason": "Met MSE and latency targets", "suggestions": ""}
        return {"status": "retry", "reason": "Targets not yet met", "suggestions": "Continue tuning"}
