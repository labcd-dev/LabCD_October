"""
Critic Agent.

Evaluates one tuning cycle against three axes — generalization gap, absolute
accuracy and real-time inference latency — and returns a structured verdict the
Actor consumes:

    {"diagnosis", "status", "lr_dir", "lr_step", "hidden_layers", "reasoning"}

The search phase changes with the cycle number: early cycles demand large
topological jumps (explicitly forcing a change in layer *count*), later cycles
switch to precise local fine-tuning. Customer context is fed in according to the
run mode (heavy = always, regular = first five cycles, fast = never).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.agents.prompt_library import block, render, system_prompt
from backend_core.AgentSysID.training.tracker import BestConfigTracker

#: Cycle after which the Critic switches from exploration to fine-tuning.
EXPLORE_LIMIT_BY_MODE: Dict[str, int] = {"fast": 3, "regular": 8, "heavy": 15}


class CriticAgent:
    def __init__(
        self,
        tracker: BestConfigTracker,
        run_mode: str = "regular",
        log_filename: Optional[str] = None,
    ):
        self.tracker = tracker
        self.run_mode = (run_mode or "regular").lower()
        self.log_filename = log_filename

    # ------------------------------------------------------------------
    def explore_limit(self) -> int:
        return EXPLORE_LIMIT_BY_MODE.get(self.run_mode, 8)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        train_mse: float,
        val_mse: float,
        current_config: Dict[str, Any],
        activation: str,
        measured_latency: float,
        max_latency: float,
        cycle_number: int,
        val_rmse: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return the Critic's structured verdict for this cycle."""
        curr_lr = current_config["learning_rate"]
        curr_hs = current_config["hidden_layers"]

        overfit_ratio = (val_mse / train_mse) if train_mse > 1e-8 else 0.0
        best_mse_str = (
            f"{self.tracker.best_mse:.6f}" if self.tracker.best_mse != float("inf") else "N/A"
        )

        if self.tracker.best_config is None:
            best_lr, best_hs = curr_lr, curr_hs
        else:
            best_lr = self.tracker.best_config["learning_rate"]
            best_hs = self.tracker.best_config["hidden_layers"]

        # --- Search phase instruction -------------------------------------
        if cycle_number <= self.explore_limit():
            phase_instruction = block("critic", "phase_explore")
        else:
            phase_instruction = block("critic", "phase_finetune")

        # --- Customer context feed, gated by run mode ----------------------
        customer_context = str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip()
        system_context_block = ""
        if customer_context and (
            self.run_mode == "heavy"
            or (self.run_mode == "regular" and cycle_number <= 5)
        ):
            system_context_block = f"\n  CUSTOMER SYSTEM CONTEXT: '{customer_context}'"

        prompt = render(
            "critic",
            system_context_block=system_context_block,
            max_latency=float(max_latency),
            measured_latency=float(measured_latency),
            curr_train_mse_str=f"{train_mse:.6f}",
            curr_val_mse_str=f"{val_mse:.6f}",
            curr_val_rmse_str=f"{val_rmse:.6f}" if val_rmse is not None else "N/A",
            overfit_ratio=float(overfit_ratio),
            curr_lr=curr_lr,
            curr_hs=curr_hs,
            best_mse_str=best_mse_str,
            best_lr=best_lr,
            best_hs=best_hs,
            best_reasoning=self.tracker.best_reasoning,
            failures_str=self.tracker.get_recent_failures_str(),
            lr_min=cfg.LEARNING_RATE_MIN,
            lr_max=cfg.LEARNING_RATE_MAX,
            hs_min=cfg.HIDDEN_SIZE_MIN,
            hs_max=cfg.HIDDEN_SIZE_MAX,
            nl_min=cfg.NUM_LAYERS_MIN,
            nl_max=cfg.NUM_LAYERS_MAX,
            phase_instruction=phase_instruction,
        )

        default: Dict[str, Any] = {
            "diagnosis": "UNKNOWN",
            "status": "NEEDS_IMPROVEMENT",
            "lr_dir": "stay",
            "lr_step": 0.0,
            "hidden_layers": list(current_config["hidden_layers"]),
            "reasoning": "LLM error",
        }

        raw = invoke_llm(
            system_prompt("critic"),
            prompt,
            agent_name="Critic Evaluation Agent",
            cycle=cycle_number,
            log_filename=self.log_filename,
            max_retries=int(getattr(cfg, "MAX_RETRIES", 3)),
            retry_wait=15.0,
            cooldown=2.0,
        )

        if not raw.strip():
            return self._heuristic_fallback(
                default, overfit_ratio, val_mse, measured_latency, max_latency, current_config
            )

        result = dict(default)
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("DIAGNOSIS:"):
                result["diagnosis"] = line.split(":", 1)[1].strip()
            elif line.startswith("STATUS:"):
                result["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("LR_DIRECTION:"):
                result["lr_dir"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("LR_STEP:"):
                try:
                    result["lr_step"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("HIDDEN_LAYERS:"):
                try:
                    parsed = json.loads(line.split(":", 1)[1].strip())
                    if isinstance(parsed, list) and parsed:
                        result["hidden_layers"] = [int(x) for x in parsed]
                except Exception:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_fallback(
        default: Dict[str, Any],
        overfit_ratio: float,
        val_mse: float,
        measured_latency: float,
        max_latency: float,
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Mathematical stand-in when the API is unreachable.

        Applies the same Master Overfit Matrix thresholds the prompt describes,
        so the loop keeps making sane decisions offline.
        """
        result = dict(default)

        if measured_latency > max_latency:
            result["diagnosis"] = "LATENCY_VIOLATION"
            result["status"] = "REJECTED"
            result["hidden_layers"] = [
                max(cfg.HIDDEN_SIZE_MIN, int(h * 0.5)) for h in current_config["hidden_layers"]
            ]
            result["lr_dir"] = "stay"
            result["reasoning"] = (
                "Measured latency exceeds the customer limit; pruning layer widths."
            )
            return result

        if overfit_ratio >= 10.0:
            result["diagnosis"] = "CRITICAL_OVERFITTING"
            result["status"] = "REJECTED"
            result["lr_dir"] = "decrease"
            result["lr_step"] = max(current_config["learning_rate"] * 0.5, cfg.LEARNING_RATE_MIN)
            result["hidden_layers"] = [
                max(cfg.HIDDEN_SIZE_MIN, int(h * 0.5)) for h in current_config["hidden_layers"]
            ]
            result["reasoning"] = "Validation loss is 10x training loss; forcing an LR backoff and reset."
        elif overfit_ratio >= 7.0:
            result["diagnosis"] = "HIGH_OVERFITTING"
            result["hidden_layers"] = [
                max(cfg.HIDDEN_SIZE_MIN, int(h * 0.75)) for h in current_config["hidden_layers"]
            ]
            result["reasoning"] = "Large generalization gap; applying topological shrinkage."
        elif overfit_ratio >= 4.0:
            result["diagnosis"] = "MILD_OVERFITTING"
            result["reasoning"] = "Moderate generalization gap; increasing regularization."
        elif val_mse <= cfg.MSE_TARGET:
            result["diagnosis"] = "CONVERGING"
            result["status"] = "GOOD"
            result["reasoning"] = "Target accuracy and latency both satisfied."
        else:
            result["diagnosis"] = "NORMAL"
            result["reasoning"] = "Healthy ratio; continuing the architecture search."

        return result
