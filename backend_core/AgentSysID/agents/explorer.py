"""
Explorer Agent – the repulsive barrier.

Summoned when the Actor-Critic loop stagnates (three cycles without a new
best). It treats the trapped configuration as an obstacle and generates a
structurally *opposite* topology: shallow-and-wide becomes deep-and-narrow and
vice versa, with a large learning-rate displacement to leave the gradient
trench.

If the LLM is unavailable or proposes something already visited, a strict
mathematical inversion via bounds reflection takes over, so the loop always
escapes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.agents.prompt_library import render, system_prompt
from backend_core.AgentSysID.training.tracker import BestConfigTracker


class ExplorerAgent:
    def __init__(
        self,
        initial_config: Optional[Dict[str, Any]] = None,
        log_filename: Optional[str] = None,
    ):
        self.log_filename = log_filename

        if initial_config is not None:
            self.lr_min = initial_config.get("lr_search_min", cfg.LEARNING_RATE_MIN)
            self.lr_max = initial_config.get("lr_search_max", cfg.LEARNING_RATE_MAX)
            self.hs_min = initial_config.get("hidden_size_search_min", cfg.HIDDEN_SIZE_MIN)
            self.hs_max = initial_config.get("hidden_size_search_max", cfg.HIDDEN_SIZE_MAX)
            self.num_layers_min = initial_config.get("num_layers_search_min", cfg.NUM_LAYERS_MIN)
            self.num_layers_max = initial_config.get("num_layers_search_max", cfg.NUM_LAYERS_MAX)
        else:
            self.lr_min = cfg.LEARNING_RATE_MIN
            self.lr_max = cfg.LEARNING_RATE_MAX
            self.hs_min = cfg.HIDDEN_SIZE_MIN
            self.hs_max = cfg.HIDDEN_SIZE_MAX
            self.num_layers_min = cfg.NUM_LAYERS_MIN
            self.num_layers_max = cfg.NUM_LAYERS_MAX

    # ------------------------------------------------------------------
    @staticmethod
    def _visited_key(lr: float, hidden: Iterable[int]) -> str:
        lr_rounded = round(lr / cfg.LR_TOLERANCE) * cfg.LR_TOLERANCE
        return f"LR={lr_rounded:.5f}, HL={tuple(hidden)}"

    def _clamp_hidden(self, hidden: List[int]) -> List[int]:
        hidden = [int(np.clip(int(x), self.hs_min, self.hs_max)) for x in hidden] or [
            int(self.hs_min)
        ]
        if len(hidden) < self.num_layers_min:
            hidden = [hidden[0]] * self.num_layers_min
        if len(hidden) > self.num_layers_max:
            hidden = hidden[: self.num_layers_max]
        return hidden

    # ------------------------------------------------------------------
    def generate_radical_escape(
        self,
        tracker: BestConfigTracker,
        stuck_config: Dict[str, Any],
        visited_configs: Iterable[str],
        cycle_number: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Return ``(new_config, reasoning)`` – a topology far from the trap."""
        visited = set(visited_configs or [])
        stuck_lr = stuck_config["learning_rate"]
        stuck_hidden = list(stuck_config["hidden_layers"])

        if cycle_number is None:
            cycle_number = len(visited) + 1

        # --- Customer context feed, gated by run mode ---------------------
        run_mode = str(getattr(cfg, "RUN_MODE", "regular")).strip().lower()
        customer_context = str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip()
        system_context_block = ""
        if customer_context and (
            run_mode == "heavy" or (run_mode == "regular" and cycle_number <= 5)
        ):
            # heavy = always, regular = first five cycles, fast = never
            system_context_block = f"\n  CUSTOMER SYSTEM CONTEXT: '{customer_context}'"

        prompt = render(
            "explorer",
            system_context_block=system_context_block,
            stuck_lr=stuck_lr,
            stuck_hidden=stuck_hidden,
            failures_str=tracker.get_recent_failures_str(),
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            hs_min=self.hs_min,
            hs_max=self.hs_max,
            nl_min=self.num_layers_min,
            nl_max=self.num_layers_max,
        )

        new_lr = stuck_lr
        new_hidden = stuck_hidden
        reasoning = "LLM Generation Failed. Applying mathematical inversion."

        raw = invoke_llm(
            system_prompt("explorer"),
            prompt,
            agent_name="Explorer Escape Agent",
            cycle=cycle_number,
            log_filename=self.log_filename,
        )

        if raw.strip():
            try:
                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("LEARNING_RATE:"):
                        new_lr = float(line.split(":", 1)[1].strip())
                    elif line.startswith("HIDDEN_LAYERS:"):
                        parsed = json.loads(line.split(":", 1)[1].strip())
                        if isinstance(parsed, list) and parsed:
                            new_hidden = [int(x) for x in parsed]
                    elif line.startswith("REASONING:"):
                        reasoning = line.split(":", 1)[1].strip()

                new_lr = float(np.clip(new_lr, self.lr_min, self.lr_max))
                new_hidden = self._clamp_hidden(new_hidden)

                candidate = dict(stuck_config)
                candidate["learning_rate"] = new_lr
                candidate["hidden_layers"] = new_hidden

                if self._visited_key(new_lr, new_hidden) not in visited:
                    return candidate, reasoning
            except Exception as exc:  # noqa: BLE001
                print(f"        ⚠️ Explorer LLM parse failed ({exc}). Executing mathematical inversion...")

        # --- FALLBACK: strict mathematical inversion -----------------------
        # Physically force the opposite dimensions of the trapped topology.
        mid_layers = (self.num_layers_max + self.num_layers_min) / 2
        target_layers = (
            self.num_layers_max if len(stuck_hidden) <= mid_layers else self.num_layers_min
        )
        mid_neurons = (self.hs_max + self.hs_min) / 2
        target_neurons = (
            self.hs_min if float(np.mean(stuck_hidden)) >= mid_neurons else self.hs_max
        )

        for _ in range(500):
            candidate_hidden = [
                int(np.random.randint(target_neurons - 16, target_neurons + 16))
                for _ in range(int(target_layers))
            ]
            candidate_hidden = self._clamp_hidden(candidate_hidden)
            candidate_lr = float(np.random.uniform(self.lr_min, self.lr_max))

            candidate = dict(stuck_config)
            candidate["learning_rate"] = candidate_lr
            candidate["hidden_layers"] = candidate_hidden

            if self._visited_key(candidate_lr, candidate_hidden) not in visited:
                return candidate, "Forced mathematical inversion via bounds reflection."

        # Exhausted the bounded space: return the last inversion anyway.
        return candidate, "Search space exhausted; returning the nearest inversion."

    # ------------------------------------------------------------------
    def propose_escape(
        self,
        performance_history: Optional[List[Dict[str, Any]]] = None,
        current_config: Optional[Dict[str, Any]] = None,
        cycle: Optional[int] = None,
        tracker: Optional[BestConfigTracker] = None,
        visited_configs: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper returning only the escape configuration."""
        config, _ = self.generate_radical_escape(
            tracker=tracker or BestConfigTracker(),
            stuck_config=current_config or {"learning_rate": cfg.MANUAL_STARTING_LR, "hidden_layers": [64]},
            visited_configs=visited_configs or [],
            cycle_number=cycle,
        )
        return config
