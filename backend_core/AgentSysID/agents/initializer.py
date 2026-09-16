"""
Initializer Agent – analyses dataset complexity and proposes initial
hyper-parameter search bounds / starting architecture.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.data.loader import ExcelDataLoader


class InitializerAgent:
    def __init__(self, loader: ExcelDataLoader, log_filename: Optional[str] = None):
        self.loader = loader
        self.log_filename = log_filename

    def determine_initial_setup(self) -> Dict[str, Any]:
        """Return a dict of starting hyper-parameters."""
        if not cfg.CHOOSE_VIA_LLM_INITIALIZER:
            return {
                "learning_rate": cfg.MANUAL_STARTING_LR,
                "hidden_layers": list(cfg.MANUAL_STARTING_HIDDEN_LAYERS),
                "activation": cfg.MANUAL_ACTIVATION,
                "dropout_rate": cfg.MANUAL_DROPOUT_RATE,
                "weight_decay": cfg.MANUAL_WEIGHT_DECAY,
                "reasoning": "Manual override (CHOOSE_VIA_LLM_INITIALIZER=False)",
            }

        system = (
            "You are the Initializer Agent for a neural system-identification framework. "
            "Given dataset statistics, propose a sensible starting architecture. "
            "Reply with a single JSON object containing keys: "
            "learning_rate (float), hidden_layers (list of ints), activation (str), "
            "dropout_rate (float), weight_decay (float), reasoning (one paragraph)."
        )
        user = (
            f"State dim: {self.loader.state_dim}\n"
            f"Action dim: {self.loader.action_dim}\n"
            f"Complexity: {self.loader.complexity_label} (score={self.loader.complexity_score})\n"
            f"Has true xdot: {self.loader.has_xdot}\n"
            f"Architecture preference: {cfg.NETWORK_ARCHITECTURE}\n"
            f"Customer context: {cfg.CUSTOMER_SYSTEM_DESCRIPTION}\n"
            f"Allowed activations: {cfg.AVAILABLE_ACTIVATIONS}\n"
            f"Hidden size bounds: [{cfg.HIDDEN_SIZE_MIN}, {cfg.HIDDEN_SIZE_MAX}]\n"
            f"Layer count bounds: [{cfg.NUM_LAYERS_MIN}, {cfg.NUM_LAYERS_MAX}]\n"
            f"LR bounds: [{cfg.LEARNING_RATE_MIN}, {cfg.LEARNING_RATE_MAX}]"
        )

        raw = invoke_llm(
            system, user, agent_name="Initializer", log_filename=self.log_filename
        )
        try:
            # Extract JSON blob
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "learning_rate": float(data.get("learning_rate", cfg.MANUAL_STARTING_LR)),
                    "hidden_layers": [int(x) for x in data.get("hidden_layers", cfg.MANUAL_STARTING_HIDDEN_LAYERS)],
                    "activation": str(data.get("activation", cfg.MANUAL_ACTIVATION)).lower(),
                    "dropout_rate": float(data.get("dropout_rate", cfg.DROPOUT_RATE)),
                    "weight_decay": float(data.get("weight_decay", cfg.WEIGHT_DECAY)),
                    "reasoning": data.get("reasoning", ""),
                }
        except Exception as e:
            print(f"   ⚠️ Initializer parse failure: {e}. Falling back to defaults.")

        return {
            "learning_rate": cfg.MANUAL_STARTING_LR,
            "hidden_layers": list(cfg.MANUAL_STARTING_HIDDEN_LAYERS),
            "activation": cfg.MANUAL_ACTIVATION,
            "dropout_rate": cfg.DROPOUT_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "reasoning": "Fallback defaults after parse failure",
        }
