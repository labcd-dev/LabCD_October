"""
Explorer Agent – detects stagnation and proposes radical architectural changes
to escape local minima.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm


class ExplorerAgent:
    def __init__(self, log_filename: Optional[str] = None):
        self.log_filename = log_filename

    def propose_escape(
        self,
        performance_history: List[Dict[str, Any]],
        current_config: Dict[str, Any],
        cycle: int,
    ) -> Dict[str, Any]:
        """Return a radically different config when the search is stuck."""
        recent = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        system = (
            "You are the Explorer Agent. The search has stagnated. Propose a RADICAL "
            "new architecture (different depth / width / activation) as JSON with keys: "
            "learning_rate, hidden_layers, dropout_rate, weight_decay, activation, reasoning."
        )
        user = (
            f"Recent performance: {json.dumps(recent, default=str)}\n"
            f"Current config: {json.dumps(current_config)}\n"
            f"Allowed activations: {cfg.AVAILABLE_ACTIVATIONS}\n"
            f"Bounds: hidden [{cfg.HIDDEN_SIZE_MIN},{cfg.HIDDEN_SIZE_MAX}], "
            f"layers [{cfg.NUM_LAYERS_MIN},{cfg.NUM_LAYERS_MAX}]"
        )
        raw = invoke_llm(
            system, user, agent_name="Explorer", cycle=cycle, log_filename=self.log_filename
        )
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "learning_rate": float(data.get("learning_rate", 1e-3)),
                    "hidden_layers": [int(x) for x in data.get("hidden_layers", [64, 64, 64])],
                    "dropout_rate": float(data.get("dropout_rate", 0.15)),
                    "weight_decay": float(data.get("weight_decay", 1e-3)),
                    "activation": str(data.get("activation", "elu")).lower(),
                    "reasoning": data.get("reasoning", "Explorer escape"),
                }
        except Exception:
            pass

        # Deterministic radical alternative
        return {
            "learning_rate": 5e-4,
            "hidden_layers": [64, 128, 64],
            "dropout_rate": 0.2,
            "weight_decay": 1e-3,
            "activation": "swish",
            "reasoning": "Deterministic Explorer fallback – inverted topology",
        }
