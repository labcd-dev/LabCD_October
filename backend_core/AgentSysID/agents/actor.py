"""
Actor Agent – proposes the next hyper-parameter configuration given Critic feedback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm


class ActorAgent:
    def __init__(self, activation: str, log_filename: Optional[str] = None):
        self.activation = activation
        self.log_filename = log_filename

    def propose(
        self,
        current_config: Dict[str, Any],
        critic_feedback: Dict[str, Any],
        cycle: int,
        recent_failures: str = "",
    ) -> Dict[str, Any]:
        """Return a new config dict (learning_rate, hidden_layers, dropout_rate, weight_decay, activation)."""
        system = (
            "You are the Actor Agent in an actor-critic hyper-parameter search for "
            "neural system identification. Propose the NEXT configuration as JSON with keys: "
            "learning_rate, hidden_layers (list of ints), dropout_rate, weight_decay, activation, reasoning."
        )
        user = (
            f"Current config: {json.dumps(current_config)}\n"
            f"Critic status: {critic_feedback.get('status')}\n"
            f"Critic reason: {critic_feedback.get('reason')}\n"
            f"Critic suggestions: {critic_feedback.get('suggestions')}\n"
            f"Recent failures memory:\n{recent_failures}\n"
            f"Bounds – LR [{cfg.LEARNING_RATE_MIN}, {cfg.LEARNING_RATE_MAX}], "
            f"hidden [{cfg.HIDDEN_SIZE_MIN}, {cfg.HIDDEN_SIZE_MAX}], "
            f"layers [{cfg.NUM_LAYERS_MIN}, {cfg.NUM_LAYERS_MAX}], "
            f"dropout [{cfg.DROPOUT_RATE_MIN}, {cfg.DROPOUT_RATE_MAX}], "
            f"weight_decay [{cfg.WEIGHT_DECAY_MIN}, {cfg.WEIGHT_DECAY_MAX}]\n"
            f"Allowed activations: {cfg.AVAILABLE_ACTIVATIONS}"
        )
        raw = invoke_llm(
            system, user, agent_name="Actor", cycle=cycle, log_filename=self.log_filename
        )
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "learning_rate": float(data.get("learning_rate", current_config.get("learning_rate", 1e-3))),
                    "hidden_layers": [int(x) for x in data.get("hidden_layers", current_config.get("hidden_layers", [128, 128]))],
                    "dropout_rate": float(data.get("dropout_rate", current_config.get("dropout_rate", 0.1))),
                    "weight_decay": float(data.get("weight_decay", current_config.get("weight_decay", 1e-4))),
                    "activation": str(data.get("activation", self.activation)).lower(),
                    "reasoning": data.get("reasoning", ""),
                }
        except Exception as e:
            print(f"   ⚠️ Actor parse failure: {e}")

        # Conservative nudge
        cfg_out = dict(current_config)
        cfg_out["learning_rate"] = max(cfg.LEARNING_RATE_MIN, current_config.get("learning_rate", 1e-3) * 0.8)
        return cfg_out
