"""
Initializer Agent.

Analyses dataset complexity (Tier 1-5), physical jump statistics and the
customer context, then proposes the starting hyper-parameters *and* the bounded
search space the Actor / Explorer are allowed to move inside.

Every value the model returns is clamped against the client-authorised outer
limits in ``config`` before it is handed back, so a hallucinated learning rate
can never escape the contract.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm, strip_code_fences
from backend_core.AgentSysID.agents.prompt_library import render, system_prompt
from backend_core.AgentSysID.data.loader import ExcelDataLoader


class InitializerAgent:
    def __init__(self, loader: ExcelDataLoader, log_filename: Optional[str] = None):
        self.loader = loader
        self.log_filename = log_filename

    # ------------------------------------------------------------------
    def _default_config(self, state_dim: int) -> Dict[str, Any]:
        # Fall back to the thresholds the loader already calibrated from the
        # data rather than an arbitrary constant, so a failed agent call can
        # never shatter a continuous trajectory into fragments.
        calibrated = getattr(self.loader, "reset_threshold", None)
        if isinstance(calibrated, (list, tuple)) and len(calibrated) == state_dim:
            fallback_reset: Any = [float(x) for x in calibrated]
        elif isinstance(calibrated, (int, float)):
            fallback_reset = [float(calibrated)] * max(state_dim, 1)
        else:
            fallback_reset = [99999.0] * max(state_dim, 1)

        return {
            "learning_rate": 0.001,
            "hidden_layers": [64],
            "activation": "relu",
            "epochs": 300,
            "batch_size": 128,
            "early_stop_patience": 25,
            "lr_search_min": cfg.LEARNING_RATE_MIN,
            "lr_search_max": cfg.LEARNING_RATE_MAX,
            "hidden_size_search_min": cfg.HIDDEN_SIZE_MIN,
            "hidden_size_search_max": cfg.HIDDEN_SIZE_MAX,
            "num_layers_search_min": cfg.NUM_LAYERS_MIN,
            "num_layers_search_max": cfg.NUM_LAYERS_MAX,
            "dropout_rate": 0.0,
            "weight_decay": 0.0001,
            "lr_reduce_factor": 0.5,
            "use_state_filter": False,
            "auto_filter_percentiles": [2, 98],
            "derivative_filter_tau": 0.005,
            "reset_threshold": fallback_reset,
            "reasoning": "Heuristic fallback active. No reasoning generated.",
        }

    # ------------------------------------------------------------------
    def determine_initial_setup(self) -> Dict[str, Any]:
        """Return the starting configuration plus the authorised search bounds."""
        df = self.loader.df
        num_samples = len(df)
        state_dim = self.loader.state_dim
        action_dim = self.loader.action_dim

        default_config = self._default_config(state_dim)

        # Manual presets bypass the agent entirely (legacy CHOOSE_VIA_LLM_INITIALIZER)
        if not getattr(cfg, "CHOOSE_VIA_LLM_INITIALIZER", True):
            manual = dict(default_config)
            manual.update(
                {
                    "learning_rate": cfg.MANUAL_STARTING_LR,
                    "hidden_layers": list(cfg.MANUAL_STARTING_HIDDEN_LAYERS),
                    "activation": cfg.MANUAL_ACTIVATION.lower(),
                    "dropout_rate": cfg.MANUAL_DROPOUT_RATE,
                    "weight_decay": cfg.MANUAL_WEIGHT_DECAY,
                    "reasoning": "Manual presets active (CHOOSE_VIA_LLM_INITIALIZER=False).",
                }
            )
            return manual

        system_complexity = state_dim + action_dim
        samples_per_dim = int(num_samples / max(1, system_complexity))

        time_col = getattr(cfg, "TIME_COLUMN", "time")
        if time_col in df.columns and len(df) > 1:
            dt_mean: Any = float(np.mean(np.diff(df[time_col].values)))
        else:
            dt_mean = "Unknown"

        num_angles = len(getattr(self.loader, "angle_indices", []))

        # --- Normal vs teleportation boundaries ---------------------------
        is_multi = getattr(cfg, "MULTI_TRAJECTORY", True)
        if is_multi and self.loader.state_cols:
            diffs = df[self.loader.state_cols].diff().abs()
            normal_jumps = diffs.quantile(0.999).fillna(0.0).values
            teleport_jumps = diffs.max().fillna(0.0).values
            normal_jumps_str = f"[{', '.join(f'{v:.4f}' for v in normal_jumps)}]"
            teleport_jumps_str = f"[{', '.join(f'{v:.4f}' for v in teleport_jumps)}]"
            jump_data_str = (
                f"- Extreme Normal Driving Jumps (99.9th percentile): {normal_jumps_str}\n"
                f"  - Teleportation/Trajectory Jumps (Absolute Max): {teleport_jumps_str}"
            )
            reset_rule = (
                "RESET_THRESHOLD: (list of floats) Output limits (one per state) "
                "positioned between Normal Jumps and Teleportation Jumps."
            )
        else:
            jump_data_str = (
                "- Dataset Mode: Single continuous trajectory (no boundary teleportations)."
            )
            reset_rule = (
                "RESET_THRESHOLD: (list of floats) Output [99999.0] for each state "
                "dimension to disable trajectory reset."
            )

        customer_context = str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip()
        customer_prompt_block = ""
        if customer_context:
            customer_prompt_block = (
                f'\n  CUSTOMER SYSTEM DESCRIPTION:\n  "{customer_context}"\n'
                "  (Bias configuration towards physical system characteristics and "
                "noise profiles described above.)\n"
            )

        overrides = getattr(cfg, "USER_OVERRIDES", {}) or {}
        override_block = ""
        if overrides:
            override_block = (
                f"\n  CLIENT HAS LOCKED THE FOLLOWING PARAMETERS:\n  {overrides}\n"
                "  (You MUST output these exact locked values in your response.)\n"
            )

        mode_str = str(getattr(cfg, "RUN_MODE", "regular")).lower()
        if mode_str == "fast":
            reasoning_req = (
                "Write exactly TWO concise sentences on a SINGLE continuous line "
                "explaining your primary choices."
            )
        else:
            reasoning_req = (
                "Write exactly ONE concise paragraph on a SINGLE continuous line "
                "explaining your logic for all parameters."
            )

        complexity_tier = getattr(self.loader, "complexity_tier", 2)
        complexity_label = getattr(self.loader, "complexity_label", "Unknown")
        optimization_goal = str(getattr(cfg, "OPTIMIZATION_GOAL", "balanced")).strip().lower()
        goal_guidance = {
            "balanced": "Balance validation fit, generalization, and practical model size.",
            "accuracy": "Prioritize validation fit while avoiding overfitting; use the data-supported capacity and regularization.",
            "speed": "Favor a compact, fast-inference starting model and avoid unnecessary width or depth, while preserving a useful validation fit.",
            "compact": "Favor the smallest reasonable model and a narrower search; avoid spending capacity the measurements do not support.",
        }.get(optimization_goal, "Balance validation fit, generalization, and practical model size.")

        prompt = render(
            "initializer",
            customer_prompt_block=customer_prompt_block,
            optimization_goal=optimization_goal,
            goal_guidance=goal_guidance,
            override_block=override_block,
            complexity_label=complexity_label,
            complexity_tier=complexity_tier,
            num_samples=num_samples,
            state_dim=state_dim,
            action_dim=action_dim,
            samples_per_dim=samples_per_dim,
            dt_mean=dt_mean,
            num_angles=num_angles,
            jump_data_str=jump_data_str,
            lr_min=cfg.LEARNING_RATE_MIN,
            lr_max=cfg.LEARNING_RATE_MAX,
            hs_min=cfg.HIDDEN_SIZE_MIN,
            hs_max=cfg.HIDDEN_SIZE_MAX,
            nl_min=cfg.NUM_LAYERS_MIN,
            nl_max=cfg.NUM_LAYERS_MAX,
            reset_rule=reset_rule,
            reasoning_req=reasoning_req,
        )
        system_content = system_prompt("initializer")

        raw_text = strip_code_fences(
            invoke_llm(
                system_content,
                prompt,
                agent_name="Initializer Configuration Agent",
                log_filename=self.log_filename,
            )
        )
        if not raw_text:
            print("   ⚠️ Initializer Agent unavailable. Falling back to default heuristics.")
            return default_config

        try:
            result = self._parse(raw_text, default_config, state_dim)
            return self._clamp(result)
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️ Initializer Agent parsing failure: {exc}. Falling back to default heuristics.")
            return default_config

    # ------------------------------------------------------------------
    def _parse(
        self, raw_text: str, default_config: Dict[str, Any], state_dim: int
    ) -> Dict[str, Any]:
        """Parse the strict KEY: VALUE contract, keeping defaults on any miss."""
        result = dict(default_config)

        float_keys = {
            "LEARNING_RATE": "learning_rate",
            "LR_SEARCH_MIN": "lr_search_min",
            "LR_SEARCH_MAX": "lr_search_max",
            "DROPOUT_RATE": "dropout_rate",
            "WEIGHT_DECAY": "weight_decay",
            "LR_REDUCE_FACTOR": "lr_reduce_factor",
            "DERIVATIVE_FILTER_TAU": "derivative_filter_tau",
        }
        int_keys = {
            "EPOCHS": "epochs",
            "BATCH_SIZE": "batch_size",
            "EARLY_STOP_PATIENCE": "early_stop_patience",
            "HIDDEN_SIZE_SEARCH_MIN": "hidden_size_search_min",
            "HIDDEN_SIZE_SEARCH_MAX": "hidden_size_search_max",
            "NUM_LAYERS_SEARCH_MIN": "num_layers_search_min",
            "NUM_LAYERS_SEARCH_MAX": "num_layers_search_max",
        }

        for line in raw_text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, val_str = line.split(":", 1)
            key = key.strip().upper()
            val_str = val_str.strip()

            if key in float_keys:
                try:
                    result[float_keys[key]] = float(val_str)
                except ValueError:
                    pass
            elif key in int_keys:
                try:
                    result[int_keys[key]] = int(float(val_str))
                except ValueError:
                    pass
            elif key == "HIDDEN_LAYERS":
                try:
                    parsed = json.loads(val_str)
                    if isinstance(parsed, list) and parsed:
                        result["hidden_layers"] = [int(x) for x in parsed]
                except Exception:
                    pass
            elif key == "ACTIVATION":
                act_val = val_str.lower()
                if act_val in cfg.AVAILABLE_ACTIVATIONS:
                    result["activation"] = act_val
            elif key == "USE_STATE_FILTER":
                result["use_state_filter"] = val_str.lower() == "true"
            elif key == "AUTO_FILTER_PERCENTILES":
                try:
                    result["auto_filter_percentiles"] = json.loads(val_str)
                except Exception:
                    pass
            elif key == "RESET_THRESHOLD":
                try:
                    if "[" in val_str:
                        result["reset_threshold"] = json.loads(val_str)
                    else:
                        result["reset_threshold"] = [float(val_str)] * max(state_dim, 1)
                except Exception:
                    pass
            elif key == "REASONING":
                result["reasoning"] = val_str

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(result: Dict[str, Any]) -> Dict[str, Any]:
        """Mathematical clamping and bounds verification against live config limits."""
        result["lr_search_min"] = float(
            np.clip(result["lr_search_min"], cfg.LEARNING_RATE_MIN, cfg.LEARNING_RATE_MAX)
        )
        result["lr_search_max"] = float(
            np.clip(result["lr_search_max"], cfg.LEARNING_RATE_MIN, cfg.LEARNING_RATE_MAX)
        )
        if result["lr_search_min"] > result["lr_search_max"]:
            result["lr_search_min"], result["lr_search_max"] = (
                result["lr_search_max"],
                result["lr_search_min"],
            )

        result["hidden_size_search_min"] = int(
            np.clip(result["hidden_size_search_min"], cfg.HIDDEN_SIZE_MIN, cfg.HIDDEN_SIZE_MAX)
        )
        result["hidden_size_search_max"] = int(
            np.clip(result["hidden_size_search_max"], cfg.HIDDEN_SIZE_MIN, cfg.HIDDEN_SIZE_MAX)
        )
        if result["hidden_size_search_min"] > result["hidden_size_search_max"]:
            result["hidden_size_search_min"], result["hidden_size_search_max"] = (
                result["hidden_size_search_max"],
                result["hidden_size_search_min"],
            )

        result["num_layers_search_min"] = int(
            np.clip(result["num_layers_search_min"], cfg.NUM_LAYERS_MIN, cfg.NUM_LAYERS_MAX)
        )
        result["num_layers_search_max"] = int(
            np.clip(result["num_layers_search_max"], cfg.NUM_LAYERS_MIN, cfg.NUM_LAYERS_MAX)
        )
        if result["num_layers_search_min"] > result["num_layers_search_max"]:
            result["num_layers_search_min"], result["num_layers_search_max"] = (
                result["num_layers_search_max"],
                result["num_layers_search_min"],
            )

        result["learning_rate"] = float(
            np.clip(result["learning_rate"], result["lr_search_min"], result["lr_search_max"])
        )

        hidden = [
            int(np.clip(int(x), result["hidden_size_search_min"], result["hidden_size_search_max"]))
            for x in (result["hidden_layers"] or [64])
        ]
        if len(hidden) < result["num_layers_search_min"]:
            hidden = (hidden * result["num_layers_search_min"])[: result["num_layers_search_min"]]
        if len(hidden) > result["num_layers_search_max"]:
            hidden = hidden[: result["num_layers_search_max"]]
        result["hidden_layers"] = hidden

        return result
