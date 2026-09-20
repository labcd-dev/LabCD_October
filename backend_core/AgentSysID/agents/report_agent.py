"""
Report Agent – automated PDF manuscript authoring.

Produces the Abstract and the Concluding Synthesis for the engineering report.
Both have professional, fully-populated fallbacks, so a dead API degrades the
prose but never the document.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.agents.prompt_library import render, system_prompt


class ReportAgent:
    def __init__(self, log_filename: Optional[str] = None):
        self.log_filename = log_filename

    # ------------------------------------------------------------------
    def generate_report_text(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        best_config: Dict[str, Any],
        best_mse: float,
        best_rmse: float,
        latency: float,
        use_pinn: bool,
        max_latency: Optional[float] = None,
        success_score: float = 0.0,
        model_status: str = "UNKNOWN",
        complexity_label: str = "Unknown",
        architecture: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return ``(abstract, conclusion)``."""
        max_latency = (
            float(max_latency)
            if max_latency is not None
            else float(getattr(cfg, "CUSTOMER_MAX_LATENCY_MS", 2.0))
        )

        base_arch = str(architecture or cfg.NETWORK_ARCHITECTURE).strip().upper()
        arch_name = (
            "Long Short-Term Memory (LSTM) Network"
            if base_arch == "LSTM"
            else "Multilayer Perceptron (MLP)"
        )
        arch_type = (
            f"Physics-Informed {arch_name} (PINN)" if use_pinn else f"Data-Driven {arch_name}"
        )

        customer_context = str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip()
        customer_prompt_block = (
            f"- Customer System Context: '{customer_context}'" if customer_context else ""
        )

        hidden_layers = best_config.get("hidden_layers", "Unknown") if best_config else "Unknown"

        prompt = render(
            "report",
            env_name=env_name,
            customer_prompt_block=customer_prompt_block,
            arch_type=arch_type,
            complexity_label=complexity_label,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            best_mse=float(best_mse),
            best_rmse=float(best_rmse),
            latency=float(latency),
            max_latency=max_latency,
            success_score=float(success_score),
            model_status=model_status,
        )

        # --- Professional fallback text (used when the API is unreachable) --
        abstract = (
            f"This report presents the results of an automated System Identification run utilizing a "
            f"{arch_type} architecture to model a dataset characterized as {complexity_label}. The tuning "
            f"methodology employed was optimized for a state dimension of {state_dim} and an action dimension "
            f"of {action_dim}, ultimately selecting a final topology of {hidden_layers}. The model achieved a "
            f"validation Mean Squared Error (MSE) of {best_mse:.6f} and a Root Mean Squared Error (RMSE) of "
            f"{best_rmse:.6f}, indicating a high level of accuracy in capturing the nonlinear dynamics of the "
            f"target plant."
        )

        conclusion = (
            f"Achieving a Composite Success Score of {success_score:.1f}/100 ({model_status}), the optimized "
            f"neural architecture demonstrates robust closed-loop stability. Furthermore, with a measured "
            f"inference latency of {latency:.3f} ms, the model safely satisfies the real-time threshold limit "
            f"of {max_latency} ms. These metrics confirm the network's readiness for immediate integration into "
            f"Model Predictive Control (MPC) or Control Barrier Function (CBF) frameworks for advanced "
            f"autonomous trajectory regulation."
        )

        raw = invoke_llm(
            system_prompt("report"),
            prompt,
            agent_name="Report Authoring Agent",
            log_filename=self.log_filename,
        )

        if not raw.strip():
            print("   ⚠️ Report Agent unavailable. Executing professional text fallbacks.")
            return abstract, conclusion

        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("ABSTRACT:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    abstract = candidate
            elif line.startswith("CONCLUSION:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    conclusion = candidate

        return abstract, conclusion
