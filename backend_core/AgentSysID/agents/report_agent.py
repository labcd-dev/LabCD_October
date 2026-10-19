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
        physics_mode: str = "pinn",
        max_latency: Optional[float] = None,
        success_score: Optional[float] = None,
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
        physics_label = {"pcnn": "Physics-Constrained", "pinn": "Physics-Informed"}.get(
            str(physics_mode or "pinn").lower(), "Physics-Informed"
        )
        arch_type = f"{physics_label} {arch_name} ({str(physics_mode).upper()})" if use_pinn else f"Data-Driven {arch_name}"

        customer_context = str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip()
        customer_prompt_block = (
            f"- Customer System Context: '{customer_context}'" if customer_context else ""
        )

        hidden_layers = best_config.get("hidden_layers", "Unknown") if best_config else "Unknown"

        score_text = (f"{float(success_score):.1f} / 100" if success_score is not None
                      else "Unavailable (no valid held-out rollout)")
        status_text = {
            "STABLE & HIGH-FIDELITY": "Legacy high fit/rollout rating",
            "STABLE & ACCEPTABLE": "Legacy acceptable fit/rollout rating",
            "UNSTABLE ROLLOUT": "Legacy weak rollout rating",
            "UNSTABLE / FAILED": "Legacy low fit/rollout rating",
        }.get(str(model_status).strip().upper(), model_status or "Not rated")
        latency_text = (f"{float(latency):.3f} ms; configured comparison limit {max_latency:.3f} ms"
                        if float(latency) <= max_latency else
                        f"{float(latency):.3f} ms; above configured comparison limit {max_latency:.3f} ms")
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
            success_score_text=score_text,
            latency_comparison=latency_text,
            model_status=status_text,
        )

        # --- Professional fallback text (used when the API is unreachable) --
        abstract = (
            f"This report presents the results of an automated System Identification run utilizing a "
            f"{arch_type} architecture to model a dataset characterized as {complexity_label}. The tuning "
            f"methodology employed was optimized for a state dimension of {state_dim} and an action dimension "
            f"of {action_dim}, ultimately selecting a final topology of {hidden_layers}. The model achieved a "
            f"validation Mean Squared Error (MSE) of {best_mse:.6f} and a Root Mean Squared Error (RMSE) of "
            f"{best_rmse:.6f}. These values summarize the saved validation measurements for this run."
        )

        score_summary = (f"The heuristic fit and held-out rollout score is {float(success_score):.1f}/100 "
                         f"({status_text}). It combines validation fit with the saved rollout comparison."
                         if success_score is not None else
                         "A composite fit and rollout score is unavailable because no valid held-out rollout was saved.")
        timing_summary = (f"Measured inference latency was {latency:.3f} ms against the configured "
                          f"comparison limit of {max_latency:.3f} ms."
                          if latency <= max_latency else
                          f"Measured inference latency was {latency:.3f} ms, above the configured "
                          f"comparison limit of {max_latency:.3f} ms.")
        conclusion = (
            f"{score_summary} {timing_summary} "
            "The score and latency do not establish closed-loop stability, control safety, or readiness for "
            "MPC, CBF, or hardware use. Before controller testing, assess "
            "all held-out trajectories and run separate closed-loop checks with the intended sampling rate, "
            "actuator limits, operating conditions, and target hardware."
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

        # Keep the deployment interpretation deterministic and evidence-bounded.
        # The language model may improve the abstract, but cannot certify stability
        # or controller readiness from these run metrics.
        return abstract, conclusion
