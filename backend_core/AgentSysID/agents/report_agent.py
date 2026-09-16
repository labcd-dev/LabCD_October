"""
Report Agent – authors a short professional engineering summary of the run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend_core.AgentSysID.agents.llm_base import invoke_llm


class ReportAgent:
    def __init__(self, log_filename: Optional[str] = None):
        self.log_filename = log_filename

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
        complexity_label: str = "Unknown",
    ) -> str:
        system = (
            "You are a senior control-systems engineer writing a concise technical abstract "
            "for a system-identification report. Use clear professional language. "
            "Do not invent numbers that were not provided."
        )
        user = (
            f"Environment / dataset: {env_name}\n"
            f"State dim: {state_dim}, Action dim: {action_dim}\n"
            f"Complexity: {complexity_label}\n"
            f"Best validation MSE: {best_mse:.6e}, RMSE: {best_rmse:.6e}\n"
            f"Inference latency: {latency:.3f} ms\n"
            f"Architecture: {best_config}\n"
            f"Physics-informed (PINN): {use_pinn}\n"
            "Write a 150-250 word abstract suitable for the executive summary of the PDF report."
        )
        text = invoke_llm(
            system, user, agent_name="Report", log_filename=self.log_filename
        )
        if not text.strip():
            text = (
                f"Automated system identification completed for '{env_name}' "
                f"({state_dim} states, {action_dim} actions, {complexity_label}). "
                f"Best validation MSE reached {best_mse:.4e} (RMSE {best_rmse:.4e}) "
                f"with measured inference latency of {latency:.2f} ms. "
                f"Final architecture: {best_config}. "
                f"PINN physics residual was {'enabled' if use_pinn else 'disabled'}."
            )
        return text.strip()
