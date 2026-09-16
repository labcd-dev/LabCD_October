"""
Data Inspector – Human-in-the-Loop agent.

Examines the loaded dataset for mathematical anomalies and, when interactive,
asks the engineer for clarification / column drops.
"""

from __future__ import annotations

from typing import List, Optional

from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.data.loader import ExcelDataLoader


def run_data_inspector_agent(
    loader: ExcelDataLoader,
    interactive: bool = True,
    log_filename: Optional[str] = None,
) -> ExcelDataLoader:
    """
    Run the Data Inspector HIL step.

    Parameters
    ----------
    loader : ExcelDataLoader
    interactive : bool
        If False (CI / headless), skip the input() loop and only print findings.
    log_filename : optional path for agent log
    """
    print("\n" + "=" * 60)
    print("🔍 DATA INSPECTOR AGENT (Human-in-the-Loop)")
    print("=" * 60)

    findings: List[str] = []
    if loader.state_dim == 0:
        findings.append("No state columns (s_*) detected.")
    if loader.action_dim == 0:
        findings.append("No action columns (a_*) detected – open-loop identification only.")
    if not loader.has_xdot:
        findings.append("No true xdot_* columns – derivatives will be estimated (noisy).")
    if loader.complexity_score >= 4:
        findings.append(f"High complexity ({loader.complexity_label}) – expect longer tuning.")

    if findings:
        print("   Findings:")
        for f in findings:
            print(f"   • {f}")
    else:
        print("   ✅ No critical data issues detected.")

    # Optional LLM-assisted question (kept lightweight)
    if findings and interactive:
        system = (
            "You are the Data Inspector Agent for a deep-learning system-identification "
            "framework. Given a short list of data issues, ask the engineer ONE concise "
            "clarifying question or recommend a concrete column drop. Reply in plain text."
        )
        user = "Issues found:\n" + "\n".join(f"- {f}" for f in findings)
        question = invoke_llm(system, user, agent_name="Data Inspector", log_filename=log_filename)
        if question:
            print(f"\n🤖 Agent: {question.strip()}")
            try:
                answer = input("   Your reply (or press Enter to continue): ").strip()
                if answer.lower().startswith("drop "):
                    cols = [c.strip() for c in answer[5:].split(",")]
                    loader.drop_columns(cols)
            except EOFError:
                # Non-interactive terminal
                pass

    print("=" * 60 + "\n")
    return loader
