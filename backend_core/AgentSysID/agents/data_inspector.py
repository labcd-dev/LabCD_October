"""
Data Inspector – Human-in-the-Loop (HIL) agent.

Reads the mathematical issues the loader found, asks the lead engineer for
clarification when an anomaly is real, and parses the reply into concrete
column drops that are applied to the dataset before training starts.

Headless / CI runs pass ``interactive=False``: the findings are printed and the
pipeline proceeds, exactly as the Streamlit and API adapters require.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from backend_core.AgentSysID.agents.llm_base import invoke_llm, strip_code_fences
from backend_core.AgentSysID.agents.prompt_library import render, system_prompt
from backend_core.AgentSysID.data.loader import ExcelDataLoader


def run_data_inspector_agent(
    loader: ExcelDataLoader,
    interactive: bool = True,
    log_filename: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Run the Data Inspector HIL step.

    Returns
    -------
    (engineer_notes, cols_to_drop)
        ``engineer_notes`` is appended to the customer system description so
        every downstream agent sees the clarification; ``cols_to_drop`` is the
        list of columns the engineer agreed to remove.
    """
    print("\n" + "=" * 80)
    print("🧠 LLM DATA INSPECTOR AGENT: Analyzing Mathematical Health...")
    print("=" * 80)

    df = loader.df
    stats_summary = []
    for col in df.columns:
        stats_summary.append(
            f"- '{col}': Variance={df[col].var():.4f}, NaNs={df[col].isna().sum()}"
        )
    stats_text = "\n".join(stats_summary)

    issues = getattr(loader, "quality_issues", None)
    engineering_issues = (
        "\n".join(issues) if issues else "No mathematical anomalies detected by DataLoader."
    )

    prompt = render(
        "data_inspector",
        complexity_label=getattr(loader, "complexity_label", "Unknown"),
        stats_text=stats_text,
        engineering_issues=engineering_issues,
    )

    agent_reply = invoke_llm(
        system_prompt("data_inspector"),
        prompt,
        agent_name="Data Inspector Agent",
        log_filename=log_filename,
    ).strip()

    if not agent_reply:
        print("  ⚠️ Inspector unavailable. Defaulting to [PROCEED].")
        print("=" * 80 + "\n")
        return "", []

    engineer_notes = ""
    cols_to_drop: List[str] = []

    if "[ASK_HUMAN]" in agent_reply:
        question = agent_reply.replace("[ASK_HUMAN]", "").strip()
        print("\n" + "⚠️ " * 30)
        print("🛑 AGENT DETECTED ANOMALY / QUESTION:")
        print(f"🤖 Agent: {question}")
        print("⚠️ " * 30)

        if not interactive:
            print("  ℹ️ Headless mode: skipping the clarification prompt and proceeding.")
            print("=" * 80 + "\n")
            return "", []

        try:
            engineer_notes = input(
                "\n👨‍💻 Your Clarification (or press Enter/type 'skip' to ignore): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            engineer_notes = ""

        # --- Action parser: turn the reply into physical column drops --------
        if engineer_notes and engineer_notes.lower() not in ["skip", "no", "none"]:
            parse_prompt = render(
                "data_inspector",
                key="parser_template",
                question=question,
                engineer_notes=engineer_notes,
                columns=list(df.columns),
            )
            raw_list = invoke_llm(
                render("data_inspector", key="parser_system"),
                parse_prompt,
                agent_name="Data Inspector Parser",
                log_filename=log_filename,
            )
            try:
                parsed = json.loads(strip_code_fences(raw_list))
                if isinstance(parsed, list):
                    cols_to_drop = [str(c) for c in parsed]
            except Exception:
                cols_to_drop = []
    else:
        print("  ✅ LLM confirms dataset is healthy. Proceeding...")

    print("=" * 80 + "\n")
    return engineer_notes, cols_to_drop
