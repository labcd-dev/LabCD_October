"""
Shared LLM access for AgentSysID agents.

Prefers labcd_agents when available; falls back to the local factory in config.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.utils.cost_tracker import cost_tracker
from backend_core.AgentSysID.utils.logging_utils import log_agent_interaction


def get_chat_model():
    """Return the configured LangChain chat model."""
    try:
        # Prefer shared monorepo factory when present
        from labcd_agents import LLMFactory  # type: ignore

        return LLMFactory.create(
            provider=cfg.API_PROVIDER,
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
        )
    except Exception:
        return cfg.get_llm()


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    agent_name: str = "Agent",
    cycle: Optional[int] = None,
    log_filename: Optional[str] = None,
) -> str:
    """Invoke the LLM, track cost, and optionally log the interaction."""
    llm = get_chat_model()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        cost_tracker.update(response)
        if log_filename:
            log_agent_interaction(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=text,
                cycle=cycle,
                log_filename=log_filename,
                api_provider=cfg.API_PROVIDER,
            )
        return text
    except Exception as e:
        print(f"   ⚠️  LLM call failed for {agent_name}: {e}")
        return ""
