"""
Shared LLM access for AgentSysID agents.

Prefers labcd_agents when available; falls back to the local factory in config.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.utils.cost_tracker import cost_tracker
from backend_core.AgentSysID.utils.logging_utils import log_agent_interaction


def get_chat_model():
    """Return the configured LangChain chat model."""
    try:
        from labcd_agents import LLMFactory  # type: ignore

        return LLMFactory.create(
            provider=cfg.API_PROVIDER,
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
        )
    except Exception:
        return cfg.get_llm()


def _extract_usage(response: Any, llm_time: float) -> Dict[str, Any]:
    usage: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_time": llm_time,
    }
    try:
        meta = getattr(response, "response_metadata", None) or {}
        raw = meta.get("token_usage") or meta.get("usage") or {}
        if "prompt_tokens" in raw:
            usage["prompt_tokens"] = raw.get("prompt_tokens", 0)
            usage["completion_tokens"] = raw.get("completion_tokens", 0)
        elif "input_tokens" in raw:
            usage["prompt_tokens"] = raw.get("input_tokens", 0)
            usage["completion_tokens"] = raw.get("output_tokens", 0)
    except Exception:
        pass
    return usage


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    agent_name: str = "Agent",
    cycle: Optional[int] = None,
    log_filename: Optional[str] = None,
) -> str:
    """Invoke the LLM, track cost, and append to conversation history log."""
    llm = get_chat_model()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        t0 = time.perf_counter()
        response = llm.invoke(messages)
        llm_time = time.perf_counter() - t0
        text = response.content if hasattr(response, "content") else str(response)
        cost_tracker.update(response)
        usage = _extract_usage(response, llm_time)
        if log_filename:
            log_agent_interaction(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=text,
                cycle=cycle,
                log_filename=log_filename,
                api_provider=cfg.API_PROVIDER,
                model_name=cfg.LLM_MODEL,
                usage=usage,
            )
        return text
    except Exception as e:
        print(f"   ⚠️  LLM call failed for {agent_name}: {e}")
        if log_filename:
            log_agent_interaction(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=f"[ERROR] {e}",
                cycle=cycle,
                log_filename=log_filename,
                api_provider=cfg.API_PROVIDER,
                model_name=cfg.LLM_MODEL,
                usage={"llm_time": 0.0},
            )
        return ""
