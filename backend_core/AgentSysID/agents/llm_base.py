"""
Shared LLM access for AgentSysID agents.

Prefers ``labcd_agents.LLMFactory`` when the monorepo package is installed;
falls back to the local factory in ``config.get_llm``.

``invoke_llm`` centralises what every legacy agent used to repeat inline:
cost tracking, conversation logging, an optional rate-limit cooldown, and the
Critic's retry-with-backoff behaviour.
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


def strip_code_fences(text: str) -> str:
    """Remove markdown fences some models wrap around key-value output."""
    return (
        (text or "")
        .replace("```json", "")
        .replace("```text", "")
        .replace("```yaml", "")
        .replace("```", "")
        .strip()
    )


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    agent_name: str = "Agent",
    cycle: Optional[int] = None,
    log_filename: Optional[str] = None,
    max_retries: int = 1,
    retry_wait: float = 15.0,
    cooldown: float = 0.0,
) -> str:
    """
    Invoke the LLM, track cost, and append the turn to the conversation log.

    Parameters
    ----------
    max_retries : total attempts before giving up (the Critic uses 3).
    retry_wait  : seconds to wait between attempts, to clear rate limits.
    cooldown    : seconds to pause *before* the first call.

    Returns the response text, or "" when every attempt failed.
    """
    try:
        llm = get_chat_model()
    except Exception as exc:  # missing API key, bad provider, offline install
        print(f"      ⚠️ {agent_name}: no LLM backend available ({exc}).")
        print("      ↪ Falling back to the deterministic mathematical path.")
        return ""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    if cooldown > 0:
        print("    ⏳ Cooldown pause ...", end="", flush=True)
        time.sleep(cooldown)
        print(f" Done. 🧠 Awaiting {agent_name} LLM response...", flush=True)

    last_error: Optional[Exception] = None
    for attempt in range(max(1, int(max_retries))):
        try:
            t0 = time.perf_counter()
            response = llm.invoke(messages)
            llm_time = time.perf_counter() - t0
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
                    model_name=cfg.LLM_MODEL,
                    usage=_extract_usage(response, llm_time),
                )
            return text
        except Exception as exc:  # noqa: BLE001 - agents must survive API errors
            last_error = exc
            remaining = max(1, int(max_retries)) - attempt - 1
            print(
                f"      ⚠️ {agent_name} API Error on attempt "
                f"{attempt + 1}/{max(1, int(max_retries))}: {exc}"
            )
            if remaining > 0:
                print(f"      ⏳ Waiting {retry_wait:.0f} seconds to clear API rate limits...")
                time.sleep(retry_wait)

    print(f"      ❌ {agent_name} API completely failed. Falling back to mathematical parameters.")
    if log_filename:
        log_agent_interaction(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=f"[ERROR] {last_error}",
            cycle=cycle,
            log_filename=log_filename,
            api_provider=cfg.API_PROVIDER,
            model_name=cfg.LLM_MODEL,
            usage={"llm_time": 0.0},
        )
    return ""
