"""
Agent interaction / conversation-history logging for a single SysID run.

Log file lives inside the run artifact folder (see run_cli.setup_run_dir).
Format inspired by LabCD agent harness logging_utils.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_run_dir(
    base_dir: str | Path = "artifacts_sysid",
    env_name: str = "system",
) -> Path:
    """
    Create artifacts_sysid/run_YYYYMMDD_HHMMSS_<env>/ and return that path.
    All outputs for one CLI run should land here.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_env = "".join(c if c.isalnum() or c in "-_" else "_" for c in (env_name or "system"))
    run_dir = Path(base_dir) / f"run_{stamp}_{safe_env}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: str | Path) -> Path:
    """
    Create conversation history log inside the run directory.
    Returns path to llm_conversation_history.txt
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "llm_conversation_history.txt"
    # Header so the file is non-empty from the start
    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"AgentSysID LLM conversation history\n"
                f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"{'=' * 80}\n"
            )
    return log_path


def log_to_file(log_path: str | Path, message: str, error: bool = False) -> None:
    prefix = "ERROR: " if error else ""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(prefix + message + "\n")


def formatted_log(
    log_path: str | Path,
    agent_name: str,
    prompt: str,
    response: str,
    model_name: str,
    usage: Optional[Dict[str, Any]] = None,
    cycle: Optional[int] = None,
) -> None:
    """
    Append a formatted turn: timestamp, agent, model, prompt, response, usage.
    """
    usage = usage or {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0
    llm_time = usage.get("llm_time", 0.0) or 0.0
    cycle_line = f"CYCLE: [{cycle}]\n" if cycle is not None else ""

    log_message = f"""
================================================================================
TIMESTAMP: [{timestamp}]
AGENT: [{agent_name}]
{cycle_line}LLM Model: [{model_name}]
--------------------------------------------------------------------------------
PROMPT:
{(prompt or '').strip()}
--------------------------------------------------------------------------------
RESPONSE:
{(response or '').strip()}
--------------------------------------------------------------------------------
USAGE STATISTICS:
Tokens In: {tokens_in:,}
Tokens Out: {tokens_out:,}
Total Tokens: {tokens_in + tokens_out:,}
LLM Time: {llm_time:.3f}s
================================================================================
"""
    log_to_file(log_path, log_message)


def log_agent_interaction(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    cycle: Optional[int] = None,
    log_filename: Optional[str | Path] = None,
    api_provider: str = "unknown",
    model_name: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a full agent turn. Combines system + user into one PROMPT block
    for harness-style conversation history.
    """
    if log_filename is None:
        return

    combined = (
        f"[SYSTEM]\n{(system_prompt or '').strip()}\n\n"
        f"[USER]\n{(user_prompt or '').strip()}"
    )
    formatted_log(
        log_path=log_filename,
        agent_name=agent_name,
        prompt=combined,
        response=response_text or "",
        model_name=model_name or api_provider,
        usage=usage,
        cycle=cycle,
    )
