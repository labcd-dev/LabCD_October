"""Agent interaction logging."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str | Path = "logs") -> Path:
    """Ensure log directory exists and return a timestamped log file path."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_path / f"agents_sysid_{stamp}.log"


def log_agent_interaction(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    cycle: Optional[int] = None,
    log_filename: Optional[str | Path] = None,
    api_provider: str = "unknown",
) -> None:
    """Append a full agent turn to the log file."""
    if log_filename is None:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cycle_header = f" | [CYCLE {cycle}]" if cycle is not None else ""

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"📅 TIME: {timestamp}{cycle_header}\n")
        log_file.write(f"🤖 AGENT: {agent_name} ({api_provider.upper()})\n")
        log_file.write("-" * 80 + "\n")
        log_file.write(f"🧠 SYSTEM INSTRUCTIONS:\n{(system_prompt or '').strip()}\n")
        log_file.write("-" * 80 + "\n")
        log_file.write(f"📥 USER PROMPT:\n{(user_prompt or '').strip()}\n")
        log_file.write("-" * 80 + "\n")
        log_file.write(f"📤 RAW LLM RESPONSE:\n{(response_text or '').strip()}\n")
        log_file.write("=" * 80 + "\n")
