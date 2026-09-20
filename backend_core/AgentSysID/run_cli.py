#!/usr/bin/env python3
"""
AgentSysID CLI entry point.

Usage (from repository root):
    PYTHONPATH=. python -m backend_core.AgentSysID.run_cli
    PYTHONPATH=. python -m backend_core.AgentSysID.run_cli --data path/to/data.csv --mode fast
    PYTHONPATH=. python -m backend_core.AgentSysID.run_cli --data data.csv --headless

The engineer questionnaire, the HIL Data Inspector and the Initializer config
review are ON BY DEFAULT, exactly as the legacy main() ran them. Pass
--headless (or --no-interactive) for API / Streamlit / CI callers, where
nothing may block on a prompt.

This module only parses argv and delegates: the pipeline itself lives in
``pipeline.run_pipeline``, so the CLI, the Streamlit UI and the FastAPI
adapter all drive exactly the same code.

All outputs for one run are written under:
    artifacts_sysid/run_YYYYMMDD_HHMMSS_<env>/
      figures/          # MSE, RMSE, latency, hyperparams, contour, verification
      deployment/       # .pth, deployed_controller_*.py, NN.py
      report/           # PDF
      Agents_log/       # full LLM conversation history
      SystemID_RunResults_<timestamp>.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo root is on path when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.pipeline import (
    OVERFIT_PENALTY_MSE,
    OVERFIT_PENALTY_RMSE,
    STAGNATION_LIMIT,
    SysIDOptions,
    run_pipeline,
)

__all__ = [
    "main",
    "build_options",
    "OVERFIT_PENALTY_MSE",
    "OVERFIT_PENALTY_RMSE",
    "STAGNATION_LIMIT",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentSysID – Agentic System Identification")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV/Excel dataset")
    parser.add_argument(
        "--mode", type=str, default=cfg.RUN_MODE, choices=["fast", "regular", "heavy"]
    )
    # The legacy main() always asked the engineer these questions, so the
    # interactive path is the DEFAULT. Adapters (API / Streamlit / CI) opt out
    # with --headless, which is the only safe mode when nobody is at a terminal.
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dataset questionnaire, HIL Data Inspector and Initializer config review "
        "(default: on; use --no-interactive or --headless to disable)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Alias for --no-interactive: never block on a prompt",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["MLP", "LSTM", "mlp", "lstm"],
        help="Override NETWORK_ARCHITECTURE for this run",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None, help="Override the run mode's cycle limit"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the per-cycle epoch budget"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts_sysid",
        help="Base directory; each run creates a subfolder run_<timestamp>_<env>/",
    )
    return parser


def build_options(args: argparse.Namespace) -> SysIDOptions:
    """Translate parsed argv into the pipeline's options object."""
    return SysIDOptions(
        data_path=args.data or cfg.EXCEL_FILE_PATH,
        run_mode=args.mode,
        output_dir=args.output_dir,
        interactive=args.interactive,
        max_cycles=args.max_cycles,
        epochs=args.epochs,
        architecture=args.arch,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.headless:
        args.interactive = False

    # A questionnaire needs a real terminal. If stdin is not a TTY (piped input,
    # cron, a worker thread) fall back to headless rather than hanging on EOF.
    if args.interactive and not sys.stdin.isatty():
        print("ℹ️  No interactive terminal detected (stdin is not a TTY) — running headless.")
        args.interactive = False

    options = build_options(args)

    if not Path(options.data_path).is_file():
        print(f"❌ Dataset not found: {options.data_path}")
        print("   Provide --data path/to/file.csv or place system_data.csv in the working directory.")
        return 1

    if args.interactive:
        print(
            "ℹ️  Press Ctrl+C at any time to stop training early and get results "
            "from the best checkpoint so far."
        )

    result = run_pipeline(options)

    if result.status == "completed":
        return 0
    if result.message:
        print(f"\n{result.message}")
    return 0 if result.status == "failed" and "PINN" in result.message else 1


if __name__ == "__main__":
    raise SystemExit(main())
