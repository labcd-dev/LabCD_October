#!/usr/bin/env python3
"""
Launcher for the AgentSysID Streamlit UI.

    python frontend_streamlit/run_agent_sysid_ui.py
    python frontend_streamlit/run_agent_sysid_ui.py --port 8600 --headless

It sets PYTHONPATH to the repository root so ``backend_core.AgentSysID``
imports cleanly regardless of the working directory, then hands off to
``streamlit run``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent / "agent_sysid_app.py"
DEFAULT_PORT = 8504


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the AgentSysID Streamlit UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--address", type=str, default="localhost")
    parser.add_argument(
        "--headless", action="store_true", help="Do not open a browser window"
    )
    args = parser.parse_args()

    if not APP.is_file():
        print(f"❌ UI not found: {APP}")
        return 1

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(args.port),
        "--server.address", args.address,
        "--server.headless", "true" if args.headless else "false",
        "--theme.base", "dark",
    ]

    print(f"🚀 AgentSysID UI → http://{args.address}:{args.port}")
    print(f"   repo root: {REPO_ROOT}")
    try:
        return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))
    except FileNotFoundError:
        print("❌ Streamlit is not installed. Run:  pip install streamlit altair")
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
