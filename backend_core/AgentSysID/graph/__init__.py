"""
LangGraph workflow scaffold for AgentSysID.

The current production path uses a sequential orchestrator in run_cli.py.
A full StateGraph can be wired here later without changing agent contracts.
"""

from .state import SysIDState
from .workflow import build_sysid_graph

__all__ = ["SysIDState", "build_sysid_graph"]
