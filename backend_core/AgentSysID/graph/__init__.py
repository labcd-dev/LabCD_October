"""
LangGraph workflow for AgentSysID.

``run_cli.py`` is the production path (it owns the questionnaire and the
interactive overrides). ``build_sysid_graph`` exposes the same pipeline as a
compiled StateGraph for orchestrators that prefer LangGraph; it returns None
when langgraph is not installed.
"""

from .state import SysIDState
from .workflow import build_sysid_graph, should_continue

__all__ = ["SysIDState", "build_sysid_graph", "should_continue"]
