"""
LangGraph workflow builder (scaffold).

When langgraph is available this can be expanded into a real StateGraph.
Until then `build_sysid_graph` returns None and callers fall back to the
sequential runner in run_cli.py.
"""

from __future__ import annotations

from typing import Any, Optional


def build_sysid_graph() -> Optional[Any]:
    """
    Attempt to construct a LangGraph StateGraph for the SysID pipeline.

    Returns
    -------
    Compiled graph or None if langgraph is not installed / not yet wired.
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
        from backend_core.AgentSysID.graph.state import SysIDState

        # Placeholder – nodes can be added as agents are fully graph-ified
        graph = StateGraph(SysIDState)
        # Example future wiring:
        # graph.add_node("inspect", ...)
        # graph.add_node("initialize", ...)
        # graph.add_node("train_cycle", ...)
        # graph.add_edge(...)
        # graph.set_entry_point("inspect")
        # return graph.compile()
        return None
    except Exception:
        return None
