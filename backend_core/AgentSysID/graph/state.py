"""Shared state for a future LangGraph SysID workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SysIDState(TypedDict, total=False):
    # Inputs
    data_path: str
    run_mode: str
    interactive: bool

    # Data
    state_dim: int
    action_dim: int
    complexity_label: str
    trajectories: List[Dict[str, Any]]

    # Config / search
    current_config: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    cycle: int
    max_cycles: int

    # Results
    best_config: Dict[str, Any]
    best_mse: float
    best_rmse: float
    latency_ms: float
    abstract: str
    pdf_path: str
    zip_path: str

    # Control
    status: str  # running | completed | cancelled | failed
    messages: List[str]
