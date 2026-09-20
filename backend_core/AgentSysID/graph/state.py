"""Shared state for the LangGraph SysID workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SysIDState(TypedDict, total=False):
    # --- Inputs ---
    data_path: str
    run_mode: str
    interactive: bool
    output_dir: str
    architecture: str

    # --- Data ---
    state_dim: int
    action_dim: int
    complexity_tier: int
    complexity_label: str
    quality_issues: List[str]
    engineer_notes: str
    dropped_columns: List[str]
    trajectories: List[Dict[str, Any]]
    train_trajs: List[Dict[str, Any]]
    val_trajs: List[Dict[str, Any]]
    test_trajs: List[Dict[str, Any]]

    # --- Config / search ---
    setup_config: Dict[str, Any]
    current_config: Dict[str, Any]
    activation: str
    performance_history: List[Dict[str, Any]]
    cycle: int
    max_cycles: int
    stagnation: int
    critic_output: Dict[str, Any]

    # --- Results ---
    best_config: Dict[str, Any]
    best_mse: float
    best_rmse: float
    latency_ms: float
    success_score: float
    model_status: str
    abstract: str
    conclusion: str
    pdf_path: str
    zip_path: str

    # --- Control ---
    status: str  # running | completed | cancelled | failed
    messages: List[str]
