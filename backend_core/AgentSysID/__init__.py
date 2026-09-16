"""
AgentSysID – Agentic System Identification
==========================================
Multi-agent deep-learning framework for state-constrained system identification.
Tunes MLP / LSTM dynamics models (optionally physics-informed) via
Initializer → Data Inspector (HIL) → Critic / Actor / Explorer loop → Report.

Usage (from repository root with PYTHONPATH=.):

    from backend_core.AgentSysID.run_cli import main
    # or
    from backend_core.AgentSysID.model.dynamics_model import DynamicsModel
"""

__version__ = "0.1.0"
