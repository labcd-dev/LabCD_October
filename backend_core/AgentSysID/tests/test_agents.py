"""
Agent tests.

Every agent must behave correctly with no LLM reachable — the deterministic
fallbacks are what keep a long tuning run alive when the API rate-limits or
the key is missing — and must parse the strict KEY: VALUE contract when the
model does answer.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents import (
    ActorAgent,
    CriticAgent,
    ExplorerAgent,
    InitializerAgent,
    ReportAgent,
)
from backend_core.AgentSysID.agents import prompt_library
from backend_core.AgentSysID.training.tracker import BestConfigTracker


def _no_llm(monkeypatch, module: str) -> None:
    """Force that module's invoke_llm to behave as if the API is unreachable."""
    monkeypatch.setattr(f"backend_core.AgentSysID.agents.{module}.invoke_llm", lambda *a, **k: "")


def _fake_llm(monkeypatch, module: str, reply: str) -> None:
    monkeypatch.setattr(
        f"backend_core.AgentSysID.agents.{module}.invoke_llm", lambda *a, **k: reply
    )


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["initializer", "critic", "actor", "explorer", "report", "data_inspector"]
)
def test_every_prompt_file_loads_with_a_system_and_user_block(name: str):
    data = prompt_library.load_prompt(name)
    assert data.get("system"), f"{name}.yaml is missing a system block"
    assert data.get("user_template"), f"{name}.yaml is missing a user_template block"


def test_render_substitutes_and_tolerates_missing_keys():
    out = prompt_library.render("explorer", stuck_lr=0.001, stuck_hidden=[64, 32])
    assert "0.001" in out and "[64, 32]" in out


def test_critic_prompt_carries_the_overfit_matrix():
    text = prompt_library.load_prompt("critic")["user_template"]
    for threshold in ["4.0", "7.0", "10.0"]:
        assert threshold in text
    assert "LATENCY_VIOLATION" in text


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
def _critic_config() -> Dict[str, Any]:
    return {"learning_rate": 0.001, "hidden_layers": [128, 128], "dropout_rate": 0.1}


def test_critic_parses_the_key_value_contract(monkeypatch):
    _fake_llm(
        monkeypatch, "critic",
        "DIAGNOSIS: MILD_OVERFITTING\n"
        "STATUS: NEEDS_IMPROVEMENT\n"
        "LR_DIRECTION: decrease\n"
        "LR_STEP: 0.0002\n"
        "HIDDEN_LAYERS: [64, 32]\n"
        "REASONING: Validation error is drifting from training error.",
    )
    critic = CriticAgent(BestConfigTracker(), run_mode="regular")
    out = critic.evaluate(0.001, 0.005, _critic_config(), "relu", 0.5, 2.0, 1, val_rmse=0.07)
    assert out["diagnosis"] == "MILD_OVERFITTING"
    assert out["lr_dir"] == "decrease"
    assert out["lr_step"] == pytest.approx(0.0002)
    assert out["hidden_layers"] == [64, 32]
    assert "drifting" in out["reasoning"]


def test_critic_fallback_flags_latency_violation(monkeypatch):
    _no_llm(monkeypatch, "critic")
    critic = CriticAgent(BestConfigTracker(), run_mode="fast")
    out = critic.evaluate(0.001, 0.002, _critic_config(), "relu",
                          measured_latency=9.0, max_latency=2.0, cycle_number=1)
    assert out["diagnosis"] == "LATENCY_VIOLATION"
    assert out["status"] == "REJECTED"
    # It must shrink the network to buy latency back.
    assert sum(out["hidden_layers"]) < sum(_critic_config()["hidden_layers"])


@pytest.mark.parametrize(
    "train,val,expected",
    [
        (0.001, 0.0035, "NORMAL"),              # ratio 3.5
        (0.001, 0.005, "MILD_OVERFITTING"),     # ratio 5
        (0.001, 0.008, "HIGH_OVERFITTING"),     # ratio 8
        (0.001, 0.02, "CRITICAL_OVERFITTING"),  # ratio 20
    ],
)
def test_critic_fallback_applies_the_overfit_matrix(monkeypatch, train, val, expected):
    _no_llm(monkeypatch, "critic")
    critic = CriticAgent(BestConfigTracker(), run_mode="regular")
    out = critic.evaluate(train, val, _critic_config(), "relu", 0.5, 2.0, 1)
    assert out["diagnosis"] == expected


def test_critic_explore_limit_follows_run_mode():
    assert CriticAgent(BestConfigTracker(), run_mode="fast").explore_limit() == 3
    assert CriticAgent(BestConfigTracker(), run_mode="regular").explore_limit() == 8
    assert CriticAgent(BestConfigTracker(), run_mode="heavy").explore_limit() == 15


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
def _actor(**overrides) -> ActorAgent:
    base = {
        "learning_rate": 0.0005,
        "hidden_layers": [128, 128],
        "dropout_rate": 0.1,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "early_stop_patience": 25,
        "lr_search_min": 1e-5,
        "lr_search_max": 1e-2,
        "hidden_size_search_min": 16,
        "hidden_size_search_max": 256,
        "num_layers_search_min": 1,
        "num_layers_search_max": 4,
    }
    base.update(overrides)
    return ActorAgent("relu", initial_config=base)


def test_actor_parses_llm_config_and_respects_bounds(monkeypatch):
    _fake_llm(
        monkeypatch, "actor",
        "LEARNING_RATE: 0.5\n"          # far above lr_search_max -> clamped
        "HIDDEN_LAYERS: [9999, 9999]\n"  # above hidden max -> clamped
        "DROPOUT_RATE: 0.25\n"
        "WEIGHT_DECAY: 0.001\n"
        "BATCH_SIZE: 64\n"
        "PATIENCE: 20",
    )
    actor = _actor()
    new = actor.apply_critic_feedback({"diagnosis": "NORMAL"}, BestConfigTracker())
    assert new["learning_rate"] <= 1e-2
    assert all(h <= 256 for h in new["hidden_layers"])
    assert new["batch_size"] == 64
    assert new["patience"] == 20


def test_actor_never_repeats_a_visited_configuration(monkeypatch):
    """Even if the LLM keeps proposing the same thing, the Actor must move on."""
    _fake_llm(
        monkeypatch, "actor",
        "LEARNING_RATE: 0.0005\nHIDDEN_LAYERS: [128, 128]\nDROPOUT_RATE: 0.1\n"
        "WEIGHT_DECAY: 0.0001\nBATCH_SIZE: 128\nPATIENCE: 25",
    )
    actor = _actor()
    seen = {actor._config_to_key(actor.current_config)}
    for _ in range(5):
        new = actor.apply_critic_feedback({"diagnosis": "NORMAL"}, BestConfigTracker())
        key = actor._config_to_key(new)
        assert key not in seen, "Actor returned an already-visited configuration"
        seen.add(key)


def test_actor_fallback_tightens_regularization_on_overfitting(monkeypatch):
    _no_llm(monkeypatch, "actor")
    cfg.ADAPTIVE_REGULARIZATION = True
    actor = _actor()
    before = dict(actor.current_config)
    new = actor.apply_critic_feedback(
        {"diagnosis": "HIGH_OVERFITTING", "hidden_layers": [64, 64], "lr_dir": "stay", "lr_step": 0.0},
        BestConfigTracker(),
    )
    assert new["dropout_rate"] > before["dropout_rate"]
    assert new["batch_size"] <= before["batch_size"]
    assert new["patience"] <= before["patience"]


def test_actor_respects_locked_regularization(monkeypatch):
    _no_llm(monkeypatch, "actor")
    saved = cfg.ADAPTIVE_REGULARIZATION
    cfg.ADAPTIVE_REGULARIZATION = False
    try:
        actor = _actor()
        before = dict(actor.current_config)
        new = actor.apply_critic_feedback(
            {"diagnosis": "HIGH_OVERFITTING", "hidden_layers": [64, 64],
             "lr_dir": "stay", "lr_step": 0.0},
            BestConfigTracker(),
        )
        assert new["dropout_rate"] == before["dropout_rate"]
        assert new["weight_decay"] == before["weight_decay"]
    finally:
        cfg.ADAPTIVE_REGULARIZATION = saved


def test_actor_applies_critic_lr_direction(monkeypatch):
    _no_llm(monkeypatch, "actor")
    actor = _actor()
    before = actor.current_config["learning_rate"]
    new = actor.apply_critic_feedback(
        {"diagnosis": "NORMAL", "hidden_layers": [128, 128],
         "lr_dir": "decrease", "lr_step": 0.0002},
        BestConfigTracker(),
    )
    assert new["learning_rate"] < before


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------
def test_explorer_inverts_a_shallow_wide_topology(monkeypatch):
    _no_llm(monkeypatch, "explorer")
    explorer = ExplorerAgent(
        initial_config={
            "lr_search_min": 1e-5, "lr_search_max": 1e-2,
            "hidden_size_search_min": 16, "hidden_size_search_max": 256,
            "num_layers_search_min": 1, "num_layers_search_max": 4,
        }
    )
    stuck = {"learning_rate": 0.001, "hidden_layers": [256]}  # shallow + wide
    new, reasoning = explorer.generate_radical_escape(BestConfigTracker(), stuck, set())
    # Should become deep and narrow.
    assert len(new["hidden_layers"]) > len(stuck["hidden_layers"])
    assert float(np.mean(new["hidden_layers"])) < float(np.mean(stuck["hidden_layers"]))
    assert reasoning


def test_explorer_avoids_visited_configurations(monkeypatch):
    _no_llm(monkeypatch, "explorer")
    explorer = ExplorerAgent()
    stuck = {"learning_rate": 0.001, "hidden_layers": [256]}
    visited = set()
    for _ in range(5):
        new, _ = explorer.generate_radical_escape(BestConfigTracker(), stuck, visited)
        key = explorer._visited_key(new["learning_rate"], new["hidden_layers"])
        assert key not in visited
        visited.add(key)


def test_explorer_keeps_static_parameters_from_the_stuck_config(monkeypatch):
    _no_llm(monkeypatch, "explorer")
    explorer = ExplorerAgent()
    stuck = {"learning_rate": 0.001, "hidden_layers": [256], "batch_size": 64, "patience": 15}
    new, _ = explorer.generate_radical_escape(BestConfigTracker(), stuck, set())
    assert new["batch_size"] == 64
    assert new["patience"] == 15


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------
class _FakeLoader:
    def __init__(self):
        import pandas as pd

        t = np.arange(0, 5, 0.01)
        self.df = pd.DataFrame({"time": t, "s_x": np.sin(t), "a_u": np.cos(t)})
        self.state_cols, self.action_cols, self.xdot_cols = ["s_x"], ["a_u"], []
        self.state_dim, self.action_dim = 1, 1
        self.angle_indices = []
        self.complexity_tier, self.complexity_label = 3, "Level 3 (Moderately Complex)"
        self.has_xdot = False
        self.reset_threshold = [0.05]


def test_initializer_parses_and_clamps(monkeypatch):
    _fake_llm(
        monkeypatch, "initializer",
        "LEARNING_RATE: 99.0\n"                 # above the outer limit
        "HIDDEN_LAYERS: [512, 512, 512, 512]\n"  # too wide and too deep
        "ACTIVATION: elu\n"
        "EPOCHS: 400\nBATCH_SIZE: 64\nEARLY_STOP_PATIENCE: 30\n"
        "LR_SEARCH_MIN: 0.00001\nLR_SEARCH_MAX: 0.01\n"
        "HIDDEN_SIZE_SEARCH_MIN: 8\nHIDDEN_SIZE_SEARCH_MAX: 1024\n"
        "NUM_LAYERS_SEARCH_MIN: 1\nNUM_LAYERS_SEARCH_MAX: 9\n"
        "DROPOUT_RATE: 0.2\nWEIGHT_DECAY: 0.001\nLR_REDUCE_FACTOR: 0.5\n"
        "USE_STATE_FILTER: True\nAUTO_FILTER_PERCENTILES: [5, 95]\n"
        "DERIVATIVE_FILTER_TAU: 0.01\nRESET_THRESHOLD: [99999.0]\n"
        "REASONING: Moderate complexity warrants a standard network.",
    )
    result = InitializerAgent(_FakeLoader()).determine_initial_setup()

    assert result["activation"] == "elu"
    assert result["epochs"] == 400
    assert result["use_state_filter"] is True
    assert result["auto_filter_percentiles"] == [5, 95]
    # Clamped to the client-authorised outer limits
    assert cfg.LEARNING_RATE_MIN <= result["learning_rate"] <= cfg.LEARNING_RATE_MAX
    assert result["hidden_size_search_max"] <= cfg.HIDDEN_SIZE_MAX
    assert result["num_layers_search_max"] <= cfg.NUM_LAYERS_MAX
    assert len(result["hidden_layers"]) <= result["num_layers_search_max"]
    assert all(h <= result["hidden_size_search_max"] for h in result["hidden_layers"])


def test_initializer_prompt_uses_the_clients_training_priority(monkeypatch):
    previous_goal = cfg.OPTIMIZATION_GOAL
    captured = {}
    reply = "LEARNING_RATE: 0.001\nHIDDEN_LAYERS: [32]\nACTIVATION: tanh\nREASONING: Compact starting model."

    def fake_invoke(system, prompt, **kwargs):
        captured["prompt"] = prompt
        return reply

    monkeypatch.setattr("backend_core.AgentSysID.agents.initializer.invoke_llm", fake_invoke)
    cfg.OPTIMIZATION_GOAL = "speed"
    try:
        InitializerAgent(_FakeLoader()).determine_initial_setup()
        assert "Client priority: speed" in captured["prompt"]
        assert "Favor a compact, fast-inference starting model" in captured["prompt"]
    finally:
        cfg.OPTIMIZATION_GOAL = previous_goal


def test_initializer_fallback_keeps_the_calibrated_reset_threshold(monkeypatch):
    """A failed agent call must not shatter trajectories with a bogus threshold."""
    _no_llm(monkeypatch, "initializer")
    loader = _FakeLoader()
    result = InitializerAgent(loader).determine_initial_setup()
    assert result["reset_threshold"] == [0.05]


def test_initializer_manual_mode_bypasses_the_llm():
    saved = cfg.CHOOSE_VIA_LLM_INITIALIZER
    cfg.CHOOSE_VIA_LLM_INITIALIZER = False
    try:
        result = InitializerAgent(_FakeLoader()).determine_initial_setup()
        assert result["learning_rate"] == cfg.MANUAL_STARTING_LR
        assert result["hidden_layers"] == list(cfg.MANUAL_STARTING_HIDDEN_LAYERS)
    finally:
        cfg.CHOOSE_VIA_LLM_INITIALIZER = saved


# ---------------------------------------------------------------------------
# Report agent
# ---------------------------------------------------------------------------
def test_report_agent_parses_abstract_and_conclusion(monkeypatch):
    _fake_llm(
        monkeypatch, "report_agent",
        "ABSTRACT: A concise technical abstract.\nCONCLUSION: A concise conclusion.",
    )
    abstract, conclusion = ReportAgent().generate_report_text(
        "plant", 3, 2, {"hidden_layers": [64]}, 0.001, 0.03, 0.4, False,
        max_latency=2.0, success_score=88.0, model_status="STABLE & HIGH-FIDELITY",
    )
    assert abstract == "A concise technical abstract."
    assert conclusion == "A concise conclusion."


def test_report_agent_fallback_is_fully_populated(monkeypatch):
    _no_llm(monkeypatch, "report_agent")
    abstract, conclusion = ReportAgent().generate_report_text(
        "plant", 3, 2, {"hidden_layers": [64, 64]}, 0.001234, 0.035, 0.42, True,
        max_latency=2.0, success_score=88.5, model_status="STABLE & HIGH-FIDELITY",
        complexity_label="Level 4 (Highly Complex / Fast Transient)", architecture="LSTM",
    )
    # The fallback must still carry the real numbers, not placeholders.
    assert "0.001234" in abstract
    assert "[64, 64]" in abstract
    assert "Physics-Informed" in abstract
    assert "88.5/100" in conclusion
    assert "STABLE & HIGH-FIDELITY" in conclusion
