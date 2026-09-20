"""Tests for BestConfigTracker and the composite success score."""

from __future__ import annotations

import numpy as np
import pytest

from backend_core.AgentSysID.training.scoring import (
    calculate_success_score,
    parse_complexity_stage,
)
from backend_core.AgentSysID.training.tracker import BestConfigTracker


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
def test_tracker_updates_best():
    t = BestConfigTracker()
    assert t.update(0.5, {"lr": 1e-3}) is True
    assert t.best_mse == 0.5
    assert t.update(0.8, {"lr": 1e-4}) is False
    assert t.best_mse == 0.5
    assert t.update(0.1, {"lr": 5e-4}) is True
    assert t.best_config["lr"] == 5e-4


def test_tracker_records_failures_and_renders_them_for_prompts():
    t = BestConfigTracker(run_mode="regular")
    t.update(0.5, {"hidden_layers": [64]})
    t.update(0.9, {"hidden_layers": [32]})
    t.add_reasoning_to_memory("Too small to capture the dynamics.")
    rendered = t.get_recent_failures_str()
    assert "[32]" in rendered
    assert "Too small to capture" in rendered
    assert "0.900000" in rendered


def test_reasoning_attaches_to_the_best_cycle_when_it_was_a_best():
    t = BestConfigTracker()
    t.update(0.5, {"hidden_layers": [64]})
    t.add_reasoning_to_memory("Great generalization.")
    assert t.best_reasoning == "Great generalization."


@pytest.mark.parametrize(
    "mode,expected_len",
    [("fast", 5), ("regular", 10), ("heavy", 15)],  # heavy keeps everything
)
def test_failure_memory_depth_follows_run_mode(mode: str, expected_len: int):
    t = BestConfigTracker(run_mode=mode)
    t.update(0.1, {"hidden_layers": [64]})  # establishes the best
    for i in range(15):
        t.update(1.0 + i, {"hidden_layers": [i]})
    assert len(t.recent_failures) == expected_len


def test_no_failures_renders_as_none():
    assert BestConfigTracker().get_recent_failures_str() == "None"


def test_history_records_every_cycle():
    t = BestConfigTracker()
    t.update(0.5, {"a": 1}, current_rmse=0.7)
    t.update(0.9, {"a": 2}, current_rmse=0.95)
    assert len(t.history) == 2
    assert t.history[0]["rmse"] == 0.7


# ---------------------------------------------------------------------------
# Success score
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,expected",
    [
        ("Level 1 (Trivial / Quasi-Steady)", 1),
        ("Level 2 (Mildly Nonlinear)", 2),
        ("Level 3 (Moderately Complex)", 3),
        ("Level 4 (Highly Complex / Fast Transient)", 4),
        ("Level 5 (Severe / Chaotic / Discontinuous)", 5),
        (4, 4),
        ("nonsense", 3),   # defaults to moderate
        (99, 5),           # clamped
    ],
)
def test_parse_complexity_stage(label, expected):
    assert parse_complexity_stage(label) == expected


def test_perfect_model_scores_near_100():
    traj = np.linspace(0, 1, 50).reshape(-1, 1)
    score, status = calculate_success_score(0.0, traj, traj, "Level 1", verbose=False)
    assert score == pytest.approx(100.0, abs=1e-6)
    assert status == "STABLE & HIGH-FIDELITY"


def test_drifting_rollout_is_penalised():
    true_traj = np.zeros((50, 1))
    good = np.zeros((50, 1))
    drifting = np.ones((50, 1)) * 5.0
    good_score, _ = calculate_success_score(0.01, true_traj, good, "Level 3", verbose=False)
    bad_score, bad_status = calculate_success_score(0.01, true_traj, drifting, "Level 3", verbose=False)
    assert bad_score < good_score
    assert "UNSTABLE" in bad_status or bad_score < 80


def test_higher_complexity_earns_leniency():
    """The same raw error must score higher on a harder plant."""
    true_traj = np.zeros((50, 1))
    nn_traj = np.ones((50, 1)) * 0.8
    easy, _ = calculate_success_score(2.0, true_traj, nn_traj, "Level 1", verbose=False)
    hard, _ = calculate_success_score(2.0, true_traj, nn_traj, "Level 5", verbose=False)
    assert hard > easy


def test_score_handles_empty_trajectories():
    score, status = calculate_success_score(
        0.001, np.zeros((0, 2)), np.zeros((0, 2)), "Level 3", verbose=False
    )
    assert 0.0 <= score <= 100.0
    assert status


def test_score_handles_mismatched_trajectory_lengths():
    score, _ = calculate_success_score(
        0.001, np.zeros((60, 1)), np.zeros((50, 1)), "Level 2", verbose=False
    )
    assert 0.0 <= score <= 100.0
