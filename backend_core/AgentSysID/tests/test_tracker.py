"""Tests for BestConfigTracker."""

from backend_core.AgentSysID.training.tracker import BestConfigTracker


def test_tracker_updates_best():
    t = BestConfigTracker()
    assert t.update(0.5, {"lr": 1e-3}) is True
    assert t.best_mse == 0.5
    assert t.update(0.8, {"lr": 1e-4}) is False
    assert t.best_mse == 0.5
    assert t.update(0.1, {"lr": 5e-4}) is True
    assert t.best_config["lr"] == 5e-4
