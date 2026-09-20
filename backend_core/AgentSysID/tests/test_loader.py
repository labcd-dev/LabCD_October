"""Unit tests for ExcelDataLoader (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.data.loader import ExcelDataLoader


@pytest.fixture(autouse=True)
def _restore_config():
    """Loader behaviour is driven by module-level config; restore it per test."""
    saved = {
        k: getattr(cfg, k)
        for k in (
            "MULTI_TRAJECTORY",
            "DERIVATIVE_METHOD",
            "DERIVATIVE_FILTER_TAU",
            "ANGLE_INDICES",
            "AUTO_DETECT_ANGLES",
            "USE_STATE_FILTER",
            "RESET_THRESHOLD",
            "MANUAL_TRAJECTORY_SPLIT_TIMES",
        )
    }
    cfg.MULTI_TRAJECTORY = False
    cfg.DERIVATIVE_METHOD = "finite_difference"
    cfg.DERIVATIVE_FILTER_TAU = 0.0
    cfg.ANGLE_INDICES = []
    cfg.AUTO_DETECT_ANGLES = False
    cfg.USE_STATE_FILTER = False
    cfg.RESET_THRESHOLD = 99999.0
    cfg.MANUAL_TRAJECTORY_SPLIT_TIMES = []
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)


def _write(tmp_path: Path, df: pd.DataFrame, name: str = "data.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    t = np.linspace(0, 1, 50)
    df = pd.DataFrame(
        {
            "time": t,
            "s_x": np.sin(t),
            "s_v": np.cos(t),
            "a_u": np.sin(2 * t) * 0.1,
        }
    )
    return _write(tmp_path, df, "synthetic.csv")


def test_load_and_trajectories(synthetic_csv: Path):
    loader = ExcelDataLoader(synthetic_csv)
    assert loader.state_dim == 2
    assert loader.action_dim == 1
    trajs = loader.get_trajectories()
    assert len(trajs) >= 1
    t0 = trajs[0]
    assert "states" in t0 and "actions" in t0 and "xdots" in t0 and "dts" in t0
    assert t0["states"].shape[1] == 2
    assert t0["actions"].shape[1] == 1
    # dt must be strictly positive everywhere
    assert np.all(t0["dts"] > 0)


def test_missing_time_raises(tmp_path: Path):
    df = pd.DataFrame({"s_x": [1, 2, 3], "a_u": [0, 0, 0]})
    path = _write(tmp_path, df, "bad.csv")
    with pytest.raises(ValueError, match="time"):
        ExcelDataLoader(path)


def test_unnamed_ghost_columns_are_dropped(tmp_path: Path):
    t = np.linspace(0, 1, 30)
    df = pd.DataFrame({"time": t, "s_x": t, "a_u": t, "Unnamed: 4": np.nan})
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert not any(c.startswith("Unnamed") for c in loader.df.columns)


def test_true_xdot_columns_are_used_verbatim(tmp_path: Path):
    t = np.linspace(0, 2, 60)
    df = pd.DataFrame(
        {"time": t, "s_x": np.sin(t), "a_u": np.zeros_like(t), "xdot_x": np.cos(t)}
    )
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert loader.has_xdot is True
    traj = loader.get_trajectories()[0]
    np.testing.assert_allclose(traj["xdots"][:, 0], np.cos(t), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "method,tolerance",
    [("finite_difference", 1e-3), ("savitzky_golay", 5e-3), ("sliding_mode", 0.2)],
)
def test_derivative_estimators_recover_a_known_derivative(
    tmp_path: Path, method: str, tolerance: float
):
    """Each estimator must approximate d(sin t)/dt = cos t on clean data."""
    t = np.arange(0, 20, 0.01)
    df = pd.DataFrame({"time": t, "s_x": np.sin(t), "a_u": np.zeros_like(t)})
    cfg.DERIVATIVE_METHOD = method
    loader = ExcelDataLoader(_write(tmp_path, df))
    est = loader.get_trajectories()[0]["xdots"][:, 0]
    # Ignore the edges, where every differentiator degrades.
    err = float(np.mean(np.abs(est[100:-100] - np.cos(t)[100:-100])))
    assert err < tolerance, f"{method} error {err} exceeded {tolerance}"


def test_derivative_filter_tau_smooths_noise(tmp_path: Path):
    """The Simulink-style low-pass must reduce derivative noise energy."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.01)
    noisy = np.sin(t) + 0.01 * rng.standard_normal(len(t))
    df = pd.DataFrame({"time": t, "s_x": noisy, "a_u": np.zeros_like(t)})
    path = _write(tmp_path, df)

    cfg.DERIVATIVE_FILTER_TAU = 0.0
    raw = ExcelDataLoader(path).get_trajectories()[0]["xdots"][:, 0]
    cfg.DERIVATIVE_FILTER_TAU = 0.05
    filtered = ExcelDataLoader(path).get_trajectories()[0]["xdots"][:, 0]

    # Roughness = mean absolute second difference
    rough = lambda a: float(np.mean(np.abs(np.diff(a, 2))))  # noqa: E731
    assert rough(filtered) < rough(raw)


def test_multi_trajectory_autodetect_splits_on_state_jump(tmp_path: Path):
    """A teleportation between stacked runs must start a new trajectory."""
    t = np.arange(0, 6, 0.01)
    x = np.concatenate([np.sin(t[: len(t) // 2]), 500.0 + np.sin(t[len(t) // 2 :])])
    df = pd.DataFrame({"time": t, "s_x": x, "a_u": np.zeros_like(t)})
    cfg.MULTI_TRAJECTORY = True
    loader = ExcelDataLoader(_write(tmp_path, df))
    trajs = loader.get_trajectories()
    assert len(trajs) == 2
    assert len(trajs[0]["states"]) + len(trajs[1]["states"]) == len(t)


def test_manual_split_timestamps(tmp_path: Path):
    t = np.round(np.arange(0, 3, 0.01), 4)
    df = pd.DataFrame({"time": t, "s_x": np.sin(t), "a_u": np.zeros_like(t)})
    cfg.MULTI_TRAJECTORY = True
    cfg.MANUAL_TRAJECTORY_SPLIT_TIMES = [1.0, 2.0]
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert len(loader.get_trajectories()) == 3


def test_angle_auto_detection(tmp_path: Path):
    """A yaw signal wrapping at +/- pi must be flagged as angular."""
    t = np.arange(0, 20, 0.01)
    yaw = np.mod(t + np.pi, 2 * np.pi) - np.pi  # sawtooth wrapping at +/- pi
    df = pd.DataFrame({"time": t, "s_yaw": yaw, "a_u": np.zeros_like(t)})
    cfg.AUTO_DETECT_ANGLES = True
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert loader.angle_indices == [0]


def test_complexity_tier_is_between_1_and_5(synthetic_csv: Path):
    loader = ExcelDataLoader(synthetic_csv)
    assert 1 <= loader.complexity_tier <= 5
    assert loader.complexity_label.startswith("Level")
    assert loader.complexity_score == float(loader.complexity_tier)


def test_quality_agent_flags_dead_action_and_duplicate_state(tmp_path: Path):
    t = np.arange(0, 5, 0.01)
    x = np.sin(t)
    df = pd.DataFrame(
        {
            "time": t,
            "s_x": x,
            "s_x_copy": x,               # perfectly collinear duplicate
            "a_dead": np.zeros_like(t),  # no excitation
        }
    )
    loader = ExcelDataLoader(_write(tmp_path, df))
    issues = " ".join(loader.quality_issues)
    assert "near-zero variance" in issues
    assert "correlated" in issues


def test_drop_columns_removes_state_and_its_derivative(tmp_path: Path):
    t = np.arange(0, 2, 0.01)
    df = pd.DataFrame(
        {
            "time": t,
            "s_x": np.sin(t),
            "s_y": np.cos(t),
            "xdot_x": np.cos(t),
            "xdot_y": -np.sin(t),
            "a_u": np.zeros_like(t),
        }
    )
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert loader.state_dim == 2
    loader.drop_columns(["s_y"])
    assert loader.state_dim == 1
    assert "xdot_y" not in loader.df.columns
    assert loader.xdot_cols == ["xdot_x"]


def test_state_filter_clips_outliers(tmp_path: Path):
    t = np.arange(0, 5, 0.01)
    x = np.sin(t)
    x[100] = 1000.0  # single extreme spike
    df = pd.DataFrame({"time": t, "s_x": x, "a_u": np.zeros_like(t)})
    cfg.USE_STATE_FILTER = True
    cfg.AUTO_FILTER_PERCENTILES = (2, 98)
    loader = ExcelDataLoader(_write(tmp_path, df))
    assert loader.df["s_x"].max() < 10.0
