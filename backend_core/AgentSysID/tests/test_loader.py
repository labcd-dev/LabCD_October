"""Unit tests for ExcelDataLoader (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend_core.AgentSysID.data.loader import ExcelDataLoader


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    t = np.linspace(0, 1, 50)
    df = pd.DataFrame({
        "time": t,
        "s_x": np.sin(t),
        "s_v": np.cos(t),
        "a_u": np.sin(2 * t) * 0.1,
    })
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


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


def test_missing_time_raises(tmp_path: Path):
    df = pd.DataFrame({"s_x": [1, 2, 3], "a_u": [0, 0, 0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="time"):
        ExcelDataLoader(path)
