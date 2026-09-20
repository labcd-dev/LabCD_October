"""Tests for splitting, plots, deployable export, PDF and ZIP packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.data.splitter import split_trajectories
from backend_core.AgentSysID.model.dynamics_model import DynamicsModel
from backend_core.AgentSysID.reporting import (
    export_standalone_inference_script,
    generate_all_plots,
    generate_final_pdf,
    package_final_results_to_zip,
    plot_test_dataset_verification,
    save_best_model,
    write_nn_inference_helper,
)


def _traj(n: int = 100, state_dim: int = 2, action_dim: int = 1) -> Dict[str, np.ndarray]:
    t = np.arange(n) * 0.01
    return {
        "states": np.column_stack([np.sin(t + i) for i in range(state_dim)]).astype(np.float32),
        "actions": np.column_stack([np.cos(t + i) for i in range(action_dim)]).astype(np.float32),
        "xdots": np.column_stack([np.cos(t + i) for i in range(state_dim)]).astype(np.float32),
        "dts": np.full(n, 0.01, dtype=np.float32),
        "times": t,
    }


def _history(n: int = 4) -> List[dict]:
    return [
        {
            "cycle": i + 1,
            "iteration": i,
            "config": {"learning_rate": 0.001 * (i + 1), "hidden_layers": [64] * (1 + i % 3)},
            "train_mse": 0.01 / (i + 1),
            "val_mse": 0.02 / (i + 1),
            "rmse": float(np.sqrt(0.02 / (i + 1))),
            "latency": 0.5 + 0.1 * i,
            "performance": {"mse": 0.02 / (i + 1), "rmse": float(np.sqrt(0.02 / (i + 1))), "latency": 0.5 + 0.1 * i},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
def test_split_many_trajectories_holds_out_the_last_two():
    trajs = [_traj(60) for _ in range(6)]
    train, val, test = split_trajectories(trajs, architecture="MLP", verbose=False)
    assert len(test) == 2
    assert len(train) + len(val) == 4


def test_split_two_trajectories_does_not_crash():
    trajs = [_traj(120), _traj(120)]
    train, val, test = split_trajectories(trajs, architecture="MLP", verbose=False)
    assert len(test) == 1
    assert train and val


def test_split_single_trajectory_holds_out_final_ten_percent():
    traj = _traj(1000)
    train, val, test = split_trajectories(
        [traj], architecture="MLP", chunk_size=0, shuffle=False, verbose=False
    )
    assert len(test[0]["states"]) == pytest.approx(100, abs=2)
    # Test data must be the chronological tail, never shuffled
    np.testing.assert_allclose(test[0]["states"][0], traj["states"][900], rtol=1e-5)


def test_split_single_trajectory_with_chunking():
    train, val, test = split_trajectories(
        [_traj(1000)], architecture="MLP", chunk_size=100, shuffle=True, verbose=False
    )
    assert len(train) > 1
    assert len(val) >= 1


def test_lstm_forces_chronological_split():
    """Shuffling must be disabled for LSTM so temporal memory stays contiguous."""
    traj = _traj(1000)
    train, val, _ = split_trajectories(
        [traj], architecture="LSTM", chunk_size=0, shuffle=True, verbose=False
    )
    # Contiguous slice: times must be strictly increasing
    assert np.all(np.diff(train[0]["times"]) > 0)


def test_split_rejects_empty_input():
    with pytest.raises(ValueError):
        split_trajectories([], verbose=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def test_generate_all_plots_writes_five_figures(tmp_path: Path):
    paths = generate_all_plots(_history(), "plant", tmp_path, "20250101_000000", max_latency=2.0)
    assert len(paths) == 5
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    names = " ".join(p.name for p in paths)
    for expected in [
        "mse_convergence", "rmse_convergence", "hyperparameters",
        "latency_evolution", "layers_neurons_contour",
    ]:
        assert expected in names


def test_plots_survive_a_single_cycle_history(tmp_path: Path):
    """A one-cycle run must not crash the contour/LogNorm code path."""
    paths = generate_all_plots(_history(1), "plant", tmp_path, "20250101_000000")
    assert len(paths) == 5


def test_verification_rollout_returns_aligned_trajectories(tmp_path: Path):
    model = DynamicsModel(2, 1, [16], activation="relu", architecture="MLP")
    model.set_scalers(
        [0.0, 0.0], [1.0, 1.0], [0.0], [1.0], [0.0, 0.0], [1.0, 1.0]
    )
    true_traj, nn_traj = plot_test_dataset_verification(
        model, [_traj(300)], state_dim=2, action_dim=1,
        output_dir=tmp_path, timestamp="20250101_000000",
        env_name="plant", architecture="MLP", integrator="RK4",
    )
    assert true_traj.shape == nn_traj.shape
    assert true_traj.shape[1] == 2
    assert (tmp_path / "figures").exists()


def test_verification_handles_an_empty_test_set(tmp_path: Path):
    model = DynamicsModel(2, 1, [8], architecture="MLP")
    true_traj, nn_traj = plot_test_dataset_verification(
        model, [], state_dim=2, action_dim=1,
        output_dir=tmp_path, timestamp="t", env_name="plant",
    )
    assert len(true_traj) == 0 and len(nn_traj) == 0


# ---------------------------------------------------------------------------
# Deployable export
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ["MLP", "LSTM"])
def test_exported_controller_is_importable_and_predicts(tmp_path: Path, arch: str):
    """The generated standalone script must load the weights and run with no framework."""
    import subprocess
    import sys

    state_dim, action_dim, hidden = 2, 1, [16, 16]
    model = DynamicsModel(state_dim, action_dim, hidden, activation="relu", architecture=arch)
    model.set_scalers(
        [0.1, 0.2], [1.5, 2.0], [0.3], [1.1], [0.0, 0.0], [1.0, 1.0]
    )

    pth = save_best_model(
        model, hidden, state_dim, action_dim, 0.001234, {}, "relu",
        env_name="plant", timestamp="20250101_000000", output_dir=tmp_path,
    )
    script = export_standalone_inference_script(
        model=model, hidden_layers=hidden, activation="relu",
        state_dim=state_dim, action_dim=action_dim, pth_stem=pth.name,
        env_name="plant", output_dir=tmp_path, architecture=arch, lstm_seq_length=5,
    )
    write_nn_inference_helper("plant", tmp_path, integrator="RK4")

    assert script.exists()
    assert (tmp_path / "deployment" / "NN.py").exists()

    # Run the generated controller in a clean interpreter.
    code = (
        "import numpy as np, sys;"
        f"sys.path.insert(0, r'{script.parent}');"
        "from deployed_controller_plant import NeuralController;"
        f"c = NeuralController(weights_path=r'{pth}');"
        "out = c.predict_xdot(np.array([0.1,0.2],dtype=np.float32),"
        " np.array([0.3],dtype=np.float32), 0.01);"
        "c.commit_to_memory(np.array([0.1,0.2],dtype=np.float32),"
        " np.array([0.3],dtype=np.float32), 0.01);"
        "assert out.shape == (2,), out.shape;"
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# PDF & ZIP
# ---------------------------------------------------------------------------
def test_pdf_contains_all_nine_sections(tmp_path: Path):
    pytest.importorskip("pypdf")
    from pypdf import PdfReader

    generate_all_plots(_history(), "plant", tmp_path, "20250101_000000")
    pdf_path = generate_final_pdf(
        env_name="plant", state_dim=2, action_dim=1,
        best_config={"hidden_layers": [64, 32]},
        best_mse=0.001234, best_rmse=0.035, latency=0.42, use_pinn=False,
        timestamp="20250101_000000",
        abstract_text="Abstract body text.", conclusion_text="Conclusion body text.",
        success_score=88.5, model_status="STABLE & HIGH-FIDELITY",
        complexity_label="Level 3 (Moderately Complex)",
        output_dir=tmp_path, architecture="MLP",
    )
    text = "".join(p.extract_text() for p in PdfReader(pdf_path).pages)
    for section in [
        "Abstract", "1. System Structure", "2. Optimal Neural",
        "3. System Identification Convergence", "4. Absolute Error",
        "5. Architecture Topology", "6. Hyperparameter",
        "7. Real-Time Deployment", "8. Data-Driven Held-Out", "9. Concluding",
    ]:
        assert section in text, f"PDF is missing section: {section}"
    assert "Composite Success Score" in text
    assert "88.5" in text
    # Every referenced figure must have been found on disk.
    assert "[Error:" not in text


def test_pdf_sanitizes_unicode_that_latin1_cannot_encode(tmp_path: Path):
    """Smart quotes and em dashes from an LLM must not crash fpdf."""
    pdf_path = generate_final_pdf(
        env_name="plant", state_dim=1, action_dim=1,
        best_config={"hidden_layers": [8]},
        best_mse=0.01, best_rmse=0.1, latency=0.2, use_pinn=True,
        timestamp="20250101_000000",
        abstract_text="The model’s “fit” — excellent • stable – ready.",
        conclusion_text="Ready ‑ for deployment now.",
        success_score=70.0, model_status="STABLE & ACCEPTABLE",
        complexity_label="Level 2 (Mildly Nonlinear)",
        output_dir=tmp_path,
    )
    assert Path(pdf_path).exists()


def test_zip_has_the_expected_delivery_layout(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "deployment").mkdir(parents=True)
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "Agents_log").mkdir(parents=True)

    (run_dir / "figures" / "a.png").write_bytes(b"\x89PNG\r\n")
    (run_dir / "deployment" / "deployed_controller_plant.py").write_text("# controller")
    (run_dir / "report" / "System_Report_plant.pdf").write_bytes(b"%PDF-1.4")
    (run_dir / "Agents_log" / "agent_prompt_history.log").write_text("log")

    zip_path = package_final_results_to_zip(run_dir, "plant", "20250101_000000")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    assert "README.md" in names
    assert any(n.startswith("figures/") for n in names)
    assert any(n.startswith("deployment/") for n in names)
    assert any(n.startswith("report/") for n in names)
    assert any(n.startswith("Agents_log/") for n in names)
