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


@pytest.mark.parametrize("rows,sequence_length", [(41, 10), (10, 3)])
def test_small_single_lstm_recording_keeps_a_full_window_in_each_split(monkeypatch, rows, sequence_length):
    monkeypatch.setattr(cfg, "LSTM_SEQ_LENGTH", sequence_length)
    monkeypatch.setattr(cfg, "ROLLOUT_HORIZON", 1)
    monkeypatch.setattr(cfg, "TRAJECTORY_CHUNK_SIZE", 0)

    train, validation, test = split_trajectories(
        [_traj(rows)], architecture="LSTM", chunk_size=0, shuffle=False, verbose=False
    )

    assert len(train[0]["states"]) >= sequence_length
    assert len(validation[0]["states"]) >= sequence_length
    assert len(test[0]["states"]) >= sequence_length


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


def test_deployment_controller_preserves_periodic_angle_encoding(tmp_path: Path):
    import torch

    model = DynamicsModel(
        1, 0, [8], activation="tanh", architecture="MLP", angle_indices=[0]
    )
    model.set_scalers([0.0, 0.0], [1.0, 1.0], [], [], [0.0], [1.0])
    path = export_standalone_inference_script(
        model, [8], "tanh", 1, 0, "angle_weights.pth", "angle_plant",
        tmp_path, architecture="MLP",
    )
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    deployed = namespace["DeployableDynamicsModel"]()
    controller = object.__new__(namespace["NeuralController"])
    controller.model = deployed

    output = deployed(
        torch.tensor([[np.pi - 1e-4]]), torch.zeros((1, 0)), torch.tensor([[0.01]])
    )

    assert output.shape == (1, 1)
    assert deployed.angle_indices == [0]
    assert controller.wrap_state(np.array([np.pi + 0.1]))[0] == pytest.approx(-np.pi + 0.1)
    helper = write_nn_inference_helper("angle_plant", tmp_path)
    helper_text = helper.read_text(encoding="utf-8")
    assert "plant_model.model.state_dim" in helper_text
    assert "plant_model.wrap_state(next_state_prediction)" in helper_text


@pytest.mark.parametrize("arch", ["MLP", "LSTM"])
def test_exported_pcnn_controller_embeds_equation_and_keeps_exact_state(tmp_path: Path, arch: str):
    import torch

    equation = tmp_path / "physics_equation.py"
    equation.write_text(
        "def compute_analytical_xdot(states, actions):\n"
        "    return torch.stack((states[:, 0] + actions[:, 0], 2 * states[:, 1]), dim=1)\n",
        encoding="utf-8",
    )
    model = DynamicsModel(2, 1, [8], activation="tanh", architecture=arch, angle_indices=[0])
    model.set_scalers([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0], [1.0], [0.0, 0.0], [1.0, 3.0])
    model.configure_physics(
        "pcnn", lambda states, actions: torch.stack((states[:, 0] + actions[:, 0], 2 * states[:, 1]), dim=1),
        hard_constraint_indices=[0],
    )
    head = model.fc_out if arch == "LSTM" else model.network[-1]
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.5, 1.0]))
    weights = tmp_path / "pcnn_model.pth"
    torch.save(model.state_dict(), weights)
    script = export_standalone_inference_script(
        model=model, hidden_layers=[8], activation="tanh", state_dim=2, action_dim=1,
        pth_stem=weights.name, env_name="pcnn_plant", output_dir=tmp_path,
        architecture=arch, lstm_seq_length=3, physics_mode="pcnn",
        hard_constraint_indices=[0], physics_equation_file=equation,
    )
    namespace = {}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    controller = namespace["NeuralController"](weights_path=str(weights))
    output = controller.predict_xdot(
        np.array([0.2, 0.3], dtype=np.float32), np.array([0.4], dtype=np.float32), 0.01,
    )

    # State 0 is exactly equation-defined; state 1 includes its learned correction.
    np.testing.assert_allclose(output, [0.6, 3.6], atol=1e-6)


def test_pcnn_pdf_reports_equation_mode_and_exact_states(tmp_path: Path, monkeypatch):
    from backend_core.AgentSysID.reporting import report as report_module

    rows = []
    original_add_table_row = report_module.SystemIDReport.add_table_row

    def capture_row(self, label, value):
        rows.append((str(label), str(value)))
        return original_add_table_row(self, label, value)

    monkeypatch.setattr(report_module.SystemIDReport, "add_table_row", capture_row)

    pdf_path = generate_final_pdf(
        env_name="plant", state_dim=2, action_dim=1,
        best_config={"hidden_layers": [8]}, best_mse=0.01, best_rmse=0.1,
        latency=0.2, use_pinn=True, physics_mode="pcnn",
        pcnn_hard_constraint_states=["s_pitch"], timestamp="pcnn-report",
        abstract_text="A physics-constrained model fit measured derivatives.",
        conclusion_text="The held-out score does not prove closed-loop stability.",
        success_score=None, model_status="ROLLOUT UNVERIFIED",
        complexity_label="Level 2", output_dir=tmp_path,
    )
    assert Path(pdf_path).is_file()
    assert ("Architecture Mode", "PCNN (equation-constrained) + MLP (Memoryless State)") in rows
    assert ("Exact PCNN State Equations", "s_pitch") in rows


def test_verification_rolls_out_each_held_out_trajectory_independently(tmp_path: Path):
    model = DynamicsModel(2, 1, [8], activation="relu", architecture="MLP")
    first, second = _traj(14), _traj(19)
    second["states"] = second["states"] + np.array([3.0, -2.0], dtype=np.float32)
    truth, predicted, ranges = plot_test_dataset_verification(
        model, [first, second], state_dim=2, action_dim=1,
        output_dir=tmp_path, timestamp="multi", env_name="plant", architecture="MLP",
        save_plot=False, return_trajectory_ranges=True,
    )
    assert truth.shape == predicted.shape == (33, 2)
    assert ranges == [
        {"trajectory_index": 1, "start": 0, "stop": 14},
        {"trajectory_index": 2, "start": 14, "stop": 33},
    ]
    np.testing.assert_allclose(truth[0], first["states"][0])
    np.testing.assert_allclose(truth[14], second["states"][0])


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
        "7. Inference Timing", "8. Data-Driven Held-Out", "9. Conclusion & Validation Limits",
    ]:
        assert section in text, f"PDF is missing section: {section}"
    assert "Fit + Held-out Rollout Score" in text
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


def test_pdf_shows_missing_rollout_score_as_unavailable(tmp_path: Path):
    pytest.importorskip("pypdf")
    from pypdf import PdfReader
    pdf_path = generate_final_pdf(
        env_name="plant", state_dim=1, action_dim=1,
        best_config={"hidden_layers": [8]}, best_mse=0.01, best_rmse=0.1,
        latency=0.2, use_pinn=False, timestamp="missing-rollout",
        abstract_text="Validation summary.",
        conclusion_text="A composite score is unavailable because no rollout was saved.",
        success_score=None, model_status="ROLLOUT UNVERIFIED",
        complexity_label="Level 2", output_dir=tmp_path,
    )
    text = "".join(page.extract_text() for page in PdfReader(pdf_path).pages)
    assert "Fit + Held-out Rollout Score" in text
    assert "Unavailable" in text


def test_zip_has_the_expected_delivery_layout(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "deployment").mkdir(parents=True)
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "Agents_log").mkdir(parents=True)

    (run_dir / "figures" / "a.png").write_bytes(b"\x89PNG\r\n")
    (run_dir / "deployment" / "deployed_controller_plant.py").write_text("# controller")
    (run_dir / "deployment" / "NN.py").write_text("# inference script")
    (run_dir / "deployment" / "best_model_plant.pth").write_bytes(b"model weights")
    (run_dir / "report" / "System_Report_plant.pdf").write_bytes(b"%PDF-1.4")
    (run_dir / "Agents_log" / "agent_prompt_history.log").write_text("log")

    zip_path = package_final_results_to_zip(run_dir, "plant", "20250101_000000")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    assert "README.md" in names
    assert "deployment/deployed_controller_plant.py" in names
    assert "deployment/NN.py" in names
    assert "deployment/best_model_plant.pth" in names
    assert "report/System_Report_plant.pdf" in names
    assert "figures/a.png" not in names
    assert "Agents_log/agent_prompt_history.log" not in names
