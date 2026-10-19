"""Tests for the training loop, rollout windowing and PINN hook."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest
import torch

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.data.splitter import split_trajectories
from backend_core.AgentSysID.training.trainer import (
    _build_windows,
    compute_rmse,
    measure_inference_latency,
    train_dynamics_model,
)


@pytest.fixture(autouse=True)
def _restore_config():
    saved = {
        k: getattr(cfg, k)
        for k in (
            "NETWORK_ARCHITECTURE",
            "LSTM_SEQ_LENGTH",
            "ROLLOUT_HORIZON",
            "USE_PINN",
            "USE_STATE_FILTER",
            "OVERFIT_RATIO_LIMIT",
            "ANGLE_INDICES",
            "PHYSICS_MODE",
            "PCNN_HARD_CONSTRAINT_INDICES",
        )
    }
    cfg.NETWORK_ARCHITECTURE = "MLP"
    cfg.LSTM_SEQ_LENGTH = 10
    cfg.ROLLOUT_HORIZON = 1
    cfg.USE_PINN = False
    cfg.USE_STATE_FILTER = False
    cfg.ANGLE_INDICES = []
    cfg.PHYSICS_MODE = "pinn"
    cfg.PCNN_HARD_CONSTRAINT_INDICES = []
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)


def _linear_trajectory(n: int = 400, state_dim: int = 2, action_dim: int = 1) -> Dict[str, np.ndarray]:
    """x_dot is an exact linear function of (state, action): perfectly learnable."""
    rng = np.random.default_rng(0)
    states = rng.standard_normal((n, state_dim)).astype(np.float32)
    actions = rng.standard_normal((n, action_dim)).astype(np.float32)
    # Every state derivative depends on all states and the summed action.
    action_effect = actions.sum(axis=1, keepdims=True)
    xdots = (0.5 * states + action_effect).astype(np.float32)
    dts = np.full(n, 0.01, dtype=np.float32)
    times = np.arange(n, dtype=np.float64) * 0.01
    return {"states": states, "actions": actions, "xdots": xdots, "dts": dts, "times": times}


def test_compute_rmse_handles_negative_and_zero():
    assert compute_rmse(4.0) == 2.0
    assert compute_rmse(0.0) == 0.0
    assert compute_rmse(-1.0) == 0.0


def test_build_windows_shapes():
    traj = _linear_trajectory(n=50, state_dim=3, action_dim=2)
    s, a, d, x = _build_windows([traj], total_len=10, state_dim=3, action_dim=2)
    assert s.shape == (41, 10, 3)
    assert a.shape == (41, 10, 2)
    assert d.shape == (41, 10, 1)
    assert x.shape == (41, 10, 3)


def test_build_windows_skips_trajectories_shorter_than_the_window():
    short = _linear_trajectory(n=4, state_dim=2, action_dim=1)
    s, _, _, _ = _build_windows([short], total_len=10, state_dim=2, action_dim=1)
    assert len(s) == 0


def test_mlp_training_reduces_error_on_a_learnable_system():
    train = [_linear_trajectory(n=600)]
    val = [_linear_trajectory(n=200)]
    model, train_mse, val_mse, val_rmse, _, _ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[32, 32], learning_rate=1e-2, epochs=60,
        batch_size=64, patience=25, activation="relu", architecture="MLP",
        verbose=False,
    )
    assert np.isfinite(val_mse)
    # The mapping is exactly representable, so the fit should be tight.
    assert val_mse < 0.05
    assert val_rmse == pytest.approx(np.sqrt(val_mse), rel=1e-4)


def test_lstm_training_runs_with_sequence_windows():
    cfg.NETWORK_ARCHITECTURE = "LSTM"
    cfg.LSTM_SEQ_LENGTH = 5
    train = [_linear_trajectory(n=300)]
    val = [_linear_trajectory(n=120)]
    model, _, val_mse, _, _, _ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[16, 16], learning_rate=5e-3, epochs=15,
        batch_size=32, patience=10, activation="tanh", architecture="LSTM",
        verbose=False,
    )
    assert np.isfinite(val_mse)
    assert model.arch == "LSTM"


def test_rollout_horizon_greater_than_one_trains():
    cfg.ROLLOUT_HORIZON = 3
    train = [_linear_trajectory(n=300)]
    val = [_linear_trajectory(n=120)]
    _, _, val_mse, _, _, _ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[16], learning_rate=5e-3, epochs=10,
        batch_size=32, patience=10, activation="relu", architecture="MLP",
        verbose=False,
    )
    assert np.isfinite(val_mse)


def test_scalers_are_baked_into_the_model_buffers():
    train = [_linear_trajectory(n=300)]
    val = [_linear_trajectory(n=100)]
    model, _, _, _, _, _ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[8], learning_rate=1e-3, epochs=3,
        batch_size=64, patience=5, activation="relu", architecture="MLP",
        verbose=False,
    )
    # Standardised data: means near 0, scales near 1, and never zero.
    assert torch.all(model.state_scale > 0)
    assert torch.all(model.output_scale > 0)
    assert model.state_mean.shape == (2,)


def test_trainer_fits_scaler_on_periodic_features_for_confirmed_angles():
    def angular_trajectory(n):
        theta = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=np.float32)
        return {
            "states": theta[:, None],
            "actions": np.zeros((n, 0), dtype=np.float32),
            "xdots": np.cos(theta)[:, None].astype(np.float32),
            "dts": np.full(n, 0.01, dtype=np.float32),
            "times": np.arange(n, dtype=np.float64) * 0.01,
        }

    model, *_ = train_dynamics_model(
        [angular_trajectory(120)], [angular_trajectory(60)], state_dim=1,
        action_encoding_dim=0, hidden_layers=[8], learning_rate=1e-3,
        epochs=2, batch_size=32, patience=2, architecture="MLP",
        angle_indices=[0], verbose=False,
    )

    assert model.angle_indices == (0,)
    assert model.state_feature_dim == 2
    assert model.state_mean.shape == (2,)


def test_pinn_residual_is_applied_when_enabled(monkeypatch):
    """With USE_PINN on, the physics hook must be consulted during training."""
    calls = {"n": 0}

    def fake_physics(states, actions):
        calls["n"] += 1
        return torch.zeros(states.shape[0], states.shape[1], device=states.device)

    monkeypatch.setattr(
        "backend_core.AgentSysID.training.trainer.load_analytical_xdot",
        lambda _path: fake_physics,
    )
    cfg.USE_PINN = True

    train = [_linear_trajectory(n=200)]
    val = [_linear_trajectory(n=80)]
    train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[8], learning_rate=1e-3, epochs=2,
        batch_size=64, patience=5, activation="relu", architecture="MLP",
        verbose=False,
    )
    assert calls["n"] > 0, "PINN physics function was never called"


def test_pcnn_trains_equation_anchored_model_and_preserves_exact_state(monkeypatch):
    calls = {"n": 0}

    def fake_physics(states, actions):
        calls["n"] += 1
        return torch.stack((2 * states[:, 0] + actions[:, 0], states[:, 1] - actions[:, 0]), dim=1)

    monkeypatch.setattr(
        "backend_core.AgentSysID.training.trainer.load_analytical_xdot",
        lambda _path: fake_physics,
    )
    cfg.USE_PINN = True
    cfg.PHYSICS_MODE = "pcnn"
    cfg.PCNN_HARD_CONSTRAINT_INDICES = [0]

    train = [_linear_trajectory(n=200)]
    val = [_linear_trajectory(n=80)]
    model, *_ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[8], learning_rate=1e-3, epochs=2,
        batch_size=64, patience=5, activation="relu", architecture="MLP",
        verbose=False,
    )
    sample_states = torch.tensor([[0.4, -0.2]])
    sample_actions = torch.tensor([[0.3]])
    exact = fake_physics(sample_states, sample_actions)
    model.cpu()
    predicted, _ = model(sample_states, sample_actions, torch.tensor([[0.01]]))

    assert calls["n"] > 0
    assert model.physics_mode == "pcnn"
    assert model.hard_constraint_indices == [0]
    torch.testing.assert_close(predicted[:, 0], exact[:, 0])


def test_equation_loader_does_not_reuse_same_named_file_from_another_run(tmp_path):
    from backend_core.AgentSysID.model.pinn import load_analytical_xdot

    functions = []
    for folder, value in (("run_one", 1.0), ("run_two", 2.0)):
        target = tmp_path / folder / "physics_equation.py"
        target.parent.mkdir()
        target.write_text(
            "import torch\n"
            "def compute_analytical_xdot(states, actions):\n"
            f"    return torch.full((len(states), 1), {value}, device=states.device)\n",
            encoding="utf-8",
        )
        functions.append(load_analytical_xdot(target))

    probe = torch.zeros((1, 1))
    action = torch.zeros((1, 0))
    assert functions[0] is not None and functions[1] is not None
    assert functions[0](probe, action).item() == 1.0
    assert functions[1](probe, action).item() == 2.0


def test_pinn_enabled_fails_preflight_instead_of_silently_training_without_physics(monkeypatch):
    monkeypatch.setattr(
        "backend_core.AgentSysID.training.trainer.load_analytical_xdot",
        lambda _path: None,
    )
    cfg.USE_PINN = True
    train = [_linear_trajectory(n=100)]
    with pytest.raises(ValueError, match="could not be loaded"):
        train_dynamics_model(
            train, train, state_dim=2, action_encoding_dim=1,
            hidden_layers=[8], epochs=1, batch_size=32, patience=3,
            architecture="MLP", verbose=False,
        )


def test_pinn_equation_output_shape_is_checked_before_training(monkeypatch):
    monkeypatch.setattr(
        "backend_core.AgentSysID.training.trainer.load_analytical_xdot",
        lambda _path: lambda states, _actions: torch.zeros((len(states), 1), device=states.device),
    )
    cfg.USE_PINN = True
    train = [_linear_trajectory(n=100)]
    with pytest.raises(ValueError, match="must return a tensor with shape"):
        train_dynamics_model(
            train, train, state_dim=2, action_encoding_dim=1,
            hidden_layers=[8], epochs=1, batch_size=32, patience=3,
            architecture="MLP", verbose=False,
        )


def test_training_raises_when_no_window_fits():
    tiny = [_linear_trajectory(n=3)]
    cfg.LSTM_SEQ_LENGTH = 50
    with pytest.raises(ValueError, match="No training windows"):
        train_dynamics_model(
            tiny, tiny, state_dim=2, action_encoding_dim=1,
            hidden_layers=[8], learning_rate=1e-3, epochs=2,
            batch_size=8, patience=3, activation="relu", architecture="LSTM",
            verbose=False,
        )


def test_measure_inference_latency_is_positive():
    from backend_core.AgentSysID.model.dynamics_model import DynamicsModel

    model = DynamicsModel(3, 2, [16], activation="relu", architecture="MLP")
    latency = measure_inference_latency(model, 3, 2, n_runs=20)
    assert latency > 0.0


def test_graceful_stop_halts_training(monkeypatch):
    from backend_core.AgentSysID.utils import stop_control

    stop_control.request_stop()
    try:
        train = [_linear_trajectory(n=200)]
        val = [_linear_trajectory(n=80)]
        model, train_mse, val_mse, _, _, _ = train_dynamics_model(
            train, val, state_dim=2, action_encoding_dim=1,
            hidden_layers=[8], learning_rate=1e-3, epochs=500,
            batch_size=32, patience=25, activation="relu", architecture="MLP",
            verbose=False,
        )
        # Stopped before any epoch completed: metrics stay at their sentinels
        # but a usable model object is still returned.
        assert model is not None
        assert train_mse == float("inf")
    finally:
        stop_control.reset_stop_flag()


def test_empty_validation_set_raises_instead_of_reporting_a_perfect_score():
    """
    Regression: a validation split shorter than the rollout window used to
    average 0/0 and report val MSE == 0.0 — a perfect score from no data,
    which the Critic would then tune against. It must fail loudly instead.
    """
    cfg.NETWORK_ARCHITECTURE = "LSTM"
    cfg.LSTM_SEQ_LENGTH = 10
    train = [_linear_trajectory(n=300)]
    too_short = [_linear_trajectory(n=6)]  # 6 rows < the 10-step window

    with pytest.raises(ValueError, match="no windows"):
        train_dynamics_model(
            train, too_short, state_dim=2, action_encoding_dim=1,
            hidden_layers=[16], learning_rate=1e-3, epochs=60,
            batch_size=32, patience=5, activation="relu", architecture="LSTM",
            verbose=False,
        )


def test_small_lstm_recording_trains_with_automatically_sized_windows():
    cfg.NETWORK_ARCHITECTURE = "LSTM"
    cfg.LSTM_SEQ_LENGTH = 3
    cfg.ROLLOUT_HORIZON = 1
    train, validation, _ = split_trajectories(
        [_linear_trajectory(n=10)], architecture="LSTM", chunk_size=0, shuffle=False, verbose=False
    )

    model, train_mse, val_mse, val_rmse, _, _ = train_dynamics_model(
        train, validation, state_dim=2, action_encoding_dim=1,
        hidden_layers=[8], learning_rate=1e-3, epochs=1, batch_size=8,
        patience=1, activation="relu", architecture="LSTM", verbose=False,
        include_state_metrics=True,
    )

    assert model is not None
    assert np.isfinite(train_mse) and np.isfinite(val_mse) and np.isfinite(val_rmse)
    assert len(model.state_validation_mse) == 2
    assert np.isfinite(model.state_validation_mse).all()


def test_empty_validation_error_names_the_offending_settings():
    cfg.LSTM_SEQ_LENGTH = 8
    cfg.ROLLOUT_HORIZON = 3
    with pytest.raises(ValueError) as excinfo:
        train_dynamics_model(
            [_linear_trajectory(n=200)], [_linear_trajectory(n=4)],
            state_dim=2, action_encoding_dim=1, hidden_layers=[8],
            learning_rate=1e-3, epochs=5, batch_size=16, patience=5,
            activation="relu", architecture="LSTM", verbose=False,
        )
    message = str(excinfo.value)
    assert "LSTM_SEQ_LENGTH=8" in message
    assert "ROLLOUT_HORIZON=3" in message
    assert "4 rows" in message          # what the split actually held
    assert "length 10" in message       # 8 + 3 - 1


def test_zero_best_loss_does_not_divide_by_zero():
    """The epoch-extension ratio is undefined at a zero best loss."""
    train = [_linear_trajectory(n=200)]
    val = [_linear_trajectory(n=80)]
    # Exercising the extension branch must not raise.
    model, train_mse, val_mse, _, _, _ = train_dynamics_model(
        train, val, state_dim=2, action_encoding_dim=1,
        hidden_layers=[8], learning_rate=1e-3, epochs=3,
        batch_size=32, patience=25, activation="relu", architecture="MLP",
        verbose=False,
    )
    assert np.isfinite(val_mse)
