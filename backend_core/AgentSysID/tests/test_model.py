"""Unit tests for DynamicsModel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend_core.AgentSysID.model.dynamics_model import DynamicsModel


def test_mlp_forward_shape():
    model = DynamicsModel(
        state_dim=4, action_dim=2, hidden_layers=[32, 32],
        activation="relu", architecture="MLP",
    )
    s = torch.randn(8, 4)
    a = torch.randn(8, 2)
    dt = torch.ones(8, 1) * 0.01
    out_phys, out_norm = model(s, a, dt)
    assert out_phys.shape == (8, 4)
    assert out_norm.shape == (8, 4)


def test_lstm_forward_shape():
    model = DynamicsModel(
        state_dim=3, action_dim=1, hidden_layers=[16, 16],
        activation="elu", architecture="LSTM",
    )
    s = torch.randn(5, 3)
    a = torch.randn(5, 1)
    dt = torch.ones(5, 1) * 0.02
    out_phys, out_norm = model(s, a, dt)
    assert out_phys.shape == (5, 3)


def test_set_scalers_and_step():
    model = DynamicsModel(state_dim=2, action_dim=1, hidden_layers=[8], architecture="MLP")
    model.set_scalers(
        state_mean=[0.0, 0.0], state_scale=[1.0, 1.0],
        action_mean=[0.0], action_scale=[1.0],
        output_mean=[0.0, 0.0], output_scale=[1.0, 1.0],
    )
    s = torch.zeros(1, 2)
    a = torch.zeros(1, 1)
    next_s = model.step_environment(s, a, 0.01)
    assert next_s.shape == (1, 2)


def test_periodic_angle_inputs_are_nearly_identical_across_the_pi_seam():
    torch.manual_seed(4)
    model = DynamicsModel(
        state_dim=1, action_dim=0, hidden_layers=[16], activation="tanh",
        architecture="MLP", angle_indices=[0],
    )
    model.set_scalers(
        state_mean=[0.0, 0.0], state_scale=[1.0, 1.0],
        action_mean=[], action_scale=[], output_mean=[0.0], output_scale=[1.0],
    )
    angle = torch.tensor([[np.pi - 1e-4], [-np.pi + 1e-4]], dtype=torch.float32)
    action = torch.zeros((2, 0))
    dt = torch.full((2, 1), 0.01)

    predicted, _ = model(angle, action, dt)

    assert model.state_feature_dim == 2
    assert model.state_mean.shape == (2,)
    assert torch.max(torch.abs(predicted[0] - predicted[1])).item() < 1e-3


@pytest.mark.parametrize("architecture", ["MLP", "LSTM"])
def test_pcnn_anchors_derivatives_to_physics_and_can_lock_selected_states(architecture):
    model = DynamicsModel(
        state_dim=2, action_dim=1, hidden_layers=[8], activation="tanh",
        architecture=architecture,
    )
    model.set_scalers(
        state_mean=[0.0, 0.0], state_scale=[1.0, 1.0],
        action_mean=[0.0], action_scale=[1.0],
        output_mean=[5.0, -3.0], output_scale=[2.0, 3.0],
    )

    def physics(states, actions):
        return torch.stack((states[:, 0] + actions[:, 0], 2 * states[:, 1]), dim=1)

    model.configure_physics("pcnn", physics, hard_constraint_indices=[0])
    head = model.fc_out if architecture == "LSTM" else model.network[-1]
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.5, 1.0]))

    states = torch.tensor([[[0.2, 0.3], [0.4, 0.5]]]) if architecture == "LSTM" else torch.tensor([[0.4, 0.5]])
    actions = torch.tensor([[[0.1], [0.2]]]) if architecture == "LSTM" else torch.tensor([[0.2]])
    dts = torch.full((1, 2, 1), 0.01) if architecture == "LSTM" else torch.tensor([[0.01]])
    physical, normalized, correction = model(states, actions, dts, return_components=True)

    # The first state is exactly the equation. The second adds one learned
    # residual (normalized head value multiplied by its physical output scale).
    torch.testing.assert_close(physical, torch.tensor([[0.6, 4.0]]))
    torch.testing.assert_close(normalized, (physical - model.output_mean) / model.output_scale)
    torch.testing.assert_close(correction, torch.tensor([[0.5, 1.0]]))
