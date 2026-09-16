"""Unit tests for DynamicsModel."""

from __future__ import annotations

import numpy as np
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
