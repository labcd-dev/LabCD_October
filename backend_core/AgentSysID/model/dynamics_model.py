"""
DynamicsModel – MLP or LSTM network that predicts state derivatives (x_dot).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        architecture: str = "MLP",
    ):
        super().__init__()

        self.arch = (architecture or "MLP").strip().upper()

        # Mapping string activation to layer types
        act_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "swish": nn.SiLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        act_fn = act_map.get(activation.lower(), nn.Tanh)

        in_dim = state_dim + action_dim + 1  # state + action + dt

        if self.arch == "LSTM":
            hidden_size = int(hidden_layers[0]) if hidden_layers else 128
            num_layers = int(len(hidden_layers)) if hidden_layers else 2

            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )
            self.fc_out = nn.Linear(hidden_size, state_dim)
        else:  # Standard MLP
            layers: list[nn.Module] = []
            curr_dim = in_dim
            for h_size in hidden_layers:
                layers.append(nn.Linear(curr_dim, int(h_size)))
                layers.append(act_fn())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                curr_dim = int(h_size)
            layers.append(nn.Linear(curr_dim, state_dim))
            self.network = nn.Sequential(*layers)

        # Buffers for Data Normalization (saved with state_dict)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("output_mean", torch.zeros(state_dim))
        self.register_buffer("output_scale", torch.ones(state_dim))

    def set_scalers(
        self,
        state_mean,
        state_scale,
        action_mean,
        action_scale,
        output_mean,
        output_scale,
    ):
        """Sets the scaling parameters calculated from the training dataset."""
        state_scale = np.where(np.asarray(state_scale) < 1e-8, 1.0, state_scale)
        action_scale = np.where(np.asarray(action_scale) < 1e-8, 1.0, action_scale)
        output_scale = np.where(np.asarray(output_scale) < 1e-8, 1.0, output_scale)

        self.state_mean.copy_(torch.tensor(state_mean, dtype=torch.float32))
        self.state_scale.copy_(torch.tensor(state_scale, dtype=torch.float32))
        self.action_mean.copy_(torch.tensor(action_mean, dtype=torch.float32))
        self.action_scale.copy_(torch.tensor(action_scale, dtype=torch.float32))
        self.output_mean.copy_(torch.tensor(output_mean, dtype=torch.float32))
        self.output_scale.copy_(torch.tensor(output_scale, dtype=torch.float32))

    def forward(self, state: torch.Tensor, action: torch.Tensor, dt: torch.Tensor):
        """
        Takes raw physical inputs, normalizes them, passes them through the network,
        and returns both physical (un-normalized) and normalized outputs.
        """
        state_norm = (state - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale

        model_input = torch.cat([state_norm, action_norm, dt], dim=-1)

        if self.arch == "LSTM":
            if model_input.dim() == 2:
                model_input = model_input.unsqueeze(1)
            lstm_out, _ = self.lstm(model_input)
            last_step_out = lstm_out[:, -1, :]
            out_norm = self.fc_out(last_step_out)
        else:
            out_norm = self.network(model_input)

        out_phys = out_norm * self.output_scale + self.output_mean
        return out_phys, out_norm

    def step_environment(self, current_state: torch.Tensor, action: torch.Tensor, dt):
        """Simulates the next environment state step using physical units (Euler)."""
        if not torch.is_tensor(dt):
            dt = torch.full(
                (current_state.shape[0], 1),
                float(dt),
                dtype=current_state.dtype,
                device=current_state.device,
            )
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        out_phys, _ = self.forward(current_state, action, dt)
        next_state = current_state + out_phys * dt
        return next_state
