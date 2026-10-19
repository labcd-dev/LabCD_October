"""
DynamicsModel – MLP or LSTM network that predicts state derivatives (x_dot).

Input  : [state, action, dt] (normalised internally)
Output : (out_phys, out_norm) – physical derivatives for integration and
         normalised derivatives for the loss.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from backend_core.AgentSysID.model.angles import valid_angle_indices


class DynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        architecture: Optional[str] = None,
        angle_indices: Optional[list[int]] = None,
        physics_mode: str = "pinn",
        physics_fn: Optional[Callable] = None,
        hard_constraint_indices: Optional[list[int]] = None,
        pcnn_output_mode: str = "hard_constraints",
    ):
        super().__init__()

        # Determine architecture mode: explicit argument wins, otherwise fall
        # back to the live config (legacy behaviour), then to MLP.
        if architecture is None:
            try:
                from backend_core.AgentSysID import config as cfg

                architecture = cfg.NETWORK_ARCHITECTURE
            except Exception:
                architecture = "MLP"

        self.arch = (architecture or "MLP").strip().upper()
        self.state_dim = int(state_dim)
        self.angle_indices = valid_angle_indices(angle_indices, state_dim)
        self.state_feature_dim = state_dim + len(self.angle_indices)
        self.physics_mode = str(physics_mode or "pinn").strip().lower()
        if self.physics_mode not in {"pinn", "pcnn"}:
            raise ValueError("physics_mode must be 'pinn' or 'pcnn'.")
        self.physics_fn = physics_fn
        self.pcnn_output_mode = str(pcnn_output_mode or "hard_constraints").strip().lower()
        if self.pcnn_output_mode not in {"hard_constraints", "equation_residual"}:
            raise ValueError("pcnn_output_mode must be 'hard_constraints' or 'equation_residual'.")
        self.hard_constraint_indices = self._validated_constraint_indices(hard_constraint_indices)

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

        in_dim = self.state_feature_dim + action_dim + 1  # encoded state + action + dt

        if self.arch == "LSTM":
            widths = [int(width) for width in hidden_layers] or [128, 128]
            if len(set(widths)) == 1:
                # Keep the original stacked module and state-dict keys for
                # existing equal-width LSTM checkpoints.
                hidden_size = widths[0]
                num_layers = len(widths)
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0.0,
                )
                self.lstm_layers = None
                self.lstm_dropout = None
            else:
                # PyTorch's stacked nn.LSTM requires one shared hidden size.
                # A stack of single-layer modules supports independent widths.
                recurrent_layers: list[nn.Module] = []
                layer_input_dim = in_dim
                for width in widths:
                    recurrent_layers.append(nn.LSTM(
                        input_size=layer_input_dim,
                        hidden_size=width,
                        num_layers=1,
                        batch_first=True,
                    ))
                    layer_input_dim = width
                self.lstm = None
                self.lstm_layers = nn.ModuleList(recurrent_layers)
                self.lstm_dropout = nn.Dropout(p=dropout_rate)
            self.fc_out = nn.Linear(widths[-1], state_dim)
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
        self.register_buffer("state_mean", torch.zeros(self.state_feature_dim))
        self.register_buffer("state_scale", torch.ones(self.state_feature_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("output_mean", torch.zeros(state_dim))
        self.register_buffer("output_scale", torch.ones(state_dim))
        if self.physics_mode == "pcnn":
            self.configure_physics("pcnn", physics_fn, self.hard_constraint_indices)

    def _validated_constraint_indices(self, indices: Optional[list[int]]) -> list[int]:
        try:
            normalized = sorted({int(index) for index in (indices or [])})
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCNN hard constraints must be integer state indices.") from exc
        if any(index < 0 or index >= self.state_dim for index in normalized):
            raise ValueError("PCNN hard-constraint index is outside the model output range.")
        if (self.physics_mode == "pcnn" and self.pcnn_output_mode == "hard_constraints"
                and normalized and len(normalized) >= self.state_dim):
            raise ValueError("PCNN must leave at least one state derivative for the neural network to learn.")
        return normalized

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

    def configure_physics(self, mode: str, physics_fn: Optional[Callable],
                          hard_constraint_indices: Optional[list[int]] = None,
                          pcnn_output_mode: Optional[str] = None) -> None:
        """Attach a validated equation and select soft PINN or hard PCNN outputs."""
        normalized_mode = str(mode or "pinn").strip().lower()
        if normalized_mode not in {"pinn", "pcnn"}:
            raise ValueError("physics_mode must be 'pinn' or 'pcnn'.")
        if normalized_mode == "pcnn" and physics_fn is None:
            raise ValueError("PCNN needs a validated equation to build its constrained output.")
        self.physics_mode = normalized_mode
        self.physics_fn = physics_fn
        if pcnn_output_mode is not None:
            self.pcnn_output_mode = str(pcnn_output_mode or "hard_constraints").strip().lower()
            if self.pcnn_output_mode not in {"hard_constraints", "equation_residual"}:
                raise ValueError("pcnn_output_mode must be 'hard_constraints' or 'equation_residual'.")
        self.hard_constraint_indices = self._validated_constraint_indices(hard_constraint_indices)
        if normalized_mode == "pcnn" and self.pcnn_output_mode == "hard_constraints":
            if not self.hard_constraint_indices:
                raise ValueError("PCNN needs at least one selected hard-constrained state.")
            if len(self.hard_constraint_indices) >= self.state_dim:
                raise ValueError("PCNN must leave at least one state derivative for the neural network to learn.")

    def forward(self, state: torch.Tensor, action: torch.Tensor, dt: torch.Tensor,
                *, return_components: bool = False):
        """
        Takes raw physical inputs, normalizes them, passes them through the network,
        and returns both physical (un-normalized) and normalized outputs.
        """
        if self.angle_indices:
            features = []
            for index in range(self.state_dim):
                value = state[..., index:index + 1]
                if index in self.angle_indices:
                    features.extend((torch.sin(value), torch.cos(value)))
                else:
                    features.append(value)
            state_features = torch.cat(features, dim=-1)
        else:
            state_features = state
        state_norm = (state_features - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale

        model_input = torch.cat([state_norm, action_norm, dt], dim=-1)

        if self.arch == "LSTM":
            if model_input.dim() == 2:
                model_input = model_input.unsqueeze(1)
            if self.lstm is not None:
                lstm_out, _ = self.lstm(model_input)
            else:
                lstm_out = model_input
                for index, layer in enumerate(self.lstm_layers):
                    lstm_out, _ = layer(lstm_out)
                    if index < len(self.lstm_layers) - 1:
                        lstm_out = self.lstm_dropout(lstm_out)
            last_step_out = lstm_out[:, -1, :]
            out_norm = self.fc_out(last_step_out)
        else:
            out_norm = self.network(model_input)

        residual_norm = None
        if self.physics_mode == "pcnn":
            if self.physics_fn is None:
                raise ValueError("PCNN is active but no validated physics equation is attached.")
            physics_state = state[:, -1, :] if state.dim() == 3 else state
            physics_action = action[:, -1, :] if action.dim() == 3 else action
            physics_xdot = self.physics_fn(physics_state, physics_action)
            expected_shape = (physics_state.shape[0], self.state_dim)
            if not torch.is_tensor(physics_xdot) or tuple(physics_xdot.shape) != expected_shape:
                raise ValueError(f"PCNN equation must return a tensor with shape {expected_shape}.")
            if not torch.isfinite(physics_xdot).all():
                raise ValueError("PCNN equation returned a non-finite derivative.")
            if self.pcnn_output_mode == "equation_residual":
                # Preserve inference behavior for PCNN runs saved before the
                # hard-output semantics were introduced.
                residual_norm = out_norm
                correction = residual_norm * self.output_scale
                if self.hard_constraint_indices:
                    free_mask = torch.ones(self.state_dim, device=state.device, dtype=state.dtype)
                    free_mask[self.hard_constraint_indices] = 0.0
                    correction = correction * free_mask
                out_phys = physics_xdot + correction
            else:
                # Exact constraints replace only selected derivative outputs.
                # Every remaining output is a normal data-trained NN prediction.
                out_phys = out_norm * self.output_scale + self.output_mean
                hard_mask = torch.zeros(self.state_dim, device=state.device, dtype=torch.bool)
                hard_mask[self.hard_constraint_indices] = True
                out_phys = torch.where(hard_mask.unsqueeze(0), physics_xdot, out_phys)
            out_norm = (out_phys - self.output_mean) / self.output_scale
        else:
            out_phys = out_norm * self.output_scale + self.output_mean
        if return_components:
            return out_phys, out_norm, residual_norm
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
