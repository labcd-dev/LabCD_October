import torch
import torch.nn as nn
import numpy as np


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation='relu', dropout_rate=0.0):
        super().__init__()

        # Determine Architecture Mode
        try:
            from config import NETWORK_ARCHITECTURE
            self.arch = NETWORK_ARCHITECTURE.strip().upper()
        except ImportError:
            self.arch = "MLP"

        # Mapping string activation to layer types
        if activation.lower() == 'relu':
            act_fn = nn.ReLU
        elif activation.lower() == 'elu':
            act_fn = nn.ELU
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU
        elif activation.lower() == 'swish':
            act_fn = nn.SiLU
        else:
            act_fn = nn.Tanh

        in_dim = state_dim + action_dim + 1

        if self.arch == "LSTM":
            # --- FIX: Explicitly cast to Python int() to prevent PyTorch TypeErrors ---
            hidden_size = int(hidden_layers[0]) if hidden_layers else 128
            num_layers = int(len(hidden_layers)) if hidden_layers else 2

            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0
            )
            self.fc_out = nn.Linear(hidden_size, state_dim)

        else:  # Standard MLP
            layers = []
            curr_dim = in_dim
            for h_size in hidden_layers:
                layers.append(nn.Linear(curr_dim, h_size))
                layers.append(act_fn())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                curr_dim = h_size
            layers.append(nn.Linear(curr_dim, state_dim))
            self.network = nn.Sequential(*layers)

        # Buffers for Data Normalization (Saved automatically with the model state_dict)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_scale', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))
        self.register_buffer('output_mean', torch.zeros(state_dim))
        self.register_buffer('output_scale', torch.ones(state_dim))

    def set_scalers(self, state_mean, state_scale, action_mean, action_scale, output_mean, output_scale):
        """Sets the scaling parameters calculated from the training dataset."""
        # Prevent division by zero
        state_scale = np.where(state_scale < 1e-8, 1.0, state_scale)
        action_scale = np.where(action_scale < 1e-8, 1.0, action_scale)
        output_scale = np.where(output_scale < 1e-8, 1.0, output_scale)

        self.state_mean.copy_(torch.tensor(state_mean, dtype=torch.float32))
        self.state_scale.copy_(torch.tensor(state_scale, dtype=torch.float32))
        self.action_mean.copy_(torch.tensor(action_mean, dtype=torch.float32))
        self.action_scale.copy_(torch.tensor(action_scale, dtype=torch.float32))
        self.output_mean.copy_(torch.tensor(output_mean, dtype=torch.float32))
        self.output_scale.copy_(torch.tensor(output_scale, dtype=torch.float32))

    def forward(self, state, action, dt):
        """
        Takes raw physical inputs, normalizes them, passes them through the network,
        and returns both physical (un-normalized) and normalized outputs.
        """
        # 1. Normalize Inputs
        state_norm = (state - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale

        # 2. Forward Pass
        # IMPORTANT: dim=-1 ensures safe concatenation for both 2D (MLP) and 3D (LSTM) tensors
        model_input = torch.cat([state_norm, action_norm, dt], dim=-1)

        if self.arch == "LSTM":
            # LSTM expects 3D inputs [batch, sequence, features]
            if model_input.dim() == 2:
                model_input = model_input.unsqueeze(1)  # Auto-fix for single-step inferences

            lstm_out, _ = self.lstm(model_input)

            # We only care about the physics prediction at the FINAL time step of the sequence
            last_step_out = lstm_out[:, -1, :]
            out_norm = self.fc_out(last_step_out)
        else:
            out_norm = self.network(model_input)

        # 3. Un-normalize Output to Physical Space
        out_phys = out_norm * self.output_scale + self.output_mean

        # Return both: physical for integration/display, norm for loss calculation
        return out_phys, out_norm

    def step_environment(self, current_state, action, dt):
        """Simulates the next environment state step using physical units."""
        if not torch.is_tensor(dt):
            dt = torch.full((current_state.shape[0], 1), float(dt),
                            dtype=current_state.dtype, device=current_state.device)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        # Extract only the physical output for simulation
        out_phys, _ = self.forward(current_state, action, dt)

        # RK4 or External integrator should be used for rollouts,
        # but for simple 1-step logic Euler is provided:
        next_state = current_state + out_phys * dt

        return next_state