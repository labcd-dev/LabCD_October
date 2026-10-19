"""
Standalone deployable controller export (from _legacy export_standalone_inference_script).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn


def export_standalone_inference_script(
    model: nn.Module,
    hidden_layers: List[int],
    activation: str,
    state_dim: int,
    action_dim: int,
    pth_stem: str,
    env_name: str,
    output_dir: Union[str, Path],
    architecture: str = "MLP",
    lstm_seq_length: int = 10,
    physics_mode: str = "pinn",
    hard_constraint_indices: Optional[List[int]] = None,
    physics_equation_file: Optional[str | Path] = None,
) -> Path:
    """
    Write deployed_controller_<env>.py into output_dir/deployment/.
    ``pth_stem`` is the weights filename without path (e.g. best_model_foo.pth or without .pth).
    """
    output_dir = Path(output_dir)
    deploy_dir = output_dir / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    state_mean = model.state_mean.cpu().numpy().flatten().tolist()
    state_scale = model.state_scale.cpu().numpy().flatten().tolist()
    action_mean = model.action_mean.cpu().numpy().flatten().tolist()
    action_scale = model.action_scale.cpu().numpy().flatten().tolist()
    output_mean = model.output_mean.cpu().numpy().flatten().tolist()
    output_scale = model.output_scale.cpu().numpy().flatten().tolist()
    angle_indices = list(getattr(model, "angle_indices", ()) or ())
    in_dim = state_dim + len(angle_indices) + action_dim + 1
    ARCH = (architecture or "MLP").strip().upper()
    PHYSICS_MODE = str(physics_mode or "pinn").strip().lower()
    HARD_CONSTRAINTS = sorted({int(index) for index in (hard_constraint_indices or [])
                               if 0 <= int(index) < int(state_dim)})
    PHYSICS_PARAMETER_ESTIMATES = dict(getattr(model, "physics_parameter_estimates", {}) or {})
    physics_source = ""
    if PHYSICS_MODE == "pcnn":
        equation_path = Path(physics_equation_file or "")
        if not equation_path.is_file():
            raise ValueError("PCNN export needs the validated physics equation saved with this run.")
        physics_source = equation_path.read_text(encoding="utf-8")
        if "def compute_analytical_xdot" not in physics_source:
            raise ValueError("The saved PCNN equation does not define compute_analytical_xdot.")

    weights_name = pth_stem if pth_stem.endswith(".pth") else f"{pth_stem}.pth"

    if ARCH == "LSTM":
        widths = [int(width) for width in hidden_layers] or [128, 128]
        SEQ_LEN = int(lstm_seq_length)
        if len(set(widths)) == 1:
            # Preserve the original key names for previously trained models.
            hidden_size = widths[0]
            network_init = f"""
        self.lstm = nn.LSTM(
            input_size={in_dim},
            hidden_size={hidden_size},
            num_layers={len(widths)},
            batch_first=True
        )
        self.fc_out = nn.Linear({widths[-1]}, {state_dim})"""
            forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out_norm = self.fc_out(lstm_out[:, -1, :])"""
        else:
            recurrent_modules = []
            layer_input_dim = in_dim
            for index, width in enumerate(widths):
                recurrent_modules.append(
                    f"            nn.LSTM(input_size={layer_input_dim}, "
                    f"hidden_size={width}, num_layers=1, batch_first=True)"
                )
                layer_input_dim = width
            module_lines = ",\n".join(recurrent_modules)
            network_init = f"""
        self.lstm_layers = nn.ModuleList([
{module_lines}
        ])
        self.fc_out = nn.Linear({widths[-1]}, {state_dim})"""
            forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out = x
        for layer in self.lstm_layers:
            lstm_out, _ = layer(lstm_out)
        out_norm = self.fc_out(lstm_out[:, -1, :])"""
        controller_class = f"""
class NeuralController:
    def __init__(self, weights_path="{weights_name}", device="cpu"):
        from collections import deque
        self.device = torch.device(device)
        self.seq_len = {SEQ_LEN}
        self.state_buffer = deque(maxlen=self.seq_len)
        self.action_buffer = deque(maxlen=self.seq_len)
        self.dt_buffer = deque(maxlen=self.seq_len)
        self.model = DeployableDynamicsModel().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        state = {{key: value for key, value in state.items() if not key.startswith("physics_fn.")}}
        self.model.load_state_dict(state)
        self.model.eval()

    def commit_to_memory(self, state, action, dt):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.dt_buffer.append(dt)

    def wrap_state(self, state):
        # Project periodic angles to [-pi, pi) after integrating the state.
        wrapped = np.asarray(state, dtype=float).copy()
        for index in self.model.angle_indices:
            wrapped[..., index] = (wrapped[..., index] + np.pi) % (2.0 * np.pi) - np.pi
        return wrapped

    def predict_xdot(self, state, action, dt):
        temp_s = list(self.state_buffer) + [state]
        temp_a = list(self.action_buffer) + [action]
        temp_dt = list(self.dt_buffer) + [dt]

        # Pad the buffer if the simulation just started
        while len(temp_s) < self.seq_len:
            temp_s.insert(0, temp_s[0])
            temp_a.insert(0, temp_a[0])
            temp_dt.insert(0, temp_dt[0])

        with torch.no_grad():
            s_t = torch.tensor(np.array(temp_s[-self.seq_len:]), dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(np.array(temp_a[-self.seq_len:]), dtype=torch.float32).unsqueeze(0).to(self.device)
            dt_t = torch.tensor(np.array(temp_dt[-self.seq_len:]), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(self.device)
            out_phys = self.model(s_t, a_t, dt_t)
            return out_phys.squeeze(0).cpu().numpy()
"""
    else:
        act_map = {
            "relu": "nn.ReLU",
            "elu": "nn.ELU",
            "leaky_relu": "nn.LeakyReLU",
            "swish": "nn.SiLU",
            "silu": "nn.SiLU",
            "tanh": "nn.Tanh",
        }
        act_cls = act_map.get((activation or "relu").lower(), "nn.ReLU")
        layer_lines = []
        curr = in_dim
        for h in hidden_layers:
            layer_lines.append(f"            nn.Linear({curr}, {int(h)}),")
            layer_lines.append(f"            {act_cls}(),")
            curr = int(h)
        layer_lines.append(f"            nn.Linear({curr}, {state_dim}),")
        layers_body = "\n".join(layer_lines)
        network_init = f"""
        self.network = nn.Sequential(
{layers_body}
        )"""
        forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        out_norm = self.network(x)"""
        controller_class = f"""
class NeuralController:
    def __init__(self, weights_path="{weights_name}", device="cpu"):
        self.device = torch.device(device)
        self.model = DeployableDynamicsModel().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        state = {{key: value for key, value in state.items() if not key.startswith("physics_fn.")}}
        self.model.load_state_dict(state)
        self.model.eval()

    def commit_to_memory(self, state, action, dt):
        pass

    def wrap_state(self, state):
        # Project periodic angles to [-pi, pi) after integrating the state.
        wrapped = np.asarray(state, dtype=float).copy()
        for index in self.model.angle_indices:
            wrapped[..., index] = (wrapped[..., index] + np.pi) % (2.0 * np.pi) - np.pi
        return wrapped

    def predict_xdot(self, state, action, dt):
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
            dt_t = torch.tensor([[dt]], dtype=torch.float32).to(self.device)
            out_phys = self.model(s_t, a_t, dt_t)
            return out_phys.squeeze(0).cpu().numpy()
"""

    physics_call = ("compute_analytical_xdot(physics_states, physics_actions)"
                    if not PHYSICS_PARAMETER_ESTIMATES else
                    f"compute_analytical_xdot(physics_states, physics_actions, parameter_values={PHYSICS_PARAMETER_ESTIMATES!r})")
    pcnn_forward = f"""
        physics_states = raw_state[:, -1, :] if raw_state.dim() == 3 else raw_state
        physics_actions = action[:, -1, :] if action.dim() == 3 else action
        physics_xdot = {physics_call}
        out_phys = (out_norm * self.output_scale) + self.output_mean
        if self.hard_constraint_indices:
            hard_mask = torch.zeros(self.state_dim, device=state.device, dtype=torch.bool)
            hard_mask[self.hard_constraint_indices] = True
            out_phys = torch.where(hard_mask.unsqueeze(0), physics_xdot, out_phys)
        return out_phys""" if PHYSICS_MODE == "pcnn" else """
        out_phys = (out_norm * self.output_scale) + self.output_mean
        return out_phys"""

    script_content = f'''"""
STANDALONE NEURAL DYNAMICS CONTROLLER ({ARCH})
Generated by AgentSysID – place next to the .pth weights file.
"""
import torch
import torch.nn as nn
import numpy as np

{physics_source}


class DeployableDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = {int(state_dim)}
        self.angle_indices = {angle_indices!r}
        self.physics_mode = {PHYSICS_MODE!r}
        self.hard_constraint_indices = {HARD_CONSTRAINTS!r}
        self.register_buffer("state_mean", torch.tensor({state_mean}, dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor({state_scale}, dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor({action_mean}, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor({action_scale}, dtype=torch.float32))
        self.register_buffer("output_mean", torch.tensor({output_mean}, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor({output_scale}, dtype=torch.float32))
{network_init}

    def forward(self, state, action, dt):
        raw_state = state
        if self.angle_indices:
            state_features = []
            for index in range(self.state_dim):
                value = state[..., index:index + 1]
                if index in self.angle_indices:
                    state_features.extend((torch.sin(value), torch.cos(value)))
                else:
                    state_features.append(value)
            state = torch.cat(state_features, dim=-1)
        state_norm = (state - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale
{forward_pass}
{pcnn_forward}

{controller_class}
'''
    out_name = deploy_dir / f"deployed_controller_{env_name}.py"
    out_name.write_text(script_content, encoding="utf-8")
    print(f"    💾 Deployable controller → {out_name}")
    return out_name


def write_nn_inference_helper(
    env_name: str,
    output_dir: Union[str, Path],
    integrator: str = "RK4",
) -> Path:
    """Write deployment/NN.py usage snippet."""
    deploy_dir = Path(output_dir) / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    INT_TYPE = (integrator or "EULER").strip().upper()
    if INT_TYPE == "RK4":
        integration_block = """# 3. KINEMATIC INTEGRATION (4th-Order Runge-Kutta)
k1 = plant_model.predict_xdot(current_state, proposed_action, dt)
k2 = plant_model.predict_xdot(current_state + 0.5 * dt * k1, proposed_action, dt)
k3 = plant_model.predict_xdot(current_state + 0.5 * dt * k2, proposed_action, dt)
k4 = plant_model.predict_xdot(current_state + dt * k3, proposed_action, dt)
predicted_xdot = (k1 + 2*k2 + 2*k3 + k4) / 6.0
next_state_prediction = current_state + (predicted_xdot * dt)
try:
    plant_model.commit_to_memory(current_state, proposed_action, dt)
except AttributeError:
    pass
"""
    else:
        integration_block = """# 3. KINEMATIC INTEGRATION (1st-Order Euler)
predicted_xdot = plant_model.predict_xdot(current_state, proposed_action, dt)
next_state_prediction = current_state + (predicted_xdot * dt)
try:
    plant_model.commit_to_memory(current_state, proposed_action, dt)
except AttributeError:
    pass
"""
    content = f'''# ==============================================================================
# NEURAL NETWORK INFERENCE DEPLOYMENT SCRIPT
# ==============================================================================
import numpy as np
from deployed_controller_{env_name} import NeuralController

# 1. INITIALIZATION
plant_model = NeuralController(device="cpu")  # or "cuda"

# 2. CONTROL LOOP
dt = 0.01
current_state = np.zeros(plant_model.model.state_dim)
proposed_action = np.zeros(plant_model.model.action_mean.shape)

{integration_block}
next_state_prediction = plant_model.wrap_state(next_state_prediction)
print("Predicted Next State:", next_state_prediction)
'''
    path = deploy_dir / "NN.py"
    path.write_text(content, encoding="utf-8")
    return path


def save_best_model(
    dyn_model: nn.Module,
    hidden_layers: List[int],
    state_dim: int,
    action_encoding_dim: int,
    mse: float,
    config: dict,
    activation: str,
    env_name: str,
    timestamp: str,
    output_dir: Union[str, Path],
    rollout_horizon: int = 1,
) -> Path:
    """
    Persist the winning weights under a self-describing filename.

    The name encodes the environment, run timestamp, loss mode, prediction
    target and achieved MSE, e.g.::

        best_model_<env>_<timestamp>_SingleStep_Xdot_MSE_0.001234.pth

    Returns the written path (the stem is what the deployable controller
    references as its weights file).
    """
    deploy_dir = Path(output_dir) / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"Rollout{rollout_horizon}" if rollout_horizon > 1 else "SingleStep"
    target_label = "Xdot"
    base = f"best_model_{env_name}_{timestamp}_{mode_label}_{target_label}_MSE_{mse:.6f}"
    path = deploy_dir / f"{base}.pth"

    torch.save(dyn_model.state_dict(), path)
    print(f"    💾 Weights → {path.name}")
    return path
