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
    in_dim = state_dim + action_dim + 1
    ARCH = (architecture or "MLP").strip().upper()

    weights_name = pth_stem if pth_stem.endswith(".pth") else f"{pth_stem}.pth"

    if ARCH == "LSTM":
        hidden_size = int(hidden_layers[0]) if hidden_layers else 128
        num_layers = int(len(hidden_layers)) if hidden_layers else 2
        SEQ_LEN = int(lstm_seq_length)
        network_init = f"""
        self.lstm = nn.LSTM(
            input_size={in_dim},
            hidden_size={hidden_size},
            num_layers={num_layers},
            batch_first=True
        )
        self.fc_out = nn.Linear({hidden_size}, {state_dim})"""
        forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
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
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def commit_to_memory(self, state, action, dt):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.dt_buffer.append(dt)

    def predict_xdot(self, state, action, dt):
        from collections import deque
        import numpy as np
        with torch.no_grad():
            if len(self.state_buffer) < self.seq_len:
                self.commit_to_memory(state, action, dt)
            s = np.array(list(self.state_buffer) if self.state_buffer else [state], dtype=np.float32)
            a = np.array(list(self.action_buffer) if self.action_buffer else [action], dtype=np.float32)
            d = np.array([[x] for x in (self.dt_buffer if self.dt_buffer else [dt])], dtype=np.float32)
            if len(s) < self.seq_len:
                pad = self.seq_len - len(s)
                s = np.concatenate([np.repeat(s[:1], pad, axis=0), s], axis=0)
                a = np.concatenate([np.repeat(a[:1], pad, axis=0), a], axis=0)
                d = np.concatenate([np.repeat(d[:1], pad, axis=0), d], axis=0)
            s_t = torch.tensor(s[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(a[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            dt_t = torch.tensor(d[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
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
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def commit_to_memory(self, state, action, dt):
        pass

    def predict_xdot(self, state, action, dt):
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
            dt_t = torch.tensor([[dt]], dtype=torch.float32).to(self.device)
            out_phys = self.model(s_t, a_t, dt_t)
            return out_phys.squeeze(0).cpu().numpy()
"""

    script_content = f'''"""
STANDALONE NEURAL DYNAMICS CONTROLLER ({ARCH})
Generated by AgentSysID – place next to the .pth weights file.
"""
import torch
import torch.nn as nn
import numpy as np


class DeployableDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state_mean", torch.tensor({state_mean}, dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor({state_scale}, dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor({action_mean}, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor({action_scale}, dtype=torch.float32))
        self.register_buffer("output_mean", torch.tensor({output_mean}, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor({output_scale}, dtype=torch.float32))
{network_init}

    def forward(self, state, action, dt):
        state_norm = (state - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale
{forward_pass}
        out_phys = (out_norm * self.output_scale) + self.output_mean
        return out_phys

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
current_state = np.zeros(plant_model.model.state_mean.shape)
proposed_action = np.zeros(plant_model.model.action_mean.shape)

{integration_block}
print("Predicted Next State:", next_state_prediction)
'''
    path = deploy_dir / "NN.py"
    path.write_text(content, encoding="utf-8")
    return path
