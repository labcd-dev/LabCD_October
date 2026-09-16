"""
Core training utilities for DynamicsModel.

A simplified but functional trainer that preserves the public contract
expected by the orchestration loop. Full original logic lives in _legacy/framework.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.model.dynamics_model import DynamicsModel
from backend_core.AgentSysID.utils.device import DEVICE


def compute_rmse(mse_value: float) -> float:
    return float(np.sqrt(max(mse_value, 0.0)))


def _trajs_to_tensors(
    trajs: List[Dict[str, np.ndarray]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack trajectories into flat tensors for supervised x_dot regression."""
    states, actions, xdots, dts = [], [], [], []
    for t in trajs:
        states.append(t["states"])
        actions.append(t["actions"])
        xdots.append(t["xdots"])
        dts.append(t["dts"].reshape(-1, 1))
    s = torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32)
    a = torch.tensor(np.concatenate(actions, axis=0), dtype=torch.float32)
    x = torch.tensor(np.concatenate(xdots, axis=0), dtype=torch.float32)
    d = torch.tensor(np.concatenate(dts, axis=0), dtype=torch.float32)
    return s, a, x, d


def train_dynamics_model(
    train_trajs: List[Dict[str, np.ndarray]],
    val_trajs: List[Dict[str, np.ndarray]],
    state_dim: int,
    action_dim: int,
    hidden_layers: List[int],
    learning_rate: float = 1e-3,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    architecture: Optional[str] = None,
) -> Tuple[DynamicsModel, float, float]:
    """
    Train a DynamicsModel and return (model, train_mse, val_mse).
    """
    epochs = epochs or cfg.EPOCHS
    batch_size = batch_size or cfg.BATCH_SIZE
    architecture = architecture or cfg.NETWORK_ARCHITECTURE

    model = DynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        architecture=architecture,
    ).to(DEVICE)

    s_tr, a_tr, x_tr, d_tr = _trajs_to_tensors(train_trajs)
    s_va, a_va, x_va, d_va = _trajs_to_tensors(val_trajs)

    # Simple standardisation
    s_mean, s_std = s_tr.mean(0), s_tr.std(0).clamp(min=1e-6)
    a_mean, a_std = a_tr.mean(0), a_tr.std(0).clamp(min=1e-6)
    x_mean, x_std = x_tr.mean(0), x_tr.std(0).clamp(min=1e-6)
    model.set_scalers(
        s_mean.numpy(), s_std.numpy(),
        a_mean.numpy(), a_std.numpy(),
        x_mean.numpy(), x_std.numpy(),
    )

    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(s_tr, a_tr, x_tr, d_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for sb, ab, xb, db in loader:
            sb, ab, xb, db = sb.to(DEVICE), ab.to(DEVICE), xb.to(DEVICE), db.to(DEVICE)
            opt.zero_grad()
            pred_phys, pred_norm = model(sb, ab, db)
            # Loss in normalised space
            target_norm = (xb - model.output_mean) / model.output_scale
            loss = loss_fn(pred_norm, target_norm)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * sb.size(0)
            n += sb.size(0)
        # Lightweight early progress print every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"      epoch {epoch+1}/{epochs}  train_loss={epoch_loss/max(n,1):.6e}")

    # Final metrics
    model.eval()
    with torch.no_grad():
        def _mse(s, a, x, d):
            s, a, x, d = s.to(DEVICE), a.to(DEVICE), x.to(DEVICE), d.to(DEVICE)
            _, pred_norm = model(s, a, d)
            target_norm = (x - model.output_mean) / model.output_scale
            return float(loss_fn(pred_norm, target_norm).item())

        train_mse = _mse(s_tr, a_tr, x_tr, d_tr)
        val_mse = _mse(s_va, a_va, x_va, d_va)

    return model, train_mse, val_mse


def measure_inference_latency(
    model: DynamicsModel,
    state_dim: int,
    action_dim: int,
    n_runs: int = 50,
) -> float:
    """Return average inference latency in milliseconds (single-sample)."""
    model.eval()
    s = torch.randn(1, state_dim, device=DEVICE)
    a = torch.randn(1, action_dim, device=DEVICE)
    dt = torch.ones(1, 1, device=DEVICE) * 0.01

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(s, a, dt)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(s, a, dt)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return ((t1 - t0) / n_runs) * 1000.0  # ms
