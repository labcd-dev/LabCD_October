"""
Core training for DynamicsModel – the full legacy loop.

Features preserved from ``_legacy/framework.train_dynamics_model``:

  * universal autoregressive rollout (``ROLLOUT_HORIZON``) for both MLP and
    LSTM, with a sliding memory window of ``LSTM_SEQ_LENGTH`` for LSTM
  * StandardScaler normalisation baked into the model's buffers
  * optional localized state-space filtering of training windows
  * dual loss bookkeeping: normalised loss for gradients, physical MSE/RMSE
    for reporting
  * optional PINN physics residual (``PINN_LOSS_WEIGHT``)
  * AdamW + ReduceLROnPlateau, gradient clipping
  * early stopping on the normalised validation loss with best-weight restore
  * adaptive epoch extension while the model is still improving
  * live overfit abort (``OVERFIT_RATIO_LIMIT``)
  * cooperative graceful stop (``utils.stop_control``)
  * CPU-resident datasets with per-batch transfer, to keep VRAM low
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.model.dynamics_model import DynamicsModel
from backend_core.AgentSysID.model.pinn import load_analytical_xdot
from backend_core.AgentSysID.utils.device import DEVICE
from backend_core.AgentSysID.utils.stop_control import stop_requested

Trajectory = Dict[str, np.ndarray]


def compute_rmse(mse_value: float) -> float:
    """RMSE in the same physical units as the state derivatives."""
    import math

    return math.sqrt(max(float(mse_value), 0.0))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def _build_windows(
    trajs: Sequence[Trajectory],
    total_len: int,
    state_dim: int,
    action_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice every trajectory into overlapping windows of ``total_len`` steps.

    Returns arrays shaped:
      states  (N, total_len, state_dim)
      actions (N, total_len, action_dim)
      dts     (N, total_len, 1)
      xdots   (N, total_len, state_dim)
    """
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    dts: List[np.ndarray] = []
    xdots: List[np.ndarray] = []

    for traj in trajs:
        s = np.asarray(traj["states"], dtype=np.float32)
        a = np.asarray(traj["actions"], dtype=np.float32)
        d = np.asarray(traj["dts"], dtype=np.float32).reshape(-1, 1)
        x = np.asarray(traj["xdots"], dtype=np.float32)

        n = len(s)
        if n < total_len:
            continue
        for i in range(n - total_len + 1):
            states.append(s[i : i + total_len])
            actions.append(a[i : i + total_len])
            dts.append(d[i : i + total_len])
            xdots.append(x[i : i + total_len])

    if not states:
        return (
            np.zeros((0, total_len, state_dim), dtype=np.float32),
            np.zeros((0, total_len, action_dim), dtype=np.float32),
            np.zeros((0, total_len, 1), dtype=np.float32),
            np.zeros((0, total_len, state_dim), dtype=np.float32),
        )

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(dts, dtype=np.float32),
        np.asarray(xdots, dtype=np.float32),
    )


def _rollout_losses(
    dyn_model: DynamicsModel,
    b_states: torch.Tensor,
    b_actions: torch.Tensor,
    b_dts: torch.Tensor,
    b_xdot: torch.Tensor,
    b_target_norm: torch.Tensor,
    arch: str,
    seq_len: int,
    horizon: int,
    criterion: nn.Module,
    physics_fn: Optional[Any] = None,
) -> Tuple[torch.Tensor, float, Any]:
    """
    Run one autoregressive rollout over ``horizon`` steps.

    Returns (mean normalised loss tensor, mean physical MSE float,
    accumulated physics loss or 0.0).
    """
    total_norm_loss: Any = 0.0
    batch_phys_loss = 0.0
    physics_loss_accum: Any = 0.0

    if arch == "LSTM":
        current_inputs = b_states[:, :seq_len, :].clone()
    else:
        current_inputs = b_states[:, 0, :].clone()

    for t in range(horizon):
        target_idx = (seq_len - 1 + t) if arch == "LSTM" else t

        if arch == "LSTM":
            curr_action = b_actions[:, t : t + seq_len, :]
            curr_dt = b_dts[:, t : t + seq_len, :]
        else:
            curr_action = b_actions[:, target_idx, :]
            curr_dt = b_dts[:, target_idx, :]

        out_phys, out_norm = dyn_model(current_inputs, curr_action, curr_dt)

        true_xdot = b_xdot[:, target_idx, :]
        true_target_norm = b_target_norm[:, target_idx, :]

        step_norm_loss = criterion(out_norm, true_target_norm)
        total_norm_loss = total_norm_loss + step_norm_loss
        batch_phys_loss += criterion(out_phys, true_xdot).detach().item()

        # --- Physics-informed residual -----------------------------------
        if physics_fn is not None:
            try:
                if arch == "LSTM":
                    phys_states = current_inputs[:, -1, :]
                    phys_actions = curr_action[:, -1, :]
                else:
                    phys_states = current_inputs
                    phys_actions = curr_action

                physics_xdot_pred = physics_fn(phys_states, phys_actions)
                norm_physics_target = (
                    physics_xdot_pred - dyn_model.output_mean
                ) / dyn_model.output_scale
                physics_loss_accum = physics_loss_accum + criterion(out_norm, norm_physics_target)
            except Exception:
                pass

        # --- Advance the rollout ------------------------------------------
        if arch == "LSTM":
            dt_val = curr_dt[:, -1, :]
            next_state_pred = current_inputs[:, -1, :] + (out_phys * dt_val)
            current_inputs = torch.cat(
                [current_inputs[:, 1:, :], next_state_pred.unsqueeze(1)], dim=1
            )
        else:
            next_state_pred = current_inputs + (out_phys * curr_dt)
            current_inputs = next_state_pred

    total_norm_loss = total_norm_loss / horizon
    batch_phys_loss = batch_phys_loss / horizon
    return total_norm_loss, batch_phys_loss, physics_loss_accum


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train_dynamics_model(
    train_trajs: Sequence[Trajectory],
    val_trajs: Sequence[Trajectory],
    state_dim: int,
    action_encoding_dim: int,
    hidden_layers: List[int],
    learning_rate: float = 1e-3,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    lr_min: float = 0.00001,
    architecture: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DynamicsModel, float, float, float, np.ndarray, np.ndarray]:
    """
    Train one candidate architecture.

    Returns
    -------
    (model, best_train_phys_mse, best_val_phys_mse, best_val_phys_rmse,
     X_val_states, y_val_xdot)
    """
    epochs = int(epochs if epochs is not None else cfg.EPOCHS)
    batch_size = int(batch_size if batch_size is not None else cfg.BATCH_SIZE)
    patience = int(patience if patience is not None else cfg.EARLY_STOP_PATIENCE)
    max_allowed_epochs = epochs

    arch = str(architecture or cfg.NETWORK_ARCHITECTURE).strip().upper()
    seq_len = int(cfg.LSTM_SEQ_LENGTH) if arch == "LSTM" else 1
    horizon = max(1, int(cfg.ROLLOUT_HORIZON))

    # Total window length extracted from the dataset
    total_len = seq_len + horizon - 1

    # --- Build training windows ------------------------------------------
    train_states, train_actions, train_dts, train_xdots = _build_windows(
        train_trajs, total_len, state_dim, action_encoding_dim
    )
    if len(train_states) == 0:
        raise ValueError(
            f"No training windows of length {total_len} could be built. "
            "The trajectories are shorter than LSTM_SEQ_LENGTH + ROLLOUT_HORIZON - 1."
        )

    flat_states = train_states.reshape(-1, state_dim)
    flat_actions = train_actions.reshape(-1, action_encoding_dim)
    flat_xdots = train_xdots.reshape(-1, state_dim)

    state_scaler = StandardScaler().fit(flat_states)
    output_scaler = StandardScaler().fit(flat_xdots)
    if action_encoding_dim > 0:
        action_scaler = StandardScaler().fit(flat_actions)
        action_mean, action_scale = action_scaler.mean_, action_scaler.scale_
    else:
        action_mean = np.zeros(0, dtype=np.float64)
        action_scale = np.ones(0, dtype=np.float64)

    # --- Optional localized state-space filtering -------------------------
    if getattr(cfg, "USE_STATE_FILTER", False):
        bounds_filter = getattr(cfg, "STATE_BOUNDS_FILTER", {}) or {}
        percentiles = getattr(cfg, "AUTO_FILTER_PERCENTILES", (2, 98))

        chosen_bounds: Dict[int, Tuple[float, float]] = {}
        if not bounds_filter:
            for dim in range(state_dim):
                p_min, p_max = np.percentile(
                    flat_states[:, dim], [percentiles[0], percentiles[1]]
                )
                if abs(p_max - p_min) < 1e-4:
                    p_min, p_max = float(np.min(flat_states[:, dim])), float(
                        np.max(flat_states[:, dim])
                    )
                chosen_bounds[dim] = (float(p_min), float(p_max))
        else:
            chosen_bounds = dict(bounds_filter)

        valid_mask = np.ones(len(train_states), dtype=bool)
        for dim, (low_b, high_b) in chosen_bounds.items():
            if dim < state_dim:
                valid_mask &= (train_states[:, 0, dim] >= low_b) & (
                    train_states[:, 0, dim] <= high_b
                )

        if valid_mask.any():
            train_states = train_states[valid_mask]
            train_actions = train_actions[valid_mask]
            train_dts = train_dts[valid_mask]
            train_xdots = train_xdots[valid_mask]

    # Keep datasets on CPU (system RAM) and move mini-batches to the device.
    X_train_states_t = torch.tensor(train_states, dtype=torch.float32)
    X_train_acts_t = torch.tensor(train_actions, dtype=torch.float32)
    X_train_dt_t = torch.tensor(train_dts, dtype=torch.float32)
    y_train_xdot_t = torch.tensor(train_xdots, dtype=torch.float32)

    val_states, val_actions, val_dts, val_xdots = _build_windows(
        val_trajs, total_len, state_dim, action_encoding_dim
    )
    X_val_states_t = torch.tensor(val_states, dtype=torch.float32)
    X_val_acts_t = torch.tensor(val_actions, dtype=torch.float32)
    X_val_dt_t = torch.tensor(val_dts, dtype=torch.float32)
    y_val_xdot_t = torch.tensor(val_xdots, dtype=torch.float32)

    # --- Model ------------------------------------------------------------
    dyn_model = DynamicsModel(
        state_dim,
        action_encoding_dim,
        hidden_layers,
        activation,
        dropout_rate=dropout_rate,
        architecture=arch,
    ).to(DEVICE)
    dyn_model.set_scalers(
        state_mean=state_scaler.mean_,
        state_scale=state_scaler.scale_,
        action_mean=action_mean,
        action_scale=action_scale,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
    )

    optimizer = optim.AdamW(dyn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(getattr(cfg, "LR_REDUCE_FACTOR", 0.5)),
        patience=max(1, int(patience / 2)),
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=lr_min,
    )
    criterion = nn.MSELoss()

    # --- PINN hook --------------------------------------------------------
    physics_fn = None
    if getattr(cfg, "USE_PINN", False):
        physics_fn = load_analytical_xdot(getattr(cfg, "PINN_EQUATION_FILE", "physics_env.py"))
        if physics_fn is None and verbose:
            print("      ⚠️ USE_PINN is on but no compute_analytical_xdot was found; training data-driven.")
    pinn_weight = float(getattr(cfg, "PINN_LOSS_WEIGHT", 0.5))

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_train_loss = float("inf")
    best_val_norm_loss = float("inf")
    best_val_phys_loss = float("inf")
    best_val_phys_rmse = float("inf")
    epochs_no_improve = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    # Pre-calculate normalised targets safely on the CPU
    norm_xdot_train_target = (
        y_train_xdot_t - dyn_model.output_mean.cpu()
    ) / dyn_model.output_scale.cpu()
    norm_xdot_val_target = (
        y_val_xdot_t - dyn_model.output_mean.cpu()
    ) / dyn_model.output_scale.cpu()

    improvement_threshold = float(getattr(cfg, "ADAPTIVE_IMPROVEMENT_THRESHOLD", 0.005))
    extension_steps = int(getattr(cfg, "EPOCH_EXTENSION_STEPS", 50))
    overfit_limit = float(getattr(cfg, "OVERFIT_RATIO_LIMIT", 10.0))

    epoch = 0
    while epoch < max_allowed_epochs:
        if stop_requested():
            break

        current_epoch_lr = optimizer.param_groups[0]["lr"]
        dyn_model.train()

        # Permutation on CPU to match the CPU-resident dataset
        perm = torch.randperm(X_train_states_t.size(0))

        epoch_norm_loss = 0.0
        epoch_phys_loss = 0.0
        num_batches = 0
        stopped_mid_epoch = False

        for i in range(0, X_train_states_t.size(0), batch_size):
            if stop_requested():
                stopped_mid_epoch = True
                break
            idx = perm[i : i + batch_size]

            # Move only the current mini-batch to the device
            b_states = X_train_states_t[idx].to(DEVICE)
            b_actions = X_train_acts_t[idx].to(DEVICE)
            b_dts = X_train_dt_t[idx].to(DEVICE)
            b_xdot = y_train_xdot_t[idx].to(DEVICE)
            b_target_norm = norm_xdot_train_target[idx].to(DEVICE)

            optimizer.zero_grad()

            total_norm_loss, batch_phys_loss, physics_loss_accum = _rollout_losses(
                dyn_model,
                b_states,
                b_actions,
                b_dts,
                b_xdot,
                b_target_norm,
                arch,
                seq_len,
                horizon,
                criterion,
                physics_fn,
            )

            if physics_fn is not None and torch.is_tensor(physics_loss_accum):
                physics_loss_accum = physics_loss_accum / horizon
                total_norm_loss = total_norm_loss + (pinn_weight * physics_loss_accum)

            total_norm_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dyn_model.parameters() if p.grad is not None], 1.0
            )
            optimizer.step()

            epoch_norm_loss += total_norm_loss.item()
            epoch_phys_loss += batch_phys_loss
            num_batches += 1

        if stopped_mid_epoch:
            break

        current_train_norm_loss = epoch_norm_loss / max(1, num_batches)
        current_train_phys_loss = epoch_phys_loss / max(1, num_batches)

        # --- VALIDATION LOOP ---------------------------------------------
        dyn_model.eval()
        val_norm_loss_sum = 0.0
        val_phys_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, X_val_states_t.size(0), batch_size):
                b_states = X_val_states_t[i : i + batch_size].to(DEVICE)
                b_actions = X_val_acts_t[i : i + batch_size].to(DEVICE)
                b_dts = X_val_dt_t[i : i + batch_size].to(DEVICE)
                b_xdot = y_val_xdot_t[i : i + batch_size].to(DEVICE)
                b_target_norm = norm_xdot_val_target[i : i + batch_size].to(DEVICE)

                batch_norm_loss, batch_phys_loss, _ = _rollout_losses(
                    dyn_model,
                    b_states,
                    b_actions,
                    b_dts,
                    b_xdot,
                    b_target_norm,
                    arch,
                    seq_len,
                    horizon,
                    criterion,
                    None,
                )

                val_norm_loss_sum += float(batch_norm_loss.item())
                val_phys_loss_sum += batch_phys_loss
                val_batches += 1

        current_val_norm_loss = val_norm_loss_sum / max(1, val_batches)
        current_val_phys_loss = val_phys_loss_sum / max(1, val_batches)

        train_losses.append(current_train_phys_loss)
        val_losses.append(current_val_phys_loss)

        current_train_rmse = compute_rmse(current_train_phys_loss)
        current_val_rmse = compute_rmse(current_val_phys_loss)

        scheduler.step(current_val_norm_loss)

        if verbose and (epoch % 50 == 0 or epoch == max_allowed_epochs - 1):
            print(
                f"      🔄 Epoch {epoch:4d}/{max_allowed_epochs} | "
                f"Train MSE: {current_train_phys_loss:.6f} (RMSE: {current_train_rmse:.6f}) | "
                f"Val MSE: {current_val_phys_loss:.6f} (RMSE: {current_val_rmse:.6f}) | "
                f"LR: {current_epoch_lr:.6f}"
            )

        # --- Adaptive epoch extension -------------------------------------
        if epoch == max_allowed_epochs - 1:
            triggered_extension = False
            if best_train_loss != float("inf"):
                train_improvement = (best_train_loss - current_train_norm_loss) / best_train_loss
                if train_improvement > improvement_threshold:
                    max_allowed_epochs += extension_steps
                    triggered_extension = True

            if best_val_norm_loss != float("inf"):
                val_improvement = (best_val_norm_loss - current_val_norm_loss) / best_val_norm_loss
                if val_improvement > improvement_threshold:
                    if not triggered_extension:
                        max_allowed_epochs += extension_steps
                    triggered_extension = True

            if triggered_extension and verbose:
                print(f"    ✨ Progress active! Extending runway to {max_allowed_epochs} epochs.")

        if current_train_norm_loss < best_train_loss:
            best_train_loss = current_train_norm_loss

        # --- Early stopping on the normalised validation loss --------------
        if current_val_norm_loss < best_val_norm_loss:
            best_val_norm_loss = current_val_norm_loss
            best_val_phys_loss = current_val_phys_loss
            best_val_phys_rmse = current_val_rmse
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in dyn_model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if epoch >= 50:
                    if verbose:
                        print(f"      Early stopping triggered at epoch {epoch}")
                    break

        # --- Live overfit abort -------------------------------------------
        if epoch >= 50 and current_train_phys_loss > 1e-8:
            live_overfit_ratio = current_val_phys_loss / current_train_phys_loss
            if live_overfit_ratio > overfit_limit:
                if verbose:
                    print(
                        f"      ⚠️ Overfit Limit Exceeded at epoch {epoch} "
                        f"({live_overfit_ratio:.1f}x limit). Aborting early to save time!"
                    )
                break

        epoch += 1

    if best_model_state is not None:
        dyn_model.load_state_dict(best_model_state)

    final_train_mse = min(train_losses) if train_losses else float("inf")
    if best_val_phys_loss == float("inf") and val_losses:
        best_val_phys_loss = min(val_losses)
        best_val_phys_rmse = compute_rmse(best_val_phys_loss)

    return (
        dyn_model,
        final_train_mse,
        best_val_phys_loss,
        best_val_phys_rmse,
        X_val_states_t.cpu().numpy(),
        y_val_xdot_t.cpu().numpy(),
    )


def measure_inference_latency(
    model: DynamicsModel,
    state_dim: int,
    action_dim: int,
    n_runs: int = 100,
) -> float:
    """Simulate real-time control deployment to measure average latency (ms)."""
    model.eval()

    device = next(model.parameters()).device
    dummy_state = torch.randn(1, state_dim).to(device)
    dummy_action = torch.randn(1, action_dim).to(device)
    dummy_dt = torch.ones(1, 1).to(device) * 0.01

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_state, dummy_action, dummy_dt)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_state, dummy_action, dummy_dt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    return ((end_time - start_time) / float(n_runs)) * 1000.0
