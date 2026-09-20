"""
Diagnostic plots for AgentSysID runs.

Adapted from _legacy/framework.py plot_* helpers. Writes PNGs into
``output_dir/figures/`` (or ``output_dir`` if figures_subdir is False).

performance_history entries (current CLI format):
  {"cycle": int, "val_mse": float, "train_mse": float, "latency": float, "config": dict}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive; safe for CLI / servers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def _figures_dir(output_dir: str | Path) -> Path:
    d = Path(output_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cycles(history: List[Dict[str, Any]]) -> List[int]:
    out = []
    for i, p in enumerate(history):
        c = p.get("cycle", p.get("iteration"))
        out.append(int(c) if c is not None else i + 1)
    return out


def _val_mse(p: Dict[str, Any]) -> float:
    if "val_mse" in p:
        return float(p["val_mse"])
    perf = p.get("performance") or {}
    return float(perf.get("mse", perf.get("val_mse", float("nan"))))


def _rmse(p: Dict[str, Any]) -> float:
    if "rmse" in p and p["rmse"] is not None:
        return float(p["rmse"])
    mse = _val_mse(p)
    return float(np.sqrt(max(mse, 0.0)))


def _latency(p: Dict[str, Any]) -> float:
    if "latency" in p:
        return float(p["latency"])
    perf = p.get("performance") or {}
    return float(perf.get("latency", 0.0))


def _config(p: Dict[str, Any]) -> Dict[str, Any]:
    return p.get("config") or {}


def plot_mse_convergence(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
) -> Optional[Path]:
    if not performance_history:
        return None
    iters = _cycles(performance_history)
    mses = [_val_mse(p) for p in performance_history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, mses, "r-o", linewidth=2, markersize=8)
    ax.set_xlabel("Actor-Critic Cycle")
    ax.set_ylabel("Validation Single-Step MSE")
    ax.set_title(f"Single-Step System Identification Error for {env_name}")
    ax.grid(True)
    ax.set_yscale("log")
    out = _figures_dir(output_dir) / f"{prefix}_{env_name}_Xdot_mse_convergence_{timestamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Saved MSE Convergence Plot → {out.name}")
    return out


def plot_nrmse_convergence(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
) -> Optional[Path]:
    if not performance_history:
        return None
    iters = _cycles(performance_history)
    rmses = [_rmse(p) for p in performance_history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, rmses, "r-o", linewidth=2, markersize=8)
    ax.set_xlabel("Actor-Critic Cycle")
    ax.set_ylabel("Validation Single-Step RMSE")
    ax.set_title(f"Single-Step System Identification Error (RMSE) for {env_name}")
    ax.grid(True)
    ax.set_yscale("log")
    out = _figures_dir(output_dir) / f"{prefix}_{env_name}_Xdot_rmse_convergence_{timestamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Saved RMSE Convergence Plot → {out.name}")
    return out


def plot_hyperparameter_evolution(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
) -> Optional[Path]:
    if not performance_history:
        return None
    cycles = _cycles(performance_history)
    lrs = [float(_config(p).get("learning_rate", 0)) for p in performance_history]
    total_neurons = [sum(_config(p).get("hidden_layers") or [0]) for p in performance_history]
    num_layers = [len(_config(p).get("hidden_layers") or []) for p in performance_history]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(cycles, lrs, "b-o")
    ax1.set_yscale("log")
    ax1.set_ylabel("Learning Rate")
    ax1.grid(True)
    ax2.plot(cycles, total_neurons, "g-s")
    ax2.set_ylabel("Total Neurons")
    ax2.grid(True)
    ax3.plot(cycles, num_layers, "m-d")
    ax3.set_ylabel("Number of Hidden Layers")
    ax3.set_xlabel("Cycle")
    ax3.grid(True)
    fig.tight_layout()
    out = _figures_dir(output_dir) / f"{prefix}_{env_name}_Xdot_hyperparameters_{timestamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Saved Hyperparameter Evolution Plot → {out.name}")
    return out


def plot_latency_evolution(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    max_latency: float,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
) -> Optional[Path]:
    if not performance_history:
        return None
    cycles = _cycles(performance_history)
    latencies = [_latency(p) for p in performance_history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles, latencies, "c-o", linewidth=2, markersize=8, label="Measured Latency")
    ax.axhline(y=max_latency, color="red", linestyle="--", linewidth=2, label="Max Allowed Limit")
    ax.set_xlabel("Actor-Critic Cycle", fontweight="bold")
    ax.set_ylabel("Inference Latency (ms)", fontweight="bold")
    ax.set_title(f"Network Inference Latency Evolution for {env_name}", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _figures_dir(output_dir) / f"{prefix}_{env_name}_latency_evolution_{timestamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Saved Latency Evolution Plot → {out.name}")
    return out


def plot_layers_neurons_mse_contour(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
) -> Optional[Path]:
    if not performance_history:
        return None
    num_layers = np.array([len(_config(p).get("hidden_layers") or []) for p in performance_history], dtype=float)
    avg_neurons = np.array(
        [
            (sum(_config(p).get("hidden_layers") or [0]) / max(len(_config(p).get("hidden_layers") or [1]), 1))
            for p in performance_history
        ],
        dtype=float,
    )
    mse = np.array([_val_mse(p) for p in performance_history], dtype=float)
    mse = np.where(np.isinf(mse) | np.isnan(mse), 9999.0, mse)
    mse = np.clip(mse, a_min=1e-8, a_max=None)
    vmin_val = float(np.min(mse))
    vmax_val = float(np.max(mse))
    if vmin_val >= vmax_val:
        vmax_val = vmin_val * 10.0

    fig, ax = plt.subplots(figsize=(9, 7))
    contour_ok = False
    if len(performance_history) >= 3 and np.ptp(num_layers) > 0 and np.ptp(avg_neurons) > 0:
        try:
            norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
            contour = ax.tricontourf(num_layers, avg_neurons, mse, levels=14, cmap="viridis", norm=norm)
            fig.colorbar(contour, ax=ax, label="Validation MSE")
            contour_ok = True
        except Exception as e:
            print(f"    ⚠️ Contour triangulation failed ({e}); scatter only.")

    scatter_norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
    scatter = ax.scatter(
        num_layers, avg_neurons, c=mse, cmap="viridis", norm=scatter_norm,
        s=90, edgecolors="white", linewidths=1.2, zorder=3,
    )
    if not contour_ok:
        fig.colorbar(scatter, ax=ax, label="Validation MSE")

    best_idx = int(np.argmin(mse))
    ax.scatter(
        num_layers[best_idx], avg_neurons[best_idx], marker="*", s=500,
        c="red", edgecolors="black", linewidths=1.2, zorder=4, label="Best Config",
    )
    ax.set_xlabel("Number of Hidden Layers", fontweight="bold")
    ax.set_ylabel("Avg. Neurons per Layer", fontweight="bold")
    ax.set_title(f"Validation MSE over Architecture Search Space (Xdot) for {env_name}", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _figures_dir(output_dir) / f"{prefix}_{env_name}_Xdot_layers_neurons_contour_{timestamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Saved Layers/Neurons MSE Contour Plot → {out.name}")
    return out


def generate_all_plots(
    performance_history: List[Dict[str, Any]],
    env_name: str,
    output_dir: str | Path,
    timestamp: str,
    max_latency: float = 2.0,
    prefix: str = "system_id",
) -> List[Path]:
    """Run the standard diagnostic suite; return list of written PNG paths."""
    paths: List[Path] = []
    for fn in (
        lambda: plot_mse_convergence(performance_history, env_name, output_dir, timestamp, prefix),
        lambda: plot_nrmse_convergence(performance_history, env_name, output_dir, timestamp, prefix),
        lambda: plot_hyperparameter_evolution(performance_history, env_name, output_dir, timestamp, prefix),
        lambda: plot_latency_evolution(performance_history, env_name, max_latency, output_dir, timestamp, prefix),
        lambda: plot_layers_neurons_mse_contour(performance_history, env_name, output_dir, timestamp, prefix),
    ):
        try:
            p = fn()
            if p is not None:
                paths.append(p)
        except Exception as e:
            print(f"    ⚠️ Plot generation skipped: {e}")
    return paths


# ---------------------------------------------------------------------------
# Pure data-driven held-out verification (no physics equations)
# ---------------------------------------------------------------------------
def plot_test_dataset_verification(
    dyn_model,
    test_trajs: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    output_dir: str | Path,
    timestamp: str,
    prefix: str = "system_id",
    env_name: str = "system",
    architecture: Optional[str] = None,
    lstm_seq_length: Optional[int] = None,
    integrator: Optional[str] = None,
    target_horizon_seconds: float = 10.0,
    save_plot: bool = True,
) -> tuple:
    """
    Roll the trained network forward over the held-out test set and plot it
    against the true trajectory.

    The network is fed its own predictions (closed loop), integrated with
    either Euler or RK4, so the plot shows accumulated drift rather than
    single-step accuracy. For an LSTM the rolling memory window is pre-filled
    with real data from just before each chunk, then updated with the verified
    states as the rollout proceeds.

    Returns ``(true_trajectory, nn_trajectory)`` as arrays, which feed the
    composite success score.
    """
    import math
    from collections import deque

    import torch

    from backend_core.AgentSysID import config as cfg

    dyn_model.eval()
    print("\n📊 Generating pure data-driven verification from the held-out Test set...")

    arch = str(architecture or cfg.NETWORK_ARCHITECTURE).strip().upper()
    seq_len = int(lstm_seq_length or cfg.LSTM_SEQ_LENGTH) if arch == "LSTM" else 1
    int_type = str(integrator or cfg.INTEGRATOR_TYPE).strip().upper()

    if not test_trajs:
        print("    ⚠️ No test trajectories available; skipping verification.")
        return np.zeros((0, state_dim)), np.zeros((0, state_dim))

    device = next(dyn_model.parameters()).device

    test_seq = test_trajs[0]
    true_all = np.asarray(test_seq["states"], dtype=np.float32)
    act_all = np.asarray(test_seq["actions"], dtype=np.float32)
    dt_all = np.asarray(test_seq["dts"], dtype=np.float32)

    total_steps = len(true_all)
    if total_steps < 2:
        print("    ⚠️ Test trajectory too short; skipping verification.")
        return np.zeros((0, state_dim)), np.zeros((0, state_dim))

    mean_dt = float(np.mean(dt_all)) if len(dt_all) else 0.01
    chunk_size = max(1, int(round(target_horizon_seconds / max(mean_dt, 1e-9))))
    num_chunks = math.ceil(total_steps / chunk_size)

    master_true_traj: List[np.ndarray] = []
    master_nn_traj: List[np.ndarray] = []
    saved_path: Optional[Path] = None

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_steps)
        n_steps = end_idx - start_idx
        if n_steps < 2:
            continue

        true_states = true_all[start_idx:end_idx]
        actions = act_all[start_idx:end_idx]
        dts = dt_all[start_idx:end_idx]

        nn_history: List[np.ndarray] = []

        # Rolling memory window for the LSTM
        window_states: deque = deque(maxlen=seq_len)
        window_actions: deque = deque(maxlen=seq_len)
        window_dts: deque = deque(maxlen=seq_len)

        # Pre-fill the buffer with the real data right before this chunk starts
        for k in range(seq_len):
            real_idx = max(0, start_idx - seq_len + 1 + k)
            window_states.append(true_all[real_idx])
            window_actions.append(act_all[real_idx])
            window_dts.append(float(dt_all[real_idx]))

        current_state = true_states[0].copy()

        for i in range(n_steps):
            action = actions[i]
            dt = float(dts[i])

            with torch.no_grad():

                def get_k(state_val: np.ndarray) -> np.ndarray:
                    if arch == "LSTM":
                        temp_states = list(window_states) + [state_val]
                        temp_actions = list(window_actions) + [action]
                        temp_dts = list(window_dts) + [dt]

                        while len(temp_states) < seq_len:
                            temp_states.insert(0, temp_states[0])
                            temp_actions.insert(0, temp_actions[0])
                            temp_dts.insert(0, temp_dts[0])

                        s_temp = torch.tensor(
                            np.array(temp_states[-seq_len:]), dtype=torch.float32
                        ).unsqueeze(0).to(device)
                        a_temp = torch.tensor(
                            np.array(temp_actions[-seq_len:]), dtype=torch.float32
                        ).unsqueeze(0).to(device)
                        dt_temp = torch.tensor(
                            np.array(temp_dts[-seq_len:]), dtype=torch.float32
                        ).unsqueeze(-1).unsqueeze(0).to(device)
                        out, _ = dyn_model(s_temp, a_temp, dt_temp)
                    else:
                        s_temp = torch.tensor(state_val, dtype=torch.float32).unsqueeze(0).to(device)
                        a_temp = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
                        dt_temp = torch.tensor([[dt]], dtype=torch.float32).to(device)
                        out, _ = dyn_model(s_temp, a_temp, dt_temp)
                    return out.squeeze(0).cpu().numpy()

                if int_type == "RK4":
                    k1 = get_k(current_state)
                    k2 = get_k(current_state + 0.5 * dt * k1)
                    k3 = get_k(current_state + 0.5 * dt * k2)
                    k4 = get_k(current_state + dt * k3)
                    pred_xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                else:
                    pred_xdot = get_k(current_state)

                next_state = current_state + (pred_xdot * dt)

            # Push the verified state into the rolling memory AFTER predicting
            window_states.append(current_state.copy())
            window_actions.append(action)
            window_dts.append(dt)

            nn_history.append(current_state.copy())
            current_state = next_state

        master_true_traj.extend(true_states)
        master_nn_traj.extend(nn_history)

        # --- Plot this chunk ------------------------------------------------
        steps_axis = np.cumsum(dts) - dts[0]
        plot_dim = min(state_dim, 12)
        fig, axes = plt.subplots(plot_dim, 1, figsize=(12, 2.5 * plot_dim), sharex=True)
        if plot_dim == 1:
            axes = np.array([axes])
        nn_arr = np.array(nn_history)
        for idx in range(plot_dim):
            ax = axes[idx]
            ax.plot(steps_axis, true_states[:, idx], "--", color="blue", linewidth=2.5,
                    label="True Test Data")
            ax.plot(steps_axis, nn_arr[:, idx], "-", color="black", linewidth=1.5,
                    label="NN Prediction")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f"State {idx}", fontsize=10, fontweight="bold")
            if idx == 0:
                ax.legend(loc="best")
        fig.tight_layout()

        # Only the first chunk is written to disk so the PDF does not duplicate them
        if save_plot and chunk_idx == 0:
            out = _figures_dir(output_dir) / (
                f"{prefix}_{env_name}_Xdot_test_verification_{timestamp}.png"
            )
            fig.savefig(out, dpi=150, bbox_inches="tight")
            saved_path = out
            print(f"    💾 Saved Data-Driven Test Plot for PDF Report → {out.name}")
        plt.close(fig)

    if saved_path is None:
        print("    ⚠️ No verification plot was written.")

    return np.array(master_true_traj), np.array(master_nn_traj)
