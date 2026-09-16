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
