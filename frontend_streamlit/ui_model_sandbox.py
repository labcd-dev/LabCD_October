"""Interactive, local time-series preview for a completed system ID run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from backend_core.AgentSysID.agents.run_evidence import read_json, save_verification_series
from backend_core.AgentSysID.model.dynamics_model import DynamicsModel

MAX_PREVIEW_STEPS = 5_000


def is_completed(chat: dict) -> bool:
    run_dir = chat.get("run_dir")
    if not run_dir:
        return False
    manifest = read_json(Path(run_dir) / "run_manifest.json", {}) or {}
    return manifest.get("status") == "completed"


def file_item(chat: dict) -> dict | None:
    """Return the virtual Files-panel entry for a completed run's sandbox."""
    if not is_completed(chat):
        return None
    run_dir = str(Path(chat["run_dir"]).resolve())
    identity = hashlib.sha1(run_dir.encode()).hexdigest()[:16]
    return {
        "id": f"artifact:sandbox-{identity}", "kind": "sandbox", "label": "Sandbox",
        "path": run_dir, "file_name": "Input preview and data verification",
        "subtitle": "Input preview · data verification",
        "icon": "science",
    }


def _run_metadata(run_dir: Path) -> tuple[dict, dict, Path, Path, dict]:
    manifest = read_json(run_dir / "run_manifest.json", {}) or {}
    context = read_json(run_dir / "run_context.json", {}) or {}
    config = context.get("effective_config") or context.get("initial_config") or {}
    summary = context.get("data_summary") or {}
    options = context.get("options") or {}
    verification = read_json(run_dir / "verification_summary.json", {}) or {}

    dataset_path = Path(str(manifest.get("dataset") or options.get("data_path") or ""))
    if not dataset_path.is_file():
        raise ValueError("The dataset used for this run is no longer available.")

    raw_weights = str(manifest.get("pth_path") or "")
    weights_path = Path(raw_weights) if raw_weights else Path()
    if not weights_path.is_file():
        weights_path = run_dir / "deployment" / Path(raw_weights).name if raw_weights else Path()
    if not weights_path.is_file():
        candidates = sorted((run_dir / "deployment").glob("best_model*.pth"))
        weights_path = candidates[0] if candidates else Path()
    if not weights_path.is_file():
        raise ValueError("This run has no saved model weights to preview.")

    states = list(summary.get("state_columns") or [])
    actions = list(summary.get("action_columns") or [])
    if not states:
        raise ValueError("This run is missing its saved state column map.")
    state_dim = manifest.get("state_dim")
    action_dim = manifest.get("action_dim")
    if state_dim is not None and len(states) != int(state_dim):
        raise ValueError("The saved state map does not match the trained model.")
    if action_dim is not None and len(actions) != int(action_dim):
        raise ValueError("The saved input map does not match the trained model.")

    periodic_states = set(summary.get("periodic_angle_states") or [])
    angles = [index for index, name in enumerate(states) if name in periodic_states]
    if not angles:
        option_angles = options.get("angle_indices") or config.get("ANGLE_INDICES") or []
        angles = [int(index) for index in option_angles
                  if isinstance(index, (int, float)) and int(index) == index and 0 <= int(index) < len(states)]

    best_config = manifest.get("best_config") or {}
    dropout_rate = best_config.get("dropout_rate")
    if dropout_rate is None:
        dropout_rate = config.get("DROPOUT_RATE", 0.0)
    test_lengths = []
    for value in summary.get("test_trajectory_lengths", []) or []:
        try:
            length = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if length > 0:
            test_lengths.append(length)
    metadata = {
        "states": states,
        "actions": actions,
        "angle_indices": angles,
        "architecture": str(manifest.get("architecture") or config.get("NETWORK_ARCHITECTURE") or "MLP").upper(),
        "hidden_layers": list(best_config.get("hidden_layers") or [64]),
        "activation": str(manifest.get("activation") or best_config.get("activation") or "relu"),
        "dropout_rate": float(dropout_rate or 0.0),
        "history_steps": max(1, int(options.get("lstm_seq_length") or config.get("LSTM_SEQ_LENGTH") or 1)),
        "sample_period": summary.get("median_dt"),
        "time_column": str(config.get("TIME_COLUMN") or "time"),
        "integrator": str(options.get("integrator_type") or config.get("INTEGRATOR_TYPE") or "EULER").upper(),
        "test_trajectory_lengths": test_lengths,
        "verification_horizon_seconds": float(
            (verification.get("protocol") or {}).get("target_chunk_horizon_seconds") or 10.0
        ),
        "physics_mode": str(manifest.get("physics_mode") or "off").strip().lower(),
        "pcnn_output_mode": str(manifest.get("pcnn_output_mode") or
                                 ("equation_residual" if str(manifest.get("physics_mode") or "").lower() == "pcnn"
                                  else "hard_constraints")),
        "physics_parameter_estimates": dict(manifest.get("physics_parameter_estimates") or {}),
        "pcnn_hard_constraint_states": list(manifest.get("pcnn_hard_constraint_states") or []),
    }
    metadata["hard_constraint_indices"] = [
        index for index, name in enumerate(states)
        if name in metadata["pcnn_hard_constraint_states"]
    ]
    if metadata["physics_mode"] in {"pinn", "pcnn"}:
        equation_path = run_dir / "physics_equation.py"
        if not equation_path.is_file():
            raise ValueError("This physics-informed run is missing its saved validated equation.")
        metadata["physics_equation_file"] = str(equation_path.resolve())
    if not metadata["test_trajectory_lengths"]:
        for item in verification.get("per_trajectory", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                length = int(item.get("aligned_samples") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if length > 0:
                metadata["test_trajectory_lengths"].append(length)
    if metadata["architecture"] not in {"LSTM", "MLP"}:
        raise ValueError("The saved model architecture is not supported by this sandbox.")
    return manifest, context, dataset_path, weights_path, metadata


@st.cache_data(show_spinner=False)
def _read_recent_data(dataset_path: str, modified_ns: int, states: tuple[str, ...],
                      actions: tuple[str, ...], time_column: str, keep_rows: int) -> dict:
    del modified_ns
    path = Path(dataset_path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator) if path.suffix.lower() in {".csv", ".tsv"} else pd.read_excel(path)
    missing = [name for name in (*states, *actions) if name not in frame.columns]
    if missing:
        raise ValueError("The saved dataset is missing columns used by this run: " + ", ".join(missing[:5]))
    numeric = frame.loc[:, [*states, *actions]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        raise ValueError("No finite state and input samples are available for the sandbox.")
    recent = numeric.tail(max(1, keep_rows))

    interval = None
    if time_column in frame.columns and len(frame) > 1:
        times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
        deltas = np.diff(times)
        positive = deltas[np.isfinite(deltas) & (deltas > 0)]
        if positive.size:
            interval = float(np.median(positive))

    return {
        "states": recent.loc[:, list(states)].to_numpy(dtype=np.float32),
        "actions": recent.loc[:, list(actions)].to_numpy(dtype=np.float32),
        "last_state": numeric.loc[:, list(states)].iloc[-1].to_numpy(dtype=np.float32),
        "last_action": numeric.loc[:, list(actions)].iloc[-1].to_numpy(dtype=np.float32),
        "action_min": numeric.loc[:, list(actions)].min().to_numpy(dtype=float),
        "action_max": numeric.loc[:, list(actions)].max().to_numpy(dtype=float),
        "sample_period": interval,
        "sample_count": int(len(numeric)),
    }


@st.cache_data(show_spinner=False)
def _read_verification_source(dataset_path: str, modified_ns: int,
                              states: tuple[str, ...], actions: tuple[str, ...],
                              time_column: str) -> dict:
    """Read aligned source rows for rebuilding an older run's saved test view."""
    del modified_ns
    path = Path(dataset_path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator) if path.suffix.lower() in {".csv", ".tsv"} else pd.read_excel(path)
    required = [time_column, *states, *actions]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("The saved dataset is missing verification columns: " + ", ".join(missing[:5]))
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < 2:
        raise ValueError("There are not enough finite saved samples to draw this verification.")
    return {
        "times": numeric[time_column].to_numpy(dtype=float),
        "states": numeric.loc[:, list(states)].to_numpy(dtype=np.float32),
        "actions": numeric.loc[:, list(actions)].to_numpy(dtype=np.float32),
    }


@st.cache_data(show_spinner=False)
def _read_saved_verification(path: str, modified_ns: int) -> dict:
    del modified_ns
    with np.load(path, allow_pickle=False) as saved:
        return {
            "time_seconds": np.asarray(saved["time_seconds"], dtype=float),
            "measured": np.asarray(saved["measured"], dtype=float),
            "predicted": np.asarray(saved["predicted"], dtype=float),
            "state_names": [str(name) for name in saved["state_names"].tolist()],
            "trajectory_index": int(np.asarray(saved["trajectory_index"]).reshape(-1)[0]),
        }


def _verification_series(run_dir: Path, dataset_path: Path,
                         metadata: dict, model: DynamicsModel) -> dict:
    """Load the one saved held-out series, rebuilding it for earlier runs."""
    series_path = run_dir / "verification" / "single_heldout_trajectory.npz"
    if series_path.is_file():
        return _read_saved_verification(str(series_path.resolve()), series_path.stat().st_mtime_ns)

    lengths = [int(length) for length in metadata.get("test_trajectory_lengths", []) if int(length) > 0]
    if not lengths:
        raise ValueError("This older run did not save the held-out trajectory boundaries.")
    source = _read_verification_source(
        str(dataset_path), dataset_path.stat().st_mtime_ns,
        tuple(metadata["states"]), tuple(metadata["actions"]), metadata["time_column"],
    )
    count = lengths[-1]
    if count < 2 or sum(lengths) > len(source["states"]):
        raise ValueError("The saved test split no longer matches the dataset rows.")

    start = len(source["states"]) - count
    states = source["states"][start:]
    actions = source["actions"][start:]
    times = source["times"][start:]
    differences = np.diff(times)
    positive = differences[np.isfinite(differences) & (differences > 1e-9)]
    fallback_dt = float(np.median(positive)) if positive.size else float(metadata.get("sample_period") or 0.01)
    dt_values = np.concatenate((differences, [positive[-1] if positive.size else fallback_dt])).astype(np.float32)
    dt_values[~np.isfinite(dt_values) | (dt_values < 1e-6)] = max(fallback_dt, 1e-6)
    trajectory = {"states": states, "actions": actions, "times": times, "dts": dt_values}
    from backend_core.AgentSysID.reporting.plots import plot_test_dataset_verification

    measured, forecast = plot_test_dataset_verification(
        model, [trajectory], len(metadata["states"]), len(metadata["actions"]),
        output_dir=run_dir, timestamp="sandbox", prefix="sandbox",
        env_name="verification", architecture=metadata["architecture"],
        lstm_seq_length=metadata["history_steps"], integrator=metadata["integrator"],
        target_horizon_seconds=metadata.get("verification_horizon_seconds", 10.0),
        save_plot=False, angle_indices=metadata["angle_indices"],
    )
    count = min(len(measured), len(forecast), len(times))
    if count < 2:
        raise ValueError("The saved model did not produce enough points for this held-out trajectory.")
    elapsed = times - times[0]
    series = {
        "time_seconds": elapsed[:count],
        "measured": measured[:count].astype(float),
        "predicted": forecast[:count].astype(float),
        "state_names": list(metadata["states"]),
        "trajectory_index": len(lengths),
    }
    save_verification_series(
        run_dir, series["time_seconds"], series["measured"], series["predicted"], metadata["states"],
        trajectory_index=len(lengths),
    )
    return series


@st.cache_resource(show_spinner=False)
def _load_model(run_dir: str, weights_path: str, weights_mtime_ns: int,
                metadata_json: str) -> DynamicsModel:
    del run_dir, weights_mtime_ns  # Used as cache keys so replaced runs reload cleanly.
    metadata = json.loads(metadata_json)
    states, actions = metadata["states"], metadata["actions"]
    model = DynamicsModel(
        state_dim=len(states), action_dim=len(actions),
        hidden_layers=[int(width) for width in metadata["hidden_layers"]],
        activation=metadata["activation"], dropout_rate=float(metadata["dropout_rate"]),
        architecture=metadata["architecture"], angle_indices=metadata["angle_indices"],
    ).cpu()
    if metadata.get("physics_mode") in {"pinn", "pcnn"}:
        from backend_core.AgentSysID.model.pinn import load_analytical_xdot

        physics_fn = load_analytical_xdot(metadata["physics_equation_file"])
        set_estimates = getattr(physics_fn, "set_parameter_estimates", None)
        if callable(set_estimates):
            set_estimates(metadata.get("physics_parameter_estimates") or {})
        model.configure_physics(
            metadata["physics_mode"], physics_fn,
            hard_constraint_indices=metadata.get("hard_constraint_indices", []),
            pcnn_output_mode=metadata.get("pcnn_output_mode", "hard_constraints"),
        )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _step_size(value: float) -> float:
    return max(abs(float(value)) * 0.02, 1e-4)


def _predict(model: DynamicsModel, data: dict, metadata: dict,
             current_state: np.ndarray, action: np.ndarray, dt: float,
             history_states: np.ndarray | None = None,
             history_actions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    def derivative_at(state: np.ndarray) -> np.ndarray:
        if metadata["architecture"] == "LSTM":
            length = metadata["history_steps"]
            states = np.asarray(history_states if history_states is not None else data["states"], dtype=np.float32)
            actions = np.asarray(history_actions if history_actions is not None else data["actions"], dtype=np.float32)
            states, actions = states[-length:].copy(), actions[-length:].copy()
            if len(states) < length:
                pad = length - len(states)
                states = np.concatenate([np.repeat(states[:1], pad, axis=0), states], axis=0)
                actions = np.concatenate([np.repeat(actions[:1], pad, axis=0), actions], axis=0)
            states[-1], actions[-1] = state, action
            state_tensor = torch.as_tensor(states, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0)
            dt_tensor = torch.full((1, length, 1), float(dt), dtype=torch.float32)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            dt_tensor = torch.tensor([[float(dt)]], dtype=torch.float32)
        with torch.inference_mode():
            derivative, _ = model(state_tensor, action_tensor, dt_tensor)
        return derivative.squeeze(0).cpu().numpy().astype(float)

    k1 = derivative_at(current_state)
    if metadata["integrator"] == "RK4":
        k2 = derivative_at(current_state + 0.5 * dt * k1)
        k3 = derivative_at(current_state + 0.5 * dt * k2)
        k4 = derivative_at(current_state + dt * k3)
        rates = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    else:
        rates = k1
    predicted = current_state.astype(float) + rates * float(dt)
    for index in metadata["angle_indices"]:
        predicted[index] = (predicted[index] + np.pi) % (2 * np.pi) - np.pi
    if not np.isfinite(rates).all() or not np.isfinite(predicted).all():
        raise ValueError("The model returned a non-finite prediction for these values.")
    return rates, predicted


def _input_profile(profile: str, times: np.ndarray, baseline: np.ndarray,
                   input_index: int, amplitude: float, duration: float,
                   frequency: float = 0.2, change_time: float = 0.0) -> np.ndarray:
    """Create the selected signal while holding every other input at baseline."""
    actions = np.repeat(np.asarray(baseline, dtype=float)[None, :], len(times), axis=0)
    elapsed = np.asarray(times, dtype=float)
    if profile == "Sine":
        change = float(amplitude) * np.sin(2.0 * np.pi * float(frequency) * elapsed)
    elif profile == "Step":
        change = float(amplitude) * (elapsed >= float(change_time))
    elif profile == "Ramp":
        change = float(amplitude) * np.clip(elapsed / max(float(duration), 1e-12), 0.0, 1.0)
    else:
        raise ValueError("Choose a sine, step, or ramp input.")
    actions[:, int(input_index)] += change
    return actions


def _input_profiles(times: np.ndarray, baseline: np.ndarray,
                    profiles: list[dict]) -> np.ndarray:
    """Apply independent signals to chosen inputs and hold all others at baseline."""
    baseline = np.asarray(baseline, dtype=float)
    actions = np.repeat(baseline[None, :], len(times), axis=0)
    duration = float(np.asarray(times, dtype=float)[-1]) if len(times) else 0.0
    for item in profiles:
        index = int(item["input_index"])
        if index < 0 or index >= len(baseline):
            raise ValueError("An input profile refers to a missing input channel.")
        channel = _input_profile(
            str(item["profile"]), times, baseline, index,
            float(item["amplitude"]), duration,
            float(item.get("frequency", 0.2)), float(item.get("change_time", 0.0)),
        )
        actions[:, index] = channel[:, index]
    return actions


def _verification_figure(series: dict, *, run_key: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    times = np.asarray(series["time_seconds"], dtype=float).reshape(-1)
    measured = np.asarray(series["measured"], dtype=float)
    predicted = np.asarray(series["predicted"], dtype=float)
    state_names = list(series["state_names"])
    if measured.ndim != 2 or predicted.shape != measured.shape or len(times) != len(measured):
        raise ValueError("The saved held-out verification series has inconsistent dimensions.")
    if not state_names or measured.shape[1] != len(state_names):
        raise ValueError("The saved state names do not match this verification series.")

    # Keep even long recordings responsive while preserving the full trajectory shape.
    if len(times) > 4_000:
        indices = np.linspace(0, len(times) - 1, 4_000, dtype=int)
        times, measured, predicted = times[indices], measured[indices], predicted[indices]

    rows = len(state_names)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.045, 0.28 / max(rows, 1)),
        subplot_titles=state_names,
    )
    for index, name in enumerate(state_names, start=1):
        fig.add_trace(go.Scatter(
            x=times, y=measured[:, index - 1], mode="lines", name="Measured",
            legendgroup="measured", showlegend=index == 1,
            line={"color": "#53dfc0", "width": 2.6},
            hovertemplate="%{x:.4g} s · %{y:.5g}<extra>Measured</extra>",
        ), row=index, col=1)
        fig.add_trace(go.Scatter(
            x=times, y=predicted[:, index - 1], mode="lines", name="Model prediction",
            legendgroup="prediction", showlegend=index == 1,
            line={"color": "#aa9bff", "width": 2.2, "dash": "dash"},
            hovertemplate="%{x:.4g} s · %{y:.5g}<extra>Model prediction</extra>",
        ), row=index, col=1)
        fig.update_yaxes(
            title_text=name, showgrid=True,
            gridcolor="rgba(148, 170, 166, 0.14)",
            zeroline=False, tickfont={"size": 11},
            title_font={"size": 12, "color": "#dce8e5"},
            row=index, col=1,
        )

    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(148, 170, 166, 0.12)",
        zeroline=False, tickfont={"size": 10},
    )
    fig.update_xaxes(title_text="Elapsed time (s)", row=rows, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(13, 24, 25, 0.92)",
        font={"family": "Inter, sans-serif", "color": "#dce8e5", "size": 12},
        height=max(380, min(1_020, 150 * rows)),
        margin={"l": 18, "r": 24, "t": 56, "b": 44},
        hovermode="x unified", uirevision=run_key,
        legend={
            "orientation": "h", "y": 1.025, "x": 1, "xanchor": "right",
            "yanchor": "bottom", "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"size": 11}, "groupclick": "togglegroup",
        },
    )
    fig.update_annotations(font={"size": 12, "color": "#9fc3b9"})
    return fig


def _render_data_verification(run_dir: Path, dataset_path: Path, metadata: dict,
                              model: DynamicsModel, run_key: str) -> None:
    try:
        with st.spinner("Preparing one held-out trajectory…"):
            series = _verification_series(run_dir, dataset_path, metadata, model)
    except Exception as exc:
        st.info(f"This run's data verification is unavailable: {exc}")
        return

    st.markdown("**Measured data vs model prediction**")
    st.caption("One held-out trajectory from this run. Other trajectories are kept separate and are not combined here.")
    samples = len(series["time_seconds"])
    duration = float(series["time_seconds"][-1] - series["time_seconds"][0]) if samples else 0.0
    trajectory_index = int(series.get("trajectory_index") or 1)
    a, b, c = st.columns(3, gap="small")
    a.metric("Held-out trajectory", str(trajectory_index))
    b.metric("Samples", f"{samples:,}")
    c.metric("Duration", f"{duration:.4g} s")

    try:
        figure = _verification_figure(series, run_key=run_key)
    except ImportError:
        st.info("The Plotly chart library is not installed in this environment. Install the project requirements to enable this view.")
        return
    except Exception as exc:
        st.info(f"The verification chart could not be drawn: {exc}")
        return
    st.plotly_chart(
        figure, width="stretch", theme=None, key=f"sandbox_{run_key}_verification",
        config={"displaylogo": False, "scrollZoom": False, "responsive": True},
    )
    st.caption("Solid mint lines show recorded states; dashed violet lines show the model rollout, initialized from the measured starting state.")


def _rollout(model: DynamicsModel, data: dict, metadata: dict, initial_state: np.ndarray,
             action_profile: np.ndarray, dt: float,
             prediction_steps: int | None = None) -> np.ndarray:
    """Integrate model derivatives forward, appending each predicted state to LSTM context."""
    history_states = np.asarray(data["states"], dtype=np.float32).copy()
    history_actions = np.asarray(data["actions"], dtype=np.float32).copy()
    current = np.asarray(initial_state, dtype=float).copy()
    trajectory = [current.copy()]
    length = max(1, int(metadata["history_steps"]))
    actions = np.asarray(action_profile, dtype=float)
    available_steps = max(0, len(actions) - 1)
    steps = available_steps if prediction_steps is None else min(
        available_steps, max(0, int(prediction_steps)),
    )

    for action in actions[:steps]:
        _, current = _predict(
            model, data, metadata, current, action, dt,
            history_states=history_states[-length:], history_actions=history_actions[-length:],
        )
        trajectory.append(current.copy())
        history_states = np.concatenate((history_states, current.astype(np.float32)[None, :]), axis=0)
        history_actions = np.concatenate((history_actions, action.astype(np.float32)[None, :]), axis=0)

    result = np.asarray(trajectory, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("The simulation became non-finite. Try a smaller input change or shorter duration.")
    return result


def _render_plot(chat: dict, run_dir: Path) -> None:
    try:
        _, _, dataset_path, weights_path, metadata = _run_metadata(run_dir)
        data = _read_recent_data(
            str(dataset_path), dataset_path.stat().st_mtime_ns,
            tuple(metadata["states"]), tuple(metadata["actions"]), metadata["time_column"],
            max(metadata["history_steps"], 2),
        )
        sample_period = float(metadata.get("sample_period") or data.get("sample_period") or 0.01)
        if not np.isfinite(sample_period) or sample_period <= 0:
            raise ValueError("A positive sample interval could not be read from this run.")
        model = _load_model(str(run_dir.resolve()), str(weights_path.resolve()),
                            weights_path.stat().st_mtime_ns, json.dumps(metadata, sort_keys=True))
    except Exception as exc:
        st.info(f"The sandbox is unavailable for this run: {exc}")
        return

    state_names, input_names = metadata["states"], metadata["actions"]
    run_key = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:10]
    view = st.selectbox(
        "Sandbox view", ["Input preview", "With your data verification"],
        key=f"sandbox_{run_key}_view",
    )
    if view == "With your data verification":
        _render_data_verification(run_dir, dataset_path, metadata, model, run_key)
        return

    max_duration = max(sample_period * 2, min(sample_period * max(1, data["sample_count"] - 1),
                                               sample_period * MAX_PREVIEW_STEPS))
    default_duration = min(max_duration, max(sample_period, 5.0))

    st.markdown("**Choose an input profile**")
    st.caption(f"{metadata['architecture']} · {len(state_names)} states · {len(input_names)} inputs · {sample_period:.5g} s/sample")
    st.caption("Explore a predicted time response with the saved model. This preview does not control hardware.")
    with st.container(key="model_sandbox_summary"):
        st.markdown('<div class="sandbox-preview-note"><span>MODEL PREVIEW</span><b>Measured starting state</b><small>Inputs are simulated locally · no physical system is connected</small></div>',
                    unsafe_allow_html=True)

    duration_key = f"sandbox_{run_key}_duration"
    duration = float(st.slider(
        "Simulation duration (s)", min_value=float(sample_period), max_value=float(max_duration),
        value=float(default_duration), step=float(sample_period), key=duration_key,
        help="The plot updates to this simulated time. The preview is limited to 5,000 model steps.",
    ))
    steps = min(MAX_PREVIEW_STEPS, max(1, int(round(duration / sample_period))))
    times = np.arange(steps + 1, dtype=float) * sample_period
    effective_duration = float(times[-1])
    prediction_steps = steps
    st.caption("Prediction automatically covers the full selected duration.")

    if not input_names:
        st.info("This run has no saved input channels to vary.")
        return

    selected_inputs = st.multiselect(
        "Inputs to vary", options=input_names, default=input_names,
        key=f"sandbox_{run_key}_inputs",
        help="Choose every input or just the channels you want to change. Unselected inputs stay at their final measured values.",
    )
    if selected_inputs:
        st.caption("Choose a separate signal for each input. Remove a name above to keep that input at its measured value.")
    else:
        st.info("No inputs selected. The preview will hold every input at its final measured value.")

    baseline_actions = np.asarray(data["last_action"], dtype=float).copy()
    profiles = []
    for selected_position, selected_input in enumerate(selected_inputs):
        input_index = input_names.index(selected_input)
        spread = max(float(data["action_max"][input_index] - data["action_min"][input_index]),
                     abs(float(data["last_action"][input_index])), 1e-4)
        control_key = f"sandbox_{run_key}_input_{input_index}"
        with st.expander(f"{selected_input} · input profile", expanded=selected_position == 0):
            profile = st.segmented_control(
                "Signal", ["Sine", "Step", "Ramp"], default="Sine",
                key=f"{control_key}_profile", width="stretch",
            ) or "Sine"
            baseline_col, amplitude_col = st.columns(2, gap="small")
            with baseline_col:
                baseline_actions[input_index] = float(st.number_input(
                    "Starting input", value=float(data["last_action"][input_index]),
                    step=_step_size(spread), key=f"{control_key}_baseline",
                ))
            with amplitude_col:
                amplitude = float(st.number_input(
                    "Input change", value=float(spread * 0.5), step=_step_size(spread),
                    key=f"{control_key}_amplitude",
                    help="The sine amplitude, step height, or ramp's total change.",
                ))

            frequency, change_time = 0.2, effective_duration * 0.5
            if profile == "Sine":
                maximum_frequency = max(0.002, min(10.0, 0.5 / sample_period))
                frequency = float(st.number_input(
                    "Frequency (Hz)", min_value=0.001, max_value=float(maximum_frequency),
                    value=float(min(0.2, maximum_frequency)), step=float(min(0.1, maximum_frequency)),
                    format="%.4g", key=f"{control_key}_frequency",
                ))
            elif profile == "Step":
                step_time_key = f"{control_key}_step_time"
                if step_time_key in st.session_state:
                    st.session_state[step_time_key] = min(
                        max(0.0, float(st.session_state[step_time_key])), effective_duration)
                change_time = float(st.slider(
                    "Step begins at (s)", min_value=0.0, max_value=float(effective_duration),
                    value=float(effective_duration * 0.5), step=float(sample_period),
                    key=step_time_key,
                ))
            else:
                st.caption("The ramp reaches the full input change at the end of the selected duration.")
            profiles.append({
                "input_index": input_index, "profile": profile,
                "amplitude": amplitude, "frequency": frequency,
                "change_time": change_time,
            })

    with st.expander("Starting state", expanded=False):
        st.caption("Defaults to the latest measured state from this run.")
        state_cols = st.columns(min(2, len(state_names)), gap="small")
        start_state = []
        for index, name in enumerate(state_names):
            with state_cols[index % len(state_cols)]:
                start_state.append(float(st.number_input(
                    name, value=float(data["last_state"][index]),
                    step=_step_size(float(data["last_state"][index])),
                    key=f"sandbox_{run_key}_state_{index}",
                )))

    actions = _input_profiles(times, baseline_actions, profiles)
    try:
        with st.spinner("Simulating the selected input…"):
            states = _rollout(
                model, data, metadata, np.asarray(start_state), actions, sample_period,
                prediction_steps=prediction_steps,
            )
    except Exception as exc:
        st.error(f"The time response could not be simulated: {exc}")
        return

    prediction_times = times[:len(states)]
    result = pd.DataFrame(states, columns=state_names)
    result.insert(0, "Time (s)", prediction_times)
    st.markdown("**Predicted state response**")
    st.line_chart(result, x="Time (s)", y=state_names, height=300)
    st.markdown("**Applied input profiles**")
    if selected_inputs:
        input_result = pd.DataFrame(
            actions[:len(states), :][:, [input_names.index(name) for name in selected_inputs]],
            columns=selected_inputs,
        )
        input_result.insert(0, "Time (s)", prediction_times)
        st.line_chart(input_result, x="Time (s)", y=selected_inputs, height=200)
    else:
        st.caption("Every input stays at its final measured value during this preview.")

    out_of_range = []
    for name in selected_inputs:
        index = input_names.index(name)
        predicted_actions = actions[:prediction_steps, index]
        if ((predicted_actions < data["action_min"][index]) |
                (predicted_actions > data["action_max"][index])).any():
            out_of_range.append(name)
    if out_of_range:
        st.warning("These inputs go outside the range seen during training: " + ", ".join(out_of_range) + ". The affected response is an extrapolation.")
    profile_summary = ", ".join(f"{item['profile']} · {input_names[item['input_index']]}" for item in profiles)
    predicted_duration = float(prediction_times[-1]) if len(prediction_times) else 0.0
    st.caption(
        f"{profile_summary or 'Inputs held'} · {prediction_steps:,} prediction steps · "
        f"{predicted_duration:.4g} s predicted · unselected inputs held at their final measured values"
    )


def render(chat: dict) -> None:
    run_dir = Path(chat.get("run_dir") or "")
    _render_plot(chat, run_dir)
