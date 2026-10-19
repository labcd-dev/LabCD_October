"""Measured evidence tools for the conversational system-identification agent."""
from __future__ import annotations

import hashlib
import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend_core.AgentSysID.agents.run_evidence import read_json


ANGLE_WORDS = ("angle", "pitch", "yaw", "roll", "heading", "theta", "phi", "psi")


def read_dataset(name: str, data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data)) if Path(name).suffix.lower() == ".csv" else pd.read_excel(io.BytesIO(data))


def profile_frame(frame: pd.DataFrame, states: list[str], actions: list[str], derivatives: list[str]) -> dict:
    """Return compact, JSON-safe facts from the actual measurements."""
    count = len(frame)
    times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float) if "time" in frame else np.array([])
    dts = np.diff(times)
    positive = dts[np.isfinite(dts) & (dts > 0)]
    median_dt = float(np.median(positive)) if len(positive) else None
    irregular = int(np.sum(np.abs(positive - median_dt) > max(1e-8, 0.01 * median_dt))) if median_dt else 0
    reset_rows = np.flatnonzero(dts < 0).astype(int) + 1
    duplicate_rows = np.flatnonzero(dts == 0).astype(int) + 1
    start = max(1, int(count * 0.9))
    signals = []
    detected_wrap = []
    possible_boundary_rows = set()
    possible_boundary_states = {}
    for name in states:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        steps = np.abs(np.diff(values))
        valid_steps = steps[np.isfinite(steps) & (dts > 0)] if len(dts) == len(steps) else steps[np.isfinite(steps)]
        near_radians = float(np.min(finite)) >= -3.2 and float(np.max(finite)) <= 3.2
        wraps = bool(near_radians and np.any(valid_steps > 5.0))
        if wraps:
            detected_wrap.append(name)
        if len(valid_steps) and not wraps:
            threshold = max(8 * float(np.median(valid_steps)), 0.5 * float(np.ptp(finite)), 1e-10)
            for row in (np.flatnonzero(steps > threshold) + 1).astype(int).tolist():
                possible_boundary_rows.add(row)
                possible_boundary_states.setdefault(row, []).append(name)
        baseline = None
        if count - start >= 2 and len(dts) == count - 1:
            working = np.unwrap(values) if wraps else values
            actual, previous = working[start + 1:], working[start:-1]
            valid = np.isfinite(actual) & np.isfinite(previous) & (dts[start:] > 0)
            if int(np.sum(valid)) >= 2:
                error = actual[valid] - previous[valid]
                scale = float(np.std(actual[valid]))
                rmse = float(np.sqrt(np.mean(error * error)))
                baseline = {"rmse": rmse, "over_test_std": rmse / scale if scale > 0 else None,
                            "samples": int(np.sum(valid))}
        early = values[:start]
        late = values[start:]
        early = early[np.isfinite(early)]
        late = late[np.isfinite(late)]
        early_mean = float(np.mean(early)) if len(early) else None
        late_mean = float(np.mean(late)) if len(late) else None
        shift = ((late_mean - early_mean) / float(np.std(finite))
                 if early_mean is not None and late_mean is not None and float(np.std(finite)) > 0 else None)
        signals.append({"name": name, "min": float(np.min(finite)), "max": float(np.max(finite)),
                        "std": float(np.std(finite)), "span": float(np.ptp(finite)),
                        "first": float(values[0]) if np.isfinite(values[0]) else None,
                        "last": float(values[-1]) if np.isfinite(values[-1]) else None,
                        "q10": float(np.quantile(finite, .1)), "median": float(np.median(finite)),
                        "q90": float(np.quantile(finite, .9)),
                        "last_tenth_mean_shift_over_full_std": shift,
                        "angle_named": any(word in name.lower() for word in ANGLE_WORDS),
                        "wrap_jumps": int(np.sum(valid_steps > 5.0)) if wraps else 0,
                        "persistence": baseline})
    inputs = []
    for name in actions:
        finite = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            inputs.append({"name": name, "min": float(np.min(finite)), "max": float(np.max(finite)),
                           "std": float(np.std(finite)), "constant": float(np.ptp(finite)) < 1e-10})
    duration = float(np.sum(positive)) if len(positive) else None
    derivative_checks = []
    wrapped_states = set(detected_wrap)
    for state in states:
        derivative = "xdot_" + state.removeprefix("s_")
        if derivative not in derivatives or len(times) < 3:
            continue
        values = pd.to_numeric(frame[state], errors="coerce").to_numpy(dtype=float)
        unwrapped = state in wrapped_states
        if unwrapped:
            values = np.unwrap(values)
        observed = pd.to_numeric(frame[derivative], errors="coerce").to_numpy(dtype=float)[1:]
        valid = (np.isfinite(observed) & np.isfinite(values[1:]) & np.isfinite(values[:-1]) &
                 np.isfinite(dts) & (dts > 0))
        if int(np.sum(valid)) < 3:
            continue
        finite_difference = np.diff(values)[valid] / dts[valid]
        reference = observed[valid]
        scale = float(np.std(reference))
        reference_center = float(np.mean(reference))
        variable_scale = scale > max(1e-12, abs(reference_center) * 1e-6)
        rmse = float(np.sqrt(np.mean((finite_difference - reference) ** 2)))
        derivative_checks.append({"state": state, "derivative": derivative, "valid_pairs": int(np.sum(valid)),
                                  "finite_difference_vs_supplied_rmse": rmse,
                                  "rmse_over_supplied_std": rmse / scale if variable_scale else None,
                                  "angle_unwrapped_for_check": unwrapped,
                                  "note": ("Angular samples were unwrapped before differencing. " if unwrapped else "") +
                                          "Descriptive consistency check; finite differences amplify measurement noise. " +
                                          ("Normalized error is omitted because supplied derivatives are effectively constant." if not variable_scale else "")})
    correlated_state_pairs = []
    # Bound this optional advisory check for uploads near the workspace limits.
    correlation_states = list(dict.fromkeys(states))[:64]
    if len(correlation_states) > 1 and not frame.columns.duplicated().any():
        stride = max(1, (len(frame) + 4999) // 5000)
        numeric_states = frame.iloc[::stride][correlation_states].apply(pd.to_numeric, errors="coerce")
        correlations = numeric_states.corr().abs()
        for left_index, left in enumerate(correlation_states):
            for right in correlation_states[left_index + 1:]:
                value = correlations.loc[left, right]
                if pd.notna(value) and float(value) >= 0.95:
                    correlated_state_pairs.append({"state_a": left, "state_b": right,
                                                   "abs_correlation": float(value)})
                    if len(correlated_state_pairs) >= 12:
                        break
            if len(correlated_state_pairs) >= 12:
                break
    reset_row_indices = set(reset_rows.tolist())
    candidate_boundary_indices = sorted(reset_row_indices | possible_boundary_rows)
    boundary_events = []
    for row in candidate_boundary_indices[:20]:
        before = float(times[row - 1]) if row > 0 and row - 1 < len(times) and np.isfinite(times[row - 1]) else None
        after = float(times[row]) if row < len(times) and np.isfinite(times[row]) else None
        boundary_events.append({
            "data_row": int(row + 1),
            "time_before": before,
            "time_after": after,
            "time_reset": row in reset_row_indices,
            "state_jumps": sorted(possible_boundary_states.get(row, [])),
        })
    return {"rows": count, "time_start": float(times[0]) if len(times) and np.isfinite(times[0]) else None,
            "time_end": float(times[-1]) if len(times) and np.isfinite(times[-1]) else None,
            "observed_duration": duration, "median_dt": median_dt,
            "irregular_intervals": irregular, "positive_intervals": int(len(positive)),
            "time_reset_rows": reset_rows[:20].tolist(), "time_reset_count": int(len(reset_rows)),
            "boundary_events": boundary_events,
            "boundary_event_count": len(candidate_boundary_indices),
            "duplicate_time_rows": duplicate_rows[:20].tolist(), "duplicate_time_count": int(len(duplicate_rows)),
            "states": signals, "inputs": inputs, "derivative_count": len(derivatives),
            "derivative_consistency": derivative_checks,
            "complete_derivatives": len(derivatives) == len(states) and len(states) > 0,
            "detected_wrap_states": detected_wrap,
            "angle_named_states": [s["name"] for s in signals if s["angle_named"]],
            "possible_boundary_count": len(possible_boundary_rows),
            "correlated_state_pairs": correlated_state_pairs}


def ensure_profile(dataset: dict) -> dict:
    """Backfill analysis for chats saved before profiles were introduced."""
    if (isinstance(dataset.get("analysis"), dict) and
            "correlated_state_pairs" in dataset["analysis"] and
            "boundary_events" in dataset["analysis"]):
        return dataset["analysis"]
    path = Path(dataset["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != dataset["sha256"]:
        raise ValueError("The saved attachment changed. Attach the dataset again to analyze it.")
    frame = read_dataset(dataset["name"], data)
    dataset["analysis"] = profile_frame(frame, dataset["states"], dataset["actions"], dataset["derivatives"])
    return dataset["analysis"]


def state_comparison_question(question: str) -> bool:
    q = question.lower()
    state_ref = bool(re.search(r"states?|s_[a-z0-9_]+|pitch|yaw|roll|position|velocity", q))
    performance = bool(re.search(r"predict|accur|error|rmse|mae|better|best|worst|compare|perform", q))
    return state_ref and performance


def compare_run_states(run_dir: str) -> dict:
    path = Path(run_dir) / "verification_summary.json"
    verification = read_json(path, {}) or {}
    trajectory_states = {}
    for trajectory in verification.get("per_trajectory", []):
        if not isinstance(trajectory, dict):
            continue
        trajectory_number = trajectory.get("trajectory_index")
        for item in trajectory.get("states", []):
            if isinstance(item, dict) and item.get("state"):
                trajectory_states.setdefault(str(item["state"]), []).append({
                    "trajectory_index": trajectory_number,
                    "rmse": item.get("rmse"),
                    "mae": item.get("mae"),
                    "normalized_rmse": item.get("rmse_over_test_std"),
                    "error_type": item.get("error_type", "linear"),
                    "valid_samples": item.get("valid_samples"),
                })
    rows = []
    for item in verification.get("states", []):
        try:
            rmse, ratio = float(item["rmse"]), item.get("rmse_over_test_std")
            ratio = float(ratio) if ratio is not None else None
            if not math.isfinite(rmse) or (ratio is not None and not math.isfinite(ratio)):
                continue
            name = str(item["state"])
            rows.append({"state": name, "rmse": rmse, "mae": float(item["mae"]),
                         "normalized_rmse": ratio, "valid_samples": int(item["valid_samples"]),
                         "error_type": item.get("error_type", "linear"),
                         "per_trajectory": trajectory_states.get(name, [])})
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return {"rows": [], "source": str(path)}
    return {"rows": sorted(rows, key=lambda r: (
                r["normalized_rmse"] is None,
                r["normalized_rmse"] if r["normalized_rmse"] is not None else math.inf)),
            "aligned_samples": verification.get("aligned_samples"), "protocol": verification.get("protocol", {}),
            "source": str(path)}
