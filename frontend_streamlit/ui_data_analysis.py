"""In-chat measurement plots, setup choices and per-state run evidence."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from . import conversation_analysis as analysis, ui_activity as activity, ui_history as hist
except ImportError:
    import conversation_analysis as analysis
    import ui_activity as activity
    import ui_history as hist


@st.cache_data(show_spinner=False, max_entries=24)
def _preview(path: str, digest: str, name: str) -> pd.DataFrame:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("The saved dataset changed. Attach it again to see its plots.")
    frame = analysis.read_dataset(name, data)
    if len(frame) > 800:
        frame = frame.iloc[::max(1, len(frame) // 800)].copy()
    return frame


def _signal(dataset: dict, name: str, key: str, *, height=205):
    try:
        frame = _preview(dataset["path"], dataset["sha256"], dataset["name"])
        if name not in frame:
            return
        if "time" in frame:
            st.line_chart(frame[["time", name]], x="time", y=name, x_label="Time (s)",
                          y_label=name, height=height)
        else:
            st.line_chart(frame[[name]], y=name, x_label="Sample", y_label=name, height=height)
    except (OSError, ValueError) as exc:
        st.warning(str(exc))


def _cycle_history(entry: dict) -> pd.DataFrame:
    """Read finite per-candidate scores saved in the manifest or activity feed."""
    rows = entry.get("performance_history") or []
    if not rows and entry.get("run_dir"):
        rows = [row.get("details", {}) for row in activity.load(entry["run_dir"])
                if row.get("id", "").startswith("actor:") and row.get("state") == "complete"]
    clean = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        try:
            cycle = int(row.get("cycle", row.get("iteration", index)))
            train = float(row["train_mse"])
            validation = float(row.get("val_mse", row.get("mse")))
            if not (math.isfinite(train) and math.isfinite(validation)):
                continue
            clean.append({"Cycle": cycle, "Training MSE": train,
                          "Validation MSE": validation,
                          "Training seconds": float(row.get("training_seconds") or 0),
                          "Configuration": row.get("config") if isinstance(row.get("config"), dict) else {}})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return pd.DataFrame(clean).sort_values("Cycle") if clean else pd.DataFrame()


def _plot_label(path: str) -> str:
    stem = Path(path).stem.lower()
    for token, label in [("test_verification", "Held-out predictions"),
                         ("mse_convergence", "Validation MSE by cycle"),
                         ("rmse_convergence", "Validation RMSE by cycle"),
                         ("latency_evolution", "Inference latency"),
                         ("hyperparameters", "Training settings explored"),
                         ("layers_neurons", "Architecture search")]:
        if token in stem:
            return label
    return Path(path).stem.replace("_", " ").replace("-", " ").title()


def _cycle_curve(history: pd.DataFrame, key: str, *, inspect=True):
    if history.empty:
        st.caption("Per-cycle scores were not saved for this run.")
        return
    chart = history.set_index("Cycle")[["Training MSE", "Validation MSE"]]
    st.line_chart(chart, height=230, y_label="Mean squared error")
    st.caption("Each point is one candidate model. Lower validation error is better; training and validation gaps can reveal overfitting.")
    if inspect:
        cycles = sorted(int(value) for value in history["Cycle"].dropna().unique())
        if len(cycles) == 1:
            cycle = cycles[0]
            st.caption(f"Only candidate cycle recorded: {cycle}.")
        else:
            cycle = st.select_slider("Inspect candidate cycle", options=cycles,
                                     value=cycles[0], key=f"inspect_cycle_{key}")
        row = history.loc[history["Cycle"] == cycle].iloc[-1]
        a, b, c = st.columns(3)
        a.metric("Training MSE", f"{row['Training MSE']:.4g}")
        b.metric("Validation MSE", f"{row['Validation MSE']:.4g}")
        c.metric("Training time", f"{row['Training seconds']:.1f} s" if row["Training seconds"] else "—")
        if row["Configuration"]:
            with st.expander(f"Cycle {cycle} · candidate settings"):
                st.json(row["Configuration"])


def render_run_result(entry: dict, key: str, requested_cycles: int | None = None):
    """Show a compact, interactive evidence dashboard directly in the chat."""
    status = str(entry.get("status") or "unknown").replace("_", " ").title()
    history = _cycle_history(entry)
    actual_cycles = int(entry.get("cycles_run") or (int(history["Cycle"].max()) if not history.empty else 0))
    requested = max(1, int(requested_cycles or actual_cycles or 1))
    best_mse = entry.get("best_mse")
    best_rmse = entry.get("best_rmse")
    latency = entry.get("latency_ms")
    with st.container(border=True, key=f"run_dashboard_{key}"):
        st.markdown(f"#### :material/monitoring: Run snapshot · {status}")
        a, b, c, d = st.columns(4)
        a.metric("Best validation MSE", f"{float(best_mse):.4g}" if best_mse is not None and _finite(best_mse) else "—")
        b.metric("Best validation RMSE", f"{float(best_rmse):.4g}" if best_rmse is not None and _finite(best_rmse) else "—")
        c.metric("Inference latency", f"{float(latency):.3g} ms" if latency is not None and _finite(latency) else "—")
        d.metric("Search cycles", f"{actual_cycles} / {requested}")
        training_seconds = entry.get("training_seconds")
        elapsed_seconds = entry.get("elapsed_seconds")
        note = []
        if training_seconds is not None and _finite(training_seconds):
            note.append(f"{float(training_seconds):.1f} s model training")
        if elapsed_seconds is not None and _finite(elapsed_seconds):
            note.append(f"{float(elapsed_seconds):.1f} s total run time")
        if entry.get("run_mode"):
            note.append(f"{str(entry['run_mode']).title()} search")
        st.caption("  ·  ".join(note) if note else "Explore the run evidence below.")
        views = ["Performance", "State accuracy", "Run plots"]
        selection = st.segmented_control("Explore this run", views, default="Performance",
                                         key=f"run_view_{key}", label_visibility="collapsed", width="stretch")
        if selection == "Performance":
            _cycle_curve(history, key)
        elif selection == "State accuracy":
            comparison = analysis.compare_run_states(entry.get("run_dir", ""))
            if comparison.get("rows"):
                render_state_comparison(comparison)
            else:
                st.info("Per-state held-out scores were not saved for this run.")
        else:
            plots = hist.figures_of(entry)
            if not plots:
                st.info("No diagnostic images were saved for this run.")
            else:
                selected = st.selectbox("Diagnostic plot", plots, format_func=_plot_label,
                                        key=f"run_plot_{key}")
                st.image(selected, width="stretch")


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def render_answer_visuals(question: str, key: str, *, dataset: dict | None = None,
                          run_dir: str | None = None):
    """Attach useful plots to a model answer, choosing evidence from this run or dataset."""
    if run_dir:
        entry = {"run_dir": run_dir}
        manifest_path = Path(run_dir) / "run_manifest.json"
        try:
            import json
            entry.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        entry["run_dir"] = run_dir
        history = _cycle_history(entry)
        comparison = analysis.compare_run_states(run_dir)
        plots = hist.figures_of(entry)
        q = question.casefold()
        preferred = "State accuracy" if "state" in q or re.search(r"which .*predict.*(best|better|worst)|state.*(best|better|worst)", q) else (
            "Performance" if any(word in q for word in ("cycle", "training", "converg", "overfit", "fail", "score", "validation")) else "Run plots")
        with st.expander("Visual evidence for this answer", icon=":material/insights:"):
            choices = ["Performance"]
            if comparison.get("rows"):
                choices.append("State accuracy")
            if plots:
                choices.append("Saved plots")
            selection = st.segmented_control("Evidence view", choices, default=preferred if preferred in choices else choices[0],
                                             key=f"answer_visual_{key}", label_visibility="collapsed")
            if selection == "Performance":
                _cycle_curve(history, f"answer_{key}", inspect=False)
            elif selection == "State accuracy":
                render_state_comparison(comparison)
            elif plots:
                selected = st.selectbox("Relevant run image", plots, format_func=_plot_label,
                                        key=f"answer_plot_{key}")
                st.image(selected, width="stretch")
    elif dataset:
        profile = analysis.ensure_profile(dataset)
        states, inputs = [r["name"] for r in profile.get("states", [])], [r["name"] for r in profile.get("inputs", [])]
        if not states:
            return
        preferred = "Input vs state" if inputs and any(word in question.casefold() for word in ("input", "excitation", "drive", "cause", "relationship")) else "Signal"
        with st.expander("Visual evidence from your data", icon=":material/insights:"):
            options = ["Signal", "Distribution"] + (["Input vs state"] if inputs else [])
            selection = st.segmented_control("Dataset view", options, default=preferred,
                                             key=f"dataset_visual_{key}", label_visibility="collapsed")
            if selection == "Signal":
                name = st.selectbox("Measured signal", states, key=f"answer_signal_{key}")
                _signal(dataset, name, key, height=225)
            elif selection == "Distribution":
                name = st.selectbox("State distribution", states, key=f"answer_distribution_{key}")
                try:
                    frame = _preview(dataset["path"], dataset["sha256"], dataset["name"])
                    values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy()
                    if len(values):
                        counts, edges = np.histogram(values, bins=min(32, max(8, int(np.sqrt(len(values))))))
                        dist = pd.DataFrame({"Value range": [f"{edges[i]:.3g}–{edges[i+1]:.3g}" for i in range(len(counts))],
                                             "Samples": counts})
                        st.bar_chart(dist, x="Value range", y="Samples", height=225)
                except (OSError, ValueError, KeyError) as exc:
                    st.warning(str(exc))
            else:
                a, b = st.columns(2)
                input_name = a.selectbox("Input", inputs, key=f"answer_input_{key}")
                state_name = b.selectbox("State", states, key=f"answer_state_{key}")
                try:
                    frame = _preview(dataset["path"], dataset["sha256"], dataset["name"])
                    points = frame[[input_name, state_name]].apply(pd.to_numeric, errors="coerce").dropna()
                    st.scatter_chart(points, x=input_name, y=state_name, height=245)
                    if len(points) > 1:
                        corr = points[input_name].corr(points[state_name])
                        st.caption(f"Pearson correlation in the recording: {corr:.3f}. Correlation alone does not establish causation or identify dynamics.")
                except (OSError, ValueError, KeyError) as exc:
                    st.warning(str(exc))


def render_dataset(dataset: dict, key: str):
    """Show measured evidence in the message, without implying model performance."""
    profile = analysis.ensure_profile(dataset)
    interval = profile.get("median_dt")
    duration = profile.get("observed_duration")
    with st.container(border=True, key=f"data_review_{key}"):
        st.caption("MEASURED DATA · BEFORE MODEL TRAINING")
        a, b, c = st.columns(3)
        a.metric("Samples", f"{profile['rows']:,}")
        b.metric("Observed time", f"{duration:.3g} s" if duration is not None else "Unknown")
        c.metric("Median interval", f"{interval:.4g} s" if interval is not None else "Unknown")
        states = [item["name"] for item in profile["states"]]
        if states:
            selected = st.selectbox("View measured state", states, key=f"data_signal_{key}")
            _signal(dataset, selected, key)
            with st.expander("More signal checks"):
                rows = []
                for item in profile["states"]:
                    baseline = item.get("persistence") or {}
                    ratio = baseline.get("over_test_std")
                    rows.append({"State": item["name"], "Minimum": item["min"], "Maximum": item["max"],
                                 "Std. dev.": item["std"],
                                 "One-step persistence / test std.": ratio})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                             height=min(320, 42 + 35 * len(rows)))
                st.caption("The persistence score is a data-only baseline on the final 10% of samples, not a trained-model score.")
        if profile.get("inputs"):
            varying = sum(not item["constant"] for item in profile["inputs"])
            st.caption(f"Inputs varying: {varying}/{len(profile['inputs'])} · Variation alone does not prove adequate excitation.")


def _choose(label: str, value: str, key: str, *, primary=False):
    if st.button(label, key=key, type="primary" if primary else "secondary"):
        st.session_state["conversation_pending"] = value


def render_setup_question(chat: dict, key: str, active: bool):
    if not active:
        return
    stage = (chat.get("setup") or {}).get("stage")
    dataset = chat.get("dataset") or {}
    profile = dataset.get("analysis") or {}
    with st.container(border=True, key=f"setup_{key}"):
        st.caption("DATA ASSUMPTIONS · ANSWER HERE OR TYPE IN CHAT")
        if stage == "trajectory":
            left, right = st.columns(2)
            with left:
                st.markdown("**One continuous run**")
                single = pd.DataFrame({"Time": range(11), "Measured state": [0, .5, 1, 1.3, 1.5, 1.6, 1.7, 2, 2.2, 2.4, 2.5]})
                st.line_chart(single, x="Time", y="Measured state", height=145)
                st.caption("One experiment continues without a restart.")
            with right:
                st.markdown("**Several stacked runs**")
                stacked = pd.DataFrame({"Time": range(11), "Measured state": [0, .6, 1.2, 1.8, 2.2, 0, .7, 1.3, 1.8, 0, .8]})
                st.line_chart(stacked, x="Time", y="Measured state", height=145)
                st.caption("Separate experiments are joined in one file.")
            with st.container(horizontal=True, gap="small"):
                _choose("One continuous run", "One continuous run", f"one_{key}")
                _choose("Several stacked runs", "Several stacked runs", f"stacked_{key}", primary=bool(profile.get("time_reset_count")))
        elif stage == "split":
            with st.container(horizontal=True, gap="small"):
                _choose("Detect boundaries", "Detect boundaries", f"auto_split_{key}", primary=True)
                _choose("I know the timestamps", "I know split timestamps", f"known_split_{key}")
        elif stage == "split_times":
            with st.form(f"split_times_{key}", border=False):
                times = st.text_input("New-run timestamps (seconds)", placeholder="12.5, 25.0")
                if st.form_submit_button("Use these timestamps", type="primary"):
                    st.session_state["conversation_pending"] = f"Split times: {times}"
        elif stage == "angle":
            found = profile.get("detected_wrap_states") or []
            names = profile.get("angle_named_states") or []
            if found:
                st.markdown("**Possible wrap in your data**")
                _signal(dataset, found[0], f"angle_{key}", height=165)
                st.caption("A jump from near +π to −π appears in the measured signal. Confirm whether it is really an angle in radians.")
            elif names:
                st.markdown("**Angle-like column in your data**")
                _signal(dataset, names[0], f"angle_{key}", height=165)
                st.caption("Its name suggests an angle, but this recording shows no ±π crossing. The physical meaning is yours to confirm.")
            else:
                example = pd.DataFrame({"Time": range(7), "Stored angle (rad)": [2.1, 2.5, 2.9, 3.14, -3.14, -2.8, -2.4]})
                st.line_chart(example, x="Time", y="Stored angle (rad)", height=165)
                st.caption("Example: a wrapped angle jumps from +π to −π while the physical rotation continues.")
            with st.container(horizontal=True, gap="small"):
                if found:
                    _choose("Wrap detected states", "Wrap detected states", f"wrap_found_{key}", primary=True)
                _choose("No wrapped states", "No wrapped states", f"no_wrap_{key}")
                _choose("Choose states", "Choose angle states", f"choose_wrap_{key}")
                _choose("Auto-detect during run", "Auto-detect wrapped angles", f"auto_wrap_{key}")
        elif stage == "angle_select":
            states = dataset.get("states") or []
            selected = st.multiselect("Angle states in radians that wrap at ±π", states, key=f"angle_names_{key}")
            if st.button("Use selected states", type="primary", disabled=not selected, key=f"confirm_angles_{key}"):
                st.session_state["conversation_pending"] = "Wrap states: " + ", ".join(selected)


def render_state_comparison(comparison: dict):
    rows = comparison.get("rows") or []
    if not rows:
        return
    st.caption(f"SAVED RUN EVIDENCE · verification_summary.json · {comparison.get('aligned_samples') or 'unknown'} aligned held-out samples")
    plot = pd.DataFrame([{"State": r["state"], "Normalized RMSE (% of test variation)": 100*r["normalized_rmse"]}
                         for r in rows if r["normalized_rmse"] is not None])
    if not plot.empty:
        st.bar_chart(plot, x="State", y="Normalized RMSE (% of test variation)", height=220)
    table = pd.DataFrame([{"State": r["state"], "RMSE": r["rmse"], "MAE": r["mae"],
                           "RMSE / test std.": r["normalized_rmse"], "Samples": r["valid_samples"]} for r in rows])
    st.dataframe(table, hide_index=True, width="stretch", height=min(310, 42 + 35*len(rows)))
    protocol = comparison.get("protocol") or {}
    if protocol.get("initialization"):
        st.caption("Rollout protocol: " + protocol["initialization"])
