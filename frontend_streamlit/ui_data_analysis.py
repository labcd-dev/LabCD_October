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


def _wrap_angle_illustration():
    """Explain the difference between a physical turn and its wrapped file value."""
    st.markdown('''
<section class="wrap-angle-card" aria-label="Illustration of angle wrapping">
  <div class="wrap-angle-heading">ONE ROTATION · TWO WAYS TO STORE IT</div>
  <p>The physical angle can keep turning smoothly. If the file stores angles only between −π and +π, its number jumps at the boundary.</p>
  <svg class="wrap-angle-svg" viewBox="0 0 960 255" role="img" aria-label="The physical angle rises smoothly through pi, while the stored angle jumps from plus pi to minus pi and continues">
    <defs>
      <marker id="wrap-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#ffb86b"/></marker>
    </defs>
    <rect x="10" y="10" width="458" height="231" rx="15" fill="#121b1d" stroke="#30413f"/>
    <rect x="492" y="10" width="458" height="231" rx="15" fill="#121b1d" stroke="#30413f"/>
    <text x="30" y="38" fill="#e7f4f0" font-size="14" font-weight="700">PHYSICAL ANGLE</text>
    <text x="30" y="57" fill="#91aaa5" font-size="12">The rotation continues through +π</text>
    <text x="512" y="38" fill="#e7f4f0" font-size="14" font-weight="700">VALUE STORED IN THE FILE</text>
    <text x="512" y="57" fill="#91aaa5" font-size="12">The value is kept between −π and +π</text>
    <path d="M67 88H442M67 132H442M67 176H442M67 205H442" stroke="#2b3a3b" stroke-width="1"/>
    <path d="M548 88H923M548 132H923M548 176H923M548 205H923" stroke="#2b3a3b" stroke-width="1"/>
    <path d="M67 132H442" stroke="#617a76" stroke-dasharray="4 5"/>
    <path d="M548 88H923M548 205H923" stroke="#617a76" stroke-dasharray="4 5"/>
    <text x="24" y="92" fill="#a9bbb7" font-size="11">+π</text>
    <text x="43" y="136" fill="#829894" font-size="11">0</text>
    <text x="513" y="92" fill="#a9bbb7" font-size="11">+π</text>
    <text x="513" y="209" fill="#a9bbb7" font-size="11">−π</text>
    <path d="M69 173C130 164 167 145 223 126S322 102 440 72" fill="none" stroke="#66dbc5" stroke-width="3.4" stroke-linecap="round"/>
    <path d="M550 173C610 164 650 145 704 126S777 99 800 89L802 204C840 194 878 181 922 167" fill="none" stroke="#66dbc5" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="800" cy="89" r="4.5" fill="#ffb86b"/><circle cx="802" cy="204" r="4.5" fill="#ffb86b"/>
    <path d="M812 98L812 190" stroke="#ffb86b" stroke-width="1.6" stroke-dasharray="4 4" marker-end="url(#wrap-arrow)"/>
    <text x="829" y="136" fill="#ffc98e" font-size="11">wrap</text>
    <text x="252" y="226" fill="#718985" font-size="11">TIME →</text>
    <text x="734" y="226" fill="#718985" font-size="11">TIME →</text>
  </svg>
  <div class="wrap-angle-note"><span>How to decide</span> Choose wrapping only if that state is an angle in radians stored modulo 2π. A column name alone is not proof.</div>
</section>
<style>
.wrap-angle-card{margin:14px 0 12px;padding:16px 17px 13px;border:1px solid #354b48;border-radius:15px;background:linear-gradient(135deg,#172321 0%,#171b20 68%,#1a1c23 100%);box-shadow:inset 0 1px #ffffff08}
.wrap-angle-heading{font:600 10px/1.5 ui-monospace,monospace;letter-spacing:.14em;color:#8edbcb}
.wrap-angle-card p{margin:5px 0 8px;color:#b8c7c4;font-size:12px;line-height:1.55}
.wrap-angle-svg{display:block;width:100%;height:auto;overflow:visible}
.wrap-angle-note{display:flex;gap:9px;align-items:flex-start;margin-top:5px;color:#99aaa7;font-size:11px;line-height:1.5}
.wrap-angle-note span{flex:none;color:#dbede9;font-weight:650}
@media(max-width:620px){.wrap-angle-card{padding:12px 9px;overflow-x:auto}.wrap-angle-card p{font-size:11px}.wrap-angle-svg{min-width:560px}}
</style>
''', unsafe_allow_html=True)


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


def _trajectory_illustration():
    """Contrast one uninterrupted recording with separate, restarted runs."""
    st.markdown('''
<section class="trajectory-example" aria-label="How to choose between one continuous recording and separate runs">
  <style>
    .trajectory-example{margin:4px 0 8px;padding:14px;border:1px solid #354744;border-radius:16px;
      background:linear-gradient(135deg,#172321 0%,#191c21 65%,#1d1d24 100%);box-shadow:inset 0 1px #ffffff08}
    .trajectory-example-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin:0 2px 8px}
    .trajectory-example-title{color:#e7f1ed;font-size:14px;font-weight:650;letter-spacing:-.01em}
    .trajectory-example-subtitle{margin-top:3px;color:#9eafaa;font-size:11px;line-height:1.5}
    .trajectory-example-tag{flex:none;padding:4px 8px;border:1px solid #3d5851;border-radius:999px;
      background:#1b302b;color:#a9e0ce;font:600 9px ui-monospace,monospace;letter-spacing:.08em}
    .trajectory-example-svg{display:block;width:100%;height:auto}
    .trajectory-example-note{display:flex;gap:8px;align-items:flex-start;margin:6px 2px 0;padding:9px 11px;
      border-radius:10px;background:#222b2a;color:#b8c9c4;font-size:11px;line-height:1.5}
    .trajectory-example-note strong{flex:none;color:#e0eee9}
    @media(max-width:680px){.trajectory-example{padding:11px 8px;overflow-x:auto}.trajectory-example-head{gap:7px}.trajectory-example-subtitle{font-size:10px}.trajectory-example-svg{min-width:740px;max-width:none}}
    @media(prefers-reduced-motion:reduce){.trajectory-example *{scroll-behavior:auto!important;transition:none!important}}
  </style>
  <div class="trajectory-example-head">
    <div><div class="trajectory-example-title">Which picture matches your data?</div>
      <div class="trajectory-example-subtitle">A trajectory is one uninterrupted recording of the system.</div></div>
    <span class="trajectory-example-tag">DATA CHECK</span>
  </div>
  <svg class="trajectory-example-svg" viewBox="0 0 1000 302" role="img" aria-label="One continuous recording is one unbroken signal over time. Separate runs are shown as three independent traces, each restarting at its own time zero.">
    <defs>
      <linearGradient id="traj-single-line" x1="0" x2="1"><stop offset="0" stop-color="#59c8b2"/><stop offset="1" stop-color="#9be4c8"/></linearGradient>
      <linearGradient id="traj-run-one" x1="0" x2="1"><stop offset="0" stop-color="#64d7c2"/><stop offset="1" stop-color="#92e4ce"/></linearGradient>
      <linearGradient id="traj-run-two" x1="0" x2="1"><stop offset="0" stop-color="#8c9fff"/><stop offset="1" stop-color="#b6bcff"/></linearGradient>
      <linearGradient id="traj-run-three" x1="0" x2="1"><stop offset="0" stop-color="#e7b56f"/><stop offset="1" stop-color="#f1d39a"/></linearGradient>
      <marker id="traj-time-arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#829691"/></marker>
    </defs>
    <rect x="8" y="8" width="480" height="280" rx="14" fill="#111a1c" stroke="#344744"/>
    <rect x="512" y="8" width="480" height="280" rx="14" fill="#111a1c" stroke="#344744"/>
    <text x="30" y="38" fill="#e4f1ec" font-size="13" font-weight="700">ONE CONTINUOUS RUN</text>
    <text x="30" y="57" fill="#91a9a3" font-size="11">One uninterrupted experiment from start to finish.</text>
    <rect x="365" y="23" width="98" height="22" rx="11" fill="#1b302b" stroke="#3c6658"/>
    <text x="414" y="38" text-anchor="middle" fill="#a9e0ce" font-size="9" font-weight="700">ONE TRAJECTORY</text>
    <path d="M48 92H458M48 130H458M48 168H458M48 206H458" stroke="#304240" stroke-width="1"/>
    <path d="M49 218H457" stroke="#667c76" stroke-width="1.2" marker-end="url(#traj-time-arrow)"/>
    <path d="M50 195C90 185 115 161 151 156S207 162 243 143S307 125 345 113S408 89 454 82" fill="none" stroke="url(#traj-single-line)" stroke-width="4" stroke-linecap="round"/>
    <circle cx="50" cy="195" r="5" fill="#b1f1dd"/><circle cx="454" cy="82" r="5" fill="#b1f1dd"/>
    <text x="49" y="238" fill="#a7b9b3" font-size="10">start</text><text x="425" y="238" fill="#a7b9b3" font-size="10">finish</text>
    <text x="247" y="258" text-anchor="middle" fill="#91aaa3" font-size="10">time keeps moving →</text>
    <text x="532" y="38" fill="#e4f1ec" font-size="13" font-weight="700">SEVERAL STACKED RUNS</text>
    <text x="532" y="57" fill="#91a9a3" font-size="11">The system restarts; separate recordings share one file.</text>
    <rect x="864" y="23" width="101" height="22" rx="11" fill="#20263a" stroke="#4b557c"/>
    <text x="914" y="38" text-anchor="middle" fill="#c3c9ff" font-size="9" font-weight="700">3 TRAJECTORIES</text>
    <rect x="530" y="70" width="444" height="58" rx="9" fill="#182321" stroke="#2d403c"/>
    <rect x="530" y="137" width="444" height="58" rx="9" fill="#1b1e2a" stroke="#343b56"/>
    <rect x="530" y="204" width="444" height="58" rx="9" fill="#24211b" stroke="#4b4130"/>
    <text x="544" y="94" fill="#bdece0" font-size="10" font-weight="700">RUN 01</text><text x="544" y="111" fill="#7f9790" font-size="9">t: 0 → T</text>
    <text x="544" y="161" fill="#c6cdff" font-size="10" font-weight="700">RUN 02</text><text x="544" y="178" fill="#858ba9" font-size="9">t: 0 → T</text>
    <text x="544" y="228" fill="#f0d6a5" font-size="10" font-weight="700">RUN 03</text><text x="544" y="245" fill="#a99b7c" font-size="9">t: 0 → T</text>
    <path d="M626 111H950M626 178H950M626 245H950" stroke="#596e68" stroke-width="1" marker-end="url(#traj-time-arrow)"/>
    <path d="M627 105C659 102 678 84 714 86S762 110 801 100S866 81 916 88S936 94 949 91" fill="none" stroke="url(#traj-run-one)" stroke-width="3.2" stroke-linecap="round"/>
    <path d="M627 171C663 166 690 148 723 153S780 180 813 169S872 151 916 157S937 162 949 158" fill="none" stroke="url(#traj-run-two)" stroke-width="3.2" stroke-linecap="round"/>
    <path d="M627 239C660 235 687 216 722 220S777 248 813 237S872 217 916 224S936 230 949 225" fill="none" stroke="url(#traj-run-three)" stroke-width="3.2" stroke-linecap="round"/>
    <circle cx="627" cy="105" r="3.5" fill="#b1f1dd"/><circle cx="627" cy="171" r="3.5" fill="#c6cdff"/><circle cx="627" cy="239" r="3.5" fill="#f0d6a5"/>
  </svg>
  <div class="trajectory-example-note"><strong>Key difference</strong><span>Each separate run starts a new trajectory. LabCD keeps those boundaries when preparing training and test data instead of connecting the traces.</span></div>
</section>
''', unsafe_allow_html=True)


def render_setup_question(chat: dict, key: str, active: bool):
    if not active:
        return
    stage = (chat.get("setup") or {}).get("stage")
    dataset = chat.get("dataset") or {}
    profile = dataset.get("analysis") or {}
    with st.container(border=True, key=f"setup_{key}"):
        st.caption("DATA ASSUMPTIONS · ANSWER HERE OR TYPE IN CHAT")
        if stage == "trajectory":
            _trajectory_illustration()
            resets = int(profile.get("time_reset_count", 0) or 0)
            if resets:
                with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                    st.badge(f"{resets} time reset{'s' if resets != 1 else ''} found",
                             icon=":material/restart_alt:", color="orange")
                    st.caption("This can indicate stacked runs. Please choose based on how you recorded the data.")
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
                candidates = ", ".join(names[:8])
                extra = f" Possible angle-like columns: {candidates}." if candidates else ""
                st.caption("The name suggests an angle, but this recording shows no ±π crossing. Confirm the physical meaning before selecting a state." + extra +
                           " A rate or derivative should not be selected just because its name contains pitch or yaw.")
            else:
                st.caption("I did not identify a likely wrapped angle in this file. This illustration shows what wrapping would look like.")
            _wrap_angle_illustration()
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
    ranked = [r for r in rows if r["normalized_rmse"] is not None]
    if ranked:
        best = ranked[0]
        st.markdown(f"**Lowest normalized error: `{best['state']}`** · "
                    f"{100 * best['normalized_rmse']:.2f}% of held-out variation")
    table_rows = []
    rank = 0
    for row in rows:
        if row["normalized_rmse"] is not None:
            rank += 1
            rank_label = str(rank)
            normalized = f"{100 * row['normalized_rmse']:.2f}%"
        else:
            rank_label, normalized = "—", "—"
        table_rows.append({"Rank": rank_label, "State": row["state"],
                           "Normalized RMSE · lower is better": normalized,
                           "RMSE": f"{row['rmse']:.4g}", "MAE": f"{row['mae']:.4g}",
                           "Samples": row["valid_samples"]})
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch",
                 height=min(310, 42 + 35 * len(table_rows)))
    st.caption("States are ranked by normalized RMSE on the held-out rollout. Lower means closer predictions relative to that state's variation.")
    protocol = comparison.get("protocol") or {}
    if protocol.get("initialization"):
        st.caption("Rollout protocol: " + protocol["initialization"])
