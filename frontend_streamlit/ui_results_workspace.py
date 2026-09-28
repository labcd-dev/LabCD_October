"""Artifact previews and comparisons, using saved summaries rather than models."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from . import ui_history as hist
except ImportError:
    import ui_history as hist


def run_id(entry: dict) -> str:
    return str(Path(entry["run_dir"]).resolve())


def run_label(entry: dict) -> str:
    stamp = entry.get("started")
    date = stamp.strftime("%d %b %H:%M:%S") if stamp else entry.get("name", "")
    return f"{hist.display_name(entry)} · {date}" + (" · archived" if entry.get("archived") else "")


def literal(value) -> str:
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|:~-])", r"\\\1", str(value).replace("\n", " "))


def number(entry: dict, field: str):
    value = entry.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def duration(seconds) -> str:
    if seconds is None:
        return "Not recorded"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def compare_rows(left: dict, right: dict) -> list[dict]:
    """Keep absent/NaN metrics distinct from valid zero measurements."""
    specs = [
        ("Score / 100", "success_score", lambda x: f"{x:.2f}", True),
        ("Validation MSE", "best_mse", lambda x: f"{x:.3e}", False),
        ("Validation RMSE", "best_rmse", lambda x: f"{x:.3e}", False),
        ("Training time", "training_seconds", duration, False),
        ("Total run time", "elapsed_seconds", duration, False),
        ("Inference latency", "latency_ms", lambda x: f"{x:.3f} ms", False),
        ("Cycles", "cycles_run", lambda x: f"{x:g}", None),
    ]
    rows = []
    for label, field, formatter, higher in specs:
        a, b = number(left, field), number(right, field)
        change = "—"
        if a is not None and b is not None:
            if higher is None:
                change = f"{b-a:+g}"
            elif field == "success_score":
                change = f"{b-a:+.2f} points"
            elif a == b:
                change = "Same"
            elif a == 0:
                change = "Higher" if b > a else "Lower"
            else:
                change = f"{abs((b-a)/a)*100:.1f}% {'higher' if b>a else 'lower'}"
        rows.append({"Metric": label, "Run A": formatter(a) if a is not None else "Not recorded",
                     "Run B": formatter(b) if b is not None else "Not recorded", "B vs A": change})
    for label, field in (("Architecture", "architecture"), ("Activation", "activation"), ("Model status", "model_status")):
        rows.append({"Metric": label, "Run A": left.get(field) or "Not recorded",
                     "Run B": right.get(field) or "Not recorded", "B vs A": "—"})
    for label, field in (("Hidden layers", "hidden_layers"), ("Learning rate", "learning_rate"), ("Dropout", "dropout_rate")):
        rows.append({"Metric": label, "Run A": str((left.get("best_config") or {}).get(field, "Not recorded")),
                     "Run B": str((right.get("best_config") or {}).get(field, "Not recorded")), "B vs A": "—"})
    return rows


def select_run(label, entries, *, key, preferred=None):
    by_id = {run_id(entry): entry for entry in entries}
    ids = list(by_id)
    if not ids:
        return None
    remembered = st.session_state.get(f"{key}_choice", preferred)
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = remembered if remembered in by_id else ids[0]

    def remember():
        st.session_state[f"{key}_choice"] = st.session_state[key]

    chosen = st.selectbox(label, ids, format_func=lambda identity: run_label(by_id[identity]),
                          key=key, on_change=remember)
    return by_id.get(chosen)


def _remember_preview_type():
    st.session_state["preview_type_choice"] = st.session_state["preview_type"]


def render_compare(output_dir, preferred=None) -> None:
    st.markdown("### Compare two runs")
    st.caption("Choose a baseline and a candidate. Scores are better when higher; errors are better when lower.")
    archived = st.toggle("Include archived runs", key="compare_archived")
    entries = [e for e in hist.load_history(output_dir, limit=None)
               if e.get("complete") and (archived or not e.get("archived"))]
    if len(entries) < 2:
        st.info("Complete two runs to compare their results." if not archived else "There are fewer than two completed runs available.")
        return
    columns = st.columns(2, gap="medium")
    with columns[0]:
        left = select_run("Run A · baseline", entries, key="compare_a", preferred=preferred)
    other = [entry for entry in entries if run_id(entry) != run_id(left)]
    with columns[1]:
        right = select_run("Run B · candidate", other, key="compare_b")
    if left.get("dataset") and right.get("dataset") and left["dataset"] != right["dataset"]:
        st.info("These runs use different dataset paths. Keep that difference in mind when comparing their errors.")
    for column, entry, letter in zip(columns, (left, right), ("A", "B")):
        with column.container(border=True):
            st.markdown(f"**Run {letter} · {literal(hist.display_name(entry))}**")
            score = number(entry, "success_score")
            st.metric("Score / 100", f"{score:.2f}" if score is not None else "Not recorded")
            st.caption(f"{entry.get('architecture') or 'Architecture not recorded'} · {entry.get('activation') or 'Activation not recorded'} · {(entry.get('best_config') or {}).get('hidden_layers', 'Layers not recorded')}")
    st.dataframe(pd.DataFrame(compare_rows(left, right)), hide_index=True, width="stretch")
    st.caption("Training time measures model training and validation. Total run time also includes agents and reporting. Older runs may not have separate training timing.")
    with st.expander("Full model configurations", icon=":material/settings:"):
        config_columns = st.columns(2)
        for column, entry, letter in zip(config_columns, (left, right), ("A", "B")):
            with column:
                st.caption(f"Run {letter}")
                st.json(entry.get("best_config") or {})


def render_panel(output_dir, *, viewed=None, result=None, running=False) -> None:
    with st.container(border=True, key="results_preview_panel"):
        st.markdown("##### Results preview")
        entries = hist.load_history(output_dir, limit=None)
        preferred = run_id(viewed) if viewed else None
        if result is not None and not running and result.run_dir:
            current = result.to_dict()
            current.update(complete=result.status == "completed", name=Path(result.run_dir).name)
            if not any(run_id(e) == run_id(current) for e in entries):
                entries.insert(0, current)
            if preferred is None:
                preferred = run_id(current)
        if viewed and not any(run_id(e) == preferred for e in entries):
            entries.insert(0, viewed)
        if not entries:
            st.caption("Open a saved run or finish a new run to preview its plots, report, and code.")
            return
        entry = select_run("Preview run", entries, key="preview_run", preferred=preferred)
        st.caption(hist.summarise(entry))
        view = st.segmented_control("Preview type", ["Plots", "Report", "Code"],
                                    default=st.session_state.get("preview_type_choice") or "Plots",
                                    key="preview_type", on_change=_remember_preview_type,
                                    label_visibility="collapsed") or "Plots"
        scope = hashlib.sha1(run_id(entry).encode()).hexdigest()[:12]
        files = hist.existing_files(entry)
        with st.container(height=580, border=False):
            if view == "Plots":
                figures = hist.figures_of(entry)
                if not figures:
                    st.info("No plots are available for this run.")
                    return
                selected = st.selectbox("Plot", figures, format_func=lambda p: Path(p).stem.replace("_", " "), key=f"preview_plot_{scope}")
                try:
                    st.image(selected, width="stretch")
                    st.download_button("Download plot", Path(selected).read_bytes(), file_name=Path(selected).name,
                                       mime="image/png", key=f"preview_plot_download_{scope}", width="stretch")
                except OSError:
                    st.info("This plot is no longer available.")
            elif view == "Report":
                pdf = files.get("PDF report")
                if pdf:
                    try:
                        data = Path(pdf).read_bytes()
                        st.download_button("Download report", data, file_name=Path(pdf).name, mime="application/pdf",
                                           key=f"preview_report_download_{scope}", width="stretch")
                        if importlib.util.find_spec("streamlit_pdf"):
                            st.pdf(data, height=440, key=f"preview_pdf_{scope}")
                    except OSError:
                        st.info("This report is no longer available.")
                if entry.get("abstract"):
                    with st.expander("Report summary", expanded=not pdf):
                        st.markdown(literal(entry["abstract"]))
                        if entry.get("conclusion"):
                            st.markdown(literal(entry["conclusion"]))
                elif not pdf:
                    st.info("No report is available for this run.")
            else:
                codes = {label: path for label, path in files.items() if Path(path).suffix == ".py"}
                if not codes:
                    st.info("No generated code is available for this run.")
                    return
                selected = st.selectbox("Generated file", list(codes), key=f"preview_code_{scope}")
                path = Path(codes[selected])
                try:
                    data = path.read_bytes()
                    st.caption(path.name)
                    st.download_button("Download code", data, file_name=path.name, mime="text/x-python",
                                       key=f"preview_code_download_{scope}", width="stretch")
                    code = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                    st.code(code, language="python", line_numbers=True)
                except OSError:
                    st.info("This generated file is no longer available.")
