#!/usr/bin/env python3
"""
LabCD · System Identification — AgentSysID reference UI.

Run from the repository root:

    PYTHONPATH=. streamlit run frontend_streamlit/agent_sysid_app.py --server.port 8504

or use the launcher:

    PYTHONPATH=. python frontend_streamlit/run_agent_sysid_ui.py

The UI is deliberately thin: every control maps to a field on
``backend_core.AgentSysID.pipeline.SysIDOptions`` and the run itself is
``run_pipeline`` on a worker thread. No training, agent or reporting logic is
duplicated here — this file only collects options, streams events, and renders
the artefacts the core produced.

Layout
------
  Sidebar   run history (persisted via each run's run_manifest.json)
  Configure every option the interactive CLI asks for
  Monitor   live stage rail, progress, per-cycle metrics, charts, agent log
  Results   score, deliverables, figures, manuscript
"""

from __future__ import annotations

import hashlib
from collections import deque
import datetime as dt
import importlib
import json
import math
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# --- Make the repo root importable no matter where streamlit is launched ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend_core.AgentSysID import config as cfg  # noqa: E402
from backend_core.AgentSysID.pipeline import (  # noqa: E402
    STAGES,
    SysIDOptions,
    SysIDResult,
    run_pipeline,
)
from backend_core.AgentSysID.utils import request_stop, reset_stop_flag  # noqa: E402
from backend_core.AgentSysID.agents.run_evidence import write_json, redact  # noqa: E402

import ui_history as hist  # noqa: E402
import ui_activity as activity  # noqa: E402
import ui_results_workspace as workspace  # noqa: E402
import ui_run_chat as run_chat  # noqa: E402
import ui_theme as T  # noqa: E402

# Theme helpers are pure constants/functions; refresh them when the app reruns
# so styling edits appear even when the server keeps imported modules cached.
T = importlib.reload(T)

# ---------------------------------------------------------------------------
# Pristine config snapshot.
#
# A run mutates the config module in place (the Initializer rewrites the search
# bounds, the questionnaire rewrites the angular states, and so on) — that is
# how the core has always worked. The Streamlit server is long-lived, so
# reading cfg.X for a widget default would make the form drift to the previous
# run's values. Snapshot the pristine values once, at import.
# ---------------------------------------------------------------------------
_CFG_DEFAULTS = {
    name: getattr(cfg, name)
    for name in dir(cfg)
    if name.isupper() and not name.startswith("_")
}


def D(name: str, fallback=None):
    """A widget default, taken from the pristine config snapshot."""
    return _CFG_DEFAULTS.get(name, fallback)


UPLOAD_DIR = _REPO_ROOT / ".streamlit_uploads"
DEFAULT_EXAMPLE = _REPO_ROOT / "backend_core/AgentSysID/data/examples/synthetic_oscillator.csv"
SECTIONS = ["Configure", "Monitor", "Results", "Compare", "Ask run"]

st.set_page_config(
    page_title="LabCD · System Identification",
    page_icon="◆",
    layout="wide",
    # History starts open; the native panel icon lets users hide and reopen it.
    initial_sidebar_state="expanded",
)
st.markdown(T.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
class _ThreadRouter:
    """Routes the worker thread's prints to a queue; other threads pass through."""

    def __init__(self, target_thread: int, sink: "queue.Queue[str]", original):
        self._target = target_thread
        self._sink = sink
        self._original = original
        self.captured = deque()
        self.captured_size = 0

    def write(self, text: str) -> int:
        if threading.get_ident() == self._target:
            if text:
                self._sink.put(text)
                self.captured.append(text)
                self.captured_size += len(text)
                while self.captured_size > 2_000_000 and len(self.captured) > 1:
                    self.captured_size -= len(self.captured.popleft())
            return len(text)
        return self._original.write(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False


class PipelineRunner:
    """Runs ``run_pipeline`` on a thread, exposing logs, events and the result."""

    def __init__(self, options: SysIDOptions):
        self.options = options
        self.logs: "queue.Queue[str]" = queue.Queue()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.result: Optional[SysIDResult] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None
        self.run_dir: Optional[Path] = None
        self.last_stage = "Preparing run"

    def _on_event(self, kind: str, payload: Dict[str, Any]) -> None:
        if kind == "run_started":
            self.run_dir = Path(payload["run_dir"])
        elif kind == "stage":
            self.last_stage = payload["name"]
        self.events.put((kind, {**payload, "_timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}))

    def _run(self) -> None:
        original = sys.stdout
        router = _ThreadRouter(threading.get_ident(), self.logs, original)
        sys.stdout = router
        try:
            self.result = run_pipeline(
                self.options, on_event=self._on_event, install_signal_handler=False
            )
        except Exception as exc:  # noqa: BLE001 - surface it in the UI
            import traceback

            self.error = f"{exc}\n\n{traceback.format_exc()}"
            self._on_event("error", {"message": str(exc)})
        finally:
            sys.stdout = original
            run_dir = self.run_dir or (self.result.run_dir if self.result else None)
            if run_dir:
                try:
                    (Path(run_dir) / "runtime_console.log").write_text(redact("".join(router.captured)), encoding="utf-8")
                    write_json(Path(run_dir) / "diagnostic_state.json", {
                        "status": self.result.status if self.result else "failed",
                        "last_stage": self.last_stage, "error": self.error,
                        "message": self.result.message if self.result else "The pipeline raised an exception.",
                    })
                except OSError:
                    pass
            self._on_event("finished", {})

    def start(self) -> None:
        reset_stop_flag()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())


# ---------------------------------------------------------------------------
# Parsing helpers (blank input == keep the agent's / config's choice)
# ---------------------------------------------------------------------------
def _f(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _i(text: str) -> Optional[int]:
    value = _f(text)
    return int(value) if value is not None else None


def _int_list(text: str) -> Optional[List[int]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return [int(float(x.strip())) for x in text.split(",") if x.strip()]
    except ValueError:
        return None


def _float_list(text: str) -> Optional[List[float]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        return None


def _pair(text: str, cast=float):
    values = _float_list(text)
    if not values or len(values) < 2:
        return None, None
    return cast(min(values)), cast(max(values))


def goto(section: str) -> None:
    """
    Navigate programmatically.

    Streamlit forbids writing a widget's own key once that widget has rendered,
    so the section lives in its own state entry and the segmented control gets a
    fresh key each time we navigate — which makes it re-initialise on the new
    default instead of replaying the user's last click.
    """
    st.session_state["section"] = section
    st.session_state["nav_epoch"] = st.session_state.get("nav_epoch", 0) + 1


# ---------------------------------------------------------------------------
# Sidebar — run history
# ---------------------------------------------------------------------------
def _history_key(entry: Dict[str, Any]) -> str:
    return hashlib.sha1(str(Path(entry["run_dir"]).resolve()).encode()).hexdigest()[:12]


def _history_text(text: str) -> str:
    """Display names as literal text inside Streamlit's Markdown labels."""
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|:~-])", r"\\\1", str(text).replace("\n", " "))


def _open_history(entry: Dict[str, Any]) -> None:
    st.session_state["viewing"] = entry
    st.session_state["preview_run"] = workspace.run_id(entry)
    st.session_state["preview_run_choice"] = workspace.run_id(entry)
    goto("Results")


def _toggle_results_panel() -> None:
    st.session_state["results_panel_open"] = not st.session_state.get("results_panel_open", False)


def _ask_run(entry: Dict[str, Any]) -> None:
    st.session_state["diagnostic_run"] = workspace.run_id(entry)
    st.session_state["diagnostic_run_choice"] = workspace.run_id(entry)
    goto("Ask run")


def _save_history_change(entry: Dict[str, Any], field: str, value: Any) -> None:
    if not hist.update_metadata(entry["run_dir"], **{field: value}):
        st.session_state["history_error"] = "Couldn't save this change. Check that the run folder is writable."
        return
    viewed = st.session_state.get("viewing")
    if viewed and viewed.get("run_dir") == entry["run_dir"]:
        st.session_state["viewing"] = {**viewed, field: value}
    message = {
        "pinned": "Run pinned." if value else "Run unpinned.",
        "archived": "Run archived. Find it in Archived runs." if value else "Run restored.",
    }[field]
    st.toast(message)


def _cancel_history_rename() -> None:
    st.session_state.pop("history_renaming", None)


def _begin_history_rename(entry: Dict[str, Any]) -> None:
    st.session_state["history_renaming"] = entry


@st.dialog("Rename run", icon=":material/edit:", on_dismiss=_cancel_history_rename)
def rename_history_run(entry: Dict[str, Any]) -> None:
    run_key = _history_key(entry)
    with st.form(f"history_rename_form_{run_key}"):
        title = st.text_input("Run name", value=hist.display_name(entry), max_chars=120)
        save = st.form_submit_button("Save name", type="primary", width="stretch")
    if save:
        if not title.strip():
            st.error("Enter a name for this run.")
        elif hist.update_metadata(entry["run_dir"], title=title):
            viewed = st.session_state.get("viewing")
            if viewed and viewed.get("run_dir") == entry["run_dir"]:
                st.session_state["viewing"] = {**viewed, "title": title.strip()}
            _cancel_history_rename()
            st.toast("Run renamed.")
            st.rerun()
        else:
            st.error("Couldn't save the name. Check that the run folder is writable.")
    if st.button("Cancel", key=f"history_rename_cancel_{run_key}", width="stretch"):
        _cancel_history_rename()
        st.rerun()


def render_sidebar(output_dir: str, running: bool) -> None:
    sb = st.sidebar
    sb.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:2px 0 12px'>"
        f"<div class='brand-mark' style='width:28px;height:28px;font-size:14px'>L</div>"
        f"<div><div style='font-weight:600;font-size:14px'>LabCD<span style='color:{T.TEXT_FAINT};font-weight:400'>.ai</span></div>"
        f"<div class='brand-sub' style='font-size:9.5px'>System Identification</div></div></div>",
        unsafe_allow_html=True,
    )

    if sb.button("New run", icon=":material/add:", width="stretch", type="primary", disabled=running):
        st.session_state["viewing"] = None
        st.session_state["result"] = None
        st.session_state.update(runner=None, log="", history=[], critic=[], activity=[])
        st.session_state["config_step"] = 1
        goto("Configure")
        st.rerun()

    sb.markdown("<div class='side-heading'>Run history</div>", unsafe_allow_html=True)

    # Load all summaries so old pins and archived runs remain discoverable.
    runs = hist.load_history(output_dir, limit=None)
    if not runs:
        sb.markdown(
            "<div class='hist-empty'>No runs yet.<br>Configure a run and press "
            "<b>Start</b> — it will appear here.</div>",
            unsafe_allow_html=True,
        )
        return

    query = sb.text_input(
        "Search runs", placeholder="Search runs…", key="history_search",
        label_visibility="collapsed",
    )
    archived_count = sum(bool(entry.get("archived")) for entry in runs)
    archived = sb.toggle(f"Archived runs · {archived_count}", key="history_archived")
    if error := st.session_state.pop("history_error", None):
        sb.error(error)

    groups = hist.group_history(runs, archived=archived, query=query)
    if not groups:
        if query.strip():
            sb.caption("No runs match that search.")
        else:
            sb.caption("No archived runs." if archived else "All runs are archived. Switch on Archived runs to restore one.")
        return

    viewed_dir = (st.session_state.get("viewing") or {}).get("run_dir")
    with sb.container(key="history_list", gap="xxsmall"):
        for heading, entries in groups:
            st.caption(heading)
            for entry in entries:
                label = hist.display_name(entry)
                summary = hist.summarise(entry)
                age = hist.relative_age(entry.get("started"))
                try:
                    score = float(entry.get("success_score"))
                except (TypeError, ValueError):
                    score = float("nan")
                badge = f"{score:.0f}" if entry.get("complete") and math.isfinite(score) else "—"
                detail = f"{badge} · {summary} · {age}"
                run_key = _history_key(entry)
                selected = entry["run_dir"] == viewed_dir
                row_key = f"history-row-{'selected-' if selected else ''}{run_key}"
                menu_key = f"history-menu-{run_key}"
                with st.container(key=row_key, gap=None):
                    col_run, col_menu = st.columns([1, 0.16], gap=None, vertical_alignment="center", wrap=False)
                    col_run.button(
                        f"**{_history_text(label)}**\n\n{_history_text(detail)}",
                        key=f"history_open_{run_key}", width="stretch", wrap=True,
                        icon=":material/push_pin:" if entry.get("pinned") else None,
                        help=f"{_history_text(label)} · {_history_text(detail)}",
                        on_click=_open_history, args=(entry,),
                    )
                    with col_menu.popover(
                        f"Actions for {_history_text(label)}", icon=":material/more_horiz:",
                        key=menu_key, width="stretch",
                        help="Run actions", disabled=running,
                    ):
                        st.button(
                            "Unpin" if entry.get("pinned") else "Pin to top",
                            icon=":material/keep_off:" if entry.get("pinned") else ":material/push_pin:",
                            key=f"history_pin_{run_key}", width="stretch",
                            on_click=_save_history_change,
                            args=(entry, "pinned", not entry.get("pinned", False)),
                        )
                        st.button(
                            "Rename", icon=":material/edit:", key=f"history_rename_{run_key}", width="stretch",
                            on_click=_begin_history_rename, args=(entry,),
                        )
                        st.button(
                            "Restore run" if archived else "Archive run",
                            icon=":material/unarchive:" if archived else ":material/archive:",
                            key=f"history_archive_{run_key}", width="stretch",
                            on_click=_save_history_change,
                            args=(entry, "archived", not archived),
                        )
    if renaming := st.session_state.get("history_renaming"):
        rename_history_run(renaming)


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
def render_sysid_intro() -> None:
    """A lightweight animated overview of the system-identification workflow."""
    st.html(
        """
        <section class="sysid-intro" aria-label="System identification overview">
          <div class="sysid-intro-copy">
            <div class="sysid-intro-kicker">LABCD / AGENTSYSID · MODULE OVERVIEW</div>
            <h2 class="sysid-intro-title">From measured data to a model of the system.</h2>
            <p class="sysid-intro-definition">
              <strong>System identification</strong> learns how a real process behaves
              from its inputs and measured responses. The result is a dynamics model
              that can be checked against data the model has never seen.
            </p>
            <div class="sysid-intro-features">
              <span>01 · Inspect data</span>
              <span>02 · Fit dynamics</span>
              <span>03 · Verify + export</span>
            </div>
            <div class="sysid-intro-footnote">
              YOUR DATA → A VERIFIED, REUSABLE SYSTEM MODEL
            </div>
          </div>
          <div class="sysid-visual">
            <div class="scene-header">
              <span class="scene-label"><i class="scene-live-dot"></i> IDENTIFICATION LOOP</span>
              <span class="scene-state">OBSERVE · ESTIMATE · VERIFY</span>
            </div>
            <div class="scene-flow-row" role="img" aria-label="Inputs and measurements feed model fitting, then the model is checked against held-out data">
              <div class="scene-node data-block">
                <div class="scene-node-head"><span>01 / MEASUREMENTS</span><b>DATA</b></div>
                <div class="scene-equation">u(t) + x(t)</div>
                <div class="scene-copy">known input · observed state</div>
                <div class="scene-wave data-wave" aria-hidden="true">
                  <i style="height:28%"></i><i style="height:46%"></i><i style="height:68%"></i><i style="height:42%"></i>
                  <i style="height:82%"></i><i style="height:57%"></i><i style="height:35%"></i><i style="height:72%"></i>
                  <i style="height:50%"></i><i style="height:88%"></i><i style="height:62%"></i><i style="height:40%"></i>
                  <i style="height:76%"></i><i style="height:53%"></i><i style="height:31%"></i><i style="height:66%"></i>
                </div>
                <div class="scene-node-foot">TIME-ALIGNED OBSERVATIONS</div>
              </div>
              <div class="scene-wire" aria-hidden="true"><span class="wire-label">FIT</span><span class="wire-track"><i></i></span><span class="wire-type">x, u</span></div>
              <div class="scene-node model-block">
                <div class="scene-node-head"><span>02 / DYNAMICS</span><b>ESTIMATE</b></div>
                <div class="scene-equation">ẋ = f(x, u; θ)</div>
                <div class="scene-copy">estimate unknown system behavior</div>
                <div class="scene-params"><span>x</span><span>u</span><span class="param-focus">θ</span></div>
                <div class="scene-node-foot">TRAIN ↔ CRITIC · ITERATIVE FIT</div>
              </div>
              <div class="scene-wire scene-wire-test" aria-hidden="true"><span class="wire-label">TEST</span><span class="wire-track"><i></i></span><span class="wire-type">x̂</span></div>
              <div class="scene-node verify-block">
                <div class="scene-node-head"><span>03 / HELD-OUT CHECK</span><b>VERIFY</b></div>
                <div class="scene-equation">predict ↔ observe</div>
                <div class="scene-copy">check error and rollout stability</div>
                <div class="compare-traces" aria-hidden="true">
                  <div class="compare-row"><b class="compare-label">MEASURED</b><div class="compare-bars">
                    <i style="height:55%"></i><i style="height:78%"></i><i style="height:92%"></i><i style="height:62%"></i><i style="height:48%"></i><i style="height:84%"></i><i style="height:68%"></i><i style="height:96%"></i>
                  </div></div>
                  <div class="compare-row"><b class="compare-label">MODEL</b><div class="compare-bars compare-model">
                    <i style="height:52%"></i><i style="height:74%"></i><i style="height:88%"></i><i style="height:66%"></i><i style="height:51%"></i><i style="height:81%"></i><i style="height:71%"></i><i style="height:91%"></i>
                  </div></div>
                </div>
                <div class="scene-node-foot">REPORT · MODEL · INFERENCE CODE</div>
              </div>
            </div>
            <div class="scene-legend"><span><i class="legend-measured"></i> measured signal</span><span><i class="legend-model"></i> fitted response</span><span class="scene-loop">AGENT-ASSISTED MODEL SEARCH</span></div>
          </div>
        </section>
        """,
    )


def render_configure() -> tuple[Optional[SysIDOptions], bool]:
    # Initialize wizard state and remember the selected dataset across steps.
    if "config_step" not in st.session_state:
        st.session_state["config_step"] = 1
    if "data_path" not in st.session_state:
        st.session_state["data_path"] = None
    if st.session_state["config_step"] not in (1, 2, 3):
        st.session_state["config_step"] = 1
    if st.session_state["config_step"] != 1 and not st.session_state.get("data_path"):
        st.session_state["config_step"] = 1
        st.rerun()

    if st.session_state["config_step"] == 1:
        st.progress(1 / 3)
        render_sysid_intro()
        st.markdown("### 1 · Dataset")
        st.caption("Step 1 of 3 · Upload a time-series file and check the detected columns.")

        with st.container(border=True):
            uploaded = st.file_uploader(
                "Dataset", type=["csv", "xlsx", "xls"],
                help="CSV or Excel. Required: time and one or more s_* state columns.",
            )
            st.caption("Required: time, s_* · Optional: a_* actions, xdot_* derivatives")

        if uploaded is not None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            target = UPLOAD_DIR / uploaded.name
            target.write_bytes(uploaded.getbuffer())
            st.session_state["data_path"] = str(target)

        with st.expander("Data format and integration guide"):
            st.markdown(
                "Use one s_* column per state. Column order determines state and action ordering. "
                "The pipeline estimates derivatives when xdot_* columns are absent."
            )
            img_path = _REPO_ROOT / "AgentSysID_Data_Contract_Specification.png"
            if img_path.is_file():
                st.image(
                    str(img_path), width="stretch",
                    caption="System identification data schema",
                )
            guide_path = (
                _REPO_ROOT
                / "backend_core/AgentSysID/data/System Identification Data Integration Guide.pdf"
            )
            if guide_path.is_file():
                st.download_button(
                    "Download integration guide",
                    data=guide_path.read_bytes(),
                    file_name="System_Identification_Data_Integration_Guide.pdf",
                    mime="application/pdf",
                )

        if st.session_state["data_path"]:
            with st.container(border=True):
                render_data_preview(st.session_state["data_path"])

            if st.button(
                "Continue to data assumptions",
                type="primary",
                icon=":material/arrow_forward:",
            ):
                st.session_state["config_step"] = 2
                st.rerun()

        return None, False

    if st.session_state["config_step"] == 2:
        st.progress(2 / 3)
        st.markdown("### 2 · Data assumptions")
        st.caption("Step 2 of 3 · Tell us how to interpret angles and experiment boundaries.")

        with st.container(border=True):
            st.markdown(f"Dataset · **{Path(st.session_state['data_path']).name}**")

        data_path = st.session_state["data_path"]
        try:
            state_columns = list(_peek(data_path, Path(data_path).stat().st_mtime)["states"])
        except Exception as exc:  # noqa: BLE001
            state_columns = []
            st.warning(f"State columns could not be read: {exc}")
        st.session_state["configured_state_columns"] = state_columns

        customer_description = st.text_input(
            "What physical system does this data describe? (optional)",
            value=st.session_state.get(
                "configured_customer_description", D("CUSTOMER_SYSTEM_DESCRIPTION")
            ),
            placeholder="e.g. A pendulum with a motor at its pivot",
            help="A short description helps the agents choose a useful model.",
            key="customer_description_step2",
        )
        st.session_state["configured_customer_description"] = customer_description

        # Migrate assumptions created by the earlier version of this step.
        legacy_angle_mode = st.session_state.get("configured_angle_mode", "None")
        if "configured_has_wrapping_states" not in st.session_state:
            st.session_state["configured_has_wrapping_states"] = (
                "No" if legacy_angle_mode == "None" else "Yes"
            )
        if "configured_angle_method" not in st.session_state:
            st.session_state["configured_angle_method"] = (
                "Choose manually"
                if legacy_angle_mode == "Specify manually"
                else "Auto-detect"
            )
        if "configured_wrap_state_names" not in st.session_state:
            old_indices = _int_list(st.session_state.get("configured_angle_text", "")) or []
            st.session_state["configured_wrap_state_names"] = [
                state_columns[index] for index in old_indices
                if 0 <= index < len(state_columns)
            ]

        with st.container(border=True):
            st.subheader("Angle wrapping")
            st.caption(
                "An angle stored between −π and +π jumps from +π to −π when it completes a turn. "
                "Tell us whether any state behaves this way."
            )
            angle_controls, angle_visual = st.columns([1, 1.15], vertical_alignment="center")
            with angle_controls:
                has_wrapping = st.segmented_control(
                    "Do any states wrap at ±π?",
                    ["No", "Yes"],
                    default=st.session_state["configured_has_wrapping_states"],
                    required=True,
                    key="has_wrapping_states_step2",
                    wrap=True,
                )
                angle_method = st.session_state.get(
                    "configured_angle_method", "Auto-detect"
                )
                selected_wrap_states = st.session_state.get(
                    "configured_wrap_state_names", []
                )
                if has_wrapping == "Yes":
                    angle_method = st.segmented_control(
                        "How should we identify them?",
                        ["Auto-detect", "Choose manually"],
                        default=angle_method,
                        required=True,
                        key="angle_method_step2",
                        wrap=True,
                    )
                    if angle_method == "Choose manually":
                        if state_columns:
                            saved_names = [
                                name for name in selected_wrap_states if name in state_columns
                            ]
                            selected_wrap_states = st.multiselect(
                                "Select the wrapping states",
                                options=state_columns,
                                default=saved_names,
                                format_func=lambda name: (
                                    f"State {state_columns.index(name)} · {name}"
                                ),
                                help="The displayed state numbers are 0-based and match the dataset column order.",
                                key="angle_wrap_states_step2",
                            )
                            st.caption("Choose every state whose angle resets at ±π.")
                            if not selected_wrap_states:
                                st.warning("Select at least one state to continue.")
                        else:
                            st.warning("No s_* state columns were found for manual selection.")

            with angle_visual:
                st.markdown("**Wrapped angle in the dataset**")
                st.altair_chart(angle_wrap_example_chart(), width="stretch")
                st.caption("Example: the stored angle reaches +π, then wraps to −π.")

        st.session_state["configured_has_wrapping_states"] = has_wrapping or "No"
        st.session_state["configured_angle_method"] = angle_method
        st.session_state["configured_wrap_state_names"] = list(selected_wrap_states)

        with st.container(border=True):
            st.subheader("Trajectory structure")
            st.caption(
                "A single run continues as one experiment. Several stacked runs are separate "
                "experiments joined into one file, often with a state reset between them."
            )
            single_chart, stacked_chart = trajectory_structure_examples()
            chart_col_a, chart_col_b = st.columns(2)
            with chart_col_a:
                st.markdown("**Single continuous run**")
                st.altair_chart(single_chart, width="stretch")
                st.caption("One uninterrupted experiment")
            with chart_col_b:
                st.markdown("**Several stacked runs**")
                st.altair_chart(stacked_chart, width="stretch")
                st.caption("Separate experiments joined together")

            traj_mode = st.segmented_control(
                "How is your data organized?",
                ["Single continuous run", "Several stacked trajectories"],
                default=st.session_state.get(
                    "configured_traj_mode", "Single continuous run"
                ),
                required=True,
                key="traj_mode_step2",
                wrap=True,
            )
            st.session_state["configured_traj_mode"] = (
                traj_mode or "Single continuous run"
            )

            split_mode = st.session_state.get(
                "configured_split_mode", "Auto-detect boundaries"
            )
            split_times_text = st.session_state.get("configured_split_times_text", "")
            if traj_mode == "Several stacked trajectories":
                split_mode = st.segmented_control(
                    "How should we find where each run starts?",
                    ["Auto-detect boundaries", "I know the timestamps"],
                    default=split_mode,
                    required=True,
                    key="split_mode_step2",
                    wrap=True,
                )
                if split_mode == "I know the timestamps":
                    split_times_text = st.text_input(
                        "Run start times (seconds)",
                        value=split_times_text,
                        placeholder="e.g. 12.5, 25.0",
                        help="Enter each split time from the time column, separated by commas.",
                        key="split_times_step2",
                    )
                    parsed_splits = _float_list(split_times_text)
                    if not split_times_text.strip():
                        st.caption("Add at least one split time, for example: 12.5, 25.0")
                    elif parsed_splits is None:
                        st.warning("Enter numeric split times separated by commas.")
                    elif not parsed_splits:
                        st.caption("Add at least one split time, for example: 12.5, 25.0")
            st.session_state["configured_split_mode"] = split_mode
            st.session_state["configured_split_times_text"] = split_times_text

        angle_configuration_valid = not (
            has_wrapping == "Yes"
            and angle_method == "Choose manually"
            and (not selected_wrap_states or not state_columns)
        )
        trajectory_configuration_valid = not (
            traj_mode == "Several stacked trajectories"
            and split_mode == "I know the timestamps"
            and not (_float_list(split_times_text) or [])
        )

        st.caption("Press Enter to continue, or choose Continue below.")
        continue_col, back_col = st.columns([2, 1])
        with continue_col:
            continue_step = st.button(
                "Continue to run settings",
                type="primary",
                icon=":material/arrow_forward:",
                key="continue_to_run_settings",
                width="stretch",
                shortcut="Enter",
                disabled=not (angle_configuration_valid and trajectory_configuration_valid),
            )
        with back_col:
            back_step = st.button(
                "Back to dataset",
                icon=":material/arrow_back:",
                key="back_to_dataset",
                width="stretch",
            )

        if continue_step or back_step:
            st.session_state["config_step"] = 3 if continue_step else 1
            st.rerun()

        return None, False

    st.progress(1.0)
    st.markdown("### 3 · Model and run")
    st.caption("Step 3 of 3 · Choose your model and run size; adjust advanced options only if needed.")

    with st.container(border=True):
        dataset_col, assumption_col, change_col = st.columns([3, 2, 2])
        with dataset_col:
            st.markdown(f"Dataset · **{Path(st.session_state['data_path']).name}**")
        with assumption_col:
            if st.button("Edit assumptions", key="edit_sysid_assumptions", width="stretch"):
                st.session_state["config_step"] = 2
                st.rerun()
        with change_col:
            if st.button("Change dataset", key="change_sysid_dataset", width="stretch"):
                st.session_state["config_step"] = 1
                st.rerun()

    output_dir = "artifacts_sysid"

    # --- Client-facing model choices -------------------------------------
    configured_architecture = str(
        st.session_state.get("configured_architecture", D("NETWORK_ARCHITECTURE"))
    ).upper()
    default_architecture = "LSTM" if configured_architecture == "LSTM" else "MLP"
    lstm_seq_length = int(
        st.session_state.get("configured_lstm_seq_length", D("LSTM_SEQ_LENGTH"))
    )
    pinn_loss_weight = float(
        st.session_state.get("configured_pinn_loss_weight", D("PINN_LOSS_WEIGHT"))
    )
    pinn_equation_file = str(
        st.session_state.get("configured_pinn_equation_file", D("PINN_EQUATION_FILE"))
    )

    model_col, physics_col = st.columns([1.35, 1], gap="medium")
    with model_col:
        with st.container(border=True):
            st.markdown("#### Choose a model")
            st.caption("Pick how the model reads your time-series data.")
            architecture = st.segmented_control(
                "Model architecture",
                ["LSTM", "MLP"],
                default=default_architecture,
                required=True,
                key="client_model_architecture",
                help="LSTM reads a short sequence of recent samples. MLP maps one sample directly.",
            ) or default_architecture
            st.session_state["configured_architecture"] = architecture

            mlp_selected = " selected" if architecture == "MLP" else ""
            lstm_selected = " selected" if architecture == "LSTM" else ""
            st.html(f"""
            <div class="model-diagrams">
              <section class="model-diagram{mlp_selected}">
                <div class="model-diagram-head"><b>MLP</b><span>one sample at a time</span></div>
                <div class="mlp-flow">
                  <div class="mlp-input"><i>State</i><i>Input</i></div>
                  <span class="model-arrow">→</span>
                  <div class="mlp-layers" aria-label="Neural network layers">
                    <i></i><i></i><i></i>
                  </div>
                  <span class="model-arrow">→</span>
                  <div class="mlp-output">Next change</div>
                </div>
                <p>Current state + input → predicted change</p>
              </section>
              <section class="model-diagram{lstm_selected}">
                <div class="model-diagram-head"><b>LSTM</b><span>recent sequence</span></div>
                <div class="lstm-flow">
                  <div class="lstm-steps">
                    <i>t−2</i><span>→</span><i>t−1</i><span>→</span><i>t</i>
                  </div>
                  <div class="lstm-memory"><span>recent history</span><i></i></div>
                  <div class="lstm-result">→ next change</div>
                </div>
                <p>Recent samples → learns how motion evolves</p>
              </section>
            </div>
            """)
            if architecture == "LSTM":
                lstm_seq_length = st.number_input(
                    "Recent history given to LSTM (steps)",
                    min_value=2,
                    max_value=200,
                    value=int(lstm_seq_length),
                    help="How many previous rows the LSTM can use to understand motion over time.",
                    key="client_lstm_memory_window",
                )
                st.session_state["configured_lstm_seq_length"] = int(lstm_seq_length)
            st.caption(
                "LSTM is a good starting point when behavior depends on recent motion. "
                "MLP is a simpler direct mapping when each row contains enough information."
            )

    with physics_col:
        with st.container(border=True):
            st.markdown("#### Physics guidance · optional")
            st.caption(
                "A physics-informed neural network (PINN) uses your system equations "
                "with the measured data to guide plausible predictions."
            )
            use_pinn = st.toggle(
                "Use physics-informed training (PINN)",
                value=bool(st.session_state.get("configured_use_pinn", D("USE_PINN"))),
                key="client_use_pinn",
                help="Leave this off if you do not have a trusted equation for this system.",
            )
            st.session_state["configured_use_pinn"] = bool(use_pinn)
            if use_pinn:
                st.info(
                    "Use PINN only when you can provide a trusted equation. If the equation "
                    "file is missing, a starter file is created and must be completed before training.",
                    icon="ℹ️",
                )
                with st.expander("PINN setup", expanded=False):
                    pinn_loss_weight = st.number_input(
                        "Physics influence",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(pinn_loss_weight),
                        step=0.1,
                        help="How strongly the known equation guides training. The default is a balanced starting point.",
                        key="client_pinn_loss_weight",
                    )
                    pinn_equation_file = st.text_input(
                        "Physics equation file",
                        value=pinn_equation_file,
                        help="The module must define compute_analytical_xdot(states, actions).",
                        key="client_pinn_equation_file",
                    )
                    st.session_state["configured_pinn_loss_weight"] = float(pinn_loss_weight)
                    st.session_state["configured_pinn_equation_file"] = pinn_equation_file

    if not use_pinn:
        pinn_loss_weight = float(
            st.session_state.get("configured_pinn_loss_weight", D("PINN_LOSS_WEIGHT"))
        )
        pinn_equation_file = str(
            st.session_state.get("configured_pinn_equation_file", D("PINN_EQUATION_FILE"))
        )

    # --- Run budget -------------------------------------------------------
    st.markdown("#### Run size")
    run_mode = st.selectbox(
        "Run mode", ["fast", "regular", "heavy"], index=1,
        help="fast = 7 cycles / 0.5 h · regular = 20 / 1.5 h · heavy = 40 / 4 h",
    )
    limits = cfg.run_mode_limits(run_mode)
    st.caption(
        f"{limits['max_cycles']} cycles · {limits['max_hours']} h · Critic explores to cycle "
        f"{limits['critic_explore_limit']} · failure memory: {limits['memory_capacity']} · "
        f"customer context: {limits['context_status']}"
    )
    st.caption("Fit diagnostics and held-out validation plots are saved with the run for review.")

    with st.expander("Custom cycle and epoch limits"):
        max_cycles, epochs = st.columns(2)
        max_cycles = max_cycles.text_input(
            "Maximum cycles", "", placeholder=str(limits["max_cycles"]),
        )
        epochs = epochs.text_input("Epochs per cycle", "", placeholder=str(D("EPOCHS")))

    with st.expander("Advanced settings", expanded=False):
        st.markdown("##### Data and rollout")
        col_l, col_r = st.columns(2)

        with col_l.expander("Rollout and integration"):
            rollout_horizon = st.number_input(
                "Rollout horizon", 1, 50, int(D("ROLLOUT_HORIZON")),
                help=">1 trains autoregressively on its own predictions.",
            )
            integrator_type = st.selectbox(
                "Kinematic integrator", ["RK4", "EULER"],
                index=0 if str(D("INTEGRATOR_TYPE")).upper() == "RK4" else 1,
            )
            trajectory_chunk_size = st.number_input(
                "Trajectory chunk size (0 = none)", 0, 100000, int(D("TRAJECTORY_CHUNK_SIZE"))
            )
            shuffle_data = st.checkbox(
                "Shuffle data", value=bool(D("SHUFFLE_DATA")),
                help="Forced off for LSTM to preserve contiguous temporal memory.",
            )
            st.caption(
                f"Validation windows need **{int(lstm_seq_length) if architecture == 'LSTM' else 1}"
                f" + {int(rollout_horizon)} − 1 = "
                f"{(int(lstm_seq_length) if architecture == 'LSTM' else 1) + int(rollout_horizon) - 1}"
                " consecutive rows**. A shorter validation split is rejected rather than "
                "scored as a perfect fit."
            )

        with col_r.expander("Derivative estimation"):
            st.caption("Only used when the dataset has no xdot_* columns.")
            methods = ["finite_difference", "savitzky_golay", "sliding_mode"]
            derivative_method = st.selectbox(
                "Method", methods,
                index=methods.index(str(D("DERIVATIVE_METHOD")))
                if str(D("DERIVATIVE_METHOD")) in methods else 0,
            )
            derivative_filter_tau = st.number_input(
                "Simulink filter tau (0 = off)", 0.0, 5.0, float(D("DERIVATIVE_FILTER_TAU")),
                step=0.001, format="%.4f",
            )
            savgol_window = st.number_input(
                "Savitzky-Golay window (odd)", 3, 201, int(D("SAVGOL_WINDOW")),
                disabled=derivative_method != "savitzky_golay",
            )
            savgol_polyorder = st.number_input(
                "Savitzky-Golay polyorder", 1, 9, int(D("SAVGOL_POLYORDER")),
                disabled=derivative_method != "savitzky_golay",
            )
            smd_lambda_1 = st.number_input(
                "Sliding-mode λ1", 0.0, 500.0, float(D("SMD_LAMBDA_1")),
                disabled=derivative_method != "sliding_mode",
            )
            smd_lambda_2 = st.number_input(
                "Sliding-mode λ2", 0.0, 500.0, float(D("SMD_LAMBDA_2")),
                disabled=derivative_method != "sliding_mode",
            )
            reset_threshold_text = st.text_input(
                "Reset threshold", "", placeholder="auto-calibrated from the data",
                help="One float, or one per state. Blank = auto.",
            )

        with st.expander("State-space filter"):
            use_state_filter = st.checkbox("Clip state outliers", value=bool(D("USE_STATE_FILTER")))
            pct_low, pct_high = st.slider(
                "Percentile band", 0.0, 100.0,
                (float(D("AUTO_FILTER_PERCENTILES")[0]), float(D("AUTO_FILTER_PERCENTILES")[1])),
                disabled=not use_state_filter,
            )

        st.markdown("##### Training and search")
        col_l3, col_r3 = st.columns(2)
        with col_l3.expander("Training limits"):
            batch_size = st.number_input("Batch size", 8, 4096, int(D("BATCH_SIZE")))
            early_stop_patience = st.number_input(
                "Early-stop patience", 1, 500, int(D("EARLY_STOP_PATIENCE"))
            )
            mse_target = st.number_input(
                "MSE target (stops the run)", 0.0, 10.0, float(D("MSE_TARGET")),
                step=1e-5, format="%.6f",
            )
            overfit_ratio_limit = st.number_input(
                "Overfit ratio limit", 1.0, 100.0, float(D("OVERFIT_RATIO_LIMIT"))
            )
            customer_max_latency_ms = st.number_input(
                "Max inference latency (ms)", 0.01, 1000.0, float(D("CUSTOMER_MAX_LATENCY_MS"))
            )
            adaptive_regularization = st.checkbox(
                "Adaptive regularization", value=bool(D("ADAPTIVE_REGULARIZATION")),
                help="Let the Actor tune dropout / weight decay.",
            )
            lr_reduce_factor = st.number_input(
                "LR reduce factor", 0.05, 0.99, float(D("LR_REDUCE_FACTOR"))
            )
            lr_schedule_min_floor = st.number_input(
                "LR floor", 1e-9, 1e-2, float(D("LR_SCHEDULE_MIN_FLOOR")), format="%.8f"
            )
            adaptive_improvement_threshold = st.number_input(
                "Epoch-extension threshold", 0.0, 1.0,
                float(D("ADAPTIVE_IMPROVEMENT_THRESHOLD")), step=0.001, format="%.4f",
            )
            epoch_extension_steps = st.number_input(
                "Epoch-extension steps", 0, 1000, int(D("EPOCH_EXTENSION_STEPS"))
            )

        with col_r3.expander("Search bounds (client-authorised)"):
            lr_lo = st.number_input("LR min", 1e-8, 1.0, float(D("LEARNING_RATE_MIN")), format="%.8f")
            lr_hi = st.number_input("LR max", 1e-8, 1.0, float(D("LEARNING_RATE_MAX")), format="%.8f")
            hs_lo = st.number_input("Hidden size min", 1, 4096, int(D("HIDDEN_SIZE_MIN")))
            hs_hi = st.number_input("Hidden size max", 1, 4096, int(D("HIDDEN_SIZE_MAX")))
            nl_lo = st.number_input("Layer count min", 1, 32, int(D("NUM_LAYERS_MIN")))
            nl_hi = st.number_input("Layer count max", 1, 32, int(D("NUM_LAYERS_MAX")))

        st.markdown("##### Agents")
        col_l4, col_r4 = st.columns(2)
        with col_l4.expander("Initializer agent"):
            choose_via_llm = st.checkbox(
                "Let the Initializer Agent choose", value=bool(D("CHOOSE_VIA_LLM_INITIALIZER"))
            )
            st.caption(
                "Manual presets, used when the agent is switched off:" if not choose_via_llm
                else "Presets below apply only if the agent is switched off."
            )
            manual_starting_lr = st.number_input(
                "Preset LR", 1e-8, 1.0, float(D("MANUAL_STARTING_LR")),
                format="%.6f", disabled=choose_via_llm,
            )
            manual_hidden_text = st.text_input(
                "Preset topology", ", ".join(str(x) for x in D("MANUAL_STARTING_HIDDEN_LAYERS")),
                disabled=choose_via_llm,
            )
            activations = list(cfg.AVAILABLE_ACTIVATIONS)
            manual_activation = st.selectbox(
                "Preset activation", activations,
                index=activations.index(D("MANUAL_ACTIVATION"))
                if D("MANUAL_ACTIVATION") in activations else 0,
                disabled=choose_via_llm,
            )
            manual_dropout_rate = st.number_input(
                "Preset dropout", 0.0, 0.9, float(D("MANUAL_DROPOUT_RATE")), disabled=choose_via_llm
            )
            manual_weight_decay = st.number_input(
                "Preset weight decay", 0.0, 1.0, float(D("MANUAL_WEIGHT_DECAY")),
                format="%.6f", disabled=choose_via_llm,
            )
            user_overrides_text = st.text_area(
                "Client-locked parameters (JSON)", value="", height=70,
                placeholder='{"hidden_layers": [128, 128]}',
                help="Echoed into the Initializer prompt as parameters the client has "
                     "locked — the agent must return these exact values.",
            )

        with col_r4.expander("Override the agent's proposal"):
            st.caption(
                "The same fourteen parameters the terminal offers after the agent reports. "
                "Blank = keep the agent's choice."
            )
            o1, o2 = st.columns(2)
            ov_lr = o1.text_input("learning_rate", "", placeholder="e.g. 0.001")
            ov_layers = o2.text_input("hidden_layers", "", placeholder="e.g. 128, 64")
            ov_activation = o1.selectbox(
                "activation", ["(keep agent's choice)"] + list(cfg.AVAILABLE_ACTIVATIONS)
            )
            ov_dropout = o2.text_input("dropout_rate", "", placeholder="0.0 – 0.5")
            ov_wd = o1.text_input("weight_decay", "", placeholder="e.g. 0.0001")
            ov_filter = o2.selectbox("use_state_filter", ["(keep agent's choice)", "True", "False"])
            ov_pct = o1.text_input("auto_filter_percentiles", "", placeholder="2, 98")
            ov_tau = o2.text_input("derivative_filter_tau", "", placeholder="e.g. 0.005")
            ov_lr_bounds = o1.text_input("LR search bounds", "", placeholder="0.00005, 0.001")
            ov_hs_bounds = o2.text_input("Hidden size bounds", "", placeholder="32, 256")
            ov_nl_bounds = o1.text_input("Layer depth bounds", "", placeholder="1, 3")
            ov_epochs = o2.text_input("epochs", "", placeholder=str(D("EPOCHS")))
            ov_batch = o1.text_input("batch_size", "", placeholder=str(D("BATCH_SIZE")))
            ov_patience = o2.text_input(
                "early_stop_patience", "", placeholder=str(D("EARLY_STOP_PATIENCE"))
            )

        with st.expander("LLM provider"):
            p1, p2, p3 = st.columns(3)
            providers = ["openai", "groq", "openrouter"]
            api_provider = p1.selectbox(
                "Provider", providers,
                index=providers.index(D("API_PROVIDER")) if D("API_PROVIDER") in providers else 0,
            )
            llm_model = p2.text_input("Model", D("LLM_MODEL"))
            llm_temperature = p3.slider(
                "Temperature", 0.0, 2.0, float(D("LLM_TEMPERATURE")), 0.05,
            )
            st.caption(
                "No API key? Every agent falls back to its deterministic mathematical "
                "path and the run still produces the full deliverable."
            )
    # --- Assemble ---------------------------------------------------------

    # Restore dataset assumptions saved before moving between wizard steps.
    customer_description = st.session_state.get("configured_customer_description", "")
    output_dir = "artifacts_sysid"

    has_wrapping_states = (
        st.session_state.get("configured_has_wrapping_states", "No") == "Yes"
    )
    angle_method = st.session_state.get("configured_angle_method", "Auto-detect")
    auto_detect_angles = has_wrapping_states and angle_method == "Auto-detect"
    configured_state_columns = st.session_state.get("configured_state_columns", [])
    selected_wrap_state_names = st.session_state.get("configured_wrap_state_names", [])
    angle_indices = [
        configured_state_columns.index(name)
        for name in selected_wrap_state_names
        if name in configured_state_columns
    ] if has_wrapping_states and angle_method == "Choose manually" else []

    traj_mode = st.session_state.get("configured_traj_mode", "Single continuous run")
    multi_trajectory = (traj_mode == "Several stacked trajectories")
    split_mode = st.session_state.get("configured_split_mode", "Auto-detect boundaries")
    manual_split_times = sorted(
        _float_list(st.session_state.get("configured_split_times_text", "")) or []
    ) if multi_trajectory and split_mode == "I know the timestamps" else []

    # Process run settings overrides.
    reset_threshold: Optional[Any] = None
    parsed_reset = _float_list(reset_threshold_text)
    if parsed_reset:
        reset_threshold = parsed_reset[0] if len(parsed_reset) == 1 else parsed_reset

    user_overrides: Dict[str, Any] = {}
    if (user_overrides_text or "").strip():
        try:
            parsed = json.loads(user_overrides_text)
            if isinstance(parsed, dict):
                user_overrides = parsed
            else:
                st.warning("Client-locked parameters must be a JSON object.")
        except json.JSONDecodeError as exc:
            st.warning(f"Client-locked parameters: invalid JSON ({exc.msg}).")

    overrides: Dict[str, Any] = {}
    if _f(ov_lr) is not None:
        overrides["learning_rate"] = _f(ov_lr)
    if _int_list(ov_layers):
        overrides["hidden_layers"] = _int_list(ov_layers)
    if ov_activation != "(keep agent's choice)":
        overrides["activation"] = ov_activation
    if _f(ov_dropout) is not None:
        overrides["dropout_rate"] = _f(ov_dropout)
    if _f(ov_wd) is not None:
        overrides["weight_decay"] = _f(ov_wd)
    if ov_filter != "(keep agent's choice)":
        overrides["use_state_filter"] = ov_filter == "True"
    if _float_list(ov_pct):
        overrides["auto_filter_percentiles"] = _float_list(ov_pct)
    if _f(ov_tau) is not None:
        overrides["derivative_filter_tau"] = _f(ov_tau)
    lo, hi = _pair(ov_lr_bounds, float)
    if lo is not None:
        overrides["lr_search_min"], overrides["lr_search_max"] = lo, hi
    lo, hi = _pair(ov_hs_bounds, int)
    if lo is not None:
        overrides["hidden_size_search_min"], overrides["hidden_size_search_max"] = lo, hi
    lo, hi = _pair(ov_nl_bounds, int)
    if lo is not None:
        overrides["num_layers_search_min"], overrides["num_layers_search_max"] = lo, hi
    if _i(ov_epochs) is not None:
        overrides["epochs"] = _i(ov_epochs)
    if _i(ov_batch) is not None:
        overrides["batch_size"] = _i(ov_batch)
    if _i(ov_patience) is not None:
        overrides["early_stop_patience"] = _i(ov_patience)

    # 3. Finalize Options
    options = SysIDOptions(
        data_path=st.session_state.get("data_path"),
        run_mode=run_mode,
        output_dir=output_dir,
        interactive=False,
        max_cycles=_i(max_cycles),
        epochs=_i(epochs),
        customer_description=customer_description,
        angle_indices=angle_indices,
        auto_detect_angles=auto_detect_angles,
        multi_trajectory=multi_trajectory,
        manual_split_times=manual_split_times,
        architecture=architecture,
        lstm_seq_length=int(lstm_seq_length),
        rollout_horizon=int(rollout_horizon),
        integrator_type=integrator_type,
        trajectory_chunk_size=int(trajectory_chunk_size),
        shuffle_data=bool(shuffle_data),
        derivative_method=derivative_method,
        derivative_filter_tau=float(derivative_filter_tau),
        savgol_window=int(savgol_window),
        savgol_polyorder=int(savgol_polyorder),
        smd_lambda_1=float(smd_lambda_1),
        smd_lambda_2=float(smd_lambda_2),
        reset_threshold=reset_threshold,
        use_state_filter=bool(use_state_filter),
        auto_filter_percentiles=(pct_low, pct_high),
        use_pinn=bool(use_pinn),
        pinn_loss_weight=float(pinn_loss_weight),
        pinn_equation_file=pinn_equation_file or None,
        batch_size=int(batch_size),
        early_stop_patience=int(early_stop_patience),
        mse_target=float(mse_target),
        overfit_ratio_limit=float(overfit_ratio_limit),
        customer_max_latency_ms=float(customer_max_latency_ms),
        adaptive_regularization=bool(adaptive_regularization),
        lr_reduce_factor=float(lr_reduce_factor),
        lr_schedule_min_floor=float(lr_schedule_min_floor),
        adaptive_improvement_threshold=float(adaptive_improvement_threshold),
        epoch_extension_steps=int(epoch_extension_steps),
        learning_rate_min=float(lr_lo),
        learning_rate_max=float(lr_hi),
        hidden_size_min=int(hs_lo),
        hidden_size_max=int(hs_hi),
        num_layers_min=int(nl_lo),
        num_layers_max=int(nl_hi),
        choose_via_llm_initializer=bool(choose_via_llm),
        manual_starting_lr=float(manual_starting_lr),
        manual_starting_hidden_layers=_int_list(manual_hidden_text),
        manual_activation=manual_activation,
        manual_dropout_rate=float(manual_dropout_rate),
        manual_weight_decay=float(manual_weight_decay),
        user_overrides=user_overrides,
        initializer_overrides=overrides,
        api_provider=api_provider,
        llm_model=llm_model or None,
        llm_temperature=float(llm_temperature),
        save_plot=True,
    )
    return options, True

@st.cache_data(show_spinner=False)
def _peek(path: str, mtime: float) -> Dict[str, Any]:
    """Read the header of a dataset without loading all of it."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        head = pd.read_csv(p, nrows=200)
        rows = sum(1 for _ in open(p, "rb")) - 1
    else:
        head = pd.read_excel(p, nrows=200)
        rows = len(pd.read_excel(p, usecols=[0]))
    cols = list(head.columns)
    return {
        "head": head.head(6),
        "rows": rows,
        "states": [c for c in cols if c.startswith("s_")],
        "actions": [c for c in cols if c.startswith("a_")],
        "xdots": [c for c in cols if c.startswith("xdot_")],
        "has_time": "time" in cols,
    }


def render_data_preview(data_path: str) -> None:
    """Show what the loader will see — column mapping and obvious problems."""
    try:
        info = _peek(data_path, Path(data_path).stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read that file: {exc}")
        return

    n_s, n_a, n_x = len(info["states"]), len(info["actions"]), len(info["xdots"])
    tags = (
        f"<span class='tag'>{info['rows']} rows</span>"
        f"<span class='tag'>{n_s} states</span>"
        f"<span class='tag'>{n_a} inputs</span>"
        + (f"<span class='tag'>true xdot</span>" if n_x == n_s and n_s else
           "<span class='tag'>xdot estimated</span>")
    )
    st.markdown(f"<div style='margin:2px 0 8px'>{tags}</div>", unsafe_allow_html=True)

    if not info["has_time"]:
        st.error("No **time** column — the loader requires one to compute derivatives.")
    if n_s == 0:
        st.error("No **s_\\*** state columns found.")
    if n_a == 0:
        st.warning("No **a_\\*** action columns — open-loop identification only.")

    with st.expander(f"Preview · {Path(data_path).name}"):
        st.dataframe(info["head"], width="stretch", hide_index=True)
        st.caption(
            f"states: {', '.join(info['states']) or '—'}  |  "
            f"actions: {', '.join(info['actions']) or '—'}  |  "
            f"derivatives: {', '.join(info['xdots']) or 'estimated from the states'}"
        )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def angle_wrap_example_chart():
    """Show the discontinuity created when a wrapped angle crosses +pi."""
    import altair as alt

    angle_data = pd.DataFrame({
        "Time": list(range(7)),
        "Angle": [2.1, 2.5, 2.85, math.pi, -math.pi, -2.75, -2.35],
    })
    boundaries = pd.DataFrame({"Angle": [-math.pi, math.pi]})
    limits = alt.Chart(boundaries).mark_rule(
        color=T.TEXT_FAINT, strokeDash=[4, 4], strokeWidth=1,
    ).encode(y="Angle:Q")
    wrapped_line = alt.Chart(angle_data).mark_line(
        color=T.SERIES_A,
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=48, filled=True, color=T.SERIES_A),
    ).encode(
        x=alt.X("Time:Q", title="Time"),
        y=alt.Y(
            "Angle:Q",
            title="Angle (radians)",
            scale=alt.Scale(domain=[-math.pi * 1.12, math.pi * 1.12]),
            axis=alt.Axis(values=[-math.pi, 0, math.pi], format=".2f"),
        ),
        tooltip=[
            alt.Tooltip("Time:Q", title="Time"),
            alt.Tooltip("Angle:Q", title="Stored angle", format=".2f"),
        ],
    )
    return _style(limits + wrapped_line, height=205)


def trajectory_structure_examples():
    """Return paired examples of one continuous run and stacked runs."""
    import altair as alt

    single_data = pd.DataFrame({
        "Time": list(range(11)),
        "State": [0.0, 0.6, 1.1, 1.45, 1.55, 1.5, 1.7, 2.15, 2.65, 3.0, 3.2],
    })
    single_line = alt.Chart(single_data).mark_line(
        color=T.SERIES_A,
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=32, filled=True, color=T.SERIES_A),
    ).encode(
        x=alt.X("Time:Q", title="Time (s)"),
        y=alt.Y("State:Q", title="Example state"),
        tooltip=[alt.Tooltip("Time:Q"), alt.Tooltip("State:Q", format=".2f")],
    )
    single_chart = _style(single_line, height=195)

    stacked_rows = []
    examples = [
        ("Run 1", [0.0, 0.8, 1.45, 1.9, 2.2]),
        ("Run 2", [0.0, 0.55, 1.25, 1.7, 2.0]),
        ("Run 3", [0.0, 0.65, 1.05, 1.55, 2.25]),
    ]
    for run_index, (run_name, states) in enumerate(examples):
        for local_time, state_value in enumerate(states):
            stacked_rows.append({
                "Time": run_index * len(states) + local_time,
                "State": state_value,
                "Run": run_name,
            })
    stacked_data = pd.DataFrame(stacked_rows)
    run_lines = alt.Chart(stacked_data).mark_line(
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=30, filled=True),
    ).encode(
        x=alt.X("Time:Q", title="Time (s)"),
        y=alt.Y("State:Q", title="Example state"),
        color=alt.Color(
            "Run:N",
            title=None,
            scale=alt.Scale(
                domain=["Run 1", "Run 2", "Run 3"],
                range=[T.SERIES_A, "#569cd6", "#c586c0"],
            ),
            legend=alt.Legend(orient="top", labelColor=T.TEXT_DIM),
        ),
        tooltip=[
            alt.Tooltip("Run:N"),
            alt.Tooltip("Time:Q"),
            alt.Tooltip("State:Q", format=".2f"),
        ],
    )
    split_rules = alt.Chart(pd.DataFrame({"Time": [5, 10]})).mark_rule(
        color=T.WARN, strokeDash=[4, 4], strokeWidth=1,
    ).encode(x="Time:Q")
    stacked_chart = _style(split_rules + run_lines, height=195)
    return single_chart, stacked_chart


def convergence_chart(history: List[Dict[str, Any]]):
    """
    Train vs validation MSE across tuning cycles.

    Two series, so a legend is always present and hovering direct-labels each
    point. Log y-axis because the error spans orders of magnitude. Latency is a
    separate chart — never a second y-axis.
    """
    import altair as alt

    rows = []
    for p in history:
        cycle = p.get("cycle", p.get("iteration", 0) + 1)
        for label, key in (("Train MSE", "train_mse"), ("Validation MSE", "val_mse")):
            value = p.get(key)
            if value is None or value != value or value <= 0:
                continue
            rows.append({"Cycle": cycle, "Series": label, "MSE": float(value)})
    if not rows:
        return None

    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Cycle:Q", axis=alt.Axis(tickMinStep=1, title="Actor-Critic cycle")),
        y=alt.Y("MSE:Q", scale=alt.Scale(type="log"),
                axis=alt.Axis(title="MSE (log)", format=".1e")),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=["Train MSE", "Validation MSE"],
                            range=[T.SERIES_A, T.SERIES_B]),
            legend=alt.Legend(title=None, orient="top", labelColor=T.TEXT_DIM,
                              symbolType="stroke"),
        ),
    )
    line = base.mark_line(strokeWidth=2, interpolate="monotone")
    points = base.mark_point(size=80, filled=True, stroke=T.BG_CARD, strokeWidth=2).encode(
        tooltip=[
            alt.Tooltip("Cycle:Q", title="Cycle"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("MSE:Q", title="MSE", format=".3e"),
        ]
    )
    return _style(line + points, height=250)


def latency_chart(history: List[Dict[str, Any]], max_latency: float):
    """Measured latency per cycle against the customer's hard limit (one series)."""
    import altair as alt

    rows = [
        {"Cycle": p.get("cycle", p.get("iteration", 0) + 1), "Latency": float(p["latency"])}
        for p in history
        if p.get("latency") is not None
    ]
    if not rows:
        return None

    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Cycle:Q", axis=alt.Axis(tickMinStep=1, title="Actor-Critic cycle")),
        y=alt.Y("Latency:Q", axis=alt.Axis(title="Inference latency (ms)")),
    )
    line = base.mark_line(strokeWidth=2, color=T.SERIES_A, interpolate="monotone")
    points = base.mark_point(
        size=80, filled=True, color=T.SERIES_A, stroke=T.BG_CARD, strokeWidth=2
    ).encode(
        tooltip=[alt.Tooltip("Cycle:Q", title="Cycle"),
                 alt.Tooltip("Latency:Q", title="Latency (ms)", format=".3f")]
    )
    rule_df = pd.DataFrame({
        "limit": [max_latency],
        "Cycle": [float(df["Cycle"].min())],
        "text": [f"customer limit {max_latency:g} ms"],
    })
    limit = alt.Chart(rule_df).mark_rule(
        color=T.CRIT, strokeDash=[6, 4], strokeWidth=2
    ).encode(y="limit:Q")
    label = alt.Chart(rule_df).mark_text(
        align="left", baseline="bottom", dx=4, dy=-4, color=T.CRIT, fontSize=11
    ).encode(x="Cycle:Q", y="limit:Q", text="text:N")
    return _style(line + points + limit + label, height=215)


def _style(chart, height: int):
    return chart.properties(
        height=height,
        padding={"top": 12, "bottom": 18, "left": 22, "right": 14},
    ).configure_view(
        stroke=T.BORDER, fill=T.BG_CARD
    ).configure_axis(
        grid=True, gridColor=T.BORDER, gridOpacity=0.5,
        labelColor=T.TEXT_DIM, titleColor=T.TEXT_DIM,
        domainColor=T.BORDER, tickColor=T.BORDER,
    ).configure_legend(labelColor=T.TEXT_DIM).configure_title(
        color=T.TEXT_DIM, fontSize=12, anchor="start",
    ).configure(background=T.BG_CARD)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def render_stage_rail(current_stage: str, progress: float, finished: bool) -> None:
    """Render the pipeline as connected, instrument-style process blocks."""
    blocks = [
        {
            "title": "Prepare data",
            "stages": ["Questionnaire", "Loading dataset", "Data Inspector", "Splitting trajectories"],
            "route": "QUESTIONNAIRE · LOAD · INSPECT · SPLIT",
            "description": "Checks the data contract, prepares derivatives, and separates independent runs.",
            "tags": ["time + states", "derivatives", "trajectories"],
            "signal": "clean runs",
        },
        {
            "title": "Initialize model",
            "stages": ["Initializer Agent"],
            "route": "SYSTEM CONTEXT · MODEL SEED",
            "description": "Builds a starting dynamics model from the system description and search limits.",
            "tags": ["architecture", "parameters", "constraints"],
            "signal": "candidate",
        },
        {
            "title": "Tune dynamics",
            "stages": ["Tuning cycles"],
            "route": "TRAIN ↔ CRITIC ↔ EXPLORER",
            "description": "Refines the model using validation error, stability checks, and bounded search.",
            "tags": ["fit", "validate", "iterate"],
            "signal": "best model",
        },
        {
            "title": "Verify + package",
            "stages": ["Held-out verification", "Report & packaging"],
            "route": "ROLLOUT · SCORE · EXPORT",
            "description": "Tests unseen trajectories and prepares the report, model, and inference code.",
            "tags": ["held-out test", "report", "artifacts"],
            "signal": "deliverables",
        },
    ]
    current_index = STAGES.index(current_stage) if current_stage in STAGES else 0
    stage_position = {name: index for index, name in enumerate(STAGES)}
    parts = []
    for index, block in enumerate(blocks):
        block_positions = [stage_position[name] for name in block["stages"]]
        active = not finished and current_index in block_positions
        complete = finished or current_index > max(block_positions)
        state_class = "done" if complete else "active" if active else "pending"
        state_label = "✓ COMPLETE" if complete else "● RUNNING" if active else "○ QUEUED"
        route = block["route"]
        if active:
            route = f"CURRENT · {current_stage.upper()}"
        tags = "".join(f"<span>{tag}</span>" for tag in block["tags"])
        parts.append(
            f"<article class='pipeline-block {state_class}'>"
            f"<div class='pipe-block-meta'><span class='pipe-index'>BLOCK {index + 1:02d}</span>"
            f"<span class='pipe-state'>{state_label}</span></div>"
            f"<div class='pipe-title'>{block['title']}</div>"
            f"<div class='pipe-subtitle'>{route}</div>"
            f"<p class='pipe-description'>{block['description']}</p>"
            f"<div class='pipe-tags'>{tags}</div>"
            "</article>"
        )
        if index < len(blocks) - 1:
            parts.append(
                f"<div class='pipeline-link' aria-hidden='true'>"
                f"<span class='pipeline-link-icon'>›</span>"
                f"<span class='pipeline-link-label'>{block['signal']}</span></div>"
            )

    progress_value = 1.0 if finished else min(max(float(progress or 0.0), 0.0), 1.0)
    progress_percent = round(progress_value * 100)
    current_label = "Run complete" if finished else (current_stage or "Preparing run")
    st.html(
        f"""
        <section class="pipeline-board" aria-label="System identification pipeline status">
          <header class="pipeline-board-head">
            <div>
              <div class="pipeline-kicker">Signal flow / execution map</div>
              <div class="pipeline-heading">From observed data to validated dynamics</div>
            </div>
            <div class="pipeline-current">
              <div class="pipeline-current-label">Current stage</div>
              <div class="pipeline-current-value">{current_label}</div>
            </div>
          </header>
          <div class="pipeline-flow">{''.join(parts)}</div>
          <footer class="pipeline-board-footer">
            <span class="pipeline-progress-label">RUN PROGRESS&nbsp; {progress_percent:02d}%</span>
            <span class="pipeline-progress-track">
              <span class="pipeline-progress-fill" style="width:{progress_percent}%"></span>
            </span>
            <span class="pipeline-progress-label">{len(STAGES)} stages</span>
          </footer>
        </section>
        """,
    )


def render_monitor(state, options: Optional[SysIDOptions], running: bool) -> None:
    viewed = state.get("viewing")
    if viewed is not None and not running:
        st.caption(f"Viewing a past run — {_history_text(hist.display_name(viewed))}")
        activity.render(activity.load(viewed["run_dir"]), historical=True, scope=viewed["run_dir"])
        return
    if not state["log"] and not running and not state.get("activity"):
        st.info("No active run. Configure a run and press **Start** to watch it here.")
        activity.render([])
        return

    render_stage_rail(
        state["stage"],
        state["progress"],
        finished=not running and state["result"] is not None,
    )

    activity.render(state.get("activity", []))
    if state.get("activity_save_error"):
        st.warning("The activity feed could not be saved to this run's history folder.")

    history = state["history"]
    if history:
        latest = history[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cycle", f"{latest['cycle']} / {latest['max_cycles']}")
        m2.metric("Validation MSE", f"{latest['val_mse']:.3e}")
        m3.metric("Best MSE", f"{latest['best_mse']:.3e}")
        m4.metric("Topology", str(latest["config"].get("hidden_layers")))

        c1, c2 = st.columns(2)
        chart = convergence_chart(history)
        if chart is not None:
            c1.markdown("##### Identification error per cycle")
            c1.altair_chart(chart, width="stretch")
        max_lat = float(options.customer_max_latency_ms) if options else cfg.CUSTOMER_MAX_LATENCY_MS
        lat = latency_chart(history, max_lat)
        if lat is not None:
            c2.markdown("##### Latency against the customer limit")
            c2.altair_chart(lat, width="stretch")
        else:
            c2.markdown("##### Latency against the customer limit")
            c2.caption(
                "Latency is measured after a cycle that does not immediately hit the "
                "MSE target — nothing to plot yet."
            )

        with st.expander("Cycle table"):
            st.dataframe(
                pd.DataFrame([
                    {
                        "Cycle": h["cycle"],
                        "Topology": str(h["config"].get("hidden_layers")),
                        "LR": h["config"].get("learning_rate"),
                        "Dropout": h["config"].get("dropout_rate"),
                        "Train MSE": h["train_mse"],
                        "Val MSE": h["val_mse"],
                        "RMSE": h["rmse"],
                        "Latency (ms)": h.get("latency"),
                        "Best": "⭐" if h["is_best"] else "",
                    } for h in history
                ]),
                width="stretch", hide_index=True,
            )

    if state["log"]:
        with st.expander("Console output", icon=":material/terminal:"):
            st.code(state["log"][-14000:], language="text")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_results_from_entry(entry: Dict[str, Any]) -> None:
    """Render a past run, read back from its manifest."""
    if not entry.get("complete"):
        if entry.get("status") == "failed":
            st.warning(entry.get("message") or "This run recorded an execution failure. Ask about this run to investigate its evidence.")
        else:
            st.warning(
                f"`{entry.get('name')}` did not complete packaging. Any artefacts it wrote are below."
            )
    _render_result_body(
        env_name=entry.get("env_name", "—"),
        score=float(entry.get("success_score") or 0.0),
        status=entry.get("model_status", "—"),
        best_mse=float(entry.get("best_mse") or float("nan")),
        best_rmse=float(entry.get("best_rmse") or float("nan")),
        latency=float(entry.get("latency_ms") or 0.0),
        cycles=int(entry.get("cycles_run") or 0),
        complexity=entry.get("complexity_label", "—"),
        state_dim=int(entry.get("state_dim") or 0),
        action_dim=int(entry.get("action_dim") or 0),
        topology=(entry.get("best_config") or {}).get("hidden_layers"),
        activation=entry.get("activation", ""),
        llm_calls=int(entry.get("llm_calls") or 0),
        llm_cost=float(entry.get("llm_cost_usd") or 0.0),
        elapsed=float(entry.get("elapsed_seconds") or 0.0),
        abstract=entry.get("abstract", ""),
        conclusion=entry.get("conclusion", ""),
        files=hist.existing_files(entry),
        figures=hist.figures_of(entry),
        quality_issues=entry.get("quality_issues") or [],
        run_dir=entry.get("run_dir", ""),
    )


def render_results(result: SysIDResult) -> None:
    files = {
        "Results ZIP": result.zip_path,
        "PDF report": result.pdf_path,
        "Model weights (.pth)": result.pth_path,
        "Standalone controller": result.controller_path,
        "NN.py inference script": result.nn_helper_path,
        "Agent conversation log": result.log_path,
    }
    _render_result_body(
        env_name=result.env_name,
        score=result.success_score,
        status=result.model_status,
        best_mse=result.best_mse,
        best_rmse=result.best_rmse,
        latency=result.latency_ms,
        cycles=result.cycles_run,
        complexity=result.complexity_label,
        state_dim=result.state_dim,
        action_dim=result.action_dim,
        topology=result.best_config.get("hidden_layers"),
        activation=result.activation,
        llm_calls=result.llm_calls,
        llm_cost=result.llm_cost_usd,
        elapsed=result.elapsed_seconds,
        abstract=result.abstract,
        conclusion=result.conclusion,
        files={k: v for k, v in files.items() if v and Path(v).is_file()},
        figures=[f for f in result.figures if Path(f).is_file()],
        quality_issues=result.quality_issues,
        run_dir=str(result.run_dir or ""),
    )


def _render_result_body(**k) -> None:
    colour = T.status_color(k["status"])
    score_col = T.score_color(k["score"])

    st.markdown(
        f"""
        <div class="hero">
          <div>
            <span class="hero-score" style="color:{score_col}">{k['score']:.1f}</span>
            <span class="hero-of"> / 100</span>
          </div>
          <div class="hero-meta">
            <div><span class="pill" style="background:{colour}22;color:{colour};
                 border:1px solid {colour}55">{k['status']}</span></div>
            <div class="hero-sub">{k['env_name']} · {k['complexity']}</div>
            <div class="hero-sub">{k['state_dim']} states · {k['action_dim']} inputs ·
                 topology {k['topology']} · {str(k['activation']).upper()}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation MSE", f"{k['best_mse']:.3e}")
    c2.metric("Validation RMSE", f"{k['best_rmse']:.3e}")
    c3.metric("Inference latency", f"{k['latency']:.3f} ms")
    c4.metric("Cycles run", str(k["cycles"]))

    st.markdown(
        f"<div class='mono' style='color:{T.TEXT_FAINT};font-size:11px;margin:-4px 0 14px'>"
        f"{k['llm_calls']} LLM calls · ${k['llm_cost']:.5f} · "
        f"{k['elapsed'] / 60:.1f} min · <span title='{k['run_dir']}'>"
        f"{Path(k['run_dir']).name if k['run_dir'] else ''}</span></div>",
        unsafe_allow_html=True,
    )

    # --- Deliverables -----------------------------------------------------
    files: Dict[str, str] = k["files"]
    if files:
        st.markdown("##### Deliverables")
        cols = st.columns(3)
        mimes = {
            "Results ZIP": "application/zip",
            "PDF report": "application/pdf",
            "Model weights (.pth)": "application/octet-stream",
        }
        for idx, (label, path) in enumerate(files.items()):
            with cols[idx % 3]:
                st.download_button(
                    label,
                    data=Path(path).read_bytes(),
                    file_name=Path(path).name,
                    mime=mimes.get(label, "text/plain"),
                    width="stretch",
                    key=f"dl_{label}_{k['run_dir']}",
                )
    else:
        st.info("No artefacts are still on disk for this run.")

    if k["abstract"]:
        with st.expander("Engineering manuscript"):
            st.markdown(f"**Abstract.** {k['abstract']}")
            if k["conclusion"]:
                st.markdown(f"**Conclusion.** {k['conclusion']}")

    figures: List[str] = k["figures"]
    if figures:
        st.markdown("##### Diagnostic figures")
        verification = [f for f in figures if "test_verification" in f]
        others = [f for f in figures if "test_verification" not in f]
        for path in verification:
            st.image(path, caption="Held-out verification — true data vs NN rollout",
                     width="stretch")
        grid = st.columns(2)
        for i, path in enumerate(others):
            with grid[i % 2]:
                st.image(path, caption=Path(path).stem, width="stretch")

    if k["quality_issues"]:
        with st.expander("Data quality findings"):
            for issue in k["quality_issues"]:
                st.markdown(f"- {issue}")


# ---------------------------------------------------------------------------
# Live updates
# ---------------------------------------------------------------------------
def drain(runner: "PipelineRunner", state) -> bool:
    """Move everything the worker has produced into session state.

    Returns True when the run just finished, so the caller can navigate.
    """
    chunks = []
    while True:
        try:
            chunks.append(runner.logs.get_nowait())
        except queue.Empty:
            break
    if chunks:
        state["log"] = (state["log"] + "".join(chunks))[-60000:]

    just_finished = False
    while True:
        try:
            kind, payload = runner.events.get_nowait()
        except queue.Empty:
            break
        activity.record(state.setdefault("activity", []), kind, payload)
        if kind == "stage":
            state["stage"] = payload["name"]
        elif kind == "progress":
            state["progress"] = float(payload["value"])
        elif kind == "cycle":
            state["history"].append(payload)
        elif kind == "latency":
            for row in reversed(state["history"]):
                if row.get("cycle") == payload["cycle"]:
                    row["latency"] = payload["latency_ms"]
                    break
        elif kind == "critic":
            state["critic"].append(payload)
        elif kind == "done":
            state["result"] = payload["result"]
            just_finished = True

    if runner.result is not None and state["result"] is None:
        state["result"] = runner.result
        activity.record(state.setdefault("activity", []), "done", {"result": runner.result})
        just_finished = True
    result = state["result"]
    if not runner.running and not state.get("activity_saved"):
        run_dir = result.run_dir if result is not None else runner.run_dir
        if run_dir:
            saved = activity.save(run_dir, state.get("activity", []))
            state["activity_saved"] = True
            state["activity_save_error"] = not saved
    return just_finished


@st.fragment(run_every=1.2)
def live_monitor(options: Optional[SysIDOptions]) -> None:
    """
    The Monitor panel, refreshed on its own.

    A fragment re-renders only itself, so the live log and charts update
    roughly once a second without the whole page flashing stale on every tick
    — which is what a full st.rerun() polling loop does.
    """
    state = st.session_state
    runner: Optional[PipelineRunner] = state["runner"]
    running = bool(runner and runner.running)

    if runner is not None:
        finished = drain(runner, state)
        if finished or (not running and state["result"] is not None):
            # Leave the fragment and repaint the whole app on the Results tab.
            goto("Results")
            st.rerun(scope="app")

    render_monitor(state, options, running)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    state = st.session_state
    state.setdefault("runner", None)
    state.setdefault("log", "")
    state.setdefault("stage", STAGES[0])
    state.setdefault("progress", 0.0)
    state.setdefault("history", [])
    state.setdefault("result", None)
    state.setdefault("critic", [])
    state.setdefault("activity", [])
    state.setdefault("viewing", None)
    state.setdefault("section", "Configure")
    state.setdefault("nav_epoch", 0)
    state.setdefault("output_dir", "artifacts_sysid")
    state.setdefault("results_panel_open", False)

    runner: Optional[PipelineRunner] = state["runner"]
    running = bool(runner and runner.running)

    # --- Brand bar --------------------------------------------------------
    if running:
        status_label, status_kind = f"Running · {state['stage']}", "run"
    elif state["result"] is not None:
        status_label = "Completed" if state["result"].status == "completed" else "Failed"
        status_kind = "done" if state["result"].status == "completed" else "fail"
    else:
        status_label, status_kind = "Idle", "idle"
    st.markdown(T.brand_bar("System Identification", status_label, status_kind),
                unsafe_allow_html=True)

    render_sidebar(state["output_dir"], running)

    # --- Navigation -------------------------------------------------------
    nav_col, act_col, preview_control = st.columns([4, 1.6, 2], gap="small")
    with nav_col:
        # A segmented control reads as product navigation; a radio reads as a form field.
        choice = st.segmented_control(
            "section", SECTIONS, key=f"nav_{state['nav_epoch']}",
            default=state["section"], label_visibility="collapsed",
        )
        # The control allows deselection; treat that as "stay where you are".
        if choice and choice != state["section"]:
            state["section"] = choice
    section = state["section"]
    with preview_control:
        st.button("Hide results" if state["results_panel_open"] else "Show results",
                  icon=":material/dock_to_right:", key="toggle_results_panel",
                  on_click=_toggle_results_panel, width="stretch",
                  help="Show or hide plots, reports, and generated code beside the workspace.")

    if state["results_panel_open"]:
        main_workspace, results_panel = st.columns([1.8, 1], gap="medium")
    else:
        main_workspace, results_panel = st.container(), None

    options: Optional[SysIDOptions] = None
    ready = False
    with main_workspace:
        if section == "Configure":
            options, ready = render_configure()
            state["pending_options"] = options
        else:
            options = state.get("pending_options")
            ready = options is not None

    if options is not None:
        state["output_dir"] = options.output_dir

    with act_col:
        if section == "Configure" and ready and not running:
            start_clicked = st.button(
                "Start run", type="primary", icon=":material/play_arrow:",
                width="stretch",
            )
        else:
            start_clicked = False
        if start_clicked:
            state.update(log="", stage=STAGES[0], progress=0.0, history=[],
                         result=None, critic=[], viewing=None, activity=[],
                         activity_saved=False, activity_save_error=False)
            state["runner"] = PipelineRunner(options)
            state["runner"].start()
            goto("Monitor")
            st.rerun()
        if running and st.button("Stop run", icon=":material/stop:", width="stretch"):
            request_stop()
            activity.record(state["activity"], "stop_requested", {})
            st.toast("Stop requested — finishing the current step and compiling results.")

    # --- Drain the worker's queues ---------------------------------------
    if runner is not None and section != "Monitor":
        drain(runner, state)

    # --- Sections ---------------------------------------------------------
    with main_workspace:
        if section == "Monitor":
            if running:
                live_monitor(options)     # refreshes itself, no page-wide flicker
            else:
                render_monitor(state, options, running)

        elif section == "Results":
            viewing = state["viewing"]
            result: Optional[SysIDResult] = state["result"]
            ask_entry = viewing or (result.to_dict() if result is not None and result.run_dir else None)
            if ask_entry:
                st.button("Ask about this run", icon=":material/chat:",
                          on_click=_ask_run, args=(ask_entry,), key="ask_selected_run")
            if viewing is not None:
                st.caption(f"Viewing a past run — {_history_text(hist.display_name(viewing))}")
                render_results_from_entry(viewing)
            elif result is not None and not running:
                if result.status == "completed":
                    render_results(result)
                else:
                    st.error(result.message or "The run did not complete.")
            elif running:
                st.info("The run is still going — see **Monitor** for live progress.")
            else:
                st.info("No result yet. Start a run, or open one from the history sidebar.")

            if runner is not None and runner.error and not running and viewing is None:
                st.error("The pipeline raised an exception.")
                st.code(runner.error)

        elif section == "Compare":
            preferred = workspace.run_id(state["viewing"]) if state["viewing"] else None
            workspace.render_compare(state["output_dir"], preferred=preferred)

        elif section == "Ask run":
            preferred = workspace.run_id(state["viewing"]) if state["viewing"] else None
            run_chat.render_run_chat(state["output_dir"], preferred=preferred, runner=runner)

    if results_panel is not None:
        with results_panel:
            workspace.render_panel(state["output_dir"], viewed=state["viewing"],
                                   result=state["result"], running=running)

    # --- Keep non-Monitor sections fresh while a run is in flight --------
    # The Monitor refreshes itself via the fragment above; elsewhere a slow
    # page-level tick is enough to keep the brand-bar status honest.
    if running and section != "Monitor":
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
