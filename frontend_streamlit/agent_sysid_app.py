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

import json
import queue
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

import ui_history as hist  # noqa: E402
import ui_theme as T  # noqa: E402

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
SECTIONS = ["Configure", "Monitor", "Results"]

st.set_page_config(
    page_title="LabCD · System Identification",
    page_icon="◆",
    layout="wide",
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

    def write(self, text: str) -> int:
        if threading.get_ident() == self._target:
            if text:
                self._sink.put(text)
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

    def _on_event(self, kind: str, payload: Dict[str, Any]) -> None:
        self.events.put((kind, payload))

    def _run(self) -> None:
        original = sys.stdout
        sys.stdout = _ThreadRouter(threading.get_ident(), self.logs, original)
        try:
            self.result = run_pipeline(
                self.options, on_event=self._on_event, install_signal_handler=False
            )
        except Exception as exc:  # noqa: BLE001 - surface it in the UI
            import traceback

            self.error = f"{exc}\n\n{traceback.format_exc()}"
            self.events.put(("error", {"message": str(exc)}))
        finally:
            sys.stdout = original
            self.events.put(("finished", {}))

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
def render_sidebar(output_dir: str, running: bool) -> None:
    sb = st.sidebar
    sb.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:2px 0 12px'>"
        f"<div class='brand-mark' style='width:28px;height:28px;font-size:14px'>L</div>"
        f"<div><div style='font-weight:600;font-size:14px'>LabCD<span style='color:{T.TEXT_FAINT};font-weight:400'>.ai</span></div>"
        f"<div class='brand-sub' style='font-size:9.5px'>System Identification</div></div></div>",
        unsafe_allow_html=True,
    )

    if sb.button("＋  New run", width="stretch", type="primary", disabled=running):
        st.session_state["viewing"] = None
        st.session_state["result"] = None
        goto("Configure")
        st.rerun()

    sb.markdown("<div class='side-heading'>Run history</div>", unsafe_allow_html=True)

    runs = hist.load_history(output_dir)
    if not runs:
        sb.markdown(
            "<div class='hist-empty'>No runs yet.<br>Configure a run and press "
            "<b>Start</b> — it will appear here.</div>",
            unsafe_allow_html=True,
        )
        return

    query = sb.text_input(
        "Search", "", placeholder="filter by dataset or architecture…",
        label_visibility="collapsed",
    ).strip().lower()

    shown = 0
    for entry in runs:
        label = entry.get("env_name") or entry.get("name", "run")
        summary = hist.summarise(entry)
        if query and query not in f"{label} {summary}".lower():
            continue
        shown += 1

        score = entry.get("success_score")
        age = hist.relative_age(entry.get("started"))
        if entry.get("complete") and score is not None:
            badge = f"{float(score):.0f}"
            colour = T.score_color(float(score))
        else:
            badge = "—"
            colour = T.TEXT_FAINT

        sb.markdown(
            f"<div class='hist-row'>"
            f"<div class='hist-line'>"
            f"<span class='hist-score' style='color:{colour}'>{badge}</span>"
            f"<span class='hist-name'>{label}</span>"
            f"<span class='hist-age'>{age}</span>"
            f"</div>"
            f"<div class='hist-meta'>{summary}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        col_open, col_del = sb.columns([4, 1])
        if col_open.button("Open", key=f"open_{entry['name']}", width="stretch"):
            st.session_state["viewing"] = entry
            goto("Results")
            st.rerun()
        if col_del.button("🗑", key=f"del_{entry['name']}", help="Delete this run"):
            hist.delete_run(entry["run_dir"])
            if (st.session_state.get("viewing") or {}).get("run_dir") == entry["run_dir"]:
                st.session_state["viewing"] = None
            st.rerun()

    if shown == 0:
        sb.caption("No runs match that filter.")


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
def render_configure() -> tuple[Optional[SysIDOptions], bool]:
    st.markdown("#### Dataset")
    c1, c2 = st.columns([3, 2])
    uploaded = c1.file_uploader(
        "Upload CSV / Excel", type=["csv", "xlsx", "xls"],
        help="Columns: time, s_* states, a_* actions, optional xdot_* derivatives "
             "(see data/DATA_CONTRACT.md).",
    )
    use_example = c2.checkbox(
        "Use the bundled example dataset", value=not uploaded,
        help=str(DEFAULT_EXAMPLE.relative_to(_REPO_ROOT)),
    )
    output_dir = c2.text_input("Output directory", "artifacts_sysid")

    data_path: Optional[str] = None
    if uploaded is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        data_path = str(target)
    elif use_example and DEFAULT_EXAMPLE.is_file():
        data_path = str(DEFAULT_EXAMPLE)

    if data_path:
        render_data_preview(data_path)

    # --- Run budget -------------------------------------------------------
    st.markdown("#### Run budget")
    b1, b2, b3, b4 = st.columns(4)
    run_mode = b1.selectbox(
        "Run mode", ["fast", "regular", "heavy"], index=1,
        help="fast = 7 cycles / 0.5 h · regular = 20 / 1.5 h · heavy = 40 / 4 h",
    )
    limits = cfg.run_mode_limits(run_mode)
    max_cycles = b2.text_input("Max cycles", "", placeholder=str(limits["max_cycles"]))
    epochs = b3.text_input("Epochs / cycle", "", placeholder=str(D("EPOCHS")))
    save_plot = b4.checkbox("Diagnostic figures", value=True)
    st.caption(
        f"{limits['max_cycles']} cycles · {limits['max_hours']} h · Critic explores to cycle "
        f"{limits['critic_explore_limit']} · failure memory: {limits['memory_capacity']} · "
        f"customer context: {limits['context_status']}"
    )

    # --- Questionnaire ----------------------------------------------------
    st.markdown("#### Dataset questionnaire")
    st.caption("The three questions the terminal asks before training.")
    q1, q2 = st.columns([3, 2])
    customer_description = q1.text_area(
        "System description", value=D("CUSTOMER_SYSTEM_DESCRIPTION"), height=110,
        help="Free text fed to every agent as physical context.",
    )
    angle_mode = q2.radio("Angular states (wrap at ±π)",
                          ["None", "Specify manually", "Auto-detect"])
    angle_text = ""
    if angle_mode == "Specify manually":
        angle_text = q2.text_input("State indices (0-based)", "", placeholder="2, 4")

    t1, t2 = st.columns([3, 2])
    traj_mode = t1.radio("Trajectory structure",
                         ["Single continuous run", "Several stacked trajectories"],
                         horizontal=True)
    split_mode, split_times_text = "Auto-detect boundaries", ""
    if traj_mode == "Several stacked trajectories":
        split_mode = t2.radio("Boundaries",
                              ["Auto-detect boundaries", "I know the timestamps"])
        if split_mode == "I know the timestamps":
            split_times_text = t2.text_input("Split timestamps (s)", "", placeholder="12.5, 25.0")

    # --- Advanced ---------------------------------------------------------
    st.markdown("#### Model & physics")
    col_l, col_r = st.columns(2)

    with col_l.expander("Architecture & rollout", expanded=True):
        architecture = st.selectbox(
            "Network", ["LSTM", "MLP"],
            index=0 if str(D("NETWORK_ARCHITECTURE")).upper() == "LSTM" else 1,
        )
        lstm_seq_length = st.number_input(
            "LSTM memory window (steps)", 2, 200, int(D("LSTM_SEQ_LENGTH")),
            disabled=architecture != "LSTM",
        )
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

    with col_r.expander("Derivative estimation", expanded=True):
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

    col_l2, col_r2 = st.columns(2)
    with col_l2.expander("State-space filter"):
        use_state_filter = st.checkbox("Clip state outliers", value=bool(D("USE_STATE_FILTER")))
        pct_low, pct_high = st.slider(
            "Percentile band", 0.0, 100.0,
            (float(D("AUTO_FILTER_PERCENTILES")[0]), float(D("AUTO_FILTER_PERCENTILES")[1])),
            disabled=not use_state_filter,
        )

    with col_r2.expander("Physics-informed (PINN)"):
        use_pinn = st.checkbox("Enable physics residual", value=bool(D("USE_PINN")))
        pinn_loss_weight = st.number_input(
            "Physics loss weight λ", 0.0, 10.0, float(D("PINN_LOSS_WEIGHT")),
            step=0.1, disabled=not use_pinn,
        )
        pinn_equation_file = st.text_input(
            "Equation file", str(D("PINN_EQUATION_FILE")), disabled=not use_pinn,
            help="Must define compute_analytical_xdot(states, actions). "
                 "A template is generated if missing.",
        )

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

    st.markdown("#### Agents")
    col_l4, col_r4 = st.columns(2)
    with col_l4.expander("Initializer Agent"):
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
        ov_patience = o2.text_input("early_stop_patience", "", placeholder=str(D("EARLY_STOP_PATIENCE")))

    with st.expander("LLM provider"):
        p1, p2, p3 = st.columns(3)
        providers = ["openai", "groq", "openrouter"]
        api_provider = p1.selectbox(
            "Provider", providers,
            index=providers.index(D("API_PROVIDER")) if D("API_PROVIDER") in providers else 0,
        )
        llm_model = p2.text_input("Model", D("LLM_MODEL"))
        llm_temperature = p3.slider("Temperature", 0.0, 2.0, float(D("LLM_TEMPERATURE")), 0.05)
        st.caption(
            "No API key? Every agent falls back to its deterministic mathematical "
            "path and the run still produces the full deliverable."
        )

    if data_path is None:
        return None, False

    # --- Assemble ---------------------------------------------------------
    if angle_mode == "None":
        angle_indices, auto_detect_angles = [], False
    elif angle_mode == "Specify manually":
        angle_indices, auto_detect_angles = (_int_list(angle_text) or []), False
    else:
        angle_indices, auto_detect_angles = [], True

    multi_trajectory = traj_mode != "Single continuous run"
    manual_split_times: List[float] = []
    if multi_trajectory and split_mode == "I know the timestamps":
        manual_split_times = sorted(_float_list(split_times_text) or [])

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

    options = SysIDOptions(
        data_path=data_path,
        run_mode=run_mode,
        output_dir=output_dir or "artifacts_sysid",
        interactive=False,  # a web UI must never block on input()
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
        save_plot=bool(save_plot),
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
    return chart.properties(height=height).configure_view(
        stroke=T.BORDER, fill=T.BG_CARD
    ).configure_axis(
        grid=True, gridColor=T.BORDER, gridOpacity=0.5,
        labelColor=T.TEXT_DIM, titleColor=T.TEXT_DIM,
        domainColor=T.BORDER, tickColor=T.BORDER,
    ).configure_legend(labelColor=T.TEXT_DIM).configure(background=T.BG_CARD)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def render_stage_rail(current_stage: str, finished: bool) -> None:
    done = True
    parts = []
    for name in STAGES:
        if finished:
            cls, icon = "rail-done", "✓"
        elif name == current_stage:
            cls, icon, done = "rail-active", "●", False
        elif done:
            cls, icon = "rail-done", "✓"
        else:
            cls, icon = "rail-todo", "○"
        parts.append(f"<span class='rail-step {cls}'>{icon} {name}</span>")
    st.markdown(f"<div class='rail'>{''.join(parts)}</div>", unsafe_allow_html=True)


def render_monitor(state, options: Optional[SysIDOptions], running: bool) -> None:
    if not state["log"] and not running:
        st.info("No active run. Configure a run and press **Start** to watch it here.")
        return

    render_stage_rail(state["stage"], finished=not running and state["result"] is not None)
    st.progress(min(max(state["progress"], 0.0), 1.0))

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
            c1.altair_chart(chart, use_container_width=True)
        max_lat = float(options.customer_max_latency_ms) if options else cfg.CUSTOMER_MAX_LATENCY_MS
        lat = latency_chart(history, max_lat)
        if lat is not None:
            c2.markdown("##### Latency against the customer limit")
            c2.altair_chart(lat, use_container_width=True)
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

    if state["critic"]:
        with st.expander("Critic decisions"):
            for c in state["critic"]:
                st.markdown(
                    f"**Cycle {c['cycle']} — `{c.get('diagnosis')}`** "
                    f"(LR {c.get('lr_dir')} by {c.get('lr_step')}, next {c.get('hidden_layers')})"
                    f"  \n{c.get('reasoning', '')}"
                )

    if state["log"]:
        st.markdown("##### Agent log")
        st.caption("Live console output from the core — the same text the terminal prints.")
        with st.container(height=380, border=True):
            st.code(state["log"][-14000:], language="text")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_results_from_entry(entry: Dict[str, Any]) -> None:
    """Render a past run, read back from its manifest."""
    if not entry.get("complete"):
        st.warning(
            f"`{entry.get('name')}` has no manifest — it was interrupted before packaging. "
            "Any artefacts it did write are below."
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
        just_finished = True
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
    state.setdefault("viewing", None)
    state.setdefault("section", "Configure")
    state.setdefault("nav_epoch", 0)
    state.setdefault("output_dir", "artifacts_sysid")

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
    nav_col, act_col = st.columns([3, 2])
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

    options: Optional[SysIDOptions] = None
    ready = False
    if section == "Configure":
        options, ready = render_configure()
        state["pending_options"] = options
    else:
        options = state.get("pending_options")
        ready = options is not None

    if options is not None:
        state["output_dir"] = options.output_dir

    with act_col:
        a1, a2 = st.columns(2)
        if a1.button("▶  Start", type="primary", disabled=running or not ready,
                     width="stretch"):
            state.update(log="", stage=STAGES[0], progress=0.0, history=[],
                         result=None, critic=[], viewing=None)
            state["runner"] = PipelineRunner(options)
            state["runner"].start()
            goto("Monitor")
            st.rerun()
        if a2.button("■  Stop", disabled=not running, width="stretch"):
            request_stop()
            st.toast("Stop requested — finishing the current step and compiling results.")

    if section == "Configure" and not ready:
        st.info("Upload a dataset, or tick **Use the bundled example dataset**, to enable Start.")

    # --- Drain the worker's queues ---------------------------------------
    if runner is not None and section != "Monitor":
        drain(runner, state)

    # --- Sections ---------------------------------------------------------
    if section == "Monitor":
        if running:
            live_monitor(options)     # refreshes itself, no page-wide flicker
        else:
            render_monitor(state, options, running)

    elif section == "Results":
        viewing = state["viewing"]
        result: Optional[SysIDResult] = state["result"]
        if viewing is not None:
            st.caption(f"Viewing a past run — {viewing.get('name')}")
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

    # --- Keep non-Monitor sections fresh while a run is in flight --------
    # The Monitor refreshes itself via the fragment above; elsewhere a slow
    # page-level tick is enough to keep the brand-bar status honest.
    if running and section != "Monitor":
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
