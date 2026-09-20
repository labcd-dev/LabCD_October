#!/usr/bin/env python3
"""
AgentSysID — Streamlit reference UI.

Run from the repository root:

    PYTHONPATH=. streamlit run frontend_streamlit/agent_sysid_app.py --server.port 8504

or use the launcher:

    PYTHONPATH=. python frontend_streamlit/run_agent_sysid_ui.py

The UI is deliberately thin: every control maps to a field on
``backend_core.AgentSysID.pipeline.SysIDOptions`` and the run itself is
``run_pipeline`` on a worker thread. No training, agent or reporting logic is
duplicated here — this file only collects options, streams events, and renders
the artefacts the core produced.

Everything the interactive CLI asks for is exposed:
  * the dataset questionnaire (context, angular states, trajectory structure)
  * the HIL Data Inspector findings
  * all fourteen Initializer overrides (blank = keep the agent's choice)
  * architecture, rollout, integrator, derivative estimator, state filter,
    PINN, training limits, search bounds, manual presets and the LLM provider
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# --- Make the repo root importable no matter where streamlit is launched ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend_core.AgentSysID import config as cfg  # noqa: E402
from backend_core.AgentSysID.pipeline import (  # noqa: E402
    STAGES,
    SysIDOptions,
    SysIDResult,
    run_pipeline,
)
from backend_core.AgentSysID.utils import request_stop, reset_stop_flag  # noqa: E402

# ---------------------------------------------------------------------------
# Pristine config snapshot.
#
# A run mutates the config module in place (the Initializer rewrites the search
# bounds, the questionnaire rewrites the angular states, and so on) — that is
# how the core has always worked. The Streamlit server is long-lived, so
# reading cfg.X for a widget default would make the form drift to the previous
# run's values. Snapshot the pristine values once, at import, and build the
# form from those.
# ---------------------------------------------------------------------------
_CFG_DEFAULTS = {
    name: getattr(cfg, name)
    for name in dir(cfg)
    if name.isupper() and not name.startswith("_")
}


def D(name: str, fallback=None):
    """A widget default, taken from the pristine config snapshot."""
    return _CFG_DEFAULTS.get(name, fallback)


# ---------------------------------------------------------------------------
# Design tokens — carried over from frontend_mockup/labcd_sysid.html.
# The two series colors are validated for the dark chart surface (#131316):
# lightness band, chroma floor, CVD separation, normal-vision floor, contrast.
# ---------------------------------------------------------------------------
BG = "#0b0b0d"
BG_CARD = "#131316"
BORDER = "#212126"
TEXT = "#e9e8e4"
TEXT_DIM = "#949398"
ACCENT = "#6e79f0"      # series 1 — train
CURVE_B = "#d4794a"     # series 2 — validation (snapped into the dark band)
GOOD = "#4bd1a0"        # status only, never a series
WARN = "#e0a94f"
CRIT = "#e06a5a"

UPLOAD_DIR = _REPO_ROOT / ".streamlit_uploads"
DEFAULT_EXAMPLE = _REPO_ROOT / "backend_core/AgentSysID/data/examples/synthetic_oscillator.csv"

st.set_page_config(
    page_title="AgentSysID — LabCD Studio",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
      .stApp {{ background:{BG}; color:{TEXT}; }}
      html, body, [class*="css"] {{ font-family:'IBM Plex Sans', -apple-system, sans-serif; }}
      code, pre, .mono {{ font-family:'IBM Plex Mono', ui-monospace, monospace !important; }}
      h1, h2, h3 {{ color:{TEXT}; font-weight:600; letter-spacing:-0.01em; }}
      /* Streamlit's own chrome, themed to the app surface */
      header[data-testid="stHeader"] {{ background:transparent; }}
      section[data-testid="stSidebar"] {{ background:{BG_CARD}; border-right:1px solid {BORDER}; }}
      section[data-testid="stSidebar"] label, .stApp label {{ color:{TEXT} !important; }}
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] .stCaption,
      div[data-testid="stCaptionContainer"] {{ color:{TEXT_DIM} !important; }}
      section[data-testid="stSidebar"] h4 {{
          color:{TEXT}; font-size:12px; text-transform:uppercase;
          letter-spacing:0.08em; margin:18px 0 2px;
      }}
      div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:11px; }}
      div[data-testid="stMetric"] {{
          background:{BG_CARD}; border:1px solid {BORDER};
          border-radius:11px; padding:14px 16px;
      }}
      div[data-testid="stMetricLabel"] {{ color:{TEXT_DIM}; font-size:11px;
          text-transform:uppercase; letter-spacing:0.06em; }}
      .stButton>button {{ border-radius:7px; border:1px solid {BORDER}; font-weight:500; }}
      .pill {{
          display:inline-block; padding:3px 10px; border-radius:999px;
          font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.04em;
      }}
      .stage-done {{ color:{GOOD}; }}
      .stage-active {{ color:{ACCENT}; font-weight:600; }}
      .stage-todo {{ color:#5c5b61; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
class _ThreadRouter:
    """
    A stdout proxy that sends the worker thread's prints to a queue and
    leaves every other thread's output on the real stdout.
    """

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
# Small parsing helpers (blank input == keep the agent's / config's choice)
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


# ---------------------------------------------------------------------------
# Sidebar — every option the CLI exposes
# ---------------------------------------------------------------------------
def build_sidebar() -> tuple[Optional[SysIDOptions], bool]:
    st.sidebar.markdown("### ◆ AgentSysID")
    st.sidebar.caption("Agentic system identification · LabCD")

    # --- Dataset ---------------------------------------------------------
    st.sidebar.markdown("#### Dataset")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV / Excel", type=["csv", "xlsx", "xls"],
        help="Columns: time, s_* states, a_* actions, optional xdot_* derivatives "
             "(see data/DATA_CONTRACT.md).",
    )
    use_example = st.sidebar.checkbox(
        "Use the bundled example dataset", value=not uploaded,
        help=str(DEFAULT_EXAMPLE.relative_to(_REPO_ROOT)),
    )

    data_path: Optional[str] = None
    if uploaded is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        data_path = str(target)
        st.sidebar.success(f"Loaded `{uploaded.name}`")
    elif use_example and DEFAULT_EXAMPLE.is_file():
        data_path = str(DEFAULT_EXAMPLE)

    # --- Run budget -------------------------------------------------------
    st.sidebar.markdown("#### Run budget")
    run_mode = st.sidebar.selectbox(
        "Run mode", ["fast", "regular", "heavy"], index=1,
        help="fast = 7 cycles / 0.5 h · regular = 20 / 1.5 h · heavy = 40 / 4 h",
    )
    limits = cfg.run_mode_limits(run_mode)
    st.sidebar.caption(
        f"{limits['max_cycles']} cycles · {limits['max_hours']} h · "
        f"explore to cycle {limits['critic_explore_limit']} · memory: {limits['memory_capacity']}"
    )
    col_a, col_b = st.sidebar.columns(2)
    max_cycles = col_a.text_input("Max cycles", "", placeholder=str(limits["max_cycles"]))
    epochs = col_b.text_input("Epochs / cycle", "", placeholder=str(D("EPOCHS")))
    output_dir = st.sidebar.text_input("Output directory", "artifacts_sysid")

    # --- Questionnaire ----------------------------------------------------
    with st.sidebar.expander("Dataset questionnaire", expanded=True):
        st.caption("The three questions the terminal asks before training.")
        customer_description = st.text_area(
            "System description",
            value=D("CUSTOMER_SYSTEM_DESCRIPTION"),
            height=90,
            help="Free text fed to every agent as physical context.",
        )
        angle_mode = st.radio(
            "Angular states (wrap at ±π)",
            ["None", "Specify manually", "Auto-detect"],
            horizontal=False,
        )
        angle_text = ""
        if angle_mode == "Specify manually":
            angle_text = st.text_input("State indices (0-based)", "", placeholder="2, 4")

        traj_mode = st.radio(
            "Trajectory structure",
            ["Single continuous run", "Several stacked trajectories"],
        )
        split_mode, split_times_text = "Auto-detect boundaries", ""
        if traj_mode == "Several stacked trajectories":
            split_mode = st.radio(
                "Boundaries", ["Auto-detect boundaries", "I know the timestamps"]
            )
            if split_mode == "I know the timestamps":
                split_times_text = st.text_input(
                    "Split timestamps (s)", "", placeholder="12.5, 25.0"
                )

    # --- Architecture -----------------------------------------------------
    with st.sidebar.expander("Architecture & rollout"):
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

    # --- Derivatives ------------------------------------------------------
    with st.sidebar.expander("Derivative estimation"):
        st.caption("Only used when the dataset has no xdot_* columns.")
        derivative_method = st.selectbox(
            "Method", ["finite_difference", "savitzky_golay", "sliding_mode"],
            index=["finite_difference", "savitzky_golay", "sliding_mode"].index(
                str(D("DERIVATIVE_METHOD"))
            ) if str(D("DERIVATIVE_METHOD")) in
            ["finite_difference", "savitzky_golay", "sliding_mode"] else 0,
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

    # --- State filter -----------------------------------------------------
    with st.sidebar.expander("State-space filter"):
        use_state_filter = st.checkbox("Clip state outliers", value=bool(D("USE_STATE_FILTER")))
        pct_low, pct_high = st.slider(
            "Percentile band", 0.0, 100.0,
            (float(D("AUTO_FILTER_PERCENTILES")[0]), float(D("AUTO_FILTER_PERCENTILES")[1])),
            disabled=not use_state_filter,
        )

    # --- PINN -------------------------------------------------------------
    with st.sidebar.expander("Physics-informed (PINN)"):
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

    # --- Training ---------------------------------------------------------
    with st.sidebar.expander("Training limits"):
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

    # --- Search bounds ----------------------------------------------------
    with st.sidebar.expander("Search bounds (client-authorised)"):
        lr_lo = st.number_input("LR min", 1e-8, 1.0, float(D("LEARNING_RATE_MIN")), format="%.8f")
        lr_hi = st.number_input("LR max", 1e-8, 1.0, float(D("LEARNING_RATE_MAX")), format="%.8f")
        hs_lo = st.number_input("Hidden size min", 1, 4096, int(D("HIDDEN_SIZE_MIN")))
        hs_hi = st.number_input("Hidden size max", 1, 4096, int(D("HIDDEN_SIZE_MAX")))
        nl_lo = st.number_input("Layer count min", 1, 32, int(D("NUM_LAYERS_MIN")))
        nl_hi = st.number_input("Layer count max", 1, 32, int(D("NUM_LAYERS_MAX")))

    # --- Initializer ------------------------------------------------------
    with st.sidebar.expander("Initializer Agent"):
        choose_via_llm = st.checkbox(
            "Let the Initializer Agent choose", value=bool(D("CHOOSE_VIA_LLM_INITIALIZER"))
        )
        st.caption(
            "Manual presets, used when the agent is switched off:"
            if not choose_via_llm
            else "Overrides below replace the terminal's review prompt. "
                 "Leave a field blank to keep the agent's choice."
        )
        manual_starting_lr = st.number_input(
            "Preset LR", 1e-8, 1.0, float(D("MANUAL_STARTING_LR")),
            format="%.6f", disabled=choose_via_llm,
        )
        manual_hidden_text = st.text_input(
            "Preset topology", ", ".join(str(x) for x in D("MANUAL_STARTING_HIDDEN_LAYERS")),
            disabled=choose_via_llm,
        )
        manual_activation = st.selectbox(
            "Preset activation", cfg.AVAILABLE_ACTIVATIONS,
            index=cfg.AVAILABLE_ACTIVATIONS.index(D("MANUAL_ACTIVATION"))
            if D("MANUAL_ACTIVATION") in cfg.AVAILABLE_ACTIVATIONS else 0,
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
            "Client-locked parameters (JSON)",
            value="",
            height=70,
            placeholder='{"hidden_layers": [128, 128]}',
            help="Echoed into the Initializer prompt as parameters the client has "
                 "locked — the agent must return these exact values.",
        )

    # --- The fourteen overrides ------------------------------------------
    with st.sidebar.expander("Override the agent's proposal"):
        st.caption(
            "The same fourteen parameters the terminal offers. "
            "Blank = keep the agent's choice."
        )
        ov_lr = st.text_input("learning_rate", "", placeholder="e.g. 0.001")
        ov_layers = st.text_input("hidden_layers", "", placeholder="e.g. 128, 64")
        ov_activation = st.selectbox(
            "activation", ["(keep agent's choice)"] + list(cfg.AVAILABLE_ACTIVATIONS)
        )
        ov_dropout = st.text_input("dropout_rate", "", placeholder="0.0 – 0.5")
        ov_wd = st.text_input("weight_decay", "", placeholder="e.g. 0.0001")
        ov_filter = st.selectbox("use_state_filter", ["(keep agent's choice)", "True", "False"])
        ov_pct = st.text_input("auto_filter_percentiles", "", placeholder="2, 98")
        ov_tau = st.text_input("derivative_filter_tau", "", placeholder="e.g. 0.005")
        ov_lr_bounds = st.text_input("LR search bounds", "", placeholder="0.00005, 0.001")
        ov_hs_bounds = st.text_input("Hidden size bounds", "", placeholder="32, 256")
        ov_nl_bounds = st.text_input("Layer depth bounds", "", placeholder="1, 3")
        ov_epochs = st.text_input("epochs", "", placeholder=str(D("EPOCHS")))
        ov_batch = st.text_input("batch_size", "", placeholder=str(D("BATCH_SIZE")))
        ov_patience = st.text_input("early_stop_patience", "", placeholder=str(D("EARLY_STOP_PATIENCE")))

    # --- LLM --------------------------------------------------------------
    with st.sidebar.expander("LLM provider"):
        api_provider = st.selectbox(
            "Provider", ["openai", "groq", "openrouter"],
            index=["openai", "groq", "openrouter"].index(D("API_PROVIDER"))
            if D("API_PROVIDER") in ["openai", "groq", "openrouter"] else 0,
        )
        llm_model = st.text_input("Model", D("LLM_MODEL"))
        llm_temperature = st.slider("Temperature", 0.0, 2.0, float(D("LLM_TEMPERATURE")), 0.05)
        st.caption(
            "No API key? Every agent falls back to its deterministic "
            "mathematical path and the run still completes."
        )

    save_plot = st.sidebar.checkbox("Generate diagnostic figures", value=True)

    # --- Assemble ---------------------------------------------------------
    if data_path is None:
        return None, False

    angle_indices: Optional[List[int]] = None
    auto_detect_angles: Optional[bool] = None
    if angle_mode == "None":
        angle_indices, auto_detect_angles = [], False
    elif angle_mode == "Specify manually":
        angle_indices, auto_detect_angles = (_int_list(angle_text) or []), False
    else:
        angle_indices, auto_detect_angles = [], True

    multi_trajectory = traj_mode != "Single continuous run"
    manual_split_times: Optional[List[float]] = []
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
                st.sidebar.warning("Client-locked parameters must be a JSON object.")
        except json.JSONDecodeError as exc:
            st.sidebar.warning(f"Client-locked parameters: invalid JSON ({exc.msg}).")

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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def convergence_chart(history: List[Dict[str, Any]]):
    """
    Train vs validation MSE across tuning cycles.

    Two series, so a legend is always present; both are also direct-labelled by
    the tooltip on hover. Log y-axis because the error spans orders of
    magnitude. Separate chart from latency — never a second y-axis.
    """
    import altair as alt

    rows = []
    for p in history:
        cycle = p.get("cycle", p.get("iteration", 0) + 1)
        for label, key in (("Train MSE", "train_mse"), ("Validation MSE", "val_mse")):
            value = p.get(key)
            if value is None or value != value or value <= 0:  # NaN / non-positive
                continue
            rows.append({"Cycle": cycle, "Series": label, "MSE": float(value)})
    if not rows:
        return None

    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Cycle:Q", axis=alt.Axis(tickMinStep=1, title="Actor-Critic cycle")),
        y=alt.Y("MSE:Q", scale=alt.Scale(type="log"), axis=alt.Axis(title="MSE (log)", format=".1e")),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=["Train MSE", "Validation MSE"], range=[ACCENT, CURVE_B]),
            legend=alt.Legend(title=None, orient="top", labelColor=TEXT_DIM, symbolType="stroke"),
        ),
    )
    line = base.mark_line(strokeWidth=2, interpolate="monotone")
    points = base.mark_point(size=80, filled=True, stroke=BG_CARD, strokeWidth=2).encode(
        tooltip=[
            alt.Tooltip("Cycle:Q", title="Cycle"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("MSE:Q", title="MSE", format=".3e"),
        ]
    )
    return (line + points).properties(height=260).configure_view(
        stroke=BORDER, fill=BG_CARD
    ).configure_axis(
        grid=True, gridColor=BORDER, gridOpacity=0.5,
        labelColor=TEXT_DIM, titleColor=TEXT_DIM, domainColor=BORDER, tickColor=BORDER,
    ).configure_legend(labelColor=TEXT_DIM).configure(background=BG_CARD)


def latency_chart(history: List[Dict[str, Any]], max_latency: float):
    """
    Measured inference latency per cycle against the customer's hard limit.

    One series, so no legend — the title names it. The limit is a reference
    rule, not a second series.
    """
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
    line = base.mark_line(strokeWidth=2, color=ACCENT, interpolate="monotone")
    points = base.mark_point(size=80, filled=True, color=ACCENT, stroke=BG_CARD, strokeWidth=2).encode(
        tooltip=[
            alt.Tooltip("Cycle:Q", title="Cycle"),
            alt.Tooltip("Latency:Q", title="Latency (ms)", format=".3f"),
        ]
    )
    rule_df = pd.DataFrame(
        {
            "limit": [max_latency],
            "Cycle": [float(df["Cycle"].min())],
            "text": [f"customer limit {max_latency:g} ms"],
        }
    )
    limit = (
        alt.Chart(rule_df)
        .mark_rule(color=CRIT, strokeDash=[6, 4], strokeWidth=2)
        .encode(y="limit:Q")
    )
    # Direct-label the rule inside the plot area so it is never clipped.
    label = (
        alt.Chart(rule_df)
        .mark_text(align="left", baseline="bottom", dx=4, dy=-4, color=CRIT, fontSize=11)
        .encode(x="Cycle:Q", y="limit:Q", text="text:N")
    )
    return (line + points + limit + label).properties(height=220).configure_view(
        stroke=BORDER, fill=BG_CARD
    ).configure_axis(
        grid=True, gridColor=BORDER, gridOpacity=0.5,
        labelColor=TEXT_DIM, titleColor=TEXT_DIM, domainColor=BORDER, tickColor=BORDER,
    ).configure(background=BG_CARD)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_stages(current_stage: str) -> None:
    done = True
    parts = []
    for name in STAGES:
        if name == current_stage:
            parts.append(f"<span class='stage-active'>● {name}</span>")
            done = False
        elif done:
            parts.append(f"<span class='stage-done'>✓ {name}</span>")
        else:
            parts.append(f"<span class='stage-todo'>○ {name}</span>")
    st.markdown(
        "<div class='mono' style='font-size:11.5px; line-height:2.0'>"
        + " &nbsp;·&nbsp; ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_results(result: SysIDResult) -> None:
    st.markdown("### Result")

    status_color = {
        "STABLE & HIGH-FIDELITY": GOOD,
        "STABLE & ACCEPTABLE": GOOD,
        "UNSTABLE ROLLOUT": WARN,
        "UNSTABLE / FAILED": CRIT,
    }.get(result.model_status, TEXT_DIM)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Composite score", f"{result.success_score:.1f} / 100")
    c2.metric("Validation MSE", f"{result.best_mse:.3e}")
    c3.metric("Validation RMSE", f"{result.best_rmse:.3e}")
    c4.metric("Inference latency", f"{result.latency_ms:.3f} ms")
    c5.metric("Cycles run", str(result.cycles_run))

    st.markdown(
        f"<span class='pill mono' style='background:{status_color}22;color:{status_color};"
        f"border:1px solid {status_color}55'>{result.model_status}</span>"
        f"&nbsp;&nbsp;<span class='mono' style='color:{TEXT_DIM};font-size:11.5px'>"
        f"{result.complexity_label} · {result.state_dim} states · {result.action_dim} inputs · "
        f"topology {result.best_config.get('hidden_layers')} · {result.activation.upper()} · "
        f"{result.llm_calls} LLM calls (${result.llm_cost_usd:.5f})</span>",
        unsafe_allow_html=True,
    )

    # --- Downloads --------------------------------------------------------
    st.markdown("#### Deliverables")
    files = [
        ("Results ZIP", result.zip_path, "application/zip"),
        ("PDF report", result.pdf_path, "application/pdf"),
        ("Model weights (.pth)", result.pth_path, "application/octet-stream"),
        ("Standalone controller", result.controller_path, "text/x-python"),
        ("NN.py inference script", result.nn_helper_path, "text/x-python"),
        ("Agent conversation log", result.log_path, "text/plain"),
    ]
    cols = st.columns(3)
    for idx, (label, path, mime) in enumerate(files):
        if not path or not Path(path).is_file():
            continue
        with cols[idx % 3]:
            st.download_button(
                label,
                data=Path(path).read_bytes(),
                file_name=Path(path).name,
                mime=mime,
                width="stretch",
            )

    # --- Report text ------------------------------------------------------
    if result.abstract:
        with st.expander("Engineering manuscript", expanded=False):
            st.markdown(f"**Abstract.** {result.abstract}")
            st.markdown(f"**Conclusion.** {result.conclusion}")

    # --- Figures ----------------------------------------------------------
    figures = [f for f in result.figures if Path(f).is_file()]
    if figures:
        st.markdown("#### Diagnostic figures")
        verification = [f for f in figures if "test_verification" in f]
        others = [f for f in figures if "test_verification" not in f]
        for path in verification:
            st.image(path, caption="Held-out verification: true data vs NN rollout",
                     width="stretch")
        grid = st.columns(2)
        for i, path in enumerate(others):
            with grid[i % 2]:
                st.image(path, caption=Path(path).stem, width="stretch")

    if result.quality_issues:
        with st.expander("Data quality findings"):
            for issue in result.quality_issues:
                st.markdown(f"- {issue}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown("## AgentSysID")
    st.caption(
        "Multi-agent neural system identification — Initializer → Data Inspector → "
        "Critic / Actor / Explorer → verification → report."
    )

    options, ready = build_sidebar()

    state = st.session_state
    state.setdefault("runner", None)
    state.setdefault("log", "")
    state.setdefault("stage", STAGES[0])
    state.setdefault("progress", 0.0)
    state.setdefault("history", [])
    state.setdefault("result", None)
    state.setdefault("critic", [])

    runner: Optional[PipelineRunner] = state["runner"]
    running = bool(runner and runner.running)

    col_start, col_stop, col_clear = st.columns([1, 1, 4])
    if col_start.button("▶  Start run", type="primary", disabled=running or not ready,
                        width="stretch"):
        state.update(log="", stage=STAGES[0], progress=0.0, history=[], result=None, critic=[])
        state["runner"] = PipelineRunner(options)
        state["runner"].start()
        st.rerun()

    if col_stop.button("■  Stop", disabled=not running, width="stretch"):
        request_stop()
        st.toast("Stop requested — finishing the current step and compiling results.")

    if not ready:
        st.info("Upload a dataset, or tick **Use the bundled example dataset**, to begin.")

    # --- Drain the worker's queues ---------------------------------------
    if runner is not None:
        chunks = []
        while True:
            try:
                chunks.append(runner.logs.get_nowait())
            except queue.Empty:
                break
        if chunks:
            state["log"] = (state["log"] + "".join(chunks))[-60000:]

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

        if runner.result is not None and state["result"] is None:
            state["result"] = runner.result

    # --- Live panel -------------------------------------------------------
    if running or state["log"]:
        st.markdown("---")
        render_stages(state["stage"] if running else "")
        st.progress(min(max(state["progress"], 0.0), 1.0))

    history = state["history"]
    if history:
        latest = history[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cycle", f"{latest['cycle']} / {latest['max_cycles']}")
        m2.metric("Validation MSE", f"{latest['val_mse']:.3e}")
        m3.metric("Best MSE", f"{latest['best_mse']:.3e}")
        m4.metric("Topology", str(latest["config"].get("hidden_layers")))

        chart = convergence_chart(history)
        if chart is not None:
            st.markdown("##### Identification error per cycle")
            st.altair_chart(chart, width="stretch")

        lat = latency_chart(
            history,
            float(options.customer_max_latency_ms) if options else cfg.CUSTOMER_MAX_LATENCY_MS,
        )
        if lat is not None:
            st.markdown("##### Inference latency against the customer limit")
            st.altair_chart(lat, width="stretch")

        with st.expander("Cycle table"):
            st.dataframe(
                pd.DataFrame(
                    [
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
                        }
                        for h in history
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    if state["critic"]:
        with st.expander("Critic decisions"):
            for c in state["critic"]:
                st.markdown(
                    f"**Cycle {c['cycle']} — `{c.get('diagnosis')}`** "
                    f"(LR {c.get('lr_dir')} by {c.get('lr_step')}, "
                    f"next {c.get('hidden_layers')})  \n{c.get('reasoning', '')}"
                )

    if state["log"]:
        st.markdown("##### Agent log")
        st.caption(
            "Live console output from the core — the same text the terminal prints, "
            "including every agent decision."
        )
        # A native scroll container keeps the log readable without hand-rolled
        # HTML (which would need escaping and would not scroll reliably).
        with st.container(height=420, border=True):
            st.code(state["log"][-14000:], language="text")

    # --- Final result -----------------------------------------------------
    result: Optional[SysIDResult] = state["result"]
    if result is not None and not running:
        st.markdown("---")
        if result.status == "completed":
            render_results(result)
        else:
            st.error(result.message or "The run did not complete.")
    if runner is not None and runner.error and not running:
        st.error("The pipeline raised an exception.")
        st.code(runner.error)

    # --- Poll while the worker is alive ----------------------------------
    if running:
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
