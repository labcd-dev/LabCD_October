"""
AgentSysID configuration.

All knobs that used to live in the flat ``_legacy/config.py`` are collected
here, value-for-value, with optional environment-variable overrides so the
package can be deployed without editing source.

Runtime mutation is expected and supported: the Initializer Agent and the
interactive questionnaire overwrite module attributes (``config.RESET_THRESHOLD``,
``config.LEARNING_RATE_MIN`` …) exactly like the legacy framework did. Every
consumer reads through the live module object (``from ... import config as cfg``)
so those overrides take effect immediately.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


# ============================================================================
#  CUSTOMER CONTEXT PROMPT
# ============================================================================
# Free-text description of the system, data, or physical properties.
# The agents read this to guide architecture and filtering choices.
# Leave as "" if not used.
CUSTOMER_SYSTEM_DESCRIPTION: str = os.getenv(
    "LABCD_SYSID_CUSTOMER_DESCRIPTION",
    "This dataset represents a high-speed autonomous sports car. "
    "The steering data is very clean, but the lateral velocity transitions "
    "can be highly nonlinear.",
)

# ============================================================================
#  HYBRID MODE: CLIENT PARAMETER OVERRIDES
# ============================================================================
# Lock specific parameters but let the agents decide the rest.
# Leave empty ({}) to give the agents full control.
# Example: {"hidden_layers": [128, 128], "derivative_filter_tau": 0.05}
USER_OVERRIDES: Dict[str, Any] = {}

# Regularization adaptation control:
#   True  = Actor tunes dropout / weight decay dynamically
#   False = locked to the initial values
ADAPTIVE_REGULARIZATION: bool = _env_bool("LABCD_SYSID_ADAPTIVE_REG", True)

# ---------------------------------------------------------------------------
# Architecture & rollout
# ---------------------------------------------------------------------------
NETWORK_ARCHITECTURE: str = os.getenv("LABCD_SYSID_ARCH", "LSTM")  # 'MLP' | 'LSTM'
LSTM_SEQ_LENGTH: int = _env_int("LABCD_SYSID_LSTM_SEQ", 10)  # memory window size
ROLLOUT_HORIZON: int = _env_int("LABCD_SYSID_HORIZON", 1)  # autoregressive steps
TRAJECTORY_CHUNK_SIZE: int = _env_int("LABCD_SYSID_CHUNK", 0)  # 0 = no chunking
SHUFFLE_DATA: bool = _env_bool("LABCD_SYSID_SHUFFLE", True)
INTEGRATOR_TYPE: str = os.getenv("LABCD_SYSID_INTEGRATOR", "RK4")  # 'EULER' | 'RK4'

# ============================================================================
#  DATASET STRUCTURE
# ============================================================================
# True  -> the file contains multiple stacked runs / trajectories
# False -> the file is one single continuous session
MULTI_TRAJECTORY: bool = _env_bool("LABCD_SYSID_MULTI_TRAJ", False)

# Timestamps (seconds) where new trajectories begin. Empty = auto-detect.
# Populated interactively by the CLI questionnaire.
MANUAL_TRAJECTORY_SPLIT_TIMES: List[float] = []

# ---- Derivative estimation method ------------------------------------------
# "finite_difference" | "sliding_mode" | "savitzky_golay"
DERIVATIVE_METHOD: str = os.getenv("LABCD_SYSID_DERIV_METHOD", "finite_difference")

# Savitzky-Golay settings (used only when DERIVATIVE_METHOD = "savitzky_golay")
SAVGOL_WINDOW: int = 15  # must be odd; higher = smoother but clips sharp peaks
SAVGOL_POLYORDER: int = 2  # 2 or 3 is best for physical kinematics

# Levant's sliding-mode differentiator tuning gains
SMD_LAMBDA_1: float = 5.0  # proportional gain (scales with sqrt of error)
SMD_LAMBDA_2: float = 10.0  # integral switching gain

# ---- Derivative estimation filter (Simulink style) --------------------------
# Time constant (tau) of the first-order low-pass filter applied to finite
# differences. Higher = smoother but more lag. 0.0 disables it.
DERIVATIVE_FILTER_TAU: float = _env_float("LABCD_SYSID_DERIV_TAU", 0.005)

# ---- Physics-Informed Neural Network (PINN) toggle --------------------------
USE_PINN: bool = _env_bool("LABCD_SYSID_USE_PINN", False)
PINN_EQUATION_FILE: str = os.getenv("LABCD_SYSID_PINN_FILE", "physics_env.py")
PINN_LOSS_WEIGHT: float = _env_float("LABCD_SYSID_PINN_WEIGHT", 0.5)  # lambda

# ---- Variable time-step configuration --------------------------------------
TIME_COLUMN: str = os.getenv("LABCD_SYSID_TIME_COLUMN", "time")
MIN_DT: float = 1e-6

# ---- Dataset path ----------------------------------------------------------
# Auto-detect whether to use the CSV or the Excel file, as the legacy config did.
_DEFAULT_DATA = "system_data.csv"
if os.getenv("LABCD_SYSID_DATA"):
    EXCEL_FILE_PATH: str = os.environ["LABCD_SYSID_DATA"]
elif os.path.exists("system_data.csv"):
    EXCEL_FILE_PATH = "system_data.csv"
elif os.path.exists("system_data.xlsx"):
    EXCEL_FILE_PATH = "system_data.xlsx"
else:
    EXCEL_FILE_PATH = _DEFAULT_DATA

# Strip the extension so generated module / artifact names never break imports.
ENV_NAME: str = Path(EXCEL_FILE_PATH).stem if EXCEL_FILE_PATH else "system"

# ---------------------------------------------------------------------------
# LLM API provider configuration
# ---------------------------------------------------------------------------
API_PROVIDER: str = os.getenv("LABCD_SYSID_API_PROVIDER", "openai").strip().lower()
LLM_TEMPERATURE: float = _env_float("LABCD_SYSID_LLM_TEMP", 0.5)

if API_PROVIDER == "groq":
    LLM_MODEL: str = os.getenv("LABCD_SYSID_LLM_MODEL", "llama-3.3-70b-versatile")
elif API_PROVIDER == "openai":
    LLM_MODEL = os.getenv("LABCD_SYSID_LLM_MODEL", "gpt-4o-mini")
else:  # openrouter
    LLM_MODEL = os.getenv("LABCD_SYSID_LLM_MODEL", "cohere/north-mini-code:free")

# ============================================================================
#  TUNING FRAMEWORK RUN MODE
# ============================================================================
RUN_MODE: str = os.getenv("LABCD_SYSID_RUN_MODE", "regular")  # fast | regular | heavy
OPTIMIZATION_GOAL: str = "balanced"

# Per-mode loop limits (legacy main.py values).
RUN_MODE_LIMITS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "max_cycles": 7,
        "max_hours": 0.5,
        "critic_explore_limit": 3,
        "failure_memory": 5,
        "memory_capacity": "Last 5 Failures",
        "context_status": "Disabled",
        "reasoning_profile": "2 Concise Sentences",
    },
    "regular": {
        "max_cycles": 20,
        "max_hours": 1.5,
        "critic_explore_limit": 8,
        "failure_memory": 10,
        "memory_capacity": "Last 10 Failures",
        "context_status": "Cycles 1-5 Only",
        "reasoning_profile": "1 Single Paragraph",
    },
    "heavy": {
        "max_cycles": 40,
        "max_hours": 4.0,
        "critic_explore_limit": 15,
        "failure_memory": 0,  # 0 = unlimited
        "memory_capacity": "Unlimited (All Cycles)",
        "context_status": "Always Active",
        "reasoning_profile": "1 Single Paragraph",
    },
}


def run_mode_limits(mode: Optional[str] = None) -> Dict[str, Any]:
    """Return the cycle / time / memory limits for a run mode."""
    key = (mode or RUN_MODE or "regular").strip().lower()
    return dict(RUN_MODE_LIMITS.get(key, RUN_MODE_LIMITS["regular"]))


# ---- Initializer logic override --------------------------------------------
CHOOSE_VIA_LLM_INITIALIZER: bool = _env_bool("LABCD_SYSID_LLM_INIT", True)

# ---- Manual setup parameters (used if CHOOSE_VIA_LLM_INITIALIZER = False) ---
MANUAL_STARTING_LR: float = 0.001
MANUAL_STARTING_HIDDEN_LAYERS: List[int] = [256, 256, 256]
MANUAL_ACTIVATION: str = "elu"  # relu | leaky_relu | elu | tanh

# ---- Regularization & dropout configuration --------------------------------
DROPOUT_RATE: float = 0.2  # active global dropout (agents overwrite this)
WEIGHT_DECAY: float = 1e-3  # active global L2 (agents overwrite this)

MANUAL_DROPOUT_RATE: float = 0.0  # dropout p in [0.0, 0.5] when LLM is disabled
MANUAL_WEIGHT_DECAY: float = 1e-4  # L2 for AdamW when LLM is disabled

DROPOUT_RATE_MIN: float = 0.0
DROPOUT_RATE_MAX: float = 0.5
WEIGHT_DECAY_MIN: float = 0.0
WEIGHT_DECAY_MAX: float = 1e-2

# ---- Adaptive learning-rate scheduler (ReduceLROnPlateau) ------------------
LR_REDUCE_FACTOR: float = 0.5  # e.g. 0.5 halves the LR
LR_SCHEDULE_MIN_FLOOR: float = 1e-6  # absolute LR floor
MANUAL_LR_MIN: float = 0.00005  # LR floor used when manual settings are active

# ---- Localized state-space partition filtering -----------------------------
USE_STATE_FILTER: bool = False
AUTO_FILTER_PERCENTILES: Tuple[int, int] = (2, 98)
# Leave empty ({}) to trim outliers across ALL dimensions with the percentiles.
STATE_BOUNDS_FILTER: Dict[int, Tuple[float, float]] = {}

# ---- Neural network & training limits (per cycle) --------------------------
EPOCHS: int = _env_int("LABCD_SYSID_EPOCHS", 300)
BATCH_SIZE: int = _env_int("LABCD_SYSID_BATCH", 128)
EARLY_STOP_PATIENCE: int = _env_int("LABCD_SYSID_PATIENCE", 25)
HIDDEN_SIZE_MIN: int = 32
HIDDEN_SIZE_MAX: int = 256
NUM_LAYERS_MIN: int = 1
NUM_LAYERS_MAX: int = 3
LEARNING_RATE_MIN: float = 0.00005
LEARNING_RATE_MAX: float = 0.001

# ---- Activation options ----------------------------------------------------
AVAILABLE_ACTIVATIONS: List[str] = ["relu", "leaky_relu", "elu", "tanh", "swish"]

# ---- Hyperparameter memory -------------------------------------------------
LR_TOLERANCE: float = 0.0001
MAX_RETRIES: int = 3
RECENT_FAILURES_MEMORY: int = 5

# ---- Actor-Critic loop -----------------------------------------------------
MSE_TARGET: float = 0.00005

# ---- Plot & output ---------------------------------------------------------
SAVE_PLOT: bool = _env_bool("LABCD_SYSID_SAVE_PLOT", True)
PLOT_FILENAME_PREFIX: str = "system_id"
LOG_FILENAME: str = "agent_prompt_history.log"

# ---- Training & extension configuration ------------------------------------
ADAPTIVE_IMPROVEMENT_THRESHOLD: float = 0.005
EPOCH_EXTENSION_STEPS: int = 50

# ---- Physics data configuration --------------------------------------------
# State indices that are angles (radians) needing [-pi, pi] wrapping.
#   Quadcopter (roll, pitch, yaw): [6, 7, 8]
#   Pure linear system           : []
ANGLE_INDICES: List[int] = []
# When True the loader statistically scans for wrapped angular states.
AUTO_DETECT_ANGLES: bool = False

# State-change threshold within one dt that triggers a trajectory reset.
# Single float (all states) or a list of floats (one per state).
# Values > 90000 mean "auto-calibrate from the data".
RESET_THRESHOLD: Union[float, List[float]] = 99999.0

# ---- Macro-level termination limits ----------------------------------------
OVERFIT_RATIO_LIMIT: float = 10.0  # abort if val loss is 10x train loss

# ---- Customer hard constraints ---------------------------------------------
CUSTOMER_MAX_LATENCY_MS: float = _env_float("LABCD_SYSID_MAX_LATENCY_MS", 2.0)

# ============================================================================
#  END OF CONFIGURATION
# ============================================================================

RUN_TIMESTAMP: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# LLM client factory (lazy, cached)
# ---------------------------------------------------------------------------
_llm_client = None


def get_llm():
    """Return a configured LangChain chat model (OpenAI / Groq / OpenRouter)."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    provider = (API_PROVIDER or "openai").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        _llm_client = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        _llm_client = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        _llm_client = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(
            f"Unsupported API_PROVIDER setting: {API_PROVIDER}. "
            "Use 'groq', 'openai', or 'openrouter'."
        )
    return _llm_client


def reset_llm_cache() -> None:
    """Drop the cached client so provider / model changes take effect."""
    global _llm_client
    _llm_client = None


def __getattr__(name: str):
    """
    Backwards-compatible ``config.model`` attribute.

    The legacy code did ``from config import model`` and called
    ``model.invoke(...)``. Building the client eagerly at import time made the
    package unusable without API keys, so it is constructed on first access.
    """
    if name == "model":
        return get_llm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
