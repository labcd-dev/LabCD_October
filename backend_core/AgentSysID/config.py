"""
AgentSysID configuration.

All knobs that used to live in the flat config.py are collected here.
Prefer environment variables / .env for secrets and deployment overrides.
A thin compatibility layer re-exports the most-used names so legacy
extraction code can still `from backend_core.AgentSysID.config import X`.
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Customer / system context
# ---------------------------------------------------------------------------
CUSTOMER_SYSTEM_DESCRIPTION: str = os.getenv(
    "LABCD_SYSID_CUSTOMER_DESCRIPTION",
    "This dataset represents a high-speed autonomous sports car. "
    "The steering data is very clean, but the lateral velocity transitions "
    "can be highly nonlinear.",
)

USER_OVERRIDES: Dict[str, Any] = {}  # e.g. {"hidden_layers": [128, 128]}

# ---------------------------------------------------------------------------
# Architecture & rollout
# ---------------------------------------------------------------------------
NETWORK_ARCHITECTURE: str = os.getenv("LABCD_SYSID_ARCH", "LSTM")  # MLP | LSTM
LSTM_SEQ_LENGTH: int = int(os.getenv("LABCD_SYSID_LSTM_SEQ", "10"))
ROLLOUT_HORIZON: int = 1
TRAJECTORY_CHUNK_SIZE: int = 0  # 0 = no chunking
SHUFFLE_DATA: bool = True
INTEGRATOR_TYPE: str = "RK4"  # EULER | RK4

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MULTI_TRAJECTORY: bool = False
DERIVATIVE_METHOD: str = "finite_difference"  # finite_difference | sliding_mode | savitzky_golay
SAVGOL_WINDOW: int = 15
SAVGOL_POLYORDER: int = 2
SMD_LAMBDA_1: float = 5.0
SMD_LAMBDA_2: float = 10.0
DERIVATIVE_FILTER_TAU: float = 0.005

USE_PINN: bool = False
PINN_EQUATION_FILE: str = "physics_env.py"
PINN_LOSS_WEIGHT: float = 0.5

TIME_COLUMN: str = "time"
MIN_DT: float = 1e-6

# Auto-detect data file (relative to CWD or explicit)
_DEFAULT_DATA = "system_data.csv"
if os.path.exists("system_data.csv"):
    EXCEL_FILE_PATH = "system_data.csv"
elif os.path.exists("system_data.xlsx"):
    EXCEL_FILE_PATH = "system_data.xlsx"
else:
    EXCEL_FILE_PATH = os.getenv("LABCD_SYSID_DATA", _DEFAULT_DATA)

ENV_NAME = (
    Path(EXCEL_FILE_PATH).stem
    if EXCEL_FILE_PATH
    else "system"
)

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
API_PROVIDER: str = os.getenv("LABCD_SYSID_API_PROVIDER", "openai").lower()
LLM_TEMPERATURE: float = float(os.getenv("LABCD_SYSID_LLM_TEMP", "0.5"))

if API_PROVIDER == "groq":
    LLM_MODEL = os.getenv("LABCD_SYSID_LLM_MODEL", "llama-3.3-70b-versatile")
elif API_PROVIDER == "openai":
    LLM_MODEL = os.getenv("LABCD_SYSID_LLM_MODEL", "gpt-4o-mini")
else:  # openrouter
    LLM_MODEL = os.getenv("LABCD_SYSID_LLM_MODEL", "cohere/north-mini-code:free")

# ---------------------------------------------------------------------------
# Run mode & search bounds
# ---------------------------------------------------------------------------
RUN_MODE: str = os.getenv("LABCD_SYSID_RUN_MODE", "regular")  # fast | regular | heavy

CHOOSE_VIA_LLM_INITIALIZER: bool = True
MANUAL_STARTING_LR: float = 0.001
MANUAL_STARTING_HIDDEN_LAYERS: List[int] = [256, 256, 256]
MANUAL_ACTIVATION: str = "elu"

ADAPTIVE_REGULARIZATION: bool = True
DROPOUT_RATE: float = 0.2
WEIGHT_DECAY: float = 1e-3
MANUAL_DROPOUT_RATE: float = 0.0
MANUAL_WEIGHT_DECAY: float = 1e-4
DROPOUT_RATE_MIN: float = 0.0
DROPOUT_RATE_MAX: float = 0.5
WEIGHT_DECAY_MIN: float = 0.0
WEIGHT_DECAY_MAX: float = 1e-2

LR_REDUCE_FACTOR: float = 0.5
LR_SCHEDULE_MIN_FLOOR: float = 1e-6
MANUAL_LR_MIN: float = 5e-5

USE_STATE_FILTER: bool = False
AUTO_FILTER_PERCENTILES: Tuple[int, int] = (2, 98)
STATE_BOUNDS_FILTER: Dict = {}

EPOCHS: int = 300
BATCH_SIZE: int = 128
EARLY_STOP_PATIENCE: int = 25
HIDDEN_SIZE_MIN: int = 32
HIDDEN_SIZE_MAX: int = 256
NUM_LAYERS_MIN: int = 1
NUM_LAYERS_MAX: int = 3
LEARNING_RATE_MIN: float = 5e-5
LEARNING_RATE_MAX: float = 1e-3

AVAILABLE_ACTIVATIONS: List[str] = ["relu", "leaky_relu", "elu", "tanh", "swish"]

LR_TOLERANCE: float = 1e-4
MAX_RETRIES: int = 3
RECENT_FAILURES_MEMORY: int = 5
MSE_TARGET: float = 5e-5

SAVE_PLOT: bool = True
PLOT_FILENAME_PREFIX: str = "system_id"
LOG_FILENAME: str = "agent_prompt_history.log"

ADAPTIVE_IMPROVEMENT_THRESHOLD: float = 0.005
EPOCH_EXTENSION_STEPS: int = 50

ANGLE_INDICES: List[int] = []
RESET_THRESHOLD: Union[float, List[float]] = 99999.0
OVERFIT_RATIO_LIMIT: float = 10.0
CUSTOMER_MAX_LATENCY_MS: float = 2.0

RUN_TIMESTAMP: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# LLM client factory (lazy)
# ---------------------------------------------------------------------------
_llm_client = None


def get_llm():
    """Return a configured LangChain chat model (OpenAI / Groq / OpenRouter)."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    if API_PROVIDER == "groq":
        _llm_client = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif API_PROVIDER == "openrouter":
        _llm_client = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    elif API_PROVIDER == "openai":
        _llm_client = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(
            f"Unsupported API_PROVIDER={API_PROVIDER!r}. "
            "Use 'groq', 'openai', or 'openrouter'."
        )
    return _llm_client


# Backwards-compat alias used by extracted agent code
model = property(lambda self: get_llm())  # type: ignore
