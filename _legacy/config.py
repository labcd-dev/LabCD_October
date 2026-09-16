import os
import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# ============================================================================
#  CONFIGURATION – CHANGE THESE PARAMETERS AS NEEDED
# ============================================================================

# ---- Customer Context Prompt ------------------------------------------------
# Optional: A free-text description of the system, data, or physical properties.
# The AI agents will read this to guide their architecture and filtering choices.
# Leave as an empty string "" if not used.
CUSTOMER_SYSTEM_DESCRIPTION = "This dataset represents a high-speed autonomous sports car. The steering data is very clean, but the lateral velocity transitions can be highly nonlinear."

# ============================================================================
#  HYBRID MODE: CLIENT PARAMETER OVERRIDES
# ============================================================================
# If the client wants to lock specific parameters but let the AI decide the rest,
# set them here. Leave the dictionary empty {} if you want the AI to control everything.
# Example: {"hidden_layers": [128, 128], "derivative_filter_tau": 0.05}
USER_OVERRIDES = {

}

# Regularization Adaptation Control
ADAPTIVE_REGULARIZATION = True  # True = Actor tunes dropout/weight decay dynamically; False = locked to initial values

# --- Architecture & Rollout Settings ---
NETWORK_ARCHITECTURE = "LSTM" # Choose 'MLP' or 'LSTM'
LSTM_SEQ_LENGTH = 10          # Total size of the memory window
ROLLOUT_HORIZON = 1           # How many steps into the future the network simulates itself
TRAJECTORY_CHUNK_SIZE = 0    # Set to 0 for NO CHUNKING (pure continuous sequence)
SHUFFLE_DATA = True            # Set to True to shuffle chunks, False for chronological chunks
INTEGRATOR_TYPE = "RK4"       # Choose "EULER" (1st-Order) or "RK4" (4th-Order)

# ============================================================================
#  DATASET STRUCTURE
# ============================================================================
# Set to True if the Excel file contains multiple stacked runs/trajectories.
# Set to False if the Excel file is one single, continuous driving session.
MULTI_TRAJECTORY = False

# ---- Derivative Estimation Method -------------------------------------------
# Options: "finite_difference", "sliding_mode", "savitzky_golay"
DERIVATIVE_METHOD = "finite_difference"

# Savitzky-Golay Settings (Used only if DERIVATIVE_METHOD = "savitzky_golay")
SAVGOL_WINDOW = 15     # Must be an odd number. Higher = smoother, but can clip sharp peaks.
SAVGOL_POLYORDER = 2   # Polynomial order. 2 or 3 is best for physical kinematics.

# Levant's Sliding Mode Differentiator Tuning Gains
SMD_LAMBDA_1 = 5.0   # Proportional gain (scales with the square root of error)
SMD_LAMBDA_2 = 10.0  # Integral switching gain

# ---- Physics-Informed Neural Network (PINN) Toggle --------------------------
USE_PINN = False            # Set to True to enable physics-guided training
PINN_EQUATION_FILE = "physics_env.py"
PINN_LOSS_WEIGHT = 0.5      # Balances Data MSE vs. Physics MSE (lambda)

# Automatically detect whether to use the large CSV or the Excel file
if os.path.exists("system_data.csv"):
    EXCEL_FILE_PATH = "system_data.csv"
elif os.path.exists("system_data.xlsx"):
    EXCEL_FILE_PATH = "system_data.xlsx"

# ---- Variable Time-Step Configuration ---------------------------------------
TIME_COLUMN = "time"
MIN_DT = 1e-6

# ---- LLM API Provider Configuration -----------------------------------------
API_PROVIDER = "openai"
LLM_TEMPERATURE = 0.5

if API_PROVIDER == "groq":
    LLM_MODEL = "llama-3.3-70b-versatile"

elif API_PROVIDER == "openai":
    LLM_MODEL = "gpt-4o-mini"

else:  # openrouter
    # LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
    LLM_MODEL = "cohere/north-mini-code:free"

# ============================================================================
#  TUNING FRAMEWORK RUN MODE
# ============================================================================
# Options: "fast", "regular", "heavy"
RUN_MODE = "regular"

# ---- Initializer Logic Override --------------------------------------------
CHOOSE_VIA_LLM_INITIALIZER = True

# ---- Manual Setup Parameters (Used if CHOOSE_VIA_LLM_INITIALIZER = False) ----
MANUAL_STARTING_LR = 0.001
MANUAL_STARTING_HIDDEN_LAYERS = [256, 256, 256]
MANUAL_ACTIVATION = "elu"  # Options: "relu", "leaky_relu", "elu", "tanh"

# ---- Regularization & Dropout Configuration ---------------------------------
DROPOUT_RATE = 0.2            # Active global dropout rate (Agent will overwrite this)
WEIGHT_DECAY = 1e-3           # Active global L2 Regularization (Agent will overwrite this)

MANUAL_DROPOUT_RATE = 0       # Dropout probability p in [0.0, 0.5] (used if LLM is disabled)
MANUAL_WEIGHT_DECAY = 1e-4    # L2 Regularization weight for Adam optimizer (used if LLM is disabled)

DROPOUT_RATE_MIN = 0.0
DROPOUT_RATE_MAX = 0.5
WEIGHT_DECAY_MIN = 0.0
WEIGHT_DECAY_MAX = 1e-2

# ---- Adaptive Learning Rate Scheduler (ReduceLROnPlateau) -----------------
LR_REDUCE_FACTOR = 0.5        # Multiplicative factor (e.g., 0.5 cuts the LR in half)
LR_SCHEDULE_MIN_FLOOR = 1e-6  # Absolute floor the LR is never allowed to go below
MANUAL_LR_MIN = 0.00005       # Minimum LR floor used when manual settings are active

# ---- Environment -----------------------------------------------------------
# ⚠️ CRITICAL FIX: Strip extensions so Python imports don't crash!
ENV_NAME = EXCEL_FILE_PATH.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')

# ---- Localized State Space Partition Filtering -----------------------------
USE_STATE_FILTER = False

AUTO_FILTER_PERCENTILES = (2, 98)

STATE_BOUNDS_FILTER = {
    # Leave empty ({}) to automatically trim outliers across ALL dimensions using AUTO_FILTER_PERCENTILES
}

# ---- Neural network & training limits (per cycle) ---------------------------
EPOCHS = 300
BATCH_SIZE = 128            # Lowered from 256
EARLY_STOP_PATIENCE = 25
HIDDEN_SIZE_MIN = 32
HIDDEN_SIZE_MAX = 256      # Lowered from 512
NUM_LAYERS_MIN = 1
NUM_LAYERS_MAX = 3         # Lowered from 4
LEARNING_RATE_MIN = 0.00005
LEARNING_RATE_MAX = 0.001

# ---- Activation options ----------------------------------------------------
AVAILABLE_ACTIVATIONS = ["relu", "leaky_relu", "elu", "tanh", "swish"]

# ---- Hyperparameter memory ------------------------------------------------
LR_TOLERANCE = 0.0001
MAX_RETRIES = 3
RECENT_FAILURES_MEMORY = 5

# ---- Actor‑Critic loop -----------------------------------------------------
MSE_TARGET = 0.00005

# ---- Plot & output ---------------------------------------------------------
SAVE_PLOT = True
PLOT_FILENAME_PREFIX = "system_id"
LOG_FILENAME = "agent_prompt_history.log"

#  TRAINING & EXTENSION CONFIGURATION
ADAPTIVE_IMPROVEMENT_THRESHOLD = 0.005
EPOCH_EXTENSION_STEPS = 50

# ---- Physics Data Configuration ----
# Define which state indices represent angles (in radians) that need [-pi, pi] wrapping.
# - Quadcopter (Roll, Pitch, Yaw): [6, 7, 8]
# - No angles (pure linear system): []
# - All angles (e.g., 4-joint robotic arm): [0, 1, 2, 3]
ANGLE_INDICES = []

# The threshold of state-change in a single time-step (dt) that triggers a trajectory reset.
# Can be a single float (applied to all states) OR a list of floats (one for each state).
RESET_THRESHOLD = 99999.0

# OR Option B: Turn it off completely to see if you get your MSE of 1 back!
# RESET_THRESHOLD = 99999.0

# ---- Macro-Level Termination Limits ----
OVERFIT_RATIO_LIMIT = 10.0        # If Validation Loss is 10x higher than Training Loss, abort

# --- MANUAL SETTINGS ---
CUSTOMER_MAX_LATENCY_MS = 2.0  # The hard maximum inference time allowed in milliseconds

# ---- Derivative Estimation Filter (Simulink Style) --------------------------
# Time constant (tau) for the first-order low-pass filter applied to finite differences.
# Higher = smoother but more lag. Set to 0.0 to disable (pure finite difference).
DERIVATIVE_FILTER_TAU = 0.005

# ============================================================================
#  END OF CONFIGURATION
# ============================================================================

load_dotenv()

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if API_PROVIDER.lower() == "groq":
    model = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY")
    )
elif API_PROVIDER.lower() == "openrouter":
    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
elif API_PROVIDER.lower() == "openai":
    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    # ⚠️ FIXED: Updated the error message to include 'openai'
    raise ValueError(f"Unsupported API_PROVIDER setting: {API_PROVIDER}. Use 'groq', 'openai', or 'openrouter'.")