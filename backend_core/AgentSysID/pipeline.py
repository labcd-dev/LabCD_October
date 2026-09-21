"""
Headless pipeline API.

This is the single implementation of the AgentSysID run. ``run_cli.py`` is a
thin argv wrapper around it, and the Streamlit / FastAPI adapters call it
directly — so there is exactly one tuning loop in the codebase.

    from backend_core.AgentSysID.pipeline import SysIDOptions, run_pipeline

    result = run_pipeline(
        SysIDOptions(data_path="system_data.csv", run_mode="fast"),
        on_event=lambda kind, payload: print(kind, payload),
    )

``SysIDOptions`` exposes every knob the interactive CLI asks for: the
questionnaire answers, the physics / derivative settings, the architecture and
rollout settings, the PINN toggle, the LLM provider, and the fourteen
Initializer parameters an engineer may override. Any field left as ``None``
keeps whatever ``config.py`` already holds.
"""

from __future__ import annotations

import datetime
import json
import signal
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents import (
    ActorAgent,
    CriticAgent,
    ExplorerAgent,
    InitializerAgent,
    ReportAgent,
    run_data_inspector_agent,
)
from backend_core.AgentSysID.data import ExcelDataLoader, describe_split, split_trajectories
from backend_core.AgentSysID.model.pinn import ensure_pinn_template
from backend_core.AgentSysID.reporting import (
    export_standalone_inference_script,
    generate_all_plots,
    generate_final_pdf,
    package_final_results_to_zip,
    plot_test_dataset_verification,
    save_best_model,
    write_nn_inference_helper,
)
from backend_core.AgentSysID.training import (
    BestConfigTracker,
    calculate_success_score,
    compute_rmse,
    measure_inference_latency,
    train_dynamics_model,
)
from backend_core.AgentSysID.utils import (
    cost_tracker,
    describe_device,
    request_stop,
    reset_stop_flag,
    setup_logging,
    setup_run_dir,
    stop_requested,
)

#: Penalty MSE for a cycle rejected for severe overfitting, so the
#: configuration can never win and the Critic registers a hard failure.
OVERFIT_PENALTY_MSE = 9999.0
OVERFIT_PENALTY_RMSE = 99.0

#: Consecutive failed cycles before the Explorer is summoned.
STAGNATION_LIMIT = 3

#: Ordered pipeline stages, for UI progress bars.
STAGES: Tuple[str, ...] = (
    "Questionnaire",
    "Loading dataset",
    "Data Inspector",
    "Splitting trajectories",
    "Initializer Agent",
    "Tuning cycles",
    "Held-out verification",
    "Report & packaging",
)

EventCallback = Callable[[str, Dict[str, Any]], None]


def _emit(on_event: Optional[EventCallback], kind: str, **payload: Any) -> None:
    """Fire a UI event, never letting a bad listener break the run."""
    if on_event is None:
        return
    try:
        on_event(kind, payload)
    except Exception:  # noqa: BLE001 - a UI bug must not abort training
        pass


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
@dataclass
class SysIDOptions:
    """
    Every engineer-facing decision, as data.

    ``None`` means "leave config.py alone", so a caller only sets what it
    actually wants to change.
    """

    # --- Run basics ---------------------------------------------------
    data_path: str
    run_mode: str = "regular"
    output_dir: str = "artifacts_sysid"
    interactive: bool = False
    max_cycles: Optional[int] = None

    # --- Questionnaire answers ----------------------------------------
    customer_description: Optional[str] = None
    angle_indices: Optional[List[int]] = None
    auto_detect_angles: Optional[bool] = None
    multi_trajectory: Optional[bool] = None
    manual_split_times: Optional[List[float]] = None

    # --- Architecture & rollout ---------------------------------------
    architecture: Optional[str] = None          # MLP | LSTM
    lstm_seq_length: Optional[int] = None
    rollout_horizon: Optional[int] = None
    integrator_type: Optional[str] = None       # EULER | RK4
    trajectory_chunk_size: Optional[int] = None
    shuffle_data: Optional[bool] = None

    # --- Derivative estimation ----------------------------------------
    derivative_method: Optional[str] = None     # finite_difference | sliding_mode | savitzky_golay
    derivative_filter_tau: Optional[float] = None
    savgol_window: Optional[int] = None
    savgol_polyorder: Optional[int] = None
    smd_lambda_1: Optional[float] = None
    smd_lambda_2: Optional[float] = None
    reset_threshold: Optional[Any] = None

    # --- State filtering -----------------------------------------------
    use_state_filter: Optional[bool] = None
    auto_filter_percentiles: Optional[Sequence[float]] = None

    # --- PINN ----------------------------------------------------------
    use_pinn: Optional[bool] = None
    pinn_loss_weight: Optional[float] = None
    pinn_equation_file: Optional[str] = None

    # --- Training limits -----------------------------------------------
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    early_stop_patience: Optional[int] = None
    mse_target: Optional[float] = None
    overfit_ratio_limit: Optional[float] = None
    customer_max_latency_ms: Optional[float] = None
    adaptive_regularization: Optional[bool] = None
    lr_reduce_factor: Optional[float] = None
    lr_schedule_min_floor: Optional[float] = None
    adaptive_improvement_threshold: Optional[float] = None
    epoch_extension_steps: Optional[int] = None

    # --- Search bounds --------------------------------------------------
    learning_rate_min: Optional[float] = None
    learning_rate_max: Optional[float] = None
    hidden_size_min: Optional[int] = None
    hidden_size_max: Optional[int] = None
    num_layers_min: Optional[int] = None
    num_layers_max: Optional[int] = None

    # --- Initializer control --------------------------------------------
    choose_via_llm_initializer: Optional[bool] = None
    manual_starting_lr: Optional[float] = None
    manual_starting_hidden_layers: Optional[List[int]] = None
    manual_activation: Optional[str] = None
    manual_dropout_rate: Optional[float] = None
    manual_weight_decay: Optional[float] = None

    #: Client-locked parameters echoed into the Initializer prompt.
    user_overrides: Dict[str, Any] = field(default_factory=dict)

    #: Applied to the Initializer's proposal, replacing the interactive
    #: review prompt. Keys are the same fourteen the CLI exposes.
    initializer_overrides: Dict[str, Any] = field(default_factory=dict)

    # --- LLM -------------------------------------------------------------
    api_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None

    # --- Output ----------------------------------------------------------
    save_plot: Optional[bool] = None

    #: Config attribute name for each option field, where they differ.
    _CONFIG_MAP: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def apply_to_config(self) -> None:
        """Push every non-None option onto the live config module."""
        mapping = {
            "customer_description": "CUSTOMER_SYSTEM_DESCRIPTION",
            "angle_indices": "ANGLE_INDICES",
            "auto_detect_angles": "AUTO_DETECT_ANGLES",
            "multi_trajectory": "MULTI_TRAJECTORY",
            "manual_split_times": "MANUAL_TRAJECTORY_SPLIT_TIMES",
            "architecture": "NETWORK_ARCHITECTURE",
            "lstm_seq_length": "LSTM_SEQ_LENGTH",
            "rollout_horizon": "ROLLOUT_HORIZON",
            "integrator_type": "INTEGRATOR_TYPE",
            "trajectory_chunk_size": "TRAJECTORY_CHUNK_SIZE",
            "shuffle_data": "SHUFFLE_DATA",
            "derivative_method": "DERIVATIVE_METHOD",
            "derivative_filter_tau": "DERIVATIVE_FILTER_TAU",
            "savgol_window": "SAVGOL_WINDOW",
            "savgol_polyorder": "SAVGOL_POLYORDER",
            "smd_lambda_1": "SMD_LAMBDA_1",
            "smd_lambda_2": "SMD_LAMBDA_2",
            "reset_threshold": "RESET_THRESHOLD",
            "use_state_filter": "USE_STATE_FILTER",
            "auto_filter_percentiles": "AUTO_FILTER_PERCENTILES",
            "use_pinn": "USE_PINN",
            "pinn_loss_weight": "PINN_LOSS_WEIGHT",
            "pinn_equation_file": "PINN_EQUATION_FILE",
            "epochs": "EPOCHS",
            "batch_size": "BATCH_SIZE",
            "early_stop_patience": "EARLY_STOP_PATIENCE",
            "mse_target": "MSE_TARGET",
            "overfit_ratio_limit": "OVERFIT_RATIO_LIMIT",
            "customer_max_latency_ms": "CUSTOMER_MAX_LATENCY_MS",
            "adaptive_regularization": "ADAPTIVE_REGULARIZATION",
            "lr_reduce_factor": "LR_REDUCE_FACTOR",
            "lr_schedule_min_floor": "LR_SCHEDULE_MIN_FLOOR",
            "adaptive_improvement_threshold": "ADAPTIVE_IMPROVEMENT_THRESHOLD",
            "epoch_extension_steps": "EPOCH_EXTENSION_STEPS",
            "learning_rate_min": "LEARNING_RATE_MIN",
            "learning_rate_max": "LEARNING_RATE_MAX",
            "hidden_size_min": "HIDDEN_SIZE_MIN",
            "hidden_size_max": "HIDDEN_SIZE_MAX",
            "num_layers_min": "NUM_LAYERS_MIN",
            "num_layers_max": "NUM_LAYERS_MAX",
            "choose_via_llm_initializer": "CHOOSE_VIA_LLM_INITIALIZER",
            "manual_starting_lr": "MANUAL_STARTING_LR",
            "manual_starting_hidden_layers": "MANUAL_STARTING_HIDDEN_LAYERS",
            "manual_activation": "MANUAL_ACTIVATION",
            "manual_dropout_rate": "MANUAL_DROPOUT_RATE",
            "manual_weight_decay": "MANUAL_WEIGHT_DECAY",
            "api_provider": "API_PROVIDER",
            "llm_model": "LLM_MODEL",
            "llm_temperature": "LLM_TEMPERATURE",
            "save_plot": "SAVE_PLOT",
        }

        for option_name, config_name in mapping.items():
            value = getattr(self, option_name, None)
            if value is None:
                continue
            if option_name == "architecture":
                value = str(value).strip().upper()
            elif option_name == "integrator_type":
                value = str(value).strip().upper()
            elif option_name == "derivative_method":
                value = str(value).strip().lower()
            elif option_name == "manual_activation":
                value = str(value).strip().lower()
            elif option_name == "auto_filter_percentiles":
                value = tuple(value)
            elif option_name == "api_provider":
                value = str(value).strip().lower()
            setattr(cfg, config_name, value)

        if self.user_overrides:
            cfg.USER_OVERRIDES = dict(self.user_overrides)

        # A provider / model / temperature change invalidates the cached client.
        if self.api_provider or self.llm_model or self.llm_temperature is not None:
            cfg.reset_llm_cache()

        cfg.RUN_MODE = self.run_mode
        cfg.EXCEL_FILE_PATH = self.data_path
        cfg.ENV_NAME = Path(self.data_path).stem


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class SysIDResult:
    """Everything a UI needs to render the outcome and offer downloads."""

    status: str = "completed"  # completed | failed | no_model
    run_dir: Optional[Path] = None
    env_name: str = ""
    state_dim: int = 0
    action_dim: int = 0
    complexity_tier: int = 0
    complexity_label: str = ""
    quality_issues: List[str] = field(default_factory=list)

    best_config: Dict[str, Any] = field(default_factory=dict)
    best_mse: float = float("nan")
    best_rmse: float = float("nan")
    latency_ms: float = 0.0
    success_score: float = 0.0
    model_status: str = ""
    activation: str = ""

    cycles_run: int = 0
    performance_history: List[Dict[str, Any]] = field(default_factory=list)

    abstract: str = ""
    conclusion: str = ""

    pdf_path: Optional[str] = None
    zip_path: Optional[str] = None
    pth_path: Optional[str] = None
    controller_path: Optional[str] = None
    nn_helper_path: Optional[str] = None
    log_path: Optional[str] = None
    figures: List[str] = field(default_factory=list)

    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    message: str = ""

    #: What produced this run — echoed into the manifest so a past run can be
    #: reopened, compared, or reproduced.
    dataset: str = ""
    run_mode: str = ""
    architecture: str = ""
    use_pinn: bool = False
    finished_at: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view, for manifests and HTTP responses."""
        from dataclasses import asdict

        data = asdict(self)
        data["run_dir"] = str(self.run_dir) if self.run_dir else None
        return data

    def save_manifest(self) -> Optional[Path]:
        """
        Write ``run_manifest.json`` into the run folder.

        This is what lets a UI list previous runs across restarts. It is
        deliberately written at the run root, which the ZIP packager does not
        collect, so the delivered archive is unchanged.
        """
        if self.run_dir is None:
            return None
        path = Path(self.run_dir) / "run_manifest.json"
        try:
            path.write_text(
                json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
            )
            return path
        except Exception as exc:  # noqa: BLE001 - a manifest must never fail a run
            print(f"    ⚠️ Could not write the run manifest: {exc}")
            return None

    @staticmethod
    def load_manifest(path: str | Path) -> Optional[Dict[str, Any]]:
        """Read one ``run_manifest.json``; returns None when unreadable."""
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Console banners (shared by the CLI and captured by the UI log pane)
# ---------------------------------------------------------------------------
def print_run_mode_banner(mode: str, limits: Dict[str, Any]) -> None:
    explore = limits["critic_explore_limit"]
    max_hours = limits["max_hours"]
    print("=" * 80)
    print(f"🚀 INITIALIZING FRAMEWORK IN [{mode.upper()}] MODE")
    print(f"   ├── Cycle Limit            : {limits['max_cycles']} Cycles")
    print(f"   ├── Time Limit             : {max_hours} Hours ({int(max_hours * 60)} mins)")
    print(
        f"   ├── Critic Search Profile  : Explore (Cycles 1-{explore}) | "
        f"Fine-Tune (Cycles {explore + 1}+)"
    )
    print(f"   ├── Failure Memory Depth   : {limits['memory_capacity']}")
    print(f"   ├── System Context Feed    : {limits['context_status']}")
    print(f"   └── Initializer Reasoning  : {limits['reasoning_profile']}")
    print("=" * 80)


def print_pipeline_banner(data_path: str, arch: str) -> None:
    horizon = int(cfg.ROLLOUT_HORIZON)
    seq_len = int(cfg.LSTM_SEQ_LENGTH)
    int_type = str(cfg.INTEGRATOR_TYPE).strip().upper()

    arch_display = (
        f"LSTM (Temporal Memory Window: {seq_len} steps)"
        if arch == "LSTM"
        else "MLP (Memoryless Instantaneous State)"
    )

    if cfg.USE_PINN:
        architecture_mode = f"PINN ({arch_display} + Physics) | Weight: {cfg.PINN_LOSS_WEIGHT}"
    else:
        architecture_mode = f"Pure {arch_display} (Data-Driven System ID)"

    if horizon > 1:
        strategy_desc = f"Autoregressive Multi-Step Rollout (Horizon = {horizon} steps)"
        target_desc = "Predicting X_dot & Accumulating Trajectory Drift Penalty"
    else:
        strategy_desc = "Single-Step Supervised Loss"
        target_desc = "Predicting Instantaneous State Derivatives (X_dot)"

    integrator_desc = "4th-Order Runge-Kutta (RK4)" if int_type == "RK4" else "1st-Order Euler"

    print("=" * 80)
    print(f"🤖 SYSTEM ID OF DATA SOURCE: '{data_path}'")
    print(f"⏱️  Time-Step Mode: Exact dt calculated row-by-row (from column '{cfg.TIME_COLUMN}')")
    print(f"🏗️  Architecture Mode: {architecture_mode}")
    print(f"⚙️  Kinematic Integrator: {integrator_desc}")
    print(f"🧠 Backend Core Driver: {cfg.API_PROVIDER.upper()} | Model Assigned: {cfg.LLM_MODEL}")
    print(f"📊 Loss Profile: {strategy_desc}")
    print(f"🎯 Target Profile: {target_desc}")
    print("=" * 80)


def query_initializer_with_progress(
    initializer: InitializerAgent, animate: bool = True
) -> Dict[str, Any]:
    """Run the Initializer on a worker thread behind an animated progress bar."""
    if not animate:
        return initializer.determine_initial_setup()

    import threading

    from tqdm import tqdm

    agent_result: List[Optional[Dict[str, Any]]] = [None]
    api_status = [False]

    def fetch_llm() -> None:
        try:
            agent_result[0] = initializer.determine_initial_setup()
        finally:
            api_status[0] = True

    llm_thread = threading.Thread(target=fetch_llm, daemon=True)
    llm_thread.start()

    with tqdm(
        total=100,
        desc="    ⏳ Awaiting AI response",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}",
    ) as pbar:
        current_val = 0.0
        while not api_status[0]:
            step = (99.0 - current_val) * 0.15
            current_val += step
            pbar.update(step)
            time.sleep(0.5)
        pbar.update(100.0 - pbar.n)

    llm_thread.join()
    return agent_result[0] or {}


def apply_initializer_to_config(setup_config: Dict[str, Any], loader: ExcelDataLoader) -> None:
    """
    Push the Initializer's decisions onto the live config module.

    This is what lets the Critic, Actor and Explorer read the *agent-authorised*
    bounds rather than the factory defaults.
    """
    cfg.DROPOUT_RATE = setup_config.get("dropout_rate", cfg.DROPOUT_RATE)
    cfg.WEIGHT_DECAY = setup_config.get("weight_decay", cfg.WEIGHT_DECAY)
    cfg.LR_REDUCE_FACTOR = setup_config.get("lr_reduce_factor", cfg.LR_REDUCE_FACTOR)
    cfg.USE_STATE_FILTER = setup_config.get("use_state_filter", cfg.USE_STATE_FILTER)
    cfg.AUTO_FILTER_PERCENTILES = tuple(
        setup_config.get("auto_filter_percentiles", cfg.AUTO_FILTER_PERCENTILES)
    )
    cfg.DERIVATIVE_FILTER_TAU = setup_config.get(
        "derivative_filter_tau", cfg.DERIVATIVE_FILTER_TAU
    )

    cfg.LEARNING_RATE_MIN = setup_config.get("lr_search_min", cfg.LEARNING_RATE_MIN)
    cfg.LEARNING_RATE_MAX = setup_config.get("lr_search_max", cfg.LEARNING_RATE_MAX)
    cfg.HIDDEN_SIZE_MIN = setup_config.get("hidden_size_search_min", cfg.HIDDEN_SIZE_MIN)
    cfg.HIDDEN_SIZE_MAX = setup_config.get("hidden_size_search_max", cfg.HIDDEN_SIZE_MAX)
    cfg.NUM_LAYERS_MIN = setup_config.get("num_layers_search_min", cfg.NUM_LAYERS_MIN)
    cfg.NUM_LAYERS_MAX = setup_config.get("num_layers_search_max", cfg.NUM_LAYERS_MAX)

    if "reset_threshold" in setup_config:
        cfg.RESET_THRESHOLD = setup_config["reset_threshold"]
        loader.reset_threshold = setup_config["reset_threshold"]


def print_initializer_dashboard(setup_config: Dict[str, Any], chosen_activation: str) -> None:
    import textwrap

    if str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip():
        print("\n    📝 CUSTOMER SYSTEM CONTEXT:")
        print(
            textwrap.fill(
                cfg.CUSTOMER_SYSTEM_DESCRIPTION.strip(),
                width=90,
                initial_indent="        ",
                subsequent_indent="        ",
            )
        )

    raw_reasoning = setup_config.get(
        "reasoning", "Heuristic fallback active. No reasoning generated."
    )
    paragraphs = [p.strip() for p in str(raw_reasoning).split("\n") if p.strip()]
    formatted_reasoning = "\n\n".join(
        textwrap.fill(p, width=90, initial_indent="        ", subsequent_indent="        ")
        for p in paragraphs
    )
    print(f"\n    🎯 INITIALIZER AGENT REASONING:\n{formatted_reasoning}")

    thresh_val = cfg.RESET_THRESHOLD
    if isinstance(thresh_val, (list, tuple)):
        thresh_display = "[" + ", ".join(
            f"{x:.4f}" if isinstance(x, (int, float)) else str(x) for x in thresh_val
        ) + "]"
    elif isinstance(thresh_val, (int, float)):
        thresh_display = f"{thresh_val:.4f}"
    else:
        thresh_display = str(thresh_val)

    reg_mode = (
        "WORKING (Dynamic)" if getattr(cfg, "ADAPTIVE_REGULARIZATION", True) else "CONSTANT (Locked)"
    )

    print("\n    🛠️  AGENT-AUTHORIZED SEARCH BOUNDS & STARTING CONFIG:")
    print(f"       ├── Initial Activation    : {chosen_activation.upper()}")
    print(
        f"       ├── Initial Learning Rate : {setup_config.get('learning_rate', 0.001):.6f}  "
        f"(Search Bounds: [{cfg.LEARNING_RATE_MIN}, {cfg.LEARNING_RATE_MAX}])"
    )
    print(f"       ├── Initial Topology      : {setup_config.get('hidden_layers', [64])}")
    print(f"       ├── Layer Depth Bounds    : [{cfg.NUM_LAYERS_MIN}, {cfg.NUM_LAYERS_MAX}] layers")
    print(f"       ├── Layer Width Bounds    : [{cfg.HIDDEN_SIZE_MIN}, {cfg.HIDDEN_SIZE_MAX}] neurons")
    print(
        f"       ├── Regularization        : Dropout={cfg.DROPOUT_RATE} | "
        f"L2 Weight Decay={cfg.WEIGHT_DECAY}"
    )
    print(f"       ├── Adaptive Reg Status   : [{reg_mode}]")
    print(f"       ├── LR Scheduler Factor   : {cfg.LR_REDUCE_FACTOR}")
    print(
        f"       ├── State Space Filter    : {'ENABLED' if cfg.USE_STATE_FILTER else 'DISABLED'} | "
        f"Percentiles: {list(cfg.AUTO_FILTER_PERCENTILES)}"
    )
    print(
        f"       └── Kinematic Dynamics    : Reset Threshold={thresh_display} | "
        f"Simulink Filter Tau={cfg.DERIVATIVE_FILTER_TAU}"
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    options: SysIDOptions,
    on_event: Optional[EventCallback] = None,
    install_signal_handler: bool = True,
) -> SysIDResult:
    """
    Run the full identification pipeline and return a structured result.

    ``on_event(kind, payload)`` receives:
      ``stage``     {name, index, total}
      ``cycle``     {cycle, max_cycles, config, train_mse, val_mse, rmse, is_best, best_mse}
      ``latency``   {cycle, latency_ms, max_latency_ms}
      ``critic``    {cycle, diagnosis, status, lr_dir, lr_step, hidden_layers, reasoning}
      ``explorer``  {cycle, hidden_layers, reasoning}
      ``progress``  {value}  0.0 -> 1.0
      ``done``      {result}
      ``error``     {message}
    """
    started = time.time()
    result = SysIDResult()

    def stage(name: str) -> None:
        idx = STAGES.index(name) if name in STAGES else 0
        _emit(on_event, "stage", name=name, index=idx, total=len(STAGES))
        _emit(on_event, "progress", value=idx / len(STAGES))

    data_path = options.data_path
    if not Path(data_path).is_file():
        result.status = "failed"
        result.message = f"Dataset not found: {data_path}"
        _emit(on_event, "error", message=result.message)
        return result

    options.apply_to_config()
    result.env_name = cfg.ENV_NAME

    limits = cfg.run_mode_limits(options.run_mode)
    max_cycles = int(options.max_cycles or limits["max_cycles"])
    max_seconds = float(limits["max_hours"]) * 3600.0

    # --- PINN template gate --------------------------------------------
    if cfg.USE_PINN and not ensure_pinn_template(cfg.PINN_EQUATION_FILE):
        result.status = "failed"
        result.message = (
            f"PINN is enabled and a fresh template was written to "
            f"'{cfg.PINN_EQUATION_FILE}'. Define compute_analytical_xdot, then run again."
        )
        _emit(on_event, "error", message=result.message)
        return result

    print_run_mode_banner(options.run_mode, limits)

    reset_stop_flag()
    if install_signal_handler:
        try:
            signal.signal(signal.SIGINT, request_stop)
        except ValueError:
            pass  # not on the main thread (a web worker); the UI owns stopping

    run_dir = setup_run_dir(base_dir=options.output_dir, env_name=cfg.ENV_NAME)
    log_file = setup_logging(run_dir)
    result.run_dir = run_dir
    result.log_path = str(log_file)

    print(describe_device())
    print(f"📂 Dataset: {data_path}")
    print(f"⚙  Run mode: {options.run_mode}  |  Interactive HIL: {options.interactive}")
    print(f"📁 Run folder: {run_dir.resolve()}")
    print(f"📝 Conversation history: {log_file.name}")

    # --- 1. Questionnaire ------------------------------------------------
    stage("Questionnaire")
    if options.interactive:
        from backend_core.AgentSysID.questionnaire import run_questionnaire

        run_questionnaire()
    else:
        print(
            "\nℹ️  Headless mode: the dataset questionnaire is supplied by the caller "
            "(angular states, trajectory structure and customer context)."
        )

    # --- 2. Load dataset & Data Inspector --------------------------------
    stage("Loading dataset")
    loader = ExcelDataLoader(data_path)
    result.state_dim = loader.state_dim
    result.action_dim = loader.action_dim
    result.complexity_tier = loader.complexity_tier
    result.complexity_label = loader.complexity_label
    result.quality_issues = list(loader.quality_issues)

    stage("Data Inspector")
    engineer_notes, cols_to_drop = run_data_inspector_agent(
        loader, interactive=options.interactive, log_filename=str(log_file)
    )
    if cols_to_drop:
        loader.drop_columns(cols_to_drop)
        result.state_dim = loader.state_dim
        result.action_dim = loader.action_dim

    if engineer_notes and engineer_notes.lower() != "skip":
        if str(getattr(cfg, "CUSTOMER_SYSTEM_DESCRIPTION", "")).strip():
            cfg.CUSTOMER_SYSTEM_DESCRIPTION += (
                f" | Engineer Clarification on Data: {engineer_notes}"
            )
        else:
            cfg.CUSTOMER_SYSTEM_DESCRIPTION = f"Engineer Clarification on Data: {engineer_notes}"
        print("   🧠 Added engineer context to system profile.")

    # --- 3. Trajectories & split -----------------------------------------
    stage("Splitting trajectories")
    trajectories = loader.get_trajectories()
    state_dim = loader.state_dim
    action_encoding_dim = loader.action_dim

    arch = str(cfg.NETWORK_ARCHITECTURE).strip().upper()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajectories, architecture=arch)
    describe_split(train_trajs, val_trajs, test_trajs, len(trajectories))

    print_pipeline_banner(data_path, arch)

    # --- 4. Initializer ---------------------------------------------------
    stage("Initializer Agent")
    prev_tau = cfg.DERIVATIVE_FILTER_TAU
    prev_reset = cfg.RESET_THRESHOLD

    if cfg.CHOOSE_VIA_LLM_INITIALIZER:
        print("\n🧠 Querying Initializer Agent to establish starting hyperparameters + activation...")
        initializer = InitializerAgent(loader, log_filename=str(log_file))
        setup_config = query_initializer_with_progress(initializer, animate=options.interactive)

        if options.interactive:
            from backend_core.AgentSysID.questionnaire import review_initializer_config

            setup_config = review_initializer_config(setup_config)
        elif options.initializer_overrides:
            # The UI's equivalent of the interactive review prompt.
            applied = {
                k: v for k, v in options.initializer_overrides.items() if v is not None
            }
            setup_config.update(applied)
            print(f"\n   ✏️  Engineer overrides applied to the Initializer proposal: {applied}")

        chosen_activation = setup_config.get("activation", "relu")
        apply_initializer_to_config(setup_config, loader)
        print_initializer_dashboard(setup_config, chosen_activation)
    else:
        print("\n⚙️ Manual presets active. Bypassing Initializer Agent...")
        chosen_activation = cfg.MANUAL_ACTIVATION.lower()
        setup_config = {
            "learning_rate": cfg.MANUAL_STARTING_LR,
            "hidden_layers": list(cfg.MANUAL_STARTING_HIDDEN_LAYERS),
            "activation": chosen_activation,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        }
        cfg.DROPOUT_RATE = cfg.MANUAL_DROPOUT_RATE
        cfg.WEIGHT_DECAY = cfg.MANUAL_WEIGHT_DECAY

        print("       ├── User Manual Status    : [ACTIVE]")
        print(f"       ├── Preset Activation     : {chosen_activation.upper()}")
        print(f"       ├── Preset Starting LR    : {setup_config['learning_rate']:.6f}")
        print(f"       ├── Preset Topology       : {setup_config['hidden_layers']}")
        print(
            f"       ├── Regularization        : Dropout={cfg.DROPOUT_RATE} | "
            f"L2 Weight Decay={cfg.WEIGHT_DECAY}"
        )
        print(f"       ├── LR Scheduler Factor   : {cfg.LR_REDUCE_FACTOR}")
        print(
            f"       ├── State Space Filter    : {'ENABLED' if cfg.USE_STATE_FILTER else 'DISABLED'} | "
            f"Percentiles: {list(cfg.AUTO_FILTER_PERCENTILES)}"
        )
        print(
            f"       └── Kinematic Dynamics    : Reset Threshold={cfg.RESET_THRESHOLD} | "
            f"Simulink Filter Tau={cfg.DERIVATIVE_FILTER_TAU}"
        )

    result.activation = chosen_activation

    # The Initializer can change how derivatives are reconstructed. When it
    # does, re-extract so the agent's choice actually reaches the data.
    if cfg.DERIVATIVE_FILTER_TAU != prev_tau or cfg.RESET_THRESHOLD != prev_reset:
        print(
            "\n    ♻️  Initializer changed the derivative filter / reset threshold — "
            "re-extracting trajectories..."
        )
        trajectories = loader.get_trajectories()
        train_trajs, val_trajs, test_trajs = split_trajectories(
            trajectories, architecture=arch, verbose=False
        )
        describe_split(train_trajs, val_trajs, test_trajs, len(trajectories))

    starting_config = {
        "learning_rate": setup_config.get("learning_rate", 0.001),
        "hidden_layers": setup_config.get("hidden_layers", [64]),
        "activation": chosen_activation,
        "dropout_rate": setup_config.get("dropout_rate", cfg.DROPOUT_RATE),
        "weight_decay": setup_config.get("weight_decay", cfg.WEIGHT_DECAY),
        "batch_size": setup_config.get("batch_size", cfg.BATCH_SIZE),
        "early_stop_patience": setup_config.get("early_stop_patience", cfg.EARLY_STOP_PATIENCE),
        "lr_search_min": cfg.LEARNING_RATE_MIN,
        "lr_search_max": cfg.LEARNING_RATE_MAX,
        "hidden_size_search_min": cfg.HIDDEN_SIZE_MIN,
        "hidden_size_search_max": cfg.HIDDEN_SIZE_MAX,
        "num_layers_search_min": cfg.NUM_LAYERS_MIN,
        "num_layers_search_max": cfg.NUM_LAYERS_MAX,
    }

    # --- 5. Actor / Critic / Explorer loop --------------------------------
    stage("Tuning cycles")
    tracker = BestConfigTracker(run_mode=options.run_mode)
    actor = ActorAgent(
        chosen_activation, initial_config=starting_config, log_filename=str(log_file)
    )
    critic = CriticAgent(tracker, run_mode=options.run_mode, log_filename=str(log_file))
    explorer = ExplorerAgent(initial_config=starting_config, log_filename=str(log_file))

    perf_history: List[Dict[str, Any]] = []
    best_model = None
    stagnation_count = 0
    tuning_span = 1.0 / len(STAGES)
    tuning_base = STAGES.index("Tuning cycles") / len(STAGES)

    for iteration in range(max_cycles):
        elapsed_seconds = time.time() - started
        if elapsed_seconds > max_seconds:
            print("\n" + "!" * 80)
            print(f"⏳ TIME LIMIT REACHED: {limits['max_hours']} hours elapsed.")
            print("🛑 Gracefully halting tuning loop to compile the best model found so far...")
            print("!" * 80 + "\n")
            break

        cycle_num = iteration + 1
        print("\n" + "─" * 80)
        print(f" 🔄  [CYCLE {cycle_num:02d} / {max_cycles:02d}]  TRAINING INITIALIZED")
        print("─" * 80)

        current_lr = actor.current_config["learning_rate"]
        current_hl = actor.current_config["hidden_layers"]

        print("    🛠️  Active Architecture Configurations:")
        print(f"        ├── Learning Rate (η)   : {current_lr:.6f}")
        print(f"        ├── Dropout Rate (p)    : {actor.current_config.get('dropout_rate', 0.0):.3f}")
        print(f"        ├── Weight Decay (L2)   : {actor.current_config.get('weight_decay', 0.0):.6f}")
        print(f"        ├── Hidden Layers Count : {len(current_hl)} layer(s)")
        print(f"        └── Neurons per Layer   : {current_hl}")

        model_trained, final_train_mse, final_mse, final_rmse, _X, _y = train_dynamics_model(
            train_trajs,
            val_trajs,
            state_dim,
            action_encoding_dim,
            hidden_layers=current_hl,
            learning_rate=current_lr,
            epochs=setup_config.get("epochs", cfg.EPOCHS),
            batch_size=actor.current_config.get("batch_size", cfg.BATCH_SIZE),
            patience=actor.current_config.get("patience", cfg.EARLY_STOP_PATIENCE),
            activation=chosen_activation,
            dropout_rate=actor.current_config.get("dropout_rate", cfg.MANUAL_DROPOUT_RATE),
            weight_decay=actor.current_config.get("weight_decay", cfg.MANUAL_WEIGHT_DECAY),
            lr_min=cfg.LR_SCHEDULE_MIN_FLOOR,
            architecture=arch,
        )

        # --- Generalization gap check -------------------------------------
        if final_train_mse > 1e-8:
            overfit_ratio = final_mse / final_train_mse
            if overfit_ratio > cfg.OVERFIT_RATIO_LIMIT:
                print("\n⚠️ CYCLE REJECTED: Severe Overfitting Detected!")
                print(
                    f"   Validation MSE ({final_mse:.6f}) is {overfit_ratio:.1f}x higher than "
                    f"Training MSE ({final_train_mse:.6f})."
                )
                print("   The network memorized the dataset instead of learning the dynamics.")
                print("   Applying a penalty to this configuration and moving to the next cycle...")
                final_mse = OVERFIT_PENALTY_MSE
                final_rmse = OVERFIT_PENALTY_RMSE

        perf_history.append(
            {
                "iteration": iteration,
                "cycle": cycle_num,
                "config": dict(actor.current_config),
                "train_mse": final_train_mse,
                "val_mse": final_mse,
                "rmse": final_rmse,
                "performance": {"mse": final_mse, "rmse": final_rmse},
            }
        )

        is_new_best = tracker.update(final_mse, actor.current_config, current_rmse=final_rmse)
        if is_new_best or best_model is None:
            best_model = model_trained

        print(f"\n    📋  [CYCLE {cycle_num:02d} COMPLETED PERFORMANCE REPORT]")
        print(f"        ├── Tested Topology     : Layers={current_hl} | LR={current_lr:.6f}")
        print(f"        ├── Resulting Eval MSE  : {final_mse:.6f}  (RMSE: {final_rmse:.6f})")
        if is_new_best:
            print("        ├── ⭐ NEW HISTORICAL BEST TRACKED MODEL OVERTAKEN! ⭐")
        best_rmse_str = f"{tracker.best_rmse:.6f}" if tracker.best_rmse is not None else "N/A"
        print(
            f"        └── Current Global Best (MSE)  : {tracker.best_mse:.6f}  (RMSE: {best_rmse_str})"
        )

        _emit(
            on_event,
            "cycle",
            cycle=cycle_num,
            max_cycles=max_cycles,
            config=dict(actor.current_config),
            train_mse=final_train_mse,
            val_mse=final_mse,
            rmse=final_rmse,
            is_best=is_new_best,
            best_mse=tracker.best_mse,
            best_rmse=tracker.best_rmse,
        )
        _emit(
            on_event, "progress", value=tuning_base + tuning_span * (cycle_num / max(max_cycles, 1))
        )

        if final_mse <= cfg.MSE_TARGET:
            print("\n🎯 Target Performance reached! Terminating optimization run sequence.")
            break

        if stop_requested():
            print("\n🛑 Stop requested — skipping further cycles and compiling results now.")
            break

        print("\n    🧠 Querying Critic Agent regarding current hyperparameter topology...")

        measured_latency_ms = measure_inference_latency(
            model_trained, state_dim=state_dim, action_dim=action_encoding_dim
        )
        print(
            f"    ⏱️ Model Inference Latency: {measured_latency_ms:.3f} ms "
            f"(Customer Limit: {cfg.CUSTOMER_MAX_LATENCY_MS} ms)"
        )
        perf_history[-1]["latency"] = measured_latency_ms
        perf_history[-1]["performance"]["latency"] = measured_latency_ms
        _emit(
            on_event,
            "latency",
            cycle=cycle_num,
            latency_ms=measured_latency_ms,
            max_latency_ms=cfg.CUSTOMER_MAX_LATENCY_MS,
        )

        critic_output = critic.evaluate(
            train_mse=final_train_mse,
            val_mse=final_mse,
            current_config=actor.current_config,
            activation=chosen_activation,
            measured_latency=measured_latency_ms,
            max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
            cycle_number=cycle_num,
            val_rmse=final_rmse,
        )

        tracker.add_reasoning_to_memory(critic_output.get("reasoning", "No reasoning provided."))

        print("    🎯 CRITIC AGENT RESPONSE COMPILATION:")
        print(f"        ├── Network Diagnosis   : [{critic_output.get('diagnosis', 'UNKNOWN')}]")
        print(f"        ├── Assessment Status   : [{critic_output.get('status')}]")
        print(
            f"        ├── Path Segment Move   : {str(critic_output.get('lr_dir', 'stay')).upper()} "
            f"(Delta step adjustment value: {critic_output.get('lr_step')})"
        )
        print(f"        ├── Next Network Shape  : {critic_output.get('hidden_layers')}")
        print(f"        └── Critic Reasoning    : {critic_output.get('reasoning')}")

        _emit(on_event, "critic", cycle=cycle_num, **critic_output)

        stagnation_count = 0 if is_new_best else stagnation_count + 1

        if stagnation_count >= STAGNATION_LIMIT:
            print(
                f"\n    🚨 STAGNATION DETECTED ({stagnation_count} failed cycles). "
                "Suspending Actor Agent..."
            )
            print("    🌌 Summoning Explorer Agent to force a repulsive architectural shift...")

            new_config, explorer_reasoning = explorer.generate_radical_escape(
                tracker=tracker,
                stuck_config=actor.current_config,
                visited_configs=actor.visited_configs,
                cycle_number=cycle_num,
            )

            print(f"        ├── Target Escape Layers : {new_config['hidden_layers']}")
            print(f"        └── Explorer Reasoning   : {explorer_reasoning}")
            _emit(
                on_event,
                "explorer",
                cycle=cycle_num,
                hidden_layers=new_config["hidden_layers"],
                reasoning=explorer_reasoning,
            )

            actor.current_config = new_config
            actor._add_to_visited(new_config)
            stagnation_count = 0
        else:
            actor.apply_critic_feedback(critic_output, tracker, cycle=cycle_num)

    result.cycles_run = len(perf_history)
    result.performance_history = perf_history

    # --- 6. Final artefacts ------------------------------------------------
    if best_model is None:
        print("\n❌ No model was trained; nothing to report.")
        result.status = "no_model"
        result.message = "No model was trained."
        _emit(on_event, "error", message=result.message)
        return result

    best_cfg = tracker.best_config or dict(actor.current_config)
    best_mse = tracker.best_mse if tracker.best_mse < float("inf") else float("nan")
    best_rmse = tracker.best_rmse if tracker.best_rmse is not None else compute_rmse(best_mse)

    run_stamp = run_dir.name.replace("run_", "", 1)
    parts = run_stamp.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        run_stamp = f"{parts[0]}_{parts[1]}"

    pth_path = save_best_model(
        best_model,
        best_cfg["hidden_layers"],
        state_dim,
        action_encoding_dim,
        best_mse,
        best_cfg,
        chosen_activation,
        env_name=cfg.ENV_NAME,
        timestamp=run_stamp,
        output_dir=run_dir,
        rollout_horizon=int(cfg.ROLLOUT_HORIZON),
    )
    controller_path = export_standalone_inference_script(
        model=best_model,
        hidden_layers=list(best_cfg["hidden_layers"]),
        activation=chosen_activation,
        state_dim=state_dim,
        action_dim=action_encoding_dim,
        pth_stem=pth_path.name,
        env_name=cfg.ENV_NAME,
        output_dir=run_dir,
        architecture=arch,
        lstm_seq_length=int(cfg.LSTM_SEQ_LENGTH),
    )
    nn_helper = write_nn_inference_helper(
        env_name=cfg.ENV_NAME, output_dir=run_dir, integrator=str(cfg.INTEGRATOR_TYPE)
    )

    final_latency_ms = measure_inference_latency(best_model, state_dim, action_encoding_dim)

    figures: List[Path] = []
    if cfg.SAVE_PLOT and perf_history:
        print("\n📊 Generating diagnostic figures...")
        figures = generate_all_plots(
            performance_history=perf_history,
            env_name=cfg.ENV_NAME,
            output_dir=run_dir,
            timestamp=run_stamp,
            max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
            prefix=cfg.PLOT_FILENAME_PREFIX,
        )

    stage("Held-out verification")
    true_traj_data, nn_traj_data = plot_test_dataset_verification(
        best_model,
        test_trajs,
        state_dim,
        action_encoding_dim,
        output_dir=run_dir,
        timestamp=run_stamp,
        prefix=cfg.PLOT_FILENAME_PREFIX,
        env_name=cfg.ENV_NAME,
        architecture=arch,
        lstm_seq_length=int(cfg.LSTM_SEQ_LENGTH),
        integrator=str(cfg.INTEGRATOR_TYPE),
        save_plot=bool(cfg.SAVE_PLOT),
    )

    final_score, model_status = calculate_success_score(
        val_mse=best_mse,
        true_trajectory=true_traj_data,
        nn_trajectory=nn_traj_data,
        complexity_label=getattr(loader, "complexity_label", "Stage 3"),
    )

    stage("Report & packaging")
    print("\n📝 Querying Report Agent to author the final manuscript text...")
    report_agent = ReportAgent(log_filename=str(log_file))
    abstract_text, conclusion_text = report_agent.generate_report_text(
        env_name=cfg.ENV_NAME,
        state_dim=state_dim,
        action_dim=action_encoding_dim,
        best_config=best_cfg,
        best_mse=best_mse,
        best_rmse=best_rmse,
        latency=final_latency_ms,
        use_pinn=cfg.USE_PINN,
        max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
        success_score=final_score,
        model_status=model_status,
        complexity_label=getattr(loader, "complexity_label", "Unknown"),
        architecture=arch,
    )

    print("📄 Compiling automated engineering manuscript (PDF)...")
    pdf_path = generate_final_pdf(
        env_name=cfg.ENV_NAME,
        state_dim=state_dim,
        action_dim=action_encoding_dim,
        best_config=best_cfg,
        best_mse=best_mse,
        best_rmse=best_rmse,
        latency=final_latency_ms,
        use_pinn=cfg.USE_PINN,
        timestamp=run_stamp,
        abstract_text=abstract_text,
        conclusion_text=conclusion_text,
        success_score=final_score,
        model_status=model_status,
        complexity_label=getattr(loader, "complexity_label", "Unknown"),
        output_dir=run_dir,
        architecture=arch,
        rollout_horizon=int(cfg.ROLLOUT_HORIZON),
        lstm_seq_length=int(cfg.LSTM_SEQ_LENGTH),
        prefix=cfg.PLOT_FILENAME_PREFIX,
    )
    print(f"    ✅ PDF Report successfully generated: {pdf_path}")

    cost_tracker.print_summary(cfg.LLM_MODEL)

    # Move the conversation history into Agents_log/ under the legacy log name.
    agents_log_dir = run_dir / "Agents_log"
    agents_log_dir.mkdir(parents=True, exist_ok=True)
    dest_log = agents_log_dir / cfg.LOG_FILENAME
    if log_file.exists():
        dest_log.write_text(log_file.read_text(encoding="utf-8"), encoding="utf-8")
        log_file.unlink()

    zip_path = package_final_results_to_zip(
        run_dir=run_dir,
        env_name=cfg.ENV_NAME,
        timestamp=run_stamp,
        pdf_filename=pdf_path,
    )

    # --- Fill the result ---------------------------------------------------
    result.status = "completed"
    result.best_config = dict(best_cfg)
    result.best_mse = best_mse
    result.best_rmse = best_rmse
    result.latency_ms = final_latency_ms
    result.success_score = final_score
    result.model_status = model_status
    result.abstract = abstract_text
    result.conclusion = conclusion_text
    result.pdf_path = pdf_path
    result.zip_path = zip_path
    result.pth_path = str(pth_path)
    result.controller_path = str(controller_path)
    result.nn_helper_path = str(nn_helper)
    result.log_path = str(dest_log)
    result.figures = [str(p) for p in figures] + [
        str(p) for p in sorted((run_dir / "figures").glob("*test_verification*.png"))
    ]
    result.llm_calls = cost_tracker.total_calls
    result.llm_cost_usd = cost_tracker.estimated_cost(cfg.LLM_MODEL)
    result.elapsed_seconds = time.time() - started
    result.dataset = data_path
    result.run_mode = options.run_mode
    result.architecture = arch
    result.use_pinn = bool(cfg.USE_PINN)
    result.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    result.save_manifest()

    elapsed_min = result.elapsed_seconds / 60.0
    print("✅ AgentSysID finished.")
    print(f"   Cycles run : {result.cycles_run} | Wall clock: {elapsed_min:.1f} min")
    print(
        f"   Best MSE   : {best_mse:.6f} (RMSE {best_rmse:.6f}) | "
        f"Score: {final_score:.1f}/100 [{model_status}]"
    )
    print(f"   Run folder : {run_dir.resolve()}")
    print(f"   History    : {dest_log}")
    print(f"   PDF        : {pdf_path}")
    print(f"   ZIP        : {zip_path}")

    _emit(on_event, "progress", value=1.0)
    _emit(on_event, "done", result=result)
    return result
