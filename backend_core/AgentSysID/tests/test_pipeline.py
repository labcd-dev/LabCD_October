"""
Tests for the headless pipeline API and its parity with the Streamlit UI.

The UI is only allowed to be thin if every engineer decision is a field on
``SysIDOptions`` and every field is actually reachable from the UI — these
tests enforce both directions.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path
from typing import Set

import pytest

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID import run_cli
from backend_core.AgentSysID.pipeline import (
    STAGES,
    SysIDOptions,
    SysIDResult,
    apply_actor_human_choice,
    apply_initializer_human_choice,
    run_pipeline,
)
from backend_core.AgentSysID.agents.actor import ActorAgent

REPO_ROOT = Path(__file__).resolve().parents[3]
UI_APP = REPO_ROOT / "frontend_streamlit" / "agent_sysid_app.py"
EXAMPLE = REPO_ROOT / "backend_core/AgentSysID/data/examples/synthetic_oscillator.csv"

#: Fields that are plumbing rather than an engineer-facing knob.
_NON_KNOB_FIELDS = {"data_path", "run_mode", "output_dir", "interactive", "human_in_the_loop", "_CONFIG_MAP"}


@pytest.fixture(autouse=True)
def _restore_config():
    saved = {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if name.isupper() and not name.startswith("_")
    }
    yield
    for name, value in saved.items():
        setattr(cfg, name, value)


# ---------------------------------------------------------------------------
# Options -> config
# ---------------------------------------------------------------------------
def test_none_fields_leave_config_untouched():
    before = cfg.EPOCHS, cfg.NETWORK_ARCHITECTURE, cfg.USE_PINN
    SysIDOptions(data_path="x.csv").apply_to_config()
    assert (cfg.EPOCHS, cfg.NETWORK_ARCHITECTURE, cfg.USE_PINN) == before


def test_apply_to_config_maps_the_knobs():
    SysIDOptions(
        data_path="plant.csv",
        run_mode="heavy",
        architecture="mlp",
        optimization_goal="speed",
        lstm_seq_length=7,
        rollout_horizon=4,
        integrator_type="euler",
        derivative_method="SAVITZKY_GOLAY",
        derivative_filter_tau=0.02,
        use_pinn=True,
        pinn_loss_weight=0.75,
        epochs=123,
        batch_size=64,
        customer_max_latency_ms=5.5,
        learning_rate_min=1e-6,
        learning_rate_max=1e-2,
        auto_filter_percentiles=[5, 95],
        angle_indices=[1, 3],
        multi_trajectory=True,
        customer_description="A quadcopter.",
        manual_activation="TANH",
        api_provider="GROQ",
    ).apply_to_config()

    assert cfg.NETWORK_ARCHITECTURE == "MLP"          # upper-cased
    assert cfg.OPTIMIZATION_GOAL == "speed"
    assert cfg.INTEGRATOR_TYPE == "EULER"
    assert cfg.DERIVATIVE_METHOD == "savitzky_golay"  # lower-cased
    assert cfg.MANUAL_ACTIVATION == "tanh"
    assert cfg.API_PROVIDER == "groq"
    assert cfg.LSTM_SEQ_LENGTH == 7
    assert cfg.ROLLOUT_HORIZON == 4
    assert cfg.DERIVATIVE_FILTER_TAU == 0.02
    assert cfg.USE_PINN is True
    assert cfg.PINN_LOSS_WEIGHT == 0.75
    assert cfg.EPOCHS == 123
    assert cfg.BATCH_SIZE == 64
    assert cfg.CUSTOMER_MAX_LATENCY_MS == 5.5
    assert cfg.AUTO_FILTER_PERCENTILES == (5, 95)     # coerced to a tuple
    assert cfg.ANGLE_INDICES == [1, 3]
    assert cfg.MULTI_TRAJECTORY is True
    assert cfg.CUSTOMER_SYSTEM_DESCRIPTION == "A quadcopter."
    assert cfg.RUN_MODE == "heavy"
    assert cfg.ENV_NAME == "plant"


def test_initializer_human_choices_stay_within_authorized_ranges():
    authorized = {
        "learning_rate_min": 0.00001, "learning_rate_max": 0.01,
        "hidden_size_min": 16, "hidden_size_max": 256,
        "num_layers_min": 1, "num_layers_max": 4,
    }
    proposal = {
        "learning_rate": 0.001, "lr_search_min": 0.0001, "lr_search_max": 0.002,
        "hidden_layers": [128, 128], "hidden_size_search_min": 32,
        "hidden_size_search_max": 192, "num_layers_search_min": 1,
        "num_layers_search_max": 3, "dropout_rate": 0.0, "weight_decay": 0.005,
    }

    compact = apply_initializer_human_choice(proposal, "compact", authorized)
    assert len(compact["hidden_layers"]) == 1
    assert compact["hidden_layers"][0] < proposal["hidden_layers"][0]
    assert compact["hidden_size_search_max"] <= proposal["hidden_size_search_max"]
    assert compact["num_layers_search_max"] == 1

    wider = apply_initializer_human_choice(proposal, "widen", authorized)
    assert wider["lr_search_min"] == authorized["learning_rate_min"]
    assert wider["hidden_size_search_max"] == authorized["hidden_size_max"]
    assert wider["num_layers_search_max"] == authorized["num_layers_max"]

    regularized = apply_initializer_human_choice(proposal, "regularize", authorized)
    assert regularized["dropout_rate"] > proposal["dropout_rate"]
    assert regularized["weight_decay"] > proposal["weight_decay"]
    assert regularized["weight_decay"] <= cfg.WEIGHT_DECAY_MAX
    over_limit = apply_initializer_human_choice(
        {**proposal, "weight_decay": 0.5}, "regularize", authorized
    )
    assert over_limit["weight_decay"] == cfg.WEIGHT_DECAY_MAX


def test_cycle_human_choice_guides_next_candidate_within_outer_limits():
    actor = ActorAgent("relu", initial_config={
        "learning_rate": 0.001, "hidden_layers": [128, 128],
        "dropout_rate": 0.0, "weight_decay": 0.005,
        "batch_size": 64, "early_stop_patience": 20,
        "lr_search_min": 0.0001, "lr_search_max": 0.002,
        "hidden_size_search_min": 32, "hidden_size_search_max": 192,
        "num_layers_search_min": 1, "num_layers_search_max": 3,
    })
    authorized = {
        "learning_rate_min": 0.00001, "learning_rate_max": 0.01,
        "hidden_size_min": 16, "hidden_size_max": 256,
        "num_layers_min": 1, "num_layers_max": 4,
    }
    smaller = apply_actor_human_choice(actor, "compact", authorized)
    assert len(smaller["hidden_layers"]) == 1
    assert max(smaller["hidden_layers"]) <= actor.hs_max

    apply_actor_human_choice(actor, "widen", authorized)
    assert actor.hs_max == 256 and actor.num_layers_max == 4
    assert cfg.HIDDEN_SIZE_MAX == 256 and cfg.NUM_LAYERS_MAX == 4
    guided = apply_actor_human_choice(actor, "regularize", authorized)
    assert guided["dropout_rate"] > 0
    assert guided["weight_decay"] == cfg.WEIGHT_DECAY_MAX
    actor.current_config["weight_decay"] = 0.5
    capped = apply_actor_human_choice(actor, "regularize", authorized)
    assert capped["weight_decay"] == cfg.WEIGHT_DECAY_MAX


def test_every_knob_reaches_a_real_config_attribute():
    """A field that maps to nothing would silently do nothing."""
    SysIDOptions(data_path="x.csv").apply_to_config()  # populate the mapping
    import inspect

    source = inspect.getsource(SysIDOptions.apply_to_config)
    mapped = dict(re.findall(r'"([a-z0-9_]+)":\s*"([A-Z0-9_]+)"', source))

    knobs = {f.name for f in fields(SysIDOptions)} - _NON_KNOB_FIELDS
    knobs -= {"max_cycles", "user_overrides", "initializer_overrides"}  # handled separately

    unmapped = sorted(k for k in knobs if k not in mapped)
    assert not unmapped, f"SysIDOptions fields with no config mapping: {unmapped}"

    missing_attr = sorted(v for v in mapped.values() if not hasattr(cfg, v))
    assert not missing_attr, f"Mapped to non-existent config attributes: {missing_attr}"


def test_user_overrides_reach_the_initializer_prompt():
    SysIDOptions(data_path="x.csv", user_overrides={"hidden_layers": [128]}).apply_to_config()
    assert cfg.USER_OVERRIDES == {"hidden_layers": [128]}


# ---------------------------------------------------------------------------
# UI parity
# ---------------------------------------------------------------------------
def test_ui_exposes_every_option_field():
    from frontend_streamlit.conversation_core import RunSettings
    knobs = {f.name for f in fields(SysIDOptions)} - _NON_KNOB_FIELDS
    # Credentials/model and trusted PINN code stay in the current server config;
    # export and override dictionaries are supplied by the conversation controller.
    managed = {"pinn_equation_file", "api_provider", "llm_model", "llm_temperature",
               "save_plot", "user_overrides", "initializer_overrides"}
    assert knobs - managed <= set(RunSettings.model_fields)
    assert set(RunSettings.model_fields) <= {f.name for f in fields(SysIDOptions)}


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_validates_initializer_settings_for_conversation_runs():
    from frontend_streamlit import conversation_core as core
    manual = {"epochs": 8, "batch_size": 32, "manual_starting_hidden_layers": [16],
              "hidden_size_min": 16, "manual_starting_lr": 0.0005,
              "manual_activation": "tanh", "manual_dropout_rate": 0.1,
              "manual_weight_decay": 0.002, "use_state_filter": True,
              "auto_filter_percentiles": [2.0, 98.0], "derivative_filter_tau": 0.01}
    settings = core.apply_changes(core.RunSettings().model_dump(), manual)
    assert settings["learning_rate_min"] < settings["learning_rate_max"]
    assert settings["num_layers_min"] <= len(settings["manual_starting_hidden_layers"]) <= settings["num_layers_max"]
    assert settings["manual_activation"] == "tanh"


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_never_runs_interactively_and_duplicates_no_core_logic():
    source = UI_APP.read_text(encoding="utf-8")
    from frontend_streamlit import conversation_core as core
    assert "interactive=False" in __import__("inspect").getsource(core.make_options), "A web UI must never block on input()"
    for forbidden in ("train_dynamics_model", "BestConfigTracker", "CriticAgent("):
        assert forbidden not in source, f"UI duplicates core logic: {forbidden}"


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_module_parses_and_imports_the_pipeline():
    runtime = REPO_ROOT / "frontend_streamlit" / "ui_pipeline_runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    imported: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "backend_core.AgentSysID.pipeline" in imported


# ---------------------------------------------------------------------------
# CLI delegation
# ---------------------------------------------------------------------------
def test_cli_builds_options_from_argv():
    args = run_cli.build_parser().parse_args(
        ["--data", "d.csv", "--mode", "fast", "--arch", "mlp",
         "--max-cycles", "3", "--epochs", "40", "--output-dir", "out"]
    )
    options = run_cli.build_options(args)
    assert isinstance(options, SysIDOptions)
    assert options.data_path == "d.csv"
    assert options.run_mode == "fast"
    assert options.architecture == "mlp"
    assert options.max_cycles == 3
    assert options.epochs == 40
    assert options.output_dir == "out"
    assert options.interactive is True  # legacy default


def test_missing_dataset_is_reported_not_raised(tmp_path):
    result = run_pipeline(
        SysIDOptions(data_path="/nonexistent/nope.csv", output_dir=str(tmp_path)), install_signal_handler=False
    )
    assert result.status == "failed"
    assert "not found" in result.message
    assert result.run_dir and (result.run_dir / "run_context.json").is_file()
    assert (result.run_dir / "diagnostic_state.json").is_file()
    assert (result.run_dir / "run_manifest.json").is_file()


# ---------------------------------------------------------------------------
# A real (tiny) run through the public API
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not EXAMPLE.is_file(), reason="example dataset missing")
def test_end_to_end_run_emits_events_and_artifacts(tmp_path):
    seen_kinds = []
    stages_seen = []
    training_times = []

    def on_event(kind, payload):
        seen_kinds.append(kind)
        if kind == "stage":
            stages_seen.append(payload["name"])
        elif kind == "cycle":
            training_times.append(payload["training_seconds"])

    result = run_pipeline(
        SysIDOptions(
            data_path=str(EXAMPLE),
            run_mode="fast",
            output_dir=str(tmp_path),
            max_cycles=1,
            epochs=8,
            architecture="MLP",
            multi_trajectory=False,
            angle_indices=[],
            mse_target=0.0,  # never short-circuit, so a full cycle runs
            initializer_overrides={"hidden_layers": [16], "activation": "relu"},
        ),
        on_event=on_event,
        install_signal_handler=False,
    )

    assert isinstance(result, SysIDResult)
    assert result.status == "completed"
    # The engineer's override must survive into the trained model.
    assert result.best_config["hidden_layers"] == [16]
    assert result.activation == "relu"

    # Artefacts exist on disk
    for path in (result.pdf_path, result.zip_path, result.pth_path,
                 result.controller_path, result.nn_helper_path, result.log_path):
        assert path and Path(path).is_file(), f"missing artefact: {path}"

    # Events the UI depends on
    assert "stage" in seen_kinds and "progress" in seen_kinds
    assert "cycle" in seen_kinds and "done" in seen_kinds
    assert stages_seen[0] == STAGES[0]
    assert "Report & packaging" in stages_seen

    assert 0.0 <= result.success_score <= 100.0
    assert result.model_status
    assert result.cycles_run == 1
    assert result.elapsed_seconds > 0
    assert result.training_seconds and result.training_seconds > 0
    assert result.training_seconds <= result.elapsed_seconds
    assert result.training_seconds == pytest.approx(sum(training_times))
    assert result.to_dict()["training_seconds"] == result.training_seconds
    import json
    context = json.loads((result.run_dir / "run_context.json").read_text(encoding="utf-8"))
    verification = json.loads((result.run_dir / "verification_summary.json").read_text(encoding="utf-8"))
    assert context["options"]["epochs"] == 8
    assert len(context["prompts"]) == 6
    assert context["phase"] == "initialized"
    assert context["data_summary"]["state_columns"]
    assert verification["aligned_samples"] > 0
    assert verification["states"]
    assert verification["protocol"]["selected_test_trajectory_index"] == 0
    assert verification["protocol"]["target_chunk_horizon_seconds"] == 10.0
