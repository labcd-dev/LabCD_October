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
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
UI_APP = REPO_ROOT / "frontend_streamlit" / "agent_sysid_app.py"
EXAMPLE = REPO_ROOT / "backend_core/AgentSysID/data/examples/synthetic_oscillator.csv"

#: Fields that are plumbing rather than an engineer-facing knob.
_NON_KNOB_FIELDS = {"data_path", "run_mode", "output_dir", "interactive", "_CONFIG_MAP"}


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
@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_exposes_every_option_field():
    """'Use all of them in the UI' — enforced, not assumed."""
    source = UI_APP.read_text(encoding="utf-8")
    knobs = {f.name for f in fields(SysIDOptions)} - _NON_KNOB_FIELDS
    missing = sorted(k for k in knobs if k not in source)
    assert not missing, f"SysIDOptions fields absent from the Streamlit UI: {missing}"


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_exposes_all_fourteen_initializer_overrides():
    source = UI_APP.read_text(encoding="utf-8")
    for key in [
        "learning_rate", "hidden_layers", "activation", "dropout_rate", "weight_decay",
        "use_state_filter", "auto_filter_percentiles", "derivative_filter_tau",
        "lr_search_min", "lr_search_max", "hidden_size_search_min",
        "hidden_size_search_max", "num_layers_search_min", "num_layers_search_max",
    ]:
        assert key in source, f"Initializer override missing from the UI: {key}"


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_never_runs_interactively_and_duplicates_no_core_logic():
    source = UI_APP.read_text(encoding="utf-8")
    assert "interactive=False" in source, "A web UI must never block on input()"
    for forbidden in ("train_dynamics_model", "BestConfigTracker", "CriticAgent("):
        assert forbidden not in source, f"UI duplicates core logic: {forbidden}"


@pytest.mark.skipif(not UI_APP.is_file(), reason="Streamlit UI not present")
def test_ui_module_parses_and_imports_the_pipeline():
    tree = ast.parse(UI_APP.read_text(encoding="utf-8"))
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
