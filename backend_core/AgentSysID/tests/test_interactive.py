"""
Tests for the engineer-facing interactive path.

The legacy ``main()`` always asked these questions, so they must be ON BY
DEFAULT and must apply their answers to the live config. Adapters opt out
explicitly with --headless / --no-interactive.
"""

from __future__ import annotations

from typing import List

import pytest

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID import run_cli
from backend_core.AgentSysID.questionnaire import (
    review_initializer_config,
    run_questionnaire,
)


@pytest.fixture(autouse=True)
def _restore_config():
    saved = {
        k: getattr(cfg, k)
        for k in (
            "CUSTOMER_SYSTEM_DESCRIPTION",
            "ANGLE_INDICES",
            "AUTO_DETECT_ANGLES",
            "MULTI_TRAJECTORY",
            "MANUAL_TRAJECTORY_SPLIT_TIMES",
        )
    }
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)


def _scripted(answers: List[str]):
    """A fake input() that replays a list of answers."""
    it = iter(answers)

    def reader(_prompt: str) -> str:
        return next(it)

    return reader


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------
def _parse(argv):
    """Parse argv the way main() does, including the --headless alias."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.headless:
        args.interactive = False
    return args


def test_interactive_is_on_by_default():
    """Legacy always asked; a bare invocation must too."""
    assert _parse([]).interactive is True


def test_headless_flag_disables_interactive():
    assert _parse(["--headless"]).interactive is False


def test_no_interactive_flag_disables_interactive():
    assert _parse(["--no-interactive"]).interactive is False


def test_explicit_interactive_flag_still_works():
    assert _parse(["--interactive"]).interactive is True


def test_run_cli_exposes_both_flags():
    """Guard against the flags being renamed or dropped."""
    import argparse
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        run_cli.main(["--help"])
    help_text = buf.getvalue()
    assert "--headless" in help_text
    assert "--no-interactive" in help_text


# ---------------------------------------------------------------------------
# Questionnaire
# ---------------------------------------------------------------------------
def test_questionnaire_manual_angles_and_single_trajectory():
    result = run_questionnaire(
        _scripted([
            "A high-speed autonomous sports car.",  # description
            "yes",                                   # has angular states
            "2, 4",                                  # indices
            "yes",                                   # single continuous trajectory
        ])
    )
    assert cfg.CUSTOMER_SYSTEM_DESCRIPTION == "A high-speed autonomous sports car."
    assert cfg.ANGLE_INDICES == [2, 4]
    assert cfg.AUTO_DETECT_ANGLES is False
    assert cfg.MULTI_TRAJECTORY is False
    assert result["angle_indices"] == [2, 4]


def test_questionnaire_auto_angles_and_manual_split_times():
    run_questionnaire(
        _scripted([
            "",          # blank description
            "auto",      # auto-scan for angles
            "no",        # not a single trajectory
            "yes",       # knows the split timestamps
            "25.0, 12.5",
        ])
    )
    assert cfg.CUSTOMER_SYSTEM_DESCRIPTION == ""
    assert cfg.AUTO_DETECT_ANGLES is True
    assert cfg.ANGLE_INDICES == []
    assert cfg.MULTI_TRAJECTORY is True
    # Timestamps are sorted regardless of entry order
    assert cfg.MANUAL_TRAJECTORY_SPLIT_TIMES == [12.5, 25.0]


def test_questionnaire_no_angles_and_auto_split_detection():
    run_questionnaire(_scripted(["", "no", "no", "auto"]))
    assert cfg.ANGLE_INDICES == []
    assert cfg.AUTO_DETECT_ANGLES is False
    assert cfg.MULTI_TRAJECTORY is True
    assert cfg.MANUAL_TRAJECTORY_SPLIT_TIMES == []


def test_questionnaire_reprompts_on_invalid_input():
    """A typo must re-ask, not crash or silently accept."""
    run_questionnaire(
        _scripted([
            "",
            "maybe", "banana", "no",  # two invalid answers, then a valid one
            "yes",
        ])
    )
    assert cfg.ANGLE_INDICES == []


def test_questionnaire_reprompts_on_bad_angle_indices():
    run_questionnaire(
        _scripted([
            "",
            "yes",
            "a, b",   # not numbers -> re-ask
            "1, 3",
            "yes",
        ])
    )
    assert cfg.ANGLE_INDICES == [1, 3]


# ---------------------------------------------------------------------------
# Initializer override review
# ---------------------------------------------------------------------------
def _agent_config() -> dict:
    return {
        "learning_rate": 0.001,
        "hidden_layers": [64],
        "activation": "relu",
        "dropout_rate": 0.0,
        "weight_decay": 0.0001,
        "use_state_filter": False,
        "auto_filter_percentiles": [2, 98],
        "derivative_filter_tau": 0.005,
        "lr_search_min": 5e-5,
        "lr_search_max": 1e-3,
        "hidden_size_search_min": 32,
        "hidden_size_search_max": 256,
        "num_layers_search_min": 1,
        "num_layers_search_max": 3,
        "epochs": 300,
        "batch_size": 128,
        "early_stop_patience": 25,
    }


def test_declining_override_leaves_the_agent_config_untouched():
    original = _agent_config()
    result = review_initializer_config(dict(original), _scripted(["no"]))
    assert result == original


def test_every_legacy_overridable_parameter_can_be_changed():
    """All 14 parameters the legacy prompt exposed must still be reachable."""
    result = review_initializer_config(
        _agent_config(),
        _scripted([
            "yes",          # do you want to override
            "0.007",        # learning_rate
            "128, 64, 32",  # hidden_layers
            "elu",          # activation
            "0.25",         # dropout_rate
            "0.002",        # weight_decay
            "true",         # use_state_filter
            "5, 95",        # auto_filter_percentiles (asked because filter is on)
            "0.02",         # derivative_filter_tau
            "0.0001, 0.05", # LR search bounds
            "16, 512",      # hidden size bounds
            "2, 4",         # layer depth bounds
        ]),
    )
    assert result["learning_rate"] == pytest.approx(0.007)
    assert result["hidden_layers"] == [128, 64, 32]
    assert result["activation"] == "elu"
    assert result["dropout_rate"] == pytest.approx(0.25)
    assert result["weight_decay"] == pytest.approx(0.002)
    assert result["use_state_filter"] is True
    assert result["auto_filter_percentiles"] == [5.0, 95.0]
    assert result["derivative_filter_tau"] == pytest.approx(0.02)
    assert (result["lr_search_min"], result["lr_search_max"]) == (0.0001, 0.05)
    assert (result["hidden_size_search_min"], result["hidden_size_search_max"]) == (16, 512)
    assert (result["num_layers_search_min"], result["num_layers_search_max"]) == (2, 4)


def test_blank_answers_keep_the_agent_choice():
    """Pressing Enter must preserve the AI's value, as the legacy prompt promised."""
    original = _agent_config()
    result = review_initializer_config(
        dict(original),
        _scripted(["yes"] + [""] * 11),
    )
    assert result["learning_rate"] == original["learning_rate"]
    assert result["hidden_layers"] == original["hidden_layers"]
    assert result["activation"] == original["activation"]
    assert result["lr_search_min"] == original["lr_search_min"]


def test_percentile_prompt_is_skipped_when_filter_stays_off():
    """The percentile question is conditional on the state filter being enabled."""
    result = review_initializer_config(
        _agent_config(),
        _scripted([
            "yes",
            "", "", "",       # lr, hidden_layers, activation
            "", "",           # dropout, weight_decay
            "false",          # use_state_filter -> percentiles NOT asked
            "0.03",           # derivative_filter_tau
            "", "", "",       # the three bounds
        ]),
    )
    assert result["use_state_filter"] is False
    assert result["derivative_filter_tau"] == pytest.approx(0.03)


def test_invalid_numeric_override_is_ignored_not_fatal():
    original = _agent_config()
    result = review_initializer_config(
        dict(original),
        _scripted(["yes", "not-a-number"] + [""] * 10),
    )
    assert result["learning_rate"] == original["learning_rate"]


def test_unknown_activation_is_rejected():
    original = _agent_config()
    result = review_initializer_config(
        dict(original),
        _scripted(["yes", "", "", "banana"] + [""] * 8),
    )
    assert result["activation"] == original["activation"]
