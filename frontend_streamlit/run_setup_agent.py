"""Dataset-grounded first-pass architecture and search recommendations."""
from __future__ import annotations

import copy
import json
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.prompt_library import system_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings
from backend_core.AgentSysID.agents.run_evidence import redact

try:
    from . import conversation_analysis as analysis
except ImportError:
    import conversation_analysis as analysis


class SetupRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    architecture: Literal["LSTM", "MLP"]
    architecture_reason: str = Field(min_length=1, max_length=1600)
    history_steps: int = Field(ge=2, le=500)
    search_effort: Literal["fast", "regular", "heavy"]
    cycles: int = Field(ge=1, le=100)
    effort_reason: str = Field(min_length=1, max_length=1200)
    confidence: Literal["low", "moderate", "high"]


def _profile(chat: dict) -> dict:
    dataset = chat.get("dataset") or {}
    physics_cues = _physics_cues(chat)
    analysis = dataset.get("analysis") or {}
    analysis_fields = (
        "observed_duration", "median_dt", "time_reset_count", "detected_wrap_states",
        "angle_named_states", "states", "inputs", "derivatives", "correlated_state_pairs",
    )
    safe_analysis = {key: analysis[key] for key in analysis_fields if key in analysis}
    return {
        "file_name": dataset.get("name"),
        "rows": dataset.get("rows"),
        "columns": dataset.get("columns") or [],
        "states": dataset.get("states") or [],
        "inputs": dataset.get("actions") or [],
        "derivatives": dataset.get("derivatives") or [],
        "sample_period_seconds": dataset.get("sample_period"),
        "quality_warnings": dataset.get("warnings") or [],
        "profile": safe_analysis,
        "physics_cues": physics_cues,
    }


def _physics_cues(chat: dict) -> list[str]:
    """Surface only measurable reasons a known equation could be worth discussing."""
    dataset = chat.get("dataset") or {}
    profile = dataset.get("analysis") or {}
    if (len(dataset.get("states") or []) > 1 and
            "correlated_state_pairs" not in profile and dataset.get("path")):
        try:
            profile = analysis.ensure_profile(dataset)
        except (KeyError, OSError, ValueError):
            pass

    cues = []
    state_names = set(dataset.get("states") or [])
    for pair in profile.get("correlated_state_pairs") or []:
        left, right = pair.get("state_a"), pair.get("state_b")
        if left not in state_names or right not in state_names:
            continue
        correlation = pair.get("abs_correlation")
        suffix = f" (|correlation| {float(correlation):.2f})" if correlation is not None else ""
        cues.append(f"{left} and {right} move very closely together{suffix}; check whether they are distinct states.")
        if len(cues) == 2:
            break

    inputs = profile.get("inputs") or []
    weak_inputs = []
    for item in inputs:
        std = item.get("std")
        if item.get("constant") or (std is not None and float(std) ** 2 < 1e-4):
            weak_inputs.append(item.get("name", "an input"))
    if weak_inputs:
        names = ", ".join(str(name) for name in weak_inputs[:4])
        cues.append(f"{names} show little variation, so this recording may not reveal their effects clearly.")

    rows = dataset.get("rows")
    if isinstance(rows, int) and 10 <= rows < 50:
        cues.append(f"The file contains only {rows} samples, so the measured data provide limited evidence for the dynamics.")
    return cues


def physics_advisory(chat: dict) -> dict:
    """Build a transparent, optional PINN prompt from deterministic data checks."""
    cues = _physics_cues(chat)
    return {"ask": bool(cues), "cues": cues}


def recommend(chat: dict, *, client=None) -> dict:
    """Ask the currently configured API for a strict, data-profile-based setup."""
    limits = {mode: int(cfg.run_mode_limits(mode)["max_cycles"])
              for mode in ("fast", "regular", "heavy")}
    request = {
        "dataset": _profile(chat),
        "current_settings": {
            name: (chat.get("settings") or {}).get(name)
            for name in ("architecture", "run_mode", "max_cycles", "lstm_seq_length")
        },
        "optional_system_and_unit_context": (chat.get("settings") or {}).get("customer_description", ""),
        "cycle_limits_by_effort": limits,
    }
    payload = redact(json.dumps(request, ensure_ascii=False, default=str))
    llm = client or DiagnosticClient(DiagnosticSettings.defaults())
    raw = llm.complete(system_prompt("run_setup_agent"), payload).strip()
    if raw.startswith("```json"):
        raw = raw.removeprefix("```json").removesuffix("```").strip()
    elif raw.startswith("```"):
        raw = raw.removeprefix("```").removesuffix("```").strip()
    recommendation = SetupRecommendation.model_validate_json(raw)
    cap = limits[recommendation.search_effort]
    # Fast is the product's default first search: use its full seven-candidate
    # allowance unless the client changes the cycle slider in the setup card.
    # The separate time cap can still end the search earlier.
    recommendation.cycles = (
        cap if recommendation.search_effort == "fast"
        else min(recommendation.cycles, cap)
    )
    model = getattr(getattr(llm, "settings", None), "model", "configured model")
    return {**recommendation.model_dump(), "physics_advisory": physics_advisory(chat), "model": model}


class RunSetupAgentJob:
    """Background recommendation job, stored in the existing planning registry."""
    purpose = "run_setup_recommendation"

    def __init__(self, chat: dict):
        self.snapshot = copy.deepcopy(chat)
        self.answer = None
        self.error = None
        self.error_detail = None
        self.phase = "Reading data patterns to recommend a starting model"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            self.answer = recommend(self.snapshot)
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:1000]
            self.error = "The Run setup agent couldn't complete its recommendation through the configured API. You can still choose the model and search effort below."
