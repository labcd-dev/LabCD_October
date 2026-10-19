"""Suggest an editable system-identification measurement schema for a new chat."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend_core.AgentSysID.agents.prompt_library import render, system_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings
from backend_core.AgentSysID.agents.run_evidence import redact


class SuggestedChannel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=80)
    label: str = Field(default="", max_length=140)
    unit: str = Field(default="", max_length=40)


class SheetProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = Field(default="", max_length=400)
    states: list[SuggestedChannel] = Field(default_factory=list, max_length=5)
    inputs: list[SuggestedChannel] = Field(default_factory=list, max_length=4)
    derivatives: list[dict[str, Any]] = Field(default_factory=list, max_length=5)


def _items(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _slug(value: str, fallback: str) -> str:
    value = re.sub(r"^(?:s_|a_|xdot_)", "", str(value or "").strip(), flags=re.I)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()[:48]
    return value or fallback


def _unique_channels(items, prefix, fallback_prefix):
    channels, used, aliases = [], set(), {}
    for index, item in enumerate(items[:5 if prefix == "s_" else 4], start=1):
        try:
            parsed = SuggestedChannel.model_validate(item)
        except ValidationError:
            continue
        base = _slug(parsed.name, f"{fallback_prefix}_{index}")
        name, suffix = base, 2
        while name in used:
            name, suffix = f"{base}_{suffix}", suffix + 1
        used.add(name)
        column = prefix + name
        label = " ".join((parsed.label or parsed.name).split())[:140]
        channel = {"name": name, "column": column, "label": label,
                   "unit": " ".join(parsed.unit.split())[:40]}
        channels.append(channel)
        for alias in (parsed.name, parsed.label, parsed.name.removeprefix(prefix), name, column):
            if alias:
                aliases[str(alias).casefold()] = channel
    return channels, aliases


def default_schema() -> dict:
    """A clearly generic fallback for when the configured AI cannot suggest signals."""
    states, aliases = _unique_channels([
        {"name": "state_1", "label": "First measured state"},
        {"name": "state_2", "label": "Second measured state"},
    ], "s_", "state")
    inputs, _ = _unique_channels([
        {"name": "input_1", "label": "Applied input or action"},
    ], "a_", "input")
    derivatives = [
        {"name": state["name"], "column": f"xdot_{state['name']}",
         "state_column": state["column"], "label": f"Rate of change of {state['label']}", "unit": ""}
        for state in states
    ]
    return {"summary": "Generic starting point; rename these signals to match what your sensors and controls measure.",
            "states": states, "inputs": inputs, "derivatives": derivatives}


def normalize_proposal(value: dict) -> dict:
    """Validate model output and convert signal names to LabCD's column prefixes."""
    proposal = SheetProposal.model_validate(value)
    states, state_aliases = _unique_channels(
        [item.model_dump() for item in proposal.states], "s_", "state")
    inputs, _ = _unique_channels(
        [item.model_dump() for item in proposal.inputs], "a_", "input")
    if not states or not inputs:
        fallback = default_schema()
        states = states or fallback["states"]
        inputs = inputs or fallback["inputs"]

    derivative_by_state = {}
    for item in _items(proposal.derivatives):
        state_ref = str(item.get("state_name", item.get("state", ""))).casefold()
        state = state_aliases.get(state_ref)
        if state and state["column"] not in derivative_by_state:
            label = " ".join(str(item.get("label", "")).split())[:140]
            unit = " ".join(str(item.get("unit", "")).split())[:40]
            derivative_by_state[state["column"]] = {"name": state["name"],
                "column": f"xdot_{state['name']}", "state_column": state["column"],
                "label": label or f"Rate of change of {state['label']}", "unit": unit}
    derivatives = [derivative_by_state.get(state["column"]) or {
        "name": state["name"], "column": f"xdot_{state['name']}",
        "state_column": state["column"], "label": f"Rate of change of {state['label']}", "unit": ""}
        for state in states]
    summary = " ".join(proposal.summary.split())[:400]
    if not summary:
        summary = "A tentative signal list based on your description; confirm it against your sensors and controls."
    return {"summary": summary, "states": states, "inputs": inputs, "derivatives": derivatives}


def proposal_message(schema: dict) -> str:
    states = ", ".join(item["label"] for item in schema["states"])
    inputs = ", ".join(item["label"] for item in schema["inputs"])
    derivative_names = ", ".join(item["column"] for item in schema["derivatives"])
    return (f"{schema['summary']}\n\n"
            f"A starting sheet could measure **states**: {states}; and **inputs**: {inputs}. "
            f"For training, keep at least one measured state and one applied input/control. "
            f"We want a time rate for every measured state, using `{derivative_names}` in matching `xdot_*` columns. "
            "Enter measured rates when you have them; if the source file does not contain rates, LabCD estimates them from the state measurements. "
            "Please check these suggestions against your actual sensors and controls. Fill the sheet below, "
            "or attach your existing file.")


class ModelSheetJob:
    """Non-blocking call to the installation's configured AI service."""

    purpose = "model_sheet"

    def __init__(self, description: str, *, client=None):
        self.question = str(description).strip()
        self.client = client
        self.answer = None
        self.error = None
        self.error_detail = None
        self.phase = "Thinking about likely measurements"
        self.started_at = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def running(self):
        return self.thread.is_alive()

    def start(self):
        self.started_at = time.monotonic()
        self.thread.start()

    def _run(self):
        try:
            self.phase = "Suggesting states, inputs, and state rates"
            client = self.client or DiagnosticClient(DiagnosticSettings.defaults())
            raw = client.complete(system_prompt("model_sheet"), render(
                "model_sheet", description_json=json.dumps(self.question[:4000], ensure_ascii=False)))
            raw = str(raw).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
            schema = normalize_proposal(json.loads(raw))
            model = getattr(getattr(client, "settings", None), "model", "configured model")
            self.answer = {"schema": schema, "message": proposal_message(schema),
                           "model": model, "fallback": False}
        except Exception as exc:
            schema = default_schema()
            detail = redact(f"{type(exc).__name__}: {exc}")[:700]
            self.error_detail = detail
            self.answer = {
                "schema": schema,
                "message": ("I couldn't get a system-specific suggestion from the configured AI service. "
                            "Here is a generic editable sheet to start from; rename the signals to match your "
                            "measurements. The sheet includes a matching `xdot_*` rate for every state."),
                "model": "generic fallback", "fallback": True,
            }
