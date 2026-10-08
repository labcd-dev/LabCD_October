import json
from types import SimpleNamespace

import pytest

from frontend_streamlit import conversation_core as core, run_setup_agent as setup_agent
from frontend_streamlit import ui_conversation as ui
from backend_core.AgentSysID import config as cfg


class FakeClient:
    settings = SimpleNamespace(model="configured-test-model")

    def __init__(self, answer):
        self.answer = answer
        self.payload = None

    def complete(self, system, user):
        self.payload = json.loads(user)
        return json.dumps(self.answer)


def sample_chat(rows=180):
    chat = core.new_chat()
    chat["dataset"] = {
        "name": "measurements.csv", "rows": rows, "ready": True,
        "columns": ["time", "s_position", "a_force"],
        "states": ["s_position"], "actions": ["a_force"], "derivatives": [],
        "sample_period": 0.01, "warnings": ["Small sample"],
        "path": "private/local/path.csv", "sha256": "dataset-hash",
        "analysis": {"observed_duration": 1.79, "median_dt": 0.01,
                     "time_reset_count": 0, "states": [{"name": "s_position", "std": 2.1}],
                     "inputs": [{"name": "a_force", "constant": False}]},
    }
    return chat


def test_recommendation_uses_profile_and_clamps_cycles_to_effort_limit():
    client = FakeClient({
        "architecture": "MLP", "architecture_reason": "A compact first comparison fits this sample size.",
        "history_steps": 8, "search_effort": "fast", "cycles": 3,
        "effort_reason": "Use the standard fast first-search allowance.", "confidence": "moderate",
    })

    recommendation = setup_agent.recommend(sample_chat(), client=client)

    assert recommendation["architecture"] == "MLP"
    assert recommendation["cycles"] == cfg.run_mode_limits("fast")["max_cycles"]
    assert client.payload["dataset"]["rows"] == 180
    assert "path" not in client.payload["dataset"]
    assert client.payload["cycle_limits_by_effort"]["heavy"] == cfg.run_mode_limits("heavy")["max_cycles"]
    assert recommendation["model"] == "configured-test-model"
    assert client.payload["current_settings"]["max_cycles"] == 7


def test_fast_setup_recommendation_uses_seven_cycles_even_for_legacy_three_cycle_setting():
    chat = sample_chat(rows=6500)
    chat["settings"]["max_cycles"] = 3  # Existing chats may still carry the former default.
    client = FakeClient({
        "architecture": "LSTM", "architecture_reason": "A sequence model is a testable first choice.",
        "history_steps": 10, "search_effort": "fast", "cycles": 3,
        "effort_reason": "A focused fast pass fits the initial review.", "confidence": "moderate",
    })

    recommendation = setup_agent.recommend(chat, client=client)

    assert recommendation["cycles"] == 7


def test_saved_fast_setup_recommendation_uses_full_default_budget():
    chat = sample_chat(rows=6500)
    ui._save_setup_recommendation(chat, {
        "architecture": "LSTM", "architecture_reason": "Sequence context is worth testing.",
        "history_steps": 10, "search_effort": "fast", "cycles": 3,
        "effort_reason": "A fast first pass.", "confidence": "moderate",
    })

    assert chat["settings"]["run_mode"] == "fast"
    assert chat["settings"]["max_cycles"] == 7
    assert chat["run_setup_flow"]["recommended_cycles"] == 7


def test_recommendation_rejects_unavailable_architecture():
    answer = {
        "architecture": "TRANSFORMER", "architecture_reason": "Use a larger model.",
        "history_steps": 8, "search_effort": "fast", "cycles": 3,
        "effort_reason": "Quick first check.", "confidence": "low",
    }
    with pytest.raises(ValueError):
        setup_agent.recommend(sample_chat(), client=FakeClient(answer))


def test_finishing_angle_assumption_starts_recommendation_automatically(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(core, "REGISTRY", core.JobRegistry())
    chat = sample_chat()
    chat["setup"] = {"stage": "angle", "dataset_sha256": chat["dataset"]["sha256"]}
    started = {}

    class CapturedJob:
        purpose = "run_setup_recommendation"
        running = True

        def __init__(self, snapshot):
            started["chat"] = snapshot

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.run_setup_agent, "RunSetupAgentJob", CapturedJob)

    ui._submit(chat, "No wrapped states", [], hooks={})

    assert chat["setup"]["stage"] == "complete"
    assert started["started"]
    assert core.REGISTRY.planning[chat["id"]].purpose == "run_setup_recommendation"
