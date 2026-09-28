"""Event lifecycle, saved replay, and the native expandable feed."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from frontend_streamlit import ui_activity as activity


def test_stage_completion_and_agent_details_are_not_duplicated():
    items = []
    activity.record(items, "stage", {"name": "Loading dataset"})
    activity.record(items, "stage", {"name": "Data Inspector"})
    assert items[0]["state"] == "complete"
    assert items[1]["state"] == "running"
    activity.record(items, "inspector", {"outcome": "reviewed", "response": "Dataset is healthy"})
    activity.record(items, "inspector_data", {"quality_issues": [], "state_dim": 2})
    activity.record(items, "stage", {"name": "Splitting trajectories"})
    assert items[1]["label"] == "Inspector checked the dataset"
    assert items[1]["details"]["response"] == "Dataset is healthy"
    assert items[1]["details"]["state_dim"] == 2
    activity.record(items, "cycle_started", {"cycle": 1, "config": {"hidden_layers": [32]}})
    activity.record(items, "cycle", {"cycle": 1, "val_mse": 0.01, "is_best": True})
    activity.record(items, "latency", {"cycle": 1, "latency_ms": 0.5})
    actor = [r for r in items if r["id"] == "actor:1"]
    assert len(actor) == 1 and actor[0]["state"] == "complete"
    assert actor[0]["details"]["latency_ms"] == 0.5
    assert actor[0]["details"]["config"]["hidden_layers"] == [32]


def test_failure_marks_pending_steps_and_success_finishes_packaging():
    items = []
    activity.record(items, "critic_started", {"cycle": 3})
    activity.record(items, "error", {"message": "Provider timeout"})
    assert all(r["state"] == "error" for r in items)
    assert items[-1]["details"]["message"] == "Provider timeout"
    activity.record(items, "finished", {})
    assert len(items) == 2
    successful = []
    activity.record(successful, "stage", {"name": "Report & packaging"})
    activity.record(successful, "done", {"result": SimpleNamespace(status="completed", cycles_run=3)})
    assert successful[0]["label"] == "Report and downloads prepared"
    assert all(r["state"] == "complete" for r in successful)


def test_manual_initializer_and_unavailable_inspector_are_truthful():
    items = []
    activity.record(items, "initializer", {"mode": "manual", "config": {"hidden_layers": [16]}})
    activity.record(items, "inspector", {"outcome": "unavailable"})
    assert items[0]["label"] == "Manual starting model settings applied"
    assert "unavailable" in items[1]["label"]


def test_worker_drain_keeps_explorer_feedback_and_saves_replay(tmp_path):
    from frontend_streamlit.agent_sysid_app import PipelineRunner, drain
    from backend_core.AgentSysID.pipeline import SysIDOptions, SysIDResult

    runner = PipelineRunner(SysIDOptions(data_path="unused.csv"))
    runner.result = SysIDResult(status="completed", run_dir=tmp_path)
    runner._on_event("explorer_started", {"cycle": 3})
    runner._on_event("explorer", {"cycle": 3, "hidden_layers": [64, 32], "reasoning": "Escape stagnation"})
    runner._on_event("done", {"result": runner.result})
    state = dict(log="", history=[], critic=[], result=None)
    assert drain(runner, state)
    assert state["activity_saved"] and not state["activity_save_error"]
    replay = activity.load(tmp_path)
    assert replay[0]["label"] == "Explorer proposed a new architecture after cycle 3"
    assert replay[0]["details"]["reasoning"] == "Escape stagnation"
    assert replay[0]["timestamp"]
    assert replay[-1]["details"]["best_mse"] is None  # NaN is valid JSON null
    assert not drain(runner, state)
    assert len(state["activity"]) == 2
    assert not list(tmp_path.glob(".activity-*.tmp"))
    (tmp_path / "agent_activity.json").write_text("{broken", encoding="utf-8")
    assert activity.load(tmp_path) == []


def test_worker_exception_saves_console_failure_and_activity(tmp_path, monkeypatch):
    from frontend_streamlit import agent_sysid_app as app
    from backend_core.AgentSysID.pipeline import SysIDOptions
    from backend_core.AgentSysID.agents.run_evidence import read_json
    def fail(options, on_event, **kwargs):
        on_event("run_started", {"run_dir":str(tmp_path)})
        on_event("stage", {"name":"Loading dataset"})
        print("Loader diagnostics before failure")
        raise ValueError("No state columns found")
    monkeypatch.setattr(app, "run_pipeline", fail)
    runner = app.PipelineRunner(SysIDOptions(data_path="unused.csv"))
    runner._run()
    assert "No state columns found" in runner.error
    assert "Loader diagnostics" in (tmp_path / "runtime_console.log").read_text(encoding="utf-8")
    saved = read_json(tmp_path / "diagnostic_state.json")
    assert saved["status"] == "failed" and saved["last_stage"] == "Loading dataset"
    state = dict(log="", history=[], critic=[], result=None)
    app.drain(runner, state)
    assert state["activity_saved"] and activity.load(tmp_path)


def test_monitor_renders_live_details_and_saved_history(tmp_path):
    items = []
    activity.record(items, "inspector", {"outcome": "reviewed", "response": "Healthy dataset"})
    activity.record(items, "critic", {"cycle": 1, "diagnosis": "UNDERFIT", "reasoning": "Increase capacity", "hidden_layers": [64]})
    assert activity.save(tmp_path, items)
    script = '''
import streamlit as st
from frontend_streamlit.agent_sysid_app import render_monitor
render_monitor(st.session_state["test_state"], None, False)
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["test_state"] = dict(log="", stage="Tuning cycles", progress=0.5,
                                           history=[], result=None, activity=items)
    app.run()
    assert not app.exception
    # Native step expanders are exposed as Status elements by AppTest.
    assert [e.label for e in app.status] == ["Inspector checked the dataset", "Critic reviewed cycle 1 and proposed adjustments"]
    assert [t.value for t in app.text] == ["Healthy dataset", "Increase capacity"]
    app.session_state["test_state"] = {"viewing": {"run_dir": str(tmp_path), "env_name": "Test run"}}
    app.run()
    assert not app.exception
    assert len(app.status) == 2
    assert any("Saved with this run" in c.value for c in app.caption)


@pytest.mark.parametrize("reply,expected", [("", "unavailable"), ("[ASK_HUMAN] Explain s_0", "clarification_skipped"), ("[PROCEED] Healthy data", "reviewed")])
def test_inspector_emits_actual_outcome(monkeypatch, reply, expected):
    import pandas as pd
    from backend_core.AgentSysID.agents import data_inspector
    monkeypatch.setattr(data_inspector, "invoke_llm", lambda *a, **kw: reply)
    loader = SimpleNamespace(df=pd.DataFrame({"s_0": [0.0, 1.0]}), quality_issues=[], complexity_label="Simple")
    outcomes = []
    data_inspector.run_data_inspector_agent(loader, interactive=False, on_review=lambda outcome, response: outcomes.append((outcome, response)))
    assert outcomes[0][0] == expected
