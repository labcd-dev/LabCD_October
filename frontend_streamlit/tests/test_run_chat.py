"""The chat uses one selected run and makes model calls only after submission."""
import hashlib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from backend_core.AgentSysID.agents.run_diagnostic import append_exchange, load_chat
from frontend_streamlit import ui_run_chat as chat
from frontend_streamlit.tests.test_results_workspace import make_run


def app_for(base):
    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit.ui_run_chat import render_run_chat
render_run_chat(st.session_state["test_output_dir"], runner=st.session_state.get("test_runner"))
''')
    app.session_state["test_output_dir"] = str(base)
    return app


def diagnostic_answer(label="Recorded answer"):
    return {"summary":label, "mode":"ai", "reviewed":True, "model":"test-model", "elapsed_seconds":1,
            "findings":[{"title":"A grounded finding", "explanation":"An observed fact", "kind":"observed", "confidence":"high",
                         "evidence":[{"source_id":"M1", "quote":'"status": "completed"', "title":"run_manifest.json", "provenance":"Saved with this run"}]}],
            "experiments":[], "limitations":["Stability is unverified."], "coverage":{"sources_used":1, "agent_turns":1, "templates":6, "notes":[]}}


def test_initial_render_does_not_call_ai_and_uses_current_api(tmp_path, monkeypatch):
    from backend_core.AgentSysID import config as cfg
    make_run(tmp_path, "1")
    monkeypatch.setattr(cfg, "API_PROVIDER", "openai")
    monkeypatch.setattr(cfg, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(chat, "DiagnosticJob", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unsolicited call")))
    app = app_for(tmp_path).run()
    assert not app.exception
    assert app.session_state["diagnostic_settings"].model == "gpt-4o-mini"
    assert app.session_state["diagnostic_settings"].effort == "high"
    assert len(app.chat_input) == 1
    assert any("Understand what happened" in m.value for m in app.markdown)


def test_existing_session_settings_are_replaced_and_follow_api_changes(tmp_path, monkeypatch):
    from backend_core.AgentSysID import config as cfg
    from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticSettings
    make_run(tmp_path, "1")
    monkeypatch.setattr(cfg, "API_PROVIDER", "openai")
    monkeypatch.setattr(cfg, "LLM_MODEL", "gpt-4o-mini")
    app = app_for(tmp_path)
    app.session_state["diagnostic_settings"] = DiagnosticSettings("openai", "old-diagnostic-model")
    app.run()
    assert not app.exception
    assert app.session_state["diagnostic_settings"].model == "gpt-4o-mini"
    monkeypatch.setattr(cfg, "API_PROVIDER", "groq")
    monkeypatch.setattr(cfg, "LLM_MODEL", "current-groq-model")
    app.run()
    assert not app.exception
    settings = app.session_state["diagnostic_settings"]
    assert settings.provider == "groq" and settings.model == "current-groq-model"


def test_saved_chats_and_evidence_are_isolated_by_selected_run(tmp_path):
    first, second = make_run(tmp_path, "1"), make_run(tmp_path, "2")
    append_exchange(first, "Question for run one", diagnostic_answer("Answer for run one"))
    append_exchange(second, "Question for run two", diagnostic_answer("Answer for run two"))
    app = app_for(tmp_path).run()
    app.selectbox(key="diagnostic_run").set_value(str(first.resolve())).run()
    assert not app.exception
    assert any("Answer for run one" in m.value for m in app.markdown)
    assert not any("Answer for run two" in m.value for m in app.markdown)
    assert app.session_state["preview_run_choice"] == str(first.resolve())
    assert any('"status": "completed"' in t.value for t in app.text)
    app.selectbox(key="diagnostic_run").set_value(str(second.resolve())).run()
    assert any("Answer for run two" in m.value for m in app.markdown)
    assert not any("Answer for run one" in m.value for m in app.markdown)


def test_rejected_ai_answer_is_distinguished_from_unavailable_service(tmp_path):
    run = make_run(tmp_path, "1")
    result = diagnostic_answer("The AI answer could not be verified.")
    result.update(mode="local_checks", reviewed=False,
                  diagnostic_error={"code":"unverified_citations", "message":"Quotations did not match."},
                  warnings=["The AI answer failed the diagnosis checks."])
    append_exchange(run, "Why did it fail?", result)
    app = app_for(tmp_path).run()
    assert not app.exception
    assert any("Local checks · AI answer rejected" in item.value for item in app.caption)
    assert not any("AI unavailable" in item.value for item in app.caption)
    assert any("failed the diagnosis checks" in item.value for item in app.warning)


def test_chat_submission_and_clear_only_affect_selected_run(tmp_path, monkeypatch):
    first, second = make_run(tmp_path, "1"), make_run(tmp_path, "2")
    class Job:
        running, error, answer, saved = False, None, None, False
        def __init__(self, run_dir, question, settings, **kwargs):
            self.run_dir, self.question, self.settings = run_dir, question, settings
        def start(self):
            self.answer = diagnostic_answer("The submitted question was checked")
            self.saved = append_exchange(self.run_dir, self.question, self.answer)
    monkeypatch.setattr(chat, "DiagnosticJob", Job)
    app = app_for(tmp_path).run()
    app.selectbox(key="diagnostic_run").set_value(str(first.resolve())).run()
    app.chat_input[0].set_value("Why does rollout drift?").run()
    assert not app.exception
    assert load_chat(first)[0]["content"] == "Why does rollout drift?"
    assert not load_chat(second)
    assert any("submitted question" in m.value for m in app.markdown)
    short_id = hashlib.sha256(str(first.resolve()).encode()).hexdigest()[:12]
    app.button(key=f"diagnostic_clear_{short_id}").click().run()
    assert not app.exception and not load_chat(first)


def test_active_training_disables_only_current_runs_chat(tmp_path):
    first, second = make_run(tmp_path, "1"), make_run(tmp_path, "2")
    class Runner:
        run_dir, running = second, True
    app = app_for(tmp_path)
    app.session_state["test_runner"] = Runner()
    app.run()
    assert not app.exception and app.chat_input[0].disabled
    app.selectbox(key="diagnostic_run").set_value(str(first.resolve())).run()
    assert not app.exception and not app.chat_input[0].disabled


def test_empty_history_explains_how_to_start(tmp_path):
    app = app_for(tmp_path).run()
    assert not app.exception
    assert "Start a run first" in app.info[0].value
    assert not app.chat_input


def test_background_progress_disables_duplicate_submission_and_shows_finished_answer(tmp_path):
    run = make_run(tmp_path, "1")
    class Job:
        running, saved, error, answer = True, False, None, None
        question, phase = "Why did it fail?", "Checking evidence and alternatives"
    job = Job()
    app = app_for(tmp_path)
    app.session_state["diagnostic_jobs"] = {str(run.resolve()):job}
    app.run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any(status.label == job.phase for status in app.status)
    assert any(b.disabled for b in app.button if b.label == "Clear this conversation")
    job.running, job.saved, job.answer = False, True, diagnostic_answer("The reviewed answer is ready")
    append_exchange(run, job.question, job.answer)
    app.run()
    assert not app.exception and not app.chat_input[0].disabled
    assert any("reviewed answer is ready" in m.value for m in app.markdown)


def test_saved_run_opens_its_conversation_and_existing_diagnoses(tmp_path, monkeypatch):
    from frontend_streamlit import conversation_core as core
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path)
    run = make_run(tmp_path, "1")
    append_exchange(run, "Explain the result", diagnostic_answer("A saved run-specific answer"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30).run()
    next(b for b in app.button if (b.key or "").startswith("legacy_run_")).click().run()
    assert not app.exception
    assert app.session_state["conversation"]["run_dir"] == str(run.resolve())
    assert any("saved run-specific answer" in m.value.replace("\\", "") for m in app.markdown)
    assert len(app.chat_input) == 1
