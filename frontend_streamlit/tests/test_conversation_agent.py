import json
from types import SimpleNamespace

import pytest

from frontend_streamlit import conversation_agent as agent, conversation_core as core


def test_question_drives_fresh_llm_answer_from_measured_profile(tmp_path):
    chat = core.new_chat()
    data = ("time,s_angle,a_motor\n" + "\n".join(
        f"{i / 10},{i / 100},{i % 3}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("angle.csv", data, chat["id"], upload_dir=tmp_path)

    class Client:
        def __init__(self):
            self.calls = []
        def complete(self, system, user):
            self.calls.append((system, user))
            packet = json.loads(user.split("\nDRAFT ANSWER:")[0])
            profile = json.loads(packet["sources"][0]["content"])["profile"]
            assert profile["median_dt"] == pytest.approx(.1)
            response = "The median interval is 0.1 s [D1]." if "sampling" in packet["question"] else "The angle grows through this recording [D1]."
            return json.dumps({"answer": response, "sources": ["D1"], "uncertainty": ""})

    client = Client()
    sampling = agent.answer_question(chat, "What is the sampling interval?", client=client)
    angle = agent.answer_question(chat, "How does the angle change?", client=client)
    assert sampling["answer"] != angle["answer"]
    assert len(client.calls) == 4  # draft and evidence review for each question
    assert sampling["evidence"][0]["id"] == "D1"


def test_question_receives_soft_client_journey_context_without_forced_cta():
    chat = core.new_chat()

    class Client:
        calls = 0

        def complete(self, system, user):
            self.calls += 1
            packet = json.loads(user.split("\nDRAFT ANSWER:\n", 1)[0])
            assert packet["client_journey"]["data_status"] == "not_attached"
            assert packet["client_journey"]["measurement_paths"] == [
                "paste a table into chat", "attach CSV or Excel"]
            assert packet["client_journey"]["run_attached"] is False
            if self.calls == 1:
                assert "answer the exact question first" in system.lower()
                assert "do not append a call to action" in system.lower()
                assert "status `clarification`" in system.lower()
            else:
                assert "low-pressure tone" in system.lower()
            return json.dumps({"answer": "An LSTM learns from ordered sequences; an MLP maps the supplied features directly.",
                               "sources": [], "uncertainty": ""})

    client = Client()
    result = agent.answer_question(chat, "What is the difference between an LSTM and MLP?", client=client)

    assert "ordered sequences" in result["answer"]
    assert "upload" not in result["answer"].lower()
    assert client.calls == 2


def test_question_gets_open_setup_choice_and_agent_recommendation_context(tmp_path):
    chat = core.new_chat()
    data = ("time,s_angle,a_motor\n" + "\n".join(
        f"{i / 10},{i / 100},{i % 3}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("angle.csv", data, chat["id"], upload_dir=tmp_path)
    chat["setup"] = {"stage": "angle", "dataset_sha256": chat["dataset"]["sha256"]}
    chat["run_setup_flow"] = {
        "dataset_sha256": chat["dataset"]["sha256"], "stage": "effort",
        "recommended_architecture": "LSTM", "architecture_reason": "The measured response may depend on recent history.",
        "recommended_effort": "regular", "effort_reason": "A balanced first search.",
        "recommended_cycles": 5, "confidence": "moderate",
    }

    class Client:
        def complete(self, system, user):
            packet = json.loads(user.split("\nDRAFT ANSWER:\n", 1)[0])
            journey = packet["client_journey"]
            assert journey["data_status"] == "ready"
            assert journey["open_data_decision"] == "angle"
            assert journey["run_setup"]["recommended_architecture"] == "LSTM"
            assert journey["run_setup"]["architecture_reason"].startswith("The measured response")
            return json.dumps({"answer": "Wrapping is useful when an angle is stored modulo ±π [D1]; the choice below remains yours.",
                               "sources": ["D1"], "uncertainty": ""})

    result = agent.answer_question(chat, "Why are you asking about angle wrapping?", client=Client())
    assert "choice below remains yours" in result["answer"]


def test_simple_setup_architecture_and_dataset_facts_work_without_provider():
    chat = core.new_chat()
    chat["dataset"] = {"states": ["s_speed", "s_heading", "s_temp"], "ready": True}
    chat["run_setup_flow"] = {
        "recommended_architecture": "LSTM",
        "architecture_reason": "The measured response may depend on recent history.",
    }

    class MustNotCallModel:
        def complete(self, *_args):
            raise AssertionError("Saved setup facts should not require a provider call")

    architecture = agent.answer_question(
        chat, "Should I start with an LSTM or an MLP for these measurements?", client=MustNotCallModel())
    states = agent.answer_question(
        chat, "Which state names from the uploaded dataset are listed in my starting setup?",
        client=MustNotCallModel())

    assert architecture["answer_scope"] == "setup"
    assert "LSTM" in architecture["answer"] and "recent history" in architecture["answer"]
    assert architecture["sources"] == ["U1"]
    assert states["answer_scope"] == "setup"
    assert all(name in states["answer"] for name in chat["dataset"]["states"])
    assert states["sources"] == ["D1"]


def test_conversation_job_answers_saved_setup_facts_before_calling_planner(monkeypatch):
    chat = core.new_chat()
    chat["dataset"] = {"states": ["s_speed", "s_heading"], "ready": True}
    chat["run_setup_flow"] = {
        "recommended_architecture": "LSTM",
        "architecture_reason": "Recent measurements may carry useful history.",
    }

    class PlannerMustNotRun:
        def __init__(self, *_args):
            raise AssertionError("Saved setup facts should be answered without planning or provider calls")

    monkeypatch.setattr(agent, "QuestionOrchestratorAgent", PlannerMustNotRun)
    job = agent.ConversationJob(chat, "Should I start with an LSTM or an MLP for these measurements?")
    job._run()

    assert job.error is None
    assert job.answer_scope == "setup"
    assert job.answer["sources"] == ["U1"]
    assert "LSTM" in job.answer["answer"]


def test_conversation_job_answers_requested_cycle_from_exact_saved_record(monkeypatch, tmp_path):
    run_dir = tmp_path / "cycle-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "performance_history": [{
            "cycle": 1, "architecture": "LSTM", "hidden_layers": [32, 32],
            "epochs": 40, "validation_mse": 0.0041, "training_time_s": 21.2,
        }],
    }), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)

    class PlannerMustNotRun:
        def __init__(self, *_args):
            raise AssertionError("An exact saved-cycle lookup should not need the planner")

    monkeypatch.setattr(agent, "QuestionOrchestratorAgent", PlannerMustNotRun)
    job = agent.ConversationJob(chat, "What happened in cycle 1?")
    job._run()

    assert job.error is None
    assert job.answer_scope == "run"
    assert job.answer["sources"] == ["C1"]
    assert "0.0041" in job.answer["answer"]


def test_conversation_job_summarizes_saved_run_and_verification_records(monkeypatch, tmp_path):
    run_dir = tmp_path / "summary-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "best_mse": 0.0018,
        "performance_history": [{"cycle": 1}, {"cycle": 2}],
    }), encoding="utf-8")
    (run_dir / "verification_summary.json").write_text(json.dumps({
        "aligned_samples": 100,
        "states": [{"state": "s_speed"}, {"state": "s_heading"}],
    }), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)

    class PlannerMustNotRun:
        def __init__(self, *_args):
            raise AssertionError("A run recap should come directly from saved run evidence")

    monkeypatch.setattr(agent, "QuestionOrchestratorAgent", PlannerMustNotRun)
    job = agent.ConversationJob(chat, "How did this run go?")
    job._run()

    assert job.error is None
    assert job.answer_scope == "run"
    assert job.answer["sources"] == ["M1", "V1"]
    assert "0.0018" in job.answer["answer"]
    assert "100 aligned samples" in job.answer["answer"]


def test_bare_yes_after_two_model_choices_asks_which_one(monkeypatch):
    chat = core.new_chat()
    chat["dataset"] = {"states": ["s_speed"], "ready": True}
    chat["run_setup_flow"] = {"recommended_architecture": "LSTM"}
    chat["messages"] = [
        {"role": "user", "content": "Which starting model should I choose?"},
        {"role": "assistant", "kind": "clarification", "answer_scope": "setup",
         "content": "Do you want to prioritize sequence memory or a simpler direct mapping?"},
        {"role": "user", "content": "yes"},
    ]

    class PlannerMustNotRun:
        def __init__(self, *_args):
            raise AssertionError("An ambiguous acknowledgement should be clarified locally")

    monkeypatch.setattr(agent, "QuestionOrchestratorAgent", PlannerMustNotRun)
    job = agent.ConversationJob(chat, "yes")
    job._run()

    assert job.error is None
    assert job.answer["status"] == "clarification"
    assert job.answer_scope == "setup"
    assert "LSTM" in job.answer["answer"] and "MLP" in job.answer["answer"]


@pytest.mark.parametrize("question", [
    "What is your name?",
    "Who made you?",
    "How you made you?",
    "What model are you using?",
    "What model powers you?",
    "What can you do?",
    "How do you work?",
    "Can you remember our chat?",
])
def test_assistant_identity_questions_are_recognized(question):
    assert agent.is_identity_question(question)


def test_ordinary_engineering_questions_are_not_identity_questions():
    assert not agent.is_identity_question("Which state predicts better?")
    assert not agent.is_identity_question("What is the median sample interval?")


def test_live_panel_greeting_is_not_misread_as_a_training_status_question():
    chat = core.new_chat()
    chat["live_run_context"] = {"cycle": 1, "stage": "Tuning cycles"}

    class MustNotCallModel:
        def complete(self, *_args):
            raise AssertionError("A greeting should not need a run recap from the API")

    result = agent.answer_question(chat, "helo", client=MustNotCallModel())

    assert result["answer"].startswith("Hi! I’m LabCD.")
    assert "cycle score" in result["answer"]
    assert "training phase" not in result["answer"]


def test_live_panel_help_explains_what_client_can_do_without_calling_api():
    chat = core.new_chat()
    chat["live_run_context"] = {"cycle": 1, "stage": "Tuning cycles"}

    class MustNotCallModel:
        def complete(self, *_args):
            raise AssertionError("Interface help should be answered directly")

    result = agent.answer_question(chat, "what can i here do", client=MustNotCallModel())

    assert "Cycle results" in result["answer"]
    assert "per-state errors" in result["answer"]
    assert "Shape the next cycle" in result["answer"]
    assert "controls apply a change or stop the run" in result["answer"]


def test_main_chat_help_is_contextual_and_does_not_require_system_description():
    chat = core.new_chat()

    class MustNotCallModel:
        def complete(self, *_args):
            raise AssertionError("Basic app help should not depend on the API")

    result = agent.answer_question(chat, "What can I do here?", client=MustNotCallModel())

    assert "paste a table" in result["answer"]
    assert "time, state, and input columns" in result["answer"]
    assert "description is optional" in result["answer"]


def test_pinn_usage_question_is_answered_from_saved_run_manifest(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "architecture": "LSTM", "use_pinn": False,
    }), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run)
    chat["settings"]["use_pinn"] = True

    question = "Do you use PINN in this run?"
    assert agent.is_pinn_usage_question(question)
    assert agent.is_pinn_usage_question("Is PINN on?")
    answer = agent.pinn_usage_answer(chat)
    assert answer.startswith("No. This saved LSTM run")
    assert "records physics constraints as off" in answer


def test_pcnn_usage_answer_explains_equation_baseline_and_exact_states(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "architecture": "MLP", "use_pinn": True, "physics_mode": "pcnn",
        "pcnn_hard_constraint_states": ["s_pitch"],
    }), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run)

    assert agent.is_pinn_usage_question("Did this run use PCNN?")
    answer = agent.pinn_usage_answer(chat)
    assert "used PCNN (physics-constrained NN)" in answer
    assert "derivative baseline" in answer
    assert "s_pitch" in answer


def test_identity_answer_uses_yaml_profile_api_model_and_longer_chat_history():
    chat = core.new_chat()
    chat["messages"] = [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": f"Earlier conversation detail {index}"}
        for index in range(16)
    ]

    class Client:
        settings = SimpleNamespace(provider="openrouter", model="project-chat-model")
        calls = []

        def complete(self, system, user):
            self.calls.append((system, user))
            packet = json.loads(user.split("\nDRAFT ANSWER:\n", 1)[0])
            identity = packet["assistant_identity"]
            assert identity["name"] == "LabCD"
            assert identity["product"] == "LabCD AgentSysID"
            assert identity["active_api_provider"] == "openrouter"
            assert identity["active_api_model"] == "project-chat-model"
            assert packet["conversation_history_truncated"] is False
            assert len(packet["previous_messages"]) == 16
            assert packet["previous_messages"][0]["content"] == "Earlier conversation detail 0"
            assert packet["sources"][0]["id"] == "A1"
            assert "does not identify one individual creator" in packet["sources"][0]["content"]
            assert "identity" in system.lower()
            return json.dumps({
                "answer": ("I am LabCD, the assistant in LabCD AgentSysID. The project developers "
                           "maintain the app; its underlying model is served by OpenRouter as "
                           "project-chat-model [A1]."),
                "sources": ["A1"], "uncertainty": "",
            })

    client = Client()
    result = agent.answer_question(chat, "Who are you, and how were you made?", client=client)

    assert "LabCD" in result["answer"]
    assert result["sources"] == ["A1"]
    assert result["evidence"] == [{"id": "A1", "title": "LabCD assistant identity and implementation",
                                   "kind": "identity"}]
    assert len(client.calls) == 2


def test_run_question_supplies_per_state_metrics_and_agent_records(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"status": "completed", "best_mse": 0.01}))
    (run / "verification_summary.json").write_text(json.dumps({"aligned_samples": 25, "states": [
        {"state": "s_a", "rmse": .5, "mae": .4, "rmse_over_test_std": .2, "valid_samples": 25},
        {"state": "s_b", "rmse": .8, "mae": .7, "rmse_over_test_std": .1, "valid_samples": 25}]}))
    logs = run / "Agents_log"
    logs.mkdir()
    (logs / "agents.txt").write_text("TIMESTAMP: [now]\nAGENT: [Inspector]\nThe input signal is constant.\n")
    chat = core.new_chat()
    chat["run_dir"] = str(run)

    class Client:
        calls = 0
        def complete(self, system, user):
            self.calls += 1
            packet = json.loads(user.split("\nDRAFT ANSWER:")[0].split("\nLOCAL CHECK FAILED:")[0])
            sources = {s["id"]: s for s in packet["sources"]}
            assert json.loads(sources["V2"]["content"])["states_sorted_by_normalized_rmse"][0]["state"] == "s_b"
            assert any(s["kind"] == "agent_turn" for s in sources.values())
            response = ("Best: s_a; worst: s_b [V2]." if self.calls < 3 else
                        "Best: s_b; worst: s_a by normalized held-out rollout RMSE [V2].")
            return json.dumps({"answer": response,
                               "sources": ["V2"], "uncertainty": ""})

    client = Client()
    result = agent.answer_question(chat, "Which state predicts better, and why?", client=client)
    assert "s_b" in result["answer"]
    assert result["sources"] == ["V2"]
    assert client.calls == 3


def test_answer_rejects_invented_evidence_id():
    with pytest.raises(ValueError, match="not supplied"):
        agent._parse('{"answer":"Unsupported [X9]", "sources":["X9"]}', {"D1"})


def test_answer_normalizes_explicit_source_object_ids():
    parsed = agent._parse(
        '{"answer":"The Actor follows its prompt [P_actor].", "sources":[{"id":"P_actor"}]}',
        {"P_actor"},
    )

    assert parsed["sources"] == ["P_actor"]


def test_evidence_answer_requires_inline_citations_matching_sources():
    repaired = agent._parse('{"answer":"The interval is 0.1 seconds.","sources":["D1"]}', {"D1"})
    assert repaired["answer"] == "The interval is 0.1 seconds [D1]."
    with pytest.raises(ValueError, match="do not match"):
        agent._parse('{"answer":"The interval is 0.1 seconds [D1].","sources":["V1"]}', {"D1", "V1"})


def test_single_selected_evidence_is_added_to_each_sentence_without_splitting_decimals():
    repaired = agent._parse(
        '{"answer":"The interval is 0.1 seconds. There are 21 samples.","sources":["D1"]}',
        {"D1"},
    )
    assert repaired["answer"] == "The interval is 0.1 seconds [D1]. There are 21 samples [D1]."


def test_missing_multi_source_citations_get_one_explicit_retry():
    from frontend_streamlit.question_orchestrator import QuestionPlan

    chat = core.new_chat()

    class Client:
        calls = []

        def complete(self, _system, user):
            self.calls.append(user)
            if len(self.calls) < 3:
                return json.dumps({"answer": "The saved recommendation is an LSTM.",
                                   "sources": ["U1", "P_initializer"], "uncertainty": ""})
            return json.dumps({"answer": "The saved recommendation is an LSTM [U1].",
                               "sources": ["U1"], "uncertainty": ""})

    client = Client()
    result = agent.answer_question(
        chat, "Explain the starting recommendation and initializer role.", client=client,
        question_plan=QuestionPlan(scope="setup", agent_ids=["initializer"],
                                   evidence_targets=["setup_recommendation", "agent_templates"]),
    )

    assert len(client.calls) == 3
    assert "CITATION REPAIR REQUIRED" in client.calls[-1]
    assert result["sources"] == ["U1"]


def test_missing_inline_citation_with_multiple_selected_sources_still_requires_repair():
    with pytest.raises(ValueError, match="did not cite"):
        agent._parse('{"answer":"This result is supported.","sources":["D1", "V1"]}', {"D1", "V1"})


def test_uncited_single_source_draft_is_cited_before_evidence_review(tmp_path):
    chat = core.new_chat()
    data = ("time,s_angle,a_motor\n" + "\n".join(
        f"{i / 10},{i / 100},{i % 3}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("angle.csv", data, chat["id"], upload_dir=tmp_path)

    class Client:
        calls = 0
        def complete(self, system, user):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"answer": "The median interval is 0.1 seconds.",
                                   "sources": ["D1"], "uncertainty": ""})
            assert "LOCAL CHECK FAILED" not in user
            assert "The median interval is 0.1 seconds [D1]." in user
            return json.dumps({"answer": "The median interval is 0.1 seconds [D1].",
                               "sources": ["D1"], "uncertainty": ""})

    client = Client()
    answer = agent.answer_question(chat, "What is the sampling interval?", client=client)
    assert answer["sources"] == ["D1"]
    assert "[D1]" in answer["answer"]
    assert client.calls == 2


def test_plain_prediction_ranking_is_answered_from_local_metrics_with_citation():
    comparison = {"rows": [
        {"state": "s_yaw", "normalized_rmse": .2},
        {"state": "s_pitch", "normalized_rmse": .4},
    ], "aligned_samples": 120}
    answer = agent._direct_best_state_answer("Which state predicts better than the other?", comparison)
    assert answer is not None
    assert "s_yaw" in answer["answer"]
    assert "[V2]" in answer["answer"]


def test_failure_and_agent_questions_route_to_quote_checked_diagnosis():
    run_chat = {"run_dir": "synthetic-run"}
    assert agent.should_use_run_diagnostic("Why did this run fail?", run_chat)
    assert agent.should_use_run_diagnostic("Audit the Inspector's cycle decisions", run_chat)
    assert not agent.should_use_run_diagnostic("Which state predicts better?", run_chat)
    assert not agent.should_use_run_diagnostic("Which state has larger validation RMSE?", run_chat)
    assert not agent.should_use_run_diagnostic("Why did this run fail?", {})


def test_live_agent_question_uses_saved_prompt_evidence_and_current_cycle(tmp_path, monkeypatch):
    run = tmp_path / "active-run"
    run.mkdir()
    chat = core.new_chat()
    chat["live_run_dir"] = str(run)
    chat["live_run_context"] = {
        "cycle": 2,
        "recent_agent_activity": [{
            "label": "Critic reviewed cycle 1 and proposed adjustments",
            "details": {"reasoning": "Validation error improved after shrinking the network."},
        }],
    }

    class FakeDiagnostic:
        def __init__(self, settings):
            pass

        def ask(self, run_dir, question, **kwargs):
            assert run_dir == str(run)
            assert question == "What did the agents decide?"
            assert kwargs["live_state"]["cycle"] == 2
            return {
                "summary": "The Critic proposed a smaller network after validation improved [A1].",
                "findings": [{"evidence": [{
                    "source_id": "A1", "title": "Critic prompt and response",
                    "kind": "agent_turn", "provenance": "Saved with this run",
                    "quote": "Validation error improved after shrinking the network.",
                }]}],
                "experiments": [], "limitations": [], "warnings": [],
                "model": "test-model",
            }

    monkeypatch.setattr(agent, "RunDiagnosticAgent", FakeDiagnostic)
    job = agent.ConversationJob(chat, "What did the agents decide?", purpose="live_run_chat")

    job._run()

    assert job.error is None
    assert "Critic proposed a smaller network" in job.answer["answer"]
    assert job.answer["evidence"] == [{"id": "A1", "title": "Critic prompt and response", "kind": "agent_turn"}]


def test_agent_action_language_routes_to_live_run_diagnosis():
    live_chat = {"live_run_dir": "active-run", "live_run_context": {"cycle": 1}}

    assert agent.should_use_run_diagnostic("What did the agents decide?", live_chat)
    assert agent.should_use_run_diagnostic("What did the Critic say about cycle 1?", live_chat)
    assert not agent.should_use_run_diagnostic("What can I do here?", live_chat)


def test_short_live_reply_continues_clarification_instead_of_starting_new_diagnosis(monkeypatch):
    run = {"live_run_dir": "active-run", "live_run_context": {"cycle": 2}}
    chat = core.new_chat()
    chat.update(run)
    chat["run_agent_conversations"] = {"live_run": [
        {"role": "user", "content": "Change the epoch to 2."},
        {"role": "assistant", "content": "For this training cycle or future runs?", "kind": "clarification"},
        {"role": "user", "content": "For this cycle."},
    ]}

    assert agent.should_use_run_diagnostic("For this cycle.", chat)
    assert agent._is_clarification_followup(chat, "For this cycle.")

    called = {}
    def conversational(snapshot, question, **kwargs):
        called["question"] = question
        assert snapshot["run_agent_conversations"]["live_run"][-3]["content"] == "Change the epoch to 2."
        assert snapshot["run_agent_conversations"]["live_run"][-2]["kind"] == "clarification"
        return {"status": "answer", "answer": "A running candidate cannot be changed midway [LIVE1].",
                "evidence": [{"id": "LIVE1"}]}

    monkeypatch.setattr(agent, "answer_question", conversational)
    monkeypatch.setattr(agent, "_run_diagnostic_answer", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("A clarification reply should not be routed as a new diagnosis")))
    job = agent.ConversationJob(chat, "For this cycle.", purpose="live_run_chat")
    job._run()

    assert job.error is None
    assert called["question"] == "For this cycle."
    assert "cannot be changed midway" in job.answer["answer"]


def test_conversation_job_uses_deep_diagnosis_and_attaches_verified_source_metadata(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"status": "failed", "message": "No model was trained."}), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run)

    class FakeDiagnostic:
        def __init__(self, settings):
            pass
        def ask(self, run_dir, question, **kwargs):
            assert question == "Why did this run fail?"
            return {"summary": "The run did not produce a model [M1].", "findings": [{
                "title": "No model was trained", "kind": "observed", "confidence": "high",
                "explanation": "The saved run manifest says training produced no model.",
                "evidence": [{"source_id": "M1", "quote": '"status": "failed"'}],
            }], "experiments": [], "limitations": [], "warnings": [], "mode": "ai",
                "reviewed": True, "model": "test-model", "elapsed_seconds": 0.1,
                "coverage": {"sources_used": 1, "agent_turns": 0, "templates": 0, "notes": [], "missing": []}}

    monkeypatch.setattr(agent, "RunDiagnosticAgent", FakeDiagnostic)
    job = agent.ConversationJob(chat, "Why did this run fail?")
    job._run()
    assert job.answer["diagnosis"]["findings"][0]["evidence"][0]["title"] == "run_manifest.json"
    assert job.answer["evidence"][0]["id"] == "M1"


def test_api_connection_error_names_internet_or_proxy_without_losing_chat(tmp_path, monkeypatch):
    class APIConnectionError(Exception):
        pass
    monkeypatch.setattr(agent, "answer_question", lambda *_a, **_k: (_ for _ in ()).throw(
        APIConnectionError("Could not connect to proxy")))
    chat = core.new_chat()
    job = agent.ConversationJob(chat, "Explain system identification")
    job._run()
    assert "internet or proxy" in job.error
    assert "Could not connect to proxy" in job.error_detail


def test_invalid_grounding_response_gets_actionable_explanation(monkeypatch):
    monkeypatch.setattr(agent, "answer_question", lambda *_a, **_k: (_ for _ in ()).throw(
        ValueError("The answer did not cite its evidence beside the supported claims.")))
    job = agent.ConversationJob(core.new_chat(), "What happened in this run?")

    job._run()

    assert "couldn't verify its answer against the available evidence" in job.error
    assert "The answer did not cite its evidence" in job.error_detail


def test_rate_limit_error_says_to_wait_and_retry(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

    monkeypatch.setattr(agent, "answer_question", lambda *_a, **_k: (_ for _ in ()).throw(
        RateLimitError("too many requests")))
    job = agent.ConversationJob(core.new_chat(), "Explain the data")

    job._run()

    assert "rate-limiting" in job.error
    assert "wait a moment" in job.error


def test_generic_next_step_question_asks_client_for_goal_instead_of_repeating_metrics():
    chat = core.new_chat()
    chat["run_dir"] = "a completed run"

    class MustNotCallModel:
        def complete(self, *_args):
            raise AssertionError("A broad next-step prompt should clarify the client's goal first")

    result = agent.answer_question(chat, "What should I try next?", client=MustNotCallModel())

    assert result["status"] == "clarification"
    assert result["sources"] == []
    assert "improve prediction on unseen measurements" in result["answer"]
    assert "controller test" in result["answer"]
    assert "do not establish controller readiness" in result["answer"]
    assert "99.8" not in result["answer"]


def test_next_step_question_uses_a_goal_the_client_already_stated():
    chat = core.new_chat()
    chat["messages"] = [
        {"role": "assistant", "kind": "result", "content": "The run completed."},
        {"role": "user", "content": "I want to improve prediction on unseen measurements."},
    ]

    assert agent._next_step_clarification(chat, "What should I try next?") is None


def test_model_can_ask_a_clarifying_question_without_citing_evidence():
    response = agent._parse(
        '{"status":"clarification","answer":"Are you optimizing accuracy or deployment?",'
        '"sources":[],"uncertainty":"The goal is unclear."}',
        {"D1", "V1"},
    )

    assert response["status"] == "clarification"
    assert response["sources"] == []


def test_reviewer_cannot_replace_a_clarification_with_a_metric_recap():
    chat = core.new_chat()

    class Client:
        calls = 0

        def complete(self, _system, _user):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({
                    "status": "clarification",
                    "answer": "Are you trying to improve accuracy or assess controller readiness?",
                    "sources": [], "uncertainty": "The goal is not clear yet.",
                })
            return json.dumps({
                "status": "answer", "answer": "The run scored 99.8 and is ready for MPC.",
                "sources": [], "uncertainty": "",
            })

    result = agent.answer_question(chat, "What is the best next step?", client=Client())

    assert result["status"] == "clarification"
    assert result["answer"].startswith("Are you trying")


def test_question_during_setup_is_not_mistaken_for_client_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    chat = core.new_chat()
    data = ("time,s_pitch,a_motor\n" + "\n".join(f"{i/10},{i/100},{i%3}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("angle.csv", data, chat["id"], upload_dir=tmp_path)
    core.begin_setup(chat)
    assert core.advance_setup(chat, "What does one continuous run mean?") is None
    assert chat["setup"]["stage"] == "trajectory"
    assert core.advance_setup(chat, "One continuous run")["stage"] == "angle"
