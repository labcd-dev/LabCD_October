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
            return json.dumps({"answer": "Wrapping is useful when an angle is stored modulo ±π; the choice below remains yours.",
                               "sources": ["D1"], "uncertainty": ""})

    result = agent.answer_question(chat, "Why are you asking about angle wrapping?", client=Client())
    assert "choice below remains yours" in result["answer"]


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
    assert "manifest records PINN as off" in answer


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
    result = agent.answer_question(chat, "Which state predicts better?", client=client)
    assert "s_b" in result["answer"]
    assert result["sources"] == ["V2"]
    assert client.calls == 3


def test_answer_rejects_invented_evidence_id():
    with pytest.raises(ValueError, match="not supplied"):
        agent._parse('{"answer":"Unsupported [X9]", "sources":["X9"]}', {"D1"})


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
