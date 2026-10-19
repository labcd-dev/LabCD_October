import json

from frontend_streamlit.question_orchestrator import QuestionOrchestratorAgent


def test_planner_discards_unknown_labels_and_keeps_model_selected_cycle_target():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "run",
                "answerer": "run_diagnostic",
                "agent_ids": ["invented_agent"],
                "evidence_targets": ["imaginary_evidence"],
                "cycle_numbers": [3],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "What happened in cycle 3?", {"run_dir": "synthetic-run"})

    assert plan.agent_ids  # Safe run-diagnostic defaults were restored.
    assert plan.cycle_numbers == [3]
    assert {"cycle_history", "run_summary"}.issubset(plan.evidence_targets)


def test_live_run_plan_always_includes_current_live_evidence():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "run",
                "answerer": "conversation",
                "agent_ids": [],
                "evidence_targets": [],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "How is the current cycle progressing?",
        {"live_run_dir": "synthetic-live-run", "live_run_context": {"cycle": 2}},
        purpose="live_run_chat",
    )

    assert "live_state" in plan.evidence_targets


def test_unrecognized_answerer_uses_conversation_answer_path():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "run",
                "answerer": "report",
                "agent_ids": ["report"],
                "evidence_targets": ["run_summary"],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "How did this run go?", {"run_dir": "synthetic-run"})

    assert plan.scope == "run"
    assert plan.answerer == "conversation"
    assert plan.agent_ids == ["report"]
    assert "run_summary" in plan.evidence_targets


def test_model_selected_agent_gets_its_saved_log():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "run", "answerer": "conversation", "agent_ids": ["initializer"],
                "evidence_targets": ["run_summary"],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "What did the initializer actually propose in this run?",
        {"run_dir": "synthetic-run"},
    )

    assert plan.agent_ids == ["initializer"]
    assert {"agent_activity", "agent_logs", "run_summary"}.issubset(plan.evidence_targets)


def test_semantic_agent_selection_is_not_overridden_by_mentioned_agent_names():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "run", "answerer": "conversation", "agent_ids": ["actor"],
                "evidence_targets": ["agent_logs", "cycle_history"], "cycle_numbers": [3],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "Not the Initializer: what did the Actor change after the critique in cycle 3?",
        {"run_dir": "synthetic-run"},
    )

    assert plan.agent_ids == ["actor"]
    assert plan.cycle_numbers == [3]


def test_live_panel_does_not_force_general_questions_into_run_scope():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "general", "answerer": "conversation", "agent_ids": [],
                "evidence_targets": ["diagnostic_reference"],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "In general, how does an LSTM retain sequence information?",
        {"live_run_dir": "synthetic-live-run", "live_run_context": {"cycle": 2}},
        purpose="live_run_chat",
    )

    assert plan.scope == "general"
    assert "live_state" not in plan.evidence_targets


def test_general_agent_question_can_select_agent_template_without_becoming_setup_scope():
    class Client:
        def complete(self, *_args):
            return json.dumps({
                "scope": "general", "answerer": "conversation", "agent_ids": ["actor"],
                "evidence_targets": [],
            })

    plan = QuestionOrchestratorAgent(Client()).plan(
        "How does the Actor choose a next candidate in general?", {})

    assert plan.scope == "general"
    assert plan.agent_ids == ["actor"]
    assert "agent_templates" in plan.evidence_targets


def test_pending_clarification_supplies_original_scope_and_question():
    from frontend_streamlit.question_orchestrator import _pending_clarification

    chat = {"messages": [
        {"role": "user", "content": "Which starting model should I choose?"},
        {"role": "assistant", "kind": "clarification", "answer_scope": "setup",
         "content": "Do you want sequence memory or a simpler direct mapping?"},
        {"role": "user", "content": "why?"},
    ]}

    context = _pending_clarification(chat, "why?")

    assert context["scope"] == "setup"
    assert context["original_question"] == "Which starting model should I choose?"
    assert context["scope_is_persistent_context"] is True
