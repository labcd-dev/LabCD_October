from pathlib import Path

from frontend_streamlit.answer_benchmark import runner
from frontend_streamlit.question_orchestrator import QuestionPlan


def test_live_diagnostic_citations_are_checked_against_diagnostic_evidence(tmp_path):
    run_dir = tmp_path / "synthetic-live-run"
    case = {
        "scenario": "live_run",
        "question": "How is the current cycle progressing?",
    }
    chat = runner._scenario_chat(case, run_dir)

    available_ids, available_kinds = runner._available_answer_sources(
        chat, case["question"], "run", "live_run_chat")

    assert "F1" in available_ids
    assert "live_run" in available_kinds


def test_planned_run_coverage_source_counts_as_available_evidence(tmp_path):
    run_dir = tmp_path / "synthetic-run"
    case = {
        "scenario": "run",
        "question": "What did the initializer actually propose in this run?",
    }
    chat = runner._scenario_chat(case, run_dir)
    plan = QuestionPlan(
        scope="run", answerer="conversation", agent_ids=["initializer"],
        evidence_targets=["run_summary", "agent_activity", "agent_logs"],
    )

    available_ids, available_kinds = runner._available_answer_sources(
        chat, case["question"], "run", "question", plan=plan)

    assert "RC1" in available_ids
    assert "coverage" in available_kinds


def test_general_question_about_agent_loads_only_selected_current_template():
    case = {"scenario": "empty", "question": "How does the Actor choose a next candidate in general?"}
    chat = runner._scenario_chat(case, Path("unused"))
    plan = QuestionPlan(scope="general", agent_ids=["actor"], evidence_targets=["agent_templates"])

    sources = runner.agent.question_sources(chat, case["question"], "general", plan=plan)

    assert [source["id"] for source in sources] == ["P_actor"]
    assert sources[0]["kind"] == "template"
    assert "not proof" in sources[0]["provenance"]


def test_planner_only_cases_are_explicitly_skipped_offline(tmp_path):
    case = {"id": "planner-only-01", "category": "semantic_scope_disambiguation",
            "question": "How does the Actor choose a next candidate in general?",
            "scenario": "empty", "planner_only": True}

    record = runner._offline_case(case, tmp_path / "unused")

    assert record["outcome"] == "skipped"
    assert "only in live mode" in record["skip_reason"]
