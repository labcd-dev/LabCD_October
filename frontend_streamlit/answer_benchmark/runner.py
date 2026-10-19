"""Run the internal LabCD question benchmark without client data.

The default offline pass checks the actual route/evidence selectors and the
deterministic settings-edit path. ``--mode live`` asks LabCD to compose answers
for a deliberately limited, explicit subset and scores the returned answers.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend_streamlit import conversation_agent as agent  # noqa: E402
from frontend_streamlit import conversation_core as core  # noqa: E402
from backend_core.AgentSysID.agents.run_evidence import collect_evidence, redact  # noqa: E402


BANK_PATH = Path(__file__).with_name("cases.json")
DEFAULT_OUTPUT = ROOT / "artifacts" / "labcd_answer_benchmark"


def load_bank(path: Path = BANK_PATH) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for group in raw["groups"]:
        for index, value in enumerate(group["questions"], start=1):
            item = dict(group)
            item.pop("questions", None)
            item["id"] = f"{group['category']}-{index:02d}"
            if isinstance(value, str):
                item["question"] = value
            else:
                item["question"] = value["text"]
                item.update({key: child for key, child in value.items() if key != "text"})
            cases.append(item)
    return cases


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _synthetic_run(run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "synthetic-benchmark-run",
        "status": "completed",
        "architecture": "LSTM",
        "best_mse": 0.0018,
        "total_run_time_s": 74.2,
        "performance_history": [
            {"cycle": 1, "validation_mse": 0.0041, "epochs": 40, "training_time_s": 21.2,
             "architecture": "LSTM", "hidden_layers": [32, 32]},
            {"cycle": 2, "validation_mse": 0.0018, "epochs": 50, "training_time_s": 26.0,
             "architecture": "LSTM", "hidden_layers": [64, 64]},
            {"cycle": 3, "validation_mse": 0.0022, "epochs": 32, "training_time_s": 27.0,
             "architecture": "LSTM", "hidden_layers": [64, 32]},
        ],
    }
    verification = {
        "aligned_samples": 100,
        "protocol": "Synthetic held-out benchmark data; reset to measured state at each chunk start.",
        "states": [
            {"state": "s_speed", "rmse": 0.12, "mae": 0.09, "rmse_over_test_std": 0.12, "valid_samples": 100},
            {"state": "s_heading", "rmse": 0.31, "mae": 0.24, "rmse_over_test_std": 0.31, "valid_samples": 100},
            {"state": "s_temp", "rmse": 0.48, "mae": 0.40, "rmse_over_test_std": 0.48, "valid_samples": 100},
        ],
    }
    context = {
        "phase": "completed",
        "initial_config": {"ARCHITECTURE": "LSTM", "EPOCHS": 50, "BATCH_SIZE": 32},
        "effective_config": {"ARCHITECTURE": "LSTM", "EPOCHS": 50, "BATCH_SIZE": 32},
        "data_summary": {"states": ["s_speed", "s_heading", "s_temp"], "inputs": ["a_drive", "a_fan"]},
        "prompts": {
            "initializer": "Synthetic benchmark prompt: select a starting model from the supplied measurement roles.",
            "critic": "Synthetic benchmark prompt: review the recorded candidate evidence.",
        },
        "implementation": {"synthetic_reference.py": "Benchmark-only synthetic implementation note."},
    }
    activity = [
        {"id": "initializer:1", "agent": "initializer", "event": "proposal", "cycle": 1},
        {"id": "critic:1", "agent": "critic", "event": "review", "cycle": 1},
        {"id": "actor:2", "agent": "actor", "event": "proposal", "cycle": 2},
    ]
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "verification_summary.json", verification)
    _write_json(run_dir / "run_context.json", context)
    _write_json(run_dir / "agent_activity.json", activity)
    _write_json(run_dir / "diagnostic_state.json", {"status": "completed", "cycle": 3})
    logs = run_dir / "Agents_log"
    logs.mkdir(exist_ok=True)
    (logs / "synthetic_agents.txt").write_text(
        "TIMESTAMP: [2026-10-14T12:00:00Z]\nAGENT: [Data Inspector]\n"
        "The synthetic input a_drive changes over time; the sample interval is 0.01 seconds.\n\n"
        "TIMESTAMP: [2026-10-14T12:01:00Z]\nAGENT: [Initializer]\n"
        "For this synthetic benchmark run, the initial proposal used an LSTM with 32 units per layer.\n\n"
        "TIMESTAMP: [2026-10-14T12:02:00Z]\nAGENT: [Critic]\n"
        "Cycle 1 review: validation MSE was 0.0041. The network may lack capacity; recommend widening the next candidate to 64 units per recurrent layer. This is a recommendation, not proof of cause.\n\n"
        "TIMESTAMP: [2026-10-14T12:03:00Z]\nAGENT: [Actor]\n"
        "For cycle 2, changed the recurrent layer widths from [32, 32] to [64, 64] after the Critic's recommendation. Kept the starting learning rate at 0.001 and batch size at 32.\n",
        encoding="utf-8")
    (run_dir / "runtime_console.log").write_text(
        "Synthetic benchmark runtime: cycle 1, cycle 2, and cycle 3 completed.\n", encoding="utf-8")
    return manifest


def _dataset_record() -> dict:
    return {
        "name": "synthetic_process.csv",
        "path": "<synthetic benchmark fixture>",
        "sha256": "synthetic-only",
        "rows": 1000,
        "columns": ["time", "s_speed", "s_heading", "s_temp", "a_drive", "a_fan"],
        "states": ["s_speed", "s_heading", "s_temp"],
        "actions": ["a_drive", "a_fan"],
        "derivatives": [],
        "sample_period": 0.01,
        "issues": [],
        "warnings": [],
        "ready": True,
        "analysis": {
            "sample_period": 0.01,
            "interval_count": 999,
            "irregular_interval_count": 0,
            "reset_rows": [],
            "boundary_events": [],
            "detected_wrap_states": [],
            "possible_boundary_count": 0,
            "correlated_state_pairs": [],
            "signals": [
                {"name": "s_speed", "std": 1.2, "min": 0.0, "max": 4.2, "wrap_jumps": 0},
                {"name": "s_heading", "std": 0.4, "min": -1.2, "max": 1.4, "wrap_jumps": 0},
                {"name": "s_temp", "std": 2.1, "min": 19.0, "max": 26.0, "wrap_jumps": 0},
            ],
            "inputs": [
                {"name": "a_drive", "min": 0.0, "max": 1.0, "std": 0.25, "constant": False},
                {"name": "a_fan", "min": 0.0, "max": 1.0, "std": 0.20, "constant": False},
            ],
        },
    }


def _setup_chat() -> dict:
    chat = core.new_chat()
    chat["dataset"] = _dataset_record()
    chat["setup"] = {"stage": "review"}
    chat["settings"].update({
        "architecture": "LSTM", "epochs": 50, "batch_size": 32,
        "manual_starting_lr": 0.001, "manual_starting_hidden_layers": [64, 64],
        "use_pinn": False, "angle_indices": [], "multi_trajectory": False,
        "num_layers_min": 2, "num_layers_max": 2,
        "learning_rate_min": 0.00001, "learning_rate_max": 0.01,
    })
    chat["run_setup_flow"] = {
        "stage": "ready", "recommended_architecture": "LSTM",
        "architecture_reason": "The synthetic example has sequential measurements.",
        "recommended_effort": "fast", "effort_reason": "A small initial search is enough for this fixture.",
        "recommended_cycles": 3, "recommended_history_steps": 10, "confidence": "medium",
        "selected_architecture": "LSTM",
        "initializer_proposal": {"architecture": "LSTM", "manual_starting_hidden_layers": [64, 64]},
        "physics_advisory": {"status": "not_provided"},
    }
    return chat


def _scenario_chat(case: dict, run_dir: Path) -> dict:
    scenario = case["scenario"]
    if scenario == "empty":
        return core.new_chat()
    if scenario in ("dataset", "setup", "setup_followup"):
        chat = _setup_chat()
        if scenario == "setup_followup":
            chat["messages"] = [
                {"role": "user", "content": "Which starting model should I choose?"},
                {"role": "assistant", "kind": "clarification", "answer_scope": "setup",
                 "content": "Do you want to prioritize sequence memory or a simpler direct mapping?"},
                {"role": "user", "content": case["question"]},
            ]
        return chat

    _synthetic_run(run_dir)
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    if scenario == "live_run":
        chat["live_run_dir"] = str(run_dir)
        chat["live_run_context"] = {
            "status": "running", "cycle": 2, "candidate": 2, "epoch": 18,
            "epoch_limit": 50, "validation_mse": 0.0023, "current_loss": 0.0031,
            "completed_cycles": 1, "architecture": "LSTM",
        }
    if scenario == "run_followup":
        chat["messages"] = [
            {"role": "user", "content": "What happened in cycle 2?"},
            {"role": "assistant", "kind": "clarification", "answer_scope": "run",
             "content": "Which candidate or cycle do you mean?"},
            {"role": "user", "content": case["question"]},
        ]
    return chat


def _matches(values: list[str], actual: set[str]) -> list[str]:
    return [value for value in values if value not in actual]


def _setting_case_result(case: dict, chat: dict) -> dict:
    question = case["question"]
    failures = []
    if not core.is_settings_edit_request(question):
        failures.append("request was not dispatched to the settings planner")
    changes = core._requested_setting_changes(chat, question)
    expected = case.get("changes", {})
    for field, wanted in expected.items():
        if field not in changes:
            failures.append(f"requested setting {field!r} was not extracted")
        elif changes[field] != wanted:
            failures.append(f"{field}: expected {wanted!r}, extracted {changes[field]!r}")
    if changes:
        try:
            validated = core.apply_changes(chat["settings"], changes)
            for field, wanted in expected.items():
                if validated.get(field) != wanted:
                    failures.append(f"{field}: validated value is {validated.get(field)!r}, expected {wanted!r}")
        except (ValueError, TypeError) as exc:
            expected_rejection = case.get("expected_rejection")
            if not expected_rejection or not re.search(expected_rejection, str(exc), re.I):
                failures.append(f"settings schema rejected the proposed change: {type(exc).__name__}: {exc}")
        else:
            if case.get("expected_rejection"):
                failures.append("the setting schema accepted a value that this case expects to reject")
    return {"handler": "planner", "parsed_changes": changes, "failures": failures}


def _local_answer_case(case: dict, chat: dict) -> dict:
    class NoNetwork:
        def complete(self, *_args, **_kwargs):
            raise AssertionError("Offline benchmark attempted an API call")

    answer = agent.answer_question(chat, case["question"], client=NoNetwork(),
                                   purpose=case.get("purpose", "question"))
    failures = []
    if answer.get("status") != case.get("expected_status", "answer"):
        failures.append(f"expected status {case.get('expected_status', 'answer')}, got {answer.get('status')}")
    if case.get("category") == "missing_run_safety" and not re.search(
            r"select(?:ion)? (?:a )?run|select(?:ed)? in this conversation", answer.get("answer", ""), re.I):
        failures.append("missing-run reply did not clearly ask for a run selection")
    if case.get("category") == "general_conversation_and_navigation" and not answer.get("answer"):
        failures.append("local greeting/help reply was empty")
    return {"status": answer.get("status"), "answer": answer.get("answer", ""), "sources": answer.get("sources", []),
            "failures": failures}


def _provider_block_reason(error: str = "", answer: str = "") -> str | None:
    if re.search(r"AI diagnosis is unavailable|AI service did not produce a usable evidence-grounded diagnosis",
                 answer or "", re.I):
        return "Provider unavailable; LabCD returned its local evidence fallback."
    if re.search(r"unsupported_country_region_territory|country, region, or territory not supported",
                 error or "", re.I):
        return "Provider rejected the call because the request region is unsupported (HTTP 403)."
    if re.search(r"\b401\b|AuthenticationError", error or "", re.I):
        return "Provider rejected the configured credentials (HTTP 401)."
    if re.search(r"\b429\b|RateLimitError", error or "", re.I):
        return "Provider rate limited the request (HTTP 429)."
    if re.search(
            r"\b403\b|PermissionDeniedError|"
            r"APIConnectionError|APITimeoutError|connection refused|timed out",
            error or "", re.I):
        return "Configured AI provider rejected or could not serve the request."
    return None


def _offline_case(case: dict, run_dir: Path) -> dict:
    if case.get("planner_only"):
        return {"id": case["id"], "category": case["category"], "question": case["question"],
                "mode": "offline", "outcome": "skipped", "failures": [],
                "skip_reason": "This case evaluates semantic LLM planning and runs only in live mode."}
    chat = _scenario_chat(case, run_dir)
    question = case["question"]
    failures: list[str] = []
    expected_handler = case.get("expected_handler", "answer")
    result: dict[str, Any] = {"id": case["id"], "category": case["category"],
                              "question": question, "mode": "offline"}

    if expected_handler == "planner":
        result.update(_setting_case_result(case, chat))
        result["setting_passed"] = not result["failures"]
        return result

    purpose = case.get("purpose", "question")
    actual_scope = agent.route_question(question, chat, purpose=purpose)
    expected_scope = case.get("expected_scope")
    route_passed = None if expected_scope is None else actual_scope == expected_scope
    if expected_scope and actual_scope != expected_scope:
        failures.append(f"route: expected {expected_scope}, got {actual_scope}")
    result.update(expected_scope=expected_scope, actual_scope=actual_scope)

    sources = agent.question_sources(chat, question, actual_scope, purpose=purpose, identity=agent.IDENTITY)
    source_kinds = {source["kind"] for source in sources}
    source_ids = {source["id"] for source in sources}
    missing_kinds = _matches(case.get("required_source_kinds", []), source_kinds)
    missing_ids = _matches(case.get("required_source_ids", []), source_ids)
    forbidden_kinds = sorted(source_kinds.intersection(case.get("forbidden_source_kinds", [])))
    if missing_kinds:
        failures.append(f"missing evidence kinds: {', '.join(missing_kinds)}")
    if missing_ids:
        failures.append(f"missing evidence IDs: {', '.join(missing_ids)}")
    if forbidden_kinds:
        failures.append(f"unexpected evidence kinds: {', '.join(forbidden_kinds)}")
    result.update(source_kinds=sorted(source_kinds), source_ids=sorted(source_ids))
    result["route_passed"] = route_passed
    result["retrieval_passed"] = not (missing_kinds or missing_ids or forbidden_kinds)

    if case.get("offline_answer"):
        try:
            result["local_answer"] = _local_answer_case(case, chat)
            failures.extend(result["local_answer"].get("failures", []))
        except Exception as exc:
            failures.append(f"offline answer path: {type(exc).__name__}: {exc}")
    if "local_answer" in result:
        result["local_answer_passed"] = not result["local_answer"].get("failures")

    if (actual_scope == "run" and chat.get("run_dir") and
            case["category"] == "run_state_ranking"):
        comparison = agent.analysis.compare_run_states(str(run_dir))
        result["ranking_oracle"] = {
            "best": min(comparison["rows"], key=lambda row: row["normalized_rmse"])["state"],
            "worst": max(comparison["rows"], key=lambda row: row["normalized_rmse"])["state"],
            "samples": comparison.get("aligned_samples"),
        }

    result["failures"] = failures
    return result


def _available_answer_sources(chat: dict, question: str, scope: str, purpose: str,
                              plan=None) -> tuple[set[str], set[str]]:
    """Return evidence from the same answer path the live response uses."""
    selected = agent.question_sources(chat, question, scope, purpose=purpose,
                                      identity=agent.IDENTITY, plan=plan)
    available_ids = {source["id"] for source in selected}
    available_kinds = {source["kind"] for source in selected}
    run_dir = chat.get("live_run_dir") or chat.get("run_dir")
    uses_diagnostic = (plan.answerer == "run_diagnostic" if plan is not None
                       else agent.should_use_run_diagnostic(question, chat))
    if scope == "run" and run_dir and uses_diagnostic:
        diagnostic_evidence = collect_evidence(
            run_dir, question, live_state=chat.get("live_run_context"),
            agent_targets=plan.agent_ids if plan else None,
            cycle_numbers=plan.cycle_numbers if plan else None,
            evidence_targets=plan.evidence_targets if plan else None,
        )
        available_ids.update(source.id for source in diagnostic_evidence.sources)
        available_kinds.update(source.kind for source in diagnostic_evidence.sources)
    return available_ids, available_kinds


def _live_case(case: dict, run_dir: Path, timeout: int) -> dict:
    chat = _scenario_chat(case, run_dir)
    if case.get("expected_handler") == "planner":
        record = _offline_case(case, run_dir)
        record.update(mode="live", outcome="failed" if record.get("failures") else "passed",
                      live_answer_skipped="setting edits are scored in the deterministic planner path",
                      answer_passed=None)
        return record

    job = agent.ConversationJob(chat, case["question"], purpose=case.get("purpose", "question"))
    started = time.monotonic()
    job.start()
    job.thread.join(timeout=max(1, timeout))
    elapsed = time.monotonic() - started
    if job.thread.is_alive():
        job.cancel()
        record = {"status": "timeout", "error": f"exceeded {timeout} seconds", "answer": "", "sources": []}
    elif job.error:
        record = {"status": "error", "error": redact(job.error_detail or job.error), "answer": "", "sources": []}
    elif job.answer is None:
        record = {"status": "error", "error": "No answer was returned", "answer": "", "sources": []}
    else:
        record = {
            "status": job.answer.get("status", "answer"),
            "answer": redact(str(job.answer.get("answer", ""))),
            "sources": list(job.answer.get("sources", [])),
            "evidence": job.answer.get("evidence", []),
            "answer_scope": job.answer.get("answer_scope", job.answer_scope),
            "model": redact(str(job.answer.get("model", ""))),
            "uncertainty": redact(str(job.answer.get("uncertainty", ""))),
            "diagnostic": ({key: job.answer["diagnosis"].get(key)
                            for key in ("mode", "reviewed", "warnings", "diagnostic_error")
                            if key in job.answer["diagnosis"]}
                           if isinstance(job.answer.get("diagnosis"), dict) else None),
        }
    provider_block_reason = _provider_block_reason(str(record.get("error", "")),
                                                   str(record.get("answer", "")))
    provider_blocked = provider_block_reason is not None
    if provider_blocked:
        record["provider_block_reason"] = provider_block_reason
    failures = []
    plan = job.question_plan
    if case.get("planner_only") and plan is None:
        failures.append("question planner did not return a semantic plan")
    expected_status = case.get("expected_status", "answer")
    if not provider_blocked and record["status"] != expected_status:
        failures.append(f"answer status: expected {expected_status}, got {record['status']}")
    expected_scope = case.get("expected_scope")
    actual_scope = record.get("answer_scope", job.answer_scope)
    planned_scope = plan.scope if plan else None
    if expected_scope and (planned_scope or actual_scope) != expected_scope:
        failures.append(f"question scope: expected {expected_scope}, got {planned_scope or actual_scope}")
    expected_agents = case.get("expected_agent_ids")
    expected_targets = case.get("required_plan_evidence_targets", [])
    forbidden_targets = case.get("forbidden_plan_evidence_targets", [])
    expected_cycles = case.get("expected_cycle_numbers")
    expected_ranking_goal = case.get("expected_ranking_goal")
    expected_answerer = case.get("expected_plan_answerer")
    if plan and expected_agents is not None and set(plan.agent_ids) != set(expected_agents):
        failures.append(f"planner roles: expected {sorted(expected_agents)}, got {sorted(plan.agent_ids)}")
    missing_plan_targets = _matches(expected_targets, set(plan.evidence_targets)) if plan else []
    unexpected_plan_targets = sorted(set(forbidden_targets).intersection(plan.evidence_targets)) if plan else []
    if plan and missing_plan_targets:
        failures.append("planner omitted evidence targets: " + ", ".join(missing_plan_targets))
    if plan and unexpected_plan_targets:
        failures.append("planner selected out-of-scope evidence targets: " + ", ".join(unexpected_plan_targets))
    if plan and expected_cycles is not None and plan.cycle_numbers != expected_cycles:
        failures.append(f"planner cycle targets: expected {expected_cycles}, got {plan.cycle_numbers}")
    if plan and expected_ranking_goal is not None and plan.ranking_goal != expected_ranking_goal:
        failures.append(f"planner ranking goal: expected {expected_ranking_goal}, got {plan.ranking_goal}")
    if plan and expected_answerer is not None and plan.answerer != expected_answerer:
        failures.append(f"planner answerer: expected {expected_answerer}, got {plan.answerer}")
    if record.get("error") and not provider_blocked:
        failures.append("live model or diagnostic call failed")
    answer_text = record.get("answer", "")
    if not provider_blocked and expected_status == "answer" and not answer_text.strip():
        failures.append("answer text is empty")
    if not provider_blocked and expected_scope and expected_scope != "general" and expected_status == "answer":
        if not record.get("sources"):
            failures.append("evidence-scoped answer returned no source IDs")
        invalid_ids = [source for source in record.get("sources", [])
                       if not isinstance(source, str) or not re.fullmatch(r"(?:[A-Z][A-Za-z0-9]*|P_[a-z_]+)", source)]
        if invalid_ids:
            failures.append("answer returned malformed evidence source IDs")
    available_ids, available_kinds = _available_answer_sources(
        chat, case["question"], actual_scope or "general", case.get("purpose", "question"),
        plan=job.question_plan,
    )
    unexpected_ids, missing_kinds, missing_ids, forbidden_kinds = [], [], [], []
    if expected_status == "answer":
        unexpected_ids = sorted(set(record.get("sources", [])) - available_ids)
        if unexpected_ids:
            failures.append("answer cited unavailable evidence IDs: " + ", ".join(unexpected_ids))
        missing_kinds = _matches(case.get("required_source_kinds", []), available_kinds)
        missing_ids = _matches(case.get("required_source_ids", []), available_ids)
        forbidden_kinds = sorted(available_kinds.intersection(case.get("forbidden_source_kinds", [])))
        if missing_kinds:
            failures.append("retrieval missed expected source kinds: " + ", ".join(missing_kinds))
        if missing_ids:
            failures.append("retrieval missed expected source IDs: " + ", ".join(missing_ids))
        if forbidden_kinds:
            failures.append("retrieval included out-of-scope source kinds: " + ", ".join(forbidden_kinds))

    if (not provider_blocked and case["category"] == "run_state_ranking" and chat.get("run_dir") and answer_text):
        mismatch = agent._check_state_ranking(
            {"answer": answer_text, "sources": record.get("sources", [])},
            agent.analysis.compare_run_states(str(run_dir)), case["question"])
        if mismatch:
            failures.append(f"measured-rank check: {mismatch}")

    for alternatives in ([] if provider_blocked else case.get("must_mention_any", [])):
        if not any(re.search(pattern, answer_text, re.I) for pattern in alternatives):
            failures.append("answer missed expected fact: " + " or ".join(alternatives))
    for pattern in ([] if provider_blocked else case.get("must_mention_all", [])):
        if not re.search(pattern, answer_text, re.I):
            failures.append(f"answer missed expected fact: {pattern}")
    for pattern in ([] if provider_blocked else case.get("must_not_mention", [])):
        if re.search(pattern, answer_text, re.I):
            failures.append(f"answer included prohibited claim or nudge: {pattern}")

    model_name = str(record.get("model") or "")
    local_model = (model_name in {"LabCD", "Computed from saved run evidence", "LabCD run router", "LabCD clarification"}
                   or model_name.startswith("LabCD "))
    warnings = []
    if (not provider_blocked and case["category"] in ("general_system_identification", "general_model_concepts") and re.search(
            r"\b(?:upload|attach|send me|paste your (?:data|dataset)|csv or excel)\b", answer_text, re.I)):
        warnings.append("general explanation may have been redirected toward uploading data")
    if answer_text and len(answer_text.split()) > 250:
        warnings.append("answer exceeded 250 words; review for unnecessary length")
    record.update({"id": case["id"], "category": case["category"], "question": case["question"],
                   "mode": "live", "elapsed_seconds": round(elapsed, 2), "failures": failures,
                   "warnings": warnings, "expected_scope": expected_scope,
                   "answer_scope": actual_scope,
                   "planner": ({"scope": plan.scope, "answerer": plan.answerer,
                                "agent_ids": plan.agent_ids, "evidence_targets": plan.evidence_targets,
                                "cycle_numbers": plan.cycle_numbers,
                                "ranking_goal": plan.ranking_goal, "confidence": plan.confidence,
                                "needs_clarification": plan.needs_clarification} if plan else None),
                   "route_passed": None if expected_scope is None else (planned_scope or actual_scope) == expected_scope,
                   "planner_scope_passed": (None if plan is None or expected_scope is None
                                            else plan.scope == expected_scope),
                   "planner_agent_selection_passed": (None if plan is None or expected_agents is None
                                                       else set(plan.agent_ids) == set(expected_agents)),
                   "planner_evidence_selection_passed": (None if plan is None or
                       (not expected_targets and not forbidden_targets) else
                       not (missing_plan_targets or unexpected_plan_targets)),
                   "planner_cycle_selection_passed": (None if plan is None or expected_cycles is None
                                                      else plan.cycle_numbers == expected_cycles),
                   "planner_ranking_classification_passed": (None if plan is None or expected_ranking_goal is None
                                                              else plan.ranking_goal == expected_ranking_goal),
                   "planner_answerer_selection_passed": (None if plan is None or expected_answerer is None
                                                          else plan.answerer == expected_answerer),
                   "retrieval_passed": (expected_status != "answer" or
                                        not (unexpected_ids or missing_kinds or missing_ids or forbidden_kinds)),
                   "provider_blocked": provider_blocked,
                   "outcome": "failed" if failures else ("blocked" if provider_blocked else "passed"),
                   "answer_passed": None if provider_blocked or "answer" not in record else not failures,
                   "model_answer_passed": None if provider_blocked or not model_name or local_model else not failures})
    return record


def _select_cases(cases: list[dict], category: str | None, limit: int | None) -> list[dict]:
    if category:
        cases = [case for case in cases if case["category"] == category]
    if limit is None or limit >= len(cases):
        return cases
    # Round-robin across categories so a small live sample still covers intent types.
    buckets: dict[str, list[dict]] = {}
    for case in cases:
        buckets.setdefault(case["category"], []).append(case)
    selected = []
    while len(selected) < limit:
        progressed = False
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _count(records: list[dict], category: str | None = None) -> tuple[int, int]:
    subset = [record for record in records if category is None or record["category"] == category]
    return sum(not record.get("failures") for record in subset), len(subset)


def _make_report(records: list[dict], mode: str, selected: list[dict]) -> dict:
    categories = sorted({record["category"] for record in records})
    by_category = {}
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        by_category[category] = {
            "passed": sum(row.get("outcome", "failed" if row.get("failures") else "passed") == "passed" for row in rows),
            "failed": sum(row.get("outcome", "failed" if row.get("failures") else "passed") == "failed" for row in rows),
            "blocked": sum(row.get("outcome") == "blocked" for row in rows),
            "skipped": sum(row.get("outcome") == "skipped" for row in rows),
            "total": len(rows),
            "failures": [row["id"] for row in rows if row.get("failures")],
        }
    component_specs = (
        ("routing", "route_passed"),
        ("llm_scope_selection", "planner_scope_passed"),
        ("agent_role_selection", "planner_agent_selection_passed"),
        ("evidence_target_selection", "planner_evidence_selection_passed"),
        ("cycle_target_selection", "planner_cycle_selection_passed"),
        ("state_ranking_intent", "planner_ranking_classification_passed"),
        ("answerer_selection", "planner_answerer_selection_passed"),
        ("evidence_retrieval", "retrieval_passed"),
        ("settings_edits", "setting_passed"),
        ("local_answers", "local_answer_passed"),
        ("completed_live_responses", "answer_passed"),
        ("model_generated_answers", "model_answer_passed"),
    )
    components = {}
    for name, field in component_specs:
        eligible = [row for row in records if row.get(field) is not None]
        passed = sum(bool(row[field]) for row in eligible)
        components[name] = {
            "passed": passed,
            "total": len(eligible),
            "pass_rate": round(passed / len(eligible), 4) if eligible else None,
        }
    passed = sum(row.get("outcome", "failed" if row.get("failures") else "passed") == "passed" for row in records)
    failed = sum(row.get("outcome", "failed" if row.get("failures") else "passed") == "failed" for row in records)
    blocked = sum(row.get("outcome") == "blocked" for row in records)
    skipped = sum(row.get("outcome") == "skipped" for row in records)
    scored = passed + failed
    return {
        "benchmark": "LabCD answer routing and quality benchmark",
        "benchmark_version": 1,
        "mode": mode,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "synthetic_data_only": True,
        "total_cases": len(records),
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "skipped": skipped,
        "scored_cases": scored,
        "pass_rate": round(passed / scored, 4) if scored else None,
        "provider_blocked_cases": sum(bool(row.get("provider_blocked")) for row in records),
        "warning_count": sum(len(row.get("warnings", [])) for row in records),
        "by_category": by_category,
        "components": components,
        "live_api_answer_cases": sum(row.get("mode") == "live" and "answer" in row for row in records),
        "planner_cases": sum(row.get("expected_handler") == "planner" for row in selected),
        "records": records,
    }


def _reclassify_provider_blocked(records: list[dict]) -> None:
    """Re-score saved live output when transport failures were initially counted as answer failures."""
    independent_prefixes = ("route:", "answer route:", "question scope:", "planner ",
                            "retrieval", "answer cited unavailable")
    for row in records:
        if row.get("mode") != "live":
            continue
        reason = _provider_block_reason(str(row.get("error", "")), str(row.get("answer", "")))
        if reason:
            row["provider_blocked"] = True
            row["provider_block_reason"] = reason
            row["failures"] = [failure for failure in row.get("failures", [])
                               if failure.lower().startswith(independent_prefixes)]
            row["answer_passed"] = None
            row["outcome"] = "failed" if row["failures"] else "blocked"
        else:
            row["provider_blocked"] = False
            row["outcome"] = "failed" if row.get("failures") else "passed"
            row["answer_passed"] = (not row["failures"] if "answer" in row and
                                    not row.get("live_answer_skipped") else None)


def _markdown(report: dict) -> str:
    lines = [
        "# LabCD answer benchmark",
        "",
        f"- Mode: **{report['mode']}**",
        f"- Cases: **{report['total_cases']}** · passed **{report['passed']}** · failed **{report['failed']}** · blocked **{report['blocked']}**",
        f"- Pass rate on scored cases ({report['scored_cases']}): **{report['pass_rate']:.1%}**" if report["pass_rate"] is not None else "- Pass rate: n/a",
        "- Dataset/run inputs: synthetic fixtures only",
        "",
        "## Score by evaluation layer",
        "",
        "| Layer | Passed | Pass rate |",
        "|---|---:|---:|",
    ]
    for layer, values in report["components"].items():
        rate = f"{values['pass_rate']:.1%}" if values["pass_rate"] is not None else "n/a"
        lines.append(f"| `{layer}` | {values['passed']} / {values['total']} | {rate} |")
    lines.extend([
        "",
        "## Category results",
        "",
        "| Category | Passed | Failed | Blocked | Skipped |",
        "|---|---:|---:|---:|---:|",
    ])
    for category, values in report["by_category"].items():
        lines.append(f"| `{category}` | {values['passed']} | {values['failed']} | {values['blocked']} | {values['skipped']} |")
    lines.extend(["", "## Failed cases", ""])
    failed_rows = [row for row in report["records"] if row.get("failures")]
    if not failed_rows:
        lines.append("No scored benchmark checks failed.")
    else:
        for row in failed_rows:
            lines.append(f"### {row['id']} · {row['category']}")
            lines.append("")
    blocked_rows = [row for row in report["records"] if row.get("outcome") == "blocked"]
    if blocked_rows:
        lines.extend(["", "## Unscored provider-blocked cases", ""])
        for row in blocked_rows:
            reason = row.get("provider_block_reason", "The configured provider did not return an answer.")
            lines.append(f"- `{row['id']}`: {reason}")
            lines.append(f"> {row['question']}")
            lines.append("")
            lines.extend(f"- {failure}" for failure in row["failures"])
            if row.get("answer"):
                snippet = " ".join(row["answer"].split())[:700]
                lines.extend(["", f"Answer excerpt: {snippet}"])
            lines.append("")
    warning_rows = [row for row in report["records"] if row.get("warnings")]
    if warning_rows:
        lines.extend(["", "## Review warnings", ""])
        for row in warning_rows:
            for warning in row["warnings"]:
                lines.append(f"- `{row['id']}`: {warning}")
    return "\n".join(lines).rstrip() + "\n"


def _load_dotenv_for_live() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline",
                        help="offline checks routing, retrieval, and setting edits; live also asks LabCD to answer")
    parser.add_argument("--limit", type=int,
                        help="maximum cases; required for live mode so API usage is always explicit")
    parser.add_argument("--category", help="only run one exact category name (see --list-categories)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for timestamped JSON and Markdown reports")
    parser.add_argument("--timeout", type=int, default=180, help="per-question live answer timeout in seconds")
    parser.add_argument("--reclassify-report", type=Path,
                        help="re-score a saved live report without making any model/API calls")
    parser.add_argument("--list-categories", action="store_true", help="list benchmark categories and exit")
    args = parser.parse_args(argv)

    cases = load_bank()
    if args.list_categories:
        for category in sorted({case["category"] for case in cases}):
            print(category)
        return 0
    if args.reclassify_report:
        source_path = args.reclassify_report.expanduser().resolve()
        saved = json.loads(source_path.read_text(encoding="utf-8"))
        records = saved.get("records")
        if not isinstance(records, list):
            parser.error("the selected report has no records list")
        _reclassify_provider_blocked(records)
        case_by_id = {case["id"]: case for case in cases}
        selected_for_report = [case_by_id[row["id"]] for row in records if row.get("id") in case_by_id]
        report = _make_report(records, saved.get("mode", "live"), selected_for_report)
        report["reanalysis_of"] = str(source_path)
        report["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        report_dir = args.output.expanduser()
        if not report_dir.is_absolute():
            report_dir = ROOT / report_dir
        stamp = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S_UTC_reanalysis")
        report_dir = report_dir / stamp
        report_dir.mkdir(parents=True, exist_ok=False)
        _write_json(report_dir / "report.json", report)
        (report_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        print(f"Re-scored {report['total_cases']} saved cases without calling the provider.")
        print(f"Passed: {report['passed']} · failed: {report['failed']} · blocked: {report['blocked']}.")
        print(f"Report: {report_dir / 'report.md'}")
        return 1 if report["failed"] else 0
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.mode == "live" and args.limit is None:
        parser.error("live mode requires an explicit --limit; use --mode live --limit 20 for a stratified 20-question sample")
    selected = _select_cases(cases, args.category, args.limit)
    if not selected:
        parser.error("no cases selected; check the category name with --list-categories")
    if args.mode == "live":
        _load_dotenv_for_live()

    records = []
    with tempfile.TemporaryDirectory(prefix="labcd-answer-benchmark-") as temporary:
        temp = Path(temporary)
        run_dir = temp / "synthetic_run"
        _synthetic_run(run_dir)
        for number, case in enumerate(selected, start=1):
            if args.mode == "live":
                record = _live_case(case, run_dir, args.timeout)
            else:
                record = _offline_case(case, run_dir)
                record["expected_handler"] = case.get("expected_handler", "answer")
                if record.get("outcome") != "skipped":
                    record["outcome"] = "failed" if record.get("failures") else "passed"
            records.append(record)
            outcome = record.get("outcome")
            if outcome == "blocked":
                status = "BLOCKED"
            elif outcome == "skipped":
                status = "SKIP"
            elif outcome == "failed" or record.get("failures"):
                status = "FAIL"
            else:
                status = "PASS"
            print(f"[{number:03d}/{len(selected):03d}] {status} {case['id']}")

    report = _make_report(records, args.mode, selected)
    report_dir = args.output.expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_dir = report_dir / f"run_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    _write_json(report_dir / "report.json", report)
    (report_dir / "report.md").write_text(_markdown(report), encoding="utf-8")

    rate = f"{report['pass_rate']:.1%}" if report["pass_rate"] is not None else "n/a"
    print(f"\nPassed: {report['passed']} · failed: {report['failed']} · blocked: {report['blocked']} · scored pass rate: {rate}.")
    print(f"Report: {report_dir / 'report.md'}")
    print(f"Data:   {report_dir / 'report.json'}")
    if args.mode == "live":
        print("Live answers used the configured LabCD provider; planner setting edits were checked locally.")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
