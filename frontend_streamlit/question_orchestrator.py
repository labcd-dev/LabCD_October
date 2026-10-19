"""Semantic routing and evidence planning for LabCD conversational questions."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from backend_core.AgentSysID.agents.prompt_library import load_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings, redact


AGENT_CATALOG = {
    "data_inspector": "Checks measured data, signal roles, sampling, and engineering quality.",
    "run_setup_agent": "Before training, recommends a starting architecture and search effort from the data profile.",
    "initializer": "At run start, chooses the initial training configuration for the first candidate.",
    "explorer": "Proposes a structural escape candidate when the search is stuck.",
    "critic": "Reviews candidate metrics and identifies strengths, weaknesses, and risks.",
    "actor": "Uses critic feedback and history to set the next candidate's hyperparameters.",
    "report": "Summarizes the completed run and its measured results.",
}

EVIDENCE_TARGETS = {
    "dataset_profile", "setup_recommendation", "agent_templates", "run_summary",
    "cycle_history", "recorded_settings", "verification", "live_state",
    "agent_activity", "agent_logs", "runtime_logs", "implementation",
    "diagnostic_reference", "assistant_identity", "state_ranking",
}
EVIDENCE_CATALOG = {
    "dataset_profile": "Profile computed from the uploaded measurements and confirmed column roles.",
    "setup_recommendation": "Saved Run Setup Agent recommendation and editable starting settings.",
    "agent_templates": "Prompt templates; not proof that an agent actually ran.",
    "run_summary": "Saved run manifest and overall status/metrics.",
    "cycle_history": "Recorded metrics and configuration for individual training cycles.",
    "recorded_settings": "Settings and data split frozen for the selected run.",
    "verification": "Saved held-out rollout measurements and per-state errors.",
    "live_state": "Latest UI-captured live training state; may be incomplete.",
    "agent_activity": "Structured record of agent activity during the run.",
    "agent_logs": "Actual prompts and responses recorded for training agents.",
    "runtime_logs": "Run console output and execution errors.",
    "implementation": "Code snapshot captured with the run, when available.",
    "diagnostic_reference": "General diagnostic guidance, not an observation from the run.",
    "assistant_identity": "Application-owned LabCD identity and capability metadata.",
    "state_ranking": "Computed ranking from held-out verification metrics.",
}

class QuestionPlan(BaseModel):
    """A bounded plan; it selects only known LabCD scopes, agents, and evidence."""

    scope: Literal["run", "setup", "general"]
    answerer: Literal["conversation", "run_diagnostic"] = "conversation"
    agent_ids: list[str] = Field(default_factory=list, max_length=7)
    evidence_targets: list[str] = Field(default_factory=list, max_length=15)
    cycle_numbers: list[int] = Field(default_factory=list, max_length=24)
    ranking_goal: Literal["none", "best_state", "worst_state", "all_states"] = "none"
    confidence: float = Field(default=0.7, ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str = Field(default="", max_length=500)


_SYSTEM = str(load_prompt("question_router").get("system", "")).strip()


def _recent_context(chat: dict, question: str) -> list[dict]:
    result = []
    messages = chat.get("messages", [])
    current_question_index = next((index for index in range(len(messages) - 1, -1, -1)
                                   if messages[index].get("role") == "user" and
                                   str(messages[index].get("content", "")).strip() == question.strip()), None)
    for index, message in enumerate(messages):
        if message.get("role") not in ("user", "assistant"):
            continue
        content = str(message.get("content", "")).strip()
        if not content or index == current_question_index:
            continue
        result.append({
            "role": message["role"],
            "scope": message.get("answer_scope"),
            "kind": message.get("kind"),
            "content": content[-1800:],
        })
    return result[-10:]


def _workspace_context(chat: dict, purpose: str) -> dict:
    dataset = chat.get("dataset") or {}
    setup = chat.get("setup") or {}
    flow = chat.get("run_setup_flow") or {}
    live = chat.get("live_run_context")
    return {
        "surface_purpose": purpose,
        "run_selected": bool(chat.get("run_dir") or chat.get("live_run_dir")),
        "live_run_available": bool(chat.get("live_run_dir") and isinstance(live, dict)),
        "dataset_available": bool(dataset),
        "dataset_ready": bool(dataset.get("ready")),
        "dataset_summary": {
            "sample_count": dataset.get("rows"),
            "state_columns": dataset.get("states", []),
            "input_columns": dataset.get("actions", dataset.get("inputs", [])),
            "has_data_issues": bool(dataset.get("issues")),
        } if dataset else None,
        "setup_available": bool(dataset or flow or setup),
        "setup_stage": setup.get("stage"),
        "recommendation_stage": flow.get("stage"),
        "selected_architecture": flow.get("selected_architecture") or (chat.get("settings") or {}).get("architecture"),
        "training_active": bool(chat.get("live_run_dir")),
    }


def _pending_clarification(chat: dict, question: str) -> dict | None:
    """Expose the exact clarification being answered, including its saved scope."""
    messages = chat.get("messages", [])
    current_index = next((index for index in range(len(messages) - 1, -1, -1)
                          if messages[index].get("role") == "user" and
                          str(messages[index].get("content", "")).strip() == question.strip()), None)
    if current_index is None or current_index == 0:
        return None
    prompt = messages[current_index - 1]
    if prompt.get("role") != "assistant" or prompt.get("kind") != "clarification":
        return None
    scope = prompt.get("answer_scope")
    if scope not in ("run", "setup", "general"):
        return None
    previous_user = next((item for item in reversed(messages[:current_index - 1])
                          if item.get("role") == "user"), None)
    return {
        "scope": scope,
        "original_question": str((previous_user or {}).get("content", ""))[-1800:],
        "clarification": str(prompt.get("content", ""))[-1200:],
            "scope_is_persistent_context": True,
    }


class QuestionOrchestratorAgent:
    """Use the configured model to select answer scope and relevant evidence roles."""

    def __init__(self, client=None):
        self.client = client

    def plan(self, question: str, chat: dict, *, purpose: str = "question") -> QuestionPlan:
        if not isinstance(question, str) or not question.strip() or len(question) > 6000:
            raise ValueError("Ask a question between 1 and 6000 characters.")
        client = self.client or DiagnosticClient(DiagnosticSettings.defaults())
        payload = {
            "question": question,
            "workspace": _workspace_context(chat, purpose),
            "recent_conversation": _recent_context(chat, question),
            "pending_clarification": _pending_clarification(chat, question),
            "available_agents": AGENT_CATALOG,
            "available_evidence": EVIDENCE_CATALOG,
            "allowed_evidence_targets": sorted(EVIDENCE_TARGETS),
        }
        raw = client.complete(_SYSTEM, redact(json.dumps(payload, ensure_ascii=False, default=str)))
        value = raw.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        decoded = json.loads(value)
        if isinstance(decoded, dict) and decoded.get("answerer") not in ("conversation", "run_diagnostic"):
            # A planner may name an evidence-producing role (for example,
            # "report") where the schema expects the answer path. Keep its
            # valid scope and targets, and use the normal conversation answerer.
            decoded["answerer"] = "conversation"
        plan = QuestionPlan.model_validate(decoded)
        # Keep planning failures recoverable. Language models occasionally
        # invent a role or evidence label; discard only those labels and let
        # the normal scoped evidence builder supply its safe defaults.
        # Scope controls where facts come from, not which agent role can be
        # discussed. A general question about the Actor can still need its
        # prompt template without being treated as a run-specific question.
        plan.agent_ids = [agent for agent in plan.agent_ids if agent in AGENT_CATALOG]
        plan.evidence_targets = [target for target in plan.evidence_targets if target in EVIDENCE_TARGETS]
        # General-scope agent explanations come from the selected role's
        # template. This invariant depends on the planner's semantic role choice,
        # not on searching the client's wording for role names.
        if plan.scope == "general" and plan.agent_ids and "agent_templates" not in plan.evidence_targets:
            plan.evidence_targets.append("agent_templates")
        if plan.scope != "run" and plan.answerer == "run_diagnostic":
            plan.answerer = "conversation"
        if plan.scope == "run":
            # The run manifest is the stable baseline for every run-scoped answer.
            if "run_summary" not in plan.evidence_targets:
                plan.evidence_targets.append("run_summary")
        if plan.scope == "run" and purpose == "live_run_chat" and "live_state" not in plan.evidence_targets:
            plan.evidence_targets.append("live_state")
        if (plan.agent_ids and plan.scope == "run" and "agent_logs" not in plan.evidence_targets
                and "agent_templates" not in plan.evidence_targets):
            plan.evidence_targets.append("agent_logs")
        if (plan.agent_ids and plan.scope == "run" and "agent_logs" in plan.evidence_targets
                and "agent_activity" not in plan.evidence_targets):
            plan.evidence_targets.append("agent_activity")
        if plan.scope == "run" and plan.ranking_goal != "none":
            for target in ("state_ranking", "verification"):
                if target not in plan.evidence_targets:
                    plan.evidence_targets.append(target)
        if "state_ranking" in plan.evidence_targets and "verification" not in plan.evidence_targets:
            plan.evidence_targets.append("verification")
        if plan.scope == "run" and plan.answerer == "run_diagnostic":
            for target in ("run_summary", "recorded_settings", "cycle_history", "verification",
                           "agent_activity", "agent_logs", "runtime_logs", "diagnostic_reference"):
                if target not in plan.evidence_targets:
                    plan.evidence_targets.append(target)
            if not plan.agent_ids:
                plan.agent_ids = ["data_inspector", "initializer", "explorer", "critic", "actor", "report"]
        if plan.confidence < 0.35 and not plan.needs_clarification:
            plan.needs_clarification = True
        if plan.needs_clarification and not plan.clarification_question.strip():
            plan.clarification_question = (
                "Should I answer this from the starting setup, a particular training run, "
                "or general system-identification knowledge?"
            )
        return plan


def plan_payload(plan: QuestionPlan | dict | None) -> dict | None:
    if plan is None:
        return None
    if isinstance(plan, QuestionPlan):
        return plan.model_dump()
    return plan
