"""Question-driven assistant grounded in uploaded measurements and saved run evidence."""
from __future__ import annotations

import copy
import json
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend_core.AgentSysID.agents.prompt_library import load_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings
from backend_core.AgentSysID.agents.run_evidence import collect_evidence, redact

try:
    from . import conversation_analysis as analysis
except ImportError:
    import conversation_analysis as analysis


class Answer(BaseModel):
    status: Literal["answer", "clarification"] = "answer"
    answer: str = Field(min_length=1, max_length=9000)
    sources: list[str] = Field(default_factory=list, max_length=15)
    uncertainty: str = Field(default="", max_length=1000)


_PROMPT = load_prompt("conversation_agent")
SYSTEM = str(_PROMPT.get("system", "")).strip()
REVIEW = str(_PROMPT.get("reviewer", "")).strip()
IDENTITY = _PROMPT.get("identity", {})

_IDENTITY_PATTERNS = (
    r"\b(?:who|what)\s+are\s+you\b",
    r"\bwhat(?:'s|\s+is)\s+your\s+(?:name|role|purpose|job|identity|creator|maker|developer)\b",
    r"\bwho\s+(?:made|built|created|developed|designed|coded|programmed|trained)\s+(?:you|this\s+(?:assistant|agent|app|application))\b",
    r"\bhow\s+(?:were|are)\s+you\s+(?:made|built|created|developed|designed|trained)\b",
    r"\bhow\s+you\s+(?:(?:were|are)\s+)?(?:made|built|created|developed|designed)\b",
    r"\bhow\s+(?:did|do)\s+you\s+(?:make|build|create|develop)\s+(?:yourself|you)\b",
    r"\b(?:what|which)\s+(?:ai\s+)?model\s+(?:(?:are|is)\s+(?:you|this)\s+(?:using|powered)|powers\s+you)\b",
    r"\b(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|what\s+do\s+you\s+know|how\s+do\s+you\s+work)\b",
    r"\b(?:are\s+you\s+(?:chatgpt|openai|an?\s+ai|a\s+language\s+model)|"
    r"(?:your|this\s+assistant'?s?)\s+(?:creator|maker|developer|identity|design|architecture))\b",
    r"\bwho\s+(?:is|was)\s+your\s+(?:creator|maker|developer)\b",
    r"\b(?:do|can)\s+you\s+(?:remember|see|read|access)\b",
    r"\b(?:who|what)\s+(?:is|was)\s+this\s+(?:assistant|agent)\b",
)
_PINN_USAGE_QUESTION = re.compile(
    r"\b(?:do|does|did)\s+(?:you|we|this(?:\s+run|\s+model)?|the\s+run|the\s+model)\s+"
    r"(?:use|apply|include|train(?:ed)?\s+with)\b.{0,100}\b(?:pinn|physics[- ]informed)\b|"
    r"\b(?:is|are|was|were|has|have)\b.{0,80}\b(?:pinn|physics[- ]informed)\b.{0,60}"
    r"\b(?:used|enabled|included|applied|active)\b.{0,40}\b(?:run|training|model)\b|"
    r"^\s*(?:is|are|was|were|has|have)\b.{0,60}\b(?:pinn|physics[- ]informed)\b.{0,40}"
    r"\b(?:on|off|used|enabled|included|applied|active)\b|"
    r"\b(?:pinn|physics[- ]informed)\b.{0,50}\b(?:in|used in|enabled in)\b.{0,30}"
    r"\b(?:this|the|current)?\s*(?:run|training|model)\b",
    re.I,
)
_GENERIC_NEXT_STEP_QUESTION = re.compile(
    r"^\s*(?:what\s+(?:should|could|can|do)\s+i\s+(?:try|do|test|work\s+on)\s+next|"
    r"what\s+next|what(?:'s|\s+is)\s+the\s+next\s+step|what\s+do\s+you\s+recommend)\s*[?.!]*$",
    re.I,
)
_NEXT_STEP_GOAL = re.compile(
    r"\b(?:accuracy|prediction|predictive|generaliz\w*|overfit\w*|error|mse|rmse|"
    r"physics|pinn|equation|deploy\w*|controller|control|mpc|cbf|hardware|"
    r"validat\w*|test\s+data|new\s+experiment|data\s+collection|improv\w*|"
    r"optimis\w*|optimiz\w*|reduce|lower|increase)\b", re.I)


def is_identity_question(question: str) -> bool:
    """Identify questions about this assistant so chat routes them to the LLM."""
    return any(re.search(pattern, question or "", flags=re.IGNORECASE)
               for pattern in _IDENTITY_PATTERNS)


def is_pinn_usage_question(question: str) -> bool:
    """Catch direct PINN status questions before they reach equation preparation."""
    return bool(_PINN_USAGE_QUESTION.search(question or ""))


def pinn_usage_answer(chat: dict) -> str:
    """Answer PINN usage from the saved run configuration or current setup."""
    run_dir = chat.get("run_dir")
    manifest = None
    if run_dir:
        path = Path(run_dir) / "run_manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                manifest = value
        except (OSError, ValueError, TypeError):
            manifest = None
    if manifest is not None and type(manifest.get("use_pinn")) is bool:
        architecture = str(manifest.get("architecture") or chat.get("settings", {}).get("architecture") or "model")
        if manifest["use_pinn"]:
            return (f"Yes. This saved {architecture} run used PINN. The run manifest confirms that "
                    "physics-informed training was enabled, so the validated equation contributed "
                    "a physics loss during training.")
        return (f"No. This saved {architecture} run used data-driven training. Its run manifest "
                "records PINN as off, so no equation-based physics loss was used.")

    settings = chat.get("settings") or {}
    if run_dir:
        if settings.get("use_pinn") is True:
            return ("PINN is selected in the conversation settings, but this run's manifest does not "
                    "record whether it was used. I can't confirm that it trained with PINN.")
        return ("I can't confirm PINN usage because this run's saved manifest has no PINN setting.")

    try:
        from . import pinn_maker
    except ImportError:
        import pinn_maker
    if settings.get("use_pinn") and pinn_maker.is_equation_ready(chat):
        return ("PINN is enabled in the current setup and its equation is validated, but no training "
                "run has started yet.")
    source = chat.get("pinn_source") or {}
    if source:
        return ("No run has used PINN yet. The source file is saved, but its equation has not passed "
                "validation, so PINN is not active.")
    return ("No run has used PINN yet, and PINN is currently off in the setup.")


def _assistant_identity(client) -> dict:
    """Combine reviewed app identity from YAML with non-secret runtime model info."""
    identity = dict(IDENTITY) if isinstance(IDENTITY, dict) else {}
    settings = getattr(client, "settings", None)
    identity["active_api_provider"] = getattr(settings, "provider", None) or "not exposed"
    identity["active_api_model"] = getattr(settings, "model", None) or "not exposed"
    return identity


def _conversation_history(messages: list[dict]) -> tuple[list[dict], bool]:
    """Keep a useful, bounded transcript with the newest context taking priority."""
    relevant = [m for m in messages if m.get("role") in ("user", "assistant")
                and str(m.get("content", "")).strip()]
    candidates = relevant[-24:]
    selected, used = [], 0
    clipped = len(relevant) > len(candidates)
    for message in reversed(candidates):
        content = str(message.get("content", "")).strip()
        if len(content) > 3200:
            content = content[:2300] + "\n[... middle of long message omitted ...]\n" + content[-800:]
            clipped = True
        remaining = 28000 - used
        if remaining <= 0:
            clipped = True
            break
        if len(content) > remaining:
            content = content[-remaining:]
            clipped = True
        selected.append({"role": message["role"], "content": content})
        used += len(content)
    selected.reverse()
    return selected, clipped



def _parse(raw: str, allowed: set[str]) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    reply = Answer.model_validate_json(value).model_dump()
    unknown = set(reply["sources"]) - allowed
    if unknown:
        raise ValueError("The answer referenced evidence that was not supplied.")
    cited = set(re.findall(r"\[([A-Z][A-Za-z0-9_]+)\]", reply["answer"]))
    if cited - allowed:
        raise ValueError("The answer contains an unknown source citation.")
    if reply["status"] == "clarification" and "?" not in reply["answer"]:
        raise ValueError("A clarification response must ask one clear client question.")
    # Force evidence-aware replies to identify at least one actual source.
    if allowed and reply["status"] == "answer" and not reply["sources"]:
        raise ValueError("The answer did not identify supporting evidence.")
    return reply


def _next_step_clarification(chat: dict, question: str) -> dict | None:
    """Ask for the missing goal instead of substituting a generic run recap."""
    if not _GENERIC_NEXT_STEP_QUESTION.fullmatch(question or ""):
        return None
    messages = chat.get("messages", [])
    result_positions = [index for index, message in enumerate(messages) if message.get("kind") == "result"]
    recent = messages[result_positions[-1] + 1:] if result_positions else messages[-12:]
    for message in recent:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content and content.casefold() != question.strip().casefold() and _NEXT_STEP_GOAL.search(content):
            return None
    return {
        "status": "clarification",
        "answer": (
            "I can help choose a useful next step, but it depends on what you want to achieve. "
            "Would you like to improve prediction on unseen measurements, check whether the learned "
            "dynamics match the physical system, or figure out what validation is needed before a "
            "controller test? The reported score and inference time alone do not establish controller "
            "readiness. Choose one, or tell me a different goal, and I’ll tailor the next step to it."
        ),
        "sources": [],
        "uncertainty": "The run results do not identify which next-step goal matters most to the client.",
    }


def _run_sources(run_dir: str, question: str) -> list[dict]:
    evidence = collect_evidence(run_dir, question, max_chars=85000)
    sources = evidence.sources
    terms = set(re.findall(r"[a-z]{4,}", question.lower()))
    def relevance(source):
        body = source.content.lower()
        return sum(body.count(term) for term in terms)
    fixed = [s for s in sources if s.id in ("M1", "V1", "R1", "S1")]
    dynamic = [s for s in sources if s.kind in ("cycle", "agent_turn", "console", "activity")]
    references = [s for s in sources if s.kind in ("template", "implementation", "knowledge")]
    ordered = fixed + sorted(dynamic, key=relevance, reverse=True) + sorted(references, key=relevance, reverse=True)
    selected, seen, remaining = [], set(), 61000
    comparison = analysis.compare_run_states(run_dir)
    if comparison.get("rows"):
        ranked = [row for row in comparison["rows"] if row["normalized_rmse"] is not None]
        selected.append({"id": "V2", "title": "Computed comparison of held-out state rollout errors",
                         "kind": "calculation", "provenance": "Calculated from verification_summary.json",
                         "content": json.dumps({"states_sorted_by_normalized_rmse": comparison["rows"],
                                                "best_by_normalized_rmse": ranked[0]["state"] if ranked else None,
                                                "worst_by_normalized_rmse": ranked[-1]["state"] if ranked else None,
                                                "aligned_samples": comparison.get("aligned_samples"),
                                                "metric": "RMSE / held-out state standard deviation; lower is better"})})
    for source in ordered:
        if source.id in seen or remaining < 500:
            continue
        seen.add(source.id)
        limit = min(len(source.content), 7500 if source.kind == "agent_turn" else 11000, remaining)
        if limit < 30:
            continue
        selected.append({"id": source.id, "title": source.title, "kind": source.kind,
                         "provenance": source.provenance, "content": source.content[:limit]})
        remaining -= limit
    return selected


def _dataset_sources(dataset: dict) -> list[dict]:
    profile = analysis.ensure_profile(dataset)
    return [{"id": "D1", "title": "Measured uploaded dataset profile", "kind": "measurement",
             "provenance": "Computed from the uploaded dataset, before training",
             "content": json.dumps({"file": dataset["name"], "states": dataset["states"],
                                    "inputs": dataset["actions"], "derivatives": dataset["derivatives"],
                                    "issues": dataset["issues"], "profile": profile}, ensure_ascii=False)}]


def _client_journey(chat: dict) -> dict:
    """Summarize the client's current, visible path without turning it into a script."""
    dataset = chat.get("dataset") or {}
    setup = chat.get("setup") or {}
    flow = chat.get("run_setup_flow") or {}
    if not dataset:
        data_status = "not_attached"
    elif dataset.get("column_review_pending"):
        data_status = "confirming_column_roles"
    elif dataset.get("issues"):
        data_status = "needs_data_checks"
    else:
        data_status = "ready"

    recommendation = {}
    for key in ("stage", "recommended_architecture", "architecture_reason", "recommended_effort",
                "effort_reason", "recommended_cycles", "confidence", "selected_architecture"):
        if key in flow:
            recommendation[key] = flow[key]

    return {
        "data_status": data_status,
        "sample_count": dataset.get("rows"),
        "columns": dataset.get("columns", []),
        "column_role_suggestions": dataset.get("column_review") if dataset.get("column_review_pending") else None,
        "data_checks": dataset.get("issues", []),
        "open_data_decision": setup.get("stage"),
        "run_setup": recommendation or None,
        "run_attached": bool(chat.get("run_dir")),
        "measurement_paths": (["paste a table into chat", "attach CSV or Excel"] if not dataset else []),
    }


def _check_state_ranking(reply: dict, comparison: dict, question: str = "") -> str:
    """Check only the ranking claims requested or actually made in the answer."""
    ranked = sorted(
        (r for r in comparison.get("rows", []) if r["normalized_rmse"] is not None),
        key=lambda row: row["normalized_rmse"],
    )
    if len(ranked) < 2:
        return ""
    answer = reply["answer"].lower().replace("**", "")
    state_token = r"(s_[a-z0-9_]+)"
    best_words = r"(?:best|lowest|smallest|most accurate|predicts? better|performs? better)"
    worst_words = r"(?:worst|highest|largest|hardest|least accurate|predicts? worse|performs? worse)"

    def claim(words):
        # Check each sentence/clause independently. A broad state-to-claim
        # regex can accidentally bind "worst" to the state named in the
        # preceding "best" clause (for example, "Best: s_b; worst: s_a").
        for clause in re.split(r"[;.!?\n]+", answer):
            terms = list(re.finditer(rf"\b{words}\b", clause))
            states = list(re.finditer(state_token, clause))
            if terms and states:
                term = terms[0]
                nearest = min(states, key=lambda state: min(
                    abs(state.start() - term.end()), abs(term.start() - state.end())))
                return nearest.group(1)
        return None

    best, worst = claim(best_words), claim(worst_words)
    expected_best, expected_worst = ranked[0]["state"].lower(), ranked[-1]["state"].lower()
    q = question.lower()
    asks_best = bool(re.search(best_words, q))
    asks_worst = bool(re.search(worst_words, q))
    mismatches = []
    if (asks_best and not best) or (best and best != expected_best):
        mismatches.append(f"best is {expected_best}")
    if (asks_worst and not worst) or (worst and worst != expected_worst):
        mismatches.append(f"worst is {expected_worst}")
    if mismatches:
        return ("State ranking mismatch. From V2, " + " and ".join(mismatches) +
                " by normalized held-out RMSE. Correct only the ranking claim(s) requested or stated; "
                "do not infer a cause from this ranking alone.")
    return ""


def _direct_best_state_answer(question: str, comparison: dict) -> dict | None:
    """Answer a plain best-state lookup directly from the saved held-out metrics."""
    q = question.lower()
    asks_best = bool(re.search(r"\b(?:best|lowest|smallest|most accurate)\b", q))
    asks_other = bool(re.search(r"\b(?:worst|highest|largest|why|explain|reason|cause|versus|vs\.?|compare|comparison)\b", q))
    if not asks_best or asks_other:
        return None
    ranked = sorted(
        (r for r in comparison.get("rows", []) if r["normalized_rmse"] is not None),
        key=lambda row: row["normalized_rmse"],
    )
    if not ranked:
        return None
    best = ranked[0]
    value = 100 * best["normalized_rmse"]
    samples = comparison.get("aligned_samples")
    sample_text = f" across {samples:,} aligned held-out samples" if isinstance(samples, int) else ""
    return {
        "status": "answer",
        "answer": (f"**{best['state']}** was predicted best by normalized held-out RMSE "
                   f"({value:.2f}% of that state's held-out variation; lower is better){sample_text}. "
                   "This ranks prediction error; it does not explain why that state was easier to predict."),
        "sources": ["V2"],
        "uncertainty": "The ranking uses normalized RMSE from the saved held-out comparison.",
        "evidence": [{"id": "V2", "title": "Computed comparison of held-out state rollout errors",
                      "kind": "calculation"}],
        "model": "Computed from saved run evidence",
    }


def answer_question(chat: dict, question: str, *, client=None, progress=None) -> dict:
    if not question.strip() or len(question) > 6000:
        raise ValueError("Ask a question between 1 and 6000 characters.")
    clarification = _next_step_clarification(chat, question)
    if clarification is not None:
        return {**clarification, "evidence": [], "model": "LabCD clarification"}
    if chat.get("run_dir") and analysis.state_comparison_question(question):
        comparison = analysis.compare_run_states(chat["run_dir"])
        direct = _direct_best_state_answer(question, comparison)
        if direct:
            return direct
    if progress:
        progress("Reading measurements and saved run evidence")
    llm = client or DiagnosticClient(DiagnosticSettings.defaults())
    identity = _assistant_identity(llm)
    dataset = chat.get("dataset") or {}
    sources = _dataset_sources(dataset) if dataset.get("path") else []
    if chat.get("run_dir"):
        sources += _run_sources(chat["run_dir"], question)
    if is_identity_question(question):
        sources.insert(0, {"id": "A1", "title": "LabCD assistant identity and implementation",
                           "kind": "identity", "provenance": "Application-owned identity metadata",
                           "content": json.dumps(identity, ensure_ascii=False)})
    allowed = {source["id"] for source in sources}
    history, history_truncated = _conversation_history(chat.get("messages", []))
    request = {"question": question, "setup_stage": (chat.get("setup") or {}).get("stage"),
               "client_journey": _client_journey(chat),
               "current_settings": {k: chat["settings"].get(k) for k in
                                    ("architecture", "run_mode", "max_cycles", "use_pinn",
                                     "pinn_loss_weight", "multi_trajectory", "manual_split_times",
                                     "auto_detect_angles", "angle_indices")},
               "assistant_identity": identity, "previous_messages": history,
               "conversation_history_truncated": history_truncated, "sources": sources}
    payload = redact(json.dumps(request, ensure_ascii=False))
    if progress:
        progress("Reasoning about your question")
    draft = _parse(llm.complete(SYSTEM, payload), allowed)
    if progress:
        progress("Checking the answer against evidence")
    try:
        reviewed = _parse(llm.complete(REVIEW, payload + "\nDRAFT ANSWER:\n" + json.dumps(draft, ensure_ascii=False)), allowed)
    except ValueError:
        if draft["status"] != "clarification":
            raise
        reviewed = draft
    if draft["status"] == "clarification" and reviewed["status"] != "clarification":
        reviewed = draft
    if chat.get("run_dir") and analysis.state_comparison_question(question):
        comparison = analysis.compare_run_states(chat["run_dir"])
        problem = _check_state_ranking(reviewed, comparison, question)
        if problem:
            if progress:
                progress("Correcting a mismatch with the measured ranking")
            reviewed = _parse(llm.complete(REVIEW, payload + "\nLOCAL CHECK FAILED:\n" + problem +
                                           "\nDRAFT ANSWER:\n" + json.dumps(reviewed, ensure_ascii=False)), allowed)
            if _check_state_ranking(reviewed, comparison, question):
                raise ValueError("The AI answer could not be reconciled with the measured state ranking.")
    return {**reviewed, "evidence": [{"id": s["id"], "title": s["title"], "kind": s["kind"]}
                                    for s in sources if s["id"] in reviewed["sources"]],
            "model": llm.settings.model if hasattr(llm, "settings") else "configured model"}


class ConversationJob:
    def __init__(self, chat: dict, question: str, *, purpose="question"):
        self.snapshot = copy.deepcopy(chat)
        self.question, self.purpose = question, purpose
        self.answer, self.error, self.error_detail = None, None, None
        self.phase = "Preparing evidence"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            self.answer = answer_question(self.snapshot, self.question,
                                          progress=lambda label: setattr(self, "phase", label))
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:1000]
            if "unsupported_country_region_territory" in str(exc):
                self.error = "The configured AI service denied this request from the current region. I can still show the measured data and saved results here."
            elif getattr(exc, "status_code", None) == 401:
                self.error = "The configured AI service rejected its credentials. Check the current API configuration."
            elif "could not be reconciled with the measured state ranking" in str(exc):
                self.error = "The AI answer disagreed with this run's measured state ranking after a correction attempt. The held-out comparison is shown below; please retry for interpretation."
            else:
                self.error = "I couldn't get a grounded answer from the configured AI service. Please retry your question."
