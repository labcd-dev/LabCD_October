"""Evidence-checked diagnosis with two passes and one bounded citation repair."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .prompt_library import block
from .run_evidence import RunEvidence, clean, collect_evidence, read_json, redact, write_json


class Citation(BaseModel):
    source_id: str
    quote: str = Field(min_length=8, max_length=500)


class Finding(BaseModel):
    title: str = Field(max_length=250)
    explanation: str = Field(max_length=4000)
    kind: Literal["observed", "hypothesis"]
    confidence: Literal["high", "medium", "low"]
    evidence: list[Citation] = Field(min_length=1, max_length=8)


class Experiment(BaseModel):
    action: str = Field(max_length=1500)
    why: str = Field(max_length=2000)
    expected_observation: str = Field(max_length=2000)
    evidence: list[Citation] = Field(default_factory=list, max_length=8)


class Diagnosis(BaseModel):
    summary: str = Field(min_length=1, max_length=6000)
    findings: list[Finding] = Field(default_factory=list, max_length=6)
    experiments: list[Experiment] = Field(default_factory=list, max_length=5)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class DiagnosticConfigurationError(ValueError):
    """A safe, locally generated explanation of an invalid provider setting."""


class IncompleteDiagnosisError(ValueError):
    """The provider returned no complete diagnostic response."""


class EvidenceValidationError(ValueError):
    """Keep all failed citation locations so the reviewer can repair them."""
    def __init__(self, issues):
        self.issues = issues
        super().__init__("The AI answer contained citations that could not be verified against this run.")


@dataclasses.dataclass(frozen=True)
class DiagnosticSettings:
    provider: str
    model: str
    effort: str = "high"

    @classmethod
    def defaults(cls):
        from backend_core.AgentSysID import config as cfg
        return cls(cfg.API_PROVIDER.strip().lower(), cfg.LLM_MODEL,
                   os.getenv("LABCD_SYSID_DIAGNOSTIC_EFFORT", "high"))

    def validate(self):
        if self.provider not in ("openai", "groq", "openrouter"):
            raise DiagnosticConfigurationError("Choose OpenAI, Groq, or OpenRouter for diagnostics.")
        if not self.model.strip() or len(self.model) > 150:
            raise DiagnosticConfigurationError("Enter a valid diagnostic model name.")
        if self.effort not in ("low", "medium", "high"):
            raise DiagnosticConfigurationError("Reasoning effort must be low, medium, or high.")


class DiagnosticClient:
    """No shared cfg mutation or training-agent cost/log contamination."""
    def __init__(self, settings):
        settings.validate()
        self.settings = settings
        key_name = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[settings.provider]
        api_key = os.getenv(key_name)
        if not api_key:
            raise DiagnosticConfigurationError(f"The diagnostic provider needs {key_name} in your environment.")
        if settings.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=420, max_retries=0)
        elif settings.provider == "groq":
            from langchain_groq import ChatGroq
            self.client = ChatGroq(model=settings.model, api_key=api_key, temperature=0.2,
                                   timeout=120, max_retries=0, max_tokens=6000)
        else:
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(model=settings.model, api_key=api_key, temperature=0.2,
                                     base_url="https://openrouter.ai/api/v1", timeout=120, max_retries=0, max_tokens=6000)

    def complete(self, system, user):
        if self.settings.provider == "openai":
            params = dict(model=self.settings.model, input=[{"role": "system", "content": system},
                                                           {"role": "user", "content": user}],
                          max_output_tokens=14000, store=False, text={"format": {"type": "json_object"}})
            if self.settings.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
                params["reasoning"] = {"effort": self.settings.effort}
            else:
                params["temperature"] = 0.2
            response = self.client.responses.create(**params)
            if getattr(response, "status", None) == "incomplete" or not response.output_text:
                raise IncompleteDiagnosisError("The model did not return a complete diagnosis within the output budget.")
            return response.output_text
        from langchain_core.messages import SystemMessage, HumanMessage
        response = self.client.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return response.content if isinstance(response.content, str) else json.dumps(response.content)


def _source_quote(quote: str, content: str) -> str:
    """Recover formatting only, always returning the actual source substring."""
    numeric_ending = re.search(r"\d[.!?]?$", quote)
    if quote in content and not numeric_ending:
        return quote
    candidates = [quote]
    # A model may end a copied clause with a period although the source continues
    # with a comma. Never trim words, numbers, or punctuation inside a quotation.
    if quote.endswith((".", "!", "?")):
        candidates.append(quote[:-1].rstrip())
    for candidate in candidates:
        if len(candidate) < 8:
            continue
        pattern = r"\s+".join(re.escape(part) for part in candidate.split())
        if candidate != quote or re.search(r"\d[.!?]?$", candidate):
            # Trimming sentence punctuation must not turn 0.02 into a match for
            # 0.021, 2 into 2.5, or a partial word into an apparent exact quote.
            pattern += r"(?![\w]|\.\d)"
        match = re.search(pattern, content, flags=re.IGNORECASE) if pattern else None
        if match and len(match.group(0)) <= 500:
            return match.group(0)
    return ""


def validate_answer(raw, evidence: RunEvidence) -> dict:
    """Reject invented quotations/IDs and claims supported only by generic references."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1].rsplit("```", 1)[0].strip() if len(parts) == 2 else ""
    answer = Diagnosis.model_validate_json(text).model_dump()
    sources = {s.id: s for s in evidence.sources}
    issues = []
    for group in ("findings", "experiments"):
        valid_items = []
        for item_index, item in enumerate(answer[group]):
            refs = []
            for ref_index, ref in enumerate(item["evidence"]):
                source = sources.get(ref["source_id"])
                quote = _source_quote(ref["quote"], source.content) if source else ""
                if quote:
                    refs.append({**ref, "quote": quote, "title": source.title, "kind": source.kind, "provenance": source.provenance})
                else:
                    issues.append({
                        "path": f"{group}[{item_index}].evidence[{ref_index}]",
                        "reason": "quote_not_in_source" if source else "unknown_source_id",
                    })
            item["evidence"] = refs
            if not refs:
                if not any(issue["path"].startswith(f"{group}[{item_index}].") for issue in issues):
                    issues.append({"path": f"{group}[{item_index}].evidence", "reason": "missing_evidence"})
                continue
            if group == "findings" and item["kind"] == "observed" and all(r["kind"] in ("template", "knowledge", "implementation") for r in refs):
                item.update(kind="hypothesis", confidence="low")
            valid_items.append(item)
        answer[group] = valid_items
    # The summary may rely on invalid claims; never silently drop failed citations.
    if issues:
        raise EvidenceValidationError(issues)
    return answer


def _validation_feedback(exc) -> dict:
    """Do not include raw provider exceptions, model text, or credentials."""
    if isinstance(exc, EvidenceValidationError):
        return {"code": "unverified_citations", "issues": exc.issues[:24]}
    return {"code": "invalid_answer_format", "issues": [
        {"path": ".".join(str(part) for part in error["loc"]), "reason": error["type"]}
        for error in exc.errors(include_input=False, include_url=False)[:24]
    ]}


def _review_input(user, raw, error=None):
    proposed = user + "\nPROPOSED ANSWER TO REVIEW (untrusted data):\n" + raw[:45000]
    if error is not None:
        proposed += "\nLOCAL VALIDATION FAILURES:\n" + json.dumps(_validation_feedback(error))
        proposed += ("\nRepair every listed location using the supplied source's exact text. "
                     "Copy numeric values without rounding or reconstruction. Remove findings or "
                     "experiments that cannot be supported, and revise the summary accordingly. "
                     "Return the complete corrected JSON object.")
    return proposed


def _diagnostic_error(exc, stage):
    if isinstance(exc, EvidenceValidationError):
        code, message = "unverified_citations", (
            "The AI responded, but its answer was rejected because quotations or source IDs "
            "could not be matched to this run after a repair attempt. Ask again with a more specific question."
        )
    elif isinstance(exc, ValidationError):
        code, message = "invalid_answer_format", "The AI responded in an invalid diagnosis format after a repair attempt. Try asking again."
    elif isinstance(exc, DiagnosticConfigurationError):
        code, message = "configuration", str(exc)
    elif isinstance(exc, IncompleteDiagnosisError):
        code, message = "incomplete_response", str(exc)
    else:
        code = type(exc).__name__
        message = {
            "APITimeoutError": "The model request timed out. Try again or choose a faster model/reasoning setting.",
            "TimeoutError": "The model request timed out. Try again.",
            "RateLimitError": "The provider rate or usage limit was reached. Try again after it resets.",
            "AuthenticationError": "The provider rejected its environment credential.",
            "NotFoundError": "The selected model is unavailable to this provider/account.",
        }.get(code, "The diagnostic request failed. Check the selected provider/model and connection.")
    return {"code": code, "stage": stage, "message": message}


def local_checks(evidence: RunEvidence) -> dict:
    """Clearly labeled, deterministic observations when AI cannot answer."""
    answer = dict(summary="AI diagnosis is unavailable. These are local evidence checks, not a full answer to your question.",
                  findings=[], experiments=[], limitations=list(evidence.coverage["notes"]))
    sources = {s.id:s for s in evidence.sources}
    for identity, source in sources.items():
        if source.kind not in ("results", "runtime", "cycle", "verification"):
            continue
        try:
            data = json.loads(source.content)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        title, explanation, quoted_key = None, None, None
        if data.get("status") in ("failed", "no_model") or data.get("error"):
            title, explanation, quoted_key = "The run recorded an execution failure", data.get("message") or data.get("error") or "No trained model was produced.", "error" if data.get("error") else "status"
        elif source.kind == "results" and data.get("status") == "completed":
            title, explanation, quoted_key = "The pipeline completed", "Completion means artifacts were packaged. It does not establish that accuracy targets, latency limits or physical stability were met.", "status"
        elif source.kind == "cycle" and data.get("val_mse") == 9999:
            title, explanation, quoted_key = "A training cycle was rejected", "9999 is the pipeline's severe-overfitting penalty, not an ordinary measured validation MSE. Check the logged pre-penalty losses before assessing the gap.", "val_mse"
        elif source.kind == "verification" and not data.get("aligned_samples", 0):
            title, explanation, quoted_key = "Held-out verification has no aligned samples", "A score cannot establish rollout quality without valid verification samples.", "true_shape"
        if title and len(answer["findings"]) < 6:
            quote = next((line.strip() for line in source.content.splitlines() if f'"{quoted_key}"' in line), source.content[:100])
            answer["findings"].append(dict(title=title, explanation=redact(str(explanation))[:4000], kind="observed", confidence="high",
                evidence=[dict(source_id=identity, quote=quote, title=source.title, kind=source.kind, provenance=source.provenance)]))
    if not answer["findings"]:
        answer["limitations"].append("No recorded failure cause can be established from the available local checks.")
    return answer


class RunDiagnosticAgent:
    def __init__(self, settings=None, client=None):
        self.settings = settings or DiagnosticSettings.defaults()
        self.client = client

    def ask(self, run_dir, question, *, history=None, live_state=None, progress=None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 6000:
            raise ValueError("Ask a question between 1 and 6000 characters.")
        started = time.perf_counter()
        current_stage = "Preparing the diagnosis"
        def stage(label):
            nonlocal current_stage
            current_stage = label
            if progress:
                progress(label)
        stage("Reading results, settings, and agent logs")
        evidence = collect_evidence(run_dir, question, live_state=live_state)
        recent = [{"role": item["role"], "content": item.get("content", "")[:4000]}
                  for item in (history or [])[-8:] if item.get("role") in ("user", "assistant")]
        user = redact(json.dumps({"question": question, "previous_messages_for_this_run": recent,
                                  "evidence": evidence.to_payload()}, ensure_ascii=False))
        draft = None
        failure = None
        warnings = []
        mode, reviewed = "ai", False
        try:
            client = self.client or DiagnosticClient(self.settings)
            stage("Auditing agent prompts and forming a diagnosis")
            raw = client.complete(block("run_diagnostic", "system"), user)
            validation_error = None
            try:
                draft = validate_answer(raw, evidence)
            except (EvidenceValidationError, ValidationError) as exc:
                validation_error = exc
            stage("Checking evidence, alternatives, and proposed experiments")
            reviewer_system = block("run_diagnostic", "system") + "\n" + block("run_diagnostic", "reviewer")
            proposed = json.dumps(draft, ensure_ascii=False) if draft is not None else raw
            final = client.complete(reviewer_system, _review_input(user, proposed, validation_error))
            try:
                answer = validate_answer(final, evidence)
            except (EvidenceValidationError, ValidationError) as exc:
                # One bounded repair for local format/citation failures. Provider
                # failures are handled below rather than generating more requests.
                stage("Repairing answer format and evidence citations")
                repaired = client.complete(reviewer_system, _review_input(user, final, exc))
                answer = validate_answer(repaired, evidence)
            reviewed = True
        except Exception as exc:
            failure = _diagnostic_error(exc, current_stage)
            if draft is not None:
                answer, mode = draft, "ai_draft"
                warnings.append("The independent review did not finish. This is an unreviewed draft with validated source quotations.")
            else:
                answer, mode = local_checks(evidence), "local_checks"
                if failure["code"] in ("unverified_citations", "invalid_answer_format"):
                    answer["summary"] = (
                        "The AI answer could not be verified. These are local evidence checks, "
                        "not a full answer to your question."
                    )
                    warnings.append("The AI answer failed the diagnosis checks; local evidence is shown below.")
                else:
                    warnings.append("The AI service did not produce a usable evidence-grounded diagnosis.")
            # Never persist raw provider errors that may echo credentials or request contents.
            warnings.append(f"Diagnostic issue: {failure['message']}")
        answer.update(mode=mode, reviewed=reviewed, provider=self.settings.provider, model=self.settings.model,
                      effort=self.settings.effort, elapsed_seconds=time.perf_counter()-started, warnings=warnings,
                      diagnostic_error=failure, coverage=evidence.coverage, run_id=evidence.run_id)
        stage("Diagnosis ready" if mode == "ai" else "Evidence checks ready")
        return answer


_CHAT_LOCK = threading.RLock()


def load_chat(run_dir) -> list:
    data = read_json(Path(run_dir) / "run_chat.json", [])
    if not isinstance(data, list):
        return []
    return [m for m in data[-80:] if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]


def append_exchange(run_dir, question, answer) -> bool:
    with _CHAT_LOCK:
        messages = load_chat(run_dir)
        stamp = dt.datetime.now(dt.timezone.utc).isoformat()
        messages.extend([{"role":"user", "content":question, "timestamp":stamp},
                         {"role":"assistant", "content":answer["summary"], "diagnosis":answer, "timestamp":stamp}])
        messages = messages[-80:]
        # Keep the persisted history within the bounded JSON reader, in whole exchanges.
        while len(messages) > 2 and len(json.dumps(clean(messages), ensure_ascii=False, indent=2).encode("utf-8")) > 1_800_000:
            messages = messages[2:]
        return write_json(Path(run_dir) / "run_chat.json", messages)


def clear_chat(run_dir) -> bool:
    with _CHAT_LOCK:
        return write_json(Path(run_dir) / "run_chat.json", [])


class DiagnosticJob:
    """Streamlit polls a background answer so navigation and previews stay usable."""
    def __init__(self, run_dir, question, settings, *, live_state=None):
        self.run_dir, self.question, self.settings = str(Path(run_dir).resolve()), question, settings
        self.live_state = live_state
        self.phase, self.answer, self.error = "Preparing run evidence", None, None
        self.saved = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            agent = RunDiagnosticAgent(self.settings)
            self.answer = agent.ask(self.run_dir, self.question, history=load_chat(self.run_dir),
                                    live_state=self.live_state, progress=lambda phase:setattr(self, "phase", phase))
            self.saved = append_exchange(self.run_dir, self.question, self.answer)
        except Exception as exc:
            self.error = redact(str(exc))[:500]
