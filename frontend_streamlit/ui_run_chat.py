"""Run-scoped diagnostic chat. Background work never calls Streamlit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from backend_core.AgentSysID.agents.run_diagnostic import (
    DiagnosticJob, DiagnosticSettings, clear_chat, load_chat,
)
from backend_core.AgentSysID.agents.run_evidence import collect_evidence

try:
    from . import ui_history as hist
    from . import ui_results_workspace as workspace
except ImportError:
    import ui_history as hist
    import ui_results_workspace as workspace


SUGGESTIONS = (
    ("Explain this result", "Did this model actually fail? Explain the result and the most important remaining risks."),
    ("Audit the agents", "Audit the other agents' prompts and decisions. Identify contradictions or possible problems and what the run evidence actually proves."),
    ("Choose my next test", "What should I test next to improve identification and held-out rollout quality? Prioritize controlled experiments supported by this run."),
)


def _text(value):
    st.markdown("\n".join(workspace.literal(line) for line in str(value).splitlines()))


def _citations(refs):
    for ref in refs:
        st.caption(f"{ref.get('source_id', '')} · {ref.get('title', 'Run evidence')} · {ref.get('provenance', 'Saved with this run')}")
        st.text(ref.get("quote", ""))


def render_answer(answer):
    """Show conclusions and verifiable evidence, never private reasoning."""
    mode = answer.get("mode")
    issue = answer.get("diagnostic_error") or {}
    rejected = issue.get("code") in ("unverified_citations", "invalid_answer_format")
    label = "Evidence reviewed" if answer.get("reviewed") else (
        "Unreviewed AI draft" if mode == "ai_draft" else (
            "Local checks · AI answer rejected" if rejected else "Local checks · AI unavailable"
        )
    )
    st.caption(f"{label} · {answer.get('model', 'Diagnostic model')} · {workspace.duration(answer.get('elapsed_seconds'))}")
    _text(answer.get("summary", "No answer was returned."))
    for warning in answer.get("warnings", []):
        st.warning(warning)
    findings = answer.get("findings", [])
    if findings:
        with st.expander(f"Findings & evidence · {len(findings)}", icon=":material/fact_check:"):
            for i, finding in enumerate(findings):
                if i:
                    st.divider()
                st.markdown(f"**{i+1}. {workspace.literal(finding['title'])}**")
                st.caption(f"{'Recorded observation' if finding.get('kind') == 'observed' else 'Hypothesis'} · {finding.get('confidence', 'low').title()} confidence")
                _text(finding.get("explanation", ""))
                _citations(finding.get("evidence", []))
    experiments = answer.get("experiments", [])
    if experiments:
        with st.expander(f"Suggested next tests · {len(experiments)}", icon=":material/science:"):
            for i, experiment in enumerate(experiments):
                if i:
                    st.divider()
                st.markdown(f"**{i+1}. {workspace.literal(experiment['action'])}**")
                _text(experiment.get("why", ""))
                st.caption("What to look for")
                _text(experiment.get("expected_observation", ""))
                _citations(experiment.get("evidence", []))
    with st.expander("Scope & uncertainty", icon=":material/info:"):
        coverage = answer.get("coverage", {})
        st.caption(f"{coverage.get('sources_used', 0)} evidence excerpts · {coverage.get('agent_turns', 0)} recorded agent turns · {coverage.get('templates', 0)} prompt templates")
        for limitation in dict.fromkeys(answer.get("limitations", []) + coverage.get("notes", [])):
            _text(f"• {limitation}")
        if coverage.get("missing"):
            st.caption("Not recorded: " + ", ".join(coverage["missing"]))


def _settings():
    # Refresh on each rerun, including sessions that still hold the old model.
    settings = DiagnosticSettings.defaults()
    st.session_state["diagnostic_settings"] = settings
    with st.popover("Diagnostic model", icon=":material/psychology:"):
        st.markdown("**Uses your current API settings**")
        st.text(f"Provider: {settings.provider}\nModel: {settings.model}")
        st.caption("Follows the provider and model configured for the other agents. Each answer still uses an investigator pass and an evidence review with your existing credentials.")
    return settings


def _inventory(run_dir):
    # File metadata invalidates the cache when training writes new evidence.
    base = Path(run_dir)
    files = list(base.glob("*.json")) + list(base.glob("*.log")) + list((base / "Agents_log").glob("*"))
    stamp = tuple(sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file()))
    return _cached_inventory(run_dir, stamp)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _cached_inventory(run_dir, stamp):
    evidence = collect_evidence(run_dir)
    return evidence.coverage, [{"Source": s.id, "Evidence": s.title, "Provenance": s.provenance} for s in evidence.sources]


def _submit(run_dir, question, settings, live_state=None):
    jobs = st.session_state.setdefault("diagnostic_jobs", {})
    if jobs.get(run_dir) and jobs[run_dir].running:
        return
    settings.validate()
    job = DiagnosticJob(run_dir, question, settings, live_state=live_state)
    jobs[run_dir] = job
    job.start()


def _conversation(run_dir, job):
    messages = load_chat(run_dir)
    with st.container(border=True, height=520 if messages or job else "content", key="diagnostic_conversation"):
        if not messages and not job:
            st.markdown("#### Understand what happened")
            st.caption("Ask about losses, rollout drift, data quality, agent decisions, or the next experiment. Answers stay with this run.")
        for message in messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and isinstance(message.get("diagnosis"), dict):
                    render_answer(message["diagnosis"])
                else:
                    _text(message["content"])
        if job and job.running:
            with st.chat_message("user"):
                _text(job.question)
            with st.chat_message("assistant"):
                with st.status(job.phase, expanded=True, state="running"):
                    st.caption("Reading the selected run, auditing agent decisions, then challenging the diagnosis against its evidence.")
                    st.caption("You can use the results preview or move to another tab while this runs.")
        elif job and job.error:
            with st.chat_message("user"):
                _text(job.question)
            st.error(f"Diagnosis could not start: {job.error}")
        elif job and job.answer and not job.saved:
            with st.chat_message("user"):
                _text(job.question)
            with st.chat_message("assistant"):
                render_answer(job.answer)
            st.warning("This answer could not be saved to the run folder. Export it before leaving this session.")


@st.fragment(run_every=1.0)
def _poll_conversation(run_dir, job):
    if not job.running:
        st.rerun()
    _conversation(run_dir, job)


def render_run_chat(output_dir, *, preferred=None, runner=None):
    st.markdown("### Ask about this run")
    st.caption("A system identification diagnosis grounded in results, recorded agent prompts, and logs.")
    entries = hist.load_history(output_dir, limit=None)
    if not entries:
        st.info("Start a run first. Completed and interrupted runs can both be investigated here.")
        return
    entry = workspace.select_run("Run to investigate", entries, key="diagnostic_run", preferred=preferred)
    identity = workspace.run_id(entry)
    if st.session_state.get("diagnostic_preview_linked_run") != identity:
        st.session_state["preview_run"] = identity
        st.session_state["preview_run_choice"] = identity
        st.session_state["diagnostic_preview_linked_run"] = identity
    short_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
    jobs = st.session_state.setdefault("diagnostic_jobs", {})
    job = jobs.get(identity)
    busy = bool(job and job.running)
    current_run = bool(runner and runner.run_dir and str(Path(runner.run_dir).resolve()) == identity)
    training = bool(current_run and runner.running)
    settings_column, actions_column = st.columns([1, 1], gap="small")
    with settings_column:
        settings = _settings()
    with actions_column:
        with st.popover("Conversation", icon=":material/chat:"):
            export = load_chat(identity)
            if job and job.answer and not job.saved:
                export += [{"role":"user", "content":job.question}, {"role":"assistant", "content":job.answer['summary'], "diagnosis":job.answer}]
            st.download_button("Export conversation", data=json.dumps(export, ensure_ascii=False, indent=2),
                               file_name=f"{Path(identity).name}_diagnosis.json", mime="application/json", width="stretch",
                               key=f"diagnostic_export_{short_id}")
            if st.button("Clear this conversation", disabled=busy, width="stretch", key=f"diagnostic_clear_{short_id}"):
                if clear_chat(identity):
                    jobs.pop(identity, None)
                    st.rerun()
                else:
                    st.error("The conversation file could not be cleared.")
    st.caption(f"{settings.provider.title()} · {settings.model} · Two-pass evidence review")
    st.caption("Deep analysis can take several minutes. The workspace stays available while it runs.")
    with st.expander("What the agent can read", icon=":material/folder_open:"):
        coverage, source_rows = _inventory(identity)
        st.caption(f"{coverage['agent_turns']} recorded agent turns · {coverage['templates']} prompt templates · {coverage['sources_used']} evidence excerpts")
        st.dataframe(source_rows, hide_index=True, width="stretch")
        for note in coverage["notes"]:
            _text(note)
        if coverage["missing"]:
            st.caption("Not recorded: " + ", ".join(coverage["missing"]))
    if training:
        st.info("This run is still training. Ask when it finishes so the diagnosis has complete evidence, or choose an earlier run.")
    if busy:
        _poll_conversation(identity, job)
    else:
        _conversation(identity, job)
    if not load_chat(identity) and not busy:
        columns = st.columns(3, gap="small")
        for column, (label, question) in zip(columns, SUGGESTIONS):
            if column.button(label, width="stretch", disabled=training, key=f"diagnostic_suggestion_{short_id}_{label}"):
                _submit(identity, question, settings)
                st.rerun()
    question = st.chat_input("Ask why, check an agent decision, or plan your next experiment…", max_chars=6000,
                             disabled=busy or training, key=f"diagnostic_question_{short_id}")
    if question:
        live_state = {"stage": runner.last_stage, "error": runner.error} if current_run else None
        _submit(identity, question, settings, live_state)
        st.rerun()
