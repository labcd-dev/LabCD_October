"""Conversation-first interface for the existing AgentSysID pipeline."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import html
import importlib.util
import json
import math
import mimetypes
import re
import time
import uuid
import zipfile
from pathlib import Path

import streamlit as st
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticJob, DiagnosticSettings, load_chat
from backend_core.AgentSysID.agents.run_evidence import _read as read_evidence_file, read_json, write_json
from backend_core.AgentSysID import config as cfg

try:
    from . import ui_brand, conversation_core as core, conversation_analysis as analysis, conversation_agent as agent
    from . import ui_data_analysis as data_ui, ui_history as hist, ui_activity as activity
    from . import excel_maker, pinn_maker, run_setup_agent, ui_initializer_review as init_ui
    from . import ui_run_setup_summary as setup_summary
    from . import ui_results_workspace as workspace, ui_run_chat as diagnosis_ui, ui_onboarding
    from . import ui_model_sandbox, model_sheet, ui_model_sheet
except ImportError:
    import ui_brand
    import conversation_core as core
    import conversation_analysis as analysis
    import conversation_agent as agent
    import ui_data_analysis as data_ui
    import ui_history as hist
    import ui_activity as activity
    import excel_maker
    import pinn_maker
    import run_setup_agent
    import ui_initializer_review as init_ui
    import ui_run_setup_summary as setup_summary
    import ui_results_workspace as workspace
    import ui_run_chat as diagnosis_ui
    import ui_onboarding
    import ui_model_sandbox
    import model_sheet
    import ui_model_sheet


AUTO_DATA_REVIEW_PROMPT = (
    "Give a brief, data-based upload review in at most three short bullets: "
    "sample count and sampling interval, the main state/input observation, and any important data check. "
    "Keep it under 70 words. Do not recap architecture, run mode, or cycle settings, ask the client to approve them, "
    "or suggest starting training. The interface will ask any needed trajectory or angle question separately. "
    "Make clear these are measurements, not trained-model results."
)
QUESTION_SKIP_DELAY_SECONDS = 30
STARTING_SETUP_AUTO_START_SECONDS = 60
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "stopped", "no_model"}
SYSTEM_WELCOME_COPIES = (
    ("Map your system over time",
     "Describe what you want to study. LabCD can help map how it changes over time into a model grounded in your measurements."),
    ("Start with the system",
     "Tell me what the system does and what changes over time; we can shape a model around the measurements you have."),
    ("What would you like to model?",
     "Share what you want to study. I can help organize the system’s behavior over time into a model you can use."),
    ("Explore your system",
     "Describe what you can observe or control. LabCD can help map how your system behaves over time into a model."),
)
SYSTEM_GREETING_COPIES = (
    "Hi, I’m LabCD. What system are you exploring? Tell me what it does and how it changes over time; I can help map that into a model.",
    "Hello! What device, process, or system would you like to understand? A few words about how it behaves over time are a good place to start.",
    "Hey! Tell me what you want to model and what changes in the system. We can organize its behavior over time into a model together.",
    "Hi there. What system are you studying? Describe how it behaves over time, and I’ll help connect that to a model.",
)


def _literal(text):
    st.markdown("\n".join(workspace.literal(line) for line in str(text).splitlines()))


def _count(number, noun):
    return f"{number} {noun if number == 1 else noun + 's'}"


def _conversation_copy(chat, choices, counter_key=None):
    """Choose stable copy per conversation, rotating when the client repeats a greeting."""
    identity = str(chat.get("id") or "")
    seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)
    count = 0
    if counter_key:
        count = max(0, int(chat.get(counter_key, 0) or 0))
        chat[counter_key] = count + 1
    return choices[(seed + count) % len(choices)]


def _queue_suggested_prompt(key):
    prompt = st.session_state.get(key)
    if isinstance(prompt, str) and prompt.strip():
        st.session_state["conversation_pending"] = prompt
    st.session_state[key] = None


def _select(chat):
    st.session_state["conversation"] = chat
    st.query_params["chat"] = chat["id"]
    st.session_state["conversation_files"] = False
    st.session_state["conversation_selected_file"] = None
    st.session_state["conversation_file_panel_expanded"] = False


def _new():
    _select(core.new_chat())


def _history_chat(chat_id):
    """Load the current saved conversation so sidebar actions never mutate a stale row."""
    active = st.session_state.get("conversation")
    if isinstance(active, dict) and active.get("id") == chat_id:
        return active
    return core.read_chat(chat_id)


def _toggle_history_pin(chat_id):
    saved = _history_chat(chat_id)
    if not saved:
        return
    saved["pinned"] = not bool(saved.get("pinned"))
    core.save_chat(saved)


def _set_history_archived(chat_id, archived):
    saved = _history_chat(chat_id)
    if not saved:
        return
    saved["archived"] = bool(archived)
    core.save_chat(saved)
    active = st.session_state.get("conversation")
    if archived and isinstance(active, dict) and active.get("id") == chat_id:
        _new()


def _set_history_project(chat_id, project_id):
    saved = _history_chat(chat_id)
    if not saved:
        return
    if project_id is not None and core.read_project(project_id) is None:
        return
    saved["project_id"] = project_id
    core.save_chat(saved)


def _remove_history_chat(chat_id):
    saved = _history_chat(chat_id) or core.read_chat(chat_id)
    if not saved:
        st.session_state.pop("conversation_remove_pending", None)
        st.rerun()
    was_active = (isinstance(st.session_state.get("conversation"), dict)
                  and st.session_state["conversation"].get("id") == chat_id)
    if not core.delete_chat(chat_id):
        st.error("The conversation could not be removed.")
        return
    st.session_state.pop("conversation_remove_pending", None)
    if was_active:
        _new()
    st.rerun()


def _entry(chat):
    if not chat.get("run_dir"):
        return None
    entry = read_json(Path(chat["run_dir"]) / "run_manifest.json") or {}
    entry.update(run_dir=chat["run_dir"], name=Path(chat["run_dir"]).name,
                 complete=entry.get("status") == "completed")
    return entry


def _result_summary(entry):
    if entry.get("status") != "completed":
        return "This run did not complete. " + (entry.get("message") or "We can investigate the recorded activity and saved results.")
    mse, latency = workspace.number(entry, "best_mse"), workspace.number(entry, "latency_ms")
    error = f"validation MSE {mse:.4g}" if mse is not None else "validation error not recorded"
    timing = f" and measured inference latency {latency:.3f} ms" if latency is not None else ""
    return (f"The run finished after {entry.get('cycles_run', '—')} cycles, with {error}{timing}. "
            "I’ve added the learning curve, per-state scores, and saved diagnostic plots below. You can inspect each candidate or ask a follow-up about the results.")


def _open_run(entry):
    identity = hashlib.md5(str(Path(entry["run_dir"]).resolve()).encode()).hexdigest()
    chat = core.read_chat(identity)
    if chat is None:
        chat = core.new_chat()
        chat.update(id=identity, title=hist.display_name(entry), run_dir=str(Path(entry["run_dir"]).resolve()))
        core.add_message(chat, "assistant", _result_summary(entry), kind="result", run_dir=chat["run_dir"])
        for message in load_chat(chat["run_dir"]):
            core.add_message(chat, message["role"], message.get("content", ""),
                             **({"diagnosis": message["diagnosis"]} if message.get("diagnosis") else {}))
    _select(chat)


def _sidebar_history_row(saved, *, archived, project_names):
    active_chat = st.session_state.get("conversation")
    selected = isinstance(active_chat, dict) and saved["id"] == active_chat.get("id")
    row_state = "selected-" if selected else ""
    with st.container(key=f"history-row-{row_state}{saved['id']}"):
        title_column, menu_column = st.columns([1, 0.14], gap="small", vertical_alignment="center")
        with title_column:
            icon = ":material/push_pin:" if saved.get("pinned") else ":material/chat_bubble_outline:"
            st.button(saved["title"], icon=icon, width="stretch",
                      key=f"conversation_open_{saved['id']}", on_click=_select, args=(saved,), help=saved["title"])
        with menu_column:
            with st.popover("Conversation actions", icon=":material/more_vert:",
                            key=f"history-menu-{saved['id']}"):
                if st.button(
                    "Unpin conversation" if saved.get("pinned") else "Pin conversation",
                    icon=":material/push_pin:", key=f"history_pin_{saved['id']}",
                    width="stretch",
                ):
                    _toggle_history_pin(saved["id"])
                    st.rerun()
                if st.button(
                    "Restore conversation" if archived else "Archive conversation",
                    icon=":material/unarchive:" if archived else ":material/archive:",
                    key=f"history_archive_{saved['id']}", width="stretch",
                ):
                    _set_history_archived(saved["id"], not archived)
                    st.rerun()
                choices = [None, *project_names.keys()]
                current_project = saved.get("project_id") if saved.get("project_id") in project_names else None
                selected_project = st.selectbox(
                    "Move to project", choices,
                    index=choices.index(current_project),
                    format_func=lambda project_id: project_names.get(project_id, "No project"),
                    key=f"history_project_choice_{saved['id']}",
                )
                if st.button(
                    "Move conversation", key=f"history_move_project_{saved['id']}",
                    width="stretch", disabled=selected_project == current_project,
                ):
                    _set_history_project(saved["id"], selected_project)
                    st.rerun()
                st.divider()
                if st.button(
                    "Remove permanently", icon=":material/delete_outline:",
                    key=f"history_remove_{saved['id']}", width="stretch",
                ):
                    st.session_state["conversation_remove_pending"] = saved["id"]
                    st.rerun()
        if st.session_state.get("conversation_remove_pending") == saved["id"]:
            st.warning(f"Remove **{saved['title']}** and its messages from history?")
            st.caption("Saved run folders, reports, and uploaded files will stay on disk.")
            keep, remove = st.columns(2)
            with keep:
                if st.button("Keep conversation", key=f"history_keep_{saved['id']}", width="stretch"):
                    st.session_state.pop("conversation_remove_pending", None)
                    st.rerun()
            with remove:
                if st.button("Remove conversation", key=f"history_remove_confirm_{saved['id']}",
                             type="primary", width="stretch"):
                    _remove_history_chat(saved["id"])


def _sidebar(chat):
    with st.sidebar:
        st.markdown(
            f'<div class="conversation-brand">{ui_brand.LOGO_MARKUP}</div>',
            unsafe_allow_html=True,
        )
        st.button("New conversation", icon=":material/edit_square:", width="stretch", on_click=_new, key="conversation_new")
        with st.popover("New project", icon=":material/create_new_folder:", width="stretch"):
            project_title = st.text_input("Project name", placeholder="e.g. Pendulum study", key="project_new_title")
            if st.button("Create project", icon=":material/add:", width="stretch", key="project_create"):
                try:
                    project = core.create_project(project_title)
                    project_chat = core.new_chat(project["id"])
                    core.save_chat(project_chat)
                    _select(project_chat)
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
        query = st.text_input("Search conversations", placeholder="Search conversations", label_visibility="collapsed", key="conversation_search")
        with st.popover("Library", icon=":material/inventory_2:", width="stretch"):
            archived = st.toggle("Show archived conversations", key="conversation_archived")
            st.caption("Conversations and run results are saved in this project.")
        chats = core.list_chats()
        projects = core.list_projects()
        project_names = {project["id"]: project["title"] for project in projects}
        query_text = query.strip().casefold()
        visible = [
            saved for saved in chats
            if bool(saved.get("archived")) == archived
            and (query_text in saved["title"].casefold()
                 or query_text in project_names.get(saved.get("project_id"), "").casefold())
        ]
        grouped = {project_id: [saved for saved in visible if saved.get("project_id") == project_id]
                   for project_id in project_names}
        grouped_ids = set(project_names)
        unassigned = [saved for saved in visible if saved.get("project_id") not in grouped_ids]
        st.caption("PROJECTS")
        shown_projects = 0
        for project in projects:
            project_chats = grouped[project["id"]]
            if query_text and not project_chats and query_text not in project["title"].casefold():
                continue
            shown_projects += 1
            with st.expander(f"{project['title']} · {len(project_chats)}", key=f"project-group-{project['id']}", expanded=(
                chat.get("project_id") == project["id"] or bool(query_text)
            )):
                if st.button("New chat in project", icon=":material/add:", width="stretch",
                             key=f"project_new_chat_{project['id']}"):
                    project_chat = core.new_chat(project["id"])
                    core.save_chat(project_chat)
                    _select(project_chat)
                    st.rerun()
                for saved in project_chats[:60]:
                    _sidebar_history_row(saved, archived=archived, project_names=project_names)
        if not shown_projects:
            st.caption("Create a project to group related conversations.")
        st.caption("CONVERSATIONS")
        if not unassigned:
            st.caption("No conversations here yet." if not query_text else "No matching conversations.")
        for saved in unassigned[:60]:
            _sidebar_history_row(saved, archived=archived, project_names=project_names)
        linked = {str(Path(c["run_dir"]).resolve()) for c in chats if c.get("run_dir")}
        runs = [r for r in hist.load_history(core.OUTPUT_DIR, limit=None)
                if str(Path(r["run_dir"]).resolve()) not in linked and not r.get("archived")
                and query.lower() in hist.display_name(r).lower()]
        if runs:
            with st.expander("Previous runs", expanded=not visible):
                for entry in runs[:40]:
                    stamp = entry.get("started")
                    suffix = stamp.strftime("%d %b · %H:%M") if stamp else "Saved run"
                    st.button(f"{hist.display_name(entry)} · {suffix}", width="stretch",
                              key=f"legacy_run_{workspace.run_id(entry)}", on_click=_open_run, args=(entry,))
        st.markdown('<div class="sidebar-footnote">LabCD · System identification<br>From measurements to understanding.</div>', unsafe_allow_html=True)


def _setup_animation(step: str):
    title = "Choose a model that fits your measurements" if step == "model" else "Choose how widely to search"
    detail = ("The agent has read the data profile. Review its first model choice, then make it yours."
              if step == "model" else
              "The model choice is set. Pick a search budget that fits the time and detail you want.")
    step_number = 1 if step == "model" else 2
    model_class = "active" if step == "model" else "complete"
    search_class = "next" if step == "model" else "active"
    st.markdown(f'''
<style>
.setup-flow-hero{{position:relative;overflow:hidden;border:1px solid #35464b;border-radius:18px;
  isolation:isolate;padding:20px 22px 10px;background:radial-gradient(ellipse at 82% 0%,#46d8c11c,transparent 40%),
  radial-gradient(ellipse at 12% 100%,#4b5cb41c,transparent 38%),
  linear-gradient(125deg,#10191c 0%,#172225 48%,#171b24 100%);margin:4px 0 14px;
  box-shadow:inset 0 1px #ffffff08,0 12px 34px #0002}}
.setup-flow-hero:after{{content:"";position:absolute;inset:auto -8% -90px 30%;height:130px;z-index:-1;
  background:#42cdb41a;filter:blur(42px);border-radius:50%}}
.setup-flow-top{{display:flex;align-items:center;justify-content:space-between;gap:16px}}
.setup-flow-kicker{{font:600 10px/1.5 ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:#7dd3c7}}
.setup-flow-count{{font:11px/1.5 ui-monospace,monospace;color:#a1bbb6;border:1px solid #3b5653;
  border-radius:999px;padding:4px 9px;background:#10201f88}}
.setup-flow-hero h3{{font:500 clamp(20px,2.5vw,29px)/1.15 Arial,sans-serif;letter-spacing:-.04em;color:#f2f7f6;margin:7px 0 5px}}
.setup-flow-hero p{{max-width:690px;color:#b9c8c7;font-size:13px;line-height:1.6;margin:0}}
.setup-flow-steps{{display:flex;align-items:center;gap:8px;margin:12px 0 0;color:#819691;font:11px ui-monospace,monospace}}
.setup-flow-step{{display:flex;align-items:center;gap:6px;padding:5px 9px;border:1px solid #31403f;border-radius:999px;background:#121b1d99}}
.setup-flow-step.complete{{color:#8de2cf;border-color:#346458;background:#18302c99}}
.setup-flow-step.active{{color:#e4fff9;border-color:#53a595;background:#1b3b3699;box-shadow:0 0 18px #48bea21a}}
.setup-flow-step-mark{{width:15px;height:15px;display:grid;place-items:center;border-radius:50%;background:#253332;color:#a4bcb6;font:10px Arial,sans-serif}}
.setup-flow-step.complete .setup-flow-step-mark{{background:#245348;color:#b7f7e7}}
.setup-flow-svg{{width:100%;height:auto;max-height:132px;display:block;margin:8px 0 0;overflow:visible}}
.setup-flow-panel{{fill:#111d20;stroke:#32494a;stroke-width:1.2}}
.setup-flow-grid{{fill:none;stroke:#34504e;stroke-width:.8;opacity:.54}}
.setup-flow-heading{{fill:#c2d8d3;font:600 10px ui-monospace,monospace;letter-spacing:1.1px}}
.setup-flow-sub{{fill:#779590;font:10px Arial,sans-serif}}
.setup-flow-link{{fill:none;stroke:#4d827b;stroke-width:1.2;opacity:.52}}
.setup-flow-link.animate{{stroke-dasharray:4 7;animation:setup-dash 2.8s linear infinite}}
.setup-flow-signal{{fill:none;stroke:#69d7c2;stroke-width:2.6;stroke-linecap:round;filter:drop-shadow(0 0 4px #4bd2b488);
  stroke-dasharray:170;stroke-dashoffset:0;animation:setup-signal 4.2s ease-in-out infinite}}
.setup-flow-forecast{{fill:none;stroke:#9aa7ff;stroke-width:2.6;stroke-linecap:round;filter:drop-shadow(0 0 4px #858dff88);
  stroke-dasharray:4 6;animation:setup-forecast 3.4s linear infinite}}
.setup-flow-neuron{{fill:#182b2e;stroke:#4e8981;stroke-width:1.3}}
.setup-flow-neuron.hot{{fill:#75e1cc;stroke:#b0ffee;filter:drop-shadow(0 0 6px #67dbc3aa);animation:setup-pulse 2.4s ease-in-out infinite}}
.setup-flow-neuron.mid{{animation:setup-pulse 2.8s ease-in-out infinite;animation-delay:.6s}}
.setup-flow-packet{{fill:#b3fff0;filter:drop-shadow(0 0 5px #80f0d6);animation:setup-travel 2.8s ease-in-out infinite}}
.setup-flow-packet.out{{animation-name:setup-travel-out;animation-duration:2.6s}}
.setup-flow-packet.p2{{animation-delay:.9s}}.setup-flow-packet.p3{{animation-delay:1.8s}}
.setup-flow-packet.out.p2{{animation-delay:.86s}}.setup-flow-packet.out.p3{{animation-delay:1.72s}}
@keyframes setup-dash{{to{{stroke-dashoffset:-32}}}}
@keyframes setup-signal{{0%,100%{{stroke-dashoffset:30;opacity:.7}}50%{{stroke-dashoffset:-20;opacity:1}}}}
@keyframes setup-forecast{{to{{stroke-dashoffset:-20}}}}
@keyframes setup-pulse{{0%,100%{{r:4;opacity:.72}}50%{{r:6.5;opacity:1}}}}
@keyframes setup-travel{{0%{{transform:translateX(0);opacity:0}}12%{{opacity:1}}84%{{opacity:1}}100%{{transform:translateX(100px);opacity:0}}}}
@keyframes setup-travel-out{{0%{{transform:translateX(0);opacity:0}}12%{{opacity:1}}84%{{opacity:1}}100%{{transform:translateX(100px);opacity:0}}}}
@media(max-width:640px){{.setup-flow-hero{{padding:16px 14px 8px}}.setup-flow-svg{{max-height:98px}}.setup-flow-steps{{gap:4px;font-size:10px}}.setup-flow-step{{padding:4px 6px}}}}
@media(prefers-reduced-motion:reduce){{.setup-flow-link.animate,.setup-flow-signal,.setup-flow-forecast,.setup-flow-neuron,.setup-flow-packet{{animation:none!important}}}}
</style>
<section class="setup-flow-hero" aria-label="System identification setup">
  <div class="setup-flow-top"><div class="setup-flow-kicker">RUN SETUP · SYSTEM IDENTIFICATION</div><div class="setup-flow-count">0{step_number} / 02</div></div>
  <h3>{title}</h3><p>{detail}</p>
  <div class="setup-flow-steps" aria-label="Setup progress">
    <span class="setup-flow-step complete"><span class="setup-flow-step-mark">✓</span> Data</span>
    <span class="setup-flow-step {model_class}"><span class="setup-flow-step-mark">{'✓' if step == 'effort' else '1'}</span> Model</span>
    <span class="setup-flow-step {search_class}"><span class="setup-flow-step-mark">2</span> Search</span>
  </div>
  <svg class="setup-flow-svg" viewBox="0 0 900 132" role="img" aria-label="Measured signals pass through a neural model to produce a state prediction">
    <rect class="setup-flow-panel" x="12" y="16" rx="13" width="201" height="101"/>
    <text class="setup-flow-heading" x="28" y="37">MEASURED SIGNAL</text><text class="setup-flow-sub" x="28" y="52">state + input over time</text>
    <path class="setup-flow-grid" d="M28 68 H197 M28 86 H197 M70 62 V103 M112 62 V103 M154 62 V103"/>
    <path class="setup-flow-signal" d="M29 88 C43 88 45 70 59 70 S76 100 90 96 S105 68 119 73 S136 94 150 83 S171 64 196 69"/>
    <path class="setup-flow-link animate" d="M213 66 H313 M587 66 H687"/>
    <rect class="setup-flow-panel" x="313" y="8" rx="18" width="274" height="117"/>
    <text class="setup-flow-heading" x="450" y="29" text-anchor="middle">LEARN THE DYNAMICS</text>
    <path class="setup-flow-link" d="M361 52 L425 48 M361 52 L425 72 M361 52 L425 96 M361 75 L425 48 M361 75 L425 72 M361 75 L425 96 M361 98 L425 48 M361 98 L425 72 M361 98 L425 96 M425 48 L489 59 M425 72 L489 59 M425 96 L489 59 M425 48 L489 83 M425 72 L489 83 M425 96 L489 83"/>
    <circle class="setup-flow-neuron" cx="361" cy="52" r="4"/><circle class="setup-flow-neuron mid" cx="361" cy="75" r="4"/><circle class="setup-flow-neuron" cx="361" cy="98" r="4"/>
    <circle class="setup-flow-neuron mid" cx="425" cy="48" r="4"/><circle class="setup-flow-neuron hot" cx="425" cy="72" r="5"/><circle class="setup-flow-neuron mid" cx="425" cy="96" r="4"/>
    <circle class="setup-flow-neuron" cx="489" cy="59" r="4"/><circle class="setup-flow-neuron hot" cx="489" cy="83" r="5"/>
    <text class="setup-flow-sub" x="450" y="114" text-anchor="middle">fit · test · refine</text>
    <rect class="setup-flow-panel" x="687" y="16" rx="13" width="201" height="101"/>
    <text class="setup-flow-heading" x="703" y="37">NEXT-STEP PREDICTION</text><text class="setup-flow-sub" x="703" y="52">compare with measurement</text>
    <path class="setup-flow-grid" d="M703 68 H872 M703 86 H872 M745 62 V103 M787 62 V103 M829 62 V103"/>
    <path class="setup-flow-link" d="M704 94 C721 94 726 72 742 72 S762 94 779 91 S800 70 816 74 S837 88 853 75 S864 68 873 69"/>
    <path class="setup-flow-forecast" d="M704 99 C721 99 726 81 742 81 S762 87 779 84 S800 77 816 81 S837 82 853 79 S864 75 873 76"/>
    <circle class="setup-flow-packet" cx="216" cy="66" r="3.4"/><circle class="setup-flow-packet p2" cx="216" cy="66" r="3.4"/><circle class="setup-flow-packet p3" cx="216" cy="66" r="3.4"/>
    <circle class="setup-flow-packet out" cx="590" cy="66" r="3.4"/><circle class="setup-flow-packet out p2" cx="590" cy="66" r="3.4"/><circle class="setup-flow-packet out p3" cx="590" cy="66" r="3.4"/>
  </svg>
</section>
''', unsafe_allow_html=True)


def _model_illustration(architecture: str, *, selected: bool = False) -> str:
    """Small visual overview of what each starting architecture sees and predicts."""
    selected_class = "selected" if selected else "unselected"
    if architecture == "LSTM":
        return f'''
<svg class="architecture-diagram {selected_class}" viewBox="0 0 440 174" role="img" aria-label="An LSTM reads recent state and input samples in order, keeps a short memory, then predicts state derivatives">
  <text class="arch-label" x="93" y="27" text-anchor="middle">RECENT SAMPLES</text>
  <text class="arch-label" x="280" y="27" text-anchor="middle">SEQUENCE MEMORY</text>
  <text class="arch-label" x="390" y="27" text-anchor="middle">PREDICTION</text>
  <rect class="arch-sample" x="14" y="47" width="70" height="70" rx="10"/><rect class="arch-sample" x="94" y="47" width="70" height="70" rx="10"/><rect class="arch-sample current" x="174" y="47" width="70" height="70" rx="10"/>
  <text class="arch-main" x="49" y="78" text-anchor="middle">[x,u]</text><text class="arch-small" x="49" y="99" text-anchor="middle">t − 2</text>
  <text class="arch-main" x="129" y="78" text-anchor="middle">[x,u]</text><text class="arch-small" x="129" y="99" text-anchor="middle">t − 1</text>
  <text class="arch-main" x="209" y="78" text-anchor="middle">[x,u]</text><text class="arch-small" x="209" y="99" text-anchor="middle">now</text>
  <path class="arch-wire" d="M84 82H94M164 82H174M244 82H258M358 82H370"/>
  <path class="arch-arrow" d="M252 77l7 5-7 5M364 77l7 5-7 5"/>
  <rect class="arch-block" x="260" y="47" width="98" height="70" rx="12"/>
  <path class="arch-memory" d="M276 91c11-22 20 18 31-2s19 11 34-10"/>
  <text class="arch-main" x="309" y="69" text-anchor="middle">LSTM</text>
  <rect class="arch-output" x="372" y="57" width="58" height="50" rx="10"/>
  <text class="arch-main" x="401" y="79" text-anchor="middle">dx/dt</text><text class="arch-small" x="401" y="96" text-anchor="middle">derivative</text>
  <text class="arch-foot" x="129" y="143" text-anchor="middle">order matters</text><text class="arch-foot" x="309" y="143" text-anchor="middle">uses recent history</text>
</svg>'''
    return f'''
<svg class="architecture-diagram {selected_class}" viewBox="0 0 440 174" role="img" aria-label="An MLP maps the current state and input through connected layers to predict state derivatives">
  <text class="arch-label" x="60" y="27" text-anchor="middle">CURRENT DATA</text>
  <text class="arch-label" x="222" y="27" text-anchor="middle">CONNECTED LAYERS</text>
  <text class="arch-label" x="376" y="27" text-anchor="middle">PREDICTION</text>
  <rect class="arch-sample" x="14" y="53" width="94" height="60" rx="10"/>
  <text class="arch-main" x="61" y="79" text-anchor="middle">x(t), u(t)</text><text class="arch-small" x="61" y="97" text-anchor="middle">state + input</text>
  <path class="arch-wire faint" d="M108 64L164 56M108 64L164 84M108 64L164 112M108 83L164 56M108 83L164 84M108 83L164 112M108 102L164 56M108 102L164 84M108 102L164 112M180 56L225 68M180 84L225 68M180 112L225 68M180 56L225 99M180 84L225 99M180 112L225 99M239 68L283 84M239 99L283 84M283 84L324 84"/>
  <circle class="arch-node" cx="172" cy="56" r="6"/><circle class="arch-node" cx="172" cy="84" r="6"/><circle class="arch-node" cx="172" cy="112" r="6"/>
  <circle class="arch-node bright" cx="231" cy="68" r="6"/><circle class="arch-node" cx="231" cy="99" r="6"/><circle class="arch-node bright" cx="289" cy="84" r="6"/>
  <path class="arch-wire" d="M324 84H337"/><path class="arch-arrow" d="M331 79l7 5-7 5"/>
  <rect class="arch-output" x="340" y="57" width="86" height="54" rx="10"/>
  <text class="arch-main" x="383" y="81" text-anchor="middle">dx/dt</text><text class="arch-small" x="383" y="99" text-anchor="middle">derivative</text>
  <text class="arch-foot" x="222" y="143" text-anchor="middle">learns a direct mapping</text>
</svg>'''


def _physics_guidance_visual() -> str:
    """Animated sketch of measured data and an optional equation guiding training."""
    return '''
<svg class="pinn-guidance-diagram" viewBox="0 0 760 178" role="img" aria-label="Measured data and an optional validated physics equation both guide an LSTM or MLP training process">
  <rect class="pinn-panel" x="10" y="24" width="190" height="100" rx="13"/>
  <text class="pinn-heading" x="26" y="48">MEASURED DATA</text>
  <text class="pinn-sub" x="26" y="65">states + inputs over time</text>
  <path class="pinn-grid" d="M27 82H183M27 99H183M63 76V111M101 76V111M139 76V111"/>
  <path class="pinn-wave" d="M28 101C43 101 44 82 59 82S77 107 92 104S112 80 128 86S152 99 182 84"/>
  <path class="pinn-link" d="M200 75H282M452 75H545"/>
  <path class="pinn-arrow" d="M273 69l9 6-9 6M536 69l9 6-9 6"/>
  <rect class="pinn-panel model" x="282" y="24" width="170" height="100" rx="13"/>
  <text class="pinn-heading" x="367" y="51" text-anchor="middle">TRAIN THE MODEL</text>
  <circle class="pinn-neuron" cx="329" cy="76" r="6"/><circle class="pinn-neuron bright" cx="367" cy="76" r="7"/><circle class="pinn-neuron" cx="405" cy="76" r="6"/>
  <path class="pinn-grid" d="M335 76H360M374 76H399"/>
  <text class="pinn-sub" x="367" y="104" text-anchor="middle">LSTM or MLP</text>
  <rect class="pinn-panel output" x="545" y="24" width="205" height="100" rx="13"/>
  <text class="pinn-heading" x="561" y="48">PREDICTED STATES</text>
  <text class="pinn-sub" x="561" y="65">check against held-out data</text>
  <path class="pinn-grid" d="M562 84H733M562 102H733M605 78V111M648 78V111M691 78V111"/>
  <path class="pinn-wave" d="M563 104C580 101 581 84 598 84S622 103 638 99S657 81 676 86S705 97 732 81"/>
  <path class="pinn-equation-link" d="M367 148V125"/>
  <path class="pinn-arrow" d="M361 134l6-9 6 9"/>
  <rect class="pinn-equation" x="235" y="145" width="264" height="25" rx="12"/>
  <text class="pinn-equation-text" x="367" y="162" text-anchor="middle">OPTIONAL EQUATION · dx/dt = f(x,u)</text>
</svg>'''


def _run_setup_card(chat, message, busy):
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    if flow.get("dataset_sha256") != dataset.get("sha256"):
        st.caption("This setup recommendation belongs to an earlier dataset. I’ll prepare a fresh one for the current measurements.")
        return
    settings = core.RunSettings.model_validate(chat["settings"]).model_dump()
    setup_summary.render(
        chat, flow, settings, busy, _model_illustration,
        on_signals_saved=_finish_setup_signal_roles,
        on_interaction=lambda: _stop_starting_setup_countdown(chat, message.get("id")),
    )
    if (not busy and dataset.get("ready") and not chat.get("run_dir")
            and flow.get("stage") != "approved"):
        _render_starting_setup_countdown(chat, message.get("id"))


@st.dialog("Compare runs", width="large")
def _compare_dialog(chat):
    workspace.render_compare(str(core.OUTPUT_DIR), preferred=chat.get("run_dir"))


def _header(chat, busy, has_files=False):
    with st.container(key="conversation_header"):
        title, controls = st.columns([3, 4], vertical_alignment="center")
        title.markdown(
            f'<div class="conversation-heading"><span class="header-brand">{ui_brand.LOGO_MARKUP}</span>'
            f'<span class="conversation-heading-separator">/</span>'
            f'<span class="conversation-heading-title">{html.escape(chat["title"])}</span></div>',
            unsafe_allow_html=True,
        )
        with controls:
            with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
                is_open = st.session_state.get("conversation_files", False)
                if st.button("Files", icon=":material/dock_to_left:" if is_open else ":material/dock_to_right:",
                              key="conversation_files_toggle", disabled=not has_files,
                              type="primary" if is_open else "secondary"):
                    st.session_state["conversation_files"] = not st.session_state.get("conversation_files", False)
                    st.rerun()
                if st.button("Compare", icon=":material/compare_arrows:", key="conversation_compare"):
                    _compare_dialog(chat)
                with st.popover("Conversation options", icon=":material/more_horiz:"):
                    if st.button("Guided tutorial", icon=":material/school:", key="conversation_open_tutorial"):
                        st.session_state["labcd_tutorial_request"] = True
                        st.rerun()
                    title = st.text_input("Conversation title", value=chat["title"], key=f"title_{chat['id']}", max_chars=100)
                    if st.button("Rename", disabled=busy):
                        chat["title"] = title.strip() or "Untitled conversation"
                        core.save_chat(chat)
                        st.rerun()
                    if st.button("Unpin conversation" if chat.get("pinned") else "Pin conversation", disabled=busy):
                        chat["pinned"] = not chat.get("pinned", False)
                        core.save_chat(chat)
                        st.rerun()
                    if st.button("Restore conversation" if chat.get("archived") else "Archive conversation", disabled=busy):
                        chat["archived"] = not chat.get("archived", False)
                        core.save_chat(chat)
                        _new()
                        st.rerun()
                    st.download_button("Export conversation", json.dumps(chat, ensure_ascii=False, indent=2),
                                       file_name="labcd-conversation.json", mime="application/json")


def _plan(settings, chat, key, can_start=True):
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    if (flow.get("dataset_sha256") == dataset.get("sha256") and
            flow.get("stage") == "approved"):
        settings = chat["settings"]
    with st.container(border=True, key=f"plan_{key}"):
        physics_label = str(settings.get("physics_mode", "pinn")).upper() if settings.get("use_pinn") else ""
        model_label = settings["architecture"] + (f" + {physics_label}" if physics_label else "")
        mode_label = {"fast":"fast search", "regular":"regular search", "heavy":"heavy search", "expert":"regular search"}.get(settings["run_mode"], "regular search")
        st.markdown(f"**Run plan · {model_label}**")
        st.caption(f"{mode_label} · up to {settings['max_cycles']} search cycles · {settings['epochs']} starting epochs per cycle")
        if settings["architecture"] == "LSTM":
            st.caption(f"LSTM history · {settings['lstm_seq_length']} past time steps")
        with st.expander("Settings and assumptions"):
            st.json({name: value for name, value in settings.items() if name != "customer_description"})
        if can_start:
            setup_pending = bool(chat.get("setup") and chat["setup"].get("stage") != "complete")
            approval_pending = bool(
                chat.get("setup") and chat["setup"].get("stage") == "complete" and
                not (flow.get("dataset_sha256") == dataset.get("sha256") and flow.get("stage") == "approved"))
            if setup_pending:
                st.caption("Finish the data questions above before starting.")
            elif approval_pending:
                st.caption("Review the setup summary and edit any value with its pen before approving the run.")
            else:
                st.caption("Open the setup review, then approve it once to start training.")
            if st.button("Open setup review", type="primary", icon=":material/tune:", key=f"start_{key}",
                         disabled=not (chat.get("dataset") or {}).get("ready") or
                                  setup_pending or approval_pending):
                st.session_state["conversation_pending"] = "start"
                st.rerun()


def _result_tools(run_dir, key):
    entry = read_json(Path(run_dir) / "run_manifest.json") or {}
    entry["run_dir"] = run_dir
    with st.container(horizontal=True, gap="small"):
        if st.button("View files", icon=":material/folder_open:", key=f"files_{key}"):
            st.session_state["conversation_files"] = True
            st.rerun()
        if path := hist.existing_files(entry).get("Results ZIP"):
            st.download_button("Download results", Path(path).read_bytes(), file_name=Path(path).name,
                               key=f"zip_{key}", icon=":material/download:")


def _file_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"):
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".py", ".m"):
        return "code"
    if suffix in (".csv", ".tsv", ".xlsx", ".xls"):
        return "table"
    if suffix == ".zip":
        return "archive"
    if suffix in (".pth", ".pt", ".pkl", ".pickle", ".onnx"):
        return "binary"
    if suffix in (".txt", ".md", ".log", ".json", ".jsonl", ".yaml", ".yml", ".toml",
                  ".html", ".xml", ".ini", ".cfg"):
        return "text"
    return "file"


def _file_icon(kind):
    return {"upload": "table_view", "table": "table_view", "image": "image",
            "pdf": "picture_as_pdf", "code": "code", "archive": "folder_zip",
            "sandbox": "science", "binary": "memory", "text": "description",
            "file": "draft"}.get(kind, "draft")


def _artifact_label(path):
    name = Path(path).name
    stem = name.lower()
    for token, label in [("test_verification", "Held-out predictions"),
                         ("mse_convergence", "Validation MSE by cycle"),
                         ("rmse_convergence", "Validation RMSE by cycle"),
                         ("latency_evolution", "Inference latency"),
                         ("hyperparameters", "Training settings explored"),
                         ("layers_neurons", "Architecture search")]:
        if token in stem:
            return label
    if name == "verification_summary.json":
        return "Held-out state scores"
    if name == "run_manifest.json":
        return "Run summary"
    if name.endswith(".pth") or name.endswith(".onnx"):
        return "Model weights"
    if name.endswith(".pdf"):
        return "PDF report"
    if name.endswith(".zip"):
        return "Results ZIP"
    return name


def _generated_file_id(path):
    resolved = str(Path(path).resolve())
    return "generated:" + hashlib.sha1(resolved.encode()).hexdigest()[:16]


def _store_physics_equation(chat, artifact: dict) -> None:
    """Install a new equation and discard exact-state choices tied to an old one."""
    previous = chat.get("pinn_equation") or {}
    previous_id = previous.get("proposal_id") or previous.get("sha256")
    new_id = artifact.get("proposal_id") or artifact.get("sha256")
    if previous_id != new_id:
        settings = chat.setdefault("settings", {})
        settings["pcnn_hard_constraint_states"] = []
        chat.pop("pcnn_constraints_reviewed_for", None)
    chat["pinn_equation"] = artifact


def _file_items(chat):
    """List every uploaded attachment and readable artifact for this conversation."""
    items, seen = [], set()
    for message in chat.get("messages", []):
        attachment = message.get("attachment")
        if not isinstance(attachment, dict) or not attachment.get("path"):
            continue
        resolved_attachment = str(Path(attachment["path"]).resolve())
        if resolved_attachment in seen:
            continue
        identity = f"upload:{message.get('id') or attachment['path']}"
        seen.add(resolved_attachment)
        if attachment.get("kind") == "pinn_source":
            if pinn_maker.is_equation_ready(chat):
                status_label = ("validated · enabled in setup" if chat.get("settings", {}).get("use_pinn")
                                else "validated · optional in setup")
            else:
                status_label = "saved · not active until validated"
            items.append({"id": identity, "kind": "code",
                          "label": attachment.get("name") or Path(attachment["path"]).name,
                          "path": attachment["path"],
                          "subtitle": f"Physics source · original preserved · {status_label}",
                          "icon": _file_icon("code")})
            continue
        rows = attachment.get("rows")
        signal_count = len(attachment.get("states") or [])
        items.append({"id": identity, "kind": "upload", "label": attachment.get("name") or Path(attachment["path"]).name,
                      "path": attachment["path"], "dataset": attachment,
                      "subtitle": f"Dataset · {rows:,} rows · {signal_count} states" if isinstance(rows, int) else "Uploaded dataset",
                      "icon": _file_icon("upload")})
    dataset = chat.get("dataset")
    if isinstance(dataset, dict) and dataset.get("path") and str(Path(dataset["path"]).resolve()) not in seen:
        seen.add(str(Path(dataset["path"]).resolve()))
        items.append({"id": f"upload:current:{dataset.get('sha256', dataset['path'])}", "kind": "upload",
                      "label": dataset.get("name") or Path(dataset["path"]).name, "path": dataset["path"],
                      "dataset": dataset, "subtitle": f"Dataset · {dataset.get('rows', 0):,} rows · {len(dataset.get('states') or [])} states",
                      "icon": _file_icon("upload")})

    for message in chat.get("messages", []):
        artifact = message.get("generated_artifact")
        path = artifact.get("path") if isinstance(artifact, dict) else None
        if not path or not Path(path).is_file():
            continue
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        artifact_kind = artifact.get("kind", "table")
        file_kind = "code" if artifact_kind == "pinn_equation" else "table"
        artifact_mode = str(artifact.get("physics_mode") or "pinn").lower()
        pcnn_artifact = artifact_kind == "pinn_equation" and artifact_mode == "pcnn"
        if pcnn_artifact:
            equation_label = ("PCNN hard-constraint proposal" if artifact.get("origin") == "ai_proposal"
                              else "Validated PCNN hard constraints")
            item_count = len(artifact.get("equations") or [])
            item_unit = "rules"
        else:
            equation_label = "Physics model draft" if artifact.get("origin") == "ai_proposal" else "Validated equation"
            item_count = len(artifact.get("states") or [])
            item_unit = "states"
        subtitle = (f"{equation_label} · {item_count:,} {item_unit}" if artifact_kind == "pinn_equation"
                    else f"Excel workbook · {artifact.get('rows', 0):,} rows · {artifact.get('columns', 0)} columns")
        items.append({"id": _generated_file_id(resolved), "kind": file_kind,
                      "label": artifact.get("name") or Path(resolved).name, "path": resolved,
                      "file_name": Path(resolved).name,
                      "preview_header": 3, "subtitle": subtitle,
                      "icon": _file_icon(file_kind)})

    run_dir = chat.get("run_dir")
    if run_dir and Path(run_dir).is_dir():
        entry = _entry(chat) or {"run_dir": run_dir}
        entry["run_dir"] = run_dir
        run_files = hist.existing_files(entry)
        # Keep code and model weights inside the ZIP package instead of
        # presenting them as separate chat files.
        deliverables = (
            ("PDF report", run_files.get("PDF report"), "pdf"),
            ("Results ZIP", run_files.get("Results ZIP"), "archive"),
        )
        sandbox_item = ui_model_sandbox.file_item(chat)
        for label, path, kind in deliverables:
            if not path:
                continue
            resolved = str(Path(path).resolve())
            if resolved in seen or not Path(resolved).is_file():
                continue
            seen.add(resolved)
            size = Path(resolved).stat().st_size
            if size >= 1024 * 1024:
                size_label = f"{size / (1024 * 1024):.1f} MB"
            elif size >= 1024:
                size_label = f"{size / 1024:.0f} KB"
            else:
                size_label = f"{size} B"
            type_label = "PDF" if kind == "pdf" else "ZIP"
            items.append({"id": "artifact:" + hashlib.sha1(resolved.encode()).hexdigest()[:16],
                          "kind": kind, "label": label, "path": resolved,
                          "file_name": Path(resolved).name,
                          "subtitle": f"{type_label} · {size_label}",
                          "icon": _file_icon(kind)})
        if sandbox_item:
            items.append(sandbox_item)
    return items


def _open_file(item):
    st.session_state["conversation_selected_file"] = item["id"]
    st.session_state["conversation_files"] = True


def _is_zip_file(item):
    return item.get("kind") == "archive" or Path(item.get("path", "")).suffix.lower() == ".zip"


def _file_card(item, key):
    card_key = f"chat_file_card_{key}"
    help_text = ("Open the results archive in the right preview panel" if _is_zip_file(item)
                 else "Open this file in the right preview panel")
    with st.container(border=True, key=card_key):
        details_col, icon_col, action_col = st.columns(
            [1, 0.45, 1], gap="small", vertical_alignment="center",
        )
        icon_col.button(
            item["label"], icon=f":material/{item['icon']}:", type="tertiary", width="content",
            key=f"chat_file_icon_{key}", help=help_text, on_click=_open_file, args=(item,),
        )
        details_col.button(
            item["label"], type="tertiary", width="stretch", key=f"open_chat_file_{key}",
            help=help_text, on_click=_open_file, args=(item,),
        )
        details_col.caption(item["subtitle"])
        with action_col.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "Open", type="secondary", width="content", key=f"open_chat_file_action_{key}",
                help=help_text, on_click=_open_file, args=(item,),
            )


def _file_cards(items, scope):
    if not items:
        return
    with st.container(key=f"run_files_{scope}"):
        st.markdown("**Run artifacts**")
        for item in items:
            card_id = hashlib.sha1(item["id"].encode()).hexdigest()[:10]
            _file_card(item, f"{scope}_{card_id}")


def _show_download(path, key):
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError:
        st.info("This file is no longer available on disk.")
        return None
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    st.download_button("Download file", data, file_name=file_path.name, mime=mime,
                       key=f"download_preview_{key}", width="stretch")
    return data


def _preview_upload(item, key):
    dataset = item["dataset"]
    view = st.segmented_control("Dataset file view", ["Rows", "Signals", "Details"], default="Rows",
                                key=f"upload_preview_view_{key}", label_visibility="collapsed", width="stretch")
    if view == "Signals":
        states = dataset.get("states") or []
        if states:
            selected = st.selectbox("Measured state", states, key=f"upload_preview_state_{key}")
            data_ui._signal(dataset, selected, key, height=250)
        else:
            st.info("No measured state columns are available to plot.")
    elif view == "Details":
        st.metric("Samples", f"{dataset.get('rows', 0):,}")
        st.write("Columns: " + ", ".join(dataset.get("columns") or []))
        if dataset.get("sample_period") is not None:
            st.caption(f"Median sample interval: {dataset['sample_period']:.6g} s")
        for issue in dataset.get("issues") or []:
            st.warning(issue)
    else:
        try:
            frame = data_ui._preview(dataset["path"], dataset["sha256"], dataset["name"])
            st.caption(f"Preview · {len(frame):,} sampled rows")
            st.dataframe(frame.head(120), width="stretch", height=350, hide_index=True)
        except (KeyError, OSError, ValueError) as exc:
            st.warning(str(exc))


def _preview_file(item, key):
    path = Path(item["path"])
    try:
        if not path.is_file():
            st.info("This file is no longer available on disk.")
            return
        kind = item["kind"]
        if kind == "upload":
            _show_download(path, key)
            _preview_upload(item, key)
            return
        suffix = path.suffix.lower()
        if kind == "image":
            st.image(str(path), width="stretch")
            _show_download(path, key)
        elif kind == "pdf":
            data = _show_download(path, key)
            if data and importlib.util.find_spec("streamlit_pdf"):
                st.pdf(data, height=610, key=f"file_pdf_{key}")
            elif data:
                st.caption("PDF preview is unavailable in this environment. Download the report to open it.")
        elif kind == "code":
            data = _show_download(path, key)
            if data is not None:
                code = data[:300_000].decode("utf-8", errors="replace")
                language = {".m": "matlab", ".py": "python"}.get(suffix, "text")
                st.code(code, language=language, line_numbers=True)
                if len(data) > 300_000:
                    st.caption("Showing the first 300 KB of this source file.")
        elif kind == "table":
            _show_download(path, key)
            import pandas as pd
            if suffix in (".csv", ".tsv"):
                frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=200)
            else:
                frame = pd.read_excel(path, nrows=200, header=item.get("preview_header", 0))
            st.dataframe(frame, width="stretch", height=430, hide_index=True)
        elif kind == "archive":
            _show_download(path, key)
            with zipfile.ZipFile(path) as archive:
                members = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
            st.caption(f"Archive contents · {len(members)} files")
            st.dataframe({"File": members[:200]}, width="stretch", hide_index=True, height=360)
        elif kind == "binary":
            _show_download(path, key)
            st.info("This is a model or binary artifact. It is ready to download, but it is not rendered as text.")
        else:
            data = _show_download(path, key)
            if data is not None:
                text = data[:180_000].decode("utf-8", errors="replace")
                if suffix == ".json":
                    try:
                        st.json(json.loads(text))
                    except ValueError:
                        st.code(text, language="text")
                else:
                    st.code(text, language="markdown" if suffix == ".md" else "text")
                if len(data) > 180_000:
                    st.caption("Showing the first 180 KB of this text file.")
    except (OSError, ValueError, zipfile.BadZipFile, ImportError) as exc:
        st.warning(f"Couldn't preview this file: {exc}")


def _artifacts(chat, items=None):
    items = items if items is not None else _file_items(chat)
    expanded = bool(st.session_state.get("conversation_file_panel_expanded", False))
    panel_key = "conversation_file_panel_expanded" if expanded else "conversation_file_panel"
    # CSS anchors the keyed container to the viewport and lets its contents
    # scroll inside the fixed panel frame.
    with st.container(border=True, height="stretch", key=panel_key):
        selected_id = st.session_state.get("conversation_selected_file")
        selected = next((item for item in items if item["id"] == selected_id), None)
        title_col, controls_col = st.columns([1.05, 1.6], vertical_alignment="center")
        if selected:
            if title_col.button("All files", icon=":material/arrow_back:", key="file_panel_back"):
                st.session_state["conversation_selected_file"] = None
                st.rerun()
        else:
            title_col.markdown("#### Files")

        with controls_col.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button(
                "Minimize" if expanded else "Maximize",
                icon=":material/close_fullscreen:" if expanded else ":material/open_in_full:",
                key="file_panel_size_toggle",
                type="secondary",
            ):
                st.session_state["conversation_file_panel_expanded"] = not expanded
                st.rerun()
            st.button("Close", icon=":material/close:", key="file_panel_close",
                      on_click=lambda: st.session_state.update(
                          conversation_files=False,
                          conversation_selected_file=None,
                          conversation_file_panel_expanded=False,
                      ))

        if selected:
            st.markdown(f"#### {workspace.literal(selected['label'])}")
            file_name = selected.get("file_name") or Path(selected["path"]).name
            st.caption(f"{file_name} · {selected['subtitle']}")
            if selected.get("kind") == "sandbox":
                ui_model_sandbox.render(chat)
            else:
                preview_key = hashlib.sha1(selected["id"].encode()).hexdigest()[:12]
                _preview_file(selected, preview_key)
        else:
            st.caption("Uploaded files and final run deliverables")
            if not items:
                st.info("Files you upload or generate will appear here.")
            for index, item in enumerate(items):
                st.button(item["label"], icon=f":material/{item['icon']}:", width="stretch",
                          key=f"file_panel_item_{chat['id']}_{index}", on_click=_open_file, args=(item,))
                st.caption(item["subtitle"])


def _column_review_prompt(dataset):
    review = dataset.get("column_review") or {}
    guesses = []
    if review.get("time_required"):
        guesses.extend(f"`{name}` → `time` (time in seconds)"
                       for name in review.get("suggested_time") or [])
    guesses.extend(f"`{name}` → `{core.role_column_name(name, 's_')}` (state)"
                   for name in review.get("suggested_states") or [])
    guesses.extend(f"`{name}` → `{core.role_column_name(name, 'a_')}` (input)"
                   for name in review.get("suggested_actions") or [])
    guesses.extend(f"`{name}` → removed as a sample index"
                   for name in review.get("suggested_ignored") or [])
    proposal = " My initial guess is " + " and ".join(guesses) + "." if guesses else " I can't safely infer the state columns from their names alone."
    return ("I’ve read the table. Could you help me confirm its columns before we use it?" + proposal +
            " A run needs time, at least one measured state, and at least one input/control signal (`a_*`). "
            "We want one time-rate (`xdot_*`) for every state. LabCD uses measured rates when present and estimates them from the state data when they are not in the source file. "
            "The choices are below; if they look right, reply `yes, that's ok`. Or tell me a header or position, such as `the fourth column is the state`, and I’ll fix the working copy. "
            "Any index column you choose to remove will be left out of that copy. Your original upload stays unchanged.")


def _start_run_setup_agent(chat):
    """Start the configured-API setup recommender after data assumptions are settled."""
    dataset = chat.get("dataset") or {}
    if not dataset.get("ready") or chat.get("run_dir"):
        return False
    setup = chat.get("setup") or {}
    if setup.get("stage") not in (None, "complete"):
        return False
    if core.REGISTRY.planning.get(chat["id"]):
        return False
    flow = chat.get("run_setup_flow") or {}
    if flow.get("dataset_sha256") == dataset.get("sha256"):
        return False
    job = run_setup_agent.RunSetupAgentJob(chat)
    core.REGISTRY.planning[chat["id"]] = job
    job.start()
    return True


def _finish_column_review(chat, state_columns, action_columns, time_column=None, ignored_columns=None):
    try:
        mapping, dataset = core.apply_column_roles(chat, state_columns, action_columns,
                                                   time_column=time_column,
                                                   ignored_columns=ignored_columns)
    except ValueError as exc:
        core.add_message(chat, "assistant", str(exc))
        return
    changes = ", ".join(f"`{old}` → `{new}`" for old, new in mapping.items())
    core.add_message(chat, "assistant",
                     f"I prepared the working copy: {changes}. I saved **{dataset['name']}** and kept the original upload unchanged.",
                     kind="column_correction", column_mapping=mapping, attachment=dataset)
    if dataset["issues"]:
        core.add_message(chat, "assistant", "The column names are fixed. Before training, please address these remaining data checks:\n\n" +
                         "\n".join(f"- {issue}" for issue in dataset["issues"]))
        return
    source = chat.get("pinn_source") or {}
    if source.get("status") in ("pending", "ready") and not pinn_maker.is_equation_ready(chat):
        mode = str((chat.get("setup") or {}).get("pending_physics_mode") or
                   (chat.get("settings") or {}).get("physics_mode") or "pinn").lower()
        message = ("Your columns are confirmed. I’m mapping the saved hard-constraint source to them and checking each rule."
                   if mode == "pcnn" else "Your columns are confirmed. I’m mapping the saved PINN model to them and checking every state derivative.")
        core.add_message(chat, "assistant", message)
        _start_pinn_preparation(chat, "Prepare the saved physics source using the confirmed state and input columns.")
        return
    _start_data_review(chat, AUTO_DATA_REVIEW_PROMPT)


def _finish_setup_signal_roles(chat, state_columns, action_columns):
    """Save an edited state/control map and restart checks that depend on it."""
    previous = chat.get("dataset") or {}
    old_states = list(previous.get("states") or [])
    old_actions = list(previous.get("inputs") or previous.get("actions") or [])
    if set(old_states) == set(state_columns) and set(old_actions) == set(action_columns):
        return False

    settings = core.RunSettings.model_validate(chat["settings"]).model_dump()
    angle_names = {
        old_states[index]
        for index in (settings.get("angle_indices") or [])
        if isinstance(index, int) and 0 <= index < len(old_states)
    }
    mapping, dataset = core.apply_column_roles(
        chat, state_columns, action_columns, allow_reassign=True
    )
    mapped_angles = {mapping.get(name, name) for name in angle_names}
    new_angles = [
        index for index, name in enumerate(dataset.get("states") or [])
        if name in mapped_angles
    ]
    new_states = list(dataset.get("states") or [])
    new_actions = list(dataset.get("inputs") or dataset.get("actions") or [])
    role_changed = set(old_states) != set(new_states) or set(old_actions) != set(new_actions)
    updates = {"angle_indices": new_angles}
    if role_changed:
        # A saved equation was checked against the previous state/control map.
        updates["use_pinn"] = False
    chat["settings"] = core.apply_changes(chat["settings"], updates)
    chat["run_setup_flow"] = None
    core.save_chat(chat)

    core.add_message(
        chat,
        "assistant",
        f"I updated the working dataset to **{len(new_states)} state outputs** and "
        f"**{len(new_actions)} system inputs**. The model dimensions now follow those "
        "selected columns. The original upload is unchanged; I’m refreshing the data assumptions before setup resumes.",
        kind="column_correction",
        column_mapping=mapping,
        attachment=dataset,
    )
    if dataset.get("issues"):
        core.add_message(
            chat,
            "assistant",
            "The signal map was saved, but the updated dataset needs attention before training:\n\n"
            + "\n".join(f"- {issue}" for issue in dataset["issues"]),
        )
        return True

    source = chat.get("pinn_source") or {}
    if source.get("status") in ("pending", "ready") and not pinn_maker.is_equation_ready(chat):
        mode = str((chat.get("setup") or {}).get("pending_physics_mode") or
                   (chat.get("settings") or {}).get("physics_mode") or "pinn").lower()
        core.add_message(
            chat,
            "assistant",
            ("The signal map changed, so I’m rechecking the hard-constraint rules against the updated columns."
             if mode == "pcnn" else
             "The signal map changed, so I’m rechecking the full PINN derivative model against the updated columns."),
        )
        _start_pinn_preparation(chat, "Recheck the saved physics source using the updated state and input columns.")
    else:
        _start_data_review(chat, AUTO_DATA_REVIEW_PROMPT)
    return True


def _chat_signal_role_edit(chat, question):
    """Apply an explicit chat request to reassign uploaded columns as states or inputs."""
    dataset = chat.get("dataset") or {}
    if not dataset.get("ready"):
        return False
    text = str(question or "")
    role_pattern = re.compile(
        r"\b(?P<role>measured\s+states?|state\s+outputs?|states?|outputs?|inputs?|actions?|controls?)\b",
        re.I,
    )
    role_matches = list(role_pattern.finditer(text))
    if not role_matches:
        return False
    columns = [
        str(name) for name in dataset.get("columns", [])
        if str(name).casefold() != "time" and not str(name).startswith("xdot_")
    ]

    def names_in(segment):
        return [
            name for name in columns
            if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", segment, re.I)
        ]

    edit_intent = bool(re.search(
        r"\b(change|edit|modify|adjust|update|set|make|move|assign|switch)\b", text, re.I
    ))
    if not edit_intent:
        mentions_column = any(names_in(text))
        has_assignment = bool(re.search(r"\b(?:to|as|should\s+be|are)\b|[:=]", text, re.I))
        edit_intent = mentions_column and has_assignment and bool(re.search(r"\b(want|prefer|need)\b", text, re.I))
    if not edit_intent:
        return False

    current_states = list(dataset.get("states") or [])
    current_actions = list(dataset.get("inputs") or dataset.get("actions") or [])
    requested_states = None
    requested_actions = None

    for index, match in enumerate(role_matches):
        label = match.group("role").casefold()
        destination = "states" if ("state" in label or "output" in label) else "actions"
        next_start = role_matches[index + 1].start() if index + 1 < len(role_matches) else len(text)
        segment = text[match.end():next_start]
        found = names_in(segment)
        if found:
            if destination == "states":
                requested_states = found
            else:
                requested_actions = found

    # Also accept natural singular assignments such as “make s_pitch a state”
    # and “assign throttle as an input”.
    individually_assigned = False
    for name in columns:
        occurrence = re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text, re.I)
        if not occurrence:
            continue
        start = occurrence.start()
        tail = text[occurrence.end():occurrence.end() + 60]
        assignment = re.search(
            r"^\s*(?:(?:as|to\s+be|be|is)\s+)?(?:a|an|the)?\s*"
            r"(?P<role>measured\s+state|state|output|input|control|action)\b",
            tail, re.I,
        )
        if not assignment:
            continue
        individually_assigned = True
        destination = "states" if assignment.group("role").casefold() in {"measured state", "state", "output"} else "actions"
        if requested_states is None:
            requested_states = current_states.copy()
        if requested_actions is None:
            requested_actions = current_actions.copy()
        requested_states = [column for column in requested_states if column != name]
        requested_actions = [column for column in requested_actions if column != name]
        target = requested_states if destination == "states" else requested_actions
        target.append(name)

    if requested_states is None and requested_actions is None:
        if not individually_assigned:
            state_example = ", ".join(f"`{name}`" for name in (current_states[:2] or columns[:1])) or "`s_position`"
            input_example = ", ".join(f"`{name}`" for name in (current_actions[:2] or columns[-1:])) or "`a_control`"
            core.add_message(
                chat, "assistant",
                "I can update the signal map from chat. Name the uploaded columns you want as states and inputs, "
                f"for example: “set states to {state_example} and inputs to {input_example}.”",
            )
            return True
        requested_states = current_states
        requested_actions = current_actions

    try:
        changed = _finish_setup_signal_roles(
            chat,
            requested_states if requested_states is not None else current_states,
            requested_actions if requested_actions is not None else current_actions,
        )
    except ValueError as exc:
        core.add_message(chat, "assistant", str(exc))
        return True
    if not changed:
        core.add_message(chat, "assistant", "Those columns are already assigned that way in the current setup.")
    return True


def _start_data_review(chat, prompt):
    core.begin_setup(chat)
    job = agent.ConversationJob(chat, prompt, purpose="data_review")
    core.REGISTRY.conversations[chat["id"]] = job
    job.start()


def _start_model_sheet(chat, description):
    description = str(description or "").strip()
    if not description:
        st.warning("Tell me a little about the physical system first; sensor and unit details can be uncertain.")
        return False
    chat.pop("first_step_stage", None)
    core.save_chat(chat)
    job = model_sheet.ModelSheetJob(description)
    core.REGISTRY.conversations[chat["id"]] = job
    job.start()
    st.rerun()
    return True


def _use_starter_sheet(chat, frame):
    """Save real values entered in the starter grid, then reuse upload checks."""
    import pandas as pd

    try:
        frame = frame.copy()
        rate_columns = [name for name in frame.columns if str(name).casefold().startswith("xdot_")]
        blank_rates = []
        incomplete_rates = []
        for column in rate_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().all():
                blank_rates.append(column)
            elif numeric.isna().any():
                incomplete_rates.append(column)
        if incomplete_rates:
            core.add_message(
                chat, "assistant",
                "Please complete every row in these state-rate columns: "
                + ", ".join(f"`{name}`" for name in incomplete_rates)
                + ". If you want LabCD to estimate the rates, clear the whole `xdot_*` column instead.",
            )
            st.rerun()
            return
        if blank_rates and len(blank_rates) < len(rate_columns):
            core.add_message(
                chat, "assistant",
                "LabCD wants a rate for every state. For this sheet, either fill every listed `xdot_*` column, "
                "or clear all of them so LabCD can estimate the full set from the state measurements.",
            )
            st.rerun()
            return
        if blank_rates:
            frame = frame.drop(columns=blank_rates)
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        dataset = core.inspect_upload("labcd_measurements.csv", csv_bytes, chat["id"])
    except (OSError, ValueError, TypeError) as exc:
        core.add_message(chat, "assistant", f"I couldn't use that sheet yet: {exc}")
        st.rerun()
        return

    chat["dataset"] = dataset
    core.add_message(
        chat, "user", "I entered measurements in the suggested sheet.",
        answer_scope="setup", attachment=dataset,
    )
    if dataset.get("column_review_pending"):
        core.add_message(chat, "assistant", _column_review_prompt(dataset), kind="column_review")
    elif dataset.get("issues"):
        details = "\n".join(f"- {issue}" for issue in dataset["issues"])
        core.add_message(
            chat, "assistant",
            "I checked the measurements from the sheet. A few data checks remain before we can use them:\n\n"
            + details + "\n\nYou can add or correct rows in the sheet and try again.",
        )
    else:
        source = chat.get("pinn_source") or {}
        if (source.get("status") in ("pending", "ready")
                and not pinn_maker.is_equation_ready(chat)
                and _start_pinn_preparation(
                    chat, "Prepare the saved physics source using the confirmed state and input columns.")):
            st.rerun()
        _start_data_review(chat, AUTO_DATA_REVIEW_PROMPT)
    st.rerun()


def _start_pinn_preparation(chat, question):
    """Queue one automatic equation-preparation pass for ready, confirmed data."""
    source = chat.get("pinn_source")
    dataset = chat.get("dataset") or {}
    if not isinstance(source, dict) or not dataset.get("ready") or dataset.get("column_review_pending"):
        return False
    current = core.REGISTRY.conversations.get(chat["id"])
    if current and current.running:
        return False
    if source.get("status") == "processing":
        return False
    source["status"] = "processing"
    core.save_chat(chat)
    job = pinn_maker.PINNMakerJob(chat, question)
    core.REGISTRY.conversations[chat["id"]] = job
    job.start()
    return True


def _prepare_saved_pinn_when_ready(chat):
    """Resume the saved source automatically once confirmed data is available."""
    source = chat.get("pinn_source") or {}
    if chat["id"] in core.REGISTRY.conversations:
        return False
    # A persisted "processing" state can survive an app restart even though its
    # in-memory worker cannot. Since this is called only when the chat is idle,
    # safely reset it so the automatic preparation can resume.
    if source.get("status") == "processing":
        source["status"] = "pending"
        core.save_chat(chat)
    if source.get("status") not in ("pending", "ready"):
        return False
    if pinn_maker.is_equation_ready(chat):
        return False
    return _start_pinn_preparation(
        chat, "Prepare the saved physics source using the attached dataset's confirmed state and input columns.")


def _column_review_controls(chat, key):
    dataset = chat.get("dataset") or {}
    review = dataset.get("column_review") or {}
    options = review.get("options") or []
    state_defaults = [name for name in review.get("suggested_states") or [] if name in options]
    input_defaults = [name for name in review.get("suggested_actions") or []
                      if name in options and name not in state_defaults]
    ignored_defaults = [name for name in review.get("suggested_ignored") or []
                        if name in options and name not in state_defaults and name not in input_defaults]
    with st.form(f"column_review_form_{key}", border=True):
        selected_time = "time"
        if review.get("time_required"):
            time_options = [name for name in (review.get("time_options") or options) if name in options]
            suggested_time = [name for name in review.get("suggested_time") or [] if name in time_options]
            time_choices = ["Choose a column…", *time_options]
            time_default = suggested_time[0] if len(suggested_time) == 1 else "Choose a column…"
            selected_time = st.selectbox(
                "Time column · renamed to time",
                options=time_choices,
                index=time_choices.index(time_default),
                key=f"column_review_time_{key}",
                help="Choose the column that contains timestamps in seconds.")
            if selected_time == "Choose a column…":
                selected_time = None
        selected_states = st.multiselect(
            "Measured state columns · renamed with s_",
            options=options, default=state_defaults,
            key=f"column_review_states_{key}",
            help="Choose the measured values that describe the system's current state.")
        selected_actions = st.multiselect(
            "Input or control columns · required, renamed with a_",
            options=options, default=input_defaults,
            key=f"column_review_inputs_{key}",
            help="Choose at least one signal applied to the system, such as force, voltage, or command.")
        st.caption("At least one input is required. We want a time-rate (`xdot_*`) for every state; LabCD estimates rates from the state data when they are not in the source file.")
        selected_ignored = st.multiselect(
            "Remove index or unused columns · optional",
            options=options, default=ignored_defaults,
            key=f"column_review_ignored_{key}",
            help="For example, remove a sequential row counter such as k from the training copy.")
        overlap = ((set(selected_states) & set(selected_actions)) |
                   (set(selected_ignored) & (set(selected_states) | set(selected_actions))))
        if selected_time:
            overlap |= {selected_time} & (set(selected_states) | set(selected_actions) | set(selected_ignored))
        if overlap:
            st.warning("A column cannot be assigned multiple roles or removed while in use: " + ", ".join(sorted(overlap)))
        submitted = st.form_submit_button(
            "Confirm roles and fix headers", icon=":material/check:",
            disabled=not selected_states or bool(overlap) or not selected_time,
            key=f"column_review_confirm_{key}")
    if submitted:
        _finish_column_review(chat, selected_states, selected_actions,
                              time_column=selected_time, ignored_columns=selected_ignored)
        st.rerun()


def _messages(chat, busy, file_items=None):
    file_items = file_items if file_items is not None else _file_items(chat)
    active_training = core.REGISTRY.training.get(chat["id"])
    preparing_or_training = bool(active_training and active_training["runner"].running)
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    setup_summary_visible = (
        flow.get("dataset_sha256") == dataset.get("sha256")
        and not chat.get("run_dir")
        and any(item.get("kind") == "run_setup" for item in chat["messages"])
    )
    latest_plan = next((m["id"] for m in reversed(chat["messages"]) if m.get("kind") == "plan"), None)
    latest_run_setup = next((m["id"] for m in reversed(chat["messages"]) if m.get("kind") == "run_setup"), None)
    latest_setup = next((m["id"] for m in reversed(chat["messages"]) if m.get("kind") == "setup_question"), None)
    pending_question_id = _pending_client_question_id(chat.get("messages") or [])
    for message_index, message in enumerate(chat["messages"]):
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else str(ui_brand.ICON_PATH)):
            if message["role"] == "assistant":
                if message.get("kind") == "result" and message.get("run_dir"):
                    _render_completed_run_thinking(message["run_dir"], message["id"])
                elif message.get("kind") != "result":
                    _render_message_activity(message, message.get("id") or str(message_index))
            if message.get("kind") == "data_review" and chat.get("dataset"):
                data_ui.render_data_quality_notice(chat["dataset"], message["id"])
            content_rendered = False
            if message.get("diagnosis"):
                diagnosis_ui.render_answer(message["diagnosis"])
                content_rendered = True
            elif message.get("evidence") or message.get("kind") in ("excel_maker", "excel_help", "pinn_maker", "physics_model_proposal", "clarification", "column_review", "column_correction", "pasted_table_preview", "run_setup", "starter_sheet"):
                # Model prose is Markdown; Streamlit escapes raw HTML by default.
                safe = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", message["content"])
                if message["role"] == "assistant":
                    safe = re.sub(r"\s*\[[A-Z]{1,3}\d+\]", "", safe)
                st.markdown(safe)
                content_rendered = True
            if message.get("kind") == "excel_help":
                _excel_help_sheet(message["id"])
            elif message.get("kind") == "starter_sheet":
                entered = ui_model_sheet.render(chat, message)
                if entered is not None:
                    _use_starter_sheet(chat, entered)
            elif not content_rendered:
                _literal(message["content"])
            if attachment := message.get("attachment"):
                uploaded = next((item for item in file_items
                                 if item["id"] == f"upload:{message.get('id')}" or
                                 (item.get("path") and attachment.get("path") and
                                  Path(item["path"]).resolve() == Path(attachment["path"]).resolve())), None)
                if uploaded:
                    _file_card(uploaded, f"upload_{message['id']}")
                if attachment.get("kind") == "pinn_source":
                    st.caption("Original physics source, preserved unchanged. Open it from the Files panel to inspect or download it.")
            if artifact := message.get("generated_artifact"):
                generated = next((item for item in file_items
                                  if item["id"] == _generated_file_id(artifact.get("path", ""))), None)
                if generated:
                    if artifact.get("kind") == "pinn_equation":
                        artifact_mode = str(artifact.get("physics_mode") or
                                            (chat.get("setup") or {}).get("pending_physics_mode") or
                                            "pinn").lower()
                        if artifact.get("origin") == "ai_proposal":
                            st.markdown("**PCNN hard-constraint candidates · ready for review**" if artifact_mode == "pcnn"
                                        else "**PINN model proposal · ready for your review**")
                        else:
                            st.markdown("**PCNN hard-constraint rules · mapped and checked**" if artifact_mode == "pcnn"
                                        else "**PINN model · mapped and checked against this dataset**")
                        _file_card(generated, f"pinn_{message['id']}")
                        equations = artifact.get("equations") or []
                        if equations:
                            st.dataframe(equations, width="stretch", hide_index=True,
                                         column_config={"state": "Constrained state" if artifact_mode == "pcnn" else "Measured state",
                                                        "alias": "State variable",
                                                        "expression": "Exact derivative rule" if artifact_mode == "pcnn" else "Derivative dx/dt"})
                        if artifact.get("origin") == "ai_proposal":
                            recommendations = artifact.get("pcnn_constraint_recommendations") or []
                            if recommendations:
                                with st.container(border=True):
                                    st.markdown("**PCNN constraint suggestions**")
                                    st.dataframe(
                                        [{
                                            "State": item.get("state", ""),
                                            "Starting suggestion": (
                                                ("Agent · " if item.get("source", "agent") == "agent" else "Review · ")
                                                + ("Exact hard constraint" if item.get("recommended_exact") else "Neural output")
                                                + f" · {str(item.get('confidence', 'low')).title()}"
                                            ),
                                            "Why": item.get("rationale", ""),
                                        } for item in recommendations],
                                        width="stretch", hide_index=True,
                                    )
                                    st.caption("These suggestions identify candidate hard constraints, not proven laws. Choose the rules to enforce in Starting setup.")
                        variables = artifact.get("state_aliases", []) + artifact.get("action_aliases", [])
                        if variables:
                            st.caption("Variable map · " + " · ".join(
                                f"`{item['alias']}` = `{item['column']}`" for item in variables))
                        if parameters := artifact.get("parameters"):
                            st.markdown("**Coefficients learned during training**")
                            st.dataframe([{"Coefficient": item.get("name"), "Initial": item.get("initial"),
                                           "Allowed range": f"{item.get('lower')} to {item.get('upper')}",
                                           "Units": item.get("units") or "—"} for item in parameters],
                                         width="stretch", hide_index=True)
                        if artifact.get("data_checks"):
                            with st.expander("Data limits", expanded=False):
                                for check in artifact["data_checks"]:
                                    st.caption(check)
                        if artifact.get("origin") == "ai_proposal":
                            current_artifact = chat.get("pinn_equation") or {}
                            current_proposal = (current_artifact.get("proposal_id", current_artifact.get("sha256")) ==
                                                artifact.get("proposal_id", artifact.get("sha256")) and
                                                pinn_maker.is_equation_ready(chat))
                            if current_artifact.get("approved"):
                                st.success("You approved this physics model for the starting setup.")
                            elif current_proposal:
                                approve_col, revise_col = st.columns([1.6, 1], gap="small")
                                approve_label = "Approve hard-constraint rules" if artifact_mode == "pcnn" else "Approve PINN model"
                                if approve_col.button(approve_label, type="primary",
                                                      icon=":material/check_circle:", key=f"approve_physics_{message['id']}",
                                                      disabled=busy, width="stretch"):
                                    current_artifact["approved"] = True
                                    chat["pinn_equation"] = current_artifact
                                    setup = chat.get("setup") or {}
                                    pending_mode = setup.get("pending_physics_mode", "pinn")
                                    chat["settings"]["use_pinn"] = True
                                    chat["settings"]["physics_mode"] = pending_mode
                                    if setup.get("stage") == "physics_model_review":
                                        if pending_mode == "pcnn":
                                            reply = core.setup_completion_reply(
                                                chat, "You approved these PCNN hard-constraint candidates. You’ll select which state rules to enforce exactly in Starting setup. "
                                            )
                                            core.add_message(chat, "assistant", reply.pop("content"), **reply)
                                            _start_run_setup_agent(chat)
                                        else:
                                            setup["pinn_interest"] = True
                                            setup["stage"] = "pcnn_interest"
                                            chat["setup"] = setup
                                            core.save_chat(chat)
                                            core.add_message(chat, "assistant", core.setup_prompt(chat),
                                                             kind="setup_question", stage="pcnn_interest")
                                    else:
                                        core.save_chat(chat)
                                    st.session_state[f"run_setup_{chat['id']}_use_pinn"] = True
                                    st.rerun()
                                if revise_col.button("Discard and revise", icon=":material/edit:",
                                                     key=f"discard_physics_{message['id']}", disabled=busy,
                                                     width="stretch"):
                                    chat["pinn_equation"] = None
                                    chat["settings"]["use_pinn"] = False
                                    setup = chat.get("setup") or {}
                                    if setup.get("stage") == "physics_model_review":
                                        setup["stage"] = "physics_model_description"
                                        chat["setup"] = setup
                                    core.save_chat(chat)
                                    core.add_message(chat, "assistant", "I discarded that draft. Revise the system description below and I’ll prepare another equation set.", answer_scope="setup")
                                    if (chat.get("setup") or {}).get("stage") == "physics_model_description":
                                        core.add_message(chat, "assistant", core.setup_prompt(chat),
                                                         kind="setup_question", stage="physics_model_description")
                                    st.rerun()
                            else:
                                st.info("This draft is no longer current. Review the latest physics proposal before approving it.")
                            if (chat.get("setup") or {}).get("stage") == "physics_model_review":
                                st.caption("Approval saves these candidates. You choose which ones to enforce in Starting setup; it does not start training.")
                            else:
                                st.caption("Approval makes these rules available to PCNN. Choose exact state constraints in Starting setup; training starts only after you approve and start the run.")
                        else:
                            st.caption("The mapped physics rules are available in Files and can be selected for the next run.")
                    else:
                        st.markdown("**Excel workbook · created by Excel Maker**")
                        _file_card(generated, f"excel_{message['id']}")
                        try:
                            import pandas as pd
                            preview = pd.read_excel(artifact["path"], header=3, nrows=12)
                            st.caption(f"Workbook preview · first {len(preview):,} rows")
                            st.dataframe(preview, width="stretch", hide_index=True)
                        except (OSError, ValueError, KeyError, ImportError) as exc:
                            st.caption(f"Open the workbook in Files to preview it. {exc}")
            if message.get("kind") == "pasted_table_preview":
                preview_info = message.get("preview_dataset") or {}
                try:
                    frame = data_ui._preview(preview_info["path"], preview_info["sha256"], preview_info["name"])
                    shown = frame.head(12)
                    st.caption(f"Table preview · {preview_info.get('rows', len(frame)):,} rows · {len(frame.columns)} columns")
                    st.dataframe(shown, width="stretch", hide_index=True)
                    if len(frame) > len(shown):
                        st.caption(f"Showing {len(shown)} rows here; Excel Maker keeps all {len(frame):,} rows.")
                except (KeyError, OSError, ValueError) as exc:
                    st.warning(f"Couldn't preview the pasted table: {exc}")
                if st.button("Create Excel workbook", icon=":material/table_view:",
                             key=f"excel_from_pasted_table_{message['id']}", disabled=busy):
                    st.session_state["conversation_pending"] = "Create an Excel workbook from the pasted table above."
                    st.rerun()
            if message.get("kind") == "plan" and not preparing_or_training and not setup_summary_visible:
                _plan(message["settings"], chat, message["id"],
                      can_start=not busy and message["id"] == latest_plan and not chat.get("run_dir"))
            if message.get("kind") == "column_review":
                if message.get("resolved"):
                    mapping = message.get("column_mapping") or {}
                    st.caption("Confirmed: " + ", ".join(f"{old} → {new}" for old, new in mapping.items()))
                else:
                    _column_review_controls(chat, message["id"])
            if message.get("kind") == "result":
                entry = _entry(chat) or {"run_dir": message["run_dir"]}
                entry["run_dir"] = message["run_dir"]
                requested = (chat.get("settings") or {}).get("max_cycles")
                data_ui.render_run_result(entry, message["id"], requested_cycles=requested)
                run_items = [item for item in file_items if item["id"].startswith("artifact:")]
                _file_cards(run_items, message["id"])
                _result_tools(message["run_dir"], message["id"])
                prompts = ([
                    "Why did this run fail?",
                    "What points to the cause?",
                    "Can I retry with fewer cycles?",
                    "What should I try next?",
                ] if entry.get("status") != "completed" else [
                    "Which state was predicted best?",
                    "Explain the prediction plot",
                    "What changed across candidate cycles?",
                    "What should I try next?",
                ])
                prompt_key = f"run_prompts_{message['id']}"
                st.pills("Try asking about this run", prompts, key=prompt_key, selection_mode="single",
                         on_change=_queue_suggested_prompt, args=(prompt_key,), wrap=True)
            if message.get("kind") == "data_review" and chat.get("dataset"):
                data_ui.render_dataset(chat["dataset"], message["id"])
            if message.get("kind") == "setup_question":
                setup_stage = (chat.get("setup") or {}).get("stage")
                needs_explicit_action = setup_stage in {
                    "pinn_interest", "pcnn_interest", "physics_source_choice", "physics_source_upload", "physics_source_processing",
                    "physics_model_description", "physics_model_drafting", "physics_model_review",
                }
                is_active_question = (
                    not busy and message["id"] == latest_setup
                    and setup_stage == message.get("stage")
                )
                data_ui.render_setup_question(chat, message["id"],
                                              is_active_question,
                                              question_timer=(
                                                  lambda: _render_client_question_skip(chat, message["id"])
                                                  if message["id"] == pending_question_id and not needs_explicit_action else None
                                              ))
            if (message.get("kind") == "run_setup" and not preparing_or_training
                    and not chat.get("run_dir")
                    and message["id"] == latest_run_setup):
                _run_setup_card(chat, message, busy)
            if message.get("state_comparison"):
                data_ui.render_state_comparison(message["state_comparison"])
            if (message.get("visual_question") and not message.get("state_comparison")
                    and (chat.get("run_dir") or not chat.get("dataset"))):
                data_ui.render_answer_visuals(
                    message["visual_question"], message["id"],
                    dataset=chat.get("dataset"), run_dir=chat.get("run_dir"))
            if message.get("error_detail"):
                with st.expander("Connection details"):
                    st.text(message["error_detail"])
            _render_answer_feedback(
                chat, message,
                key_id=message.get("id") or _fallback_feedback_id(chat, message, message_index),
                save=lambda: core.save_chat(chat),
            )


_NON_ANSWER_MESSAGE_KINDS = {
    "plan", "run_setup", "setup_question", "column_review", "column_correction",
    "pasted_table_preview", "starter_sheet", "clarification", "error",
}


def _fallback_feedback_id(chat, message, index):
    identity = "\0".join((str(chat.get("id", "")), str(index),
                           str(message.get("timestamp", "")), str(message.get("content", ""))))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _render_answer_feedback(chat, message, *, key_id, save):
    """Let clients rate a completed assistant answer and persist it with the chat."""
    if message.get("role") != "assistant" or message.get("kind") in _NON_ANSWER_MESSAGE_KINDS:
        return
    previous = message.get("feedback") or {}
    rating = previous.get("rating")
    default = {"down": 0, "up": 1}.get(rating)
    st.caption("Was this answer helpful?")
    selected = st.feedback("thumbs", key=f"answer_feedback_{chat['id']}_{key_id}", default=default)
    if selected is None:
        return
    current = "up" if selected == 1 else "down"
    if rating != current:
        message["feedback"] = {
            "rating": current,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        save()
    st.caption("Feedback saved with this conversation.")


def _format_worked_time(elapsed):
    seconds = max(0, int(round(float(elapsed or 0))))
    if seconds == 0:
        return "under 1s"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s" if seconds else f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
    return f"{seconds}s"


def _render_message_activity(message, scope):
    """Show elapsed response time and concise, client-safe activity on demand."""
    elapsed = message.get("response_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
        return
    steps = message.get("response_steps")
    if not isinstance(steps, list) or not steps:
        kind = message.get("kind")
        if kind == "data_review":
            steps = ["Reviewed the uploaded dataset", "Prepared a summary of its structure"]
        elif kind == "run_setup":
            steps = ["Checked the dataset and run settings", "Prepared the starting setup"]
        elif kind == "plan":
            steps = ["Reviewed the requested setup change", "Prepared an updated run plan"]
        elif message.get("answer_scope") == "run":
            steps = ["Read your message", "Checked the selected run details", "Prepared a reply"]
        elif message.get("answer_scope") == "setup":
            steps = ["Read your message", "Checked the saved setup details", "Prepared a reply"]
        else:
            steps = ["Read your message", "Prepared a reply"]
    with st.container(key=f"message_activity_{scope}"):
        with st.status(f"Worked for {_format_worked_time(elapsed)}", state="complete",
                       expanded=False, type="compact"):
            st.caption("Activity")
            for step in steps:
                if isinstance(step, str) and step.strip():
                    st.markdown(f"- {step.strip()}")
        st.divider()


def _save_setup_recommendation(chat, recommendation=None, error=None, error_detail=None,
                               response_seconds=None):
    dataset = chat.get("dataset") or {}
    settings = chat["settings"]
    physics_advisory = ((recommendation or {}).get("physics_advisory") or
                        run_setup_agent.physics_advisory(chat))
    if recommendation:
        effort = recommendation["search_effort"]
        cycle_cap = int(cfg.run_mode_limits(effort)["max_cycles"])
        cycles = cycle_cap if effort == "fast" else min(int(recommendation["cycles"]), cycle_cap)
        chat["settings"] = core.apply_changes(settings, {
            "architecture": recommendation["architecture"],
            "lstm_seq_length": recommendation["history_steps"],
            "run_mode": effort,
            "max_cycles": cycles,
            "use_pinn": settings.get("use_pinn", False) and pinn_maker.is_equation_ready(chat),
        })
        chat["settings"], adjustments = core.fit_settings_to_sample_count(chat["settings"], dataset.get("rows", 0))
        flow = {
            "dataset_sha256": dataset.get("sha256"),
            "stage": "model",
            "recommended_architecture": recommendation["architecture"],
            "architecture_reason": recommendation["architecture_reason"],
            "recommended_history_steps": chat["settings"]["lstm_seq_length"],
            "recommended_effort": effort,
            "recommended_cycles": cycles,
            "effort_reason": recommendation["effort_reason"],
            "confidence": recommendation["confidence"],
            "agent_model": recommendation.get("model", "configured model"),
            "sample_adjustments": adjustments,
            "physics_advisory": physics_advisory,
        }
        content = (f"I looked over the measured-data profile and prepared {recommendation['architecture']} as a starting point. "
                   "You can edit the network and every run setting below. One approval starts candidate training; held-out results will give us the first measured comparison.")
    else:
        fallback_effort = settings["run_mode"] if settings["run_mode"] in ("fast", "regular", "heavy") else "regular"
        fallback_cycle_cap = int(cfg.run_mode_limits(fallback_effort)["max_cycles"])
        flow = {
            "dataset_sha256": dataset.get("sha256"),
            "stage": "model",
            "recommended_architecture": settings["architecture"],
            "architecture_reason": "Choose the first model from the measured data. The current selection is only a starting point; compare held-out predictions after training.",
            "recommended_history_steps": settings["lstm_seq_length"],
            "recommended_effort": fallback_effort,
            "recommended_cycles": fallback_cycle_cap if fallback_effort == "fast" else min(int(settings["max_cycles"]), fallback_cycle_cap),
            "effort_reason": "Select a search budget that fits your time and desired breadth.",
            "confidence": "low",
            "agent_error": error or "The setup recommendation is unavailable.",
            "physics_advisory": physics_advisory,
        }
        content = "I couldn’t get a model recommendation this time. You can choose a starting model and edit the settings below. One approval starts candidate training."
    # Keep the setup-time baseline separate from later manual edits so each
    # editor can restore exactly what was proposed for this dataset.
    flow["initializer_proposal"] = core.RunSettings.model_validate(chat["settings"]).model_dump()
    flow["initializer_dataset_roles"] = {
        "states": list(dataset.get("states") or []),
        "actions": list(dataset.get("inputs") or dataset.get("actions") or []),
    }
    chat["run_setup_flow"] = flow
    core.save_chat(chat)
    response_steps = (
        ["Reviewed the uploaded dataset", "Prepared the starting setup"]
        if recommendation else
        ["Reviewed the available dataset information", "Prepared a fallback setup"]
    )
    core.add_message(chat, "assistant", content, kind="run_setup", answer_scope="setup",
                     **({"response_seconds": response_seconds, "response_steps": response_steps}
                        if isinstance(response_seconds, (int, float)) else {}),
                     **({"error_detail": error_detail} if error_detail else {}))


def _sync(chat, hooks):
    identity, registry = chat["id"], core.REGISTRY
    if job := registry.conversations.get(identity):
        if not job.running:
            setup_followup = None
            physics_model_retry = False
            physics_source_retry = False
            if getattr(job, "answer_scope", None):
                for message in reversed(chat.get("messages", [])):
                    if (message.get("role") == "user" and
                            str(message.get("content", "")).strip() == str(job.question).strip()):
                        message["answer_scope"] = job.answer_scope
                        break
            if job.answer:
                if job.purpose == "model_sheet":
                    job.answer["answer"] = job.answer["message"]
                    job.answer.setdefault("evidence", [])
                    extra = {
                        "kind": "starter_sheet",
                        "starter_schema": job.answer["schema"],
                        "starter_fallback": bool(job.answer.get("fallback")),
                        "answer_scope": "setup",
                    }
                    if job.answer.get("model"):
                        extra["model"] = job.answer["model"]
                    if getattr(job, "error_detail", None):
                        extra["error_detail"] = job.error_detail
                else:
                    extra = {"evidence": job.answer["evidence"], "model": job.answer["model"]}
                if job.answer.get("answer_scope"):
                    extra["answer_scope"] = job.answer["answer_scope"]
                if job.answer.get("diagnosis"):
                    extra["diagnosis"] = job.answer["diagnosis"]
                if ((chat.get("run_dir") or chat.get("dataset")) and
                        job.purpose not in ("excel_maker", "pinn_maker", "physics_model_agent") and
                        job.answer.get("status") != "clarification" and
                        not job.answer.get("diagnosis")):
                    extra["visual_question"] = job.question
                if job.purpose == "excel_maker":
                    extra["kind"] = "excel_maker"
                    extra["generated_artifact"] = job.answer["generated_artifact"]
                if job.purpose == "pinn_maker":
                    extra["kind"] = "pinn_maker"
                    artifact = job.answer.get("generated_artifact")
                    if artifact:
                        extra["generated_artifact"] = artifact
                        _store_physics_equation(chat, artifact)
                        if isinstance(chat.get("pinn_source"), dict):
                            chat["pinn_source"]["status"] = "ready"
                        setup = chat.get("setup") or {}
                        if setup.get("stage") == "physics_source_processing":
                            pending_mode = setup.get("pending_physics_mode", "pinn")
                            chat["settings"]["use_pinn"] = True
                            chat["settings"]["physics_mode"] = pending_mode
                            if pending_mode == "pcnn":
                                setup_followup = core.setup_completion_reply(
                                    chat, "I mapped and checked the supplied PCNN hard-constraint rules. Select which states to enforce exactly in Starting setup. "
                                )
                            else:
                                setup["pinn_interest"] = True
                                setup["stage"] = "pcnn_interest"
                                chat["setup"] = setup
                                setup_followup = {
                                    "content": core.setup_prompt(chat), "kind": "setup_question",
                                    "stage": "pcnn_interest",
                                }
                        else:
                            chat["settings"]["use_pinn"] = True
                        st.session_state[f"run_setup_{chat['id']}_use_pinn"] = bool(chat["settings"]["use_pinn"])
                        setup_suffix = f"{chat['id']}_{(chat.get('dataset') or {}).get('sha256', '')[:8]}"
                        st.session_state[f"setup_pinn_{setup_suffix}"] = bool(chat["settings"]["use_pinn"])
                        st.session_state[f"setup_pinn_interest_{setup_suffix}"] = (
                            "Yes" if chat["settings"]["use_pinn"] else "No"
                        )
                        core.save_chat(chat)
                    elif chat.get("pinn_source"):
                        chat["pinn_source"]["status"] = "needs_clarification"
                        if (chat.get("setup") or {}).get("stage") == "physics_source_processing":
                            chat["setup"]["stage"] = "physics_source_upload"
                            physics_source_retry = True
                        core.save_chat(chat)
                if job.purpose == "physics_model_agent":
                    extra["kind"] = "physics_model_proposal"
                    artifact = job.answer.get("generated_artifact")
                    if artifact:
                        extra["generated_artifact"] = artifact
                        _store_physics_equation(chat, artifact)
                        chat["settings"]["use_pinn"] = False
                        setup = chat.get("setup") or {}
                        if setup.get("stage") == "physics_model_drafting":
                            setup["stage"] = "physics_model_review"
                            chat["setup"] = setup
                        core.save_chat(chat)
                    elif (chat.get("setup") or {}).get("stage") == "physics_model_drafting":
                        chat["setup"]["stage"] = "physics_model_description"
                        physics_model_retry = True
                        core.save_chat(chat)
                if job.purpose == "data_review":
                    extra["kind"] = "data_review"
                if job.purpose == "question" and job.answer.get("status") == "clarification":
                    extra["kind"] = "clarification"
                    extra["asked_at"] = time.time()
                if chat.get("run_dir") and analysis.state_comparison_question(job.question):
                    comparison = analysis.compare_run_states(chat["run_dir"])
                    if comparison.get("rows"):
                        extra["state_comparison"] = comparison
                extra["response_seconds"] = _response_elapsed(job)
                extra["response_steps"] = job.client_activity_steps()
                saved = core.add_message(chat, "assistant", job.answer["answer"], **extra)
                if job.purpose == "model_sheet":
                    latest_user = next((message for message in reversed(chat.get("messages", []))
                                        if message.get("role") == "user"), None)
                    if (latest_user and
                            str(latest_user.get("content", "")).strip() != str(job.question).strip() and
                            agent.is_state_definition_question(latest_user.get("content", ""))):
                        followup = agent._interaction_answer(chat, latest_user["content"])
                        if followup:
                            core.add_message(
                                chat, "assistant", followup["answer"],
                                answer_scope="setup", model=followup.get("model", "LabCD"),
                            )
                if job.purpose in ("excel_maker", "pinn_maker", "physics_model_agent") and saved.get("generated_artifact"):
                    item_id = _generated_file_id(saved["generated_artifact"]["path"])
                    if any(item["id"] == item_id for item in _file_items(chat)):
                        st.session_state["conversation_files"] = True
                        st.session_state["conversation_selected_file"] = item_id
                if setup_followup:
                    if setup_followup.get("kind") == "plan":
                        core.add_message(chat, "assistant", setup_followup.pop("content"), **setup_followup)
                        _start_run_setup_agent(chat)
                    else:
                        core.add_message(chat, "assistant", setup_followup["content"],
                                         kind="setup_question", stage=setup_followup["stage"])
            else:
                if job.purpose == "model_sheet":
                    extra = {"kind": "starter_sheet", "starter_schema": model_sheet.default_schema(),
                             "starter_fallback": True, "answer_scope": "setup"}
                else:
                    extra = {"kind": "data_review"} if job.purpose == "data_review" else {}
                if job.purpose == "pinn_maker":
                    extra["kind"] = "pinn_maker"
                    if isinstance(chat.get("pinn_source"), dict):
                        chat["pinn_source"]["status"] = "error"
                        if (chat.get("setup") or {}).get("stage") == "physics_source_processing":
                            chat["setup"]["stage"] = "physics_source_upload"
                            physics_source_retry = True
                        core.save_chat(chat)
                if job.purpose == "physics_model_agent":
                    extra["kind"] = "physics_model_proposal"
                if ((chat.get("run_dir") or chat.get("dataset")) and
                        job.purpose not in ("excel_maker", "pinn_maker", "physics_model_agent")):
                    extra["visual_question"] = job.question
                if chat.get("run_dir") and analysis.state_comparison_question(job.question):
                    comparison = analysis.compare_run_states(chat["run_dir"])
                    if comparison.get("rows"):
                        extra["state_comparison"] = comparison
                extra["response_seconds"] = _response_elapsed(job)
                extra["response_steps"] = job.client_activity_steps()
                core.add_message(chat, "assistant", job.error or "The AI request did not finish.",
                                 **extra,
                                 **({"answer_scope": job.answer_scope} if getattr(job, "answer_scope", None) else {}),
                                 **({"error_detail": job.error_detail} if job.error_detail else {}))
                if (job.purpose == "physics_model_agent" and
                        (chat.get("setup") or {}).get("stage") == "physics_model_drafting"):
                    chat["setup"]["stage"] = "physics_model_description"
                    physics_model_retry = True
                    core.save_chat(chat)
            if physics_model_retry:
                core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question",
                                 stage="physics_model_description")
            if physics_source_retry:
                core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question",
                                 stage="physics_source_upload")
            if (job.purpose == "data_review" and
                    (chat.get("setup") or {}).get("stage") not in (None, "complete")):
                core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question",
                                 stage=chat["setup"]["stage"])
            registry.conversations.pop(identity, None)
            if (job.purpose == "pinn_maker" and job.answer and job.answer.get("status") == "ready" and
                    not chat.get("run_dir") and chat.get("setup") is None):
                _start_data_review(chat, AUTO_DATA_REVIEW_PROMPT)
    if job := registry.planning.get(identity):
        if not job.running:
            if getattr(job, "purpose", None) == "run_setup_recommendation":
                _save_setup_recommendation(chat, recommendation=job.answer,
                                           error=job.error, error_detail=job.error_detail,
                                           response_seconds=_response_elapsed(job))
            elif job.answer:
                previous_settings = dict(chat.get("settings") or {})
                updated_settings = dict(job.answer["settings"])
                settings_changed = updated_settings != previous_settings
                chat["settings"] = updated_settings
                show_plan = bool(job.answer.get("show_plan") or settings_changed)
                flow = chat.get("run_setup_flow") or {}
                dataset = chat.get("dataset") or {}
                setup_review_active = (
                    not chat.get("run_dir")
                    and flow.get("dataset_sha256") == dataset.get("sha256")
                    and any(item.get("kind") == "run_setup" for item in chat.get("messages", []))
                )
                if show_plan and setup_review_active:
                    # A chat setting edit should create a fresh review at the new
                    # message, even when it repeats the current value. The editor
                    # must also use the saved values when reopened.
                    setup_summary.clear_editor_widget_state(identity)
                    core.add_message(
                        chat, "assistant", job.answer["message"], kind="run_setup", answer_scope="setup",
                        response_seconds=_response_elapsed(job),
                        response_steps=["Read your message", "Checked the requested setup change",
                                        "Prepared the updated setup"],
                    )
                else:
                    core.add_message(chat, "assistant", job.answer["message"],
                                     answer_scope="run" if chat.get("run_dir") else "setup",
                                     response_seconds=_response_elapsed(job),
                                     response_steps=["Read your message", "Checked the current run settings",
                                                     "Prepared a reply"],
                                     **({"kind": "plan", "settings": chat["settings"].copy()} if show_plan else {}))
            else:
                core.add_message(chat, "assistant", job.error or "The planning request did not finish. Please retry.",
                                 answer_scope="run" if chat.get("run_dir") else "setup",
                                 response_seconds=_response_elapsed(job),
                                 response_steps=["Read your message", "Could not complete the setup update"],
                                 **({"error_detail": job.error_detail} if job.error_detail else {}))
            registry.planning.pop(identity, None)
    if job := registry.diagnostics.get(identity):
        if not job.running:
            if job.answer:
                core.add_message(chat, "assistant", job.answer["summary"], diagnosis=job.answer,
                                 answer_scope="run")
            else:
                detail = job.error or "The diagnosis could not finish. Please retry your question."
                if "unsupported_country_region_territory" in detail:
                    core.add_message(
                        chat, "assistant",
                        "LabCD's configured AI provider denied this diagnostic request from the region where LabCD is running. "
                        "The LabCD administrator needs to configure a provider that accepts requests from that region. "
                        "The run's saved results and activity are still available here.",
                        error_detail=detail,
                    )
                else:
                    core.add_message(chat, "assistant", detail)
            registry.diagnostics.pop(identity, None)
    if task := registry.training.get(identity):
        runner, state = task["runner"], task["state"]
        hooks["drain"](runner, state)
        if runner.run_dir and chat.get("run_dir") != str(runner.run_dir.resolve()):
            chat["run_dir"] = str(runner.run_dir.resolve())
            core.save_chat(chat)
        if not runner.running and not task.get("announced"):
            entry = runner.result.to_dict() if runner.result else dict(status="failed", message=runner.error or "The run stopped without a result.")
            core.add_message(chat, "assistant", _result_summary(entry),
                             **({"kind":"result", "run_dir":chat["run_dir"]} if chat.get("run_dir") else {}))
            task["announced"] = True

    # A Streamlit session can reload after the worker has written its results
    # but before the UI saves the completion message. Recover that announcement
    # from the durable manifest so the chat never gets stuck at "I'm starting".
    run_dir = chat.get("run_dir")
    training_task = registry.training.get(identity)
    worker_active = bool(training_task and training_task["runner"].running)
    has_result = any(message.get("kind") == "result" for message in chat.get("messages", []))
    if run_dir and not worker_active and not has_result:
        run_path = Path(run_dir)
        manifest_path = run_path / "run_manifest.json"
        manifest = read_json(manifest_path) or {}
        if manifest.get("status") in TERMINAL_RUN_STATUSES:
            core.add_message(chat, "assistant", _result_summary(manifest),
                             kind="result", run_dir=run_dir, recovered_from_manifest=True)
        else:
            # Older workers could end after Stop or an exception without writing
            # a terminal manifest. Recover from their diagnostic state, or make
            # the orphaned run explicit instead of leaving the chat at "starting".
            diagnostic = read_json(run_path / "diagnostic_state.json") or {}
            recovery = dict(manifest)
            recovery.update(
                status=(diagnostic.get("status")
                        if diagnostic.get("status") in TERMINAL_RUN_STATUSES else "failed"),
                run_dir=str(run_path),
                message=(diagnostic.get("message") or
                         "The run ended before its final result was saved. Any partial activity and cycle results are available below."),
                cycles_run=manifest.get("cycles_run", 0),
                performance_history=manifest.get("performance_history", []),
            )
            if run_path.is_dir():
                write_json(manifest_path, recovery)
            core.add_message(chat, "assistant", _result_summary(recovery),
                             kind="result", run_dir=run_dir, recovered_from_manifest=True)


def _busy(chat):
    identity, registry = chat["id"], core.REGISTRY
    return bool(identity in registry.conversations or identity in registry.planning or identity in registry.diagnostics or
                (identity in registry.training and registry.training[identity]["runner"].running))


def _start(chat, hooks):
    setup = chat.get("setup") or {}
    if setup.get("stage") not in (None, "complete"):
        core.add_message(chat, "assistant", "Please finish the data questions in the conversation before starting the run.")
        return
    dataset = chat.get("dataset") or {}
    flow = chat.get("run_setup_flow") or {}
    setup_preapproved = (
        flow.get("dataset_sha256") == dataset.get("sha256")
        and flow.get("stage") == "approved"
        and bool(flow.get("preapproved_initializer"))
    )
    if setup.get("stage") == "complete":
        approved = (flow.get("dataset_sha256") == dataset.get("sha256") and
                    flow.get("stage") == "approved")
        if not approved:
            if flow.get("dataset_sha256") != dataset.get("sha256"):
                _start_run_setup_agent(chat)
                core.add_message(chat, "assistant", "I’m preparing one setup review for this dataset. You can edit any setting with its pen, then approve the setup to start training.")
            else:
                core.add_message(chat, "assistant", "Review the setup card, edit any setting with its pen, then approve the setup once to start training.")
            return
    registry = core.REGISTRY
    with registry.lock:
        if registry.active_training():
            core.add_message(chat, "assistant", "Another run is using the training worker. You can keep discussing this plan and start it when that run finishes.")
            return
        previous_settings = dict(chat.get("settings") or {})
        try:
            options = core.make_options(chat)
        except ValueError as exc:
            core.add_message(chat, "assistant", str(exc))
            return
        runner = hooks["runner"](options)
        registry.training[chat["id"]] = dict(runner=runner, started=time.time(),
            state=dict(log="", stage="Preparing run", progress=0.0, history=[], result=None, critic=[], activity=[]))
        dataset = chat.get("dataset") or {}
        notices = []
        if dataset.get("rows", 0) < 50:
            notices.append(
                f"This dataset has {dataset.get('rows', 0)} samples, so the scores may be less reliable than with a larger recording."
            )
        changed = []
        current_settings = chat.get("settings") or {}
        for key, label in (("lstm_seq_length", "LSTM memory"), ("rollout_horizon", "rollout horizon")):
            before, after = previous_settings.get(key), current_settings.get(key)
            if before != after:
                changed.append(f"{label} {before} → {after} steps")
        if changed:
            notices.append("I shortened the model window to fit the available samples: " + "; ".join(changed) + ".")
        notice_text = (" " + " ".join(notices)) if notices else ""
        status = ("Your approved setup is in. I’m preparing the starting network and training now."
                  if setup_preapproved else
                  "I’m preparing the starting network. Review and approve it when it is ready, then training will continue.")
        core.add_message(chat, "assistant", status + notice_text)
        runner.start()


def _pasted_dataset(text, chat):
    parsed = excel_maker.parse_raw_table(text)
    if not parsed:
        return None
    headers, rows = parsed
    dataset = core.inspect_upload("pasted_data.csv", excel_maker.table_csv_bytes(headers, rows), chat["id"])
    dataset["source_kind"] = "pasted_table"
    chat["dataset"] = dataset
    if chat.get("title") == "New conversation" or chat["title"].startswith("k t (s)"):
        chat["title"] = "Pasted measurements"
    core.save_chat(chat)
    return dataset


def _recover_dataset_from_chat(chat):
    """Restore a pasted table from older chat history when the saved dataset is missing."""
    if chat.get("dataset"):
        return chat["dataset"]
    for message in reversed(chat.get("messages", [])):
        if message.get("role") != "user":
            continue
        attachment = message.get("attachment")
        if isinstance(attachment, dict) and attachment.get("path"):
            if Path(attachment["path"]).is_file():
                chat["dataset"] = attachment
                core.save_chat(chat)
                return attachment
            return None
        try:
            dataset = _pasted_dataset(str(message.get("content", "")), chat)
        except ValueError:
            return None
        if dataset:
            message["attachment"] = dataset
            core.save_chat(chat)
            return dataset
    return None


def _add_pasted_table_preview(chat, dataset):
    if dataset.get("source_kind") != "pasted_table":
        return
    if any(message.get("kind") == "pasted_table_preview" and
           (message.get("preview_dataset") or {}).get("sha256") == dataset.get("sha256")
           for message in chat.get("messages", [])):
        return
    info = {key: dataset[key] for key in ("path", "sha256", "name", "rows", "columns")}
    small_data_note = (
        f" This is a small dataset ({dataset['rows']:,} samples), so model scores may be less reliable, but it can still be used for a run."
        if dataset.get("warnings") else ""
    )
    core.add_message(
        chat, "assistant",
        f"I read your pasted table as {dataset['rows']:,} rows and {len(dataset['columns'])} columns. "
        "Here is a preview. I can make an Excel workbook from these same values, or use them to prepare a run. "
        "If you know what device or process produced the signals, or their units, you can tell me to help interpret them. That context is optional."
        + small_data_note,
        kind="pasted_table_preview", preview_dataset=info)


def _no_separate_output_reply(dataset):
    states = dataset.get("states") or []
    actions = dataset.get("actions") or []
    if states and actions:
        content = (f"Right. `{states[0]}` is the measured state and `{actions[0]}` is the input. "
                   "This run uses the measured state and does not need a separate output column.")
    else:
        content = "Understood. This run uses measured-state columns and inputs; it does not need a separate output column."
    if dataset.get("issues"):
        content += "\n\nThe current data check still needs attention:\n\n" + "\n".join(
            f"- {issue}" for issue in dataset["issues"])
    return content


def _is_optional_context_skip(question):
    text = re.sub(r"\s+", " ", str(question or "").casefold().replace("’", "'")).strip()
    return bool(re.search(
        r"\b(?:i\s+(?:do\s+not|don't|dont)\s+have|i\s+have\s+no|there\s+is\s+no|no|unknown|not\s+sure)\b"
        r".{0,55}\b(?:description|details|units|system|device|model|plant)\b|"
        r"^(?:skip|unknown|not sure|i don't know)$", text))


def _is_no_separate_output(question):
    text = str(question or "").casefold().replace("’", "'")
    return bool(re.search(r"\b(?:no|remove|drop|don't|do not|dont)\b.{0,45}\boutput\b", text))


def _context_skip_reply(chat):
    dataset = chat.get("dataset") or {}
    if dataset.get("column_review_pending"):
        return ("A system description and units are optional. "
                "First, confirm the time, state, and input columns in the choices above.")
    if dataset.get("issues"):
        details = "\n".join(f"- {issue}" for issue in dataset["issues"])
        return ("A system description and units are optional. I can work from the measurements. "
                "The data still needs these checks before a run:\n\n" + details)
    return ("A system description and units are optional. I can work from the measurements. "
            "If you later know what device or process produced these measurements, or the units, tell me and I'll use that context when explaining the results. Say `start` when you're ready.")


def _is_simple_greeting(question):
    text = re.sub(r"[^a-z0-9]+", " ", str(question or "").casefold()).strip()
    return text in {
        "hi", "hello", "hey", "hiya", "hi labcd", "hello labcd", "hey labcd",
        "salam", "good morning", "good afternoon", "good evening",
    }


def _is_system_description(question):
    text = str(question or "").strip().casefold()
    if not text or re.match(r"^(?:what|why|how|can|could|should|would|is|are|do|does|explain|tell me)\b", text):
        return False
    return bool(re.search(
        r"\b(?:want to model|trying to model|model my|identify my|identify a system|"
        r"my system|the system is|system is|i measure|i control|we measure|we control|"
        r"pendulum|motor|water tank|tank system|helicopter|drone|robot|cart-pole|"
        r"battery|engine|autonomous car|autonomous vehicle|self[- ]driving (?:car|vehicle)|"
        r"vehicle|car|pump|actuator|thermal|temperature process|fluid system)\b",
        text,
    ))


def _is_system_question(question):
    text = str(question or "").strip().casefold()
    return "?" in text or bool(re.match(
        r"^(?:what|why|how|can|could|should|would|is|are|do|does|explain|tell me)\b", text))


def _is_model_confirmation(question):
    text = re.sub(r"\s+", " ", str(question or "").casefold()).strip(" .!?")
    return bool(re.fullmatch(r"(?:this|that|it) is (?:my|the) (?:model|system)", text))


def _is_unsure_about_system(question):
    text = re.sub(r"\s+", " ", str(question or "").casefold().replace("’", "'")).strip(" .!?")
    return bool(re.fullmatch(
        r"(?:i (?:do not|don't|dont) know|not sure|no idea|i cannot describe it|"
        r"i can't describe it|i dont know what it is)", text))


def _submit(chat, question, files, hooks):
    question, attached, pinn_upload = (question or "").strip(), None, None
    physics_description_prefix = "__labcd_physics_description__:"
    physics_description_request = question.startswith(physics_description_prefix)
    if physics_description_request:
        question = question[len(physics_description_prefix):].strip()
    setup_at_submit = chat.get("setup") or {}
    setup_stage_at_submit = setup_at_submit.get("stage")
    explicit_source_choice = bool(re.search(
        r"\b(?:equations?|source|upload|attach|paste|provide|send|file|python|matlab)\b|\.(?:py|m|txt)\b",
        question, re.I,
    ))
    is_model_description_reply = (
        setup_stage_at_submit == "physics_model_description" and question and not files
        and not physics_description_request and not _is_system_question(question)
    )
    is_system_description_at_source_choice = (
        setup_stage_at_submit in ("physics_source_choice", "physics_source_upload")
        and question and not files and not physics_description_request
        and not explicit_source_choice and _is_system_description(question)
        and not _is_system_question(question)
    )
    if is_model_description_reply or is_system_description_at_source_choice:
        # In the PINN wizard, a plant name/description is enough to start
        # LabCD's draft path. Do not send it to the equation-file uploader.
        physics_description_request = True
    command = question.lower().strip(" .!?")
    first_client_message = not any(message.get("role") == "user" for message in chat.get("messages", []))
    awaiting_system_description = chat.get("first_step_stage") == "awaiting_system_description"
    start_commands = ("start", "start run", "run", "run it", "go", "go ahead", "start training", "yes start", "start the run")
    current_task = core.REGISTRY.training.get(chat["id"])
    if files and current_task and current_task["runner"].running:
        core.add_message(chat, "assistant", "A run is already using this attachment. Open a new conversation to work on another dataset.")
        return
    if files:
        if chat.get("run_dir"):
            core.add_message(chat, "assistant", "This conversation is linked to a run. Start a new conversation to train on another dataset.")
            return
        try:
            uploaded = files[0]
            if Path(uploaded.name).suffix.lower() in (".m", ".py", ".txt"):
                pinn_upload = pinn_maker.save_source(chat["id"], uploaded.name, uploaded.getvalue())
                pinn_upload["chat_id"] = chat["id"]
                chat["pinn_source"] = pinn_upload
                chat["pinn_equation"] = None
                chat["settings"]["use_pinn"] = False
            else:
                attached = core.inspect_upload(uploaded.name, uploaded.getvalue(), chat["id"])
                chat["dataset"] = attached
                if not question and chat["title"] == "New conversation":
                    chat["title"] = Path(attached["name"]).stem.replace("_", " ")[:100] or "Dataset conversation"
        except ValueError as exc:
            core.add_message(chat, "user", question or "Upload dataset")
            core.add_message(chat, "assistant", str(exc))
            return
    pending_pinn_source = chat.get("pinn_source") or {}
    auto_prepare_for_dataset = bool(
        attached and attached.get("ready") and
        pending_pinn_source.get("status") in ("pending", "ready") and
        not pinn_maker.is_equation_ready(chat))
    column_roles_need_confirmation = bool(
        (chat.get("dataset") or {}).get("column_review_pending"))
    pinn_usage_question = agent.is_pinn_usage_question(question)
    setup_choice_stage = setup_stage_at_submit in ("pinn_interest", "pcnn_interest")
    pinn_requested = (
        not physics_description_request
        and (
            bool(pinn_upload)
            or auto_prepare_for_dataset
            or (
                not setup_choice_stage
                and not pinn_usage_question
                and not column_roles_need_confirmation
                and pinn_maker.is_raw_equation_request(
                    question, has_pending_source=pending_pinn_source.get("status", False)
                )
            )
        )
    )
    if pinn_requested and question and not pinn_upload and not chat.get("pinn_source"):
        try:
            source = pinn_maker.save_source(chat["id"], "pasted_physics_equations.txt", question.encode("utf-8"))
            source["chat_id"] = chat["id"]
            chat["pinn_source"] = source
            chat["pinn_equation"] = None
            chat["settings"]["use_pinn"] = False
        except ValueError as exc:
            core.add_message(chat, "user", question)
            core.add_message(chat, "assistant", str(exc))
            return
    elif question and not pinn_requested and not files and not physics_description_request:
        try:
            attached = _pasted_dataset(question, chat)
        except ValueError as exc:
            core.add_message(chat, "user", question)
            core.add_message(chat, "assistant", str(exc))
            return
        if (attached and attached.get("ready") and
                pending_pinn_source.get("status") in ("pending", "ready") and
                not pinn_maker.is_equation_ready(chat)):
            auto_prepare_for_dataset = True
            pinn_requested = True
    message_attachment = pinn_upload or (chat.get("pinn_source") if pinn_requested and not attached else attached)
    if (not attached and not chat.get("dataset") and not files and
            not excel_maker.is_excel_request(question)):
        role_reply = bool(re.search(
            r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\b.{0,35}\b(?:time|state|input|action|output|remove|drop|ignore)\b|"
            r"\b(?:no|remove|drop|ignore)\b.{0,30}\boutput\b", question, re.I))
        if command in start_commands or role_reply or _is_optional_context_skip(question):
            attached = _recover_dataset_from_chat(chat)
    source_mode = str((setup_at_submit.get("pending_physics_mode") or
                       (chat.get("settings") or {}).get("physics_mode") or "pinn")).lower()
    default_message = ((f"I've attached the PCNN hard-constraint source {pinn_upload['name']}." if source_mode == "pcnn" else
                        f"I've attached the physics source {pinn_upload['name']}.") if pinn_upload
                       else "I've attached my dataset.")
    core.add_message(chat, "user", question or default_message,
                     **({"attachment": message_attachment} if message_attachment else {}),
                     **({"kind": "physics_model_request", "answer_scope": "setup"}
                        if physics_description_request else {}))
    if physics_description_request:
        description = question.strip()
        if len(description) < 12:
            core.add_message(chat, "assistant", (
                "Please describe the exact state-derivative rule PCNN should enforce and name the state it applies to."
                if source_mode == "pcnn" else
                "Please add a little more detail about the system, states, inputs, or known physical relationships before I draft the PINN model."
            ))
            setup_at_submit["stage"] = "physics_model_description"
            chat["setup"] = setup_at_submit
            core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question", stage="physics_model_description")
            core.save_chat(chat)
            return
        chat["pinn_model_description"] = description[:4000]
        setup_at_submit["stage"] = "physics_model_drafting"
        chat["setup"] = setup_at_submit
        core.save_chat(chat)
        job = pinn_maker.PhysicsModelAgentJob(chat, description[:4000])
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return
    if pinn_requested and (pinn_upload or chat.get("pinn_source")):
        if setup_stage_at_submit == "pinn_interest":
            setup_at_submit["pending_physics_mode"] = "pinn"
        elif setup_stage_at_submit == "pcnn_interest":
            setup_at_submit["pending_physics_mode"] = "pcnn"
        if setup_stage_at_submit in (
            "pinn_interest", "pcnn_interest", "physics_source_choice", "physics_source_upload"
        ):
            setup_at_submit["stage"] = "physics_source_processing"
        chat["setup"] = setup_at_submit
        core.save_chat(chat)
    if _is_system_description(question) and not _is_system_question(question):
        chat["pinn_model_description"] = question[:4000]
    if not chat.get("run_dir"):
        setup_flow = chat.get("run_setup_flow") or {}
        setup_dataset = chat.get("dataset") or {}
        if (setup_flow.get("dataset_sha256") == setup_dataset.get("sha256")
                and setup_flow.get("stage") != "approved"):
            latest_setup = next(
                (item for item in reversed(chat.get("messages") or [])
                 if item.get("kind") == "run_setup"),
                None,
            )
            if latest_setup is not None:
                _stop_starting_setup_countdown(chat, latest_setup.get("id"))
    dataset = chat.get("dataset") or {}
    if dataset.get("source_kind") == "pasted_table" and not excel_maker.is_excel_request(question):
        _add_pasted_table_preview(chat, dataset)
    if command in ("stop", "stop run", "cancel run", "stop training"):
        task = core.REGISTRY.training.get(chat["id"])
        if task and task["runner"].running:
            task["runner"].request_stop()
            core.add_message(chat, "assistant", "Stop requested. LabCD will finish the current training epoch, score that candidate, and package the best result without starting another candidate.")
        else:
            core.add_message(chat, "assistant", "There is no active training run in this conversation.")
        return
    if not files and excel_maker.is_excel_help_request(question):
        core.add_message(
            chat, "assistant",
            "Paste your header row and data rows in chat, or upload a CSV/Excel file. Then ask me to create an Excel workbook. "
            "I’ll keep your supplied values intact. Here’s a small example of how rows and columns can look:",
            kind="excel_help",
        )
        return
    if excel_maker.is_excel_request(question):
        job = excel_maker.ExcelMakerJob(chat, question)
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return
    if not files and pinn_usage_question:
        core.add_message(chat, "assistant", agent.pinn_usage_answer(chat), kind="pinn_status",
                         answer_scope="run" if (chat.get("run_dir") or chat.get("live_run_dir")) else "setup")
        return
    if attached and attached.get("column_review_pending"):
        core.add_message(chat, "assistant", _column_review_prompt(attached), kind="column_review")
        return

    if (not attached and not chat.get("dataset") and not files and not chat.get("pinn_source")
            and not pinn_usage_question and not excel_maker.is_excel_request(question)):
        if agent.is_state_definition_question(question) or _is_model_confirmation(question):
            previous_description = next((str(message.get("content", "")).strip()
                                         for message in reversed(chat.get("messages", [])[:-1])
                                         if message.get("role") == "user" and
                                         _is_system_description(message.get("content", ""))), None)
            has_suggested_sheet = any(message.get("kind") == "starter_sheet"
                                      for message in chat.get("messages", []))
            running_job = core.REGISTRY.conversations.get(chat["id"])
            sheet_already_running = getattr(running_job, "purpose", None) == "model_sheet"
            if previous_description and not has_suggested_sheet:
                if not sheet_already_running:
                    _start_model_sheet(chat, previous_description)
                return
        # A client may describe the system after a greeting or another chat turn.
        # Do not limit model-sheet creation to the very first user message.
        if _is_system_description(question) and not _is_system_question(question):
            _start_model_sheet(chat, question)
            return
        if awaiting_system_description:
            if _is_simple_greeting(question):
                core.add_message(
                    chat, "assistant",
                    _conversation_copy(chat, SYSTEM_GREETING_COPIES, "system_greeting_count"),
                )
                return
            if _is_unsure_about_system(question):
                core.add_message(
                    chat, "assistant",
                    "That’s okay. What device or process produced the signals you want to study? "
                    "Even a rough description will help me suggest the measurements.",
                )
                return
            if not _is_system_question(question):
                _start_model_sheet(chat, question)
                return
        elif first_client_message and _is_simple_greeting(question):
            chat["first_step_stage"] = "awaiting_system_description"
            core.add_message(
                chat, "assistant",
                _conversation_copy(chat, SYSTEM_GREETING_COPIES, "system_greeting_count"),
            )
            return

    dataset = chat.get("dataset") or {}
    if (not attached and dataset.get("path") and
            (not dataset.get("states") or "time" not in dataset.get("columns", [])) and
            not dataset.get("column_review_pending")):
        # Older saved chats may already contain a dataset that was rejected
        # only because its measured-state header lacked the s_ prefix.
        try:
            refreshed = core.inspect_upload(dataset.get("name") or Path(dataset["path"]).name,
                                            Path(dataset["path"]).read_bytes(), chat["id"])
        except (OSError, ValueError):
            refreshed = None
        if refreshed and refreshed.get("column_review_pending"):
            old_digest = dataset.get("sha256")
            chat["dataset"] = refreshed
            for message in chat.get("messages", []):
                attachment = message.get("attachment")
                if isinstance(attachment, dict) and attachment.get("sha256") == old_digest:
                    message["attachment"] = refreshed
            core.save_chat(chat)
            core.add_message(chat, "assistant", _column_review_prompt(refreshed), kind="column_review")

    dataset = chat.get("dataset") or {}
    if dataset.get("column_review_pending"):
        if pinn_requested:
            core.add_message(chat, "assistant", (
                "I saved your PCNN constraint source. Confirm the time, state, and input columns above; I’ll map the rules after that."
                if source_mode == "pcnn" else
                "I saved your physics source. Confirm the time, state, and input columns above; I’ll map the model after that."
            ))
            return
        if command in start_commands:
            core.add_message(chat, "assistant", "I found your table. First confirm the time, measured-state, and input columns above. Then I can check whether it is ready to run.")
            return
        roles = core.column_roles_from_answer(dataset, question)
        if roles is None:
            core.add_message(chat, "assistant",
                             "I still need to confirm which columns are time, measured state, and input. Use the selectors above or tell me the column names or positions.")
            return
        time_column, state_columns, action_columns = roles
        _finish_column_review(chat, state_columns, action_columns, time_column=time_column)
        return
    if pinn_requested:
        if not dataset.get("ready"):
            if dataset.get("issues"):
                core.add_message(chat, "assistant", (
                    "I saved your PCNN constraint source. First fix the dataset checks; I’ll map the rules once the measurements and roles are ready."
                    if source_mode == "pcnn" else
                    "I saved your physics source. First fix the dataset checks; I’ll map the model once the measurements and roles are ready."
                ))
            else:
                core.add_message(chat, "assistant", (
                    "I saved your PCNN constraint source. Attach the measurements and confirm the time, state, and input columns; I’ll map its rules when they’re ready."
                    if source_mode == "pcnn" else
                    "I saved your physics source. Attach the measurements and confirm the time, state, and input columns; I’ll map its model when they’re ready."
                ))
            return
        if not pinn_maker.is_equation_ready(chat):
            prompt = (f"Map {chat['pinn_source']['name']} using the confirmed state and input columns."
                      if auto_prepare_for_dataset else
                      question or f"Map {chat['pinn_source']['name']} using the confirmed state and input columns.")
            if _start_pinn_preparation(chat, prompt):
                return
        core.add_message(chat, "assistant", "Your saved hard-constraint rules are already mapped to these columns." if source_mode == "pcnn" and pinn_maker.is_equation_ready(chat)
                         else "Your saved physics model is already mapped to these columns." if pinn_maker.is_equation_ready(chat)
                         else "I couldn’t start physics-source mapping while another response is being processed. Please retry in a moment.")
        return
    if _is_no_separate_output(question) and dataset:
        core.add_message(chat, "assistant", _no_separate_output_reply(dataset))
        return
    if _is_optional_context_skip(question):
        content = _context_skip_reply(chat)
        if dataset.get("ready") and (chat.get("setup") or {}).get("stage") in (None, "complete"):
            core.add_message(chat, "assistant", content, kind="plan", settings=chat["settings"].copy())
        else:
            core.add_message(chat, "assistant", content)
        return
    if attached and attached["issues"]:
        core.add_message(chat, "assistant", "I checked the attachment. A few things need sorting out before it can be used for a test:\n\n" +
                         "\n".join(f"- {issue}" for issue in attached["issues"]) +
                         "\n\nWe can work through these together; the original upload stays as it is.")
        return
    if attached and attached["ready"]:
        prompt = question or AUTO_DATA_REVIEW_PROMPT
        _start_data_review(chat, prompt)
        return
    if command in start_commands:
        if not dataset:
            message = ("I couldn't find a usable table in this conversation yet. If you'd like to try a run, paste a table "
                       "here or attach a CSV or Excel file whenever you're ready. A time column and one measured state "
                       "are a good start; include an input column if you recorded one. Unclear headers are fine - we'll "
                       "identify the roles together. A system description isn't needed.")
            core.add_message(chat, "assistant", message)
        elif dataset.get("issues"):
            details = "\n".join(f"- {issue}" for issue in dataset["issues"])
            core.add_message(chat, "assistant", "I found the dataset, but it still needs these checks before a run:\n\n" + details)
        else:
            _start(chat, hooks)
        return
    if agent.is_identity_question(question):
        job = agent.ConversationJob(chat, question)
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return
    # A client can edit setup values while the assumptions wizard or its setup
    # recommendation is visible. Route these before the wizard consumes the
    # message as an answer to its current question.
    if not chat.get("run_dir") and _chat_signal_role_edit(chat, question):
        return
    if not chat.get("run_dir") and core.is_settings_edit_request(question):
        job = core.PlanningJob(chat, question)
        core.REGISTRY.planning[chat["id"]] = job
        job.start()
        return
    if chat.get("setup") and chat["setup"].get("stage") != "complete":
        reply = core.advance_setup(chat, question)
        if reply:
            assumptions_complete = reply.get("kind") == "plan" and chat["setup"].get("stage") == "complete"
            core.add_message(chat, "assistant", reply.pop("content"), **reply)
            if assumptions_complete:
                _start_run_setup_agent(chat)
            return
        job = agent.ConversationJob(chat, question, purpose="setup_explain")
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return
    if not chat.get("run_dir") and command in ("show plan", "plan", "show my plan"):
        content = (f"I've checked {attached['name']}: {attached['rows']:,} samples, {_count(len(attached['states']), 'state')} and {_count(len(attached['actions']), 'input')}. " if attached else "Here is the current run plan. ")
        content += "You can review or adjust the setup below. If you happen to know what produced the signals or their units, that context can help interpret the results, but it isn’t needed. Start whenever the plan looks right to you."
        core.add_message(chat, "assistant", content, kind="plan", settings=chat["settings"].copy())
        return
    if chat.get("run_dir"):
        task = core.REGISTRY.training.get(chat["id"])
        if task and task["runner"].running:
            core.add_message(chat, "assistant", "The run is still working. The activity below shows its progress. Say 'stop' to finish early; we can investigate results when it finishes.")
            return
        job = agent.ConversationJob(chat, question or "Explain this run")
        core.REGISTRY.conversations[chat["id"]] = job
    else:
        model_concept = bool(re.search(r"\b(lstm|mlp|pinn|model|architecture|network)\b", question, re.I) and
                             re.search(
                                 r"\b(explain|difference|compare|what is|how does|which|why|can|could|should|would|better|fit|suit|work)\b",
                                 question, re.I))
        settings_request = core.is_settings_edit_request(question) or bool(
            re.search(r"\b(change|edit|modify|adjust|update|set|increase|decrease|raise|lower|switch|configure|choose|select|make|put)\b", question, re.I)
            or re.match(r"^\s*(?:please\s+)?(?:use|run)\b", question, re.I)
        )
        dataset_attached = bool(chat.get("dataset"))
        # Keep ordinary no-dataset conversation out of the configuration planner.
        if (model_concept or
                (not dataset_attached and not settings_request) or
                (dataset_attached and not re.search(
                    r"\b(use|change|set|increase|decrease|switch|configure|plan|recommend)\b", question, re.I
                ))):
            job = agent.ConversationJob(chat, question)
            core.REGISTRY.conversations[chat["id"]] = job
        else:
            job = core.PlanningJob(chat, question or "Help me prepare this dataset for system identification.")
            core.REGISTRY.planning[chat["id"]] = job
    job.start()


def _respond_to_human_checkpoint(task, action, edits=None):
    runner = task["runner"]
    accepted = (runner.respond_to_checkpoint(action) if edits is None else
                runner.respond_to_checkpoint(action, edits=edits))
    if accepted:
        st.rerun()


def _initializer_analysis_rows(checkpoint):
    """Create a compact, data-grounded table for reviewing the first proposal."""
    config = checkpoint.get("config") or {}
    dataset = checkpoint.get("dataset") or {}
    setup = checkpoint.get("setup") or {}
    architecture = str(setup.get("architecture", "LSTM")).upper()
    run_mode = str(setup.get("run_mode", "fast")).lower()
    run_limits = cfg.run_mode_limits(run_mode)
    cycles = int(setup.get("max_cycles", run_limits["max_cycles"]))
    samples = int(dataset.get("samples", 0) or 0)
    train_samples = int(dataset.get("train_samples", 0) or 0)
    validation_samples = int(dataset.get("validation_samples", 0) or 0)
    test_samples = int(dataset.get("test_samples", 0) or 0)
    states = list(dataset.get("states") or [])
    inputs = list(dataset.get("inputs") or [])
    dt = float(dataset.get("median_dt", 0.0) or 0.0)
    history_steps = int(setup.get("history_steps", 0) or 0)
    history_seconds = setup.get("history_seconds")
    if history_seconds is None and architecture == "LSTM" and dt > 0 and history_steps > 1:
        history_seconds = (history_steps - 1) * dt

    layer_widths = [int(width) for width in (config.get("hidden_layers") or [])]
    if architecture == "LSTM":
        topology = (
            f"{len(layer_widths)} LSTM layers × {layer_widths[0]} units"
            if layer_widths else "LSTM topology unavailable"
        )
        context_value = f"LSTM · {history_steps} past steps"
        if history_seconds is not None:
            context_value += f" · {float(history_seconds):.4g} s of history"
        context_read = (
            f"The window covers {max(history_steps - 1, 0)} measured intervals. "
            "Sequence memory is a hypothesis to test, not proof that hidden memory is required."
        )
    else:
        topology = "MLP · " + (" → ".join(map(str, layer_widths)) if layer_widths else "topology unavailable")
        context_value = "MLP · current state and input"
        context_read = "The model maps each measured state/input snapshot directly; no history window is used."

    parameter_count = setup.get("estimated_parameters")
    if parameter_count is not None:
        capacity_value = f"{topology} · {int(parameter_count):,} trainable parameters"
    else:
        capacity_value = topology
    capacity_read = (
        f"The training split contains {train_samples:,} rows. Parameter count is a size indicator, "
        "not a direct overfitting test; compare training and validation errors, then use the held-out test."
    )

    lr = config.get("learning_rate")
    dropout = config.get("dropout_rate")
    weight_decay = config.get("weight_decay")
    optimizer_value = (
        f"LR {float(lr):.3g}" if lr is not None else "Learning rate unavailable"
    )
    if dropout is not None:
        optimizer_value += f" · dropout {float(dropout):.2f}"
    if weight_decay is not None:
        optimizer_value += f" · weight decay {float(weight_decay):.3g}"
    lr_min, lr_max = config.get("lr_search_min"), config.get("lr_search_max")
    if lr_min is not None and lr_max is not None:
        optimizer_read = f"Learning rate search spans {float(lr_min):.3g}–{float(lr_max):.3g}; regularization is a tunable tradeoff, not a guarantee."
    else:
        optimizer_read = "Validation results will show whether the starting optimization and regularization settings need adjustment."

    split = f"{train_samples:,} train · {validation_samples:,} validation · {test_samples:,} test"
    split_read = (
        f"{samples:,} rows · {len(states)} states · {len(inputs)} inputs. "
        "The held-out test segment stays out of candidate selection for the final check."
    )
    varying_states = dataset.get("varying_states")
    varying_inputs = dataset.get("varying_inputs")
    complexity = str(dataset.get("complexity", "Unknown"))
    tier = dataset.get("complexity_tier")
    complexity_label = f"{complexity} · tier {tier}/5" if tier else complexity
    if varying_inputs is None:
        signal_value = "Signal variation not summarized"
    else:
        signal_value = f"{len(varying_states or [])}/{len(states)} states · {len(varying_inputs)}/{len(inputs)} inputs vary"
    derivative_count = len(dataset.get("derivatives") or [])
    derivative_note = (
        f"{derivative_count} supplied derivative columns"
        if derivative_count else "derivatives will be estimated from measured states"
    )
    signal_read = (
        f"{complexity_label}; {derivative_note}. Input variation alone does not prove sufficient excitation."
    )
    max_hours = float(setup.get("max_hours", run_limits["max_hours"]) or 0.0)
    time_budget = f"{max_hours * 60:g} min time cap" if max_hours < 1 else f"{max_hours:g} h time cap"
    budget_value = f"{run_mode.title()} · up to {cycles} candidate cycles · {time_budget}"
    budget_read = "Each cycle evaluates one candidate. You can lower the cycle count before starting; the time cap can stop it sooner."

    return [
        {"Review area": "Search budget", "Initializer proposal": budget_value, "Read against the data": budget_read},
        {"Review area": "Model context", "Initializer proposal": context_value, "Read against the data": context_read},
        {"Review area": "Network capacity", "Initializer proposal": capacity_value, "Read against the data": capacity_read},
        {"Review area": "Optimization", "Initializer proposal": optimizer_value, "Read against the data": optimizer_read},
        {"Review area": "Dataset and split", "Initializer proposal": split, "Read against the data": split_read},
        {"Review area": "Signal quality", "Initializer proposal": signal_value, "Read against the data": signal_read},
    ]


def _initializer_detail_rows(checkpoint):
    config = checkpoint.get("config") or {}
    details = []
    learning_rate = config.get("learning_rate")
    if learning_rate is not None:
        details.append({"Training detail": "Starting learning rate", "Value": f"{float(learning_rate):.3g}"})
    for label, low_key, high_key in (
        ("Learning-rate search", "lr_search_min", "lr_search_max"),
        ("Hidden-width search", "hidden_size_search_min", "hidden_size_search_max"),
        ("Layer-count search", "num_layers_search_min", "num_layers_search_max"),
    ):
        low, high = config.get(low_key), config.get(high_key)
        if low is not None and high is not None:
            value = f"{float(low):.3g}–{float(high):.3g}" if "lr" in low_key else f"{int(low)}–{int(high)}"
            details.append({"Training detail": label, "Value": value})
    for label, key in (
        ("Epoch limit per candidate", "epochs"),
        ("Batch size", "batch_size"),
        ("Early-stop patience", "early_stop_patience"),
        ("Activation", "activation"),
        ("Dropout", "dropout_rate"),
        ("Weight decay", "weight_decay"),
    ):
        if config.get(key) is not None:
            value = config[key]
            if key == "dropout_rate":
                value = f"{float(value):.2f}"
            elif key == "weight_decay":
                value = f"{float(value):.3g}"
            details.append({"Training detail": label, "Value": str(value)})
    return details


def _finite_number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _next_candidate_editor(chat, task, checkpoint):
    """Edit only parameters the Actor will use on its next candidate."""
    checkpoint_id = str(checkpoint.get("id", "cycle_review"))
    prefix = f"next_candidate_{chat['id']}_{checkpoint_id}"
    candidate = checkpoint.get("next_config") or {}
    bounds = checkpoint.get("candidate_bounds") or {}
    lr_min = float(bounds.get("learning_rate_min", cfg.LEARNING_RATE_MIN))
    lr_max = float(bounds.get("learning_rate_max", cfg.LEARNING_RATE_MAX))
    width_min = int(bounds.get("hidden_size_min", cfg.HIDDEN_SIZE_MIN))
    width_max = int(bounds.get("hidden_size_max", cfg.HIDDEN_SIZE_MAX))
    layer_min = int(bounds.get("num_layers_min", cfg.NUM_LAYERS_MIN))
    layer_max = int(bounds.get("num_layers_max", cfg.NUM_LAYERS_MAX))
    existing_widths = candidate.get("hidden_layers") or [64]
    initial_layers = min(max(len(existing_widths), layer_min), layer_max)
    st.markdown("#### Optional · change next-cycle parameters")
    st.caption(
        "These edits affect only the next candidate. Leave them as shown to keep the agents’ proposal; apply your own values when you’re ready."
    )
    with st.form(f"{prefix}_form", border=True):
        if layer_min == layer_max:
            layer_count = layer_min
            st.caption(f"Hidden layers · fixed at {layer_count}")
        else:
            layer_count = st.slider(
                "Hidden layers", min_value=layer_min, max_value=layer_max,
                value=initial_layers, key=f"{prefix}_layer_count",
            )
        width_columns = st.columns(min(layer_count, 3), gap="small")
        widths = []
        for index in range(layer_count):
            start = existing_widths[min(index, len(existing_widths) - 1)]
            value = min(max(int(start), width_min), width_max)
            with width_columns[index % len(width_columns)]:
                if width_min == width_max:
                    st.caption(f"Layer {index + 1} · {width_min} units")
                    widths.append(width_min)
                else:
                    widths.append(int(st.number_input(
                        f"Layer {index + 1} units", min_value=width_min,
                        max_value=width_max, value=value, step=8,
                        key=f"{prefix}_width_{index}",
                    )))

        left, right = st.columns(2, gap="medium")
        current_lr = _finite_number(candidate.get("learning_rate"), 0.001)
        current_lr = min(max(current_lr, lr_min), lr_max)
        with left:
            if lr_min == lr_max:
                st.caption(f"Learning rate · fixed at {lr_min:.5g}")
                learning_rate = lr_min
            else:
                learning_rate = st.number_input(
                    "Learning rate", min_value=lr_min, max_value=lr_max,
                    value=current_lr, format="%.7f", step=max((lr_max - lr_min) / 100, 1e-7),
                    key=f"{prefix}_learning_rate",
                )
            dropout = st.slider(
                "Dropout", min_value=0.0, max_value=float(cfg.DROPOUT_RATE_MAX),
                value=min(max(_finite_number(candidate.get("dropout_rate"), 0.0), 0.0), float(cfg.DROPOUT_RATE_MAX)),
                step=0.01, key=f"{prefix}_dropout",
            )
            batch_size = st.number_input(
                "Batch size", min_value=8, max_value=4096,
                value=min(max(int(candidate.get("batch_size", cfg.BATCH_SIZE)), 8), 4096),
                step=8, key=f"{prefix}_batch_size",
            )
        with right:
            weight_decay = st.number_input(
                "Weight decay", min_value=0.0, max_value=float(cfg.WEIGHT_DECAY_MAX),
                value=min(max(_finite_number(candidate.get("weight_decay"), 0.0), 0.0), float(cfg.WEIGHT_DECAY_MAX)),
                step=0.0001, format="%.6f", key=f"{prefix}_weight_decay",
            )
            patience = st.number_input(
                "Early-stop patience", min_value=1, max_value=1000,
                value=min(max(int(candidate.get("patience", cfg.EARLY_STOP_PATIENCE)), 1), 1000),
                step=1, key=f"{prefix}_patience",
            )
            epochs = st.number_input(
                "Epoch limit for next cycle", min_value=1, max_value=5000,
                value=min(max(int(candidate.get("epochs", cfg.EPOCHS)), 1), 5000),
                step=1, key=f"{prefix}_epochs",
                help="Applies to the next candidate only. A cycle already training cannot be changed midway.",
            )
            st.caption("These controls shape the next candidate; they cannot change a cycle already training.")
        submitted = st.form_submit_button(
            "Apply these parameters to the next cycle", type="primary",
            icon=":material/tune:", width="stretch", key=f"{prefix}_submit",
        )
    if submitted:
        edits = {
            "hidden_layers": widths,
            "learning_rate": float(learning_rate),
            "dropout_rate": float(dropout),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "epochs": int(epochs),
        }
        queued = chat.get("queued_live_parameter_edit")
        if isinstance(queued, dict) and int(queued.get("cycle", 0) or 0) == cycle + 1:
            runner = task.get("runner") if isinstance(task, dict) else None
            cancel = getattr(runner, "cancel_candidate_edits", None)
            if callable(cancel):
                cancel(cycle + 1)
            chat.pop("queued_live_parameter_edit", None)
        _dismiss_pending_question_timer(chat)
        _respond_to_human_checkpoint(task, "custom", edits=edits)


def _live_agent_context(state, checkpoint, cycle, queued_parameter_edit=None):
    """Keep the agent's live context small, explicit, and grounded in this run."""
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    payload = {
        "stage": state.get("stage"),
        "cycle": cycle,
        "checkpoint_phase": checkpoint.get("phase"),
        "parameter_controls": {
            "chat_applies_changes": False,
            "interface_applies_explicit_chat_edits": True,
            "in_progress_candidate_mutable": False,
            "explicit_chat_edits_apply_before_target_cycle_starts": True,
            "already_started_cycle_editable": False,
            "unspecified_edit_target": "next_candidate_that_has_not_started",
            "ask_client_to_choose_current_or_next_cycle": False,
            "next_candidate_editor_available": checkpoint.get("phase") == "early_results",
            "next_candidate_parameters": [
                "hidden layers", "neurons per layer", "learning rate", "dropout", "weight decay",
                "batch size", "early-stop patience", "epoch limit",
            ],
            "chat_editable_parameters": [
                "hidden layers", "neurons per layer", "learning rate", "dropout",
                "weight decay", "batch size", "early-stop patience", "epoch limit",
            ],
            "explicit_chat_edits_require_client_apply": False,
            "next_candidate_changes_require_client_apply": True,
        },
    }
    for key in (
        "max_cycles", "recent_results", "best_mse", "best_config", "next_config",
        "candidate_bounds", "state_validation_mse", "config", "setup",
    ):
        if checkpoint.get(key) is not None:
            payload[key] = checkpoint[key]
    if isinstance(queued_parameter_edit, dict):
        payload["queued_candidate_edit"] = {
            "cycle": queued_parameter_edit.get("cycle"),
            "edits": dict(queued_parameter_edit.get("edits") or {}),
            "status": "queued_until_cycle_start",
        }
    dataset = checkpoint.get("dataset")
    if isinstance(dataset, dict):
        payload["dataset"] = {
            key: dataset[key] for key in (
                "samples", "train_samples", "validation_samples", "test_samples",
                "states", "inputs", "median_dt",
            ) if key in dataset
        }
    payload["recent_agent_activity"] = []
    activity_detail_keys = (
        "outcome", "response", "mode", "config", "cycle", "val_mse", "train_mse",
        "diagnosis", "status", "lr_dir", "lr_step", "hidden_layers", "reasoning",
        "findings", "message",
    )
    for item in list(state.get("activity") or [])[-8:]:
        if not isinstance(item, dict):
            continue
        entry = {key: item.get(key) for key in ("label", "state", "timestamp") if item.get(key) is not None}
        details = item.get("details")
        if isinstance(details, dict):
            selected = {key: details[key] for key in activity_detail_keys if key in details}
            if selected:
                entry["details"] = selected
        payload["recent_agent_activity"].append(entry)
    payload["critic_reviews"] = list(state.get("critic") or [])[-5:]
    payload["recent_cycle_results"] = list(state.get("history") or [])[-5:]
    return payload


def _save_agent_conversation(chat):
    if (re.fullmatch(r"[a-f0-9]{32}", str(chat.get("id", "")))
            and isinstance(chat.get("messages"), list)):
        core.save_chat(chat)


_LIVE_PARAMETER_ALIASES = (
    ("epochs", re.compile(r"\b(?:maximum\s+epochs?|max\s+epochs?|epoch\s+limit|epochs?)\b", re.I), int),
    ("learning_rate", re.compile(r"\b(?:learning\s+rate|lr)\b", re.I), float),
    ("batch_size", re.compile(r"\bbatch\s+size\b", re.I), int),
    ("patience", re.compile(r"\b(?:early[- ]stop(?:ping)?\s+patience|patience)\b", re.I), int),
    ("dropout_rate", re.compile(r"\b(?:drop[\s-]*out(?:\s+rate)?|drop\s+rate)\b", re.I), float),
    ("weight_decay", re.compile(r"\bweight\s+decay\b", re.I), float),
    ("num_layers", re.compile(r"\b(?:number\s+of\s+(?:hidden\s+)?layers|hidden\s+layer\s+count|layer\s+count)\b", re.I), int),
    ("hidden_size", re.compile(r"\b(?:neurons?\s+(?:in\s+each\s+layer|per\s+layer)|units?\s+per\s+layer|hidden\s+layer\s+(?:width|size)|hidden\s+size|neuron\s+count)\b", re.I), int),
)
_LIVE_EDIT_VERB = re.compile(r"\b(?:set|change|adjust|update|make|put)\b", re.I)
_LIVE_CYCLE_REFERENCE = re.compile(r"\bcycle\s*(\d+)\b", re.I)
_LIVE_NUMERIC_VALUE = re.compile(
    r"\s*(?:(?:to|at|be|=|:|change|changed|set)\s*)*"
    r"(zero|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
    re.I,
)
_LIVE_BARE_NUMBER = re.compile(
    r"\s*(zero|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*", re.I,
)
_LIVE_PARAMETER_LABELS = {
    "epochs": "epoch limit", "learning_rate": "learning rate", "batch_size": "batch size",
    "patience": "early-stop patience", "dropout_rate": "dropout", "weight_decay": "weight decay",
    "hidden_size": "neurons per layer", "num_layers": "hidden layers",
}


def _format_live_parameter_edits(edits):
    return ", ".join(
        f"{_LIVE_PARAMETER_LABELS.get(name, name.replace('_', ' '))} {value:g}"
        if isinstance(value, (int, float)) else
        f"{_LIVE_PARAMETER_LABELS.get(name, name.replace('_', ' '))} {value}"
        for name, value in edits.items()
    )


def _live_current_cycle(state, checkpoint, fallback_cycle):
    if isinstance(checkpoint, dict) and checkpoint.get("phase") == "early_results":
        try:
            return int(checkpoint.get("cycle", fallback_cycle) or 0)
        except (TypeError, ValueError):
            return int(fallback_cycle or 0)
    active_cycles = []
    for item in list(state.get("activity") or []):
        identifier = str(item.get("id", "")) if isinstance(item, dict) else ""
        if identifier.startswith("actor:") and item.get("state") == "running":
            try:
                active_cycles.append(int(identifier.split(":", 1)[1]))
            except ValueError:
                continue
    return max(active_cycles, default=int(fallback_cycle or 0))


def _live_number_value(token):
    """Convert a parsed numeric token, including the common spoken value zero."""
    return 0.0 if str(token).strip().casefold() == "zero" else float(token)


def _live_parameter_request(question, *, allow_implicit=False):
    text = question or ""
    if not _LIVE_EDIT_VERB.search(text) and "=" not in text:
        # Once a setting is queued, clients often add another value as a short
        # continuation ("and drop out to zero"). Accept that statement, but
        # leave questions and standalone parameter mentions to the agent.
        if not allow_implicit or "?" in text:
            return None
    return _live_parameter_values(text)


def _live_parameter_values(text):
    """Read parameter/value pairs from a sentence, including an agent's clarification."""
    edits = {}
    for name, alias, converter in _LIVE_PARAMETER_ALIASES:
        match = alias.search(text)
        if not match:
            continue
        value_match = _LIVE_NUMERIC_VALUE.match(text[match.end():])
        if not value_match:
            continue
        try:
            raw = _live_number_value(value_match.group(1))
            value = converter(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if converter is int and raw != value:
            return {"error": f"{name} must be a whole number."}
        edits[name] = value
    return edits or None


def _live_parameter_names(text):
    return [name for name, alias, _converter in _LIVE_PARAMETER_ALIASES if alias.search(text or "")]


def _live_parameter_followup_request(question, messages):
    """Resolve short answers against the preceding parameter question, without another model turn."""
    text = re.sub(r"[.!?]+$", "", (question or "").strip()).casefold()
    yes_reply = text in {"yes", "yes please", "yeah", "yep", "correct", "right"}
    next_reply = text in {
        "next", "next one", "next cycle", "the next cycle", "for next cycle",
        "for the next cycle", "future cycle", "for a future cycle",
    }
    bare_number = _LIVE_BARE_NUMBER.fullmatch(question or "")
    if not (yes_reply or next_reply or bare_number):
        return None

    prior_assistant = next(
        (item for item in reversed(messages or [])
         if isinstance(item, dict) and item.get("role") == "assistant"
         and str(item.get("content") or "").strip()),
        None,
    )
    if not prior_assistant:
        return None
    assistant_text = str(prior_assistant.get("content") or "")
    assistant_lower = assistant_text.casefold()
    names = _live_parameter_names(assistant_text)
    if len(names) != 1:
        return None
    name = names[0]

    if bare_number:
        asks_for_value = (
            "?" in assistant_text
            and re.search(r"\b(?:what|which|how much|value|set|choose|provide|enter|desired)\b", assistant_lower)
        )
        if not asks_for_value:
            return None
        converter = next(converter for param, _alias, converter in _LIVE_PARAMETER_ALIASES if param == name)
        try:
            raw = _live_number_value(bare_number.group(1))
            value = converter(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if converter is int and raw != value:
            return {"error": f"{name} must be a whole number."}
        return {name: value}

    asks_about_target = bool(re.search(
        r"\b(?:which|what|current|next|future)\b.{0,50}\bcycle\b|"
        r"\bcycle\b.{0,50}\b(?:current|next|future|which|what)\b",
        assistant_lower,
    ))
    asks_for_confirmation = bool(re.search(
        r"\b(?:confirm|confirmation|are you asking|should i|do you want|would you like|apply)\b",
        assistant_lower,
    ))
    if (next_reply and not asks_about_target) or (yes_reply and not (asks_about_target or asks_for_confirmation)):
        return None

    edits = _live_parameter_values(assistant_text)
    if edits and name in edits:
        return {name: edits[name]}

    # The latest model question may omit the value even though the client already gave it.
    # Recover only a recent bare number or an explicit value for this same parameter.
    earlier_items = list(messages or [])
    try:
        assistant_index = max(
            index for index, item in enumerate(earlier_items)
            if isinstance(item, dict) and item.get("role") == "assistant"
            and item.get("content") == prior_assistant.get("content")
        )
    except ValueError:
        assistant_index = len(earlier_items)
    for item in reversed(earlier_items[max(0, assistant_index - 8):assistant_index]):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        prior_user_text = str(item.get("content") or "")
        prior_edits = _live_parameter_request(prior_user_text) or _live_parameter_values(prior_user_text)
        if prior_edits and name in prior_edits:
            return {name: prior_edits[name]}
        number = _LIVE_BARE_NUMBER.fullmatch(prior_user_text)
        if number:
            converter = next(converter for param, _alias, converter in _LIVE_PARAMETER_ALIASES if param == name)
            try:
                raw = _live_number_value(number.group(1))
                value = converter(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if converter is int and raw != value:
                return {"error": f"{name} must be a whole number."}
            return {name: value}
    return None


def _live_parameter_value_error(edits, checkpoint):
    bounds = checkpoint.get("candidate_bounds") if isinstance(checkpoint, dict) else {}
    bounds = bounds if isinstance(bounds, dict) else {}
    ranges = {
        "epochs": (1, 5000), "batch_size": (8, 4096), "patience": (1, 1000),
        "dropout_rate": (0.0, float(cfg.DROPOUT_RATE_MAX)),
        "weight_decay": (0.0, float(cfg.WEIGHT_DECAY_MAX)),
        "learning_rate": (
            float(bounds.get("learning_rate_min", cfg.LEARNING_RATE_MIN)),
            float(bounds.get("learning_rate_max", cfg.LEARNING_RATE_MAX)),
        ),
        "hidden_size": (
            int(bounds.get("hidden_size_min", cfg.HIDDEN_SIZE_MIN)),
            int(bounds.get("hidden_size_max", cfg.HIDDEN_SIZE_MAX)),
        ),
        "num_layers": (
            int(bounds.get("num_layers_min", cfg.NUM_LAYERS_MIN)),
            int(bounds.get("num_layers_max", cfg.NUM_LAYERS_MAX)),
        ),
    }
    for name, value in edits.items():
        low, high = ranges[name]
        if not math.isfinite(float(value)) or not low <= value <= high:
            return f"{name.replace('_', ' ').title()} must be between {low:g} and {high:g}. I have not changed the run."
    return None


def _resolve_live_parameter_message(question, chat, state, checkpoint, cycle):
    """Handle explicit next-candidate edits and short confirmations deterministically."""
    pending = chat.get("pending_live_parameter_edit")
    queued = chat.get("queued_live_parameter_edit")
    requested_edits = _live_parameter_request(
        question, allow_implicit=isinstance(pending, dict) or isinstance(queued, dict),
    )
    messages = (chat.get("run_agent_conversations") or {}).get("live_run", [])
    if requested_edits is None:
        requested_edits = _live_parameter_followup_request(question, messages)
    short_ack = re.fullmatch(r"(?:yes|yes please|yeah|yep|correct|right|next|next one|next cycle|the next cycle|for next cycle|for the next cycle)", (question or "").strip().rstrip(".!?").casefold())
    latest_assistant = next(
        (str(item.get("content") or "") for item in reversed(messages)
         if isinstance(item, dict) and item.get("role") == "assistant"),
        "",
    )
    is_queued_confirmation = bool(re.match(
        r"\s*(?:That change is already queued|I['’]ve queued)\b",
        latest_assistant,
        re.I,
    ))
    if requested_edits is None and isinstance(queued, dict) and short_ack and is_queued_confirmation:
        return (
            f"That change is already queued for candidate cycle **{int(queued.get('cycle', 0) or 0)}**. "
            "I’ll apply it before that candidate starts.",
            False,
        )
    if requested_edits is not None:
        if "error" in requested_edits:
            return (requested_edits["error"], False)
        value_error = _live_parameter_value_error(requested_edits, checkpoint)
        if value_error:
            return (value_error, False)
        current_cycle = _live_current_cycle(state, checkpoint, cycle)
        target_match = _LIVE_CYCLE_REFERENCE.search(question or "")
        target_cycle = int(target_match.group(1)) if target_match else current_cycle + 1
        if target_cycle <= current_cycle:
            return (
                f"Cycle {target_cycle} has already started or finished, so I cannot change its settings. "
                f"The next candidate I can edit is cycle {current_cycle + 1}.",
                False,
            )
        try:
            existing_cycle = int((pending or {}).get("cycle"))
        except (TypeError, ValueError):
            existing_cycle = None
        merged_edits = dict((pending or {}).get("edits") or {}) if existing_cycle == target_cycle else {}
        merged_edits.update(requested_edits)
        pending = {"edits": merged_edits, "cycle": target_cycle}
        chat["pending_live_parameter_edit"] = pending
    elif isinstance(pending, dict):
        text = (question or "").strip().casefold()
        target_match = _LIVE_CYCLE_REFERENCE.search(question or "")
        if re.fullmatch(r"(?:no|no thanks|cancel|cancel it|never mind|nevermind)", text):
            chat.pop("pending_live_parameter_edit", None)
            return ("Okay—I canceled the queued setting change. The run settings are unchanged.", False)
        elif target_match:
            target_cycle = int(target_match.group(1))
            current_cycle = _live_current_cycle(state, checkpoint, cycle)
            if target_cycle <= current_cycle:
                return (
                    f"Cycle {target_cycle} has already started or finished, so I cannot change it. "
                    f"The next editable candidate is cycle {current_cycle + 1}.",
                    False,
                )
            pending["cycle"] = target_cycle
            chat["pending_live_parameter_edit"] = pending
        elif re.fullmatch(r"(?:yes|yes please|yeah|yep|correct|right|next cycle\??|the next cycle\??)", text):
            pass
        else:
            return None
    else:
        return None

    pending = chat.get("pending_live_parameter_edit")
    if not isinstance(pending, dict):
        return None
    checkpoint_cycle = None
    if isinstance(checkpoint, dict) and checkpoint.get("phase") == "early_results":
        try:
            checkpoint_cycle = int(checkpoint.get("cycle", 0) or 0)
        except (TypeError, ValueError):
            checkpoint_cycle = 0
    if checkpoint_cycle is not None and int(pending["cycle"]) == checkpoint_cycle + 1:
        return ("Applying this setting to the next candidate.", True)

    summary = _format_live_parameter_edits(pending["edits"])
    epoch_note = (
        " The adaptive extension setting may add epochs, and early stopping may finish sooner."
        if "epochs" in pending["edits"] else ""
    )
    return (
        f"I’ve queued **{summary}** for candidate cycle **{int(pending['cycle'])}**. "
        "I’ll apply it immediately before that candidate starts; the current candidate keeps its existing settings. "
        + epoch_note,
        False,
    )


def _materialize_live_candidate_edits(pending, checkpoint):
    edits = dict(pending.get("edits") or {})
    if not ({"hidden_size", "num_layers"} & set(edits)):
        return edits
    config = checkpoint.get("next_config") or {}
    widths = [int(width) for width in (config.get("hidden_layers") or [cfg.HIDDEN_SIZE_MIN])]
    if "num_layers" in edits:
        layer_count = int(edits.pop("num_layers"))
        widths = (widths[:layer_count] + [widths[-1]] * max(0, layer_count - len(widths)))
    if "hidden_size" in edits:
        width = int(edits.pop("hidden_size"))
        widths = [width] * len(widths)
    edits["hidden_layers"] = widths
    return edits


def _apply_pending_live_parameter_edit(chat, task, checkpoint):
    """Apply a pending edit immediately when its review checkpoint is open."""
    pending = chat.get("pending_live_parameter_edit")
    if not isinstance(pending, dict) or not isinstance(checkpoint, dict):
        return False
    if checkpoint.get("phase") != "early_results":
        return False
    try:
        target_cycle = int(pending.get("cycle"))
        next_cycle = int(checkpoint.get("cycle", 0) or 0) + 1
    except (TypeError, ValueError):
        chat.pop("pending_live_parameter_edit", None)
        return False
    if target_cycle != next_cycle:
        return False
    runner = task.get("runner") if isinstance(task, dict) else None
    queued = chat.get("queued_live_parameter_edit")
    if isinstance(queued, dict) and int(queued.get("cycle", 0) or 0) == target_cycle:
        cancel = getattr(runner, "cancel_candidate_edits", None)
        if callable(cancel) and cancel(target_cycle):
            merged_edits = dict(queued.get("edits") or {})
            merged_edits.update(dict(pending.get("edits") or {}))
            pending["edits"] = merged_edits
            chat["pending_live_parameter_edit"] = pending
        chat.pop("queued_live_parameter_edit", None)
    edits = _materialize_live_candidate_edits(pending, checkpoint)
    value_error = _live_parameter_value_error(pending.get("edits") or {}, checkpoint)
    if value_error:
        chat.pop("pending_live_parameter_edit", None)
        messages = chat.setdefault("run_agent_conversations", {}).setdefault("live_run", [])
        messages.append({"role": "assistant", "kind": "parameter_edit", "answer_scope": "run",
                         "feedback_id": uuid.uuid4().hex,
                         "content": value_error})
        chat["run_agent_conversations"]["live_run"] = messages[-24:]
        _save_agent_conversation(chat)
        return False
    if runner is None or not runner.respond_to_checkpoint("custom", edits=edits):
        return False

    chat.pop("pending_live_parameter_edit", None)
    messages = chat.setdefault("run_agent_conversations", {}).setdefault("live_run", [])
    summary = _format_live_parameter_edits(pending["edits"])
    messages.append({
        "role": "assistant",
        "kind": "parameter_edit",
        "answer_scope": "run",
        "feedback_id": uuid.uuid4().hex,
        "content": (
            f"Done. I set **{summary}** for candidate cycle **{target_cycle}**. "
            + ("Adaptive epoch extension may add more epochs, and early stopping may finish sooner."
               if "epochs" in pending["edits"] else "")
        ),
    })
    chat["run_agent_conversations"]["live_run"] = messages[-24:]
    _save_agent_conversation(chat)
    st.rerun()
    return True


def _talk_to_agent_chat(chat, state, checkpoint, checkpoint_id, cycle, task, run_dir=None):
    """Chat with LabCD and queue explicit edits for a candidate start boundary."""
    conversations = chat.setdefault("run_agent_conversations", {})
    conversation_id = "live_run"
    messages = conversations.setdefault(conversation_id, [])
    if not isinstance(messages, list):
        messages = conversations[conversation_id] = []
    jobs = state.setdefault("talk_agent_jobs", {})
    job = jobs.get(conversation_id)
    if job is not None and not job.running:
        if job.answer:
            content = str(job.answer.get("answer") or job.answer.get("summary") or "I couldn't prepare an answer.")
            extra = {"evidence": job.answer.get("evidence", []),
                     "answer_scope": job.answer.get("answer_scope", "run"),
                     "response_seconds": _response_elapsed(job),
                     "response_steps": job.client_activity_steps()}
            if job.answer.get("status") == "clarification":
                extra.update(kind="clarification", asked_at=time.time(), id=f"live_question_{time.time_ns()}")
        else:
            content = job.error or "I couldn't get an answer from the agent. Please try again."
            extra = {"kind": "error", "response_seconds": _response_elapsed(job),
                     "response_steps": job.client_activity_steps()}
            if job.error_detail:
                extra["error_detail"] = job.error_detail
        messages.append({"role": "assistant", "content": content,
                         "feedback_id": uuid.uuid4().hex, **extra})
        conversations[conversation_id] = messages[-24:]
        _save_agent_conversation(chat)
        jobs.pop(conversation_id, None)
        job = None

    st.caption(
        "Ask about this run or request next-cycle changes: layer count/width, learning rate, epochs, "
        "batch size, patience, dropout, or weight decay. Changes default to the next candidate that has not started."
    )
    form_key = f"talk_agent_chat_{chat['id']}_{checkpoint_id}"
    message_area = st.container(
        height=180, key=f"talk_agent_messages_{chat['id']}", autoscroll=True,
    )
    with st.form(form_key, clear_on_submit=True, border=False):
        question = st.text_input(
            "Message LabCD", placeholder="e.g. Set epochs to 80 and learning rate to 0.001",
            label_visibility="collapsed", max_chars=6000,
            key=f"{form_key}_input", disabled=bool(job and job.running),
        )
        submitted = st.form_submit_button(
            "Send", icon=":material/send:", type="primary", width="stretch",
            disabled=bool(job and job.running),
        )
    question = question.strip() if submitted and question.strip() else None
    if question:
        response_started_at = time.monotonic()
        messages.append({"role": "user", "content": question, "answer_scope": "run"})
        conversations[conversation_id] = messages[-24:]
        _save_agent_conversation(chat)
        queued_edit = chat.get("queued_live_parameter_edit")
        cancel_queued = bool(
            isinstance(queued_edit, dict)
            and re.fullmatch(r"(?:no|no thanks|cancel|cancel it|never mind|nevermind)", question, re.I)
        )
        if cancel_queued:
            target_cycle = int(queued_edit.get("cycle", 0) or 0)
            runner = task.get("runner") if isinstance(task, dict) else None
            cancel = getattr(runner, "cancel_candidate_edits", None)
            if callable(cancel) and cancel(target_cycle):
                chat.pop("queued_live_parameter_edit", None)
                content = f"Okay—I canceled the queued change for candidate cycle **{target_cycle}**. The run is unchanged."
            else:
                chat.pop("queued_live_parameter_edit", None)
                content = (
                    f"Candidate cycle **{target_cycle}** has already started, so I couldn't cancel its settings. "
                    "A running candidate keeps its settings."
                )
            local_edit = (content, False)
        else:
            local_edit = _resolve_live_parameter_message(question, chat, state, checkpoint, cycle)
        if local_edit is not None:
            content, apply_now = local_edit
            if apply_now and _apply_pending_live_parameter_edit(chat, task, checkpoint):
                return
            if apply_now:
                pending = chat.get("pending_live_parameter_edit") or {}
                target_cycle = pending.get("cycle", "the next")
                content = (
                    f"The review checkpoint changed before I could apply this. It remains queued "
                    f"for candidate cycle **{target_cycle}**."
                )
            pending = chat.get("pending_live_parameter_edit")
            if not apply_now and isinstance(pending, dict):
                runner = task.get("runner") if isinstance(task, dict) else None
                queue_edits = getattr(runner, "queue_candidate_edits", None)
                target_cycle = int(pending.get("cycle", 0) or 0)
                if callable(queue_edits) and queue_edits(target_cycle, dict(pending.get("edits") or {})):
                    chat.pop("pending_live_parameter_edit", None)
                    previous_queued = chat.get("queued_live_parameter_edit")
                    merged_queued_edits = {}
                    if (isinstance(previous_queued, dict)
                            and int(previous_queued.get("cycle", 0) or 0) == target_cycle):
                        merged_queued_edits.update(dict(previous_queued.get("edits") or {}))
                    elif isinstance(previous_queued, dict):
                        cancel_old = getattr(runner, "cancel_candidate_edits", None)
                        if callable(cancel_old):
                            cancel_old(int(previous_queued.get("cycle", 0) or 0))
                    merged_queued_edits.update(dict(pending.get("edits") or {}))
                    chat["queued_live_parameter_edit"] = {
                        "cycle": target_cycle, "edits": merged_queued_edits,
                    }
                    summary = _format_live_parameter_edits(pending.get("edits") or {})
                    epoch_note = (
                        " The adaptive extension setting may add epochs, and early stopping may finish sooner."
                        if "epochs" in (pending.get("edits") or {}) else ""
                    )
                    content = (
                        f"I’ve queued **{summary}** for candidate cycle **{target_cycle}**. "
                        "I’ll apply it immediately before that candidate starts; the current candidate "
                        "keeps its existing settings." + epoch_note
                    )
                else:
                    chat.pop("pending_live_parameter_edit", None)
                    last_started = int(getattr(runner, "last_started_cycle", 0) or 0)
                    if callable(queue_edits):
                        retry_cycle = max(last_started, _live_current_cycle(state, checkpoint, cycle)) + 1
                        retry_edits = dict(pending.get("edits") or {})
                        if queue_edits(retry_cycle, retry_edits):
                            previous_queued = chat.get("queued_live_parameter_edit")
                            merged_queued_edits = {}
                            if (isinstance(previous_queued, dict)
                                    and int(previous_queued.get("cycle", 0) or 0) == retry_cycle):
                                merged_queued_edits.update(dict(previous_queued.get("edits") or {}))
                            elif isinstance(previous_queued, dict):
                                cancel_old = getattr(runner, "cancel_candidate_edits", None)
                                if callable(cancel_old):
                                    cancel_old(int(previous_queued.get("cycle", 0) or 0))
                            merged_queued_edits.update(retry_edits)
                            chat["queued_live_parameter_edit"] = {
                                "cycle": retry_cycle, "edits": merged_queued_edits,
                            }
                            summary = _format_live_parameter_edits(retry_edits)
                            content = (
                                f"Candidate cycle **{target_cycle}** started just before I could queue the change. "
                                f"I’ve queued **{summary}** for the next candidate, cycle **{retry_cycle}**."
                            )
                        else:
                            content = (
                                f"Candidate cycle **{target_cycle}** has already started. "
                                f"The next candidate that can accept a change is cycle **{retry_cycle}**."
                            )
                    else:
                        content = (
                            "This run could not confirm a safe candidate boundary, so I left its settings unchanged. "
                            "Use the next-candidate controls at a review checkpoint."
                        )
            messages = conversations[conversation_id]
            messages.append({"role": "assistant", "kind": "parameter_edit",
                             "feedback_id": uuid.uuid4().hex, "answer_scope": "run", "content": content})
            messages[-1]["response_seconds"] = max(0.0, time.monotonic() - response_started_at)
            messages[-1]["response_steps"] = [
                "Read your message", "Checked the requested change against the run state",
                "Prepared the next-cycle update",
            ]
            conversations[conversation_id] = messages[-24:]
            _save_agent_conversation(chat)
        else:
            snapshot = copy.deepcopy(chat)
            transcript = list(snapshot.get("messages") or [])
            transcript.extend({"role": item.get("role"), "content": item.get("content", ""),
                               "answer_scope": item.get("answer_scope", "run")}
                              for item in messages[-12:])
            snapshot["messages"] = transcript
            snapshot["live_run_context"] = _live_agent_context(
                state, checkpoint, cycle,
                queued_parameter_edit=chat.get("queued_live_parameter_edit"),
            )
            if run_dir:
                snapshot["live_run_dir"] = str(run_dir)
            job = agent.ConversationJob(snapshot, question, purpose="live_run_chat")
            jobs[conversation_id] = job
            job.start()

    # Render after submission so the new question appears immediately. The
    # bounded history scrolls independently while the composer stays in view.
    with message_area:
        pending_question_id = _pending_client_question_id(messages)
        first_visible = max(0, len(messages) - 8)
        for message_index, message in enumerate(messages[-8:], start=first_visible):
            with st.chat_message(message.get("role", "assistant")):
                if message.get("role") == "assistant":
                    _render_message_activity(
                        message, message.get("feedback_id") or message.get("id") or str(message_index))
                st.markdown(str(message.get("content", "")))
                if message.get("error_detail"):
                    with st.expander("Troubleshooting details"):
                        st.text(message["error_detail"])
                _render_answer_feedback(
                    chat, message,
                    key_id=(message.get("feedback_id") or message.get("id")
                            or _fallback_feedback_id(chat, message, message_index)),
                    save=lambda: _save_agent_conversation(chat),
                )
                if (message.get("kind") == "clarification"
                        and message.get("id") == pending_question_id):
                    _render_client_question_skip(chat, pending_question_id)
        if job is not None and job.running:
            elapsed = _response_elapsed(job)
            with st.container(key=f"message_activity_live_run_{chat['id']}"):
                with st.status(f"Thinking · {_format_thinking_time(elapsed)}", state="running", expanded=False):
                    st.caption(_client_progress_text(job, "Preparing the cycle context"))
def _talk_to_agent_section(chat, task, state):
    """Offer a compact, optional run chat for questions and next-candidate edits."""
    checkpoint = state.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else None
    phase = checkpoint.get("phase") if checkpoint else None
    checkpoint_id = str(checkpoint.get("id", "messages")) if checkpoint else "messages"
    if phase == "early_results" and _apply_pending_live_parameter_edit(chat, task, checkpoint):
        return
    history = list(state.get("history") or [])
    latest = history[-1] if history else {}
    cycle = _live_current_cycle(state, checkpoint, _finite_number(latest.get("cycle"), 0))
    runner = task.get("runner") if isinstance(task, dict) else None
    run_dir = getattr(runner, "run_dir", None)
    with st.expander("Talk to LabCD · next cycle", expanded=False, icon=":material/forum:"):
        _talk_to_agent_chat(chat, state, checkpoint, checkpoint_id, cycle, task, run_dir=run_dir)


def _render_cycle_checkpoint_controls(chat, task, checkpoint):
    """Keep checkpoint decisions and the full manual editor outside the chat."""
    if not isinstance(checkpoint, dict) or checkpoint.get("phase") != "early_results":
        return
    checkpoint_id = str(checkpoint.get("id", "cycle_review"))
    cycle = int(checkpoint.get("cycle", 0) or 0)
    with st.expander("Review the next candidate", expanded=False, icon=":material/tune:"):
        st.caption(
            f"Cycle {cycle} is complete. Adjust parameters for cycle {cycle + 1}, continue with the agent proposal, or stop here."
        )
        _next_candidate_editor(chat, task, checkpoint)
        action_columns = st.columns(2, gap="small")
        for column, action, label, icon in (
            (action_columns[0], "continue", "Continue with proposal", "arrow_forward"),
            (action_columns[1], "compact", "Try a smaller model", "compress"),
            (action_columns[0], "regularize", "Add regularization", "shield"),
            (action_columns[1], "widen", "Broaden search", "open_in_full"),
            (action_columns[0], "stop", "Stop and keep best", "stop"),
        ):
            if column.button(
                label, key=f"talk_agent_{checkpoint_id}_{action}",
                icon=f":material/{icon}:",
                type="primary" if action == "continue" else "secondary",
                width="stretch",
            ):
                _dismiss_pending_question_timer(chat)
                _respond_to_human_checkpoint(task, action)


def _response_elapsed(job):
    started_at = getattr(job, "started_at", None)
    if not isinstance(started_at, (int, float)):
        started_at = time.monotonic()
        try:
            job.started_at = started_at
        except (AttributeError, TypeError):
            pass
    return max(0.0, time.monotonic() - started_at)


def _format_thinking_time(elapsed):
    total_seconds = max(0, int(elapsed))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _client_progress_text(job, default="Working on your request"):
    """Return a concise progress label without exposing internal reasoning text."""
    phase = getattr(job, "phase", None)
    if isinstance(phase, dict):
        phase = phase.get(getattr(job, "answer_scope", None))
    safe = agent._public_activity_label(phase, getattr(job, "answer_scope", None))
    if safe:
        return safe
    phase_labels = {
        "reading your message and current run settings": "Checking your current setup",
        "preparing an updated run plan": "Preparing an updated setup",
        "reasoning about your question": "Preparing a reply",
    }
    return phase_labels.get(str(phase or "").strip().casefold(), default)


def _pending_client_question_id(messages):
    """Return the newest unanswered clarification or setup question."""
    last_user = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    for index in range(len(messages) - 1, last_user, -1):
        message = messages[index]
        if (message.get("role") == "assistant" and message.get("kind") in {
            "clarification", "setup_question",
        } and not message.get("question_timer_dismissed")
                and not message.get("question_resolved")):
            return str(message.get("id") or f"question_{index}")
    return None


def _question_message(messages, question_id):
    for index, message in enumerate(messages):
        if str(message.get("id") or f"question_{index}") == str(question_id):
            return message
    return None


def _dismiss_pending_question_timer(chat):
    changed = False
    collections = [chat.get("messages") or []]
    collections.extend(
        messages for messages in (chat.get("run_agent_conversations") or {}).values()
        if isinstance(messages, list)
    )
    for messages in collections:
        question_id = _pending_client_question_id(messages)
        if question_id is None:
            continue
        message = _question_message(messages, question_id)
        if message is not None:
            message["question_timer_dismissed"] = True
            changed = True
    if changed:
        _save_agent_conversation(chat)


def _client_question_started_at(message):
    value = message.get("asked_at")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    timestamp = message.get("timestamp")
    try:
        return dt.datetime.fromisoformat(str(timestamp)).timestamp()
    except (TypeError, ValueError, OverflowError):
        message["asked_at"] = time.time()
        return message["asked_at"]


def _stop_starting_setup_countdown(chat, message_id):
    changed = False
    if message_id:
        message = _question_message(chat.get("messages") or [], str(message_id))
        if message is not None and not message.get("setup_timer_dismissed"):
            message["setup_timer_dismissed"] = True
            changed = True
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    if (isinstance(flow, dict) and flow.get("dataset_sha256") == dataset.get("sha256")
            and not flow.get("setup_timer_dismissed")):
        flow["setup_timer_dismissed"] = True
        chat["run_setup_flow"] = flow
        changed = True
    if changed:
        _save_agent_conversation(chat)


def _render_starting_setup_countdown(chat, message_id):
    _render_starting_setup_countdown_fragment(chat, str(message_id))


@st.fragment(run_every=1.0)
def _render_starting_setup_countdown_fragment(chat, message_id):
    messages = chat.get("messages") or []
    message = _question_message(messages, message_id)
    latest_setup = next(
        (item for item in reversed(messages) if item.get("kind") == "run_setup"), None
    )
    dataset = chat.get("dataset") or {}
    flow = chat.get("run_setup_flow") or {}
    if (message is None or message.get("kind") != "run_setup"
            or latest_setup is None or str(latest_setup.get("id")) != message_id
            or flow.get("dataset_sha256") != dataset.get("sha256")
            or not dataset.get("ready") or chat.get("run_dir")):
        return

    timer_key = f"starting_setup_timer_{chat['id']}_{message_id}"
    if flow.get("stage") == "approved":
        return
    if message.get("setup_timer_dismissed") or flow.get("setup_timer_dismissed"):
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                          key=timer_key):
            st.caption("Auto-start stopped · setup settings saved", text_alignment="right", width="content")
        return

    started_at = message.get("setup_timer_started_at")
    if not isinstance(started_at, (int, float)) or not math.isfinite(started_at):
        started_at = time.time()
        message["setup_timer_started_at"] = started_at
        _save_agent_conversation(chat)
    wait_left = STARTING_SETUP_AUTO_START_SECONDS - (time.time() - started_at)
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                      gap="small", wrap=False, key=timer_key):
        if wait_left > 0:
            st.caption(
                f":material/hourglass_bottom: Start training with this setup in "
                f"{_format_thinking_time(math.ceil(wait_left))}",
                text_alignment="right", width="content",
            )
        else:
            if setup_summary.approve_and_start_training(chat):
                st.rerun()
            return
        if st.button(
            "Stop", key=f"stop_starting_setup_timer_{chat['id']}_{message_id}",
            icon=":material/stop:", type="tertiary", help="Stop this countdown",
        ):
            _stop_starting_setup_countdown(chat, message_id)
            st.rerun()


def _render_client_question_skip(chat, question_id):
    _render_client_question_skip_fragment(chat, question_id)


@st.fragment(run_every=1.0)
def _render_client_question_skip_fragment(chat, question_id):
    _render_client_question_skip_contents(chat, question_id)


def _render_client_question_skip_contents(chat, question_id):
    if st.session_state.get("conversation_pending"):
        return
    main_messages = chat.get("messages", [])
    question = _question_message(main_messages, question_id)
    source_messages = main_messages if question is not None else []
    live_messages = (chat.get("run_agent_conversations") or {}).get("live_run", [])
    if question is None:
        question = _question_message(live_messages, question_id)
        source_messages = live_messages
    if question is None or _pending_client_question_id(source_messages) != str(question_id):
        return

    had_timer_start = isinstance(question.get("asked_at"), (int, float))
    started_at = _client_question_started_at(question)
    if not had_timer_start and question.get("asked_at"):
        _save_agent_conversation(chat)
    wait_left = QUESTION_SKIP_DELAY_SECONDS - (time.time() - started_at)
    timer_key = f"client_question_timer_{chat['id']}_{question_id}"
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                      gap="small", wrap=False, key=timer_key):
        if wait_left > 0:
            countdown_label = "Skip this question in" if question.get("kind") == "setup_question" else "Skip option in"
            st.caption(
                f":material/hourglass_bottom: {countdown_label} {_format_thinking_time(math.ceil(wait_left))}",
                text_alignment="right", width="content",
            )
        else:
            if question.get("kind") == "setup_question":
                if _skip_setup_question(chat, str(question_id), automatic=True):
                    st.rerun()
                return
            st.caption(
                ":material/keyboard_return: Ready when you are",
                text_alignment="right", width="content",
            )
            if st.button(
                "Skip question",
                key=f"skip_client_question_{chat['id']}_{question_id}",
                icon=":material/skip_next:", type="tertiary",
            ):
                _skip_client_question(chat, str(question_id))
                st.rerun()
            return
        if st.button(
            "Stop", key=f"cancel_client_question_timer_{chat['id']}_{question_id}",
            icon=":material/stop:", type="tertiary", help="Stop this countdown",
        ):
            question["question_timer_dismissed"] = True
            _save_agent_conversation(chat)
            st.rerun()


def _skip_client_question(chat, question_id):
    main_messages = chat.get("messages") or []
    if _pending_client_question_id(main_messages) != question_id:
        live_messages = (chat.get("run_agent_conversations") or {}).get("live_run", [])
        if _pending_client_question_id(live_messages) != question_id:
            return
        question = None
    else:
        question = _question_message(main_messages, question_id)
    if question is not None and question.get("kind") == "setup_question":
        _skip_setup_question(chat, question_id)
        return
    if question is not None:
        directive = (
            "Skip this clarification and continue my original request using the best-supported information already available. "
            "State any assumption briefly."
        )
        core.add_message(chat, "user", directive)
        job = agent.ConversationJob(chat, directive, purpose="question")
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return

    live_messages = (chat.get("run_agent_conversations") or {}).get("live_run", [])
    question = _question_message(live_messages, question_id)
    training = core.REGISTRY.training.get(chat["id"])
    if question is None or training is None:
        return
    state = training["state"]
    if _pending_client_question_id(live_messages) != question_id:
        return
    directive = (
        "Skip this clarification and continue my original request using the best-supported information already available. "
        "State any assumption briefly."
    )
    live_messages.append({"role": "user", "content": directive, "id": f"live_skip_{time.time_ns()}",
                          "answer_scope": "run"})
    conversations = chat.setdefault("run_agent_conversations", {})
    conversations["live_run"] = live_messages[-24:]
    _save_agent_conversation(chat)

    snapshot = copy.deepcopy(chat)
    transcript = list(snapshot.get("messages") or [])
    transcript.extend(
        {"role": item.get("role"), "content": item.get("content", ""),
         "answer_scope": item.get("answer_scope", "run")}
        for item in conversations["live_run"][-12:]
    )
    snapshot["messages"] = transcript
    checkpoint = state.get("checkpoint")
    history = list(state.get("history") or [])
    latest = history[-1] if history else {}
    cycle = int((checkpoint or {}).get("cycle", latest.get("cycle", 0)) or 0)
    snapshot["live_run_context"] = _live_agent_context(state, checkpoint, cycle)
    run_dir = getattr(training.get("runner"), "run_dir", None)
    if run_dir:
        snapshot["live_run_dir"] = str(run_dir)
    job = agent.ConversationJob(snapshot, directive, purpose="live_run_chat")
    state.setdefault("talk_agent_jobs", {})["live_run"] = job
    job.start()


def _skip_setup_question(chat, question_id=None, *, automatic=False):
    setup = chat.get("setup") or {}
    stage = setup.get("stage")
    profile = ((chat.get("dataset") or {}).get("analysis") or {})
    question = _question_message(chat.get("messages") or [], question_id) if question_id else None
    if question is not None and question.get("question_resolved"):
        return False
    if stage == "trajectory":
        answer = "Several stacked runs" if int(profile.get("time_reset_count", 0) or 0) else "One continuous run"
    elif stage == "split":
        answer = "Detect boundaries"
    elif stage == "split_times":
        setup["stage"] = "split"
        answer = "Detect boundaries"
    elif stage in ("angle", "angle_select"):
        setup["stage"] = "angle"
        answer = "Use automatic angle detection"
    elif stage == "pinn_interest":
        answer = "No PINN"
    elif stage == "pcnn_interest":
        answer = "No PCNN"
    else:
        return False

    reply = core.advance_setup(chat, answer)
    if not reply:
        return False
    if question is not None:
        question["question_resolved"] = True
        question["question_resolution"] = {
            "source": "data_default",
            "stage": stage,
            "answer": answer,
            "resolved_at": time.time(),
        }
    content = reply.pop("content")
    if automatic:
        content = "No reply needed — I’ll skip this question and continue. " + content
    complete = reply.get("kind") == "plan"
    core.add_message(chat, "assistant", content, **reply)
    if complete:
        _start_run_setup_agent(chat)
    return True


def _run_process_diagram(stage: str, active_milestone: int, active_agent: str | None,
                        current_cycle: int, cycle_limit: int) -> str:
    """Render the live identification pipeline and its Actor–Critic feedback loop."""
    stage = str(stage)
    data_state = "done" if active_milestone > 0 else "active"
    if stage in {"Splitting trajectories", "Initializer Agent"}:
        initializer_state = "active"
    elif stage in {"Tuning cycles", "Held-out verification", "Report & packaging"}:
        initializer_state = "done"
    else:
        initializer_state = ""
    if stage == "Tuning cycles":
        search_state = "active"
    elif stage in {"Held-out verification", "Report & packaging"}:
        search_state = "done"
    else:
        search_state = ""
    verify_state = "active" if stage == "Held-out verification" else "done" if stage == "Report & packaging" else ""
    package_state = "active" if stage == "Report & packaging" else ""
    cycle_text = f"Cycle {current_cycle} / {cycle_limit}" if current_cycle else "Preparing first cycle"
    actor_state = "active" if active_agent == "actor" else ""
    critic_state = "active" if active_agent == "critic" else ""

    def block(state: str, number: str, title: str, subtitle: str) -> str:
        classes = f"run-flow-block {f'run-flow-block--{state}' if state else ''}"
        return (f'<div class="{classes}"><small>{number}</small><b>{title}</b>'
                f'<span>{subtitle}</span></div>')

    aria = html.escape(
        f"Identification flow: inspect data, choose an initial network, repeat Actor training and Critic review "
        f"for {cycle_text}, verify on held-out data, and package results.",
        quote=True,
    )
    return f'''<section class="run-process-diagram" role="img" aria-label="{aria}">
      <div class="run-process-heading"><span>IDENTIFICATION FLOW</span><b>{cycle_text}</b></div>
      <div class="run-process-flow">
        {block(data_state, "01 · DATA", "Inspect", "signals & quality")}
        <span class="run-process-arrow" aria-hidden="true">→</span>
        {block(initializer_state, "02 · START", "Initializer", "first network")}
        <span class="run-process-arrow" aria-hidden="true">→</span>
        <div class="run-search-block{' run-search-block--active' if search_state == 'active' else ''}{' run-search-block--done' if search_state == 'done' else ''}">
          <div class="run-search-heading"><small>03 · CANDIDATE LOOP</small><b>{cycle_text}</b></div>
          <div class="run-search-agents">
            <div class="run-search-agent{' run-search-agent--active' if actor_state else ''}"><small>ACTOR</small><b>Train NN</b></div>
            <span aria-hidden="true">→</span>
            <div class="run-search-score"><small>VALIDATION</small><b>Score</b></div>
            <span aria-hidden="true">→</span>
            <div class="run-search-agent{' run-search-agent--active' if critic_state else ''}"><small>CRITIC</small><b>Review</b></div>
          </div>
          <div class="run-search-feedback">↶ &nbsp;Critic feedback shapes the next candidate</div>
        </div>
        <span class="run-process-arrow" aria-hidden="true">→</span>
        {block(verify_state, "04 · TEST", "Held-out", "unseen data")}
        <span class="run-process-arrow" aria-hidden="true">→</span>
        {block(package_state, "05 · OUTPUT", "Package", "report & files")}
      </div>
    </section>'''


RUN_TOUR_INTERVAL_SECONDS = 30


def _run_tour_scene_index(elapsed_seconds: float, session_key: str, scene_count: int) -> int:
    """Return the selected scene, advancing from the last manual choice every 30 seconds."""
    selection = st.session_state.get(session_key)
    if isinstance(selection, dict):
        try:
            selected = int(selection.get("index", 0)) % scene_count
            anchor = max(0.0, float(selection.get("anchor_elapsed", 0.0)))
            elapsed_after_choice = max(0.0, elapsed_seconds - anchor)
            return (selected + int(elapsed_after_choice // RUN_TOUR_INTERVAL_SECONDS)) % scene_count
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return int(max(0.0, elapsed_seconds) // RUN_TOUR_INTERVAL_SECONDS) % scene_count


def _select_run_tour_scene(session_key: str, scene_index: int, elapsed_seconds: float) -> None:
    st.session_state[session_key] = {
        "index": int(scene_index),
        "anchor_elapsed": max(0.0, float(elapsed_seconds)),
    }


def _render_run_tour_dots(elapsed_seconds: float, session_key: str, scene_count: int) -> None:
    selected = _run_tour_scene_index(elapsed_seconds, session_key, scene_count)
    with st.container(
        horizontal=True, wrap=False, horizontal_alignment="center", gap="small", key="run_tour_dots"
    ):
        for scene_index in range(scene_count):
            st.button(
                "●",
                key=f"{session_key}_dot_{scene_index}",
                type="primary" if scene_index == selected else "tertiary",
                width="content",
                on_click=_select_run_tour_scene,
                args=(session_key, scene_index, elapsed_seconds),
            )


def _run_tour_state_key(chat: dict, task: dict) -> str:
    chat_id = str(chat.get("id", "run"))
    try:
        started = str(int(float(task.get("started", 0)) * 1000))
    except (TypeError, ValueError):
        started = "active"
    token = re.sub(r"[^A-Za-z0-9_-]", "_", f"{chat_id}_{started}")[:80]
    return f"run_tour_slide_{token}"


def _run_learning_snapshot(elapsed_seconds: float, chat: dict, state: dict,
                           current_cycle: int, cycle_limit: int, best_mse: float | None,
                           stage: str, tour_state_key: str) -> str:
    """Show an illustrated run tour with manually selectable, auto-advancing slides."""
    dataset = chat.get("dataset") or {}
    settings = chat.get("settings") or {}
    profile = dataset.get("analysis") or {}
    states = list(dataset.get("states") or [])
    inputs = list(dataset.get("actions") or dataset.get("inputs") or [])
    derivatives = list(dataset.get("derivatives") or [])
    samples = int(dataset.get("rows") or profile.get("rows") or 0)
    architecture = str(settings.get("architecture", "LSTM")).upper()
    sample_period = dataset.get("sample_period") or profile.get("median_dt")

    def names(values: list, limit: int = 3) -> str:
        shown = [str(value) for value in values[:limit]]
        suffix = f" + {len(values) - limit} more" if len(values) > limit else ""
        return ", ".join(shown) + suffix if shown else "none confirmed"

    def number(value, digits: int = 4) -> str:
        try:
            parsed = float(value)
            return f"{parsed:.{digits}g}" if math.isfinite(parsed) else "not available"
        except (TypeError, ValueError):
            return "not available"

    data_name = Path(str(dataset.get("name") or "dataset")).name
    state_names, input_names = names(states), names(inputs)
    state_count, input_count = len(states), len(inputs)
    interval_note = f"{number(sample_period)} s median sample interval" if sample_period else "sample interval not available"
    duration = profile.get("observed_duration")
    duration_note = f"{number(duration)} s recording span" if duration is not None else "recording span not available"
    irregular_count = int(profile.get("irregular_intervals", 0) or 0)
    reset_count = int(profile.get("time_reset_count", 0) or 0)
    wrap_detected = list(profile.get("detected_wrap_states") or [])
    angle_indices = [index for index in (settings.get("angle_indices") or [])
                     if isinstance(index, int) and 0 <= index < len(states)]
    selected_angles = [states[index] for index in angle_indices]
    if selected_angles:
        wrap_note = "Angle wrapping selected for " + names(selected_angles) + "."
    elif settings.get("auto_detect_angles", True):
        wrap_note = "Automatic angle-wrap detection is enabled."
    else:
        wrap_note = "Angle wrapping is off for this run."
    if wrap_detected:
        wrap_note += " Data scan flagged " + names(wrap_detected) + "."

    if architecture == "LSTM":
        history_steps = int(settings.get("lstm_seq_length", 10) or 10)
        approx_history = (f" about {number((history_steps - 1) * float(sample_period), 3)} s"
                          if sample_period else "")
        temporal_note = f"This LSTM uses {history_steps} ordered samples{approx_history} as its history window."
        temporal_svg = '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="An ordered history window feeds an LSTM memory block">
          <text class="run-tour-label" x="36" y="32">RECENT MEASUREMENTS</text>
          <g class="run-tour-step"><rect x="38" y="72" width="64" height="54" rx="11"/><text x="52" y="96">x(t−3)</text><text class="run-tour-tiny" x="56" y="114">u(t−3)</text></g>
          <g class="run-tour-step"><rect x="122" y="72" width="64" height="54" rx="11"/><text x="136" y="96">x(t−2)</text><text class="run-tour-tiny" x="140" y="114">u(t−2)</text></g>
          <g class="run-tour-step"><rect x="206" y="72" width="64" height="54" rx="11"/><text x="220" y="96">x(t−1)</text><text class="run-tour-tiny" x="224" y="114">u(t−1)</text></g>
          <g class="run-tour-step run-tour-step--current"><rect x="290" y="72" width="64" height="54" rx="11"/><text x="304" y="96">x(t)</text><text class="run-tour-tiny" x="308" y="114">u(t)</text></g>
          <path class="run-tour-wire" d="M102 99H120 M186 99H204 M270 99H288 M354 99H388"/>
          <rect class="run-tour-block run-tour-block--model" x="389" y="59" width="76" height="80" rx="17"/>
          <path class="run-tour-memory" d="M403 102c9-30 17 28 28-2s15 7 20-8"/>
          <text class="run-tour-label" x="404" y="161">LSTM</text><text class="run-tour-tiny" x="382" y="194">ORDER CARRIES CONTEXT →</text>
          <circle class="run-tour-sample" r="5"><animateMotion dur="3s" repeatCount="indefinite" path="M55 99H102 M139 99H186 M223 99H270 M307 99H354 M354 99H389"/></circle>
        </svg>'''
    else:
        history_steps = 1
        temporal_note = "This MLP maps the current measured state and input directly; it does not use a sequence-history window."
        temporal_svg = '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="A current measurement snapshot maps directly through an MLP">
          <text class="run-tour-label" x="31" y="37">ONE CURRENT SNAPSHOT</text>
          <rect class="run-tour-block" x="30" y="70" width="105" height="86" rx="15"/>
          <text class="run-tour-label" x="52" y="105">x(t), u(t)</text><text class="run-tour-tiny" x="47" y="129">state + input</text>
          <path class="run-tour-wire" d="M135 86L224 76 M135 86L224 114 M135 86L224 151 M135 135L224 76 M135 135L224 114 M135 135L224 151 M244 76L330 94 M244 114L330 94 M244 151L330 94 M350 94L415 113"/>
          <circle class="run-tour-neuron" cx="234" cy="76" r="8"/><circle class="run-tour-neuron run-tour-neuron--active" cx="234" cy="114" r="8"/><circle class="run-tour-neuron" cx="234" cy="151" r="8"/>
          <circle class="run-tour-neuron run-tour-neuron--active" cx="340" cy="94" r="9"/>
          <rect class="run-tour-block run-tour-block--output" x="414" y="83" width="77" height="60" rx="13"/>
          <text class="run-tour-label" x="431" y="108">ẋ(t)</text><text class="run-tour-tiny" x="423" y="127">state change</text>
          <circle class="run-tour-sample" r="5"><animateMotion dur="2.8s" repeatCount="indefinite" path="M82 113L234 114L340 94L452 113"/></circle>
          <text class="run-tour-tiny" x="188" y="194">DIRECT FEATURE MAPPING →</text>
        </svg>'''

    if architecture == "LSTM":
        architecture_svg = temporal_svg
        layer_values = settings.get("manual_starting_hidden_layers")
        if layer_values:
            layer_note = "Approved LSTM starting widths: " + " → ".join(str(int(width)) for width in layer_values) + " units."
        else:
            layer_note = (
                f"LSTM selected. Search bounds allow {settings.get('num_layers_min', 1)}–{settings.get('num_layers_max', 3)} layers "
                f"and {settings.get('hidden_size_min', 32)}–{settings.get('hidden_size_max', 256)} units per layer."
            )
    else:
        architecture_svg = '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="An MLP with connected dense layers between inputs and predicted state derivatives">
          <text class="run-tour-label" x="34" y="36">INPUT FEATURES</text><text class="run-tour-label" x="420" y="36">STATE DERIVATIVES</text>
          <path class="run-tour-wire" d="M75 78L193 64 M75 78L193 111 M75 78L193 158 M75 116L193 64 M75 116L193 111 M75 116L193 158 M75 154L193 64 M75 154L193 111 M75 154L193 158 M211 64L320 80 M211 64L320 142 M211 111L320 80 M211 111L320 142 M211 158L320 80 M211 158L320 142 M338 80L435 91 M338 142L435 91"/>
          <circle class="run-tour-neuron" cx="74" cy="78" r="8"/><circle class="run-tour-neuron" cx="74" cy="116" r="8"/><circle class="run-tour-neuron" cx="74" cy="154" r="8"/>
          <circle class="run-tour-neuron run-tour-neuron--active" cx="202" cy="64" r="9"/><circle class="run-tour-neuron" cx="202" cy="111" r="9"/><circle class="run-tour-neuron run-tour-neuron--active" cx="202" cy="158" r="9"/>
          <circle class="run-tour-neuron" cx="329" cy="80" r="9"/><circle class="run-tour-neuron run-tour-neuron--active" cx="329" cy="142" r="9"/>
          <rect class="run-tour-block run-tour-block--output" x="429" y="68" width="69" height="50" rx="13"/><text class="run-tour-label" x="451" y="99">ẋ(t)</text>
          <circle class="run-tour-sample" r="5"><animateMotion dur="3s" repeatCount="indefinite" path="M74 116L202 111L329 142L463 93"/></circle>
          <text class="run-tour-tiny" x="214" y="202">DENSE LAYERS LEARN A DIRECT MAPPING →</text>
        </svg>'''
        layer_values = settings.get("manual_starting_hidden_layers")
        if layer_values:
            layer_note = "Approved MLP starting widths: " + " → ".join(str(int(width)) for width in layer_values) + " units."
        else:
            layer_note = (
                f"MLP selected. Search bounds allow {settings.get('num_layers_min', 1)}–{settings.get('num_layers_max', 3)} layers "
                f"and {settings.get('hidden_size_min', 32)}–{settings.get('hidden_size_max', 256)} units per layer."
            )

    selected_geometry = "Several stacked trajectories" if settings.get("multi_trajectory") else "One continuous trajectory"
    geometry_detail = (f"Data scan found {reset_count} time reset(s). " if reset_count else "")
    trajectory_note = f"{selected_geometry}. {geometry_detail}{wrap_note}"
    pinn_artifact = chat.get("pinn_equation") or {}
    pinn_enabled = bool(settings.get("use_pinn"))
    physics_mode = str(settings.get("physics_mode", "pinn")).lower()
    equation_name = Path(str(pinn_artifact.get("name") or "equation")).name
    if pinn_enabled and physics_mode == "pcnn":
        constrained = list(settings.get("pcnn_hard_constraint_states") or [])
        learned = [state for state in (dataset.get("states") or []) if state not in constrained]
        pinn_note = (f"PCNN is enabled with {equation_name}. Exact constraints: {', '.join(constrained) or 'not selected yet'}. "
                     f"The neural network predicts the remaining {len(learned)} state(s) directly from measurements.")
        physics_tour_text = "A PCNN replaces selected state-derivative outputs with exact rules. The neural network predicts every other state directly from measurements."
    elif pinn_enabled:
        pinn_note = f"PINN is enabled with the validated equation {equation_name}; it adds a soft physics loss during training."
        physics_tour_text = "A PINN predicts derivatives from measurements and adds a validated equation as a soft training constraint."
    else:
        pinn_note = "Physics constraints are off in this run; training uses the measured-data objective."
        physics_tour_text = "Choose a soft PINN equation loss or PCNN rules that replace selected state derivatives exactly."
    pinn_svg = '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="Measured data and physics equation both guide a dynamics model">
      <rect class="run-tour-block" x="28" y="52" width="126" height="58" rx="13"/><text class="run-tour-label" x="52" y="78">MEASURED DATA</text><text class="run-tour-tiny" x="49" y="96">state + input samples</text>
      <rect class="run-tour-block run-tour-block--model" x="201" y="82" width="112" height="68" rx="16"/><text class="run-tour-label" x="226" y="112">DYNAMICS</text><text class="run-tour-tiny" x="228" y="131">model fθ</text>
      <rect class="run-tour-equation" x="28" y="145" width="126" height="51" rx="12"/><text class="run-tour-equation-text" x="48" y="176">ẋ = f(x,u)</text>
      <rect class="run-tour-block run-tour-block--output" x="366" y="82" width="124" height="68" rx="16"/><text class="run-tour-label" x="388" y="112">PREDICTION</text><text class="run-tour-tiny" x="397" y="131">state change</text>
      <path class="run-tour-wire" d="M154 81L201 103 M154 171L201 130 M313 116H366"/>
      <circle class="run-tour-sample" r="5"><animateMotion dur="3.2s" repeatCount="indefinite" path="M154 81L201 103L313 116L366 116"/></circle>
      <text class="run-tour-tiny" x="177" y="193">DATA FIT + OPTIONAL PHYSICS CONSTRAINT</text>
    </svg>'''
    if not pinn_enabled:
        pinn_svg = pinn_svg.replace('class="run-tour-equation"', 'class="run-tour-equation run-tour-equation--off"')
    if not pinn_enabled:
        pinn_svg = pinn_svg.replace("DATA FIT + OPTIONAL PHYSICS CONSTRAINT", "DATA FIT · EQUATION CONSTRAINT OFF")

    latest = list(state.get("history") or [])[-1:]
    latest_score = latest[0].get("val_mse") if latest else None
    latest_score_note = f"Latest validation MSE: {number(latest_score)}." if latest_score is not None else "No candidate has a validation score yet."
    best_score_note = f"Best validation MSE so far: {number(best_mse)}." if best_mse is not None else "Best validation score will appear after the first candidate."
    goal_labels = {"balanced": "balanced", "accuracy": "accuracy", "speed": "speed", "compact": "compact model"}
    search_note = (
        f"{str(settings.get('run_mode', 'fast')).title()} search · {goal_labels.get(str(settings.get('optimization_goal', 'balanced')), 'balanced')} priority · "
        f"candidate {current_cycle} of {cycle_limit}. {best_score_note} {latest_score_note}"
    )
    heldout_status = (
        "Held-out verification is running now."
        if stage == "Held-out verification" else
        "Verification follows the candidate search."
        if stage != "Report & packaging" else "Verification is complete; LabCD is packaging the report."
    )
    derivative_note = (
        f"{len(derivatives)} supplied derivative column(s); any missing derivatives are estimated from the measured states."
    )

    scenes = [
        (
            "System identification, at a glance",
            "System identification learns a model of how a system changes from time, measured states, and control inputs.",
            f"{data_name} · {samples:,} samples · {state_count} states ({state_names}) · {input_count} inputs ({input_names}).",
            "Time, state, and input signals flow through a model to estimate the system's state changes.",
            '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="System identification flow from observed input and state signals to a learned dynamics model">
              <text class="run-tour-label" x="28" y="37">OBSERVE THE SYSTEM</text>
              <rect class="run-tour-block" x="27" y="62" width="114" height="58" rx="13"/><text class="run-tour-label" x="50" y="88">x(t), u(t)</text><text class="run-tour-tiny" x="47" y="106">measured signals</text>
              <path class="run-tour-wire" d="M141 91H194 M303 91H359 M418 119V157H86V121"/>
              <rect class="run-tour-block run-tour-block--model" x="194" y="57" width="109" height="69" rx="16"/><text class="run-tour-label" x="218" y="87">MODEL fθ</text><text class="run-tour-tiny" x="211" y="107">learn dynamics</text>
              <rect class="run-tour-block run-tour-block--output" x="359" y="62" width="118" height="58" rx="13"/><text class="run-tour-label" x="383" y="88">ẋ(t)</text><text class="run-tour-tiny" x="376" y="106">state changes</text>
              <circle class="run-tour-sample" r="5"><animateMotion dur="3.5s" repeatCount="indefinite" path="M141 91H194 M303 91H359 M418 119V157H86V121"/></circle>
              <text class="run-tour-note" x="125" y="187">FIT THE MODEL</text><text class="run-tour-label" x="322" y="187">→</text><text class="run-tour-note" x="346" y="187">PREDICT MOTION</text>
            </svg>''',
        ),
        (
            "The recording behind this run",
            "Sampling cadence and recording length determine which changes are visible to a learner.",
            f"{data_name} contains {samples:,} rows, {interval_note}, and {duration_note}. The scan found {irregular_count} irregular interval(s).",
            "A state trace and an input signal sampled at regular time steps.",
            '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="State curve and control input sampled over time">
              <path class="run-tour-grid" d="M54 52H492 M54 91H492 M54 130H492 M54 169H492"/><path class="run-tour-axis" d="M54 42V177H494"/>
              <path class="run-tour-signal" d="M58 140C100 136 108 80 151 84S207 153 251 122S316 62 355 81S414 144 489 66"/>
              <path class="run-tour-input-wave" d="M58 194H110V178H171V194H225V164H286V194H344V179H409V194H489"/>
              <g class="run-tour-sample-ticks"><path d="M58 177v7 M113 177v7 M168 177v7 M223 177v7 M278 177v7 M333 177v7 M388 177v7 M443 177v7 M489 177v7"/></g>
              <circle class="run-tour-sample" r="5"><animateMotion dur="4.1s" repeatCount="indefinite" path="M58 140C100 136 108 80 151 84S207 153 251 122S316 62 355 81S414 144 489 66"/></circle>
              <text class="run-tour-label" x="60" y="31">STATE x(t)</text><text class="run-tour-label" x="60" y="216">INPUT u(t)</text><text class="run-tour-tiny" x="414" y="216">TIME →</text>
            </svg>''',
        ),
        (
            "What enters and leaves the model",
            "For dynamics learning, the network uses current state and input features to estimate state derivatives; integration turns them into a trajectory.",
            f"This run maps {state_count} state(s) + {input_count} input(s) + time to {state_count} state derivative(s). States: {state_names}. {derivative_note}",
            "State and input features enter the model and estimated state derivatives leave it.",
            '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="State and input features pass through a neural network to state-derivative outputs">
              <text class="run-tour-label" x="28" y="43">STATE + INPUT</text><text class="run-tour-label" x="401" y="43">DERIVATIVES</text>
              <rect class="run-tour-block" x="27" y="64" width="116" height="105" rx="15"/><text class="run-tour-label" x="49" y="98">x(t)</text><text class="run-tour-tiny" x="48" y="117">measured states</text><text class="run-tour-label" x="49" y="145">u(t)</text><text class="run-tour-tiny" x="48" y="162">control inputs</text>
              <path class="run-tour-wire" d="M143 87L219 70 M143 87L219 114 M143 87L219 158 M143 145L219 70 M143 145L219 114 M143 145L219 158 M237 70L310 90 M237 114L310 90 M237 158L310 90 M328 90L400 114"/>
              <circle class="run-tour-neuron" cx="228" cy="70" r="9"/><circle class="run-tour-neuron run-tour-neuron--active" cx="228" cy="114" r="9"/><circle class="run-tour-neuron" cx="228" cy="158" r="9"/><circle class="run-tour-neuron run-tour-neuron--active" cx="319" cy="90" r="11"/>
              <rect class="run-tour-block run-tour-block--output" x="399" y="79" width="92" height="69" rx="14"/><text class="run-tour-label" x="420" y="108">ẋ(t)</text><text class="run-tour-tiny" x="410" y="128">one per state</text>
              <circle class="run-tour-sample" r="5"><animateMotion dur="3.3s" repeatCount="indefinite" path="M84 114L228 114L319 90L445 114"/></circle>
              <text class="run-tour-tiny" x="162" y="203">TIME STEP IS INCLUDED IN THE INPUT</text>
            </svg>''',
        ),
        (
            f"Your {architecture} starting model",
            "An LSTM keeps order across a short sequence. An MLP learns a direct mapping from the current snapshot. Both can estimate system dynamics.",
            layer_note,
            "The selected model architecture for this run, showing its inputs and internal layers.",
            architecture_svg,
        ),
        (
            "How much recent history it sees",
            "A sequence model can use earlier measurements to capture lag and evolving behavior; a direct model uses the current snapshot.",
            temporal_note,
            "Recent ordered samples flow into the selected model architecture.",
            temporal_svg,
        ),
        (
            "Candidate search and results",
            "Each candidate is trained and scored on validation data. Agent feedback uses those scores to suggest the next candidate.",
            search_note,
            "An iterative train, validation, and review loop with the current cycle and measured scores.",
            '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="Candidate training, validation scoring, and agent feedback loop">
              <text class="run-tour-label" x="28" y="37">MODEL SEARCH · REPEAT AND IMPROVE</text>
              <rect class="run-tour-block run-tour-block--model" x="28" y="71" width="126" height="78" rx="16"/><text class="run-tour-label" x="59" y="103">ACTOR</text><text class="run-tour-tiny" x="46" y="124">trains candidate</text>
              <path class="run-tour-wire" d="M154 110H199 M321 110H366 M426 149V183H91V151"/>
              <rect class="run-tour-block" x="199" y="71" width="122" height="78" rx="16"/><text class="run-tour-label" x="224" y="103">VALIDATION</text><text class="run-tour-tiny" x="224" y="124">measured score</text>
              <rect class="run-tour-equation" x="366" y="71" width="122" height="78" rx="16"/><text class="run-tour-label" x="395" y="103">CRITIC</text><text class="run-tour-tiny" x="383" y="124">proposes change</text>
              <circle class="run-tour-sample" r="5"><animateMotion dur="3.8s" repeatCount="indefinite" path="M154 110H199 M321 110H366 M426 149V183H91V151"/></circle>
              <text class="run-tour-note" x="188" y="207">SCORES GUIDE THE NEXT TRY</text>
            </svg>''',
        ),
        (
            "Trajectory boundaries and angle wrapping",
            "Separate experiments should stay separate during splitting. A wrapped angle can jump from +π to −π while the physical motion remains continuous.",
            trajectory_note,
            "Separate runs remain segmented, while a wrapped angle crosses from plus pi to minus pi.",
            ('''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="Three separate experiment traces with clear reset boundaries">
              <text class="run-tour-label" x="29" y="34">STACKED EXPERIMENTS</text>
              <path class="run-tour-grid" d="M105 75H488 M105 125H488 M105 175H488"/>
              <text class="run-tour-tiny" x="29" y="79">RUN 01</text><text class="run-tour-tiny" x="29" y="129">RUN 02</text><text class="run-tour-tiny" x="29" y="179">RUN 03</text>
              <path class="run-tour-signal" d="M109 85C145 83 155 59 193 62S238 92 277 70S333 59 369 75S432 91 486 58"/>
              <path class="run-tour-input-wave" d="M109 136C145 134 155 110 193 113S238 143 277 121S333 110 369 126S432 142 486 109"/>
              <path class="run-tour-signal--purple" d="M109 186C145 184 155 160 193 163S238 193 277 171S333 160 369 176S432 192 486 159"/>
              <path class="run-tour-boundary" d="M99 50V199 M493 50V199"/><text class="run-tour-note" x="157" y="217">BOUNDARIES PRESERVED</text>
            </svg>'''
             if settings.get("multi_trajectory") else
             '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="One continuous trajectory, with an angle signal wrapping across the pi boundary">
               <text class="run-tour-label" x="29" y="34">ONE CONTINUOUS EXPERIMENT</text>
               <path class="run-tour-grid" d="M58 77H489 M58 113H489 M58 149H489"/><path class="run-tour-axis" d="M58 55V158H490"/>
               <text class="run-tour-tiny" x="31" y="81">+π</text><text class="run-tour-tiny" x="39" y="153">−π</text>
               <path class="run-tour-signal" d="M61 139C113 134 145 116 188 106S269 82 314 69L323 65L328 146C367 135 413 121 487 111"/>
               <circle class="run-tour-sample" r="5"><animateMotion dur="3.8s" repeatCount="indefinite" path="M61 139C113 134 145 116 188 106S269 82 314 69L323 65L328 146C367 135 413 121 487 111"/></circle>
               <path class="run-tour-boundary" d="M326 62V151"/><text class="run-tour-note" x="337" y="72">WRAP</text><text class="run-tour-tiny" x="58" y="183">ANGLE IS STORED MODULO 2π</text>
             </svg>'''),
        ),
        (
            "When physics joins the learning",
            physics_tour_text,
            pinn_note,
            "Measured examples and an optional validated equation guide one dynamics model.",
            pinn_svg,
        ),
        (
            "The final held-out check",
            "After tuning, an unseen portion estimates how well the chosen model predicts data it did not train on.",
            f"Current pipeline stage: {stage}. {heldout_status}",
            "Illustrative measured and predicted signals compared on held-out data; the curves are conceptual, not this run's result.",
            '''<svg class="run-tour-svg" viewBox="0 0 520 230" role="img" aria-label="Illustrative measured and predicted curves with a highlighted held-out region">
              <path class="run-tour-grid" d="M46 53H493 M46 91H493 M46 129H493 M46 167H493"/><path class="run-tour-axis" d="M46 43V175H494"/>
              <rect class="run-tour-heldout" x="316" y="47" width="176" height="123" rx="8"/><text class="run-tour-note" x="356" y="65">UNSEEN TEST DATA</text>
              <path class="run-tour-signal" d="M51 136C96 125 105 74 151 82S207 151 252 119S309 69 354 85S418 139 489 75"/>
              <path class="run-tour-prediction" d="M51 143C96 132 105 81 151 90S207 143 252 126S309 77 354 91S418 130 489 82"/>
              <circle class="run-tour-sample" r="5"><animateMotion dur="4s" repeatCount="indefinite" path="M316 97C354 85 418 139 489 75"/></circle>
              <path class="run-tour-legend-observed" d="M73 202h23"/><text class="run-tour-legend" x="103" y="206">measured</text><path class="run-tour-legend-predicted" d="M183 202h23"/><text class="run-tour-legend" x="213" y="206">predicted</text>
            </svg>''',
        ),
    ]
    scene_index = _run_tour_scene_index(elapsed_seconds, tour_state_key, len(scenes))
    title, general, run_note, aria, illustration = scenes[scene_index]
    safe_title = html.escape(title, quote=False)
    safe_general = html.escape(general, quote=False)
    safe_run_note = html.escape(run_note, quote=False)
    safe_aria = html.escape(aria, quote=True)
    illustration = re.sub(r' role="img" aria-label="[^"]*"', "", illustration, count=1)
    heading = '<div class="run-tour-heading"><span>SYSID · YOUR RUN</span></div>'
    art = (
        f'<div class="run-tour-art"><div class="run-tour-title">{safe_title}</div>'
        f'<div class="run-tour-illustration" role="img" aria-label="{safe_aria}">{illustration}</div></div>'
    )
    story = (
        '<div class="run-tour-story">'
        f'<div class="run-tour-general"><small>GENERAL SYSID IDEA</small><p>{safe_general}</p></div>'
        f'<div class="run-tour-specific"><small>IN THIS RUN</small><p>{safe_run_note}</p></div>'
        '</div>'
    )
    return (
        '<section class="run-learning-snapshot" aria-label="System identification and this run">'
        f'{heading}<div class="run-tour-layout">{art}{story}</div></section>'
    )


def _agent_reasoning_from_response(response: str) -> str:
    """Extract only an agent's explicit, user-facing reasoning field."""
    response = (response or "").strip()
    if not response:
        return ""

    json_response = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", response, flags=re.IGNORECASE)
    try:
        parsed = json.loads(json_response)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    def find_reasoning(value) -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).strip().casefold() == "reasoning" and isinstance(item, str):
                    return item.strip()
            for item in value.values():
                found = find_reasoning(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find_reasoning(item)
                if found:
                    return found
        return ""

    reasoning = find_reasoning(parsed)
    if not reasoning:
        match = re.search(
            r"(?ims)^\s*REASONING\s*:\s*(.*?)(?=^\s*[A-Z][A-Z0-9 _/-]{1,40}\s*:|\Z)",
            response,
        )
        reasoning = match.group(1).strip() if match else ""

    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    if len(reasoning) >= 2 and reasoning.startswith("(") and reasoning.endswith(")"):
        reasoning = reasoning[1:-1].strip()
    return reasoning


def _read_agent_prompt_turns(run_dir) -> tuple[list[dict], bool]:
    """Read only explicit reasoning fields from complete agent transcript records."""
    if not run_dir:
        return [], False
    try:
        base = Path(run_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return [], False

    archived_logs = base / "Agents_log"
    try:
        archived_paths = sorted(archived_logs.glob("*")) if archived_logs.is_dir() else []
    except OSError:
        archived_paths = []
    paths = archived_paths or [base / "llm_conversation_history.txt"]
    turns = []
    truncated = False
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if base not in resolved.parents or not resolved.is_file():
            continue
        content, was_truncated = read_evidence_file(resolved)
        truncated = truncated or was_truncated
        for block in re.split(r"(?=TIMESTAMP: \[)", content):
            agent = re.search(r"^AGENT: \[([^\]]+)\]", block, re.MULTILINE)
            if not agent or "PROMPT:" not in block or "RESPONSE:" not in block:
                continue
            # The logger writes each complete turn as one block. Ignore a partial
            # trailing write until its response and usage footer have arrived.
            if "USAGE STATISTICS:" not in block or not re.search(r"={10,}\s*$", block):
                continue
            response_match = re.search(
                r"(?ms)^RESPONSE:\r?\n(.*?)(?=^-{10,}\r?\n^USAGE STATISTICS:)",
                block,
            )
            if not response_match:
                continue
            reasoning = _agent_reasoning_from_response(response_match.group(1))
            if not reasoning:
                continue
            cycle_match = re.search(r"^CYCLE: \[([^\]]+)\]", block, re.MULTILINE)
            timestamp_match = re.search(r"^TIMESTAMP: \[([^\]]+)\]", block, re.MULTILINE)
            turns.append({
                "agent": agent.group(1).strip(),
                "cycle": cycle_match.group(1).strip() if cycle_match else "",
                "timestamp": timestamp_match.group(1).strip() if timestamp_match else "",
                "reasoning": reasoning,
            })
    return turns, truncated


def _activity_id_for_agent_turn(turn: dict) -> str | None:
    agent = re.sub(r"[^a-z0-9]+", "", str(turn.get("agent") or "").casefold())
    cycle_match = re.search(r"\d+", str(turn.get("cycle") or ""))
    cycle = cycle_match.group(0) if cycle_match else ""
    if "inspector" in agent:
        return "stage:Data Inspector"
    if "initializer" in agent:
        return "stage:Initializer Agent"
    if "actor" in agent and cycle:
        return f"actor:{cycle}"
    if "critic" in agent and cycle:
        return f"critic:{cycle}"
    if "explorer" in agent and cycle:
        return f"explorer:{cycle}"
    if "report" in agent:
        return "report"
    return None


def _merge_agent_logs_into_activity(activity_items: list, run_dir) -> list[dict]:
    """Attach each saved agent reasoning note to that agent's activity item."""
    merged = [
        {**row, "details": dict(row.get("details") or {})}
        for row in activity_items if isinstance(row, dict)
    ]
    rows_by_id = {row.get("id"): row for row in merged if row.get("id")}
    turns, _ = _read_agent_prompt_turns(run_dir)
    for index, turn in enumerate(turns, start=1):
        row_id = _activity_id_for_agent_turn(turn)
        row = rows_by_id.get(row_id) if row_id else None
        if row is None:
            agent = str(turn.get("agent") or "Agent").strip()
            cycle = str(turn.get("cycle") or "").strip()
            slug = re.sub(r"[^a-z0-9]+", "-", agent.casefold()).strip("-") or "agent"
            row_id = f"agent-log:{slug}:{cycle or index}"
            row = rows_by_id.get(row_id)
            if row is None:
                suffix = f" · Cycle {cycle}" if cycle else " · Run setup"
                row = {
                    "id": row_id,
                    "timestamp": turn.get("timestamp") or "",
                    "label": f"{agent} log{suffix}",
                    "state": "complete",
                    "details": {},
                }
                merged.append(row)
                rows_by_id[row_id] = row
        row["details"].setdefault("agent_log", []).append(str(turn.get("reasoning") or ""))
    return merged


def _render_completed_run_thinking(run_dir, scope: str) -> None:
    """Show elapsed run time and recorded agent activity under its result summary."""
    items = _merge_agent_logs_into_activity(activity.load(run_dir), run_dir)
    manifest = read_json(Path(run_dir) / "run_manifest.json") or {}
    elapsed = manifest.get("elapsed_seconds")
    if not items and not isinstance(elapsed, (int, float)):
        return
    label = f"Worked for {_format_worked_time(elapsed)}" if isinstance(elapsed, (int, float)) else ":shimmer[Thinking]"
    with st.container(key=f"completed_thinking_{scope}"):
        with st.status(label, state="complete", expanded=False, type="compact"):
            if items:
                activity.render(
                    items, historical=True, scope=f"completed_{scope}", heading="Activity",
                    container_height=320,
                )
            else:
                st.caption("No detailed activity was recorded for this run.")


def _cycle_validation_scores(cycle_history: list) -> list[tuple[int, float]]:
    """Return finite per-candidate validation MSE points in cycle order."""
    points = []
    for position, row in enumerate(cycle_history, start=1):
        if not isinstance(row, dict):
            continue
        try:
            cycle = int(row.get("cycle") or position)
            score = float(row.get("val_mse", row.get("mse")))
        except (TypeError, ValueError):
            continue
        if cycle > 0 and math.isfinite(score) and score >= 0:
            points.append((cycle, score))
    return sorted(points)


def _render_live_cycle_mse_plot(cycle_history: list, cycle_limit: int, chat_id: str) -> None:
    """Show real validation scores after Cycle 1 has completed validation."""
    import pandas as pd

    scores = _cycle_validation_scores(cycle_history)
    with st.container(border=True, key=f"live_cycle_mse_{chat_id}"):
        st.markdown("**Validation MSE by candidate cycle**")
        st.caption("Lower is better · each point is a completed candidate’s validation score.")
        if not scores:
            st.markdown(
                '''<div class="run-mse-empty" role="img" aria-label="Validation MSE plot waiting for the first score">
                  <svg viewBox="0 0 760 174" aria-hidden="true">
                    <path d="M54 28H744 M54 66H744 M54 104H744 M54 142H744"/>
                    <path class="run-mse-empty-axis" d="M54 18V142H744"/>
                    <text x="54" y="164">CYCLE 1</text>
                    <text x="700" y="164">CYCLE ''' + str(max(1, int(cycle_limit))) + '''</text>
                    <text class="run-mse-empty-note" x="250" y="82">WAITING FOR CYCLE 1 VALIDATION SCORE</text>
                  </svg>
                </div>''',
                unsafe_allow_html=True,
            )
            st.caption("The plot will fill in when candidate 1 finishes validation.")
            return

        best_cycle, best_score = min(scores, key=lambda point: point[1])
        latest_cycle, latest_score = scores[-1]
        best_col, latest_col = st.columns(2, gap="small")
        best_col.metric("Best validation MSE", f"{best_score:.5g}", help=f"Candidate cycle {best_cycle}")
        latest_col.metric("Latest validation MSE", f"{latest_score:.5g}", help=f"Candidate cycle {latest_cycle}")
        chart = pd.DataFrame(scores, columns=["Candidate cycle", "Validation MSE"])
        st.line_chart(
            chart,
            x="Candidate cycle",
            y="Validation MSE",
            x_label="Candidate cycle",
            y_label="Validation MSE",
            color="#73d8b4",
            height=220,
            width="stretch",
        )


def _render_agent_logs(activity_items: list, scope: str, run_dir=None) -> None:
    """Show run activity with each saved agent note inside its matching event."""
    activity.render(
        _merge_agent_logs_into_activity(activity_items, run_dir),
        scope=f"live_{scope}",
        heading=None,
        container_height=280,
    )


def _working(chat):
    identity = chat["id"]
    if task := core.REGISTRY.training.get(identity):
        if task["runner"].running:
            state = task["state"]
            checkpoint = state.get("checkpoint")
            with st.chat_message("assistant", avatar=str(ui_brand.ICON_PATH)):
                stage = str(state.get("stage", "Preparing run"))
                milestones = (
                    ("Inspect", {"Questionnaire", "Loading dataset", "Data Inspector"}),
                    ("Tune", {"Splitting trajectories", "Initializer Agent", "Tuning cycles"}),
                    ("Verify", {"Held-out verification"}),
                    ("Package", {"Report & packaging"}),
                )
                active_milestone = next(
                    (index for index, (_, stages) in enumerate(milestones) if stage in stages), 0
                )
                cycle_limit = int((chat.get("settings") or {}).get("max_cycles", 7))
                completed_cycles = list(state.get("history") or [])
                has_cycle_one_score = any(
                    cycle == 1 for cycle, _ in _cycle_validation_scores(completed_cycles)
                )
                activity = list(state.get("activity") or [])
                active_cycles = [
                    int(item["id"].split(":", 1)[1]) for item in activity
                    if str(item.get("id", "")).startswith("actor:") and item.get("state") == "running"
                    and str(item["id"]).split(":", 1)[1].isdigit()
                ]
                current_cycle = max(active_cycles, default=(
                    int(completed_cycles[-1].get("cycle", len(completed_cycles))) if completed_cycles else 0
                ))
                latest = completed_cycles[-1] if completed_cycles else {}
                best_mse = min(
                    (float(item["val_mse"]) for item in completed_cycles
                     if item.get("val_mse") is not None and math.isfinite(float(item["val_mse"]))),
                    default=None,
                )
                stage_copy = {
                    "Preparing run": "Getting the measurements and run settings ready.",
                    "Questionnaire": "Confirming the assumptions for this identification run.",
                    "Loading dataset": "Checking the measurements and preparing the working data.",
                    "Data Inspector": "Reviewing signal quality and useful patterns in the data.",
                    "Splitting trajectories": "Preparing training and held-out data fairly.",
                    "Initializer Agent": "Choosing a sensible first model from the measured data.",
                    "Tuning cycles": "Training candidates and comparing their validation scores.",
                    "Held-out verification": "Checking how the selected model performs on unseen data.",
                    "Report & packaging": "Putting the findings and downloadable files together.",
                }.get(stage, "Working through the identification pipeline.")
                waiting_for_setup = (
                    isinstance(checkpoint, dict) and checkpoint.get("phase") == "initializer"
                )
                if waiting_for_setup:
                    stage_copy = "Review the starting setup below. Candidate 1 begins after you approve it."
                if isinstance(checkpoint, dict) and checkpoint.get("phase") == "early_results":
                    cycle_number = checkpoint.get("cycle", current_cycle)
                    stage_copy = (
                        f"Cycle {cycle_number} is scored. Review its validation results and guide the next candidate, "
                        "or continue with the agents’ proposal."
                    )
                waiting_for_guidance = (
                    isinstance(checkpoint, dict) and checkpoint.get("phase") == "early_results"
                )
                elapsed = max(0.0, time.time() - task["started"])
                with st.container(border=True, key="conversation_live_run_card"):
                    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                        run_label = (
                            "Setup approval needed" if waiting_for_setup else
                            "Cycle results ready" if waiting_for_guidance else
                            "LabCD is working"
                        )
                        st.markdown(
                            f'<div class="run-live-label"><span class="run-live-dot"></span> {run_label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{workspace.duration(elapsed)} elapsed")
                    st.markdown("### System ID · this run")
                    st.caption(stage_copy)
                    candidate_started = not waiting_for_setup and (current_cycle > 0 or bool(completed_cycles))
                    active_item = next(
                        (item for item in reversed(activity) if item.get("state") == "running"), {}
                    )
                    active_item_id = str(active_item.get("id", ""))
                    active_agent = active_item_id.split(":", 1)[0] if active_item_id else ""
                    current_update = next(
                        (item["label"] for item in reversed(activity) if item.get("state") == "running"),
                        activity[-1].get("label", "Run started") if activity else "Preparing the run",
                    )
                    with st.container(border=True, key="run_details_card"):
                        st.markdown(
                            _run_process_diagram(stage, active_milestone, active_agent, current_cycle, cycle_limit),
                            unsafe_allow_html=True,
                        )
                        with st.container(horizontal=True, gap="small", wrap=True):
                            st.metric("Search cycles", f"{current_cycle} / {cycle_limit}")
                            st.metric("Best validation MSE", f"{best_mse:.5g}" if best_mse is not None else "Waiting for first score")
                            latest_cycle = latest.get("cycle") or current_cycle
                            st.metric("Latest candidate", f"Cycle {latest_cycle}" if latest_cycle else "Preparing")
                    if has_cycle_one_score:
                        _render_live_cycle_mse_plot(completed_cycles, cycle_limit, chat["id"])
                    else:
                        tour_state_key = _run_tour_state_key(chat, task)
                        st.markdown(
                            _run_learning_snapshot(
                                elapsed, chat, state, current_cycle, cycle_limit, best_mse, stage,
                                tour_state_key,
                            ),
                            unsafe_allow_html=True,
                        )
                        _render_run_tour_dots(elapsed, tour_state_key, 9)
                    prompt_log_dir = getattr(task["runner"], "run_dir", None)
                if waiting_for_setup:
                    st.caption("Starting setup review")
                    decision = init_ui.render(
                        checkpoint, f"initial_review_{chat['id']}_{checkpoint.get('id', 'initializer')}"
                    )
                    if decision is not None:
                        action, edits = decision
                        _dismiss_pending_question_timer(chat)
                        _respond_to_human_checkpoint(task, action, edits=edits)
                with st.expander("Training log", expanded=False):
                    st.code(state["log"][-12000:], language="text", height=250)
                if candidate_started:
                    _talk_to_agent_section(chat, task, state)
                thinking = st.status(
                    ":shimmer[Thinking]", state="running", expanded=False, type="compact",
                )
                with thinking:
                    thinking.caption(f"Now · {current_update}")
                    _render_agent_logs(activity, chat["id"], prompt_log_dir)
                # A context-managed status closes as complete on each rerun;
                # restore its live state so the disclosure keeps its spinner.
                thinking.update(label=":shimmer[Thinking]", state="running", expanded=False)
                if candidate_started:
                    _render_cycle_checkpoint_controls(chat, task, checkpoint)
            return
    job = (core.REGISTRY.conversations.get(identity) or core.REGISTRY.planning.get(identity)
           or core.REGISTRY.diagnostics.get(identity))
    if job:
        elapsed = _response_elapsed(job)
        with st.chat_message("assistant", avatar=str(ui_brand.ICON_PATH)):
            with st.container(key=f"message_activity_live_{identity}"):
                thinking = st.status(
                    f":shimmer[Thinking] · {_format_thinking_time(elapsed)}",
                    state="running", expanded=False, type="compact"
                )
                thinking.caption(f"Now · {_client_progress_text(job, 'Reviewing your request')}")


@st.fragment(run_every=1.0)
def _live_thread(chat, hooks):
    _sync(chat, hooks)
    if not _busy(chat):
        st.rerun(scope="app")
    _messages(chat, True)
    _working(chat)


def _welcome(chat):
    """Show useful first-chat guidance without putting a description form in the page."""
    title, description = _conversation_copy(chat, SYSTEM_WELCOME_COPIES)
    st.markdown(f'''<div class="conversation-welcome"><div class="welcome-glyph" aria-hidden="true">{ui_brand.ICON_MARKUP}</div>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(description)}</p>
      </div>''', unsafe_allow_html=True)
    suggestions = [
        ("Identify a system", "Help me identify a model for my system. What should I share first?", "add_chart"),
        ("How do I make an Excel?", "How do I make an Excel workbook with LabCD?", "table_view"),
        ("LSTM, MLP, PINN, or PCNN?", "Explain the difference between LSTM and MLP for system identification, and how PINN and PCNN add physics constraints.", "account_tree"),
        ("Plan an experiment", "Help me plan a system identification experiment. Ask only for what we need, and let me skip details I don't know.", "science"),
    ]
    with st.container(horizontal=True, horizontal_alignment="center", key="conversation_suggestions"):
        for label, prompt, icon in suggestions:
            if st.button(label, icon=f":material/{icon}:", key=f"suggest_{'table_view' if icon == 'table_view' else icon}"):
                st.session_state["conversation_pending"] = prompt
                st.rerun()


def _excel_help_sheet(message_id):
    import pandas as pd

    example = pd.DataFrame({
        "time": [0.00, 0.10, 0.20, 0.30],
        "s_1": [1.178, 1.273, 0.198, -1.060],
        "s_2": [1.910, 0.296, -1.589, -2.014],
        "a_1": [0.395, -2.119, -2.685, -0.782],
        "xdot_1": [0.950, -4.900, -11.665, -12.580],
        "xdot_2": [-16.140, -17.495, -11.550, -4.250],
    })
    with st.container(border=True, key=f"excel_help_sheet_{message_id}"):
        title, badge = st.columns([1.5, 1], vertical_alignment="center")
        title.markdown("**Example Excel sheet**")
        badge.caption("EXAMPLE · 2 STATES · 1 INPUT")
        st.dataframe(example, width="stretch", height=190, hide_index=True)
        st.caption("We want a time rate for every state: `xdot_1` matches `s_1`, and `xdot_2` matches `s_2`. When a source file does not include these rates, LabCD estimates them from the state measurements.")
        st.caption("This is only an example. Send your own data and I’ll make the workbook from those values.")


def render_app(*, runner, drain):
    hooks = dict(runner=runner, drain=drain)
    if "conversation" not in st.session_state:
        existing = core.read_chat(st.query_params.get("chat", ""))
        st.session_state["conversation"] = existing or core.new_chat()
    chat = st.session_state["conversation"]
    ui_onboarding.render_onboarding()
    _sync(chat, hooks)
    if not _busy(chat) and not chat.get("run_dir"):
        _prepare_saved_pinn_when_ready(chat)
    if (not _busy(chat) and not chat.get("run_dir") and
            (chat.get("setup") or {}).get("stage") == "complete"):
        _start_run_setup_agent(chat)
    busy = _busy(chat)
    file_items = _file_items(chat)
    _sidebar(chat)
    _header(chat, busy, has_files=bool(file_items))
    training = core.REGISTRY.training.get(chat["id"])
    training_active = bool(training and training["runner"].running)
    active = core.REGISTRY.active_training()
    if active and active[0] != chat["id"]:
        st.caption("A run is training in another conversation. Your chat is available while it works.")
        if st.button("Open active run", key="open_active_conversation"):
            _select(core.read_chat(active[0]) or chat)
            st.rerun()
    show_files = bool(st.session_state.get("conversation_files")) and bool(file_items)
    conversation = st.container(key="conversation_center")
    with conversation:
        if not chat["messages"]:
            _welcome(chat)
        elif busy:
            _live_thread(chat, hooks)
        else:
            _messages(chat, False, file_items=file_items)
    if show_files:
        _artifacts(chat, file_items)
    with st.bottom:
        with st.container(key="conversation_composer"):
            dataset = chat.get("dataset") or {}
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                if dataset:
                    st.caption(f":material/attach_file: {workspace.literal(dataset['name'])}")
                if training_active:
                    st.caption("Open Talk to LabCD under the agent activity to guide the next cycle.")
                    if st.button("Stop run", icon=":material/stop:", key="conversation_stop"):
                        training["runner"].request_stop()
                        core.add_message(chat, "assistant", "Stop requested. LabCD will finish the current training epoch, score that candidate, and package the best result without starting another candidate.")
                        st.rerun()
                else:
                    st.caption("Ask about this run" if chat.get("run_dir") else f"{chat['settings']['architecture']} · {chat['settings']['max_cycles']} cycles")
                    if not chat.get("run_dir"):
                        if dataset.get("ready"):
                            flow = chat.get("run_setup_flow") or {}
                            settings_saved = (flow.get("dataset_sha256") == dataset.get("sha256")
                                              and flow.get("stage") == "approved")
                            if settings_saved:
                                if st.button("Start training", type="primary", icon=":material/play_arrow:",
                                             key=f"composer_start_{chat['id']}", disabled=busy):
                                    st.session_state["conversation_pending"] = "start"
                                    st.rerun()
                            else:
                                st.caption("Review the setup card above; use a pen to edit a setting.")
                        elif dataset.get("column_review_pending"):
                            st.caption("Confirm the time, state, and input columns above before starting.")
                        elif dataset:
                            st.caption("Resolve the dataset checks above before starting.")
                        else:
                            st.caption("Upload a CSV or Excel dataset before starting a run.")
            value = st.chat_input("Ask LabCD, attach a dataset, or add a physics source (.py, .m, or .txt)…" if not chat.get("run_dir") else "Ask about the results, agents, or your next experiment…",
                                  accept_file=not bool(chat.get("run_dir")), file_type=["csv", "xlsx", "xls", "py", "m", "txt"], max_upload_size=25,
                                  max_chars=6000, disabled=busy, key=f"composer_{chat['id']}")
            st.caption("LabCD can make mistakes. Check the source details behind important conclusions.")
    pending = st.session_state.pop("conversation_pending", None)
    pending_source_file = st.session_state.pop("conversation_pending_source_file", None)
    if value or pending or pending_source_file:
        text = pending or (value if isinstance(value, str) else value.text if value else "Prepare my attached equation source")
        uploads = [] if pending or isinstance(value, str) or value is None else list(value.files)
        if pending_source_file is not None:
            uploads.insert(0, pending_source_file)
        _submit(chat, text, uploads, hooks)
        st.query_params["chat"] = chat["id"]
        st.rerun()
