"""Conversation-first interface for the existing AgentSysID pipeline."""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import mimetypes
import re
import time
import zipfile
from pathlib import Path

import streamlit as st
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticJob, DiagnosticSettings, load_chat
from backend_core.AgentSysID.agents.run_evidence import read_json
from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.utils import request_stop

try:
    from . import conversation_core as core, conversation_analysis as analysis, conversation_agent as agent
    from . import ui_data_analysis as data_ui, ui_history as hist, ui_activity as activity
    from . import excel_maker, pinn_maker, run_setup_agent
    from . import ui_results_workspace as workspace, ui_run_chat as diagnosis_ui
except ImportError:
    import conversation_core as core
    import conversation_analysis as analysis
    import conversation_agent as agent
    import ui_data_analysis as data_ui
    import ui_history as hist
    import ui_activity as activity
    import excel_maker
    import pinn_maker
    import run_setup_agent
    import ui_results_workspace as workspace
    import ui_run_chat as diagnosis_ui


def _literal(text):
    st.markdown("\n".join(workspace.literal(line) for line in str(text).splitlines()))


def _count(number, noun):
    return f"{number} {noun if number == 1 else noun + 's'}"


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


def _new():
    _select(core.new_chat())


def _entry(chat):
    if not chat.get("run_dir"):
        return None
    entry = read_json(Path(chat["run_dir"]) / "run_manifest.json") or {}
    entry.update(run_dir=chat["run_dir"], name=Path(chat["run_dir"]).name,
                 complete=entry.get("status") == "completed")
    return entry


def _result_summary(entry):
    if entry.get("status") != "completed":
        return "This run did not complete. " + (entry.get("message") or "We can investigate the recorded activity and available evidence.")
    mse, latency = workspace.number(entry, "best_mse"), workspace.number(entry, "latency_ms")
    error = f"validation MSE {mse:.4g}" if mse is not None else "validation error not recorded"
    timing = f" and measured inference latency {latency:.3f} ms" if latency is not None else ""
    return (f"The run finished after {entry.get('cycles_run', '—')} cycles, with {error}{timing}. "
            "I’ve added the learning curve, per-state scores, and saved diagnostic plots below. You can inspect each candidate or ask a follow-up about the evidence.")


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


def _sidebar(chat):
    with st.sidebar:
        st.markdown('<div class="conversation-brand"><span class="labcd-symbol">L</span> LabCD <span class="brand-muted">workspace</span></div>', unsafe_allow_html=True)
        st.button("New conversation", icon=":material/edit_square:", width="stretch", on_click=_new, key="conversation_new")
        query = st.text_input("Search conversations", placeholder="Search conversations", label_visibility="collapsed", key="conversation_search")
        with st.popover("Library", icon=":material/inventory_2:", width="stretch"):
            archived = st.toggle("Show archived conversations", key="conversation_archived")
            st.caption("Conversations and run evidence are saved in this project.")
        chats = core.list_chats()
        visible = [c for c in chats if bool(c.get("archived")) == archived and query.lower() in c["title"].lower()]
        st.caption("CONVERSATIONS")
        if not visible:
            st.caption("Your conversations will appear here." if not query else "No matching conversations.")
        for saved in visible[:60]:
            selected = saved["id"] == chat["id"]
            with st.container(key=f"conversation_nav_{'selected_' if selected else ''}{saved['id']}"):
                st.button(saved["title"], icon=":material/chat_bubble_outline:", width="stretch",
                          key=f"conversation_open_{saved['id']}", on_click=_select, args=(saved,), help=saved["title"])
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


def _run_setup_card(chat, message, busy):
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    if flow.get("dataset_sha256") != dataset.get("sha256"):
        st.caption("This setup recommendation belongs to an earlier dataset. I’ll prepare a fresh one for the current measurements.")
        return
    settings = core.RunSettings.model_validate(chat["settings"]).model_dump()
    stage = flow.get("stage", "model")
    suffix = f"{chat['id']}_{dataset.get('sha256', '')[:8]}"
    with st.container(border=True, key=f"run_setup_card_{suffix}"):
        if stage in ("model", "effort"):
            _setup_animation(stage)
        if stage == "model":
            st.caption("STEP 1 OF 2 · MODEL CHOICE")
            if flow.get("agent_error"):
                st.warning(flow["agent_error"])
                st.markdown("**Choose a starting architecture.** You can switch it after the first run as well.")
            else:
                st.markdown(f"**Run setup agent recommends {flow['recommended_architecture']}.**")
                st.write(flow["architecture_reason"])
                if flow.get("confidence"):
                    st.caption(f"Recommendation confidence · {flow['confidence']}. This is a starting choice, not a claim that it will score better.")
            architecture_key = f"setup_architecture_{suffix}"
            st.session_state.setdefault(architecture_key, settings["architecture"])
            architecture = st.segmented_control(
                "Starting model", ["LSTM", "MLP"], key=architecture_key,
                format_func=lambda value: "LSTM · sequence memory" if value == "LSTM" else "MLP · direct mapping",
                required=True, width="stretch")
            left, right = st.columns(2, gap="small")
            with left, st.container(border=True):
                st.markdown("**LSTM · remembers sequences**")
                st.caption("Use earlier measurements as context. A useful first test when current states and inputs may not capture all relevant history.")
            with right, st.container(border=True):
                st.markdown("**MLP · learns snapshots**")
                st.caption("Maps the current state and input directly. A simpler first test when those measurements describe the dynamics well.")
            history_key = f"setup_history_{suffix}"
            st.session_state.setdefault(history_key, settings["lstm_seq_length"])
            history = st.number_input("Past time steps · LSTM history", min_value=2, max_value=500,
                                       disabled=architecture != "LSTM", key=history_key,
                                       help="At 0.01 seconds per sample, 20 steps cover 0.2 seconds.")
            interval = (dataset.get("analysis") or {}).get("median_dt")
            if architecture == "LSTM" and interval and interval > 0:
                st.caption(f"At the measured interval of {interval:.4g} s, this history spans about {(history - 1) * interval:.4g} s.")
            pinn_ready = pinn_maker.is_equation_ready(chat)
            pinn_key = f"setup_pinn_{suffix}"
            st.session_state.setdefault(pinn_key, settings["use_pinn"] if pinn_ready else False)
            with st.container(border=True, key=f"setup_pinn_panel_{suffix}"):
                status_col, badge_col = st.columns([3, 1], vertical_alignment="center")
                status_col.markdown("**Physics-informed training**")
                badge_col.badge("Equation ready" if pinn_ready else "Optional",
                                icon=":material/science:", color="green" if pinn_ready else "gray")
                pinn_enabled = st.checkbox("Use the equation as a training constraint (PINN)",
                                           key=pinn_key, disabled=not pinn_ready)
                if pinn_ready:
                    state_count = len((chat.get("pinn_equation") or {}).get("states") or [])
                    source_name = (chat.get("pinn_equation") or {}).get("source_name") or "Saved equation"
                    st.caption(f"{source_name} · checked against {state_count} measured state{'s' if state_count != 1 else ''}. "
                               "PINN can pair with either model.")
                    weight_key = f"setup_pinn_weight_{suffix}"
                    st.session_state.setdefault(weight_key, float(settings["pinn_loss_weight"]))
                    pinn_weight = st.number_input("Physics loss strength", min_value=0.0, max_value=10000.0,
                                                  step=0.05, key=weight_key,
                                                  help="Controls how strongly the validated equation contributes to training.")
                else:
                    pinn_weight = settings["pinn_loss_weight"]
                    st.caption("Attach a physics equation or MATLAB .m file in the chat. I’ll map it to your confirmed columns and check it before PINN can be enabled.")
            context_key = f"setup_context_{suffix}"
            st.session_state.setdefault(context_key, settings["customer_description"])
            context = st.text_area("Optional: what device or process produced these signals, and what units are they in?",
                                   key=context_key, placeholder="For example: motor position in radians; torque in N·m.",
                                   help="Skip this if you do not know. It helps interpret results but is not required to train.")
            if st.button("Continue to search effort", type="primary", icon=":material/arrow_forward:",
                         key=f"setup_continue_effort_{suffix}", disabled=busy):
                try:
                    chat["settings"] = core.apply_changes(chat["settings"], {
                        "architecture": architecture, "lstm_seq_length": history,
                        "use_pinn": bool(pinn_enabled), "pinn_loss_weight": pinn_weight,
                        "customer_description": context,
                    })
                    flow["stage"] = "effort"
                    flow["selected_architecture"] = architecture
                    core.save_chat(chat)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        elif stage == "effort":
            st.caption(f"STEP 2 OF 2 · {settings['architecture']} SELECTED")
            if flow.get("effort_reason"):
                st.markdown(f"**Run setup agent’s search suggestion · {flow['recommended_effort'].title()}**")
                st.write(flow["effort_reason"])
            modes = ["fast", "regular", "heavy"]
            mode_key = f"setup_effort_{suffix}"
            st.session_state.setdefault(mode_key, flow.get("recommended_effort", settings["run_mode"] if settings["run_mode"] in modes else "regular"))
            mode = st.segmented_control("Search effort", modes, key=mode_key, format_func=str.title,
                                        required=True, width="stretch")
            limits = cfg.run_mode_limits(mode)
            cap = int(limits["max_cycles"])
            details = {
                "fast": "Short search with compact agent memory.",
                "regular": "Balanced search with recent failure history.",
                "heavy": "Broad search with full failure history.",
            }
            max_hours = float(limits["max_hours"])
            time_limit = f"{max_hours * 60:g} minutes" if max_hours < 1 else f"{max_hours:g} hours"
            st.caption(f"{details[mode]} Up to {cap} candidate cycles or {time_limit}; whichever comes first.")
            cycle_key = f"setup_cycles_{suffix}"
            default_cycles = int(flow.get("recommended_cycles", settings["max_cycles"]))
            if cycle_key not in st.session_state:
                st.session_state[cycle_key] = min(max(default_cycles, 1), cap)
            else:
                st.session_state[cycle_key] = min(max(int(st.session_state[cycle_key]), 1), cap)
            cycles = st.slider("Candidate cycles", min_value=1, max_value=cap, step=1, key=cycle_key,
                               help="Each cycle proposes and evaluates one candidate model.")
            st.caption(f"Selected search · **{mode.title()}**, up to **{cycles} cycles**.")
            epoch_key = f"setup_epochs_{suffix}"
            st.session_state.setdefault(epoch_key, settings["epochs"])
            epochs = st.number_input("Starting epochs per cycle", min_value=1, max_value=5000, key=epoch_key)
            first, second = st.columns(2)
            mse_key = f"setup_mse_{suffix}"
            latency_key = f"setup_latency_{suffix}"
            st.session_state.setdefault(mse_key, settings["mse_target"])
            st.session_state.setdefault(latency_key, settings["customer_max_latency_ms"])
            mse = first.number_input("Validation MSE target", min_value=0.00000001, format="%.8f", key=mse_key)
            latency = second.number_input("Inference time budget (ms)", min_value=0.001, key=latency_key)
            with st.expander("Advanced settings"):
                st.caption("Optional training and data-processing controls. The defaults are ready for a first run.")
                advanced_key = f"setup_advanced_{suffix}"
                st.session_state.setdefault(advanced_key, json.dumps(settings, indent=2))
                advanced = st.text_area("Advanced settings JSON", height=230, key=advanced_key)
            with st.container(horizontal=True, horizontal_alignment="distribute", gap="small"):
                if st.button("Back to model", key=f"setup_back_model_{suffix}", disabled=busy):
                    flow["stage"] = "model"
                    core.save_chat(chat)
                    st.rerun()
                if st.button("Approve search effort", type="primary", icon=":material/check:",
                             key=f"setup_approve_{suffix}", disabled=busy):
                    try:
                        updates = json.loads(advanced)
                        updates.update(run_mode=mode, max_cycles=cycles, epochs=epochs,
                                       mse_target=mse, customer_max_latency_ms=latency)
                        chat["settings"] = core.apply_changes(chat["settings"], updates)
                        flow["stage"] = "approved"
                        flow["approved_settings"] = {
                            "architecture": chat["settings"]["architecture"],
                            "run_mode": mode, "max_cycles": cycles,
                        }
                        core.save_chat(chat)
                        st.rerun()
                    except (ValueError, TypeError) as exc:
                        st.error(f"Check the advanced settings and ranges. Nothing was approved: {exc}")
        elif stage == "approved":
            st.caption("SETUP APPROVED · READY WHEN YOU ARE")
            st.markdown(f"**{settings['architecture']} · {settings['run_mode'].title()} search · up to {settings['max_cycles']} cycles**")
            if settings["architecture"] == "LSTM":
                st.caption(f"Using {settings['lstm_seq_length']} past time steps as sequence history.")
            if settings.get("use_pinn"):
                st.badge("PINN selected · validated equation", icon=":material/science:", color="green")
                st.caption(f"Physics loss strength · {settings['pinn_loss_weight']:g}")
            else:
                st.badge("Data-driven training · PINN off", icon=":material/model_training:", color="gray")
            st.caption("You can keep discussing the plan, revise either choice, or start the run.")
            with st.container(horizontal=True, gap="small"):
                if st.button("Change model", key=f"setup_edit_model_{suffix}", disabled=busy):
                    flow["stage"] = "model"
                    core.save_chat(chat)
                    st.rerun()
                if st.button("Change search effort", key=f"setup_edit_effort_{suffix}", disabled=busy):
                    flow["stage"] = "effort"
                    core.save_chat(chat)
                    st.rerun()
                if st.button("Start run", type="primary", icon=":material/play_arrow:",
                             key=f"setup_start_{suffix}", disabled=busy or not dataset.get("ready")):
                    st.session_state["conversation_pending"] = "start"
                    st.rerun()


@st.dialog("Compare runs", width="large")
def _compare_dialog(chat):
    workspace.render_compare(str(core.OUTPUT_DIR), preferred=chat.get("run_dir"))


def _header(chat, busy, has_files=False):
    with st.container(key="conversation_header"):
        title, controls = st.columns([3, 4], vertical_alignment="center")
        title.markdown(f'<div class="conversation-heading">LabCD <span>/</span> {html.escape(chat["title"])}</div>', unsafe_allow_html=True)
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
        model_label = settings["architecture"] + (" + PINN" if settings.get("use_pinn") else "")
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
                st.caption("Review the model and search effort in the setup card before starting.")
            else:
                st.caption("Say “start” to begin, or tell me what to change.")
            if st.button("Start run", type="primary", icon=":material/play_arrow:", key=f"start_{key}",
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
    with st.expander("Run activity", icon=":material/account_tree:"):
        activity.render(activity.load(run_dir), historical=True, scope=key)


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
            "binary": "memory", "text": "description", "file": "draft"}.get(kind, "draft")


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
                          "subtitle": f"PINN source · original preserved · {status_label}",
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
        subtitle = (f"Validated PINN equation · {len(artifact.get('states') or []):,} states" if artifact_kind == "pinn_equation"
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
        paths = list(hist.figures_of(entry))
        paths.extend(hist.existing_files(entry).values())
        try:
            paths.extend(str(path) for path in Path(run_dir).rglob("*")
                         if path.is_file() and not path.name.startswith(".") and
                         path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".py",
                                                 ".pth", ".pt", ".pkl", ".pickle", ".onnx", ".zip", ".txt",
                                                 ".md", ".log", ".json", ".jsonl", ".yaml", ".yml", ".toml",
                                                 ".html", ".xml", ".ini", ".cfg", ".csv", ".tsv", ".xlsx", ".xls"})
        except OSError:
            pass
        for path in paths:
            resolved = str(Path(path).resolve())
            if resolved in seen or not Path(resolved).is_file():
                continue
            seen.add(resolved)
            kind = _file_kind(resolved)
            size = Path(resolved).stat().st_size
            size_label = f"{size / 1024:.0f} KB" if size >= 1024 else f"{size} B"
            items.append({"id": "artifact:" + hashlib.sha1(resolved.encode()).hexdigest()[:16],
                          "kind": kind, "label": _artifact_label(resolved), "path": resolved,
                          "file_name": Path(resolved).name, "subtitle": f"{Path(resolved).suffix[1:].upper() or 'FILE'} · {size_label}",
                          "icon": _file_icon(kind)})
    return items


def _open_file(item):
    st.session_state["conversation_selected_file"] = item["id"]
    st.session_state["conversation_files"] = True


def _file_card(item, key):
    with st.container(border=True, key=f"chat_file_card_{key}"):
        st.button(item["label"], icon=f":material/{item['icon']}:", width="stretch",
                  key=f"open_chat_file_{key}", help="Open this file in the right preview panel",
                  on_click=_open_file, args=(item,))
        st.caption(item["subtitle"])


def _file_cards(items, scope):
    if not items:
        return
    st.markdown("**Files from this run**")
    for start in range(0, len(items), 2):
        columns = st.columns(2, gap="small")
        for column, item in zip(columns, items[start:start + 2]):
            with column:
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
    with st.container(border=True, key="conversation_file_panel", height=680):
        selected_id = st.session_state.get("conversation_selected_file")
        selected = next((item for item in items if item["id"] == selected_id), None)
        title_col, close_col = st.columns([5, 1], vertical_alignment="center")
        if selected:
            if title_col.button("All files", icon=":material/arrow_back:", key="file_panel_back"):
                st.session_state["conversation_selected_file"] = None
                st.rerun()
            close_col.button("Close", icon=":material/close:", key="file_panel_close",
                             on_click=lambda: st.session_state.update(conversation_files=False, conversation_selected_file=None))
            st.markdown(f"#### {workspace.literal(selected['label'])}")
            st.caption(f"{selected.get('file_name') or Path(selected['path']).name} · {selected['subtitle']}")
            preview_key = hashlib.sha1(selected["id"].encode()).hexdigest()[:12]
            _preview_file(selected, preview_key)
        else:
            title_col.markdown("#### Files")
            close_col.button("Close", icon=":material/close:", key="file_panel_close_list",
                             on_click=lambda: st.session_state.update(conversation_files=False, conversation_selected_file=None))
            st.caption("Uploaded datasets and generated run files")
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
            " The choices are below; if they look right, reply `yes, that's ok`. Or tell me a header or position, such as `the fourth column is the state`, and I’ll fix the working copy. "
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
        core.add_message(chat, "assistant", "Your columns are confirmed. I’m now mapping the saved PINN source to those measurements and checking it before enabling PINN.")
        _start_pinn_preparation(chat, "Prepare the saved PINN source using the just-confirmed state and input columns.")
        return
    prompt = ("Analyze this uploaded dataset in depth. Describe the measured signals, sampling and segmentation, "
              "input variation, derivative availability, and what still needs client confirmation before training.")
    _start_data_review(chat, prompt)


def _start_data_review(chat, prompt):
    core.begin_setup(chat)
    job = agent.ConversationJob(chat, prompt, purpose="data_review")
    core.REGISTRY.conversations[chat["id"]] = job
    job.start()


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
        chat, "Prepare the saved PINN source using the attached dataset's confirmed state and input columns.")


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
            "Input or control columns · optional, renamed with a_",
            options=options, default=input_defaults,
            key=f"column_review_inputs_{key}",
            help="Choose signals applied to the system, such as force, voltage, or command.")
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
    latest_plan = next((m["id"] for m in reversed(chat["messages"]) if m.get("kind") == "plan"), None)
    latest_setup = next((m["id"] for m in reversed(chat["messages"]) if m.get("kind") == "setup_question"), None)
    for message in chat["messages"]:
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/graphic_eq:"):
            if message.get("diagnosis"):
                diagnosis_ui.render_answer(message["diagnosis"])
            elif message.get("evidence") or message.get("kind") in ("excel_maker", "pinn_maker", "column_review", "column_correction", "pasted_table_preview", "run_setup"):
                # Model prose is Markdown; Streamlit escapes raw HTML by default.
                safe = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", message["content"])
                st.markdown(safe)
            else:
                _literal(message["content"])
            if attachment := message.get("attachment"):
                uploaded = next((item for item in file_items
                                 if item["id"] == f"upload:{message.get('id')}" or
                                 (item.get("path") and attachment.get("path") and
                                  Path(item["path"]).resolve() == Path(attachment["path"]).resolve())), None)
                if uploaded:
                    _file_card(uploaded, f"upload_{message['id']}")
                if attachment.get("kind") == "pinn_source":
                    st.caption("Original equation source, preserved unchanged. Open it from the Files panel to inspect or download it.")
                else:
                    with st.expander("Dataset details"):
                        st.write("Columns: " + ", ".join(attachment.get("columns") or []))
                        if attachment.get("column_mapping"):
                            st.caption("Confirmed working-copy mapping: " + ", ".join(
                                f"{old} → {new}" for old, new in attachment["column_mapping"].items()))
                        if attachment.get("sample_period") is not None:
                            st.caption(f"Median sampling interval: {attachment['sample_period']:.6g} seconds")
                        for issue in attachment.get("issues") or []:
                            st.warning(issue)
            if artifact := message.get("generated_artifact"):
                generated = next((item for item in file_items
                                  if item["id"] == _generated_file_id(artifact.get("path", ""))), None)
                if generated:
                    if artifact.get("kind") == "pinn_equation":
                        st.markdown("**PINN equation · mapped and checked against this dataset**")
                        _file_card(generated, f"pinn_{message['id']}")
                        equations = artifact.get("equations") or []
                        if equations:
                            st.dataframe(equations, width="stretch", hide_index=True,
                                         column_config={"state": "Measured state", "alias": "Equation variable",
                                                        "expression": "Derivative dx/dt"})
                        variables = artifact.get("state_aliases", []) + artifact.get("action_aliases", [])
                        if variables:
                            st.caption("Variable map · " + " · ".join(
                                f"`{item['alias']}` = `{item['column']}`" for item in variables))
                        st.caption("The generated file is available in the right Files panel and will be passed to the next PINN training run.")
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
            if message.get("kind") == "plan":
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
                    "What evidence points to the cause?",
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
                data_ui.render_setup_question(chat, message["id"],
                                              not busy and message["id"] == latest_setup and
                                              (chat.get("setup") or {}).get("stage") == message.get("stage"))
            if message.get("kind") == "run_setup":
                _run_setup_card(chat, message, busy)
            if message.get("state_comparison"):
                data_ui.render_state_comparison(message["state_comparison"])
            if message.get("visual_question") and not message.get("state_comparison"):
                data_ui.render_answer_visuals(
                    message["visual_question"], message["id"],
                    dataset=chat.get("dataset"), run_dir=chat.get("run_dir"))
            if message.get("evidence"):
                with st.expander("Evidence used"):
                    for source in message["evidence"]:
                        st.caption(f"[{source['id']}] {source['title']}")
            if message.get("error_detail"):
                with st.expander("Connection details"):
                    st.text(message["error_detail"])


def _save_setup_recommendation(chat, recommendation=None, error=None, error_detail=None):
    dataset = chat.get("dataset") or {}
    settings = chat["settings"]
    if recommendation:
        effort = recommendation["search_effort"]
        cycles = min(int(recommendation["cycles"]), int(cfg.run_mode_limits(effort)["max_cycles"]))
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
        }
        content = (f"I looked over the measured-data profile and prepared {recommendation['architecture']} as a starting point. "
                   "You can change the model and search effort below. Nothing starts until you choose to approve the setup and start the run; its held-out results will give us the first measured comparison.")
    else:
        flow = {
            "dataset_sha256": dataset.get("sha256"),
            "stage": "model",
            "recommended_architecture": settings["architecture"],
            "architecture_reason": "Choose the first model from the measured data. The current selection is only a starting point; compare held-out predictions after training.",
            "recommended_history_steps": settings["lstm_seq_length"],
            "recommended_effort": settings["run_mode"] if settings["run_mode"] in ("fast", "regular", "heavy") else "regular",
            "recommended_cycles": min(int(settings["max_cycles"]), int(cfg.run_mode_limits(settings["run_mode"])["max_cycles"])),
            "effort_reason": "Select a search budget that fits your time and desired breadth.",
            "confidence": "low",
            "agent_error": error or "The setup recommendation is unavailable.",
        }
        content = "I couldn’t get a model recommendation this time. You can choose a starting model below, or ask me what would make sense for these measurements. We’ll review the search effort before anything starts."
    chat["run_setup_flow"] = flow
    core.save_chat(chat)
    core.add_message(chat, "assistant", content, kind="run_setup",
                     **({"error_detail": error_detail} if error_detail else {}))


def _sync(chat, hooks):
    identity, registry = chat["id"], core.REGISTRY
    if job := registry.conversations.get(identity):
        if not job.running:
            if job.answer:
                extra = {"evidence": job.answer["evidence"], "model": job.answer["model"]}
                if (chat.get("run_dir") or chat.get("dataset")) and job.purpose not in ("excel_maker", "pinn_maker"):
                    extra["visual_question"] = job.question
                if job.purpose == "excel_maker":
                    extra["kind"] = "excel_maker"
                    extra["generated_artifact"] = job.answer["generated_artifact"]
                if job.purpose == "pinn_maker":
                    extra["kind"] = "pinn_maker"
                    artifact = job.answer.get("generated_artifact")
                    if artifact:
                        extra["generated_artifact"] = artifact
                        chat["pinn_equation"] = artifact
                        if isinstance(chat.get("pinn_source"), dict):
                            chat["pinn_source"]["status"] = "ready"
                        chat["settings"]["use_pinn"] = True
                        st.session_state[f"run_setup_{chat['id']}_use_pinn"] = True
                        setup_suffix = f"{chat['id']}_{(chat.get('dataset') or {}).get('sha256', '')[:8]}"
                        st.session_state[f"setup_pinn_{setup_suffix}"] = True
                        core.save_chat(chat)
                    elif chat.get("pinn_source"):
                        chat["pinn_source"]["status"] = "needs_clarification"
                        core.save_chat(chat)
                if job.purpose == "data_review":
                    extra["kind"] = "data_review"
                if chat.get("run_dir") and analysis.state_comparison_question(job.question):
                    comparison = analysis.compare_run_states(chat["run_dir"])
                    if comparison.get("rows"):
                        extra["state_comparison"] = comparison
                saved = core.add_message(chat, "assistant", job.answer["answer"], **extra)
                if job.purpose in ("excel_maker", "pinn_maker") and saved.get("generated_artifact"):
                    item_id = _generated_file_id(saved["generated_artifact"]["path"])
                    if any(item["id"] == item_id for item in _file_items(chat)):
                        st.session_state["conversation_files"] = True
                        st.session_state["conversation_selected_file"] = item_id
            else:
                extra = {"kind": "data_review"} if job.purpose == "data_review" else {}
                if job.purpose == "pinn_maker":
                    extra["kind"] = "pinn_maker"
                    if isinstance(chat.get("pinn_source"), dict):
                        chat["pinn_source"]["status"] = "error"
                        core.save_chat(chat)
                if ((chat.get("run_dir") or chat.get("dataset")) and
                        job.purpose not in ("excel_maker", "pinn_maker")):
                    extra["visual_question"] = job.question
                if chat.get("run_dir") and analysis.state_comparison_question(job.question):
                    comparison = analysis.compare_run_states(chat["run_dir"])
                    if comparison.get("rows"):
                        extra["state_comparison"] = comparison
                core.add_message(chat, "assistant", job.error or "The AI request did not finish.",
                                 **extra,
                                 **({"error_detail": job.error_detail} if job.error_detail else {}))
            if (job.purpose == "data_review" and
                    (chat.get("setup") or {}).get("stage") not in (None, "complete")):
                core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question",
                                 stage=chat["setup"]["stage"])
            registry.conversations.pop(identity, None)
            if (job.purpose == "pinn_maker" and job.answer and job.answer.get("status") == "ready" and
                    not chat.get("run_dir") and chat.get("setup") is None):
                prompt = ("Analyze this uploaded dataset in depth. Describe the measured signals, sampling and segmentation, "
                          "input variation, derivative availability, and what still needs client confirmation before training.")
                _start_data_review(chat, prompt)
    if job := registry.planning.get(identity):
        if not job.running:
            if getattr(job, "purpose", None) == "run_setup_recommendation":
                _save_setup_recommendation(chat, recommendation=job.answer,
                                           error=job.error, error_detail=job.error_detail)
            elif job.answer:
                chat["settings"] = job.answer["settings"]
                core.add_message(chat, "assistant", job.answer["message"],
                                 **({"kind":"plan", "settings":chat["settings"].copy()} if job.answer["show_plan"] else {}))
            else:
                core.add_message(chat, "assistant", job.error or "The planning request did not finish. Please retry.",
                                 **({"error_detail": job.error_detail} if job.error_detail else {}))
            registry.planning.pop(identity, None)
    if job := registry.diagnostics.get(identity):
        if not job.running:
            if job.answer:
                core.add_message(chat, "assistant", job.answer["summary"], diagnosis=job.answer)
            else:
                detail = job.error or "The diagnosis could not finish. Please retry your question."
                if "unsupported_country_region_territory" in detail:
                    core.add_message(chat, "assistant", "The configured AI service denied this diagnostic request from the current region. The run's saved results and activity are still available here.", error_detail=detail)
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
    if setup.get("stage") == "complete":
        flow = chat.get("run_setup_flow") or {}
        approved = (flow.get("dataset_sha256") == dataset.get("sha256") and
                    flow.get("stage") == "approved")
        if not approved:
            if flow.get("dataset_sha256") != dataset.get("sha256"):
                _start_run_setup_agent(chat)
                core.add_message(chat, "assistant", "Before starting, review the model recommendation and search effort in the setup card. You can change either choice there.")
            else:
                core.add_message(chat, "assistant", "Review and approve the model and search effort in the setup card before starting.")
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
        core.add_message(chat, "assistant", "I'm starting the identification run. I'll inspect the data, fit candidate models, check held-out predictions, and save the results here." + notice_text)
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


def _submit(chat, question, files, hooks):
    question, attached, pinn_upload = (question or "").strip(), None, None
    command = question.lower().strip(" .!?")
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
            if Path(uploaded.name).suffix.lower() in (".m", ".txt"):
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
    pinn_requested = (bool(pinn_upload) or auto_prepare_for_dataset or
                      (not pinn_usage_question and not column_roles_need_confirmation and pinn_maker.is_raw_equation_request(
                          question, has_pending_source=pending_pinn_source.get("status", False))))
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
    elif question and not pinn_requested and not files:
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
    default_message = (f"I've attached the PINN source {pinn_upload['name']}." if pinn_upload
                       else "I've attached my dataset.")
    core.add_message(chat, "user", question or default_message, **({"attachment":message_attachment} if message_attachment else {}))
    dataset = chat.get("dataset") or {}
    if dataset.get("source_kind") == "pasted_table" and not excel_maker.is_excel_request(question):
        _add_pasted_table_preview(chat, dataset)
    if command in ("stop", "stop run", "cancel run", "stop training"):
        task = core.REGISTRY.training.get(chat["id"])
        if task and task["runner"].running:
            request_stop()
            core.add_message(chat, "assistant", "Stop requested. The current step will finish and available results will be saved here.")
        else:
            core.add_message(chat, "assistant", "There is no active training run in this conversation.")
        return
    if excel_maker.is_excel_request(question):
        job = excel_maker.ExcelMakerJob(chat, question)
        core.REGISTRY.conversations[chat["id"]] = job
        job.start()
        return
    if not files and pinn_usage_question:
        core.add_message(chat, "assistant", agent.pinn_usage_answer(chat), kind="pinn_status")
        return
    if attached and attached.get("column_review_pending"):
        core.add_message(chat, "assistant", _column_review_prompt(attached), kind="column_review")
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
            core.add_message(chat, "assistant", "I saved your PINN source. Please confirm the time, state, and input columns in the choices above; I’ll prepare the equation automatically after that.")
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
                core.add_message(chat, "assistant", "I saved your PINN source. First fix the dataset checks shown above; I’ll prepare the equation automatically once the measurements and column roles are ready.")
            else:
                core.add_message(chat, "assistant", "I saved your PINN source. Attach the measurements and confirm the time, state, and input columns; I’ll prepare the equation automatically as soon as they’re ready.")
            return
        if not pinn_maker.is_equation_ready(chat):
            prompt = (f"Prepare {chat['pinn_source']['name']} for PINN using the confirmed state and input columns."
                      if auto_prepare_for_dataset else
                      question or f"Prepare {chat['pinn_source']['name']} for PINN using the confirmed state and input columns.")
            if _start_pinn_preparation(chat, prompt):
                return
        core.add_message(chat, "assistant", "Your saved PINN equation is already validated for these columns." if pinn_maker.is_equation_ready(chat)
                         else "I couldn’t start equation preparation while another response is being processed. Please retry in a moment.")
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
        prompt = question or ("Analyze this uploaded dataset in depth. Identify measured patterns, sampling or "
                              "segmentation issues, input variation, derivative availability, and which conclusions "
                              "cannot be made before model training. Then help me set up the run.")
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
        model_concept = bool(re.search(r"\b(lstm|mlp|pinn)\b", question, re.I) and
                             re.search(r"\b(explain|difference|compare|what is|how does|which)\b", question, re.I))
        if model_concept or (chat.get("dataset") and not re.search(r"\b(use|change|set|increase|decrease|switch|configure|plan|recommend)\b", question, re.I)):
            job = agent.ConversationJob(chat, question)
            core.REGISTRY.conversations[chat["id"]] = job
        else:
            job = core.PlanningJob(chat, question or "Help me prepare this dataset for system identification.")
            core.REGISTRY.planning[chat["id"]] = job
    job.start()


def _working(chat):
    identity = chat["id"]
    if task := core.REGISTRY.training.get(identity):
        if task["runner"].running:
            state = task["state"]
            with st.chat_message("assistant", avatar=":material/graphic_eq:"):
                with st.status(state["stage"], expanded=True):
                    st.caption(f"Working for {workspace.duration(time.time()-task['started'])}")
                    for item in state["activity"][-4:]:
                        st.caption(("✓ " if item.get("state") == "complete" else "· ") + item["label"])
                    if state["history"]:
                        last = state["history"][-1]
                        st.caption(f"Cycle {last['cycle']} · validation MSE {last['val_mse']:.5g}")
                with st.expander("Training details"):
                    st.code(state["log"][-12000:], language="text", height=250)
            return
    job = (core.REGISTRY.conversations.get(identity) or core.REGISTRY.planning.get(identity)
           or core.REGISTRY.diagnostics.get(identity))
    if job:
        with st.chat_message("assistant", avatar=":material/graphic_eq:"):
            with st.status(getattr(job, "phase", "Reviewing your request"), expanded=False):
                st.caption("Using your configured model and the context of this conversation.")


@st.fragment(run_every=1.0)
def _live_thread(chat, hooks):
    _sync(chat, hooks)
    if not _busy(chat):
        st.rerun(scope="app")
    _messages(chat, True)
    _working(chat)


def _welcome():
    st.markdown('''<div class="conversation-welcome"><div class="welcome-glyph" aria-hidden="true">∿</div>
      <h1>What are we identifying?</h1>
      <p>Ask anything about system identification. When you’re ready, paste a table or attach a CSV/Excel file.<br>
      A time column and one measured state are a good start; include inputs if you recorded them. We can sort out the headers together.</p>
      </div>''', unsafe_allow_html=True)
    with st.container(horizontal=True, horizontal_alignment="center", key="conversation_suggestions"):
        for label, question, icon in [
            ("Identify a system", "Help me identify a system from my measurements. What data should I attach?", "add_chart"),
            ("Understand my data", "What should I check in my time-series data before training a dynamics model?", "table_chart"),
            ("LSTM, MLP, or PINN?", "Explain the difference between LSTM and MLP for system identification, and how PINN relates to both.", "account_tree"),
            ("Plan an experiment",
             "Help me plan a system identification experiment. Ask only for what we need, and let me skip details I don't know.",
             "science")]:
            if st.button(label, icon=f":material/{icon}:", key=f"suggest_{icon}"):
                st.session_state["conversation_pending"] = question
                st.rerun()


def render_app(*, runner, drain):
    hooks = dict(runner=runner, drain=drain)
    if "conversation" not in st.session_state:
        existing = core.read_chat(st.query_params.get("chat", ""))
        st.session_state["conversation"] = existing or core.new_chat()
    chat = st.session_state["conversation"]
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
    if show_files:
        conversation, files_panel = st.columns([1.55, 1], gap="large")
    else:
        conversation, files_panel = st.container(key="conversation_center"), None
    with conversation:
        if not chat["messages"]:
            _welcome()
        elif busy:
            _live_thread(chat, hooks)
        else:
            _messages(chat, False, file_items=file_items)
    if files_panel is not None:
        with files_panel:
            _artifacts(chat, file_items)
    with st.bottom:
        with st.container(key="conversation_composer"):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                if dataset := chat.get("dataset"):
                    st.caption(f":material/attach_file: {workspace.literal(dataset['name'])}")
                if training_active:
                    if st.button("Stop run", icon=":material/stop:", key="conversation_stop"):
                        request_stop()
                        core.add_message(chat, "assistant", "Stop requested. I'll save the available results after the current step.")
                        st.rerun()
                else:
                    st.caption("Ask about this run" if chat.get("run_dir") else f"{chat['settings']['architecture']} · {chat['settings']['max_cycles']} cycles")
            value = st.chat_input("Ask LabCD, attach a dataset, or add a PINN equation / MATLAB .m file…" if not chat.get("run_dir") else "Ask about the results, agents, or your next experiment…",
                                  accept_file=not bool(chat.get("run_dir")), file_type=["csv", "xlsx", "xls", "m", "txt"], max_upload_size=25,
                                  max_chars=6000, disabled=busy and not training_active, key=f"composer_{chat['id']}")
            st.caption("LabCD can make mistakes. Check the evidence behind important conclusions.")
    pending = st.session_state.pop("conversation_pending", None)
    if value or pending:
        text = pending or (value if isinstance(value, str) else value.text)
        uploads = [] if pending or isinstance(value, str) else value.files
        _submit(chat, text, uploads, hooks)
        st.query_params["chat"] = chat["id"]
        st.rerun()
