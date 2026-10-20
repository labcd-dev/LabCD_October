import queue
import time
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from frontend_streamlit.ga_agent_ui.ga_agent_utils import drain_event_queue, empty_plot_data
from frontend_streamlit.mulo_designer_utils import show_performance_plots, generate_controller_name, clear_cached_plots

# Import the live decision queue from the backend
try:
    from backend_core.MuloDesigner.Agents.mulo_design_agents import LIVE_AGENT_DECISIONS
except ImportError:
    LIVE_AGENT_DECISIONS = []

# -- Colour palette ------------------------------------------------------------
_BLUE = "#2196F3"
_RED = "#F44336"
_GREEN = "#4CAF50"
_ORANGE = "#FF9800"
_PURPLE = "#9C27B0"
_TEAL = "#00BCD4"
_GRAY = "rgba(120,120,120,0.55)"
_RANGE_FILL = "rgba(33,150,243,0.13)"
_TEMPLATE = "plotly_white"

_AGENT_LABELS = {
    "supervisor": "🧑‍💼 Supervisor Agent",
    "fine_tuner_agent": "👌 Fine Tuner Agent",
    "evaluate_agent": "⚖️ Evaluate Agent",
}


# -- Public entry-point --------------------------------------------------------

def display_project_page(run_by_ga_agent_ui: bool = True) -> None:
    """Display the project page with design results."""
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator is None:
        st.error("orchestrator not initialized. Please return to home and start a new experiment.")
        return

    supervisor_agent = st.session_state.get("supervisor_agent")

    # Grab loop index to ensure unique chart keys on sequential runs
    loop_idx = orchestrator.get_loop_index()

    # -- Header ----------------------------------------------------------------
    col_title, col_btn = st.columns([6, 1])
    with col_title:
        rc = st.session_state.get("run_config", {})
        case_name = rc.get("case_study_file", "Unknown").replace(".json", "")
        if not run_by_ga_agent_ui:
            case_name = generate_controller_name()
        st.markdown(f"## 🎛️ {case_name} — Controller Tuning")
    with col_btn:
        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        st.markdown('<div id="blue_btn"></div>', unsafe_allow_html=True)
        if st.button("🏠 New Experiment", width='stretch'):
            if supervisor_agent is not None and hasattr(supervisor_agent, "force_stop"):
                supervisor_agent.force_stop()
            st.session_state["mulo_designer_stage"] = "setup"
            _reset_and_go_home()
            return

    # -- Configuration badges --------------------------------------------------
    if "Agentic" in orchestrator.get_run_config()["optimizer_choice"]:
        _render_config_badges()
        st.markdown("---")

    # -- Drain the queue and refresh plot_data ---------------------------------
    drain_event_queue()

    # -- Get plot data (now a dict keyed by controller name) -------------------
    plot_data: Dict[str, Any] = st.session_state.get("plot_data", {})
    run_complete: bool = st.session_state.get("run_complete", False)

    # -- Live status bar -------------------------------------------------------
    _render_status_bar(plot_data, supervisor_agent)

    # -- Live Workflow Pause, Rollback & User Context Injection Panel ----------
    is_editing_context = False
    if not run_by_ga_agent_ui and supervisor_agent is not None:
        is_editing_context = _render_workflow_control_panel(supervisor_agent, run_complete)

    st.markdown("")

    # -- Main plots ------------------------------------------------------------
    if not plot_data:
        st.info("⏳ Waiting for the first generation result…")
    else:
        controller_names = list(plot_data.keys())
        is_mimo = len(controller_names) > 1

        if run_by_ga_agent_ui:
            tab_cost, tab_metrics, tab_gains, tab_summary, tab_decisions, tab_context = st.tabs([
                "Baseline Cost",
                "Performance Metrics",
                "PID Gains",
                "LLM Summary",
                "Agent Decisions",
                "Agent Context",
            ])
            tabs = {
                "cost": tab_cost, "metrics": tab_metrics, "gains": tab_gains,
                "summary": tab_summary, "decisions": tab_decisions, "context": tab_context,
            }
        else:
            tab_performance, tab_cost, tab_metrics, tab_gains, tab_summary, tab_decisions, tab_context, tab_result = st.tabs([
                "Controller Lab",
                "Baseline Cost",
                "Performance Metrics",
                "PID Gains",
                "LLM Summary",
                "Agent Decisions",
                "Agent Context",
                "Final Result",
            ])
            tabs = {
                "cost": tab_cost, "metrics": tab_metrics, "gains": tab_gains,
                "summary": tab_summary, "decisions": tab_decisions, "context": tab_context,
            }

            with tab_performance:
                if orchestrator.get_controller_designed():
                    show_performance_plots()
                else:
                    st.info("Controller Lab simulation will be available once the block finishes designing...")

            with tab_result:
                idx = orchestrator.get_loop_index()
                is_complete = idx >= len(orchestrator.get_controller_structure())
                text = f" Number {idx}" if not is_complete else ""

                if orchestrator.get_controller_designed():
                    with st.expander(
                            f"🐍 View Raw Python Code Output ({'Final ' if is_complete else ''}Loop{text} Designed)"):
                        st.code(st.session_state.get("modified_code", ""))
                    with st.expander(f"📄 View Raw JSON Output ({'Final ' if is_complete else ''}Loop{text} Designed)"):
                        st.json(st.session_state.get("modified_controller_structure", {}))
                else:
                    st.info("Raw outputs will be available once the block finishes designing...")

        with tabs["cost"]:
            for controller_name, data in plot_data.items():
                if data.get("cumulative_nfe"):
                    st.plotly_chart(
                        _build_cost_fig(data, title_suffix=f" - {controller_name}"),
                        width='stretch',
                        key=f"cost_{controller_name}_{loop_idx}"
                    )
                    st.divider()
                else:
                    st.info(f"⏳ Waiting for data from {controller_name}...")

        with tabs["metrics"]:
            for controller_name, data in plot_data.items():
                if data.get("cumulative_nfe"):
                    st.plotly_chart(
                        _build_metrics_fig(data, title_suffix=f" - {controller_name}"),
                        width='stretch',
                        key=f"metrics_{controller_name}_{loop_idx}"
                    )
                    st.divider()
                else:
                    st.info(f"⏳ Waiting for data from {controller_name}...")

        with tabs["gains"]:
            for controller_name, data in plot_data.items():
                if data.get("cumulative_nfe"):
                    st.plotly_chart(
                        _build_gains_fig(data, title_suffix=f" - {controller_name}"),
                        width='stretch',
                        key=f"gains_{controller_name}_{loop_idx}"
                    )
                    st.divider()
                else:
                    st.info(f"⏳ Waiting for data from {controller_name}...")

        with tabs["summary"]:
            for controller_name, data in plot_data.items():
                if data.get("attempt_summaries"):
                    st.plotly_chart(
                        _build_summary_fig(data, title_suffix=f" - {controller_name}"),
                        width='stretch',
                        key=f"summary_{controller_name}_{loop_idx}"
                    )
                else:
                    st.info(f"⏳ Waiting for the first attempt to complete for {controller_name}...")
                if len(plot_data) > 1:
                    st.divider()

        with tabs["decisions"]:
            if not LIVE_AGENT_DECISIONS:
                st.info("⏳ Waiting for the first agent decision...")
            else:
                for i, decision in enumerate(reversed(LIVE_AGENT_DECISIONS)):
                    agent_name = decision.get("agent", "Unknown Agent")
                    decision_number = len(LIVE_AGENT_DECISIONS) - i

                    with st.expander(f"{agent_name} - Decision #{decision_number}", expanded=(i == 0)):
                        st.json(decision.get("response", {}))

        with tabs["context"]:
            if supervisor_agent is not None and hasattr(supervisor_agent, "get_workflow_status"):
                wf_info = supervisor_agent.get_workflow_status()
                uc_hist = wf_info.get("user_context_history", [])
                if uc_hist:
                    with st.expander(f"🗣️ User Context History ({len(uc_hist)} entries)", expanded=True):
                        st.json({
                            "latest_user_context": wf_info.get("user_context"),
                            "user_context_history": uc_hist,
                        })

            if not LIVE_AGENT_DECISIONS:
                st.info("⏳ Waiting for the first agent context...")
            else:
                for i, decision in enumerate(reversed(LIVE_AGENT_DECISIONS)):
                    agent_name = decision.get("agent", "Unknown Agent")
                    decision_number = len(LIVE_AGENT_DECISIONS) - i

                    context_data = decision.get("context", {})
                    if context_data != {}:
                        with st.expander(f"{agent_name} - Context #{decision_number}", expanded=(i == 0)):
                            if isinstance(context_data, str):
                                st.text(context_data)
                            else:
                                st.json(context_data)

    # -- Auto-rerun while GA is running (paused while user is typing prompt) ---
    if not run_complete:
        thread = st.session_state.get("run_thread")
        if thread and thread.is_alive():
            if not is_editing_context:
                time.sleep(0.75)
                st.rerun()
        else:
            drain_event_queue()
            st.session_state.run_complete = True
            st.rerun()
    else:
        err = st.session_state.get("run_error")
        if err:
            st.error(f"❌ Run failed: {err}")
            with st.expander("📋 Traceback"):
                st.code(st.session_state.get("run_error_tb", ""), language="python")
        else:
            st.session_state["mulo_designer_stage"] = "optimisation_complete"


# -- Live Workflow Control & Context Injection Panel ---------------------------

def _render_workflow_control_panel(supervisor_agent: Any, run_complete: bool) -> bool:
    """
    Renders live controls for pausing the LangGraph workflow and injecting UserContext:
      - Mode 1 (Graceful): Keep current step running, pause at step completion (or queue prompt),
        inject UserContext, and proceed to the next step.
      - Mode 2 (Immediate Terminate & Rollback): Immediately terminate the ongoing process,
        roll back to the latest completed graph-level agentic step (`supervisor`, `fine_tuner_agent`,
        or `evaluate_agent`), inject UserContext, and re-run that agentic step and all steps after it.
        Disabled during the initial `optimize` run before the first supervisor decision is made.

    Returns True if the user currently has the context editor open (so the 0.75s polling loop
    pauses and does not interrupt typing).
    """
    if not hasattr(supervisor_agent, "get_workflow_status"):
        return False

    wf_info = supervisor_agent.get_workflow_status()
    wf_status = wf_info["status"]
    current_node = wf_info["current_node"]
    can_rollback = wf_info["can_rollback"]
    latest_agent = wf_info["latest_agent"]
    pending_prompt = wf_info["pending_prompt"]
    uc_history = wf_info["user_context_history"]

    active_mode = st.session_state.get("context_injection_mode")
    if wf_status in ("pausing_after_step", "paused") and active_mode is None and pending_prompt is None:
        active_mode = "graceful"
        st.session_state["context_injection_mode"] = "graceful"
    elif wf_status in ("terminating", "paused_after_rollback") and active_mode is None:
        active_mode = "rollback"
        st.session_state["context_injection_mode"] = "rollback"

    latest_agent_str = (
        f"{_AGENT_LABELS.get(latest_agent['node_name'], latest_agent['node_name'])} "
        f"(Loop {latest_agent['loop_index'] + 1}, Ctrl {latest_agent['controller_index'] + 1}, Attempt {latest_agent['attempt']})"
        if latest_agent else "None yet (Initial Optimization Run)"
    )

    st.markdown("---")
    info_col1, info_col2, info_col3 = st.columns([2, 3, 2])
    with info_col1:
        st.markdown(f"**📍 Active Graph Step:** `{current_node}`")
    with info_col2:
        st.markdown(f"**🧠 Latest Completed Agent:** {latest_agent_str}")
    with info_col3:
        if pending_prompt:
            st.markdown("**📥 Context Queued:** Waiting for step to finish")
        else:
            st.markdown(f"**💬 Injected Contexts:** `{len(uc_history)}`")

    # 1. While workflow is running
    if not run_complete:
        if active_mode is None:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(
                    "⏸️ Pause After Current Step & Inject Context",
                    type="secondary",
                    width='stretch',
                    key="btn_graceful_pause",
                    help="Allows the currently running process to finish its job normally, then injects your context before the next step.",
                ):
                    supervisor_agent.request_pause_after_current_step(pause_and_wait=True)
                    st.session_state["context_injection_mode"] = "graceful"
                    st.rerun()

            with btn_col2:
                rollback_help = (
                    "Immediately terminates the ongoing process, rolls back to the latest completed agentic step, "
                    "injects your user context, and re-runs that agentic step and all processes after it."
                    if can_rollback
                    else "Disabled during the initial optimization run because no Supervisor Agent decision has been completed yet."
                )
                if st.button(
                    "🛑 Terminate Current Process & Rollback to Last Agent",
                    type="primary",
                    width='stretch',
                    disabled=not can_rollback,
                    key="btn_terminate_rollback",
                    help=rollback_help,
                ):
                    supervisor_agent.request_terminate_and_rollback(wait_for_prompt=True)
                    st.session_state["context_injection_mode"] = "rollback"
                    drain_event_queue()
                    st.rerun()

            if not can_rollback:
                st.caption(
                    "ℹ️ *Terminate & Rollback becomes available once the first Supervisor Agent decision completes. "
                    "During the initial optimization run, use **Pause After Current Step** to inject context after optimization finishes.*"
                )
            return False

        # Active Context Editor Panel (Mode 1 or Mode 2)
        with st.container(border=True):
            if active_mode == "graceful":
                if wf_status == "paused":
                    st.info(
                        f"⏸️ **Workflow Paused After Step (`{current_node}`):** "
                        f"The ongoing step finished normally. Write your prompt below to inject user context before the next step starts."
                    )
                else:
                    st.warning(
                        f"⏳ **Graceful Pause Active (Current Step `{current_node}` Finishing Normally):** "
                        f"The current process is **not** interrupted and continues in the background. "
                        f"Write your prompt below — as soon as `{current_node}` finishes, your context will be injected into the workflow."
                    )
            else:
                target_label = _AGENT_LABELS.get(latest_agent["node_name"], latest_agent["node_name"]) if latest_agent else "Latest Agent"
                st.error(
                    f"🛑 **Process Terminated & Rolled Back to {target_label}:** "
                    f"The ongoing process was terminated and state was restored to right before **{target_label}** "
                    f"(`{latest_agent['checkpoint'] if latest_agent else ''}`). "
                    f"Write your prompt below to inject user context and re-run **{target_label}** along with all subsequent steps."
                )

            prompt_text = st.text_area(
                "User Context Prompt",
                placeholder="Type your guidance or constraints for the workflow agents (e.g., 'Relax settling time target to 10s and focus on reducing overshoot')...",
                key="live_user_context_input",
                height=110,
            )

            if active_mode == "graceful" and can_rollback:
                act_col1, act_col2, act_col3, act_col4 = st.columns([3, 2, 2, 1])
            else:
                act_col1, act_col2, act_col4 = st.columns([4, 3, 1])
                act_col3 = None

            with act_col1:
                submit_label = (
                    "▶️ Inject Context & Re-run Agent Step"
                    if active_mode == "rollback"
                    else "▶️ Inject Context & Proceed"
                )
                if st.button(submit_label, type="primary", width='stretch', key="btn_submit_user_context"):
                    if not prompt_text or not prompt_text.strip():
                        st.warning("⚠️ Please enter a non-empty prompt to inject, or click 'Resume Without Context'.")
                    else:
                        supervisor_agent.resume_with_context(prompt_text.strip())
                        st.session_state.pop("context_injection_mode", None)
                        st.session_state.pop("live_user_context_input", None)
                        st.rerun()

            with act_col2:
                if st.button("▶️ Resume Without Context", type="secondary", width='stretch', key="btn_resume_no_context"):
                    supervisor_agent.resume_with_context(None)
                    st.session_state.pop("context_injection_mode", None)
                    st.session_state.pop("live_user_context_input", None)
                    st.rerun()

            if act_col3 is not None:
                with act_col3:
                    if st.button(
                        "🛑 Terminate & Rollback Instead",
                        type="secondary",
                        width='stretch',
                        key="btn_switch_to_rollback",
                        help="Immediately abort the current step and roll back to the latest completed agentic step instead of waiting.",
                    ):
                        supervisor_agent.request_terminate_and_rollback(wait_for_prompt=True)
                        st.session_state["context_injection_mode"] = "rollback"
                        drain_event_queue()
                        st.rerun()

            with act_col4:
                if st.button("🔄 Refresh", type="secondary", width='stretch', key="btn_refresh_pause_panel"):
                    drain_event_queue()
                    st.rerun()

        return True

    # 2. When the current loop has completed (`run_complete` is True)
    with st.expander("💬 Inject User Context or Re-run Latest Completed Agentic Step", expanded=False):
        post_prompt = st.text_area(
            "User Context Prompt",
            placeholder="Add context for the next loop or to re-run the latest completed agentic step...",
            key="post_run_user_context_input",
            height=90,
        )
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button(
                "📥 Add Context for Next Stage",
                type="secondary",
                width='stretch',
                key="btn_add_post_context",
            ):
                if post_prompt and post_prompt.strip():
                    supervisor_agent.inject_context_while_idle(post_prompt.strip())
                    st.success("✅ User context added to workflow state and history!")
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a non-empty context prompt.")

        with p_col2:
            if st.button(
                "🛑 Inject Context & Re-run Latest Completed Agent",
                type="primary",
                width='stretch',
                disabled=not can_rollback,
                key="btn_post_rollback_rerun",
                help="Rolls back to the latest completed agentic step in this loop, injects your prompt, and re-runs from that step.",
            ):
                if not post_prompt or not post_prompt.strip():
                    st.warning("⚠️ Please enter a prompt before re-running the latest agentic step.")
                else:
                    supervisor_agent._pending_user_prompt = post_prompt.strip()
                    supervisor_agent._start_next_run_from_rollback = True
                    st.session_state["run_complete"] = False
                    st.session_state.pop("run_error", None)
                    st.session_state.pop("run_error_tb", None)
                    st.session_state.pop("modified_code", None)
                    st.session_state.pop("modified_controller_structure", None)
                    clear_cached_plots()
                    st.session_state["orchestrator"].set_controller_designed(False)
                    st.session_state["mulo_designer_stage"] = "run_designer"
                    st.rerun()

    return False


# -- Internal helpers ----------------------------------------------------------

def _reset_and_go_home() -> None:
    try:
        from backend_core.MuloDesigner.Agents.mulo_design_agents import LIVE_AGENT_DECISIONS
        LIVE_AGENT_DECISIONS.clear()
    except ImportError:
        pass

    for key in (
            "run_config", "event_queue", "plot_data", "plot_data_snapshots", "run_thread",
            "run_complete", "run_error", "run_error_tb", "context_injection_mode",
    ):
        st.session_state.pop(key, None)
    st.session_state.plot_data = empty_plot_data()
    st.session_state.page = "home"
    st.rerun()


def _render_config_badges() -> None:
    rc = st.session_state.get("run_config", {})
    badges = [
        ("🤖 Model", rc.get("llm_model", "—")),
        ("🔁 Max Attempts", str(rc.get("max_attempts", "—"))),
        ("⏱ Wall Clock", f"{rc.get('max_wall_clock', 0):.0f} s"),
        ("💰 Cost Budget", f"${rc.get('max_cost_budget', 0):.3f}"),
        ("📝 Variant", rc.get("prompt_variant", "—")),
    ]
    for col, (label, value) in zip(st.columns(len(badges)), badges):
        col.metric(label, value)


def _render_status_bar(plot_data: Dict[str, Any], supervisor_agent: Any = None) -> None:
    run_complete: bool = st.session_state.get("run_complete", False)
    err = st.session_state.get("run_error")

    wf_status = "running"
    if supervisor_agent is not None and hasattr(supervisor_agent, "get_workflow_status"):
        wf_status = supervisor_agent.get_workflow_status().get("status", "running")

    if not run_complete:
        if wf_status in ("paused", "paused_after_rollback"):
            status_html = (
                '<div style="background:#9C27B0;color:white;border-radius:6px;'
                'padding:6px 14px;text-align:center;font-weight:600;">⏸️  Paused</div>'
            )
        elif wf_status == "pausing_after_step":
            status_html = (
                '<div style="background:#00BCD4;color:white;border-radius:6px;'
                'padding:6px 14px;text-align:center;font-weight:600;">⏳  Finishing Step…</div>'
            )
        else:
            status_html = (
                '<div style="background:#FF9800;color:white;border-radius:6px;'
                'padding:6px 14px;text-align:center;font-weight:600;">🔄  Running…</div>'
            )
    elif err:
        status_html = (
            '<div style="background:#F44336;color:white;border-radius:6px;'
            'padding:6px 14px;text-align:center;font-weight:600;">❌  Failed</div>'
        )
    else:
        status_html = (
            '<div style="background:#4CAF50;color:white;border-radius:6px;'
            'padding:6px 14px;text-align:center;font-weight:600;">✅  Complete</div>'
        )

    total_nfe = 0
    best_cost = float('inf')
    max_attempt = 0
    max_score = 0

    for data in plot_data.values():
        if data.get("cumulative_nfe"):
            total_nfe = max(total_nfe, data["cumulative_nfe"][-1] if data["cumulative_nfe"] else 0)
        if data.get("best_baseline_so_far"):
            best_cost = min(best_cost,
                            data["best_baseline_so_far"][-1] if data["best_baseline_so_far"] else float('inf'))
        if data.get("attempt"):
            max_attempt = max(max_attempt, data["attempt"][-1] if data["attempt"] else 0)
        if data.get("success_score"):
            max_score = max(max_score, data["success_score"][-1] if data["success_score"] else 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(status_html, unsafe_allow_html=True)
    c2.metric("Attempt", max_attempt if max_attempt > 0 else "—")
    c3.metric("Cumulative NFE", f"{total_nfe:,}" if total_nfe > 0 else "—")
    c4.metric("Best Baseline Cost", f"{best_cost:.4f}" if best_cost != float('inf') else "—")
    c5.metric("Success Score", f"{max_score}/100" if max_score > 0 else "—")

    if len(plot_data) > 1:
        st.caption(f"🔄 Tuning {len(plot_data)} controllers simultaneously")


# -- Shared plot utilities -----------------------------------------------------

def _attempt_nfe_extents(pd: Dict) -> Dict[int, Dict[str, int]]:
    extents: Dict[int, Dict[str, int]] = {}
    for nfe, att in zip(pd["cumulative_nfe"], pd["attempt"]):
        if att not in extents:
            extents[att] = {"nfe_min": nfe, "nfe_max": nfe}
        else:
            extents[att]["nfe_max"] = max(extents[att]["nfe_max"], nfe)
    return extents


def _add_attempt_vlines(
        fig: go.Figure,
        pd: Dict,
        row: Optional[int] = None,
        col: Optional[int] = None,
        opacity: float = 0.7,
) -> None:
    base_kwargs: Dict[str, Any] = {}
    if row is not None:
        base_kwargs["row"] = row
    if col is not None:
        base_kwargs["col"] = col

    # 1. Internal GA Agent attempt boundaries (gray dashed line)
    for nfe in pd.get("attempt_boundaries_nfe", []):
        fig.add_vline(
            x=nfe,
            line_dash="dash",
            line_color=_GRAY,
            opacity=opacity,
            **base_kwargs,
        )

    # 2. Supervisor / Fine Tuner / Evaluate Agent boundaries (red dotted line)
    for nfe in pd.get("supervisor_boundaries_nfe", []):
        fig.add_vline(
            x=nfe,
            line_dash="dot",
            line_color=_PURPLE,
            line_width=2,
            opacity=0.9,
            **base_kwargs,
        )


def _build_cost_fig(pd: Dict, title_suffix: str = "") -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd["cumulative_nfe"],
        y=pd["best_baseline_cost"],
        mode="lines+markers",
        name="Generation Best Baseline Cost",
        line=dict(color="rgba(33, 150, 243, 0.45)", width=2.5),
        marker=dict(size=4),
    ))

    _add_attempt_vlines(fig, pd, opacity=0.7)

    fig.update_layout(
        title=f"Baseline Cost vs Cumulative NFE{title_suffix}",
        xaxis_title="Cumulative NFE",
        yaxis_title="Baseline Cost",
        template=_TEMPLATE,
        height=320,
        margin=dict(l=64, r=40, t=52, b=48),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    return fig


def _build_metrics_fig(pd: Dict, title_suffix: str = "") -> go.Figure:
    metrics = ["mse", "settling_time", "overshoot", "control_effort"]
    target_keys = ["target_mse", "target_settling_time", "target_overshoot", "target_control_effort"]
    labels = ["MSE", "Settling Time", "Overshoot", "Control Effort"]
    colors = [_BLUE, _ORANGE, _PURPLE, _TEAL]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=labels,
        horizontal_spacing=0.10,
        vertical_spacing=0.20,
    )

    for metric, t_key, label, color, (r, c) in zip(metrics, target_keys, labels, colors, positions):
        # Actual performance metric trace
        fig.add_trace(
            go.Scatter(
                x=pd["cumulative_nfe"],
                y=pd[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=r, col=c,
        )

        # Historical target metric trace (plotted as a step graph)
        if pd.get(t_key) and not all(v != v for v in pd[t_key]):  # Ensure data isn't all NaNs
            fig.add_trace(
                go.Scatter(
                    x=pd["cumulative_nfe"],
                    y=pd[t_key],
                    mode="lines",
                    name=f"Target {label}",
                    line=dict(color=_RED, width=2, dash="dash"),
                    line_shape="vh",  # <--- Change this from "hv" to "vh"
                    showlegend=False,
                ),
                row=r, col=c,
            )

            # Fix: Use data coordinates and native row/col binding
            latest_target = pd[t_key][-1]
            if latest_target == latest_target:  # check for not NaN
                fig.add_annotation(
                    x=pd["cumulative_nfe"][-1],  # Anchor to the highest NFE point
                    y=latest_target,  # Anchor to the target line height
                    text=f"Target={latest_target}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(color=_GRAY),
                    row=r, col=c  # Natively target the correct subplot
                )

        _add_attempt_vlines(fig, pd, row=r, col=c, opacity=0.5)

    fig.update_layout(
        title=f"Performance Metrics vs Cumulative NFE{title_suffix}",
        template=_TEMPLATE,
        height=650,
        margin=dict(l=64, r=40, t=64, b=52),
    )
    for r, c in positions:
        fig.update_xaxes(title_text="NFE", row=r, col=c)
    return fig


def _build_gains_fig(pd: Dict, title_suffix: str = "") -> go.Figure:
    gains = ["Kp", "Ki", "Kd"]
    colors = [_BLUE, _ORANGE, _PURPLE]
    extents = _attempt_nfe_extents(pd)

    safe_ranges = {}
    for s in pd.get("attempt_summaries", []):
        safe_ranges[s["attempt"]] = s.get("param_ranges", {})

    if pd.get("attempt_ranges"):
        for att_num in extents.keys():
            if att_num not in safe_ranges:
                safe_ranges[att_num] = pd["attempt_ranges"].get(att_num, {})

    fig = make_subplots(rows=1, cols=3, subplot_titles=gains,
                        horizontal_spacing=0.08)

    for gi, (gain, color) in enumerate(zip(gains, colors), 1):
        for att_num, ext in extents.items():
            att_rng_dict = safe_ranges.get(att_num, {})
            rng = att_rng_dict.get(gain) or att_rng_dict.get(gain.lower())
            if rng:
                lo, hi = rng
                fig.add_shape(
                    type="rect",
                    x0=ext["nfe_min"], x1=ext["nfe_max"],
                    y0=lo, y1=hi,
                    fillcolor=_RANGE_FILL,
                    line_width=0,
                    row=1, col=gi,
                )
        fig.add_trace(
            go.Scatter(
                x=pd["cumulative_nfe"],
                y=pd[gain],
                mode="lines+markers",
                name=gain,
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=1, col=gi,
        )
        _add_attempt_vlines(fig, pd, row=1, col=gi, opacity=0.55)

    fig.update_layout(
        title=f"PID Gains vs Cumulative NFE{title_suffix} (shaded = search range)",
        template=_TEMPLATE,
        height=300,
        margin=dict(l=64, r=40, t=64, b=52),
    )
    for gi in range(1, 4):
        fig.update_xaxes(title_text="NFE", row=1, col=gi)
    return fig


def _build_summary_fig(pd: Dict, title_suffix: str = "") -> go.Figure:
    summaries: List[Dict] = pd.get("attempt_summaries", [])
    if not summaries:
        fig = go.Figure()
        fig.add_annotation(text="No attempt data yet", showarrow=False,
                           font=dict(size=14, color="gray"))
        return fig

    attempts = [s["attempt"] for s in summaries]

    subplot_titles = [
        "GA Config (Pop / Gens)",
        "Weights: MSE & Settling Time",
        "Weights: Overshoot & Ctrl Effort",
        "Budget Remaining (%)",
        "Success Score",
        "Search Range: Kp",
        "Search Range: Ki",
        "Search Range: Kd",
    ]
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.26,
    )

    pops = [s["pop_size"] for s in summaries]
    gens = [s["num_gen"] for s in summaries]
    fig.add_trace(go.Bar(name="Pop Size", x=attempts, y=pops,
                         marker_color=_BLUE, showlegend=True,
                         legendgroup="cfg"), row=1, col=1)
    fig.add_trace(go.Bar(name="Num Gens", x=attempts, y=gens,
                         marker_color=_ORANGE, showlegend=True,
                         legendgroup="cfg"), row=1, col=1)

    w_mse = [s["weights"].get("mse", 0) for s in summaries]
    w_st = [s["weights"].get("settling_time", 0) for s in summaries]
    fig.add_trace(go.Bar(name="W-MSE", x=attempts, y=w_mse,
                         marker_color=_BLUE, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="W-ST", x=attempts, y=w_st,
                         marker_color=_ORANGE, showlegend=False), row=1, col=2)

    w_os = [s["weights"].get("overshoot", 0) for s in summaries]
    w_ce = [s["weights"].get("control_effort", 0) for s in summaries]
    fig.add_trace(go.Bar(name="W-OS", x=attempts, y=w_os,
                         marker_color=_PURPLE, showlegend=False), row=1, col=3)
    fig.add_trace(go.Bar(name="W-CE", x=attempts, y=w_ce,
                         marker_color=_TEAL, showlegend=False), row=1, col=3)

    t_pct = [s.get("time_remaining_pct", 0) for s in summaries]
    c_pct = [s.get("cost_remaining_pct", 0) for s in summaries]
    fig.add_trace(go.Scatter(name="Time %", x=attempts, y=t_pct,
                             mode="lines+markers",
                             line=dict(color=_BLUE),
                             showlegend=True, legendgroup="budget"),
                  row=1, col=4)
    fig.add_trace(go.Scatter(name="Cost %", x=attempts, y=c_pct,
                             mode="lines+markers",
                             line=dict(color=_RED, dash="dot"),
                             showlegend=True, legendgroup="budget"),
                  row=1, col=4)
    fig.update_yaxes(range=[0, 105], row=1, col=4)

    scores = [s.get("success_score", 0) for s in summaries]
    bar_colors = [
        _GREEN if sc == 100 else (_ORANGE if sc >= 50 else _RED)
        for sc in scores
    ]
    fig.add_trace(go.Bar(name="Score", x=attempts, y=scores,
                         marker_color=bar_colors, showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color=_GREEN,
                  opacity=0.6, row=2, col=1)
    fig.update_yaxes(range=[0, 110], row=2, col=1)

    for gi, gain in enumerate(["Kp", "Ki", "Kd"], 2):
        for s in summaries:
            att = s["attempt"]
            p_ranges = s.get("param_ranges", {})
            rng = p_ranges.get(gain) or p_ranges.get(gain.lower())
            if rng is None:
                continue
            lo, hi = rng
            c_gains = s.get("controller_gains", {})
            final_val = c_gains.get(gain, c_gains.get(gain.lower(), (lo + hi) / 2))

            fig.add_trace(
                go.Scatter(
                    x=[att, att], y=[lo, hi],
                    mode="lines",
                    line=dict(color="rgba(33,150,243,0.40)", width=12),
                    showlegend=False,
                ),
                row=2, col=gi,
            )
            fig.add_trace(
                go.Scatter(
                    x=[att], y=[final_val],
                    mode="markers",
                    marker=dict(size=11, color=_RED, symbol="diamond"),
                    showlegend=False,
                ),
                row=2, col=gi,
            )

    fig.update_layout(
        title=f"LLM Decision Summary per Attempt{title_suffix}",
        barmode="group",
        template=_TEMPLATE,
        height=580,
        margin=dict(l=64, r=40, t=80, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right", x=1,
        ),
    )
    for r in range(1, 3):
        for c in range(1, 5):
            fig.update_xaxes(title_text="Attempt", dtick=1, row=r, col=c)

    return fig