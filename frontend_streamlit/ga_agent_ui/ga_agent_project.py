import time
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from frontend_streamlit.ga_agent_ui.ga_agent_utils import drain_event_queue, empty_plot_data
from frontend_streamlit.mulo_designer_utils import show_performance_plots, generate_controller_name

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


# -- Public entry-point --------------------------------------------------------

def display_project_page(run_by_ga_agent_ui: bool = True) -> None:
    """Display the project page with design results."""
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator is None:
        st.error("orchestrator not initialized. Please return to home and start a new experiment.")
        return

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

    # -- Live status bar -------------------------------------------------------
    _render_status_bar(plot_data)
    st.markdown("")

    # -- Main plots ------------------------------------------------------------
    if not plot_data:
        st.info("⏳ Waiting for the first generation result…")
    else:
        controller_names = list(plot_data.keys())
        is_mimo = len(controller_names) > 1

        if run_by_ga_agent_ui:
            tab_cost, tab_metrics, tab_gains, tab_summary, tab_decisions = st.tabs([
                "Baseline Cost",
                "Performance Metrics",
                "PID Gains",
                "LLM Summary",
                "Agent Decisions",
            ])
            tabs = {"cost": tab_cost, "metrics": tab_metrics, "gains": tab_gains, "summary": tab_summary,
                    "decisions": tab_decisions}
        else:
            tab_performance, tab_cost, tab_metrics, tab_gains, tab_summary, tab_decisions, tab_result = st.tabs([
                "Controller Lab",
                "Baseline Cost",
                "Performance Metrics",
                "PID Gains",
                "LLM Summary",
                "Agent Decisions",
                "Final Result",
            ])
            tabs = {"cost": tab_cost, "metrics": tab_metrics, "gains": tab_gains, "summary": tab_summary,
                    "decisions": tab_decisions}

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
                # Reverse the list so the most recent decisions appear at the top
                for i, decision in enumerate(reversed(LIVE_AGENT_DECISIONS)):
                    agent_name = decision.get("agent", "Unknown Agent")
                    decision_number = len(LIVE_AGENT_DECISIONS) - i

                    with st.expander(f"{agent_name} - Decision #{decision_number}", expanded=(i == 0)):
                        st.json(decision.get("response", {}))

    # -- Auto-rerun while GA is running ----------------------------------------
    run_complete: bool = st.session_state.get("run_complete", False)
    if not run_complete:
        thread = st.session_state.get("run_thread")
        if thread and thread.is_alive():
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


# -- Internal helpers ----------------------------------------------------------

def _reset_and_go_home() -> None:
    try:
        from backend_core.MuloDesigner.Agents.mulo_design_agents import LIVE_AGENT_DECISIONS
        LIVE_AGENT_DECISIONS.clear()
    except ImportError:
        pass

    for key in (
            "run_config", "event_queue", "plot_data", "run_thread",
            "run_complete", "final_state", "run_error", "run_error_tb",
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


def _render_status_bar(plot_data: Dict[str, Any]) -> None:
    run_complete: bool = st.session_state.get("run_complete", False)
    err = st.session_state.get("run_error")

    if not run_complete:
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
) -> None:
    kwargs: Dict[str, Any] = {"line_dash": "dash", "line_color": _GRAY, "opacity": 0.7}
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    for nfe in pd["attempt_boundaries_nfe"]:
        fig.add_vline(x=nfe, **kwargs)


def _build_cost_fig(pd: Dict, title_suffix: str = "") -> go.Figure:
    fig = go.Figure()

    # 1. Best baseline cost of the current generation (shows exploration spikes)
    # if "best_baseline_cost" in pd and pd["best_baseline_cost"]:
    fig.add_trace(go.Scatter(
        x=pd["cumulative_nfe"],
        y=pd["best_baseline_cost"],
        mode="lines+markers",
        name="Generation Best Baseline Cost",
        line=dict(color="rgba(33, 150, 243, 0.45)", width=2.5),
        marker=dict(size=4),
    ))

    # 2. Cumulative best baseline cost so far (all-time running champion)
    # if "best_baseline_so_far" in pd and pd["best_baseline_so_far"]:
    # fig.add_trace(go.Scatter(
    #     x=pd["cumulative_nfe"],
    #     y=pd["best_baseline_so_far"],
    #     mode="lines+markers",
    #     name="Best Baseline Cost So Far",
    #     line=dict(color=_BLUE, width=2.5),
    #     marker=dict(size=6),
    # ))

    for nfe in pd.get("attempt_boundaries_nfe", []):
        fig.add_vline(x=nfe, line_dash="dash", line_color=_GRAY, opacity=0.7)

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
    labels = ["MSE", "Settling Time", "Overshoot", "Control Effort"]
    colors = [_BLUE, _ORANGE, _PURPLE, _TEAL]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    targets = pd["fixed_targets"]
    # if isinstance(fixed_targets_data, list) and len(fixed_targets_data) > 0:
    #     targets: Dict[str, float] = fixed_targets_data[-1]
    # elif isinstance(fixed_targets_data, dict):
    #     targets: Dict[str, float] = fixed_targets_data
    # else:
    #     targets: Dict[str, float] = {}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=labels,
        horizontal_spacing=0.10,
        vertical_spacing=0.20,
    )
    for metric, label, color, (r, c) in zip(metrics, labels, colors, positions):
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
        target_val = targets.get(metric)


        if target_val is not None:
            fig.add_hline(
                y=target_val,
                line_dash="dash",
                line_color=_RED,
                opacity=0.75,
                annotation_text=f"Target={target_val}",
                annotation_position="top right",
                row=r, col=c,
            )
        for nfe in pd["attempt_boundaries_nfe"]:
            fig.add_vline(x=nfe, line_dash="dash", line_color=_GRAY,
                          opacity=0.5, row=r, col=c)

    fig.update_layout(
        title=f"Performance Metrics vs Cumulative NFE{title_suffix}",
        template=_TEMPLATE,
        height=480,
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
            rng = safe_ranges.get(att_num, {}).get(gain)
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
        for nfe in pd["attempt_boundaries_nfe"]:
            fig.add_vline(x=nfe, line_dash="dash", line_color=_GRAY,
                          opacity=0.55, row=1, col=gi)

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
            rng = s.get("param_ranges", {}).get(gain)
            if rng is None:
                continue
            lo, hi = rng
            final_val = s.get("controller_gains", {}).get(gain, (lo + hi) / 2)

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