import streamlit as st
import re
import plotly.graph_objects as go
import copy

from backend_core.MuloDesigner.utils import (
    active_controller_index,
    controller_loop_name,
    get_pid_gain_upper_bounds,
    get_pid_gain_lower_bounds,
    get_pid_gains
)

from backend_core.MuloDesigner.equation_editor import (
    replace_last_pid_controller_gains,
    apply_pid_gains_to_controller_structure
)
from backend_core.MuloDesigner.simulator import simulate_system_response


def generate_controller_name():
    orchestrator = st.session_state["orchestrator"]
    return controller_loop_name(orchestrator.get_controller_structure(), int(orchestrator.get_loop_index()))


def show_performance_plots():
    orchestrator = st.session_state["orchestrator"]
    supervisor = st.session_state["supervisor_agent"]
    loop_index = active_controller_index(orchestrator.get_loop_index())

    if "modified_code" not in st.session_state:
        st.session_state["modified_code"] = orchestrator.get_equation()
    if "modified_controller_structure" not in st.session_state:
        st.session_state["modified_controller_structure"] = copy.deepcopy(orchestrator.get_controller_structure())

    for j, cont in enumerate(st.session_state["modified_controller_structure"][loop_index]["controllers"]):
        if j != 0:
            st.markdown(50 * "_")

        left_panel, right_panel = st.columns([4, 5])

        with left_panel:
            col_ctrl, col_sig = st.columns(2)

            (code, new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound,
             sig_type, sig_amp, sig_freq, sig_phase, sig_step_time, sig_offset, sig_stop_time) = _show_constant_slidebars(
                orchestrator,
                supervisor,
                st.session_state["modified_code"],
                st.session_state["modified_controller_structure"],
                col_ctrl,
                col_sig,
                j
            )

            has_bounds_error = is_bounded and (min_bound != "" and max_bound != "") and (
                        float(min_bound) >= float(max_bound))

        with right_panel:
            if f"cached_plot_{j}" not in st.session_state:
                st.session_state[f"cached_plot_{j}"] = _show_performance_plots(
                    orchestrator, supervisor, code, st.session_state["modified_controller_structure"],
                    new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound,
                    sig_type, sig_amp, sig_freq, sig_phase, sig_step_time, sig_offset, sig_stop_time, j
                )

            cached_fig = st.session_state.get(f"cached_plot_{j}")
            if cached_fig is not None:
                st.plotly_chart(cached_fig, width='stretch')
            else:
                st.warning("⚠️ Waiting for simulation data to generate plot...")

        left_panel, right_panel = st.columns([4, 5])

        with left_panel:
            st.markdown('<div id="blue_btn"></div>', unsafe_allow_html=True)
            if st.button("💾 Apply New Values To Code", type="secondary", width='stretch', disabled=has_bounds_error,
                         key=f"apply_new_gains_{j}"):
                _save_changes(code, new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound, j)

        with right_panel:
            st.markdown('<div id="red_btn"></div>', unsafe_allow_html=True)
            if st.button("📊 Run Simulation & Update Plot", type="primary", width='stretch', key=f"run_sim_btn_{j}"):
                st.session_state[f"cached_plot_{j}"] = _show_performance_plots(
                    orchestrator, supervisor, code, st.session_state["modified_controller_structure"],
                    new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound,
                    sig_type, sig_amp, sig_freq, sig_phase, sig_step_time, sig_offset, sig_stop_time, j
                )

    if st.session_state.get("reset_sliders", False):
        st.session_state["reset_sliders"] = False

    st.markdown('<div id="blue_btn"></div>', unsafe_allow_html=True)
    if st.button("⏪️ Reset To Original", type="secondary", width='stretch', key="reset_to_original_gains"):
        _reset_values()


def _save_changes(code, new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound, controller_index):
    orchestrator = st.session_state["orchestrator"]
    loop_index = orchestrator.get_loop_index()

    st.session_state["modified_code"] = code
    st.session_state["modified_controller_structure"] = apply_pid_gains_to_controller_structure(
        st.session_state["modified_controller_structure"],
        loop_index,
        controller_index,
        new_kp,
        new_ki,
        new_kd,
        is_bounded,
        min_bound,
        max_bound
    )

    st.success("💾 Changes temporarily applied to session scratchpad!")
    st.rerun()


def _reset_values():
    orchestrator = st.session_state["orchestrator"]
    st.session_state["modified_code"] = orchestrator.get_equation()
    st.session_state["modified_controller_structure"] = copy.deepcopy(orchestrator.get_controller_structure())
    st.session_state["reset_sliders"] = True
    clear_cached_plots()
    st.info("⏪️ Reset scratchpad back to original optimization settings.")
    st.rerun()


def clear_cached_plots():
    for key in list(st.session_state.keys()):
        if key.startswith("cached_plot_"):
            del st.session_state[key]


def _show_constant_slidebars(orchestrator, supervisor, code, controller_structure, col_ctrl, col_sig, cont_index):
    loop_index = active_controller_index(orchestrator.get_loop_index())
    kp, ki, kd = get_pid_gains(controller_structure, orchestrator.get_loop_index(), cont_index)

    ctrl_output = controller_structure[loop_index]["controllers"][cont_index].get("controller_output", {})

    if st.session_state.get("reset_sliders", False):
        st.session_state[f"kp_input_{loop_index}{cont_index}"] = float(kp)
        st.session_state[f"ki_input_{loop_index}{cont_index}"] = float(ki)
        st.session_state[f"kd_input_{loop_index}{cont_index}"] = float(kd)

        is_bounded_default = bool(ctrl_output.get("is_bounded", False))
        st.session_state[f"is_bounded_check_slide_{loop_index}{cont_index}"] = is_bounded_default

        if is_bounded_default:
            st.session_state[f"min_bound_slide_{loop_index}{cont_index}"] = float(ctrl_output["min_bound"]) if ctrl_output.get(
                "min_bound") not in ["", None] else -1.0
            st.session_state[f"max_bound_slide_{loop_index}{cont_index}"] = float(ctrl_output["max_bound"]) if ctrl_output.get(
                "max_bound") not in ["", None] else 1.0

    controller = controller_structure[loop_index]["controllers"][cont_index]
    default_amp = orchestrator.generate_target(controller)

    kp_bound_up, ki_bound_up, kd_bound_up = get_pid_gain_upper_bounds(supervisor.get_final_states()[cont_index],
                                                                      controller, orchestrator.get_run_config())
    kp_bound_low, ki_bound_low, kd_bound_low = get_pid_gain_lower_bounds(supervisor.get_final_states()[cont_index],
                                                                      controller, orchestrator.get_run_config())

    with col_ctrl:
        st.markdown("### 🎛️ Controller Config")
        new_kp = st.number_input(
            "Proportional Gain (Kp)", min_value=float(kp_bound_low) * 20, max_value=100.0,
            value=float(kp), step=0.01, format="%.9f", key=f"kp_input_{loop_index}{cont_index}"
        )
        new_ki = st.number_input(
            "Integral Gain (Ki)", min_value=float(ki_bound_low) * 20, max_value=100.0,
            value=float(ki), step=0.01, format="%.9f", key=f"ki_input_{loop_index}{cont_index}"
        )
        new_kd = st.number_input(
            "Derivative Gain (Kd)", min_value=float(kd_bound_low) * 20, max_value=100.0,
            value=float(kd), step=0.01, format="%.9f", key=f"kd_input_{loop_index}{cont_index}"
        )

        is_bounded_default = bool(ctrl_output.get("is_bounded", False))
        is_bounded = st.checkbox("Apply Saturation Limits", value=is_bounded_default,
                                 key=f"is_bounded_check_slide_{loop_index}{cont_index}")

        if is_bounded:
            def_min = float(ctrl_output["min_bound"]) if ctrl_output.get("min_bound") not in ["", None] else -1.0
            def_max = float(ctrl_output["max_bound"]) if ctrl_output.get("max_bound") not in ["", None] else 1.0

            bound_col1, bound_col2 = st.columns(2)
            with bound_col1:
                min_bound = st.number_input("Lower Limit", value=def_min, step=0.1, key=f"min_bound_slide_{loop_index}{cont_index}")
            with bound_col2:
                max_bound = st.number_input("Upper Limit", value=def_max, step=0.1, key=f"max_bound_slide_{loop_index}{cont_index}")

            if min_bound >= max_bound:
                st.error("⚠️ Lower limit must be less than upper limit.")
        else:
            min_bound = ""
            max_bound = ""

    with col_sig:
        st.markdown("### 🎛️ Signal Config")

        target_signal = controller.get("signal_type", controller.get("objective", "Step"))
        sig_options = ["Step", "Ramp", "Sine", "Regulate", "Pulse"]
        default_idx = sig_options.index(target_signal) if target_signal in sig_options else 0

        sig_type = st.selectbox("Test Signal Type", sig_options, index=default_idx,
                                key=f"sig_type_{loop_index}{cont_index}")

        sig_amp, sig_freq, sig_phase, sig_step_time, sig_offset, sig_stop_time = default_amp, 0.5, 0.0, 1.0, 0.0, 2.0

        if sig_type == "Step":
            sig_amp = st.number_input("Step Value", value=default_amp, step=0.1,
                                      key=f"step_val_{loop_index}{cont_index}")
            sig_step_time = st.number_input("Step Time (s)", value=1.0, step=0.1,
                                            key=f"step_time_{loop_index}{cont_index}")
        elif sig_type == "Pulse":
            sig_amp = st.number_input("Pulse Amplitude", value=default_amp, step=0.1,
                                      key=f"pulse_amp_{loop_index}{cont_index}")
            sig_step_time = st.number_input("Pulse Start Time (s)", value=1.0, step=0.1,
                                            key=f"pulse_start_{loop_index}{cont_index}")
            sig_stop_time = st.number_input("Pulse Stop Time (s)", value=6.0, step=0.1,
                                            key=f"pulse_stop_{loop_index}{cont_index}")
        elif sig_type == "Regulate":
            sig_amp = st.number_input("Initial Error Offset", value=default_amp, step=0.1,
                                      key=f"reg_amp_{loop_index}{cont_index}")
            sig_step_time = 0.0
        elif sig_type == "Ramp":
            sig_amp = st.number_input("Ramp Slope", value=default_amp, step=0.1,
                                      key=f"ramp_slope_{loop_index}{cont_index}")
            sig_step_time = st.number_input("Ramp Start Time (s)", value=1.0, step=0.1,
                                            key=f"ramp_start_{loop_index}{cont_index}")
        elif sig_type == "Sine":
            sig_offset = st.number_input("Sine Offset (Average)", value=0.0, step=0.1,
                                         key=f"sine_offset_{loop_index}{cont_index}")
            sig_amp = st.number_input("Sine Amplitude", value=default_amp, step=0.1,
                                      key=f"sine_amp_{loop_index}{cont_index}")
            sig_freq = st.number_input("Sine Frequency (Hz)", value=0.25, step=0.1,
                                       key=f"sine_freq_{loop_index}{cont_index}")
            sig_phase = st.number_input("Sine Phase (rad)", value=0.0, step=0.1,
                                        key=f"sine_phase_{loop_index}{cont_index}")

    index = len(controller_structure[loop_index]["controllers"]) - cont_index - 1
    code = replace_last_pid_controller_gains(code, new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound, controller_index=index)

    return (code, new_kp, new_ki, new_kd, is_bounded, min_bound, max_bound,
            sig_type, sig_amp, sig_freq, sig_phase, sig_step_time, sig_offset, sig_stop_time)


def _show_performance_plots(orchestrator, supervisor, code, controller,
                            kp, ki, kd, is_bounded, min_ctrl, max_ctrl,
                            sig_type, sig_amp, sig_freq, sig_phase, sig_step_time,
                            sig_offset, sig_stop_time, controller_index):
    loop_index = max(0, orchestrator.get_loop_index() - 1)
    y_label = controller[loop_index]["controllers"][controller_index]["controlled_variable"]

    with st.spinner(f"Simulating final system response to a {sig_type} Input..."):
        reversed_index = len(controller[loop_index]["controllers"]) - controller_index - 1
        controller_structure = controller[loop_index]["controllers"][controller_index]
        input_channel_name = controller_structure["controlled_variable_in_equation"].capitalize()
        unit = controller_structure.get("target", {}).get("unit", "").capitalize()

        t, y, ref_val, trim_value, cont_signal = simulate_system_response(
            code,
            orchestrator.get_case_study(),
            controller_structure,
            reversed_index,
            kp, ki, kd, is_bounded, min_ctrl, max_ctrl,
            sig_type,
            amplitude=sig_amp,
            freq_hz=sig_freq,
            phase_rad=sig_phase,
            step_time=sig_step_time,
            offset=sig_offset,
            stop_time=sig_stop_time
        )

        match = re.search(r'\d+', input_channel_name)
        n = int(match.group())

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t, y=y[:, n], mode='lines',
            name='Actual Value', line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=t, y=ref_val, mode='lines',
            name='Reference Setpoint', line=dict(dash='dash', color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=t, y=trim_value, mode='lines',
            name='Trim Operating Point', line=dict(dash='dash', color='yellow', width=2)
        ))

        fig.update_layout(
            title=f"System Temporal Tracking Response: {sig_type} Input",
            xaxis_title="Time (seconds)",
            yaxis_title=f"{y_label} ({unit})",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )

        return fig


def display_edit_case_study_page():
    orchestrator = st.session_state["orchestrator"]

    page_container = st.empty()

    with page_container.container():
        controller_structure = copy.deepcopy(orchestrator.get_controller_structure())
        case_study = copy.deepcopy(orchestrator.get_case_study())

        # -----------------------------------------------------------------
        # 1. Initialize session_state defaults on first visit if not present
        # -----------------------------------------------------------------
        if "sim_bound_dt" not in st.session_state:
            st.session_state["sim_bound_dt"] = float(case_study.get("simulation_params", {}).get("dt", 0.001))
        if "sim_bound_max_time" not in st.session_state:
            st.session_state["sim_bound_max_time"] = float(case_study.get("simulation_params", {}).get("max_time", 50.0))

        for i, pid_loop in enumerate(controller_structure):
            metrics = pid_loop.get("metrics", {})
            if f"mse_input_{i}" not in st.session_state:
                st.session_state[f"mse_input_{i}"] = float(metrics.get("mse", 0.001))
            if f"settling_input_{i}" not in st.session_state:
                st.session_state[f"settling_input_{i}"] = float(metrics.get("settling_time", 7.0))
            if f"overshoot_input_{i}" not in st.session_state:
                st.session_state[f"overshoot_input_{i}"] = float(metrics.get("overshoot", 15.0))
            if f"effort_input_{i}" not in st.session_state:
                st.session_state[f"effort_input_{i}"] = float(metrics.get("control_effort", 0.25))

            for j, controller in enumerate(pid_loop.get("controllers", [])):
                ctrl_output = controller.get("controller_output", {})
                if f"is_bounded_check_{i}{j}" not in st.session_state:
                    st.session_state[f"is_bounded_check_{i}{j}"] = bool(ctrl_output.get("is_bounded", False))
                if f"min_bound_{i}{j}" not in st.session_state:
                    st.session_state[f"min_bound_{i}{j}"] = float(ctrl_output.get("min_bound", -1.0)) if ctrl_output.get("min_bound") not in ["", None] else -1.0
                if f"max_bound_{i}{j}" not in st.session_state:
                    st.session_state[f"max_bound_{i}{j}"] = float(ctrl_output.get("max_bound", 1.0)) if ctrl_output.get("max_bound") not in ["", None] else 1.0

                target_cfg = controller.get("target", {})
                if f"target_min_{i}{j}" not in st.session_state:
                    st.session_state[f"target_min_{i}{j}"] = float(target_cfg.get("min_value", 0.0))
                if f"target_max_{i}{j}" not in st.session_state:
                    st.session_state[f"target_max_{i}{j}"] = float(target_cfg.get("max_value", 0.0))

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            st.markdown('<div id="blue_btn"></div>', unsafe_allow_html=True)
            if st.button("⬅️ Back to Parameter Configurations", type="secondary", width='stretch',
                         key="btn_back_to_params"):
                st.session_state["mulo_designer_stage"] = "setup"
                st.rerun()

        with btn_col2:
            st.markdown('<div id="blue_btn"></div>', unsafe_allow_html=True)
            if st.button("🔄 Reset to Default Values", type="secondary", width='stretch',
                         key="btn_reset_defaults"):
                raw_case_study = orchestrator.get_case_study()
                default_cs = raw_case_study if raw_case_study else orchestrator.get_case_study()
                default_ctrl = raw_case_study.get("pid_loops", orchestrator.get_controller_structure())

                orchestrator.set_case_study(copy.deepcopy(default_cs))
                orchestrator.set_controller_structure(copy.deepcopy(default_ctrl))

                st.session_state["sim_bound_dt"] = float(default_cs.get("simulation_params", {}).get("dt", 0.001))
                st.session_state["sim_bound_max_time"] = float(default_cs.get("simulation_params", {}).get("max_time", 50.0))

                for i, pid_loop in enumerate(default_ctrl):
                    metrics = pid_loop.get("metrics", {})
                    st.session_state[f"mse_input_{i}"] = float(metrics.get("mse", 0.001))
                    st.session_state[f"settling_input_{i}"] = float(metrics.get("settling_time", 7.0))
                    st.session_state[f"overshoot_input_{i}"] = float(metrics.get("overshoot", 15.0))
                    st.session_state[f"effort_input_{i}"] = float(metrics.get("control_effort", 0.25))

                    for j, controller in enumerate(pid_loop.get("controllers", [])):
                        ctrl_output = controller.get("controller_output", {})
                        st.session_state[f"is_bounded_check_{i}{j}"] = bool(ctrl_output.get("is_bounded", False))
                        st.session_state[f"min_bound_{i}{j}"] = float(ctrl_output.get("min_bound", -1.0)) if ctrl_output.get("min_bound") not in ["", None] else -1.0
                        st.session_state[f"max_bound_{i}{j}"] = float(ctrl_output.get("max_bound", 1.0)) if ctrl_output.get("max_bound") not in ["", None] else 1.0

                        target_cfg = controller.get("target", {})
                        st.session_state[f"target_min_{i}{j}"] = float(target_cfg.get("min_value", 0.0))
                        st.session_state[f"target_max_{i}{j}"] = float(target_cfg.get("max_value", 0.0))

                st.rerun()

        with st.expander("⚙️ Live Case Study Parametric Modification", expanded=False):
            st.markdown("#### ⏳ **Simulation Parameters**")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                dt = st.number_input("Time Step Size Delta (dt)",
                                     step=0.0005, min_value=0.0001, max_value=0.1, format="%.7f", key="sim_bound_dt")
            with sim_col2:
                max_time = st.number_input("Run Time (s)",
                                           step=1.0, min_value=1.0, max_value=300.0, key="sim_bound_max_time")

            case_study["simulation_params"] = {"dt": dt, "max_time": max_time}
            st.markdown("---")

            st.markdown("#### 🎯 **Fixed Performance Targets**")
            has_bounds_error = False

            for i, pid_loop in enumerate(controller_structure):
                if i != 0:
                    st.markdown(50 * "_")
                st.markdown(f"##### **Loop Context: {pid_loop['loop_name'].replace('_', ' ').title()}**")

                loop_col1, loop_col2 = st.columns(2)
                with loop_col1:
                    mse = st.number_input("Mean Squared Error (mse)",
                                          step=100.0, min_value=0.0, format="%.3f", key=f"mse_input_{i}")

                    settling_time = st.number_input("Settling Time Threshold (s)",
                                                    step=0.5, min_value=0.0, key=f"settling_input_{i}")
                with loop_col2:
                    overshoot = st.number_input("Maximum Percentage Overshoot (%)",
                                                step=0.5, min_value=0.0, max_value=100.0, key=f"overshoot_input_{i}")

                    control_effort = st.number_input("Control Effort Penalty Weight",
                                                     min_value=0.0, step=0.1, key=f"effort_input_{i}")

                controller_structure[i]["metrics"] = {
                    "mse": mse,
                    "settling_time": settling_time,
                    "overshoot": overshoot,
                    "control_effort": control_effort
                }

                if pid_loop.get("controllers") and len(pid_loop["controllers"]) > 0:
                    for j, controller in enumerate(pid_loop["controllers"]):
                        ctrl_output_name = controller.get("output_signal", f"Signal {j+1}")

                        is_bounded = st.checkbox(
                            f"Apply Saturation Limits on {ctrl_output_name}",
                            key=f"is_bounded_check_{i}{j}"
                        )

                        controller_structure[i]["controllers"][j]["controller_output"]["is_bounded"] = is_bounded

                        if is_bounded:
                            bound_col1, bound_col2 = st.columns(2)
                            with bound_col1:
                                min_b = st.number_input("Lower Saturation Limit",
                                                        step=0.1, key=f"min_bound_{i}{j}")
                            with bound_col2:
                                max_b = st.number_input("Upper Saturation Limit",
                                                        step=0.1, key=f"max_bound_{i}{j}")

                            if min_b >= max_b:
                                st.error(
                                    f"⚠️ **Error:** Lower Saturation Limit ({min_b}) must be strictly less than Upper Saturation Limit ({max_b}).")
                                has_bounds_error = True
                        else:
                            min_b = ""
                            max_b = ""

                        controller_structure[i]["controllers"][j]["controller_output"]["min_bound"] = min_b
                        controller_structure[i]["controllers"][j]["controller_output"]["max_bound"] = max_b

                        # Inline Target Min / Max configuration
                        target_data = controller.get("target", {})
                        ctrl_var_name = controller.get("controlled_variable", f"Controller {j + 1}")
                        unit_label = f" ({target_data.get('unit')})" if target_data.get("unit") else ""

                        t_col1, t_col2 = st.columns(2)
                        with t_col1:
                            target_min = st.number_input(
                                f"{ctrl_var_name} Target Min{unit_label}",
                                step=0.01,
                                format="%.4f",
                                key=f"target_min_{i}{j}"
                            )
                        with t_col2:
                            target_max = st.number_input(
                                f"{ctrl_var_name} Target Max{unit_label}",
                                step=0.01,
                                format="%.4f",
                                key=f"target_max_{i}{j}"
                            )

                        if "target" not in controller_structure[i]["controllers"][j]:
                            controller_structure[i]["controllers"][j]["target"] = {}
                        controller_structure[i]["controllers"][j]["target"]["min_value"] = target_min
                        controller_structure[i]["controllers"][j]["target"]["max_value"] = target_max

        st.markdown('<div id="red_btn"></div>', unsafe_allow_html=True)

        if st.button(f"🚀 Run Controller Design Optimization (Loop {orchestrator.get_loop_index() + 1})",
                     type="primary",
                     width='stretch',
                     key="btn_run_optimization",
                     disabled=has_bounds_error):
            orchestrator.set_case_study(case_study)
            orchestrator.set_controller_structure(controller_structure)
            st.session_state["mulo_designer_stage"] = "run_designer"

            page_container.empty()
            st.rerun()