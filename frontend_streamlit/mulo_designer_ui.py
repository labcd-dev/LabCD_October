import streamlit as st
import queue
from typing import Any, Dict

st.set_page_config(layout="wide")

from frontend_streamlit.ga_agent_ui.ga_agent_home import display_home_page, run_worker_ui
from frontend_streamlit.ga_agent_ui.ga_agent_project import display_project_page
from frontend_streamlit.home_page_style import import_home_page_css_style
from frontend_streamlit.ga_agent_ui.ga_agent_utils import display_logo_home
from frontend_streamlit.mulo_designer_utils import display_edit_case_study_page, clear_cached_plots
from frontend_streamlit.ga_agent_ui.ga_agent_utils import empty_plot_data

from backend_core.MuloDesigner.GaAgent.src.callbacks import register_callback, unregister_callback


def _run_experiment_worker(
        supervisor_agent, event_queue) -> None:
    """
    Runs inside a daemon thread.
    1. Registers a progress callback with the thread-local registry.
    2. Calls run_ga_handler (blocking).
    3. Emits 'run_complete' or 'run_error' when finished.
    """

    def _push(event: Dict[str, Any]) -> None:
        """Non-blocking put; silently drop if queue is unexpectedly full."""
        try:
            event_queue.put_nowait(event)
        except Exception:
            pass

    register_callback(_push)
    # final_state = supervisor_agent.rule_based_control_design()
    final_state = supervisor_agent.run()
    _push({"event_type": "run_complete", "final_state": final_state})
    unregister_callback()


if "mulo_designer_stage" not in st.session_state:
    st.session_state["mulo_designer_stage"] = "setup"
if "selected_pipeline" not in st.session_state:
    st.session_state["selected_pipeline"] = "muloDesign"

import_home_page_css_style()

display_logo_home()

st.markdown(
    "<h1 style='text-align:center;margin-bottom:2px;'>Multi Loop Design</h1>"
    "<p style='text-align:center;color:gray;margin-top:0;'>"
    "LLM-Enhanced Genetic Algorithm for PID controller design"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if st.session_state["mulo_designer_stage"] == "setup":
    clear_cached_plots()
    display_home_page(False)
elif st.session_state["mulo_designer_stage"] == "edit_case_study":
    display_edit_case_study_page()
elif st.session_state["mulo_designer_stage"] == "run_designer":
    st.session_state["mulo_designer_stage"] = "project_page"
    run_worker_ui(_run_experiment_worker)
else:
    # Wrap the page in an empty container to explicitly destroy it on navigation
    project_page_container = st.empty()

    with project_page_container.container():
        display_project_page(False)
        if st.session_state["mulo_designer_stage"] == "optimisation_complete":
            orchestrator = st.session_state["orchestrator"]
            is_complete = orchestrator.get_loop_index() >= len(orchestrator.get_controller_structure())

            if not is_complete:
                st.markdown('<div id="red_btn"></div>', unsafe_allow_html=True)
                if st.button(f"🚀 Continue Controller Design (Loop {orchestrator.get_loop_index() + 1})", type="primary",
                             width='stretch'):
                    # 1. Update orchestrator states
                    cont_index = max(0, orchestrator.get_loop_index() - 1)
                    orchestrator.set_equation(st.session_state["modified_code"])
                    orchestrator.set_controller_structure(st.session_state["modified_controller_structure"])

                    # 2. Reset run states so the UI knows to start polling again!
                    st.session_state["run_complete"] = False
                    st.session_state["final_state"] = None
                    st.session_state["event_queue"] = queue.Queue()
                    st.session_state["plot_data"] = empty_plot_data()
                    st.session_state.pop("run_error", None)
                    st.session_state.pop("run_error_tb", None)
                    st.session_state.pop("modified_code", None)
                    st.session_state.pop("modified_controller_structure", None)
                    clear_cached_plots()

                    orchestrator.set_controller_designed(False)
                    orchestrator.set_final_state([])

                    # 3. Transition stage
                    st.session_state["mulo_designer_stage"] = "run_designer"

                    # 4. Completely clear the current page elements before rerunning to prevent UI ghosting
                    project_page_container.empty()
                    st.rerun()