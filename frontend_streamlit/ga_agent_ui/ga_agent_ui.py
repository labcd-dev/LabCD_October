import streamlit as st

st.set_page_config(
    page_title="GA-Agent",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend_streamlit.ga_agent_ui.ga_agent_utils import display_logo_sidebar, empty_plot_data
from frontend_streamlit.ga_agent_ui.ga_agent_home import display_home_page
from frontend_streamlit.ga_agent_ui.ga_agent_project import display_project_page


# -- Session-state initialisation ---------------------------------------------

def _init_session() -> None:
    defaults = {
        "page":         "home",
        "run_config":   None,
        "event_queue":  None,
        "plot_data":    empty_plot_data(),  # Now a dict keyed by controller name
        "run_thread":   None,
        "run_complete": False,
        "final_state":  None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -- Sidebar -------------------------------------------------------------------

def _display_sidebar() -> None:
    with st.sidebar:
        display_logo_sidebar()
        st.markdown("---")

        # TODO: Session History Manager
        st.markdown("### 📋 Session History")
        st.caption("_Session history manager — coming soon._")
        st.info(
            "Previous experiments will appear here once the session "
            "history manager is implemented.",
            icon="🗂️",
        )


# -- Main ----------------------------------------------------------------------

_init_session()
_display_sidebar()

if st.session_state.page == "home":
    display_home_page()
else:
    display_project_page()