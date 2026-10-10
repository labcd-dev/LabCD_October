from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from backend_core.MuloDesigner.data import (
    get_available_case_studies as get_backend_available_case_studies,
    load_case_study_content,
)

# -- Paths -----------------
REPO_ROOT = Path(__file__).parent
LOGO_PATH = REPO_ROOT / "assets" / "logo.svg"
CASE_STUDIES_DIR = REPO_ROOT / "case_studies" / "json"


# -- Logo helpers ----------

def _svg_content() -> str:
    """Return raw SVG string, or a minimal emoji fallback SVG."""
    if LOGO_PATH.exists():
        return LOGO_PATH.read_text(encoding="utf-8")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<text y="0.9em" font-size="56">🎛️</text></svg>'
    )


def display_logo_sidebar() -> None:
    """Render the SVG logo in the sidebar (compact)."""
    svg = _svg_content()
    svg = svg.replace('<svg', '<svg style="width:100%; height:auto; max-width:80px;"', 1)

    st.markdown(
        f'<div style="text-align:center;padding:12px 4px 4px;">'
        f'<div style="max-width:80px;margin:auto;">{svg}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def display_logo_home() -> None:
    """Render the SVG logo on the home page (larger)."""
    svg = _svg_content()
    svg = svg.replace('<svg', '<svg style="width:100%; height:auto; max-width:160px;"', 1)

    st.markdown(
        f'<div style="display:flex;justify-content:center;padding:28px 0 12px;">'
        f'<div style="max-width:160px;width:100%;">{svg}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# -- Case-study helpers --------------------------------------------------------

def get_available_case_studies() -> List[str]:
    """Return sorted list of *.json filenames in case_studies/json/."""
    return get_backend_available_case_studies(CASE_STUDIES_DIR)


def load_case_study_safe(selected_cs: str, run_by_mulo_designer: bool = True) -> Dict[str, Any]:
    """Load a case study via src.utils.load_case_study with graceful error handling."""
    if run_by_mulo_designer:
        return load_case_study_content(selected_cs, run_by_mulo_designer=True)

    try:
        return load_case_study_content(selected_cs, run_by_mulo_designer=False)
    except Exception as exc:
        st.error(f"Failed to load case study '{selected_cs}': {exc}")
        return {}


# -- Plot-data container -------------------------------------------------------

def empty_plot_data() -> Dict[str, Any]:
    """
    Return a freshly initialised container for streaming plot data.

    Now supports controller-specific data storage where each controller
    has its own dataset keyed by controller_name.
    """
    return {}


def initialize_controller_data(controller_name: str) -> Dict[str, Any]:
    """
    Initialize empty data structure for a specific controller.
    """
    return {
        "cumulative_nfe": [],
        "best_baseline_cost": [],
        "best_baseline_so_far": [],
        "mse": [],
        "settling_time": [],
        "overshoot": [],
        "control_effort": [],
        "Kp": [],
        "Ki": [],
        "Kd": [],
        "attempt": [],
        "success_score": [],
        "best_score_so_far": [],
        "attempt_boundaries_nfe": [],
        "attempt_summaries": [],
        "attempt_ranges": {},
        "controller_name": controller_name,
        "fixed_targets": {},  # <--- ADD THIS LINE
    }

def update_controller_plot_data(controller_name: str, event: Dict[str, Any]) -> None:
    """
    Update plot data for a specific controller with a new generation event.
    """
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = empty_plot_data()

    # Initialize controller data if not exists
    if controller_name not in st.session_state.plot_data:
        st.session_state.plot_data[controller_name] = initialize_controller_data(controller_name)

    data = st.session_state.plot_data[controller_name]

    # 1. Incoming instantaneous generation cost
    incoming_cost = event.get("best_baseline_cost")
    data["best_baseline_cost"].append(incoming_cost)

    # 2. UI-side running minimum computation
    prev_best = (
        data["best_baseline_so_far"][-1]
        if data["best_baseline_so_far"] and data["best_baseline_so_far"][-1] is not None
        else float("inf")
    )
    if incoming_cost is not None and incoming_cost != float("inf"):
        new_running_best = min(prev_best, float(incoming_cost))
    else:
        new_running_best = prev_best

    data["best_baseline_so_far"].append(
        new_running_best if new_running_best != float("inf") else None
    )

    # Append remaining telemetry
    data["cumulative_nfe"].append(event.get("cumulative_nfe", 0))
    data["mse"].append(event.get("mse", float('nan')))
    data["settling_time"].append(event.get("settling_time", float('nan')))
    data["overshoot"].append(event.get("overshoot", float('nan')))
    data["control_effort"].append(event.get("control_effort", float('nan')))
    data["Kp"].append(event.get("Kp", 0.0))
    data["Ki"].append(event.get("Ki", 0.0))
    data["Kd"].append(event.get("Kd", 0.0))
    data["attempt"].append(event.get("attempt", 1))
    data["success_score"].append(event.get("success_score", 0))
    data["best_score_so_far"].append(event.get("best_score_so_far", 0))
    data["fixed_targets"] = event.get("fixed_targets", {})

    # Update attempt boundaries if attempt changed
    if len(data["attempt"]) > 1 and data["attempt"][-1] != data["attempt"][-2]:
        data["attempt_boundaries_nfe"].append(data["cumulative_nfe"][-2])

def update_attempt_summary(controller_name: str, summary: Dict[str, Any]) -> None:
    """
    Update attempt summary for a specific controller.
    """
    if "plot_data" not in st.session_state:
        return

    if controller_name in st.session_state.plot_data:
        data = st.session_state.plot_data[controller_name]

        # Store attempt ranges if present
        if "param_ranges" in summary:
            att = summary.get("attempt", 1)
            if att not in data["attempt_ranges"]:
                data["attempt_ranges"][att] = {}
            data["attempt_ranges"][att] = summary.get("param_ranges", {})

        # Append summary to list
        data["attempt_summaries"].append(summary)


# -- Event-queue drain ---------------------------------------------------------

def drain_event_queue() -> None:
    """
    Non-blocking drain of st.session_state.event_queue into
    st.session_state.plot_data with controller-aware processing.
    """
    event_queue = st.session_state.get("event_queue")
    if event_queue is None:
        return

    if "plot_data" not in st.session_state:
        st.session_state.plot_data = empty_plot_data()

    # Drain all events from the queue
    while not event_queue.empty():
        try:
            event = event_queue.get_nowait()
            event_type = event.get("event_type")

            if event_type == "generation":
                # Update controller-specific plot data
                controller_name = event.get("controller_name", "Unknown Controller")
                update_controller_plot_data(controller_name, event)

            elif event_type == "attempt_complete":
                # Update attempt summary for specific controller
                controller_name = event.get("controller_name", "Unknown Controller")
                update_attempt_summary(controller_name, event)

            elif event_type == "run_complete":
                st.session_state.run_complete = True
                st.session_state.final_state = event.get("final_state")

            elif event_type == "run_error":
                st.session_state.run_complete = True
                st.session_state.run_error = event.get("error")
                st.session_state.run_error_tb = event.get("traceback")

        except Exception:
            # Queue empty or event malformed - skip
            pass