import copy
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
        # --- NEW: Historical target arrays ---
        "target_mse": [],
        "target_settling_time": [],
        "target_overshoot": [],
        "target_control_effort": [],
        # -------------------------------------
        "Kp": [],
        "Ki": [],
        "Kd": [],
        "attempt": [],
        "success_score": [],
        "best_score_so_far": [],
        "attempt_boundaries_nfe": [],
        "supervisor_boundaries_nfe": [],
        "supervisor_boundaries_labels": [],
        "attempt_summaries": [],
        "attempt_ranges": {},
        "controller_name": controller_name,
        "fixed_targets": {},
        "_nfe_offset": 0,
        "_last_raw_nfe": 0,
        "_global_attempt": 0,
        "_last_raw_attempt": None,
        "_pending_supervisor_boundary": False,
    }


def mark_supervisor_boundary(controller_name: str, event: Dict[str, Any]) -> None:
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = empty_plot_data()

    if controller_name not in st.session_state.plot_data:
        st.session_state.plot_data[controller_name] = initialize_controller_data(controller_name)

    data = st.session_state.plot_data[controller_name]
    if data["cumulative_nfe"]:
        boundary_nfe = data["cumulative_nfe"][-1]
        label = event.get("label", "Decision")  # <--- ADD THIS

        if boundary_nfe not in data["supervisor_boundaries_nfe"]:
            data["supervisor_boundaries_nfe"].append(boundary_nfe)
            data["supervisor_boundaries_labels"].append(label)  # <--- ADD THIS
        data["_pending_supervisor_boundary"] = True

def update_controller_plot_data(controller_name: str, event: Dict[str, Any]) -> None:
    """
    Update plot data for a specific controller with a new generation event,
    ensuring NFE and attempt numbers continue sequentially across runs.
    """
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = empty_plot_data()

    if controller_name not in st.session_state.plot_data:
        st.session_state.plot_data[controller_name] = initialize_controller_data(controller_name)

    data = st.session_state.plot_data[controller_name]

    raw_nfe = event.get("cumulative_nfe", 0)
    raw_attempt = event.get("attempt", 1)
    gen = event.get("generation", 1)

    # 1. Detect if the backend optimizer reset its NFE counter (e.g. new PSO / MIMO PSO / GA run)
    if data["cumulative_nfe"] and (raw_nfe < data["_last_raw_nfe"] or (gen == 1 and data["_pending_supervisor_boundary"])):
        # If raw_nfe restarted from a small number (<= last_raw_nfe), shift offset to the last plotted NFE
        if raw_nfe <= data["_last_raw_nfe"]:
            data["_nfe_offset"] = data["cumulative_nfe"][-1]

    adjusted_nfe = data["_nfe_offset"] + raw_nfe
    # Guarantee strictly non-decreasing NFE if retain=True already included partial offset
    if data["cumulative_nfe"] and adjusted_nfe <= data["cumulative_nfe"][-1]:
        data["_nfe_offset"] = data["cumulative_nfe"][-1]
        adjusted_nfe = data["_nfe_offset"] + raw_nfe

    data["_last_raw_nfe"] = raw_nfe

    # 2. Compute continuous global attempt number
    is_new_attempt = (
        data["_last_raw_attempt"] is None
        or raw_attempt != data["_last_raw_attempt"]
        or data["_pending_supervisor_boundary"]
        or (gen == 1 and len(data["cumulative_nfe"]) > 0)
    )

    if is_new_attempt:
        prev_nfe = data["cumulative_nfe"][-1] if data["cumulative_nfe"] else None
        data["_global_attempt"] += 1
        data["_last_raw_attempt"] = raw_attempt

        if prev_nfe is not None:
            if data["_pending_supervisor_boundary"]:
                # Red dotted line was already added by mark_supervisor_boundary;
                # consume the flag so we don't also draw a gray dashed line on top of it
                if prev_nfe not in data["supervisor_boundaries_nfe"]:
                    data["supervisor_boundaries_nfe"].append(prev_nfe)
            else:
                # Internal GA Agent attempt transition -> gray dashed line
                if prev_nfe not in data["attempt_boundaries_nfe"]:
                    data["attempt_boundaries_nfe"].append(prev_nfe)

        data["_pending_supervisor_boundary"] = False

    global_attempt = data["_global_attempt"]

    # 3. Incoming instantaneous generation cost
    incoming_cost = event.get("best_baseline_cost")
    data["best_baseline_cost"].append(incoming_cost)

    # 4. UI-side running minimum computation
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

    # 5. Append remaining telemetry using adjusted_nfe and global_attempt
    data["cumulative_nfe"].append(adjusted_nfe)
    data["mse"].append(event.get("mse", float('nan')))
    data["settling_time"].append(event.get("settling_time", float('nan')))
    data["overshoot"].append(event.get("overshoot", float('nan')))
    data["control_effort"].append(event.get("control_effort", float('nan')))
    targets = event.get("fixed_targets", {})
    data["target_mse"].append(targets.get("mse", float('nan')))
    data["target_settling_time"].append(targets.get("settling_time", float('nan')))
    data["target_overshoot"].append(targets.get("overshoot", float('nan')))
    data["target_control_effort"].append(targets.get("control_effort", float('nan')))
    data["Kp"].append(event.get("Kp", 0.0))
    data["Ki"].append(event.get("Ki", 0.0))
    data["Kd"].append(event.get("Kd", 0.0))
    data["attempt"].append(global_attempt)
    data["success_score"].append(event.get("success_score", 0))
    data["best_score_so_far"].append(event.get("best_score_so_far", 0))
    data["fixed_targets"] = targets  # Keep latest state

    if "param_ranges" in event and event["param_ranges"]:
        data["attempt_ranges"][global_attempt] = copy.deepcopy(event["param_ranges"])

def update_attempt_summary(controller_name: str, summary: Dict[str, Any]) -> None:
    """
    Update attempt summary for a specific controller using its continuous global attempt index.
    """
    if "plot_data" not in st.session_state:
        return

    if controller_name in st.session_state.plot_data:
        data = st.session_state.plot_data[controller_name]

        # Align summary attempt number with the continuous global attempt counter
        att = data["_global_attempt"] if data["_global_attempt"] > 0 else summary.get("attempt", 1)
        summary_copy = copy.deepcopy(summary)
        summary_copy["attempt"] = att

        if "param_ranges" in summary_copy:
            data["attempt_ranges"][att] = summary_copy.get("param_ranges", {})

        data["attempt_summaries"].append(summary_copy)

# -- Event-queue drain ---------------------------------------------------------

def drain_event_queue() -> None:
    """
    Non-blocking drain of st.session_state.event_queue into
    st.session_state.plot_data with controller-aware and rollback-aware processing.
    """
    event_queue = st.session_state.get("event_queue")
    if event_queue is None:
        return

    if "plot_data" not in st.session_state:
        st.session_state.plot_data = empty_plot_data()
    if "plot_data_snapshots" not in st.session_state:
        st.session_state.plot_data_snapshots = {}

    # Drain all events from the queue
    while not event_queue.empty():
        try:
            event = event_queue.get_nowait()
            event_type = event.get("event_type")

            if event_type == "generation":
                controller_name = event.get("controller_name", "Unknown Controller")
                update_controller_plot_data(controller_name, event)

            elif event_type == "attempt_complete":
                controller_name = event.get("controller_name", "Unknown Controller")
                update_attempt_summary(controller_name, event)

            elif event_type == "supervisor_boundary":
                controller_name = event.get("controller_name", "Unknown Controller")
                mark_supervisor_boundary(controller_name, event)  # <--- PASS event HERE

            elif event_type == "agent_snapshot":
                snap_id = event.get("snapshot_id")
                if snap_id is not None:
                    st.session_state.plot_data_snapshots[snap_id] = copy.deepcopy(st.session_state.plot_data)

            elif event_type == "rollback":
                snap_id = event.get("snapshot_id")
                if snap_id is not None and snap_id in st.session_state.plot_data_snapshots:
                    st.session_state.plot_data = copy.deepcopy(st.session_state.plot_data_snapshots[snap_id])

            elif event_type == "node_start":
                st.session_state.current_workflow_node = event.get("node_name", "")

            elif event_type == "agent_step_completed":
                st.session_state.latest_completed_agent_node = event.get("node_name", "")

            elif event_type == "run_complete":
                st.session_state.run_complete = True

            elif event_type == "run_error":
                st.session_state.run_complete = True
                st.session_state.run_error = event.get("error")
                st.session_state.run_error_tb = event.get("traceback")

        except Exception:
            # Queue empty or event malformed - skip
            pass