"""Pure data helpers for the Mulo Designer GA workflow."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


OPTIMIZATION_OPTIONS: List[str] = [
    "Dummy",
    "Agentic GA",
    "PSO Optimizer"
]


LLM_MODELS: List[str] = [
    "gpt-6-luna",
    "gpt-6-astra",
    "gpt-4o-mini",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-4o",
]

WEB_SEARCH_MODEL_OPTIONS: List[str] = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4o", "gpt-4o-mini"]

# DEFAULT_OBJECTIVES: Dict[str, str] = {
#     "aircraft": "Design a controller to regulate pitch attitude (theta) to a desired setpoint with minimal settling time, overshoot",
#     "active suspension": "Minimize chassis vertical acceleration and pitch/roll motions to improve ride comfort while maintaining tire road-holding capability.",
#     "auv depth": "Regulate depth (z) to a desired setpoint and stabilize pitch attitude (theta) using a single control input (u), compensating for hydrodynamic cross-coupling and positive buoyancy.",
#     "cstr": "Regulate reactor temperature and product concentration to desired setpoints by controlling coolant flow rate, despite highly nonlinear exothermic reaction dynamics.",
#     "diesel engine": "Regulate engine RPM and fuel injection timing to meet load demands while minimizing emissions.",
#     "quadcopter": "Stabilize roll, pitch, and yaw attitude angles along with altitude tracking control.",
#     "reaction wheel": "Regulate the angular position or attitude of the primary body by controlling the acceleration of the reaction wheel, while managing wheel saturation limits.",
#     "twin rotor": "Control the 2-DOF pitch and yaw angles to track a desired trajectory while decoupling the strong aerodynamic cross-coupling between the main and tail rotors.",
# }
DEFAULT_OBJECTIVES: Dict[str, str] = {
    "dc motor": "",
    "quanser heli": "",
    "aircraft": "",
    "active suspension": "",
    "auv depth": "",
    "cstr": "",
    "diesel engine": "",
    "quadcopter": "",
    "reaction wheel": "",
    "twin rotor": "",
}


def get_available_case_studies(case_studies_dir: Path) -> List[str]:
    """Return sorted list of JSON case-study filenames."""
    if not case_studies_dir.exists():
        return []
    return sorted(file_name for file_name in os.listdir(case_studies_dir) if file_name.endswith(".json"))


def load_case_study_content(
    selected_case_study: str,
    *,
    run_by_mulo_designer: bool = True,
    inputs_dir: Path = Path("frontend_streamlit/inputs"),
) -> Dict[str, Any]:
    """Load case-study content used by the Mulo Designer workflow."""
    if run_by_mulo_designer:
        content_paths = {
            "controller": inputs_dir / selected_case_study / "controller.json",
            "trimming": inputs_dir / selected_case_study / "trimming_result.json",
            "system": inputs_dir / selected_case_study / "system_identification.json",
            "equation": inputs_dir / selected_case_study / "equation.py",
        }
        content_list: Dict[str, Any] = {}

        for key, path in content_paths.items():
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if key != "equation":
                    content = json.loads(content)
            else:
                content = None

            content_list[key] = content

        return content_list

    from backend_core.MuloDesigner.GaAgent.src.utils import load_case_study

    return load_case_study(selected_case_study)


def empty_plot_data() -> Dict[str, Any]:
    """Return a freshly initialized container for streaming plot data."""
    return {
        "cumulative_nfe": [],
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
        "attempt_boundaries_nfe": [],
        "attempt_ranges": {},
        "attempt_weights": {},
        "attempt_pop_gen": {},
        "attempt_summaries": [],
    }


def apply_ga_event(plot_data: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a single GA event to plot data and return session-level updates."""
    event_type = event.get("event_type")
    session_updates: Dict[str, Any] = {}

    if event_type == "generation":
        attempt = event["attempt"]

        if plot_data["attempt"] and plot_data["attempt"][-1] != attempt:
            plot_data["attempt_boundaries_nfe"].append(event["cumulative_nfe"])

        plot_data["cumulative_nfe"].append(event["cumulative_nfe"])
        plot_data["best_baseline_so_far"].append(event["best_baseline_so_far"])
        plot_data["mse"].append(event.get("mse"))
        plot_data["settling_time"].append(event.get("settling_time"))
        plot_data["overshoot"].append(event.get("overshoot"))
        plot_data["control_effort"].append(event.get("control_effort"))
        plot_data["Kp"].append(event.get("Kp"))
        plot_data["Ki"].append(event.get("Ki"))
        plot_data["Kd"].append(event.get("Kd"))
        plot_data["attempt"].append(attempt)
        plot_data["success_score"].append(event.get("success_score", 0))

        if attempt not in plot_data["attempt_ranges"]:
            plot_data["attempt_ranges"][attempt] = event.get("param_ranges", {})
            plot_data["attempt_weights"][attempt] = event.get("weights", {})
            plot_data["attempt_pop_gen"][attempt] = {
                "pop": event.get("pop_size", 0),
                "gen": event.get("num_gen", 0),
            }

    elif event_type == "attempt_complete":
        plot_data["attempt_summaries"].append(event)

    elif event_type == "run_complete":
        session_updates["final_state"] = event.get("final_state")
        session_updates["run_complete"] = True

    elif event_type == "run_error":
        session_updates["run_error"] = event.get("error", "Unknown error")
        session_updates["run_error_tb"] = event.get("traceback", "")
        session_updates["run_complete"] = True

    return session_updates


def drain_event_queue(event_queue: Optional[Any], plot_data: Dict[str, Any]) -> Dict[str, Any]:
    """Drain queued GA events and return session-level updates."""
    if event_queue is None:
        return {}

    session_updates: Dict[str, Any] = {}
    while True:
        try:
            event = event_queue.get_nowait()
        except Exception:
            break

        session_updates.update(apply_ga_event(plot_data, event))

    return session_updates
