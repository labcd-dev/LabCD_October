from typing import Any, Dict, List, Tuple

def active_controller_index(controller_index: int) -> int:
    """Return the zero-based active controller index used by the UI workflow."""
    return max(0, int(controller_index) - 1)

def controller_loop_name(controller_structure: List[Dict[str, Any]], controller_index: int) -> str:
    """Return the display name for the active controller loop."""
    cont_index = active_controller_index(controller_index)
    return controller_structure[cont_index]["loop_name"].replace("_", " ")

def get_pid_gains(controller_structure: List[Dict[str, Any]], loop_index: int, controller_index: int) -> Tuple[float, float, float]:
    """Return PID gains for the active controller."""
    loop_index = active_controller_index(loop_index)
    controller = controller_structure[loop_index]["controllers"][controller_index]
    return float(controller["kp"]), float(controller["ki"]), float(controller["kd"])

def get_pid_gain_upper_bounds(final_state: Dict[str, Any], controller_structure: Dict[str, Any],
                              run_config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return PID gain bounds from the final optimizer state."""
    if run_config["optimizer_choice"] == "Agentic GA":
        param_ranges = final_state["tuning_specs"]["param_ranges"]
    else:
        param_ranges = controller_structure["param_ranges"]
    return param_ranges["Kp"][1], param_ranges["Ki"][1], param_ranges["Kd"][1]

def get_pid_gain_lower_bounds(final_state: Dict[str, Any], controller_structure: Dict[str, Any],
                              run_config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return PID gain bounds from the final optimizer state."""
    if run_config["optimizer_choice"] == "Agentic GA":
        param_ranges = final_state["tuning_specs"]["param_ranges"]
    else:
        param_ranges = controller_structure["param_ranges"]
    return param_ranges["Kp"][0], param_ranges["Ki"][0], param_ranges["Kd"][0]