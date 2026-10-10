import re
import os
from typing import Any, Dict, List, Tuple

from backend_core.MuloDesigner.utils import active_controller_index


def get_index(signal_str: str) -> int:
    match = re.search(r'([A-Za-z_]+)[\(\[](\d+)[\)\]]', signal_str)
    return int(match.group(2))

def replace_last_pid_controller_gains(
        code: str,
        kp: float,
        ki: float,
        kd: float,
        is_bounded: bool = False,
        min_bound: Any = "",
        max_bound: Any = "",
        controller_index: int = 0,
) -> str:
    """Replace the nth PIDController constructor from the end in generated controller code, safely handling tuples."""

    start_idx, end_idx, args_start = find_nth_controller(code, controller_index)
    if start_idx == -1:
        return code

    # Extract raw inner argument string
    args_str = code[args_start:end_idx]

    # 3. Smart split inner arguments by comma (top-level only)
    args_list = []
    level = 0
    current = []
    for char in args_str:
        if char == ',' and level == 0:
            args_list.append("".join(current).strip())
            current = []
        else:
            if char in '([{':
                level += 1
            elif char in ')]}':
                level -= 1
            current.append(char)
    if current:
        args_list.append("".join(current).strip())

    # 4. Filter out old output_limits while keeping other positional/keyword args (dt, trim, etc.)
    preserved_args = []
    if len(args_list) > 3:
        for arg in args_list[3:]:
            if not arg.startswith("output_limits"):
                preserved_args.append(arg)

    # 5. Build new arguments list
    new_args_list = [str(kp), str(ki), str(kd)] + preserved_args

    if is_bounded:
        new_args_list.append(f"output_limits=({min_bound}, {max_bound})")

    # 6. Reconstruct the string cleanly
    replacement = f"PIDController({', '.join(new_args_list)})"

    return code[:start_idx] + replacement + code[end_idx + 1:]


def find_nth_controller(code, controller_index):
    # 1. Find the start of the nth PIDController instantiation from the end
    target = "PIDController("
    search_end = len(code)
    start_idx = -1

    # Loop backwards 'controller_index + 1' times.
    # If controller_index = 0, it finds the very last one.
    # If controller_index = 1, it finds the second to last one, etc.
    for _ in range(controller_index + 1):
        start_idx = code.rfind(target, 0, search_end)
        if start_idx == -1:
            return -1, -1, -1

        # Shift the search window to end immediately before the current match
        search_end = start_idx

    args_start = start_idx + len(target)

    # 2. Find the true matching closing parenthesis for PIDController(...)
    bracket_level = 1
    end_idx = -1
    for i in range(args_start, len(code)):
        if code[i] == '(':
            bracket_level += 1
        elif code[i] == ')':
            bracket_level -= 1
            if bracket_level == 0:
                end_idx = i
                break

    if end_idx == -1:
        return -1, -1, -1

    return start_idx, end_idx, args_start


def apply_pid_gains_to_controller_structure(
    controller_structure: List[Dict[str, Any]],
    loop_index: int,
    controller_index: int,
    kp: float,
    ki: float,
    kd: float,
    is_bounded: bool = False,
    min_bound: Any = "",
    max_bound: Any = ""
) -> List[Dict[str, Any]]:
    """Apply PID gains and saturation bounds to the active controller structure."""
    loop_index = active_controller_index(loop_index)
    controller = controller_structure[loop_index]["controllers"][controller_index]
    
    controller["kp"] = kp
    controller["ki"] = ki
    controller["kd"] = kd
    
    controller["controller_output"]["is_bounded"] = is_bounded
    controller["controller_output"]["min_bound"] = min_bound
    controller["controller_output"]["max_bound"] = max_bound
    
    return controller_structure


def inject_all_zeroed_controllers(equation: str, case_study: Dict[str, Any], control_block: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Statically injects all controllers in this block into the Python text string
    with gains set to zero. Groups them in a new dynamics wrapper.
    """
    pid_controller_code = load_file("backend_core/MuloDesigner/pid_controller.py")

    # 1. Ensure base class exists
    if "class PIDController:" not in equation:
        equation += f"\n{pid_controller_code}\n"

    # 2. Determine wrapper function naming
    pattern = r'system_dynamics_controller_(\d+)'
    match = re.findall(pattern, equation)
    if match:
        match = [int(m) for m in match]
        new_controller = 'system_dynamics_controller_' + str(max(match) + 1)
        last_controller = 'system_dynamics_controller_' + str(max(match))
    else:
        new_controller = 'system_dynamics_controller_1'
        last_controller = 'system_dynamics'

    init_classes_code = ""
    update_logic_code = ""
    dt = case_study["simulation_params"]["dt"]

    # 3. Construct the text for initializing zero-gain controllers and calling update()
    for idx, cont in enumerate(control_block):
        # Safe unique naming
        clean_name = cont["controlled_variable_in_equation"].replace("[", "").replace("]", "")
        controller_name = f'{clean_name}PID_{idx}'
        cont["name"] = controller_name

        # Limits handling
        minB = cont['controller_output'].get('min_bound', "")
        maxB = cont['controller_output'].get('max_bound', "")
        bounds = f', output_limits=({minB}, {maxB})' if minB != "" and maxB != "" else ''

        # Trim selection
        out_var = cont['output_variable_in_equation']
        var_idx = get_index(out_var)
        if 'U' in out_var:
            trim_val = case_study["trim_values"][var_idx]
        else:
            trim_val = case_study["trim_ics"][var_idx]

        # Append the controller definition initialized at zero
        init_classes_code += f'\n{controller_name} = PIDController(0.0, 0.0, 0.0, {dt}{bounds}, trim={trim_val})'

        # Setup the execution block logic
        inp = cont["controlled_variable_in_equation"]
        out_var = out_var if 'U' in out_var else f'X[{get_index(out_var)}]'
        out = out_var if 'U' in out_var else f'setpoints["{out_var}"]'
        update_logic_code += f'\n    {out} = {controller_name}.update(setpoints["{inp}"], {inp})'
        # print(update_logic_code)

    # 4. Integrate into the main equation string
    equation += init_classes_code

    wrapper_func = f'\ndef {new_controller}(t, X, U, setpoints):'
    wrapper_func += update_logic_code
    sp_expr = "" if last_controller == "system_dynamics" else ", setpoints"
    wrapper_func += f'\n    return {last_controller}(t, X, U{sp_expr})'

    equation += f'\n\n{wrapper_func}'

    return equation, new_controller


def update_gains_on_equation(equation:str, cont:Dict[str, Any], target_index:int) -> str:
    """Uniform method to mutate the equation applying newly discovered gains"""
    kp = cont["kp"]
    ki = cont["ki"]
    kd = cont["kd"]
    is_bounded = cont["controller_output"]["is_bounded"]
    min_bound = cont["controller_output"]["min_bound"]
    max_bound = cont["controller_output"]["max_bound"]

    equation = replace_last_pid_controller_gains(
        equation,
        kp,
        ki,
        kd,
        is_bounded,
        min_bound,
        max_bound,
        target_index,
    )

    return equation

def load_file(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"The file {file_name} does not exist in the current directory.")
