import numpy as np
from typing import Dict, Any
import re
from statistics import mean

from backend_core.MuloDesigner.equation_editor import replace_last_pid_controller_gains
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger

# Initialize module logger for the simulator
logger = get_logger(__name__)


class SystemSimulator:
    """Simulates control system and evaluates performance"""

    def __init__(self, system_config: Dict[str, Any], equation: str = None):
        """
        Args:
            system_config: Dict with dt, max_time, target, etc.
            dynamics_func: Optional pre-compiled function with signature (t, x, u) -> dx/dt
            equation: Optional python string to dynamically compile the dynamics function.
        """
        self.dt = system_config['dt']
        self.max_time = system_config['max_time']
        self.target = system_config['target']
        self.num_inputs = system_config['num_inputs']
        self.input_channel = system_config['input_channel']
        self.output_channel = system_config['output_channel']
        self.min_ctrl = system_config['min_ctrl']
        self.max_ctrl = system_config['max_ctrl']
        self.trim_values = np.array(system_config['trim_values'])
        self.trim_ics = np.array(system_config['trim_ics'])
        self.num_states = system_config['num_states']
        self.working_function = system_config['working_function']
        self.input_name = system_config['input_name']
        self.controller_index = system_config['controller_index']
        self.coupled_channels = system_config['coupled_channels']
        self.cost_function = system_config['cost_function']
        self.coupling_ratio = system_config['coupling_ratio']
        self.signal_type = system_config.get('signal_type', 'Step')
        self.equation = equation
        self.dynamics_func = lambda t, x, u, *args: np.array([0, 0])
        if equation:
            self._compile_dynamics(equation)

    def _compile_dynamics(self, equation: str):
        """Compiles the dynamic python string into a callable function."""
        exec_globals = {}
        exec(equation, exec_globals)
        if self.working_function in exec_globals:
            self.dynamics_func = exec_globals[self.working_function]

    def evaluate_pid(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate PID controller with fixed initial conditions (no randomness)"""
        Kp = params['Kp']
        Ki = params['Ki']
        Kd = params['Kd']

        signal_params = {
            'signal_type': getattr(self, 'signal_type', "Step"),
            'amplitude': self.target,
            'step_time': 1.0 if getattr(self, 'signal_type', "Step") != "Regulate" else 0.0,
            'stop_time': 6.0
        }

        t, y, ref_val, trim_value, cont_signal = self._simulate_pid(Kp, Ki, Kd, signal_params=signal_params)
        y_1d = y[:, self.output_channel]

        if len(y_1d) < len(t):
            result = {'success': False, 'metrics': None}
        else:
            if self.cost_function == 'compute_metrics':
                metrics = self._compute_metrics(y_1d, cont_signal, t, self.target, signal_params, ref_val)
                result = {'success': True, 'metrics': metrics}
            elif self.cost_function == 'compute_coupling_metrics':
                metrics = self._compute_mixed_metrics(y, cont_signal, t, signal_params, ref_val)
                result = {'success': True, 'metrics': metrics}

        if not result['success']:
            penalty_metrics = {
                'mse': 1e9,
                'settling_time': float(self.max_time),
                'overshoot': 1000.0,
                'control_effort': 1e6,
                'rise_time': float(self.max_time),
                'steady_state_error': 1e6,
                'control_zero_crossings': 0
            }
            return {
                'success': False,
                'metrics': penalty_metrics,
                'warnings': ['Simulation failed (instability) with fixed initial conditions']
            }

        return {
            'success': True,
            'metrics': result['metrics'],
            'controller_parameters': params,
            'warnings': []
        }

    def generate_reference(self, t: float, signal_type: str, amplitude: float = 1.0, freq_hz: float = 0.5,
                           phase_rad: float = 0.0, step_time: float = 1.0, offset: float = 0.0, stop_time: float = 2.0) -> float:
        """Generates standard control system test signals."""
        if signal_type == "Step":
            return amplitude if t >= step_time else 0.0
        elif signal_type == "Pulse":
            return amplitude if step_time <= t <= stop_time else 0.0
        elif signal_type == "Ramp":
            return amplitude * (t - step_time) if t >= step_time else 0.0
        elif signal_type == "Sine":
            return offset + amplitude * np.sin(freq_hz * t + phase_rad) if t >= 0 else 0.0
        return 0.0

    def _simulate_pid(self, Kp: float, Ki: float, Kd: float, signal_params: Dict[str, Any] = None):

        control_signals = []
        full_states = []
        ref_signals = []

        x = self.trim_ics.copy()

        is_regulation = (signal_params and signal_params.get('signal_type') == "Regulate")
        if is_regulation:
            x[self.output_channel] += self.target

        time_steps = np.arange(0.0, self.max_time, self.dt)
        setpoints = {f"X[{idx}]": float(val) for idx, val in enumerate(self.trim_ics)}

        is_bounded = (self.min_ctrl != "" and self.max_ctrl != "")

        equation = replace_last_pid_controller_gains(
            self.equation, Kp, Ki, Kd,
            is_bounded, self.min_ctrl, self.max_ctrl, self.controller_index
        )
        equation = self._adding_control_signal(equation, self.input_name)
        self._compile_dynamics(equation)

        state_trim_val = self.trim_ics[self.output_channel]
        controlled_var_name = f"X[{self.output_channel}]"

        for i, t in enumerate(time_steps):

            if is_regulation:
                current_target = state_trim_val
            elif signal_params:
                current_target = state_trim_val + self.generate_reference(t, **signal_params)
            else:
                current_target = state_trim_val + self.target

            setpoints[controlled_var_name] = current_target

            dx, control_signal = self.dynamics_func(t, x, self.trim_values.copy(), setpoints)

            x = x + dx * self.dt

            # Instability protection
            if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
                return (time_steps, np.array(full_states), np.array(ref_signals),
                        np.full(len(time_steps), state_trim_val),
                        np.array(control_signals))

            control_signals.append(control_signal)
            ref_signals.append(current_target)
            full_states.append(x.copy())

        return (time_steps, np.array(full_states), np.array(ref_signals), np.full(len(time_steps), state_trim_val),
                np.array(control_signals))

    @staticmethod
    def _adding_control_signal(equation: str, input_name: str):
        input_name = input_name if 'U' in input_name else f'X[{get_index(input_name)}]'
        cont_signal = input_name if input_name.startswith("U") else f'setpoints["{input_name}"]'

        # Isolate the VERY LAST function definition in the entire file
        parts = equation.rsplit('def ', 1)

        if len(parts) == 2:
            func_parts = parts[1].rsplit('return', 1)

            if len(func_parts) == 2:
                original_return_val = func_parts[1].rstrip()
                modified_return = f'return {original_return_val}, {cont_signal}\n'
                parts[1] = func_parts[0] + modified_return
                equation = 'def '.join(parts)

        return equation

    def _compute_metrics(self, trajectory: np.ndarray, control: np.ndarray, time: np.ndarray,
                         final_target: float, signal_params: Dict[str, float], ref_signal: np.ndarray = None) -> Dict[str, float]:
        """Compute performance metrics from trajectory"""

        change_indices = []
        phase_end_idx = len(trajectory)

        if ref_signal is not None:
            error = trajectory - ref_signal
            true_step_size = float(np.max(ref_signal) - np.min(ref_signal))

            # If reference is flat but there is error (Regulation mode), use initial error as step size
            if true_step_size < 1e-6 and abs(error[0]) > 1e-6:
                true_step_size = float(abs(error[0]))

            step_diffs = np.abs(np.diff(ref_signal))
            step_start_idx = int(np.argmax(step_diffs)) + 1 if np.max(step_diffs) > 1e-6 else 0
            actual_target = ref_signal[-1]

            # --- DYNAMIC WINDOWING: Prevent dilution on short pulses ---
            change_indices = np.where(step_diffs > 1e-6)[0]
            if len(change_indices) == 1:
                # Step: Active from step trigger to end
                active_steps = len(ref_signal) - change_indices[0]
            elif len(change_indices) > 1:
                # Pulse: Active width of the pulse only
                active_steps = change_indices[1] - change_indices[0]
                phase_end_idx = change_indices[1]  # Cutoff before the signal drops
            else:
                # Regulate / Constant
                active_steps = len(ref_signal)

            # Prevent division by zero
            active_steps = max(active_steps, 1)
        else:
            error = trajectory - final_target
            true_step_size = abs(final_target)
            step_start_idx = 0
            actual_target = final_target
            active_steps = len(trajectory)

        if true_step_size < 1e-6:
            true_step_size = 1.0

        # 1. MSE (Normalized strictly by the active duration of the stimulus)
        mse = float(np.sum(error ** 2) / active_steps)

        # --- ISOLATE EVALUATION WINDOW FOR TRANSIENT SIGNALS ---
        eval_error = error[:phase_end_idx]
        eval_time = time[:phase_end_idx]

        # 2. Settling time (Evaluated only on the active phase)
        settling_band = 0.02 * true_step_size
        settled_idx = np.where(np.abs(eval_error) > settling_band)[0]

        if len(settled_idx) > 0:
            # If it hasn't settled by the end of the evaluation window, issue maximum penalty
            if settled_idx[-1] == len(eval_error) - 1:
                settling_time = float(self.max_time)
            else:
                settling_time = eval_time[settled_idx[-1]]
        else:
            settling_time = eval_time[0]

        # 3. Overshoot & Undershoot Calculation (Phase-Aware & Step-Isolated)
        initial_error = np.max(np.abs(eval_error)) if len(eval_error) > 0 else 0.0

        if initial_error > 1e-6 and step_start_idx < len(eval_error):
            active_error = eval_error[step_start_idx:]
            approach_threshold = 0.1 * true_step_size

            active_approach_indices = np.where(np.abs(active_error) < approach_threshold)[0]

            if len(active_approach_indices) > 0:
                first_approach = active_approach_indices[0] + step_start_idx
                initial_sign = np.sign(eval_error[step_start_idx])

                if first_approach < len(eval_error) - 1:
                    errors_after_approach = eval_error[first_approach:]
                    crossed_indices = np.where(np.sign(errors_after_approach) == -initial_sign)[0]
                    if len(crossed_indices) > 0:
                        max_overshoot_error = np.max(np.abs(errors_after_approach[crossed_indices]))
                        overshoot_val = (max_overshoot_error / true_step_size) * 100.0
                    else:
                        overshoot_val = 0.0
                else:
                    overshoot_val = 0.0

                if first_approach > step_start_idx:
                    errors_before_approach = eval_error[step_start_idx:first_approach]
                    same_indices = np.where(np.sign(errors_before_approach) == initial_sign)[0]
                    if len(same_indices) > 0:
                        max_undershoot_error = np.max(np.abs(errors_before_approach[same_indices]))
                        if max_undershoot_error > true_step_size:
                            undershoot_val = ((max_undershoot_error - true_step_size) / true_step_size) * 100.0
                        else:
                            undershoot_val = 0.0
                    else:
                        undershoot_val = 0.0
                else:
                    # FIX: Catch condition where first_approach == step_start_idx
                    undershoot_val = 0.0

                overshoot = float(max(overshoot_val, undershoot_val))
            else:
                overshoot = 0.0
        else:
            overshoot = 0.0

        # 4. Normalized Control Effort
        if self.max_ctrl != "" and self.min_ctrl != "":
            max_abs_u = max(abs(float(self.min_ctrl)), abs(float(self.max_ctrl)))
            num_steps = len(control)
            max_possible_effort = max_abs_u * num_steps
        elif "U" in self.input_name:
            max_possible_effort = abs(self.trim_values[self.input_channel])
        else:
            max_possible_effort = abs(self.trim_ics[self.input_channel])

        control_effort = np.sum(np.abs(control)) * self.dt / max_possible_effort if max_possible_effort > 0 else 0.0

        # 5. Rise time (Bounded to the active phase)
        if true_step_size > 1e-6:
            error_mag = np.abs(eval_error)
            idx_10 = np.where(error_mag <= 0.9 * true_step_size)[0]
            idx_90 = np.where(error_mag <= 0.1 * true_step_size)[0]
            rise_time = (eval_time[idx_90[0]] - eval_time[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else float(
                self.max_time)
        else:
            rise_time = 0.0

        # 6. Steady-state error (For a pulse, evaluate exactly before it drops back down)
        if ref_signal is not None and len(change_indices) > 1:
            active_target = ref_signal[phase_end_idx - 1]
            eval_period = max(1, int(0.2 / self.dt))
            start_idx = max(0, phase_end_idx - eval_period)
            steady_state_error = abs(np.mean(trajectory[start_idx:phase_end_idx]) - active_target)
        else:
            steady_state_error = abs(np.mean(trajectory[-int(0.2 / self.dt):]) - actual_target) if len(
                trajectory) > 0 else 0.0

        control_signs = np.sign(control)
        control_signs_nonzero = control_signs[control_signs != 0]
        sign_changes = np.diff(control_signs_nonzero)

        settling_time = settling_time - (signal_params["step_time"] if signal_params["signal_type"] != "Regulate" else 0.0)

        return {
            'mse': float(mse),
            'settling_time': float(settling_time),
            'overshoot': float(overshoot),
            'control_effort': float(control_effort),
            'rise_time': float(rise_time),
            'steady_state_error': float(steady_state_error),
            'control_zero_crossings': int(np.count_nonzero(sign_changes))
        }


    def _compute_coupling_metrics(self, x: np.ndarray, control: np.ndarray, time: np.ndarray, signal_params: Dict[str, float]) -> Dict[str, float]:
        metrics_list = []

        if len(self.coupled_channels) <= 1:
            return {}

        for i in self.coupled_channels:
            if i == self.output_channel:
                continue
            y = x[:, i]
            channel_ref = np.full(len(time), self.trim_ics[i])
            metrics = self._compute_metrics(y, control, time, self.trim_ics[i], signal_params, channel_ref)
            metrics_list.append(metrics)

        coupling_metrics = {k: mean(d[k] for d in metrics_list) for k in metrics_list[0]}

        return coupling_metrics

    def _compute_mixed_metrics(self, x: np.ndarray, control: np.ndarray, time: np.ndarray,
                               signal_params: Dict[str, float], ref_signal: np.ndarray = None) -> Dict[str, float]:
        alpha = self.coupling_ratio
        coupling_metrics = self._compute_coupling_metrics(x, control, time, signal_params)

        x_1d = x[:, self.output_channel]
        metrics = self._compute_metrics(x_1d, control, time, self.target, signal_params, ref_signal)

        mixed_metrics = {k: metrics[k] * (1 - alpha) + coupling_metrics[k] * alpha for k in metrics}
        return mixed_metrics


def get_index(signal_str: str) -> int:
    match = re.search(r'([A-Za-z_]+)[\(\[](\d+)[\)\]]', signal_str)
    return int(match.group(2))


def simulate_system_response(equation: str, case_study: dict, controller_structure: Dict[str, Any],
                             controller_index: int,
                             Kp: float, Ki: float, Kd: float, is_bounded: bool, min_ctrl: float, max_ctrl: float,
                             signal_type: str = "Step", amplitude: float = 1.0, freq_hz: float = 0.5,
                             phase_rad: float = 0.0,
                             step_time: float = 1.0, offset: float = 0.0, stop_time: float = 2.0):
    state_var_name = controller_structure["controlled_variable_in_equation"].strip()
    actuator_var_name = controller_structure["output_variable_in_equation"].strip()

    if not is_bounded:
        min_ctrl = ""
        max_ctrl = ""

    system_config = {
        "dt": case_study["simulation_params"]["dt"],
        "max_time": case_study["simulation_params"]["max_time"],
        "target": case_study["target"],
        "num_inputs": case_study["num_inputs"],
        "input_channel": int(re.search(r'\d+', actuator_var_name).group()),
        "output_channel": int(re.search(r'\d+', state_var_name).group()),
        "min_ctrl": min_ctrl,
        "max_ctrl": max_ctrl,
        "trim_values": case_study["trim_values"],
        "trim_ics": case_study["trim_ics"],
        "num_states": case_study["num_states"],
        "working_function": case_study["working_function"],
        "input_name": actuator_var_name,
        "controller_index": controller_index,
        'coupled_channels': case_study['coupled_channels'],
        'cost_function': 'compute_metrics',
        'coupling_ratio': 0.0,
        'signal_type': signal_type  # Set directly from the function argument
    }



    simulator = SystemSimulator(system_config, equation)

    signal_params = {
        'signal_type': signal_type,
        'amplitude': amplitude,
        'freq_hz': freq_hz,
        'phase_rad': phase_rad,
        'step_time': step_time if signal_type != "Regulate" else 0.0,
        'offset': offset,
        'stop_time': stop_time  # NEW
    }

    print()
    print(signal_params)

    t, y, ref_val, trim_value, cont_signal = simulator._simulate_pid(Kp, Ki, Kd, signal_params=signal_params)

    state_idx = system_config["output_channel"]
    y_1d = y[:, state_idx]

    metrics = simulator._compute_metrics(y_1d, cont_signal, t, system_config["target"], signal_params, ref_val)
    coupling_metrics = simulator._compute_coupling_metrics(y, cont_signal, t, signal_params)

    print()
    print(state_var_name)
    print()
    print("metrics:")
    import pprint
    pprint.pprint(metrics)
    print()
    print("coupling metrics:")
    print(simulator.coupled_channels)
    pprint.pprint(coupling_metrics)

    return t, y, ref_val, trim_value, cont_signal