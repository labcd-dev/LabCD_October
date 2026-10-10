import json
import os
import numpy as np
from typing import Any, Dict, List

from backend_core.MuloDesigner.GaAgent.src.utils import coerce_metric_targets
from backend_core.MuloDesigner.equation_editor import inject_all_zeroed_controllers, update_gains_on_equation
from backend_core.MuloDesigner.tuning_engine import (
    get_index,
    add_constraint_to_controller,
    ga_agent_controller_tuner,
    pso_controller_tuner,
    fine_tuner_pso,
    dummy_controller_tuner,
    run_fixed_pid_handler
)


class MuloTuningOrchestrator:

    def __init__(self, run_config: Dict[str, Any], controller_structure: List[Any],
                 system_identification: Dict[str, Any], trimming_result: Dict[str, Any], equation: str):
        self.run_config = run_config
        self.equation = equation
        self.case_study = generate_case_study(system_identification, trimming_result, self.equation)
        self.controller_structure = add_constraint_to_controller(self.run_config, controller_structure, trimming_result)
        self.control_block = []
        self.loop_index = 0
        self.controller_index = 0
        self.controller_designed = False
        self.working_function = ""
        self.final_state = {}

    def initialize_control_block(self, tuning_order: List[str] = None):
        self.set_controller_index(0)

        i = int(self.get_loop_index())
        if i >= len(self.get_controller_structure()):
            print("design controller none")
            return None, None

        control_block = self.get_controller_structure()[i]["controllers"]
        control_block = self.change_tuning_order(control_block, tuning_order)

        # Pre-process the metrics and simulation configurations
        self.pre_processing(self.get_controller_structure()[i])

        # INJECTION: Add all controllers into the equation with 0-gains upfront
        self.equation, self.working_function = inject_all_zeroed_controllers(self.equation, self.case_study, control_block)
        
        return control_block
        

    def design_controller(self, control_block: List[Dict[str, Any]], controller_name: str, llm_agent: Any,
                          cost_function: str = "compute_metrics", coupling_ratio: float = 0.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # for j in range(len(control_block)):
        # if len(non_working_controllers) > 0 and j not in non_working_controllers:
        #     continue

        # Tune controller to find the new optimal gains
        j = self.controller_index
        control_block = self.tune_controller(control_block, j, self.working_function, cost_function, controller_name, coupling_ratio, llm_agent)

        target_index = len(control_block) - j - 1
        self.equation = update_gains_on_equation(self.equation, control_block[j], target_index)

        return self.get_final_state(), control_block


    def fine_tune_control_block(self, control_block: List[Dict[str, Any]], channels: List[Dict[str, Any]]):
        control_block = self.fine_tuner(control_block, channels)

        for j, cont in enumerate(control_block):
            target_index = len(control_block) - j - 1
            self.equation = update_gains_on_equation(self.equation, control_block[j], target_index)

        return self.get_final_state(), control_block

    def selective_redesign_controller(self, inner_loop_idx: int, inner_cont_idx: int,
                                      outer_loop_idx: int, outer_cont_idx: int,
                                      controller_name: str, llm_agent: Any) -> tuple[dict[str, Any], list[Any]]:
        """
        Optimizes an inner controller's gains by evaluating the step response
        of an outer controller's loop.
        """
        inner_cont = self.controller_structure[inner_loop_idx]["controllers"][inner_cont_idx]
        outer_cont = self.controller_structure[outer_loop_idx]["controllers"][outer_cont_idx]

        # Target the inner controller's position for gain injection
        inner_control_block = self.controller_structure[inner_loop_idx]["controllers"]
        outer_count = sum([len(loop["controllers"]) for loop_idx, loop in
                           enumerate(self.controller_structure) if loop_idx > inner_loop_idx])
        target_file_index = len(inner_control_block) + outer_count - inner_cont_idx - 1

        print()
        print(self.equation)
        print()
        print(target_file_index)
        print()

        self.update_case_study_for_selective_redesign(
            inner_cont, outer_cont, target_file_index, controller_name)

        if "skip" in inner_cont:
            gains, final_state = run_fixed_pid_handler(
                inner_cont["kp"], inner_cont["ki"], inner_cont["kd"],
                self.case_study, self.equation, self.run_config
            )
        elif self.run_config["optimizer_choice"] == "Agentic GA":
            gains, final_state = ga_agent_controller_tuner(self.run_config, self.case_study, llm_agent)
        elif self.run_config["optimizer_choice"] == "PSO Optimizer":
            gains, final_state = pso_controller_tuner(self.case_study, self.equation, self.run_config)
        else:
            gains, final_state = dummy_controller_tuner(self.run_config)

        self.final_state = final_state

        # Apply the newly discovered gains to the inner controller
        inner_cont["kp"] = gains[0]
        inner_cont["ki"] = gains[1]
        inner_cont["kd"] = gains[2]

        self.equation = update_gains_on_equation(self.equation, inner_cont, target_file_index)

        return self.get_final_state(), self.controller_structure

    def update_case_study_for_selective_redesign(self, inner_cont: Dict[str, Any], outer_cont: Dict[str, Any]
                                                 , target_file_index: int, controller_name: str) -> None:
        """
        Hybirdizes the case study: Simulation metrics come from the outer channel,
        but mutation bounds and equation indices target the inner channel.
        """
        outer_sim_config = self.get_simulator_config(outer_cont)

        # 1. Simulation evaluation criteria (Outer Controller)
        self.case_study["controller_name"] = controller_name
        self.case_study["output_channel"] = outer_sim_config["output_channel"]
        self.case_study["input_channel"] = outer_sim_config["input_channel"]
        self.case_study["input_name"] = outer_sim_config["input_name"]
        self.case_study["target"] = outer_sim_config["target"]
        self.case_study["signal_type"] = outer_sim_config["signal_type"]

        min_b = outer_sim_config["min_ctrl"]
        max_b = outer_sim_config["max_ctrl"]
        self.case_study["min_ctrl"] = float(min_b) if min_b != "" else ""
        self.case_study["max_ctrl"] = float(max_b) if max_b != "" else ""

        # 2. Optimization mutation targets (Inner Controller)
        self.case_study["controller_index"] = target_file_index
        self.case_study["param_ranges"] = inner_cont.get("param_ranges")

        if inner_cont.get("kp") is not None:
            self.case_study["initial_conditions"] = [[inner_cont["kp"], inner_cont["ki"], inner_cont["kd"]]]

        print()
        import pprint
        pprint.pprint(self.case_study)
        print()


    def change_tuning_order(self, control_block: List[Dict[str, Any]], tuning_order: List[str]) -> List[
        Dict[str, Any]]:
        if tuning_order == None:
            return control_block
        return control_block

    def pre_processing(self, pid_loop):
        """Prepare target tracking goals and simulation settings before modifying code"""
        self.case_study["fixed_targets"] = coerce_metric_targets(pid_loop.get('metrics'))
        self.case_study["coupled_channels"] = [get_index(cont["controlled_variable_in_equation"]) for cont in
                                               pid_loop["controllers"]]


    def fine_tuner(self, control_block: List[Dict[str, Any]], channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gains = []
        for index in range(len(control_block)):
            if "skip" in control_block[index]:
                cont_gains, final_state = run_fixed_pid_handler(
                    control_block[index]["kp"],
                    control_block[index]["ki"],
                    control_block[index]["kd"],
                    self.case_study,
                    self.equation,
                    self.run_config
                )
                # Convert the returned tuple into a dictionary to prevent the TypeError
                gains.append({"Kp": cont_gains[0], "Ki": cont_gains[1], "Kd": cont_gains[2]})
            else:
                gains, final_state = fine_tuner_pso(self.case_study, self.equation, self.run_config, channels)
                break

        self.final_state = final_state

        for index in range(len(control_block)):
            control_block[index]["kp"] = gains[index]["Kp"]
            control_block[index]["ki"] = gains[index]["Ki"]
            control_block[index]["kd"] = gains[index]["Kd"]

        return control_block


    def tune_controller(self, control_block: List[Dict[str, Any]], index: int,
                        working_function: str, cost_function: str, controller_name: str, coupling_ratio: int, llm_agent: Any) -> List[Dict[str, Any]]:
        cont = control_block[index]

        # Calculate inverse index targeting specifically for replace_last_pid_controller_gains
        target_file_index = len(control_block) - index - 1


        self.update_case_study(cont, working_function, target_file_index, cost_function, controller_name, coupling_ratio)

        simulator_config = self.get_simulator_config(cont)

        if "skip" in control_block[index]:
            gains, final_state = run_fixed_pid_handler(control_block[index]["kp"], control_block[index]["ki"],
                                                       control_block[index]["kd"], self.case_study, self.equation,self.run_config)

        elif self.run_config["optimizer_choice"] == "Agentic GA":
            gains, final_state = ga_agent_controller_tuner(self.run_config, self.case_study, llm_agent, simulator_config=simulator_config)
        elif self.run_config["optimizer_choice"] == "PSO Optimizer":
            gains, final_state = pso_controller_tuner(self.case_study, self.equation, self.run_config, simulator_config=simulator_config)
        else:
            gains, final_state = dummy_controller_tuner(self.run_config)

        kp, ki, kd = gains

        self.final_state = final_state

        control_block[index]["kp"] = kp
        control_block[index]["ki"] = ki
        control_block[index]["kd"] = kd

        return control_block

    def update_case_study(self, cont: Dict[str, Any], working_function: str, target_file_index: int, cost_function: str,
                          controller_name: str, coupling_ratio: int) -> None:
        simulator_config = self.get_simulator_config(cont)
        min_b = simulator_config["min_ctrl"]
        max_b = simulator_config["max_ctrl"]

        # Inject real-time controller name for UI tracking
        self.case_study["controller_name"] =  controller_name
        self.case_study["output_channel"] = simulator_config["output_channel"]
        self.case_study["input_channel"] = simulator_config["input_channel"]
        self.case_study["input_name"] = simulator_config["input_name"]
        self.case_study["min_ctrl"] = float(min_b) if min_b != "" else ""
        self.case_study["max_ctrl"] = float(max_b) if max_b != "" else ""
        self.case_study["target"] = simulator_config["target"]
        self.case_study["signal_type"] = simulator_config["signal_type"]
        self.case_study["controller_index"] = target_file_index
        if "param_ranges" in cont:
            self.case_study["param_ranges"] = cont["param_ranges"]
        else:
            self.case_study["param_ranges"] = {'Kp': [-50.0, 50.0], 'Ki': [-10.0, 10.0], 'Kd': [-10.0, 10.0]}

        if cont.get("kp") is not None:
            self.case_study["initial_conditions"] = [[cont["kp"], cont["ki"], cont["kd"]]]
            print("\ninitial_conditions")
            print(self.case_study["initial_conditions"], "\n")

        self.case_study['python_code'] = self.equation
        self.case_study['working_function'] = working_function
        self.case_study['cost_function'] = cost_function
        self.case_study['coupling_ratio'] = coupling_ratio

    def get_simulator_config(self, cont: Dict[str, Any]):
        input_name = cont['output_variable_in_equation']
        output_name = cont['controlled_variable_in_equation']

        return {
            "output_name": output_name,
            "input_name": input_name,
            "output_channel": get_index(output_name),
            "input_channel": get_index(input_name),
            "min_ctrl": cont['controller_output']['min_bound'],
            "max_ctrl": cont['controller_output']['max_bound'],
            "target": self.generate_target(cont),
            "signal_type": cont.get("signal_type", "Step"),
        }


    def generate_target(self, cont):
        min_target = float(cont['target']['min_value'])
        max_target = float(cont['target']['max_value'])
        # input_channel = cont['controlled_variable_in_equation']
        #
        # input_index = get_index(input_channel)
        # trim_value = self.case_study["trim_ics"][input_index]

        target = 0.0
        rng = np.random.default_rng(seed=self.run_config["seed"])
        while target == 0.0:
            target = round(rng.uniform(min_target, max_target), 3)

        # return target + trim_value
        return target

    # --- GETTERS AND SETTERS ---

    def get_run_config(self) -> Dict[str, Any]:
        return self.run_config

    def set_run_config(self, run_config: Dict[str, Any]) -> None:
        self.run_config = run_config

    def get_equation(self) -> str:
        return self.equation

    def set_equation(self, equation: str) -> None:
        self.equation = equation

    def get_case_study(self) -> Dict[str, Any]:
        return self.case_study

    def set_case_study(self, case_study: Dict[str, Any]) -> None:
        self.case_study = case_study

    def get_controller_structure(self) -> List[Any]:
        return self.controller_structure

    def set_controller_structure(self, controller_structure: List[Any]) -> None:
        self.controller_structure = controller_structure

    def get_loop_index(self) -> int:
        return self.loop_index

    def set_loop_index(self, index: int) -> None:
        self.loop_index = index
        
    def get_controller_index(self) -> int:
        return self.controller_index
    
    def set_controller_index(self, index: int) -> None:
        self.controller_index = index

    def get_controller_designed(self) -> bool:
        return self.controller_designed

    def set_controller_designed(self, designed: bool) -> None:
        self.controller_designed = designed

    def get_final_state(self) -> Dict[str, Any]:
        return self.final_state

    def set_final_state(self, final_state: Dict[str, Any]) -> None:
        self.final_state = final_state


def generate_case_study(system_identification: Dict[str, Any], trimming_result: Dict[str, Any], equation: str) -> Dict[str, Any]:
    trim_values = [float(num) for num in trimming_result["equilibrium"]["u_e"]]
    trim_ics = [float(num) for num in trimming_result["equilibrium"]["x_e"]]


    return {
        "system_name": system_identification["system_name"],
        "python_code": equation,
        "system_description": system_identification["description"],
        "control_objective": "Design a controller to regulate pitch attitude (theta) to a desired setpoint with minimal settling time, overshoot, and steady-state error",
        "target": None,
        "num_inputs": trimming_result["system"]["n_inputs"],
        "trim_values": trim_values,
        "trim_ics": trim_ics,
        "input_channel": None,
        "output_channel": None,
        "num_states": trimming_result["system"]["n_states"],
        "min_ctrl": None,
        "max_ctrl": None,
        "fixed_targets": {
            "mse": 0.001,
            "settling_time": 7.0,
            "overshoot": 10.0,
            "control_effort": 0.25
        },
        "simulation_params": {
            "dt": 0.001,
            "max_time": 50.0
        }
    }


def load_file(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"The file {file_name} does not exist in the current directory.")


if __name__ == "__main__":
    controller = load_file("inputs/aircraft.json")
    trimming = load_file("inputs/aircraft_trim.json")
    trimming = json.loads(trimming)
    pyEquation = load_file("inputs/aircraft_equation.py")
    system = load_file("inputs/aircraft_system_identification.json")
    system = json.loads(system)

    config = {
        'case_study_file': '',
        'tuning_specs': None,
        'llm_model': 'openai/gpt-oss-120b',
        'web_search_model': None,
        'seed': 42,
        'run_id': 1,
        'max_attempts': 5,
        'max_wall_clock': 120.0,
        'max_cost_budget': 1.0,
        'prompt_variant': "elaborate",
        'buffer_size': 3,
        'control_objective': 'Design a controller to regulate pitch attitude (theta) to a desired setpoint with minimal settling time, overshoot',
    }

    import pprint

    controller = json.loads(controller)
    pprint.pprint(controller)
    designer = MuloTuningOrchestrator(config, controller, system, trimming, pyEquation)
    controller, equation = designer.design_controller()
    pprint.pprint(equation)