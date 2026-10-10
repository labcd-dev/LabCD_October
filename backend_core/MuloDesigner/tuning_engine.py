import json
import re
import copy
import time
import numpy as np
from typing import Any, Dict, List, Tuple

from backend_core.MuloDesigner.GaAgent.ga_handler_agent import run_ga_handler
from backend_core.MuloDesigner.PsoOptimizer.pso_handler import run_pso_handler, run_mimo_pso_handler
from backend_core.MuloDesigner.simulator import SystemSimulator
from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.Agents.mulo_design_agents import Agents

logger = get_logger(__name__)

# TODO: In final version this will be removed and replaced by an agent or user input data
def add_constraint_to_controller(run_config: Dict[str, Any], controller_structure: List[Any], trimming_result: Dict[str, Any]) -> List[Any]:
    if __name__ == "__main__":
        agents = Agents(model_name=run_config["llm_model"])
    else:
        agents = Agents(model_name=run_config["llm_model"])

    if run_config["optimizer_choice"] != "Dummy":
        if "metrics" in controller_structure["pid_loops"][0].keys():
            response = json.dumps(controller_structure)
        elif run_config.get("web_search_model") is not None:
            response = agents.constraint_estimator_web(controller_structure, trimming_result, run_config["web_search_model"], run_config["control_objective"])
        else:
            response = agents.constraint_estimator(controller_structure, trimming_result, run_config["control_objective"])

        response = response.replace("```json", "").replace("```", "")
        new_controller = json.loads(response)
        new_controller = sorted(new_controller["pid_loops"], key=lambda x: x['loop_number'])

        for i, pid_loop in enumerate(new_controller):
            for j in range(len(pid_loop["controllers"])):
                new_value = get_state_name(new_controller[i]["controllers"][j]["output_variable_in_equation"])
                new_controller[i]["controllers"][j]["output_variable_in_equation"] = new_value

        import pprint
        pprint.pprint(new_controller)

        return new_controller
    else:
        new_controller = copy.deepcopy(controller_structure['pid_loops'])
        new_controller = sorted(new_controller, key=lambda x: x.get('loop_number', 0))

        for i, pid_loop in enumerate(new_controller):
            for j, cont in enumerate(pid_loop["controllers"]):
                input_name = cont["controlled_variable"]
                input_unit = cont["input_unit"]
                output_unit = cont["output_unit"]
                output_name = cont["output_signal"]

                new_controller[i]["controllers"][j]["controller_output"] = {
                    'is_bounded': True,
                    'max_bound': 1,
                    'min_bound': -1,
                    'unit': output_unit,
                    'variable_name': output_name,
                }
                new_controller[i]["controllers"][j]["target"] = {
                    'description': input_name,
                    'max_value': 1.0,
                    'min_value': 0.0,
                    'unit': input_unit,
                }
            new_controller[i]['metrics'] = {
                'control_effort': 0.1,
                'mse': 0.01,
                'overshoot': 0.05,
                'settling_time': 1.5
            }
        return new_controller


def ga_agent_controller_tuner(run_config: Dict[str, Any], case_study: Dict[str, Any], llm_agent: Any = None,
                              simulator_config: Dict[str, Any] = None) -> Tuple[
    Tuple[float, float, float], Dict[str, Any]]:
    final_state = run_ga_handler(
        case_study_file=run_config["case_study_file"],
        tuning_specs=None,
        llm_model=run_config["llm_model"],
        seed=42,
        run_id=1,
        max_attempts=run_config["max_attempts"],
        max_wall_clock=run_config["max_wall_clock"],
        max_cost_budget=run_config["max_cost_budget"],
        prompt_variant=run_config["prompt_variant"],
        buffer_size=run_config["buffer_size"],
        control_objective=run_config.get("control_objective"),
        case_study=case_study,
        llm_agent_instance=llm_agent,
        simulator_config=simulator_config,
    )
    kp = float(final_state["best_result"]["gains"]["Kp"])
    ki = float(final_state["best_result"]["gains"]["Ki"])
    kd = float(final_state["best_result"]["gains"]["Kd"])

    return (kp, ki, kd), final_state


# In tuning_engine.py

def pso_controller_tuner(case_study: Dict[str, Any], equation: str, run_config: Dict[str, Any],
                         simulator_config: Dict[str, Any] = None) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    final_state = run_pso_handler(case_study, equation, run_config, simulator_config=simulator_config)

    kp = float(final_state["best_result"]["gains"]["Kp"])
    ki = float(final_state["best_result"]["gains"]["Ki"])
    kd = float(final_state["best_result"]["gains"]["Kd"])

    return (kp, ki, kd), final_state


def fine_tuner_pso(case_study: Dict[str, Any], equation: str, run_config: Dict[str, Any], channels: List[Dict[str, Any]])-> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    final_state = run_mimo_pso_handler(case_study, equation, run_config, channels=channels)
    gains = final_state["best_result"]["gains"]
    return gains, final_state

def run_fixed_pid_handler(
        kp: float,
        ki: float,
        kd: float,
        case_study: Dict[str, Any],
        equation: str,
        run_config: Dict[str, Any]
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    experiment_start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"FIXED PID HANDLER - Evaluating Gains: Kp={kp}, Ki={ki}, Kd={kd}")
    logger.info("=" * 80)

    controller_name = case_study.get("controller_name", "Unknown Controller")
    working_function = case_study.get('working_function', 'system_dynamics')

    system_config = {
        'dt': case_study["simulation_params"]['dt'],
        'max_time': case_study["simulation_params"]['max_time'],
        'target': case_study['target'],
        'num_inputs': case_study['num_inputs'],
        'input_channel': case_study['input_channel'],
        'output_channel': case_study['output_channel'],
        'min_ctrl': case_study['min_ctrl'],
        'max_ctrl': case_study['max_ctrl'],
        'trim_values': case_study['trim_values'] or [0.0] * case_study.get('num_inputs', 1),
        'trim_ics': case_study.get('trim_ics', [0.0] * case_study.get('num_states', 2)),
        'num_states': case_study.get('num_states', 2),
        'working_function': working_function,
        'input_name': case_study.get('input_name', ""),
        'controller_index': case_study['controller_index'],
        'coupled_channels': case_study['coupled_channels'],
        'cost_function': case_study['cost_function'],
        'coupling_ratio': case_study['coupling_ratio'],
        'signal_type': case_study.get('signal_type', 'Step'),
    }

    # 1. Initialize Simulator and Evaluate
    simulator = SystemSimulator(system_config, equation)
    result = simulator.evaluate_pid({'Kp': kp, 'Ki': ki, 'Kd': kd})

    # GUARANTEE METRICS EXIST TO AVOID KEYERRORS IN THE SUPERVISOR
    achieved_metrics = result.get('metrics', {})
    if not achieved_metrics:
        achieved_metrics = {
            "mse": float('inf'),
            "settling_time": float('inf'),
            "overshoot": float('inf'),
            "control_effort": float('inf')
        }

    baseline_cost = float('inf')
    if result['success']:
        fixed_targets = case_study.get('fixed_targets', {})
        cost = 0.0
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if metric in achieved_metrics and metric in fixed_targets:
                achieved = achieved_metrics[metric]
                target = fixed_targets[metric]
                cost += (achieved - target) / max(target, 1e-6)
        baseline_cost = cost

    param_ranges = case_study.get("param_ranges", {})
    weights = run_config.get("weights", {})
    pop_size = run_config.get("population_size", 10)
    num_gen = run_config.get("generations", 10)

    # 2. Push Callbacks (Now includes the missing 'generation' event)
    push_event = get_callback()
    if push_event:
        gen_event = {
            "event_type": "generation",
            "controller_name": controller_name,
            "attempt": 1,
            "generation": 1,
            "cumulative_nfe": 1,
            "cumulative_wall_time": time.time() - experiment_start_time,
            "best_baseline_so_far": baseline_cost,
            "best_baseline_cost": baseline_cost if baseline_cost != float('inf') else None,
            "success_score": 100 if result['success'] else 0,
            "best_score_so_far": 100 if result['success'] else 0,
            "mse": achieved_metrics.get("mse", float('inf')),
            "settling_time": achieved_metrics.get("settling_time", float('inf')),
            "overshoot": achieved_metrics.get("overshoot", float('inf')),
            "control_effort": achieved_metrics.get("control_effort", float('inf')),
            "Kp": kp,
            "Ki": ki,
            "Kd": kd,
            "param_ranges": param_ranges,
            "weights": weights,
            "pop_size": pop_size,
            "num_gen": num_gen,
        }
        push_event(gen_event)
        time.sleep(0.1)

        summary_event = {
            "event_type": "attempt_complete",
            "controller_name": controller_name,
            "attempt": 1,
            "pop_size": pop_size,
            "num_gen": num_gen,
            "weights": weights,
            "param_ranges": param_ranges,
            "controller_gains": {"Kp": kp, "Ki": ki, "Kd": kd},
            "time_remaining_pct": 100.0,
            "cost_remaining_pct": 100.0,
            "success_score": 100 if result['success'] else 0,
            "decision": "proceed",
        }
        push_event(summary_event)
        time.sleep(0.2)

        # 3. Format pre-defined final state
        final_state = {
            "best_result": {
                "gains": {"Kp": kp, "Ki": ki, "Kd": kd},
                "achieved_metrics": achieved_metrics,
                "achieved_cost": baseline_cost,
                "success_score": 100 if result['success'] else 0
            },
            "tuning_specs": {
                "target_metrics": case_study.get("fixed_targets", {}),
                "weights": weights,
                "param_ranges": param_ranges
            },
            "evolution_history": {
                "columns": ["attempt", "iteration", "cost", "Kp", "Ki", "Kd", "mse", "settling_time", "overshoot",
                            "control_effort"],
                "data": [
                    [1, 1, baseline_cost, kp, ki, kd, achieved_metrics.get("mse"),
                     achieved_metrics.get("settling_time"),
                     achieved_metrics.get("overshoot"), achieved_metrics.get("control_effort")]
                ]
            }
        }

        return (kp, ki, kd), final_state

def dummy_controller_tuner(run_config: Dict[str, Any]) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    rng = np.random.default_rng(seed=run_config.get("seed", 42))
    final_kp = round(rng.uniform(1.5, 10.5), 2)
    final_ki = round(rng.uniform(1.5, 10.5), 2)
    final_kd = round(rng.uniform(1.5, 10.5), 2)

    push_event = get_callback()
    if push_event:
        generations = 10
        nfe_per_gen = 20
        attempt = 1

        for gen in range(1, generations + 1):
            nfe = gen * nfe_per_gen
            fake_cost = 10.0 / gen
            fake_mse = 0.05 / gen
            fake_st = 15.0 / gen
            fake_os = 25.0 / gen
            fake_ce = 1.0 / gen
            fake_score = min(100, int(gen * 10))

            curr_kp = final_kp * (gen / generations)
            curr_ki = final_ki * (gen / generations)
            curr_kd = final_kd * (gen / generations)

            gen_event = {
                "event_type": "generation",
                "attempt": attempt,
                "cumulative_nfe": nfe,
                "best_baseline_so_far": fake_cost,
                "metrics": {
                    "mse": fake_mse,
                    "settling_time": fake_st,
                    "overshoot": fake_os,
                    "control_effort": fake_ce,
                },
                "gains": {"Kp": curr_kp, "Ki": curr_ki, "Kd": curr_kd},
                "search_ranges": {"Kp": [0.0, 15.0], "Ki": [0.0, 15.0], "Kd": [0.0, 15.0]},
                "success_score": fake_score,
            }

            push_event(gen_event)
            time.sleep(0.1)

        summary_event = {
            "event_type": "attempt_complete",
            "attempt": attempt,
            "pop_size": 20,
            "num_gen": generations,
            "weights": {"mse": 0.4, "settling_time": 0.3, "overshoot": 0.2, "control_effort": 0.1},
            "param_ranges": {"Kp": [0.0, 15.0], "Ki": [0.0, 15.0], "Kd": [0.0, 15.0]},
            "controller_gains": {"Kp": final_kp, "Ki": final_ki, "Kd": final_kd},
            "time_remaining_pct": 85.0,
            "cost_remaining_pct": 98.0,
            "success_score": 100,
        }
        push_event(summary_event)
        time.sleep(0.1)

    dummy_final_state = {
        "best_result": {
            "controller_name": "Dummy Controller",
            "best_attempt": attempt,
            "best_baseline_cost": 10.0 / generations,
            "best_cost": 10.0 / generations,
            "controller_parameters": {"Kp": final_kp, "Ki": final_ki, "Kd": final_kd},
            "achieved_metrics": {
                "mse": 0.05 / generations,
                "settling_time": 15.0 / generations,
                "overshoot": 25.0 / generations,
                "control_effort": 1.0 / generations,
            },
            "best_ga_config": {
                "population_size": 20,
                "generations": generations,
                "param_ranges": {"PID": {"Kp": [0.0, 15.0], "Ki": [0.0, 15.0], "Kd": [0.0, 15.0]}}
            },
            "tuning_specs": {
                "weights": {"mse": 0.4, "settling_time": 0.3, "overshoot": 0.2, "control_effort": 0.1},
                "fixed_targets": {}
            }
        },
        "evolution_history": {
            "columns": ["iteration", "cost", "Kp", "Ki", "Kd", "mse", "settling_time", "overshoot", "control_effort"],
            "data": []
        },
        "decision": {"action": "proceed", "reason": "Dummy evaluation completed."},
        "feedback_history": []
    }

    return (final_kp, final_ki, final_kd), dummy_final_state


def get_index(signal_str: str) -> int:
    match = re.search(r'([A-Za-z_]+)[\(\[](\d+)[\)\]]', signal_str)
    return int(match.group(2))


def get_state_name(signal_str: str) -> str:
    if signal_str.startswith("X_sp"):
        match = re.search(r'\d+', signal_str)
        if match:
            n = int(match.group())
            return f'X[{n}]'
    return signal_str