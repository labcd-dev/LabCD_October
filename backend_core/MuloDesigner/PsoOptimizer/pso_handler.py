import time
from typing import Dict, Any, Type, Optional, List

from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.PsoOptimizer.pso_optimizer import PSOOptimizer
from backend_core.MuloDesigner.PsoOptimizer.mimo_pso_optimizer import MIMOPSOOptimizer

logger = get_logger(__name__)


# ==============================================================================
# UNIFIED CONFIGURATION BUILDERS
# ==============================================================================
def build_system_config(case_study: Dict[str, Any], simulator_config: Optional[Dict[str, Any]] = None,
                        idx: int = None) -> Dict[str, Any]:
    sim_params = case_study["simulation_params"]
    src = simulator_config if simulator_config is not None else case_study
    idx = idx if idx is not None else case_study['controller_index']

    return {
        'dt': sim_params['dt'],
        'max_time': sim_params['max_time'],
        'target': src['target'],
        'num_inputs': case_study['num_inputs'],
        'input_channel': src['input_channel'],
        'output_channel': src['output_channel'],
        'min_ctrl': src['min_ctrl'],
        'max_ctrl': src['max_ctrl'],
        'trim_values': case_study['trim_values'],
        'trim_ics': case_study['trim_ics'],
        'num_states': case_study['num_states'],
        'working_function': case_study['working_function'],
        'input_name': src['input_name'],
        'controller_index': idx,
        'coupled_channels': case_study['coupled_channels'],
        'cost_function': case_study['cost_function'],
        'coupling_ratio': case_study['coupling_ratio'],
        'signal_type': src['signal_type'],
    }


def generate_controller_name(control_block, j):
    return control_block[j]['controlled_variable'] + " Fine Tuner Agent"


def build_optimizer_config(case_study: Dict[str, Any], run_config: Dict[str, Any],
                           channels: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_config = {
        'seed': run_config.get("seed", 42),
        'experiment_start_time': time.time(),
        'max_wall_clock': float(run_config.get("max_wall_clock", 600.0)),
        'nfe_offset': 0,
        'current_attempt': 1,
        'population_size': int(run_config.get("population_size", 50)),
        'generations': int(run_config.get("generations", 20)),
        'weights': run_config["weights"],
    }

    if channels is not None:
        base_config.update({
            'param_ranges_list': [ch['param_ranges'] for ch in channels],
            'fixed_targets_list': [ch['fixed_targets'] for ch in channels],
            'weights_list': [ch.get('weights', run_config["weights"]) for ch in channels],
            'controller_names': [ch['controller_name'] for ch in channels],
        })
    else:
        base_config.update({
            'param_ranges': case_study.get("param_ranges", {}),
            'initial_conditions': case_study.get("initial_conditions", []),
            'fixed_targets': case_study['fixed_targets'],
            'controller_name': case_study["controller_name"],
        })

    return base_config


# ==============================================================================
# STATE FORMATTING & POLYMORPHIC EXECUTION
# ==============================================================================
def broadcast_summary_event(optimizer_config: Dict[str, Any], results: Dict[str, Any]):
    push_event = get_callback()
    if not push_event:
        return

    is_mimo = "controller_parameters_list" in results

    if is_mimo:
        params_list = results["controller_parameters_list"]
        for i in range(len(params_list)):
            summary_event = {
                "event_type": "attempt_complete",
                "controller_name": optimizer_config["controller_names"][i],
                "attempt": optimizer_config.get("current_attempt", 1),
                "pop_size": optimizer_config["population_size"],
                "num_gen": optimizer_config["generations"],
                "weights": optimizer_config["weights_list"][i],
                "param_ranges": optimizer_config["param_ranges_list"][i],
                "controller_gains": params_list[i],
                "time_remaining_pct": 100.0,
                "cost_remaining_pct": 100.0,
                "success_score": results.get("final_score", 0),
                "decision": "proceed",
            }
            push_event(summary_event)
            time.sleep(0.05)
    else:
        summary_event = {
            "event_type": "attempt_complete",
            "controller_name": optimizer_config.get("controller_name", "Unknown Controller"),
            "attempt": optimizer_config.get("current_attempt", 1),
            "pop_size": optimizer_config["population_size"],
            "num_gen": optimizer_config["generations"],
            "weights": optimizer_config.get("weights", {}),
            "param_ranges": optimizer_config.get("param_ranges", {}),
            "controller_gains": results.get("controller_parameters", {}),
            "time_remaining_pct": 100.0,
            "cost_remaining_pct": 100.0,
            "success_score": results.get("final_score", 0),
            "decision": "proceed",
        }
        push_event(summary_event)
        time.sleep(0.2)


def format_final_state(optimizer_config: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    is_mimo = "controller_parameters_list" in results
    gains = results.get("controller_parameters_list") if is_mimo else results.get("controller_parameters", {})
    param_ranges = optimizer_config.get("param_ranges_list") if is_mimo else optimizer_config.get("param_ranges")
    elapsed = time.time() - optimizer_config['experiment_start_time']
    current_attempt = optimizer_config.get("current_attempt", 1)

    progress = results.get("progress", {})
    iterations = len(progress.get("iteration", []))

    columns = ["attempt", "iteration", "cost", "Kp", "Ki", "Kd", "mse", "settling_time", "overshoot", "control_effort"]
    data = []
    last_cost = float('inf')

    def _rnd(val):
        if val is None: return None
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return val

    for i in range(iterations):
        current_cost = progress.get("best_cost", [])[i] if i < len(progress.get("best_cost", [])) else float('inf')

        if current_cost < last_cost:
            last_cost = current_cost
            it = progress.get("iteration", [])[i] if i < len(progress.get("iteration", [])) else i
            kp = progress.get("Kp", [])[i] if i < len(progress.get("Kp", [])) else None
            ki = progress.get("Ki", [])[i] if i < len(progress.get("Ki", [])) else None
            kd = progress.get("Kd", [])[i] if i < len(progress.get("Kd", [])) else None
            mse = progress.get("mse", [])[i] if i < len(progress.get("mse", [])) else None
            st = progress.get("settling_time", [])[i] if i < len(progress.get("settling_time", [])) else None
            os_val = progress.get("overshoot", [])[i] if i < len(progress.get("overshoot", [])) else None
            ce = progress.get("control_effort", [])[i] if i < len(progress.get("control_effort", [])) else None

            data.append([
                current_attempt, it, _rnd(current_cost), _rnd(kp), _rnd(ki), _rnd(kd),
                _rnd(mse), _rnd(st), _rnd(os_val), _rnd(ce)
            ])

    fixed_targets = optimizer_config.get("fixed_targets", optimizer_config.get("fixed_targets_list"))
    weights = optimizer_config.get("weights", optimizer_config.get("weights_list"))

    return {
        "best_result": {
            "gains": gains,
            "achieved_metrics": results.get("achieved_metrics", results.get("achieved_metrics_list", {})),
            "achieved_cost": results.get("baseline_cost", float('inf')),
            "success_score": results.get("final_score", 0)
        },
        "tuning_specs": {
            "target_metrics": fixed_targets,
            "weights": weights,
            "param_ranges": param_ranges
        },
        "evolution_history": {
            "columns": columns,
            "data": data
        }
    }


def execute_optimization_pipeline(case_study: Dict[str, Any], equation: str, run_config: Dict[str, Any],
                                  optimizer_cls: Type, channels: List[Dict[str, Any]] = None,
                                  simulator_config: Dict[str, Any] = None) -> Dict[str, Any]:
    is_mimo = False if channels is None else True

    label = "MIMO PSO" if is_mimo else "PSO"
    logger.info("=" * 80)
    logger.info(f"{label} HANDLER - Executing Controller Tuning")
    logger.info("=" * 80)

    if is_mimo:
        opt_config = build_optimizer_config(case_study, run_config, channels)
        system_configs = [
            build_system_config(case_study, channel_data["simulator_config"], len(channels) - idx - 1)
            for idx, channel_data in enumerate(channels)
        ]
        optimizer = optimizer_cls(system_configs, equation, opt_config)
        results = optimizer.optimize_pid()
    else:
        opt_config = build_optimizer_config(case_study, run_config)
        # FIX: Pass the explicit simulator_config to ensure dynamic objectives map correctly
        system_config = build_system_config(case_study, simulator_config)
        optimizer = optimizer_cls(system_config, equation, opt_config)
        results = optimizer.optimize_pid()

    broadcast_summary_event(opt_config, results)

    logger.info("=" * 80)
    logger.info(f"{label} EXECUTION COMPLETE")
    logger.info(f"  Baseline Cost: {results.get('baseline_cost', 0):.4f}")
    logger.info(f"  Final Cost:    {results.get('cost', 0):.4f}")
    logger.info("=" * 80)

    return format_final_state(opt_config, results)


# ==============================================================================
# ROUTING HANDLERS
# ==============================================================================
def run_pso_handler(case_study: Dict[str, Any], equation: str, run_config: Dict[str, Any],
                    simulator_config: Dict[str, Any] = None) -> Dict[str, Any]:
    return execute_optimization_pipeline(case_study, equation, run_config, optimizer_cls=PSOOptimizer,
                                         simulator_config=simulator_config)

def run_mimo_pso_handler(case_study: Dict[str, Any], equation: str, run_config: Dict[str, Any],
                         channels: List[Dict[str, Any]]) -> Dict[str, Any]:
    return execute_optimization_pipeline(case_study, equation, run_config, optimizer_cls=MIMOPSOOptimizer,
                                         channels=channels)