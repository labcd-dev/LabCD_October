import numpy as np
from typing import Dict, Any
import time
from datetime import datetime

from backend_core.MuloDesigner.GaAgent.src.graph import create_ga_handler_graph
from backend_core.MuloDesigner.GaAgent.src.utils import (
    coerce_float,
    coerce_metric_targets,
    coerce_simulation_params,
    load_case_study,
)
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Each GA attempt runs design -> optimize -> evaluate (3 graph steps).
_STEPS_PER_ATTEMPT = 3
_MIN_RECURSION_LIMIT = 100


def get_recursion_limit(max_attempts: int) -> int:
    """Size LangGraph recursion limit from attempt budget (+ buffer for warm-start path)."""
    attempts = max(int(max_attempts), 1)
    return max(_MIN_RECURSION_LIMIT, attempts * _STEPS_PER_ATTEMPT + 20)


# =============================== MAIN FUNCTIONS ===============================

def initialize_ga_handler_state(
        case_study: Dict[str, Any],
        tuning_specs: Dict[str, Any],
        llm_model: str = "openai/gpt-oss-20b",
        seed: int = 42,
        run_id: int = 1,
        temperature: float = 0.0,
        max_attempts: int = 5,
        max_wall_clock: float = 60.0,
        max_cost_budget: float = 0.01,
        prompt_variant: str = "elaborate",
        buffer_size: int = 3,
        warm_start_config=None,
        system_aware: bool = True,
        llm_agent_instance: Any = None,
        optimizer_type: str = "PSO",
        simulator_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Initialize GA handler state with fixed targets and adjustable weights"""

    np.random.seed(seed)
    src = simulator_config if simulator_config is not None else case_study

    # 1. Safely extract the agentic context dictionary
    feedback_history = case_study.get("feedback_history", {})

    # 2. Extract the raw list, defaulting to [] to prevent LangGraph crashes
    restored_history = feedback_history.get("raw_feedback_history", [])
    if not isinstance(restored_history, list):
        restored_history = []

    state = {
        "current_attempt": feedback_history.get("total_attempts", 0) + 1,
        "experiment_start_time": time.time(),  # set once; shared across all GA attempts
        "max_attempts": max(int(coerce_float(max_attempts, 5)), 1),
        "max_wall_clock": coerce_float(max_wall_clock, 3600.0),
        "total_elapsed_time": 0.0,
        "prompt_variant": prompt_variant,
        "buffer_size": buffer_size,
        "feedback_history": restored_history,  # Safely injected list
        "best_result": None,
        "ga_config": None,
        "optimization_results": None,
        "decision": None,
        "tuning_specs": tuning_specs,
        "llm_model": llm_model,
        "run_id": run_id,
        "seed": seed,
        "temperature": temperature,
        "system_name": case_study["system_name"],
        "num_states": case_study["num_states"],
        "control_objective": case_study.get("control_objective", "Design a stable controller"),
        "dt": tuning_specs['simulation_params']['dt'],
        "max_time": tuning_specs['simulation_params']['max_time'],
        "target": src["target"],
        "num_inputs": case_study["num_inputs"],
        "input_channel": src["input_channel"],
        "output_channel": src["output_channel"],
        "trim_values": case_study["trim_values"],
        "trim_ics": case_study["trim_ics"],
        "min_ctrl": src["min_ctrl"],
        "max_ctrl": src["max_ctrl"],
        "system_description": case_study.get("system_description"),
        "python_code": case_study.get("python_code"),
        "max_cost_budget": coerce_float(max_cost_budget, 1.0),  # Maximum allowed cost in dollars
        "total_cost_consumed": 0.0,  # Total cost consumed so far
        "warm_start_config": warm_start_config,
        "system_aware": system_aware,
        "llm_agent_instance": llm_agent_instance,
        "working_function": case_study.get("working_function", "system_dynamics"),
        "input_name": src.get("input_name", ""),
        "controller_index": case_study.get("controller_index", 0),
        'coupled_channels': case_study['coupled_channels'],
        "initial_conditions": case_study.get("initial_conditions", None),
        'cost_function': case_study['cost_function'],
        'coupling_ratio': case_study['coupling_ratio'],
        "controller_name": case_study["controller_name"],
        "optimizer_type": optimizer_type.upper(),
        "signal_type": src["signal_type"],
    }

    return state


def run_ga_handler(
        case_study_file: str = "AircraftPitch.json",
        tuning_specs: Dict[str, Any] = None,
        llm_model: str = "openai/gpt-oss-20b",
        seed: int = 42,
        run_id: int = 1,
        max_attempts: int = 2,
        max_wall_clock: float = 3600.0,
        max_cost_budget: float = 1.0,
        prompt_variant: str = "elaborate",
        buffer_size: int = 3,
        control_objective: str = None,
        warm_start_config: Dict[str, Any] = None,
        system_aware: bool = True,
        case_study: Dict[str, Any] = None,
        llm_agent_instance: Any = None,
        optimizer_type: str = "PSO",
        simulator_config: Dict[str, Any] = None
):
    """
    Run GA handler workflow with fixed targets and adjustable weights.

    warm_start_config (optional)
    ----------------------------
    If provided, attempt 1 uses this config directly instead of calling the
    LLM.  Subsequent attempts call the LLM as normal.

    Supported keys (all optional – defaults are used for missing keys):
        weights           : dict  – initial fitness weights
        ga_population_size: int   – population size for the warm-start run
        ga_generations    : int   – generation count for the warm-start run
        param_ranges      : dict  – PID search bounds  {'PID': {'Kp':[], ...}}

    Example – lighter than the regular GA, same search bounds:
        warm_start_config = {
            "weights": {"mse": 1.0, "settling_time": 1.0,
                        "overshoot": 1.0, "control_effort": 1.0},
            "ga_population_size": 10,    # half of regular GA's 20
            "ga_generations":     25,    # 5 % of regular GA's 500
            "param_ranges": {"PID": {"Kp": [0.5, 50.0],
                                     "Ki": [0.005, 5.0],
                                     "Kd": [0.005, 10.0]}},
        }
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # configure_logging(verbose=False, log_file=f"logs/ga_run_{timestamp}.log")

    logger.info("=" * 80)
    logger.info("GA HANDLER - Tuning GA Configuration with Fixed Targets and Adjustable Weights")
    logger.info("=" * 80)

    logger.info(f"Loading case study from case_studies/json/{case_study_file}...")
    if case_study is None:
        case_study = load_case_study(case_study_file)
    if control_objective is not None:
        case_study['control_objective'] = control_objective
    logger.info(f"✓ Loaded case study: {case_study['system_name']}")

    # Extract fixed_targets and simulation_params from case study
    fixed_targets = coerce_metric_targets(case_study.get('fixed_targets'))
    simulation_params = coerce_simulation_params(case_study.get('simulation_params'))

    # Default initial weights if not provided
    default_weights = {
        'mse': 1.0,
        'settling_time': 1.0,
        'overshoot': 1.0,
        'control_effort': 1.0
    }

    # Build tuning_specs from case study + optional override
    if tuning_specs is None:
        tuning_specs = {
            'fixed_targets': fixed_targets,
            'weights': default_weights,
            'simulation_params': simulation_params
        }
    else:
        # If tuning_specs provided, merge with case study data
        # Case study takes precedence for fixed_targets and simulation_params
        tuning_specs['fixed_targets'] = fixed_targets
        tuning_specs['simulation_params'] = simulation_params

        # Use provided weights if available, otherwise use defaults
        if 'weights' not in tuning_specs:
            tuning_specs['weights'] = default_weights

    logger.info("Initializing GA handler state with FIXED TARGETS and adjustable weights...")
    state = initialize_ga_handler_state(
        case_study=case_study,
        tuning_specs=tuning_specs,
        llm_model=llm_model,
        seed=seed,
        run_id=run_id,
        max_attempts=max_attempts,
        max_wall_clock=max_wall_clock,
        max_cost_budget=max_cost_budget,
        prompt_variant=prompt_variant,
        buffer_size=buffer_size,
        warm_start_config=warm_start_config,
        system_aware=system_aware,
        llm_agent_instance=llm_agent_instance,
        optimizer_type=optimizer_type,
        simulator_config=simulator_config
    )

    logger.info(f"Fixed targets (from case study): {tuning_specs['fixed_targets']}")
    logger.info(f"Initial weights: {tuning_specs['weights']}")
    logger.info(f"Fixed sim params (from case study): {tuning_specs['simulation_params']}")

    logger.info("Creating GA handler workflow...")
    app = create_ga_handler_graph()

    recursion_limit = get_recursion_limit(max_attempts)
    logger.info("Starting GA configuration tuning workflow...")
    logger.info(f"LangGraph recursion_limit: {recursion_limit} (max_attempts={max_attempts})")
    final_state = app.invoke(state, {"recursion_limit": recursion_limit})

    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best Result (Attempt {final_state['best_result']['best_attempt']}):")
    logger.info(f"  Baseline Cost: {final_state['best_result']['best_baseline_cost']:.4f}")
    logger.info(f"  GA Cost: {final_state['best_result']['best_ga_cost']:.4f}")
    logger.info(f"  GA Config:")
    best_ga = final_state['best_result']['best_ga_config']
    logger.info(f"    Population: {best_ga['ga_population_size']}")
    logger.info(f"    Generations: {best_ga['ga_generations']}")
    logger.info(f"    Param ranges: {best_ga['param_ranges']}")
    logger.info(f"  Controller: {final_state['best_result']['controller_parameters']}")
    logger.info(f"  Achieved Metrics:")
    for metric, value in final_state['best_result']['achieved_metrics'].items():
        logger.info(f"    {metric}: {value:.4f}")

    # Log num_evaluations for the best attempt
    best_attempt = final_state['best_result']['best_attempt']
    if final_state['feedback_history'] and best_attempt <= len(final_state['feedback_history']):
        num_evaluations = final_state['feedback_history'][best_attempt - 1].get('num_evaluations', 0)
        logger.info(f"  PID Evaluations: {num_evaluations}")

    logger.info(f"Final Decision: {final_state['decision']['action']}")
    logger.info(f"Reason: {final_state['decision']['reason']}")
    logger.info(f"Total Attempts: {len(final_state['feedback_history'])}")

    total_evals = sum(entry.get('num_evaluations', 0) for entry in final_state['feedback_history'])
    logger.info(f"Total PID Evaluations: {total_evals}")

    # --- TRANSFORM TO UNIFIED SCHEMA ---
    best_res = final_state.get('best_result', {})
    best_ga_config = best_res.get('best_ga_config', {})
    final_tuning_specs = final_state.get('tuning_specs', {})

    # 1. Build evolution history data array
    history_data = []
    for attempt_record in final_state.get('feedback_history', []):
        att_num = attempt_record.get('attempt_num', 1)
        prog = attempt_record.get('progress', {})
        iters = prog.get('iteration', [])

        for i in range(len(iters)):
            def _safe_get(lst, idx):
                return lst[idx] if idx < len(lst) else None

            history_data.append([
                att_num,
                _safe_get(iters, i),
                _safe_get(prog.get('best_cost', []), i),
                _safe_get(prog.get('Kp', []), i),
                _safe_get(prog.get('Ki', []), i),
                _safe_get(prog.get('Kd', []), i),
                _safe_get(prog.get('mse', []), i),
                _safe_get(prog.get('settling_time', []), i),
                _safe_get(prog.get('overshoot', []), i),
                _safe_get(prog.get('control_effort', []), i)
            ])

    # 2. Build decision trail specifically for GA Agent
    decision_trail = []
    for attempt_record in final_state.get('feedback_history', []):
        att_num = attempt_record.get('attempt_num', 1)
        config = attempt_record.get('ga_config', {})
        specs = attempt_record.get('tuning_specs', {})
        eval_decision = attempt_record.get('decision', {'action': 'unknown', 'reason': 'missing logic'})

        decision_trail.append({
            "attempt": att_num,
            "design_phase": {
                "action": "refine_search" if att_num > 1 else "initialize",
                "reasoning": config.get('reasoning', 'No reasoning provided.'),
                "configured_ranges": config.get('param_ranges', {}).get('PID', {}),
                "configured_weights": specs.get('weights', {})
            },
            "evaluation_phase": {
                "action": eval_decision.get('action'),
                "reasoning": eval_decision.get('reason')
            }
        })

    # 3. Compile Unified Final State
    normalized_final_state = {
        "best_result": {
            "gains": best_res.get('controller_parameters', {}),
            "achieved_metrics": best_res.get('achieved_metrics', {}),
            "achieved_cost": best_res.get('best_baseline_cost', float('inf')),
            "success_score": best_res.get('best_score', 0)
        },
        "tuning_specs": {
            "target_metrics": final_tuning_specs.get('fixed_targets', {}),
            "weights": final_tuning_specs.get('weights', {}),
            "param_ranges": best_ga_config.get('param_ranges', {}).get('PID', {})
        },
        "evolution_history": {
            "columns": ["attempt", "iteration", "cost", "Kp", "Ki", "Kd", "mse", "settling_time", "overshoot",
                        "control_effort"],
            "data": history_data
        },
        # agentic_context is exclusive to GA workflow
        "agentic_context": {
            "optimizer_type": final_state.get('optimizer_type', 'GA'),
            "total_attempts": len(final_state.get('feedback_history', [])),
            "total_evaluations": sum(
                entry.get('num_evaluations', 0) for entry in final_state.get('feedback_history', [])),
            "elapsed_time_sec": final_state.get('total_elapsed_time', 0.0),
            "final_decision": final_state.get('decision', {}).get('action', 'unknown'),
            "decision_trail": decision_trail,
            # PASS RAW HISTORY: Allows LangGraph to safely restart with previously populated state lists
            "raw_feedback_history": final_state.get('feedback_history', [])
        }
    }

    return normalized_final_state


def run_ga_handler_mock(
        case_study_file: str = "DCMotor.json",
        max_attempts: int = 3,
        prompt_variant: str = "elaborate",
        initial_weights: Dict[str, float] = None,
        **kwargs
):
    """
    Run GA handler workflow in mock mode for testing.

    Args:
        case_study_file: Case study JSON file (contains fixed_targets and simulation_params)
        max_attempts: Maximum number of attempts
        prompt_variant: Prompt variant to use
        initial_weights: Optional initial weights (if None, uses defaults)
        **kwargs: Additional arguments passed to run_ga_handler
    """

    # Prepare tuning_specs with only weights (if provided)
    tuning_specs = None
    if initial_weights is not None:
        tuning_specs = {'weights': initial_weights}

    return run_ga_handler(
        case_study_file=case_study_file,
        tuning_specs=tuning_specs,
        llm_model="mock",
        seed=42,
        run_id=999,
        max_attempts=max_attempts,
        max_wall_clock=600.0,
        prompt_variant=prompt_variant,
        buffer_size=2,
        **kwargs
    )


if __name__ == "__main__":
    from src.logger import configure_logging

    # Configure logging for standalone execution
    configure_logging(verbose=False)

    # Test with initial weights override
    initial_weights = {
        'mse': 1.0,
        'settling_time': 1.0,
        'overshoot': 1.0,
        'control_effort': 1.0
    }

    fin_state = run_ga_handler(
        case_study_file="DCMotor.json",
        tuning_specs={'weights': initial_weights},  # Only override weights
        llm_model="mock",
        seed=42,
        run_id=1,
        max_attempts=3,
        max_wall_clock=5.5,
        max_cost_budget=0.021,
        prompt_variant="concise",
        buffer_size=3
    )