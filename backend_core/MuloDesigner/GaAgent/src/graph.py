import json

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated
import operator
import yaml
import time
import re
from langgraph.graph import StateGraph, END

from backend_core.MuloDesigner.GaAgent.src.ga_optimizer import GAOptimizer
from backend_core.MuloDesigner.PsoOptimizer.pso_optimizer import PSOOptimizer
from backend_core.MuloDesigner.GaAgent.src.utils import (
    METRIC_KEYS,
    coerce_float,
    coerce_metric_targets,
    format_markdown_ga_feedback_for_prompt,
)
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback

# Get logger for this module
logger = get_logger(__name__)


# ============================== STATE DEFINITION ==============================

class GAConfigState(TypedDict):
    """State for configuration tuning workflow"""

    # Optimizer type (GA or PSO)
    optimizer_type: str

    # Attempt tracking
    current_attempt: int
    max_attempts: int

    # Wall clock and timing
    max_wall_clock: float
    total_elapsed_time: float

    # Prompt variant
    prompt_variant: str  # "concise" or "elaborate"

    # Buffer size for feedback history
    buffer_size: int

    # History and results
    feedback_history: Annotated[List[Dict[str, Any]], operator.add]
    best_result: Optional[Dict[str, Any]]

    # Current iteration data
    ga_config: Optional[Dict[str, Any]]
    optimization_results: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, str]]

    # tuning_specs contains weights and fixed_targets
    tuning_specs: Dict[str, Any]  # {'weights': {...}, 'fixed_targets': {...}, 'simulation_params': {...}}

    # System configuration
    llm_model: str
    run_id: int
    seed: int
    temperature: float
    system_name: str
    num_states: int
    control_objective: str

    # Simulation parameters
    dt: float
    max_time: float
    target: float
    num_inputs: int
    input_channel: int
    output_channel: int
    trim_values: Optional[List[float]]
    trim_ics: Optional[List[float]]
    min_ctrl: float
    max_ctrl: float

    # System description
    system_description: Optional[str]
    python_code: Optional[str]

    # System awareness
    system_aware: bool

    # LLM configuration
    max_tokens: int

    # Resource tracking
    max_cost_budget: float
    total_cost_consumed: float
    llm_call_cost: float
    llm_tokens_in: int
    llm_tokens_out: int

    # Shared experiment timing
    experiment_start_time: float

    # Warm start
    warm_start_config: Optional[Dict[str, Any]]

    # System setup
    working_function: str
    input_name: str
    controller_index: int
    coupled_channels: List[int]
    initial_conditions: Optional[List[List[float]]]
    cost_function: str
    coupling_ratio: float
    controller_name: str
    llm_agent_instance: Any
    signal_type: str


# =================================== NODES ===================================

def design_ga_config_node(state: GAConfigState) -> Dict[str, Any]:
    """LLM designs optimizer configuration parameters and weights"""
    optimizer_type = state.get('optimizer_type', 'GA').upper()

    logger.info("=" * 80)
    logger.info(f"DESIGN {optimizer_type} CONFIG NODE - Attempt {state['current_attempt']}/{state['max_attempts']}")
    logger.info("=" * 80)

    # Calculate remaining resources
    total_cost_consumed = state.get('total_cost_consumed', 0.0)
    cost_remaining = state['max_cost_budget'] - total_cost_consumed
    cost_budget_pct = (cost_remaining / state['max_cost_budget']) * 100

    time_elapsed = state.get('total_elapsed_time', 0.0)
    time_remaining = state['max_wall_clock'] - time_elapsed
    time_budget_pct = (time_remaining / state['max_wall_clock']) * 100

    # Get cost from last attempt
    last_attempt_cost = 0.0
    last_attempt_time = 0.0
    if state['feedback_history']:
        last_entry = state['feedback_history'][-1]
        last_attempt_cost = last_entry.get('llm_call_cost', 0.0)
        last_attempt_time = last_entry.get('elapsed_time', 0.0)

    logger.info(f"Resource Status:")
    logger.info(
        f"  Cost Budget: {cost_budget_pct:.1f}% remaining "
        f"(${cost_remaining:.4f} / ${state['max_cost_budget']:.4f})"
    )
    if last_attempt_cost > 0:
        logger.info(f"  Last LLM call: ${last_attempt_cost:.6f}")
    logger.info(
        f"  Time Budget: {time_budget_pct:.1f}% remaining "
        f"({time_remaining:.1f}s / {state['max_wall_clock']:.1f}s)"
    )
    if last_attempt_time > 0:
        logger.info(f"  Last optimization run: {last_attempt_time:.2f}s")

    # Warm-start bypass
    warm_start_config = state.get('warm_start_config')
    if state['current_attempt'] == 1 and warm_start_config is not None:
        logger.info("WARM-START: skipping LLM call for attempt 1 – using predefined config.")

        current_weights = state['tuning_specs']['weights']

        ga_config = {
            'ga_population_size': warm_start_config.get('ga_population_size', 10),
            'ga_generations': warm_start_config.get('ga_generations', 20),
            'param_ranges': warm_start_config.get(
                'param_ranges',
                {'PID': {'Kp': [-50.0, 50.0], 'Ki': [-5.0, 5.0], 'Kd': [-10.0, 10.0]}}
            ),
            'reasoning': 'Warm-start: predefined configuration (no LLM call on attempt 1).',
        }

        new_weights = warm_start_config.get('weights', current_weights)
        updated_tuning_specs = state['tuning_specs'].copy()
        updated_tuning_specs['weights'] = new_weights

        logger.info(f"  Population size : {ga_config['ga_population_size']}")
        logger.info(f"  Generations     : {ga_config['ga_generations']}")
        logger.info(f"  Param ranges    : {ga_config['param_ranges']}")
        logger.info(f"  Weights         : {new_weights}")

        return {
            'ga_config': ga_config,
            'tuning_specs': updated_tuning_specs,
            'total_cost_consumed': total_cost_consumed,
            'llm_call_cost': 0.0,
            'llm_tokens_in': 0,
            'llm_tokens_out': 0,
        }

    # Load appropriate prompt
    prompt_variant = state.get('prompt_variant', 'elaborate')
    prompt_filename = f"ga_config_prompt_{prompt_variant}.yaml"
    prompt_path = f"backend_core/MuloDesigner/Agents/templates/{prompt_filename}"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = yaml.safe_load(f)

    system_prompt = prompt_template["system_prompt"]

    if optimizer_type == 'PSO':
        system_prompt += (
            "\n\nCRITICAL: You are currently configuring a Particle Swarm Optimization (PSO) algorithm, "
            "not a Genetic Algorithm. Please treat 'ga_population_size' as Swarm Size (particles) "
            "and 'ga_generations' as Iterations."
        )

    current_config = state.get('ga_config') or {
        'ga_population_size': 20,
        'ga_generations': 30,
        'param_ranges': {
            'PID': {
                'Kp': [-50.0, 50.0],
                'Ki': [-10.0, 10.0],
                'Kd': [-10.0, 10.0]
            }
        }
    }

    current_weights = state['tuning_specs']['weights']
    fixed_targets = state['tuning_specs']['fixed_targets']
    current_sim_params = state['tuning_specs']['simulation_params']

    feedback_section = format_markdown_ga_feedback_for_prompt(
        state['feedback_history'],
        state['max_attempts'],
        variant=prompt_variant,
        buffer_size=state.get('buffer_size', 3),
        cost_budget_pct=cost_budget_pct,
        time_budget_pct=time_budget_pct,
        cost_remaining=cost_remaining,
        time_remaining=time_remaining,
        max_cost_budget=state['max_cost_budget'],
        max_wall_clock=state['max_wall_clock']
    ) if state['current_attempt'] > 1 else (
        ("```json\n" + json.dumps({
            "attempt": "first",
            "instruction": f"Design reasonable initial {optimizer_type} configuration and weights based on system complexity.",
            "resource_constraints": {
                "cost_budget_pct_remaining": round(cost_budget_pct, 1),
                "cost_remaining_usd": round(cost_remaining, 4),
                "time_budget_pct_remaining": round(time_budget_pct, 1),
                "time_remaining_s": round(time_remaining, 1)
            },
            "strategy_recommendation": (
                f"Consider starting with smaller {optimizer_type} runs to explore "
                "weights efficiently. Once you identify promising weights, increase size for better convergence."
            )
        }, indent=2) + "\n```")
        if prompt_variant == "json" else
        f"This is the first attempt. Design reasonable initial {optimizer_type} configuration "
        f"and weights based on the system complexity and control objectives.\n\n"
        f"**RESOURCE CONSTRAINTS:**\n"
        f"- Cost Budget: {cost_budget_pct:.1f}% remaining (${cost_remaining:.4f})\n"
        f"- Time Budget: {time_budget_pct:.1f}% remaining ({time_remaining:.1f} seconds)\n\n"
        f"**STRATEGY RECOMMENDATION:**\n"
        f"Consider starting with smaller {optimizer_type} runs to explore weights efficiently.\n"
        f"Once you identify promising weights, you can increase size for better convergence."
    )

    system_aware: bool = state.get('system_aware', True)

    if system_aware:
        initial_template_key = "user_prompt_template_initial"
        followup_template_key = "user_prompt_template"
    else:
        initial_template_key = "user_prompt_template_initial_blind"
        followup_template_key = "user_prompt_template_blind"

    if state['current_attempt'] == 1:
        user_prompt = prompt_template[initial_template_key].format(
            **(
                dict(
                    system_name=state['system_name'],
                    control_objective=state['control_objective'],
                    system_description=state.get('system_description', 'N/A'),
                )
                if system_aware else {}
            ),
            fixed_target_mse=fixed_targets['mse'],
            fixed_target_settling_time=fixed_targets['settling_time'],
            fixed_target_overshoot=fixed_targets['overshoot'],
            fixed_target_control_effort=fixed_targets['control_effort'],
            dt=current_sim_params['dt'],
            max_time=current_sim_params['max_time'],
            current_attempt=state['current_attempt'],
            max_attempts=state['max_attempts'],
            feedback_section=feedback_section,
            cost_budget_pct=cost_budget_pct,
            cost_remaining=cost_remaining,
            time_budget_pct=time_budget_pct,
            time_remaining=time_remaining,
            last_attempt_cost=last_attempt_cost,
            last_attempt_time=last_attempt_time,
        )
    else:
        user_prompt = prompt_template[followup_template_key].format(
            **(
                dict(
                    system_name=state['system_name'],
                    control_objective=state['control_objective'],
                    system_description=state.get('system_description', 'N/A'),
                )
                if system_aware else {}
            ),
            fixed_target_mse=fixed_targets['mse'],
            fixed_target_settling_time=fixed_targets['settling_time'],
            fixed_target_overshoot=fixed_targets['overshoot'],
            fixed_target_control_effort=fixed_targets['control_effort'],
            dt=current_sim_params['dt'],
            max_time=current_sim_params['max_time'],
            weight_mse=current_weights['mse'],
            weight_settling=current_weights['settling_time'],
            weight_overshoot=current_weights['overshoot'],
            weight_control=current_weights['control_effort'],
            current_pop_size=current_config['ga_population_size'],
            current_generations=current_config['ga_generations'],
            current_kp_range=current_config['param_ranges']['PID']['Kp'],
            current_ki_range=current_config['param_ranges']['PID']['Ki'],
            current_kd_range=current_config['param_ranges']['PID']['Kd'],
            current_attempt=state['current_attempt'],
            max_attempts=state['max_attempts'],
            feedback_section=feedback_section,
            cost_budget_pct=cost_budget_pct,
            cost_remaining=cost_remaining,
            time_budget_pct=time_budget_pct,
            time_remaining=time_remaining,
            last_attempt_cost=last_attempt_cost,
            last_attempt_time=last_attempt_time,
        )

    agent = state.get('llm_agent_instance')

    if agent is None:
        from backend_core.MuloDesigner.Agents.mulo_design_agents import Agents
        agent = Agents(model_name=state['llm_model'], temperature=state['temperature'])
    else:
        agent.temperature = state['temperature']

    response_text, usage = agent.design_agent(system_prompt, user_prompt)
    try:
        llm_response = extract_json_from_response(response_text)
    except Exception as e:
        logger.warning(f"✗ JSON parse failed: {e}")
        llm_response = current_config
        llm_response['reasoning'] = 'Fallback to current config due to parse error'

    llm_tokens_in = usage.get('prompt_tokens', 0)
    llm_tokens_out = usage.get('completion_tokens', 0)
    llm_call_cost = agent.compute_cost(llm_tokens_in, llm_tokens_out)

    new_total_cost = total_cost_consumed + llm_call_cost

    logger.info(f"LLM call cost: ${llm_call_cost:.6f} (in: {llm_tokens_in}, out: {llm_tokens_out})")
    logger.info(f"Total cost consumed: ${new_total_cost:.6f} / ${state['max_cost_budget']:.6f}")
    logger.info(
        f"Remaining cost budget: {((state['max_cost_budget'] - new_total_cost) / state['max_cost_budget'] * 100):.1f}%")

    ga_config = {
        'ga_population_size': llm_response.get('ga_population_size', current_config['ga_population_size']),
        'ga_generations': llm_response.get('ga_generations', current_config['ga_generations']),
        'param_ranges': llm_response.get('param_ranges', current_config['param_ranges']),
        'reasoning': llm_response.get('reasoning', 'N/A')
    }

    new_weights = llm_response.get('weights', current_weights)
    new_sim_params = llm_response.get('simulation_params', current_sim_params)

    updated_tuning_specs = state['tuning_specs'].copy()
    updated_tuning_specs['weights'] = new_weights
    updated_tuning_specs['simulation_params'] = new_sim_params

    logger.info(f"✓ Designed {optimizer_type} config:")
    logger.info(f"  Population/Swarm size: {ga_config['ga_population_size']}")
    logger.info(f"  Generations/Iterations: {ga_config['ga_generations']}")
    logger.info(f"  Reasoning: {ga_config['reasoning']}")

    return {
        'ga_config': ga_config,
        'tuning_specs': updated_tuning_specs,
        'total_cost_consumed': new_total_cost,
        'llm_call_cost': llm_call_cost,
        'llm_tokens_in': llm_tokens_in,
        'llm_tokens_out': llm_tokens_out
    }


def optimize_with_ga_node(state: GAConfigState) -> Dict[str, Any]:
    """Optimize controller using GA or PSO with current configuration"""
    optimizer_type = state.get('optimizer_type', 'GA').upper()

    logger.info("=" * 80)
    logger.info(f"OPTIMIZE WITH {optimizer_type} NODE - Attempt {state['current_attempt']}")
    logger.info("=" * 80)

    start_time = time.time()

    ga_config = state['ga_config']
    tuning_specs = state['tuning_specs']
    weights = tuning_specs['weights']
    fixed_targets = coerce_metric_targets(tuning_specs.get('fixed_targets'))
    sim_params = tuning_specs['simulation_params']

    logger.info(f"Running {optimizer_type} with:")
    logger.info(f"  Population/Swarm: {ga_config['ga_population_size']}")
    logger.info(f"  Generations/Iterations: {ga_config['ga_generations']}")
    logger.info(f"  Weights: {weights}")
    logger.info(f"  Fixed targets (for baseline): {fixed_targets}")

    system_config = {
        'dt': sim_params['dt'],
        'max_time': sim_params['max_time'],
        'target': state['target'],
        'num_inputs': state['num_inputs'],
        'input_channel': state['input_channel'],
        'output_channel': state['output_channel'],
        'min_ctrl': state['min_ctrl'],
        'max_ctrl': state['max_ctrl'],
        'trim_values': state['trim_values'] or [0.0] * state['num_inputs'],
        'trim_ics': state.get('trim_ics', [0.0] * state.get('num_states', 2)),
        'num_states': state.get('num_states', 2),
        'working_function': state.get("working_function", "system_dynamics"),
        'input_name': state.get('input_name', ""),
        'controller_index': state.get('controller_index', 0),
        'coupled_channels': state['coupled_channels'],
        'cost_function': state['cost_function'],
        'coupling_ratio': state['coupling_ratio'],
        'signal_type': state['signal_type'],
    }

    nfe_offset = sum(
        entry.get('num_evaluations', 0)
        for entry in state.get('feedback_history', [])
    )

    optimizer_config = {
        'population_size': ga_config['ga_population_size'],
        'generations': ga_config['ga_generations'],
        'seed': state['seed'],
        'experiment_start_time': state.get('experiment_start_time'),
        'max_wall_clock': state.get('max_wall_clock', float('inf')),
        'nfe_offset': nfe_offset,
        'current_attempt': state['current_attempt'],
        # Change this line to fallback to an empty list if None
        'initial_conditions': state.get('initial_conditions') or [],
        'controller_name': state.get('controller_name', 'Unknown Controller')
    }

    if optimizer_type == 'PSO':
        optimizer_config.update({
            'param_ranges': ga_config['param_ranges']['PID'],
            'fixed_targets': fixed_targets,
            'weights': weights
        })
        optimizer = PSOOptimizer(system_config, state.get('python_code'), optimizer_config)
        optimization_results = optimizer.optimize_pid()
    else:
        optimizer = GAOptimizer(system_config, state.get('python_code'), optimizer_config)
        optimization_results = optimizer.optimize_pid(
            weights=weights,
            param_ranges=ga_config['param_ranges']['PID'],
            fixed_targets=fixed_targets
        )

    elapsed_time = time.time() - start_time
    optimization_results['elapsed_time'] = elapsed_time
    optimization_results['optimizer_cost_key'] = 'cost' if optimizer_type == 'PSO' else 'ga_cost'

    logger.info(f"✓ Optimization complete!")
    logger.info(f"  Success: {optimization_results['success']}")
    logger.info(f"  Controller: {optimization_results['controller_parameters']}")
    logger.info(
        f"  {optimizer_type} Cost: {optimization_results.get(optimization_results['optimizer_cost_key'], float('inf')):.4f}")
    logger.info(f"  Baseline Cost: {optimization_results['baseline_cost']:.4f}")
    logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
    logger.info(f"  Achieved metrics: {optimization_results['achieved_metrics']}")

    return {'optimization_results': optimization_results}


def evaluate_ga_node(state: GAConfigState) -> Dict[str, Any]:
    """Evaluate optimizer performance and decide next action"""
    optimizer_type = state.get('optimizer_type', 'GA').upper()

    logger.info("=" * 80)
    logger.info(f"EVALUATE {optimizer_type} NODE - Attempt {state['current_attempt']}")
    logger.info("=" * 80)

    opt_results = state['optimization_results']
    tuning_specs = state['tuning_specs']
    ga_config = state['ga_config']
    fixed_targets = coerce_metric_targets(tuning_specs.get('fixed_targets'))

    if opt_results['success']:
        baseline_cost = coerce_float(opt_results['baseline_cost'], float('inf'))
        ga_cost = coerce_float(opt_results.get('ga_cost', opt_results.get('cost', float('inf'))), float('inf'))
    else:
        baseline_cost = float('inf')
        ga_cost = float('inf')

    max_wall_clock = coerce_float(state.get('max_wall_clock'), float('inf'))
    max_cost_budget = coerce_float(state.get('max_cost_budget'), float('inf'))
    max_attempts = max(int(coerce_float(state.get('max_attempts'), 1)), 1)

    success_score = 0
    metric_comparisons = {}
    if opt_results['success']:
        achieved_metrics = opt_results.get('achieved_metrics', {})
        for metric in METRIC_KEYS:
            achieved_val = coerce_float(achieved_metrics.get(metric), float('inf'))
            target_val = fixed_targets[metric]
            passed = achieved_val <= target_val
            if passed:
                success_score += 25
            metric_comparisons[metric] = {
                'achieved': achieved_val,
                'target': target_val,
                'passed': bool(passed)
            }
    else:
        success_score = 0
        for metric in METRIC_KEYS:
            metric_comparisons[metric] = {
                'achieved': None,
                'target': fixed_targets[metric],
                'passed': False
            }

    logger.info(f"{optimizer_type} Cost: {ga_cost:.4f}")
    logger.info(f"Baseline Cost: {baseline_cost:.4f}")
    logger.info(f"Success Score: {success_score}/100")

    num_evaluations = opt_results.get('num_evaluations', 0)
    logger.info(f"PID Evaluations: {num_evaluations}")

    boundary_analysis = analyze_boundary_hitting(
        opt_results.get('controller_parameters', {}),
        ga_config['param_ranges']['PID']
    ) if opt_results.get('success') else {}

    current_score = success_score if opt_results['success'] else 0
    previous_best = state['best_result'] or {}
    previous_best_score = int(coerce_float(previous_best.get('best_score'), -1))
    previous_best_baseline = coerce_float(previous_best.get('best_baseline_cost'), float('inf'))

    if (state['best_result'] is None or
            current_score > previous_best_score or
            (current_score == previous_best_score and
             baseline_cost < previous_best_baseline)):
        best_result = {
            'best_attempt': state['current_attempt'],
            'best_baseline_cost': baseline_cost,
            'best_ga_cost': ga_cost,
            'best_score': current_score,
            'best_ga_config': ga_config,
            'controller_parameters': opt_results.get('controller_parameters', {}),
            'achieved_metrics': opt_results.get('achieved_metrics', {}),
            'best_target_hit_nfe': opt_results.get('target_hit_nfe'),
            'best_target_hit_time': opt_results.get('target_hit_time'),
        }
        logger.info(f"✓ New best result! Score: {current_score}/100, Baseline Cost: {baseline_cost:.4f}")
    else:
        best_result = state['best_result']
        logger.info(f"  Previous best remains (score: {best_result.get('best_score', 0)}/100, "
                    f"baseline cost: {best_result.get('best_baseline_cost', float('inf')):.4f})")

    current_total_time = state.get('total_elapsed_time', 0.0) + opt_results.get('elapsed_time', 0.0)
    current_total_cost = state.get('total_cost_consumed', 0.0)

    if opt_results['success'] and success_score == 100:
        decision = {'action': 'proceed', 'reason': 'Full success: all 4 metrics met or bettered (score=100/100)'}
        logger.info(f"✓ DECISION: PROCEED - {decision['reason']}")
    elif state['current_attempt'] >= max_attempts:
        decision = {'action': 'accept_suboptimal', 'reason': f'Max attempts reached ({max_attempts})'}
        logger.info(f"⚠ DECISION: ACCEPT SUBOPTIMAL - {decision['reason']}")
    elif current_total_time >= max_wall_clock:
        decision = {'action': 'accept_suboptimal',
                    'reason': f'Max wall clock reached ({current_total_time:.2f}s >= {max_wall_clock}s)'}
        logger.info(f"⏳ DECISION: ACCEPT SUBOPTIMAL - {decision['reason']}")
    elif current_total_cost >= max_cost_budget:
        decision = {'action': 'accept_suboptimal',
                    'reason': f'Max cost budget reached (${current_total_cost:.6f} >= ${max_cost_budget:.6f})'}
        logger.info(f"💰 DECISION: ACCEPT SUBOPTIMAL - {decision['reason']}")
    else:
        decision = {'action': 'retry',
                    'reason': f'Partial success (score={success_score}/100), retrying to achieve full success'}
        logger.info(f"↻ DECISION: RETRY - {decision['reason']}")

    feedback_entry = {
        'attempt_num': state['current_attempt'],
        'ga_config': ga_config,
        'tuning_specs': tuning_specs,
        'controller_gains': opt_results.get('controller_parameters', {}),
        'performance_metrics': opt_results.get('achieved_metrics', {}),
        'metric_comparisons': metric_comparisons,
        'success_score': success_score,
        'ga_cost': ga_cost,
        'baseline_cost': baseline_cost,
        'elapsed_time': opt_results.get('elapsed_time', 0.0),
        'boundary_analysis': boundary_analysis,
        'warnings': opt_results.get('warnings', []),
        'progress': opt_results.get('progress', {}),
        'llm_call_cost': state.get('llm_call_cost', 0.0),
        'llm_tokens_in': state.get('llm_tokens_in', 0),
        'llm_tokens_out': state.get('llm_tokens_out', 0),
        'num_evaluations': num_evaluations,
        'target_hit_nfe': opt_results.get('target_hit_nfe'),
        'target_hit_time': opt_results.get('target_hit_time'),
        'decision': decision
    }

    try:
        _cb = get_callback()
        if _cb is not None:
            _cb({
                'event_type': 'attempt_complete',
                'controller_name': state.get('controller_name', 'Unknown Controller'),  # ---> ADD THIS LINE
                'attempt': state['current_attempt'],
                'success_score': success_score,
                'baseline_cost': baseline_cost,
                'ga_cost': ga_cost,
                'achieved_metrics': opt_results.get('achieved_metrics', {}),
                'controller_gains': opt_results.get('controller_parameters', {}),
                'time_remaining_pct': max(0.0,
                                          (1.0 - current_total_time / max(max_wall_clock, 1e-9)) * 100),
                'cost_remaining_pct': max(0.0,
                                          (1.0 - current_total_cost / max(max_cost_budget, 1e-9)) * 100),
                'param_ranges': ga_config['param_ranges']['PID'],
                'weights': tuning_specs['weights'],
                'pop_size': ga_config['ga_population_size'],
                'num_gen': ga_config['ga_generations'],
                'decision': decision['action'],
            })
    except Exception:
        pass

    return {
        'feedback_history': [feedback_entry],
        'best_result': best_result,
        'decision': decision,
        'total_elapsed_time': current_total_time,
        'current_attempt': state['current_attempt'] + 1 if decision['action'] == 'retry' else state['current_attempt']
    }


# ============================ HELPER FUNCTIONS ================================

def analyze_boundary_hitting(params: Dict[str, float], ranges: Dict[str, List[float]]) -> Dict[str, Any]:
    """Analyze if parameters are hitting boundaries"""
    boundary_threshold = 0.05
    analysis = {}

    for param in ['Kp', 'Ki', 'Kd']:
        value = params.get(param, 0.0)
        param_range = ranges.get(param, [0, 0])
        range_span = param_range[1] - param_range[0]

        lower_dist = abs(value - param_range[0]) / range_span
        upper_dist = abs(param_range[1] - value) / range_span

        hitting_lower = lower_dist < boundary_threshold
        hitting_upper = upper_dist < boundary_threshold

        analysis[param] = {
            'value': value,
            'range': param_range,
            'hitting_lower': lower_dist,
            'hitting_upper': upper_dist,
            'boundary_issue': bool(hitting_lower) or bool(hitting_upper)
        }

    return analysis


# =============================== GRAPH DEFINITION ===============================

def should_continue(state: GAConfigState) -> str:
    """Conditional edge: decide whether to continue or end"""
    decision = state.get('decision', {})
    action = decision.get('action', 'end')

    if action != 'retry':
        return "end"

    current_attempt = state.get('current_attempt', 1)
    max_attempts = max(int(coerce_float(state.get('max_attempts'), 1)), 1)
    if current_attempt > max_attempts:
        logger.warning(
            "Stopping loop: current_attempt=%s exceeded max_attempts=%s",
            current_attempt,
            max_attempts,
        )
        return "end"

    return "design"


def create_ga_handler_graph():
    """Create and compile the agentic handler workflow"""
    workflow = StateGraph(GAConfigState)

    workflow.add_node("design", design_ga_config_node)
    workflow.add_node("optimize", optimize_with_ga_node)
    workflow.add_node("evaluate", evaluate_ga_node)

    workflow.set_entry_point("design")
    workflow.add_edge("design", "optimize")
    workflow.add_edge("optimize", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "design": "design",
            "end": END
        }
    )

    app = workflow.compile()
    return app


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    if not response_text:
        raise json.JSONDecodeError("Empty response", "", 0)
    # Remove thinking tags and their content
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    # Try to find JSON within markdown code blocks
    json_pattern = r'```json\s*\n?(.*?)\n?```'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # Try to find JSON without markdown formatting
        # Look for content that starts with { and ends with }
        brace_pattern = r'\{.*\}'
        brace_match = re.search(brace_pattern, response_text, re.DOTALL)
        if brace_match:
            json_text = brace_match.group(0).strip()
        else:
            # Last resort: use the entire cleaned response
            json_text = response_text.strip()
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # Log the problematic text for debugging
        logger.error(f"Failed to parse JSON. Original response: {repr(response_text)}")
        logger.error(f"Extracted JSON text: {repr(json_text)}")
        raise e
