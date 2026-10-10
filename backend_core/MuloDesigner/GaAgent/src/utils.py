import json
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

METRIC_KEYS = ('mse', 'settling_time', 'overshoot', 'control_effort')
METRIC_DEFAULTS = {
    'mse': 0.01,
    'settling_time': 5.0,
    'overshoot': 10.0,
    'control_effort': 0.5,
}


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert API/LLM values to float for numeric comparisons."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_metric_targets(data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Normalize performance target dict to float values."""
    source = data or {}
    return {
        key: coerce_float(source.get(key), METRIC_DEFAULTS[key])
        for key in METRIC_KEYS
    }


def coerce_simulation_params(data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Normalize simulation timing parameters to floats."""
    source = data or {}
    return {
        'dt': coerce_float(source.get('dt'), 0.01),
        'max_time': coerce_float(source.get('max_time'), 10.0),
    }


def gen_experiment_filename(
        system_name: str,
        llm_model: str,
        prompt_variant: str,
        buffer_size: int,
        run_id: int,
        include_timestamp: bool = True
) -> str:
    # Apply safe_name to all components
    safe_system = safe_name(system_name)
    safe_llm = safe_name(llm_model.split("/")[-1])  # Keep only last segment
    safe_prompt = safe_name(prompt_variant)

    # Format base filename
    filename = f"{safe_system}_R0{run_id}_{safe_llm}_{safe_prompt}_B{buffer_size}"

    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename += f"_{timestamp}"

    return f"{filename}.json"


def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s


def load_case_study(json_filename: str) -> Dict[str, Any]:
    """Load a case study JSON file from the case_studies directory."""
    base_dir = Path(__file__).parent.parent
    file_path = os.path.join(base_dir, "case_studies", "json", json_filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Case study file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Post-process: trim_values can be float or list; ensure it's a list
    if isinstance(data.get("trim_values"), (int, float)):
        data["trim_values"] = [float(data["trim_values"])]
    elif isinstance(data.get("trim_values"), list):
        data["trim_values"] = [float(v) for v in data["trim_values"]]
    else:
        data["trim_values"] = [0.0] * data.get("num_inputs", 1)

    # Post-process: trim_ics can be float or list; ensure it's a list
    if isinstance(data.get("trim_ics"), (int, float)):
        data["trim_ics"] = [float(data["trim_ics"])]
    elif isinstance(data.get("trim_ics"), list):
        data["trim_ics"] = [float(v) for v in data["trim_ics"]]
    else:
        # Default to zeros if not specified
        data["trim_ics"] = [0.0] * data.get("num_states", 2)

    data["fixed_targets"] = coerce_metric_targets(data.get("fixed_targets"))
    data["simulation_params"] = coerce_simulation_params(data.get("simulation_params"))

    return data


def format_markdown_ga_feedback_for_prompt(
        feedback_history: List[Dict[str, Any]],
        max_attempts: int,
        variant: str = "elaborate",
        buffer_size: int = 3,
        cost_budget_pct: float = 100.0,
        time_budget_pct: float = 100.0,
        cost_remaining: float = 1.0,
        time_remaining: float = 3600.0,
        max_cost_budget: float = 1.0,
        max_wall_clock: float = 3600.0
) -> str:
    """Format GA-specific feedback for LLM prompt with variant support"""
    if not feedback_history:
        return "No previous attempts."

    # Use buffer_size to limit history shown
    n_show = min(buffer_size, len(feedback_history))

    common_kwargs = dict(
        feedback_history=feedback_history,
        max_attempts=max_attempts,
        n_show=n_show,
        cost_budget_pct=cost_budget_pct,
        time_budget_pct=time_budget_pct,
        cost_remaining=cost_remaining,
        time_remaining=time_remaining,
        max_cost_budget=max_cost_budget,
        max_wall_clock=max_wall_clock,
    )

    if variant == "concise":
        return format_concise_feedback(
            feedback_history,
            cost_budget_pct,
            time_budget_pct,
            cost_remaining,
            time_remaining,
            max_cost_budget,
            max_wall_clock
        )
    else:
        # fallback to elaborate
        return format_elaborate_feedback(**common_kwargs)


def format_concise_feedback(
        feedback_history: List[Dict[str, Any]],
        cost_budget_pct: float = 100.0,
        time_budget_pct: float = 100.0,
        cost_remaining: float = 1.0,
        time_remaining: float = 3600.0,
        max_cost_budget: float = 1.0,
        max_wall_clock: float = 3600.0
) -> str:
    """Concise feedback format - minimal tables, last attempt only"""
    feedback_md = f"**LAST ATTEMPT:**\n"

    entry = feedback_history[-1]  # Only last attempt
    attempt_num = entry['attempt_num']
    ga_cfg = entry['ga_config']
    baseline_cost = entry.get('baseline_cost', float('inf'))
    ga_cost = entry.get('ga_cost', float('inf'))
    elapsed_time = entry.get('elapsed_time', 0.0)
    num_evaluations = entry.get('num_evaluations', 0)
    boundary = entry.get('boundary_analysis', {})
    controller_gains = entry.get('controller_gains', {})
    llm_call_cost = entry.get('llm_call_cost', 0.0)
    feedback_md += f"Attempt {attempt_num}: Baseline={baseline_cost:.4f}, GA={ga_cost:.4f}, Time={elapsed_time:.2f}s, Evals={num_evaluations}\n"
    feedback_md += f"GA: pop={ga_cfg['ga_population_size']}, gen={ga_cfg['ga_generations']}\n"
    feedback_md += f"LLM cost: ${llm_call_cost:.6f}\n\n"

    # Minimal parameter table
    feedback_md += "| Param | Value | Range |\n|-------|-------|-------|\n"
    param_ranges = ga_cfg['param_ranges']['PID']
    for param in ['Kp', 'Ki', 'Kd']:
        value = controller_gains.get(param, 0.0)
        param_range = param_ranges.get(param, [0, 0])
        boundary_info = boundary.get(param, {})
        flag = " ⚠️" if boundary_info.get('boundary_issue', False) else ""
        feedback_md += f"| {param}{flag} | {value:.4f} | [{param_range[0]:.2f}, {param_range[1]:.2f}] |\n"

    # Resource info
    feedback_md += f"\n**RESOURCES:**\n"
    feedback_md += f"- Cost: {cost_budget_pct:.0f}% (${cost_remaining:.6f} remaining), Last: ${llm_call_cost:.6f}\n"
    feedback_md += f"- Time: {time_budget_pct:.0f}% ({time_remaining:.1f}s remaining), Last: {elapsed_time:.2f}s\n"

    if llm_call_cost > 0:
        estimated_calls = max(1, int(cost_remaining / llm_call_cost))
        feedback_md += f"- Estimated calls remaining: ~{estimated_calls}\n"

    feedback_md += f"\n**FOR ATTEMPT {len(feedback_history) + 1}:** Adjust GA config and weights. "
    feedback_md += "If ⚠️ → expand range. Consider resources when sizing GA.\n"

    return feedback_md


def format_elaborate_feedback(
        feedback_history: List[Dict[str, Any]],
        max_attempts: int,
        n_show: int,
        cost_budget_pct: float = 100.0,
        time_budget_pct: float = 100.0,
        cost_remaining: float = 1.0,
        time_remaining: float = 3600.0,
        max_cost_budget: float = 1.0,
        max_wall_clock: float = 3600.0
) -> str:
    """Elaborate feedback format - full details with cost structure"""
    feedback_md = f"**PREVIOUS GA CONFIGURATION ATTEMPTS:**\nShowing last {n_show} attempt(s).\n\n"

    for entry in feedback_history[-n_show:]:
        attempt_num = entry['attempt_num']
        ga_cfg = entry['ga_config']
        baseline_cost = entry.get('baseline_cost', float('inf'))
        ga_cost = entry.get('ga_cost', float('inf'))
        elapsed_time = entry.get('elapsed_time', 0.0)
        num_evaluations = entry.get('num_evaluations', 0)
        boundary = entry.get('boundary_analysis', {})
        progress = entry.get('progress', {})
        warnings = entry.get('warnings', [])
        controller_gains = entry.get('controller_gains', {})
        performance_metrics = entry.get('performance_metrics', {})

        # GET PER-ATTEMPT RESOURCE CONSUMPTION
        llm_call_cost = entry.get('llm_call_cost', 0.0)
        llm_tokens_in = entry.get('llm_tokens_in', 0)
        llm_tokens_out = entry.get('llm_tokens_out', 0)

        feedback_md += f"### Attempt {attempt_num} Results:\n"
        feedback_md += f"**Baseline Cost:** {baseline_cost:.4f}\n"
        feedback_md += f"**GA Fitness Cost:** {ga_cost:.4f}\n"
        feedback_md += f"**GA Run Time:** {elapsed_time:.2f}s, Evaluations: {num_evaluations}\n"
        feedback_md += f"**LLM Call Cost:** ${llm_call_cost:.6f} (tokens: {llm_tokens_in} in, {llm_tokens_out} out)\n"
        feedback_md += (f"**Resource Consumption:** Cost={llm_call_cost / max_cost_budget * 100:.2f}% of budget, "
                        f"Time={elapsed_time / max_wall_clock * 100:.2f}% of budget\n\n")

        # Show Fixed Targets and Weights Used
        attempt_specs = entry.get('tuning_specs', {})
        if attempt_specs:
            attempt_targets = attempt_specs.get('fixed_targets', {})  # RENAMED
            attempt_weights = attempt_specs.get('weights', {})  # NEW
            attempt_sim = attempt_specs.get('simulation_params', {})

            feedback_md += "**Fixed Targets (for Baseline Cost):**\n"
            feedback_md += (f"- MSE: {attempt_targets.get('mse', 'N/A')}, "
                            f"Settling: {attempt_targets.get('settling_time', 'N/A')}s, "
                            f"Overshoot: {attempt_targets.get('overshoot', 'N/A')}, "
                            f"Control Effort: {attempt_targets.get('control_effort', 'N/A')}\n")

            feedback_md += "**Weights Used (for GA Fitness):**\n"
            feedback_md += (f"- MSE: {attempt_weights.get('mse', 'N/A')}, "
                            f"Settling: {attempt_weights.get('settling_time', 'N/A')}, "
                            f"Overshoot: {attempt_weights.get('overshoot', 'N/A')}, "
                            f"Control Effort: {attempt_weights.get('control_effort', 'N/A')}\n")

            feedback_md += f"**Sim Params:** Time: {attempt_sim.get('max_time', 'N/A')}s, dt: {attempt_sim.get('dt', 'N/A')}\n\n"

        feedback_md += "**Achieved Metrics:**\n"
        for metric, value in performance_metrics.items():
            feedback_md += f"- {metric}: {value:.4f}\n"
        feedback_md += "\n"

        # GA Configuration
        feedback_md += "#### GA Configuration Used:\n"
        feedback_md += f"- Population Size: {ga_cfg['ga_population_size']}\n"
        feedback_md += f"- Generations: {ga_cfg['ga_generations']}\n"
        feedback_md += f"- Kp Range: {ga_cfg['param_ranges']['PID']['Kp']}\n"
        feedback_md += f"- Ki Range: {ga_cfg['param_ranges']['PID']['Ki']}\n"
        feedback_md += f"- Kd Range: {ga_cfg['param_ranges']['PID']['Kd']}\n\n"

        # -- LLM Reasoning trace ---------------------------------------------
        reasoning = ga_cfg.get("reasoning", "").strip()
        if reasoning and reasoning not in ("N/A", "Warm-start: predefined configuration (no LLM call on attempt 1)."):
            feedback_md += "**LLM Reasoning (this attempt):**\n"
            # Indent each line so it renders as a block-quote-like section
            for line in reasoning.splitlines():
                feedback_md += f"> {line}\n"
            feedback_md += "\n"

        # MERGED TABLE: Obtained Parameters with Exploration Statistics
        if progress and all(k in progress for k in ['Kp', 'Ki', 'Kd']):
            feedback_md += "#### Obtained Parameters with Exploration Statistics\n"
            feedback_md += "⚠️ = parameter value within 5% of range boundary. "
            feedback_md += "Boundary hits require diagnosis: check whether the associated metric improved — "
            feedback_md += "if yes, the optimum may lie beyond the boundary (consider expanding); "
            feedback_md += "if no, the GA may be trapped in a local minimum at the boundary (consider restricting).\n"
            feedback_md += "| Parameter | Value | Range | Mean±Std | Min | Max |\n"
            feedback_md += "|-----------|-------|-------|----------|-----|-----|\n"

            param_ranges = ga_cfg['param_ranges']['PID']

            for param in ['Kp', 'Ki', 'Kd']:
                value = controller_gains.get(param, 0.0)
                param_range = param_ranges.get(param, [0, 0])

                boundary_info = boundary.get(param, {})
                boundary_flag = " ⚠️" if boundary_info.get('boundary_issue', False) else ""

                values = np.array(progress[param])
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)

                feedback_md += f"| {param}{boundary_flag} | {value:.4f} | "
                feedback_md += f"[{param_range[0]:.2f}, {param_range[1]:.2f}] | "
                feedback_md += f"{mean_val:.3f}±{std_val:.3f} | "
                feedback_md += f"{min_val:.3f} | {max_val:.3f} |\n"

            feedback_md += "\n"

            # Stats of Each Term in the Cost Function
            feedback_md += "#### Statistics of Each Objective Metric\n"
            feedback_md += "| Objective      | Value | Fixed Target | Mean±Std | Min | Max |\n"
            feedback_md += "|----------------|-------|--------------|----------|-----|-----|\n"

            metric_display = {
                'mse': 'MSE',
                'settling_time': 'Settling Time',
                'overshoot': 'Overshoot',
                'control_effort': 'Control Effort'
            }

            for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
                display_name = metric_display[metric]
                value = performance_metrics.get(metric, np.nan)
                target_val = attempt_targets.get(metric, np.nan)

                if metric in progress and len(progress[metric]) > 0:
                    values = np.array(progress[metric])
                    mean_v = np.nanmean(values)
                    std_v = np.nanstd(values)
                    min_v = np.nanmin(values)
                    max_v = np.nanmax(values)
                    mean_std_str = f"{mean_v:.3f}±{std_v:.3f}"
                    min_str = f"{min_v:.3f}"
                    max_str = f"{max_v:.3f}"
                else:
                    mean_std_str = min_str = max_str = "N/A"

                value_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
                target_str = f"{target_val:.4f}" if not np.isnan(target_val) else "N/A"

                feedback_md += f"| {display_name:<14} | {value_str:^7} | {target_str:^12} | {mean_std_str:^10} | {min_str:^5} | {max_str:^5} |\n"

            feedback_md += "\n"
        else:
            feedback_md += "#### Obtained Parameters vs Allowable Ranges\n"
            feedback_md += "| Parameter | Value | Range |\n"
            feedback_md += "|-----------|-------|-------|\n"

            param_ranges = ga_cfg['param_ranges']['PID']
            for param in ['Kp', 'Ki', 'Kd']:
                value = controller_gains.get(param, 0.0)
                param_range = param_ranges.get(param, [0, 0])
                boundary_info = boundary.get(param, {})
                boundary_flag = " ⚠️" if boundary_info.get('boundary_issue', False) else ""
                feedback_md += f"| {param}{boundary_flag} | {value:.4f} | "
                feedback_md += f"[{param_range[0]:.2f}, {param_range[1]:.2f}] |\n"

            feedback_md += "\n"

        if warnings:
            feedback_md += "#### Warnings:\n"
            for w in warnings:
                feedback_md += f"- {w}\n"
            feedback_md += "\n"

        feedback_md += "\n\n"

    # Cost trend analysis
    if len(feedback_history) >= 2:
        feedback_md += "**COST & TIMING TREND ANALYSIS:**\n"
        baseline_costs = [entry.get('baseline_cost', float('inf')) for entry in feedback_history]
        ga_costs = [entry.get('ga_cost', float('inf')) for entry in feedback_history]
        ga_times = [entry.get('elapsed_time', 0.0) for entry in feedback_history]
        llm_costs = [entry.get('llm_call_cost', 0.0) for entry in feedback_history]
        valid_baseline = [c for c in baseline_costs if c != float('inf')]

        if valid_baseline:
            best_baseline = min(valid_baseline)
            best_attempt = baseline_costs.index(best_baseline) + 1
            latest_baseline = baseline_costs[-1]
            latest_ga = ga_costs[-1]
            avg_ga_time = np.mean(ga_times) if ga_times else 0
            avg_llm_cost = np.mean(llm_costs) if llm_costs else 0
            total_llm_cost = sum(llm_costs)

            feedback_md += f"- Best baseline cost so far: {best_baseline:.4f} (Attempt {best_attempt})\n"
            feedback_md += f"- Latest baseline cost: {latest_baseline:.4f}, GA cost: {latest_ga:.4f} (Attempt {len(feedback_history)})\n"
            feedback_md += f"- Average GA run time: {avg_ga_time:.2f}s\n"
            feedback_md += f"- Average LLM call cost: ${avg_llm_cost:.6f}\n"
            feedback_md += f"- Total LLM cost consumed: ${total_llm_cost:.6f}\n"

            if len(valid_baseline) >= 2:
                if valid_baseline[-1] < valid_baseline[-2]:
                    feedback_md += f"- Trend: ✓ IMPROVING (baseline cost decreased by {valid_baseline[-2] - valid_baseline[-1]:.4f})\n"
                elif valid_baseline[-1] > valid_baseline[-2]:
                    feedback_md += f"- Trend: ✗ WORSENING (baseline cost increased by {valid_baseline[-1] - valid_baseline[-2]:.4f})\n"
                else:
                    feedback_md += f"- Trend: STAGNANT (no baseline cost change)\n"
        feedback_md += "\n"

    # Resource Status Section
    feedback_md += "**RESOURCE STATUS:**\n"
    feedback_md += f"- **Cost Budget:** {cost_budget_pct:.1f}% remaining (${cost_remaining:.6f} of ${max_cost_budget:.6f})\n"

    if feedback_history:
        last_llm_cost = feedback_history[-1].get('llm_call_cost', 0.0)
        last_ga_time = feedback_history[-1].get('elapsed_time', 0.0)
        if last_llm_cost > 0:
            estimated_calls_remaining = max(1, int(cost_remaining / last_llm_cost))
            feedback_md += f"  - Last LLM call: ${last_llm_cost:.6f} ({last_llm_cost / max_cost_budget * 100:.2f}% of total budget)\n"
            feedback_md += f"  - Estimated LLM calls remaining: ~{estimated_calls_remaining}\n"

    feedback_md += f"- **Time Budget:** {time_budget_pct:.1f}% remaining ({time_remaining:.1f}s of {max_wall_clock:.1f}s)\n"

    if feedback_history:
        if last_ga_time > 0:
            feedback_md += f"  - Last GA run: {last_ga_time:.2f}s ({last_ga_time / max_wall_clock * 100:.2f}% of total budget)\n"

    # Add resource-aware strategic guidance
    feedback_md += "\n**RESOURCE-AWARE STRATEGY:**\n"
    if cost_budget_pct < 30 or time_budget_pct < 30:
        feedback_md += "⚠️ **LOW RESOURCES:**\n"
        feedback_md += "  - Use smaller GA configurations (pop=10-20, gen=20-30) to conserve time\n"
        feedback_md += "  - Focus on refining weights rather than exploring many configurations\n"
        feedback_md += "  - Make decisive choices to maximize remaining attempts\n"
        feedback_md += "  - Prioritize: fewer attempts with good configs > many attempts with poor configs\n"
    elif cost_budget_pct > 70 and time_budget_pct > 70:
        feedback_md += "✓ **SUFFICIENT RESOURCES:**\n"
        feedback_md += "  - You can afford larger GA configurations for better convergence\n"
        feedback_md += "  - Consider thorough exploration of parameter spaces\n"
        feedback_md += "  - Early attempts: smaller configs to explore weights (pop=5-10, gen=10-15)\n"
        feedback_md += "  - Later attempts: larger configs for convergence (pop=10-20, gen=20-30)\n"
    else:
        feedback_md += "ℹ️ **MODERATE RESOURCES:**\n"
        feedback_md += "  - Balance GA configuration size with number of remaining attempts\n"
        feedback_md += "  - Use medium-sized configs (pop=10-20, gen=20-30)\n"
        feedback_md += "  - If weights are well-tuned, consider one larger final run\n"

    feedback_md += "\n"

    feedback_md += "**ANALYSIS INSTRUCTIONS:**\n"
    feedback_md += "1. **Cost Definitions:**\n"
    feedback_md += "   - GA Fitness = sum(weight_i * achieved_i^2) - what the optimizer minimizes\n"
    feedback_md += "   - Baseline Cost = sum((achieved_i - target_i) / target_i) - for comparison\n"
    feedback_md += "2. **Parameter Table** - When ⚠️ appears, diagnose before acting:\n"
    feedback_md += "   - Identify which objective metric that parameter most directly affects\n"
    feedback_md += "   - If that metric **improved** since last attempt → optimum may lie beyond boundary → expand range\n"
    feedback_md += "   - If that metric **stagnated or worsened** despite the parameter hitting the boundary → the GA is\n"
    feedback_md += "     likely trapped in a local minimum where high gain satisfies some objectives at the cost of others\n"
    feedback_md += "     → **restrict** the upper bound to force exploration of lower-gain regions\n"
    feedback_md += "   - Do not expand a range that has already been expanded once without metric improvement to show for it\n"
    feedback_md += "3. **Cross-Attempt Boundary Stagnation** - For each parameter that has shown ⚠️ across multiple\n"
    feedback_md += "   consecutive attempts, check the trend of its associated metric:\n"
    feedback_md += "   - If a metric has not improved across 2+ consecutive attempts while its parameter hits the upper\n"
    feedback_md += "     boundary → this is a local-minimum signature, not a tight-range signature\n"
    feedback_md += "   - Weight increases alone will not escape a local minimum — range restriction is the correct lever\n"
    feedback_md += "   - **Ceiling-Chasing Reversal:** If the same parameter has been\n"
    feedback_md += "     restricted ≥5 times consecutively and still hits the upper bound each time, the restriction strategy is\n"
    feedback_md += "     exhausted — the assumption that a lower gain resolves the stagnation has been falsified. Switch to range\n"
    feedback_md += "     expansion (100–200% of the current upper bound). Always cross-check with the baseline cost trend: if\n"
    feedback_md += "     baseline cost is worsening during restriction cycles, this is a strong confirmation that expansion is the\n"
    feedback_md += "     correct next step. Be physically reasonable: high gains amplify control effort — verify the control_effort\n"
    feedback_md += "     metric remains within its fixed target after expansion.\n"
    feedback_md += "4. **Exploration Statistics** - Review Mean±Std, Min, Max to understand GA search behavior:\n"
    feedback_md += "   - Tight std (low spread) near a boundary → population has collapsed into a local minimum\n"
    feedback_md += "   - Wide std away from boundaries → healthy exploration, convergence not yet reached\n"
    feedback_md += "5. **Baseline Cost Trajectories** - Analyze convergence patterns across attempts\n"
    feedback_md += "6. **Weights** - Adjust weights to emphasize objectives not meeting fixed targets.\n"
    feedback_md += "   Note: if a metric has failed its target across 2+ attempts despite weight increases,\n"
    feedback_md += "   the problem is likely structural (local minimum) rather than motivational — adjust\n"
    feedback_md += "   the corresponding parameter range in addition to or instead of increasing its weight\n\n"

    current_attempt = len(feedback_history) + 1
    feedback_md += f"**FOR ATTEMPT {current_attempt}/{max_attempts}:**\n"
    feedback_md += "Adjust GA configuration AND weights based on the analysis above. "
    feedback_md += "**YOU MUST MODIFY BOTH THE GA CONFIGURATION AND WEIGHTS FOR THIS ATTEMPT.**\n"
    feedback_md += "**Consider resource constraints when choosing population_size and generations.**\n"

    return feedback_md

