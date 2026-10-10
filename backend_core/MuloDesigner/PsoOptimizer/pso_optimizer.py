import time
import sys
import numpy as np
import concurrent.futures
import threading
import multiprocessing
import logging
from typing import Dict, Any, List, Optional
from scipy.stats import qmc

if multiprocessing.current_process().name != 'MainProcess':
    logging.getLogger('streamlit').setLevel(logging.ERROR)

from backend_core.MuloDesigner.simulator import SystemSimulator
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback

logger = get_logger(__name__)


# ==============================================================================
# TOP-LEVEL PARALLEL EVALUATION FUNCTION
# ==============================================================================
def _parallel_evaluate_fitness(Kp: float, Ki: float, Kd: float, weights: Dict[str, float],
                               system_config: Dict[str, Any], equation: str) -> tuple:
    """Creates a local simulator in the worker process and evaluates fitness."""
    if abs(Kp) < 1e-6 and abs(Ki) < 1e-6 and abs(Kd) < 1e-6:
        return float('inf'), {'success': False, 'metrics': {}, 'warnings': ['Trivial all-zero gains rejected']}

    local_simulator = SystemSimulator(system_config, equation)
    result = local_simulator.evaluate_pid({'Kp': Kp, 'Ki': Ki, 'Kd': Kd})

    if not result['success']:
        return float('inf'), result

    m = result['metrics']
    cost = sum(
        weights.get(met, 1.0) * (m[met] ** 2)
        for met in ['mse', 'settling_time', 'overshoot', 'control_effort'] if met in m
    )
    return cost, result


class PSOOptimizer:
    """Modular Particle Swarm Optimizer for controller tuning."""

    def __init__(self, system_config: Dict[str, Any], equation: str, config: Dict[str, Any]):
        self.system_config = system_config
        self.equation = equation
        self.seed = config.get('seed', 42)

        self.experiment_start_time: Optional[float] = config.get('experiment_start_time', time.time())
        self.max_wall_clock: float = float(config.get('max_wall_clock', float('inf')))
        self.nfe_offset: int = config.get('nfe_offset', 0)
        self.current_attempt: int = config.get('current_attempt', 1)
        self.initial_conditions: List[float] = config.get('initial_conditions', [])
        self.param_ranges: Dict[str, float] = config['param_ranges']
        self.fixed_targets: Dict[str, float] = config['fixed_targets']
        self.population_size: int = int(config['population_size'])
        self.generations: int = int(config['generations'])
        self.controller_name: str = config.get('controller_name', 'Unknown Controller')
        self.weights: Dict[str, float] = config['weights']

        self.num_evaluations: int = 0
        self.dim: int = 3

    # ==============================================================================
    # COST & SCORING HELPERS
    # ==============================================================================
    def compute_baseline_cost(self, metrics: Dict[str, float], fixed_targets: Dict[str, float]) -> float:
        cost = 0.0
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if metric in metrics and metric in fixed_targets:
                achieved = metrics[metric]
                target = fixed_targets[metric]
                cost += (achieved - target) / max(target, 1e-6)
        return cost

    def _compute_success_score(self, metrics: Dict[str, float], fixed_targets: Dict[str, float]) -> int:
        score = 0
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if metric in metrics and metric in fixed_targets:
                if metrics[metric] <= fixed_targets[metric]:
                    score += 25
        return score

    # ==============================================================================
    # SWARM KINEMATICS & SEARCH SPACE HELPERS
    # ==============================================================================
    def _get_bounds(self) -> np.ndarray:
        """Returns bounds array for initialization and clipping."""
        return np.array([self.param_ranges['Kp'], self.param_ranges['Ki'], self.param_ranges['Kd']])

    def _initialize_swarm(self, bounds: np.ndarray) -> tuple:
        """Initializes positions via Sobol sequence and assigns random velocities."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        sample = sampler.random_base2(m=int(np.ceil(np.log2(self.population_size))))
        sample = sample[:self.population_size]

        positions = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        if len(self.initial_conditions) > 0:
            init_conds = np.array(self.initial_conditions)
            num_init = min(len(init_conds), self.population_size)
            positions[:num_init] = init_conds[:num_init]

        return positions, velocities

    def _update_particles(self, positions: np.ndarray, velocities: np.ndarray, pbest: np.ndarray,
                          gbest: np.ndarray, bounds: np.ndarray, w: float = 0.7, c1: float = 1.5,
                          c2: float = 1.5) -> tuple:
        """Applies PSO kinematic updates and boundary clipping."""
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        velocities = (w * velocities +
                      c1 * r1 * (pbest - positions) +
                      c2 * r2 * (gbest - positions))
        positions = positions + velocities

        for d in range(self.dim):
            positions[:, d] = np.clip(positions[:, d], bounds[d, 0], bounds[d, 1])

        return positions, velocities

    def _get_thread_context(self):
        """Resolves Streamlit script runner context for cross-thread UI updates."""
        if len(sys.argv) > 0 and "streamlit" in sys.argv[0].lower():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                return get_script_run_ctx()
            except ImportError:
                pass
        return None

    # ==============================================================================
    # EVALUATION & CALLBACK HOOKS (POLYMORPHIC EXTENSION POINTS)
    # ==============================================================================
    def _submit_evaluation_future(self, executor: concurrent.futures.ProcessPoolExecutor, position: np.ndarray):
        """Dispatches an evaluation task to the process pool."""
        return executor.submit(
            _parallel_evaluate_fitness,
            position[0], position[1], position[2],
            self.weights, self.system_config, self.equation
        )

    def _evaluate_generation(self, executor: concurrent.futures.ProcessPoolExecutor,
                             positions: np.ndarray, pbest_positions: np.ndarray,
                             pbest_costs: np.ndarray, pbest_results: list,
                             gbest_position: np.ndarray, gbest_cost: float, gbest_result: Any) -> tuple:
        """Executes parallel evaluation for the swarm and updates pbest/gbest."""
        gen_costs = [float('inf')] * self.population_size
        gen_baseline_costs = [np.nan] * self.population_size
        gen_results = [None] * self.population_size

        futures = {self._submit_evaluation_future(executor, positions[i]): i for i in range(self.population_size)}

        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            cost, result = future.result()
            self.num_evaluations += 1

            gen_costs[i] = cost
            gen_results[i] = result

            if result.get('success', False):
                bc = self.compute_baseline_cost(result['metrics'], self.fixed_targets)
                gen_baseline_costs[i] = bc
            else:
                gen_baseline_costs[i] = np.nan

            if cost < pbest_costs[i]:
                pbest_costs[i] = cost
                pbest_positions[i] = np.copy(positions[i])
                pbest_results[i] = result

                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = np.copy(positions[i])
                    gbest_result = result

        return gen_costs, gen_baseline_costs, gen_results, gbest_position, gbest_cost, gbest_result

    def _append_generation_progress(self, progress: Dict[str, list], gen: int, gbest_cost: float,
                                    gen_costs: list, gbest_position: np.ndarray, gen_best_pos: np.ndarray,
                                    gen_best_m: Any, metrics_success: bool, warnings: list, bc: float,
                                    gen_baseline_costs: list, cum_nfe: int, elapsed: float,
                                    running_best_baseline: float, scr: float, running_best_score: float):
        """Records iteration telemetry. Robustly handles both single dicts and lists of dicts."""
        progress['iteration'].append(gen)
        progress['best_cost'].append(gbest_cost)
        progress['mean_cost'].append(np.mean([c for c in gen_costs if c != float('inf')]))
        progress['best_params'].append(gbest_position.tolist())
        progress['Kp'].append(gen_best_pos[0])
        progress['Ki'].append(gen_best_pos[1])
        progress['Kd'].append(gen_best_pos[2])

        # Safely parse metrics whether they are a single dict (PSO) or a list of dicts (MIMO PSO)
        for key in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if isinstance(gen_best_m, list):
                vals = [m.get(key, np.nan) for m in gen_best_m if isinstance(m, dict)]
                progress[key].append(np.nanmean(vals) if vals else np.nan)
            elif isinstance(gen_best_m, dict):
                progress[key].append(gen_best_m.get(key, np.nan))
            else:
                progress[key].append(np.nan)

        progress['metrics_success'].append(metrics_success)
        progress['warnings'].append(warnings)
        progress['best_baseline_cost'].append(bc)
        progress['mean_baseline_cost'].append(np.nanmean(gen_baseline_costs) if gen_baseline_costs else np.nan)
        progress['cumulative_nfe'].append(cum_nfe)
        progress['cumulative_wall_time'].append(elapsed)
        progress['best_baseline_so_far'].append(running_best_baseline)
        progress['success_score'].append(scr)
        progress['best_score_so_far'].append(running_best_score)

    def _dispatch_generation_callback(self, gen: int, cum_nfe: int, elapsed: float,
                                      running_best_baseline: float, bc: float, scr: float,
                                      running_best_score: float, gen_best_m: Any, gen_best_pos: np.ndarray,
                                      gen_best_baseline_cost: float):
        """Constructs and pushes real-time telemetry to callback listeners."""
        try:
            _cb = get_callback()
            if _cb is not None:
                # Fallback to safely pluck the first channel's metrics if called during MIMO execution
                m = gen_best_m[0] if isinstance(gen_best_m, list) and len(gen_best_m) > 0 else gen_best_m
                if not isinstance(m, dict):
                    m = {}

                _cb({
                    'event_type': 'generation',
                    'controller_name': self.controller_name,
                    'attempt': self.current_attempt,
                    'generation': gen,
                    'cumulative_nfe': cum_nfe,
                    'cumulative_wall_time': elapsed,
                    'best_baseline_so_far': running_best_baseline,
                    'best_baseline_cost': gen_best_baseline_cost if np.isfinite(gen_best_baseline_cost) else None,
                    'success_score': scr,
                    'best_score_so_far': running_best_score,
                    'mse': m.get('mse', np.nan),
                    'settling_time': m.get('settling_time', np.nan),
                    'overshoot': m.get('overshoot', np.nan),
                    'control_effort': m.get('control_effort', np.nan),
                    'Kp': float(gen_best_pos[0]),
                    'Ki': float(gen_best_pos[1]),
                    'Kd': float(gen_best_pos[2]),
                    'param_ranges': self.param_ranges,
                    'fixed_targets': self.fixed_targets,  # ADD THIS LINE
                    'weights': self.weights,
                    'pop_size': self.population_size,
                    'num_gen': self.generations,
                })
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def _extract_final_result(self, gbest_position: np.ndarray, gbest_result: Any, gbest_cost: float,
                              progress: dict, stop_reason: str, running_best_score: float,
                              target_hit: dict) -> Dict[str, Any]:
        """Formats the final return payload."""
        final_params = {'Kp': float(gbest_position[0]), 'Ki': float(gbest_position[1]), 'Kd': float(gbest_position[2])}
        logger.info(f"✓ Optimization complete! Controller: {final_params}")

        return {
            'success': gbest_result['success'] if gbest_result else False,
            'controller_parameters': final_params,
            'achieved_metrics': gbest_result['metrics'] if gbest_result and gbest_result['success'] else {},
            'cost': gbest_cost,
            'baseline_cost': self.compute_baseline_cost(gbest_result['metrics'], self.fixed_targets) if gbest_result and
                                                                                                        gbest_result[
                                                                                                            'success'] else float(
                'inf'),
            'warnings': gbest_result.get('warnings', []) if gbest_result else [],
            'progress': progress,
            'num_evaluations': self.num_evaluations,
            'stop_reason': stop_reason,
            'final_score': running_best_score,
            'target_hit_nfe': target_hit['nfe'],
            'target_hit_time': target_hit['time'],
            'target_hit_generation': target_hit['generation'],
        }

    # ==============================================================================
    # MAIN OPTIMIZATION LOOP
    # ==============================================================================
    def optimize_pid(self) -> Dict[str, Any]:
        logger.info(f"Running PSO | Swarm: {self.population_size} | Gens: {self.generations} | Weights: {self.weights}")
        np.random.seed(self.seed)
        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()

        bounds = self._get_bounds()
        positions, velocities = self._initialize_swarm(bounds)

        pbest_positions = np.copy(positions)
        pbest_costs = np.full(self.population_size, float('inf'))
        pbest_results = [None] * self.population_size

        gbest_position = np.copy(positions[0])
        gbest_cost = float('inf')
        gbest_result = None

        progress = {k: [] for k in ['iteration', 'best_cost', 'mean_cost', 'best_params', 'Kp', 'Ki', 'Kd',
                                    'mse', 'settling_time', 'overshoot', 'control_effort', 'metrics_success',
                                    'warnings', 'best_baseline_cost', 'mean_baseline_cost', 'cumulative_nfe',
                                    'cumulative_wall_time', 'best_baseline_so_far', 'success_score',
                                    'best_score_so_far']}

        _running_best_baseline = float('inf')
        _running_best_score = 0
        _target_hit = {'nfe': None, 'time': None, 'generation': None}
        _stop_reason = 'normal'

        ctx = self._get_thread_context()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            if ctx is not None:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                for thread in threading.enumerate():
                    add_script_run_ctx(thread, ctx)

            for gen in range(1, self.generations + 1):
                elapsed = time.time() - self.experiment_start_time

                # 1. Evaluate Current Generation
                (gen_costs, gen_baseline_costs, gen_results,
                 gbest_position, gbest_cost, gbest_result) = self._evaluate_generation(
                    executor, positions, pbest_positions, pbest_costs, pbest_results,
                    gbest_position, gbest_cost, gbest_result
                )

                # 2. Extract Generation Best
                valid_gen_costs = [c if r and r['success'] else float('inf') for c, r in zip(gen_costs, gen_results)]
                if min(valid_gen_costs) != float('inf'):
                    gen_best_idx = np.argmin(valid_gen_costs)
                    gen_best_pos = positions[gen_best_idx]
                    gen_best_m = gen_results[gen_best_idx]['metrics']
                    gen_best_baseline_cost = self.compute_baseline_cost(
                        gen_results[gen_best_idx]['metrics'], self.fixed_targets
                    )
                else:
                    gen_best_pos = gbest_position
                    gen_best_m = gbest_result['metrics'] if gbest_result else {'mse': np.nan, 'settling_time': np.nan,
                                                                               'overshoot': np.nan,
                                                                               'control_effort': np.nan}
                    gen_best_baseline_cost = float('nan')

                cum_nfe = self.nfe_offset + self.num_evaluations

                # 3. Score Global Best
                if gbest_result and gbest_result['success']:
                    bc = self.compute_baseline_cost(gbest_result['metrics'], self.fixed_targets)
                    scr = self._compute_success_score(gbest_result['metrics'], self.fixed_targets)
                    metrics_success = True
                    warnings = gbest_result.get('warnings', [])
                else:
                    bc = float('inf')
                    scr = 0
                    metrics_success = False
                    warnings = []

                _running_best_baseline = min(bc, _running_best_baseline)
                _running_best_score = max(scr, _running_best_score)

                logger.debug(f"Gen {gen}: best_bl={bc:.4f} score={scr}/100 NFE={cum_nfe} wall={elapsed:.2f}s")

                # 4. Telemetry Recording
                self._append_generation_progress(
                    progress, gen, gbest_cost, gen_costs, gbest_position, gen_best_pos,
                    gen_best_m, metrics_success, warnings, bc, gen_baseline_costs,
                    cum_nfe, elapsed, _running_best_baseline, scr, _running_best_score
                )

                if scr == 100 and _target_hit['nfe'] is None:
                    _target_hit.update({'nfe': cum_nfe, 'time': elapsed, 'generation': gen})

                # 5. UI Event Dispatch
                self._dispatch_generation_callback(
                    gen, cum_nfe, elapsed, _running_best_baseline, bc,
                    scr, _running_best_score, gen_best_m, gen_best_pos,
                    gen_best_baseline_cost
                )

                # 6. Check Termination
                if scr == 100:
                    _stop_reason = 'score_100'
                    break
                if elapsed >= self.max_wall_clock:
                    _stop_reason = 'wall_clock'
                    break

                # 7. Update Swarm Kinematics
                positions, velocities = self._update_particles(positions, velocities, pbest_positions, gbest_position,
                                                               bounds)

        return self._extract_final_result(
            gbest_position, gbest_result, gbest_cost, progress, _stop_reason,
            _running_best_score, _target_hit
        )