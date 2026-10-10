import logging
import time
import pygad
import numpy as np
import multiprocessing
from typing import Dict, Any, List, Optional

from backend_core.MuloDesigner.simulator import SystemSimulator
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.GaAgent.src.utils import coerce_float, coerce_metric_targets
from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback

logger = get_logger(__name__)


class GAFitnessEvaluator:
    """Picklable callable class to hold state for multiprocessing workers."""

    def __init__(self, weights: Dict[str, float], system_config: Dict[str, Any], equation: str):
        self.weights = weights
        self.system_config = system_config
        self.equation = equation

    def __call__(self, ga_instance, solution, solution_idx):
        Kp, Ki, Kd = solution

        if abs(Kp) < 1e-6 and abs(Ki) < 1e-6 and abs(Kd) < 1e-6:
            return -10000.0

        # Instantiate simulator strictly inside the worker process
        local_simulator = SystemSimulator(self.system_config, self.equation)
        result = local_simulator.evaluate_pid({'Kp': Kp, 'Ki': Ki, 'Kd': Kd})

        if not result['success']:
            return -10000.0

        m = result['metrics']
        cost = sum(
            self.weights.get(met, 1.0) * (m[met] ** 2)
            for met in ['mse', 'settling_time', 'overshoot', 'control_effort']
            if met in m
        )
        return -cost


class GAOptimizer:
    """Genetic Algorithm optimizer for controller tuning."""

    def __init__(self, system_config: Dict[str, Any], equation: str, config: Dict[str, Any]):
        self.system_config = system_config
        self.equation = equation
        self.population_size = config['population_size']
        self.generations = config['generations']
        self.seed = config.get('seed', None)

        self.experiment_start_time: Optional[float] = config.get('experiment_start_time', None)
        self.max_wall_clock: float = coerce_float(config.get('max_wall_clock'), float('inf'))
        self.nfe_offset: int = config.get('nfe_offset', 0)
        self.current_attempt: int = config.get('current_attempt', 1)

        self.num_evaluations: int = 0
        self.initial_conditions: Optional[List[List[float]]] = config.get('initial_conditions')
        self.controller_name: str = config.get('controller_name', 'Unknown Controller')

    # =========================================================================

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

    # =========================================================================
    # CLASS METHOD: Replaces the nested closure to allow multiprocessing pickling
    # =========================================================================
    def on_generation(self, ga_instance):
        now = time.time()
        elapsed = now - self.experiment_start_time
        gen_num = ga_instance.generations_completed

        # Mathematically compute NFE
        self.num_evaluations = gen_num * self.population_size
        cum_nfe = self.nfe_offset + self.num_evaluations

        solution, fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )

        best_params = {'Kp': solution[0], 'Ki': solution[1], 'Kd': solution[2]}

        # Local evaluation on the main thread for telemetry
        local_eval_sim = SystemSimulator(self.system_config, self.equation)
        best_eval = local_eval_sim.evaluate_pid(best_params)

        if best_eval['success']:
            m = best_eval['metrics']
            bc = self.compute_baseline_cost(m, self.fixed_targets)
            scr = self._compute_success_score(m, self.fixed_targets)

            self.progress['mse'].append(m.get('mse'))
            self.progress['settling_time'].append(m.get('settling_time'))
            self.progress['overshoot'].append(m.get('overshoot'))
            self.progress['control_effort'].append(m.get('control_effort'))
            self.progress['best_baseline_cost'].append(bc)
            self.progress['metrics_success'].append(True)
            self.progress['warnings'].append(best_eval.get('warnings', []))
            self.progress['success_score'].append(scr)

            if scr > self._running_best_score:
                self._running_best_score = scr
            self.progress['best_score_so_far'].append(self._running_best_score)
        else:
            bc = float('inf')
            scr = 0
            self.progress['mse'].append(np.nan)
            self.progress['settling_time'].append(np.nan)
            self.progress['overshoot'].append(np.nan)
            self.progress['control_effort'].append(np.nan)
            self.progress['best_baseline_cost'].append(np.nan)
            self.progress['metrics_success'].append(False)
            self.progress['warnings'].append(best_eval.get('warnings', []))

        # Mock the population means since we refuse to evaluate them sequentially
        self.progress['mean_baseline_cost'].append(np.nan)

        pop_fitness = ga_instance.last_generation_fitness
        self.progress['mean_cost'].append(-np.mean(pop_fitness))

        self.progress['iteration'].append(gen_num)
        self.progress['best_cost'].append(-fitness)
        self.progress['best_ga_cost'].append(-fitness)
        self.progress['mean_ga_cost'].append(-np.mean(pop_fitness))
        self.progress['best_params'].append(solution.tolist())
        self.progress['Kp'].append(solution[0])
        self.progress['Ki'].append(solution[1])
        self.progress['Kd'].append(solution[2])

        self.progress['cumulative_nfe'].append(cum_nfe)
        self.progress['cumulative_wall_time'].append(elapsed)

        if bc < self._running_best:
            self._running_best = bc
        self.progress['best_baseline_so_far'].append(self._running_best)

        if scr == 100 and self._target_hit['nfe'] is None:
            self._target_hit['nfe'] = cum_nfe
            self._target_hit['time'] = elapsed
            self._target_hit['generation'] = gen_num

        try:
            _cb = get_callback()
            if _cb is not None:
                _cb({
                    'event_type': 'generation',
                    'controller_name': self.controller_name,
                    'attempt': self.current_attempt,
                    'generation': gen_num,
                    'cumulative_nfe': cum_nfe,
                    'cumulative_wall_time': elapsed,
                    'best_baseline_so_far': self._running_best,
                    'best_baseline_cost': bc if bc != float('inf') else None,
                    'mse': self.progress['mse'][-1] if self.progress['mse'] else None,
                    'settling_time': self.progress['settling_time'][-1] if self.progress['settling_time'] else None,
                    'overshoot': self.progress['overshoot'][-1] if self.progress['overshoot'] else None,
                    'control_effort': self.progress['control_effort'][-1] if self.progress['control_effort'] else None,
                    'Kp': float(solution[0]),
                    'Ki': float(solution[1]),
                    'Kd': float(solution[2]),
                    'success_score': scr,
                    'best_score_so_far': self._running_best_score,
                    'param_ranges': {
                        'Kp': [self.param_ranges['Kp'][0], self.param_ranges['Kp'][1]],
                        'Ki': [self.param_ranges['Ki'][0], self.param_ranges['Ki'][1]],
                        'Kd': [self.param_ranges['Kd'][0], self.param_ranges['Kd'][1]],
                    },
                    'fixed_targets': self.fixed_targets,  # ADD THIS LINE
                    'weights': dict(self.weights),
                    'pop_size': self.population_size,
                    'num_gen': self.generations,
                })
        except Exception:
            pass

        if scr == 100:
            self._stop_reason = 'score_100'
            raise StopIteration

        if elapsed >= self.max_wall_clock:
            self._stop_reason = 'wall_clock'
            raise StopIteration

    # =========================================================================

    def optimize_pid(
            self,
            weights: Dict[str, float],
            param_ranges: Dict[str, List[float]],
            fixed_targets: Dict[str, float],
    ) -> Dict[str, Any]:

        self.weights = weights
        self.fixed_targets = coerce_metric_targets(fixed_targets)
        self.param_ranges = param_ranges
        self.num_evaluations = 0

        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()

        # Bind state to instance so the class method can access it
        self.progress = {
            'iteration': [], 'best_cost': [], 'mean_cost': [], 'best_params': [],
            'Kp': [], 'Ki': [], 'Kd': [], 'mse': [], 'settling_time': [], 'overshoot': [], 'control_effort': [],
            'metrics_success': [], 'warnings': [], 'best_baseline_cost': [], 'mean_baseline_cost': [],
            'cumulative_nfe': [], 'cumulative_wall_time': [], 'best_baseline_so_far': [],
            'best_ga_cost': [], 'mean_ga_cost': [], 'success_score': [], 'best_score_so_far': [],
        }

        self._running_best_score = 0
        self._running_best = float('inf')
        self._target_hit = {'nfe': None, 'time': None, 'generation': None}
        self._stop_reason = 'normal'

        print()
        print("objective")
        print(self.system_config["signal_type"])
        print()

        evaluator = GAFitnessEvaluator(
            weights=self.weights,
            system_config=self.system_config,
            equation=self.equation
        )

        ga_kwargs = dict(
            num_generations=self.generations,
            num_parents_mating=max(2, self.population_size // 5),
            fitness_func=evaluator,
            parallel_processing=["thread", max(1, multiprocessing.cpu_count() - 1)],
            sol_per_pop=self.population_size,
            num_genes=3,
            gene_space=[
                {'low': self.param_ranges['Kp'][0], 'high': self.param_ranges['Kp'][1]},
                {'low': self.param_ranges['Ki'][0], 'high': self.param_ranges['Ki'][1]},
                {'low': self.param_ranges['Kd'][0], 'high': self.param_ranges['Kd'][1]},
            ],
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_num_genes=1,
            on_generation=self.on_generation,  # Pass the class method
            suppress_warnings=True,
        )

        if self.seed is not None:
            ga_kwargs['random_seed'] = self.seed

            if self.initial_conditions and len(self.initial_conditions) > 0:
                initial_pop = []
                for ic in self.initial_conditions:
                    if len(initial_pop) < self.population_size:
                        initial_pop.append(ic)

                if len(initial_pop) < self.population_size:
                    rng = np.random.default_rng(self.seed)
                    rem = self.population_size - len(initial_pop)
                    for _ in range(rem):
                        kp = rng.uniform(self.param_ranges['Kp'][0], self.param_ranges['Kp'][1])
                        ki = rng.uniform(self.param_ranges['Ki'][0], self.param_ranges['Ki'][1])
                        kd = rng.uniform(self.param_ranges['Kd'][0], self.param_ranges['Kd'][1])
                        initial_pop.append([kp, ki, kd])

                ga_kwargs['initial_population'] = initial_pop
                ga_kwargs.pop('sol_per_pop', None)

        ga_instance = pygad.GA(**ga_kwargs)

        _pygad_logger = logging.getLogger('pygad.pygad')
        _prev_level = _pygad_logger.level
        _pygad_logger.setLevel(logging.CRITICAL)
        try:
            ga_instance.run()
        except StopIteration:
            logger.info(
                f"  GA halted early: {self._stop_reason} "
                f"({self.num_evaluations} evals in this run)"
            )
        finally:
            _pygad_logger.setLevel(_prev_level)

        solution, fitness, _ = ga_instance.best_solution()
        Kp_opt, Ki_opt, Kd_opt = solution

        params = {'Kp': Kp_opt, 'Ki': Ki_opt, 'Kd': Kd_opt}
        final_sim = SystemSimulator(self.system_config, self.equation)
        final_result = final_sim.evaluate_pid(params)

        return {
            'success': final_result['success'],
            'controller_parameters': params,
            'achieved_metrics': final_result.get('metrics', {}),
            'ga_cost': -fitness,
            'baseline_cost': (
                self.compute_baseline_cost(final_result['metrics'], self.fixed_targets)
                if final_result['success'] else float('inf')
            ),
            'warnings': final_result.get('warnings', []),
            'progress': self.progress,
            'num_evaluations': self.num_evaluations,
            'stop_reason': self._stop_reason,
            'target_hit_nfe': self._target_hit['nfe'],
            'target_hit_time': self._target_hit['time'],
            'target_hit_generation': self._target_hit['generation'],
        }