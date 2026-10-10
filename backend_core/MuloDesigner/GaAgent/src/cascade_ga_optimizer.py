import pygad
import numpy as np
from typing import Dict, Any, List

from backend_core.MuloDesigner.GaAgent.src.ga_optimizer import GAOptimizer
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger

logger = get_logger(__name__)


class _HighLevelStop(Exception):
    """Sentinel raised by on_generation_high_level to halt the HL-GA."""

class CascadeGAOptimizer:
    """
    Two-level cascade GA optimizer with caching
    """

    def __init__(
            self,
            simulator,
            low_level_config: Dict[str, Any],
            high_level_config: Dict[str, Any]
    ):
        self.simulator = simulator
        self.low_level_config = low_level_config
        self.high_level_config = high_level_config

        self.weight_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self.total_low_level_evaluations = 0
        self.num_high_level_evaluations = 0

        self._ll_running_best: float = float('inf')
        self.ll_history_nfe: list = []
        self.ll_history_best_baseline_so_far: list = []

        self._ll_running_best_score: int = 0  # â† NEW
        self.ll_history_success_score: list = []  # â† NEW  running-max series

        self.target_hit_nfe = None
        self.target_hit_time = None

        self.progress = {
            'iteration': [],
            'best_baseline_cost': [],
            'mean_baseline_cost': [],
            'best_weights': [],
            'weight_mse': [],
            'weight_settling_time': [],
            'weight_overshoot': [],
            'weight_control_effort': [],
            'range_Kp': [],
            'range_Ki': [],
            'range_Kd': [],
            'low_level_evals_cumulative': [],
            'high_level_evals_cumulative': [],
            'best_controller': [],
            'Kp': [],
            'Ki': [],
            'Kd': [],
            'mse': [],
            'settling_time': [],
            'overshoot': [],
            'control_effort': [],
            'ga_cost': [],
            'success': [],
            'success_score': [],  # per-HL-gen: best solution's score this gen
            'best_score_so_far': [],  # per-HL-gen: running max of success_score
            'll_success_score': [],  # dense (per-LL-gen): running max â€” written
        }

    def _compute_success_score(
            self,
            metrics: Dict[str, float],
            fixed_targets: Dict[str, float],
    ) -> int:
        """25 points per metric that meets or beats its fixed target (max 100)."""  # â† NEW
        score = 0  # â† NEW
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:  # â† NEW
            if metric in metrics and metric in fixed_targets:  # â† NEW
                if metrics[metric] <= fixed_targets[metric]:  # â† NEW
                    score += 25  # â† NEW
        return score

    def _evaluate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate a weight configuration using low-level GA with caching.
        Args:
            weights: Weight configuration
        Returns:
            Low-level GA optimization result
        """
        # Create cache key (round to avoid floating point issues)
        cache_key = tuple(round(weights[k], 6) for k in ['mse', 'settling_time', 'overshoot', 'control_effort'])

        # Check cache
        if cache_key in self.weight_cache:
            self.cache_hits += 1
            return self.weight_cache[cache_key]

        # Cache miss - run low-level GA
        self.cache_misses += 1

        import hashlib
        seed_string = str(cache_key)
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

        # Create config with unique seed
        low_level_config_with_seed = self.low_level_config.copy()
        low_level_config_with_seed['seed'] = seed_hash % (2 ** 31)  # Keep within valid range

        low_level_optimizer = GAOptimizer(self.simulator, low_level_config_with_seed)
        result = low_level_optimizer.optimize_pid(
            weights=weights,
            param_ranges=self.pid_param_ranges,
            fixed_targets=self.fixed_targets
        )

        # Store in cache
        self.weight_cache[cache_key] = result
        return result

    def optimize_weights(
            self,
            fixed_targets: Dict[str, float],
            pid_param_ranges: Dict[str, List[float]],
            weight_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize weights using high-level GA to minimize baseline cost.
        """
        self.fixed_targets = fixed_targets
        self.pid_param_ranges = pid_param_ranges

        # Default weight ranges: [0.0, 10.0] for each weight
        if weight_ranges is None:
            weight_ranges = {
                'mse': [0.0, 10.0],
                'settling_time': [0.0, 10.0],
                'overshoot': [0.0, 10.0],
                'control_effort': [0.0, 10.0]
            }
        self.weight_ranges = weight_ranges

        logger.info("=" * 80)
        logger.info("CASCADE GA OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"High-level GA: pop={self.high_level_config['population_size']}, "
                    f"gen={self.high_level_config['generations']}")
        logger.info(f"Low-level GA: pop={self.low_level_config['population_size']}, "
                    f"gen={self.low_level_config['generations']}")
        logger.info(f"Fixed targets: {fixed_targets}")
        logger.info(f"Weight ranges: {weight_ranges}")
        logger.info(f"PID ranges: {pid_param_ranges}")

        # Define high-level fitness function with caching
        def fitness_func_high_level(ga_instance, solution, solution_idx):
            """
            High-level fitness: minimize baseline cost
            Solution encoding: [w_mse, w_settling, w_overshoot, w_control]
            """
            weights = {
                'mse': solution[0],
                'settling_time': solution[1],
                'overshoot': solution[2],
                'control_effort': solution[3]
            }

            # Use cached evaluation
            result = self._evaluate_weights(weights)

            if not result['success']:
                return -1e6  # Large penalty for failure

            # Return negative baseline cost (GA maximizes fitness)
            baseline_cost = result['baseline_cost']
            return -baseline_cost

        # High-level GA progress tracking
        def on_generation_high_level(ga_instance):
            gen = ga_instance.generations_completed
            solution, fitness, _ = ga_instance.best_solution()

            # Decode best weights
            best_weights = {
                'mse': solution[0],
                'settling_time': solution[1],
                'overshoot': solution[2],
                'control_effort': solution[3]
            }

            # Use cached evaluation for best solution
            result = self._evaluate_weights(best_weights)

            # Calculate mean baseline cost across population
            pop = ga_instance.population
            pop_baseline_costs = []
            for sol in pop:
                w = {
                    'mse': sol[0],
                    'settling_time': sol[1],
                    'overshoot': sol[2],
                    'control_effort': sol[3]
                }
                ll_result = self._evaluate_weights(w)
                if ll_result['success']:
                    pop_baseline_costs.append(ll_result['baseline_cost'])
                else:
                    pop_baseline_costs.append(np.nan)

            mean_baseline = np.nanmean(pop_baseline_costs)

            # Store progress
            self.progress['iteration'].append(gen)
            self.progress['best_baseline_cost'].append(-fitness if result['success'] else np.nan)
            self.progress['mean_baseline_cost'].append(mean_baseline)

            # Store best weights
            self.progress['best_weights'].append(best_weights)
            self.progress['weight_mse'].append(best_weights['mse'])
            self.progress['weight_settling_time'].append(best_weights['settling_time'])
            self.progress['weight_overshoot'].append(best_weights['overshoot'])
            self.progress['weight_control_effort'].append(best_weights['control_effort'])

            # Store best controller and metrics
            if result['success']:
                self.progress['best_controller'].append(result['controller_parameters'])
                self.progress['Kp'].append(result['controller_parameters']['Kp'])
                self.progress['Ki'].append(result['controller_parameters']['Ki'])
                self.progress['Kd'].append(result['controller_parameters']['Kd'])

                self.progress['mse'].append(result['achieved_metrics']['mse'])
                self.progress['settling_time'].append(result['achieved_metrics']['settling_time'])
                self.progress['overshoot'].append(result['achieved_metrics']['overshoot'])
                self.progress['control_effort'].append(result['achieved_metrics']['control_effort'])

                self.progress['ga_cost'].append(result['ga_cost'])
                self.progress['success'].append(True)
            else:
                self.progress['best_controller'].append(None)
                self.progress['Kp'].append(np.nan)
                self.progress['Ki'].append(np.nan)
                self.progress['Kd'].append(np.nan)

                self.progress['mse'].append(np.nan)
                self.progress['settling_time'].append(np.nan)
                self.progress['overshoot'].append(np.nan)
                self.progress['control_effort'].append(np.nan)

                self.progress['ga_cost'].append(np.nan)
                self.progress['success'].append(False)

            logger.info(f"High-Level Gen {gen}/{self.high_level_config['generations']}: "
                        f"Best Baseline Cost = {-fitness:.4f}, Mean Baseline Cost = {mean_baseline:.4f}")
            logger.info(f"  Cache: {self.cache_hits} hits, {self.cache_misses} misses")
            logger.info(f"  Best Weights: MSE={best_weights['mse']:.4f}, "
                        f"Settling={best_weights['settling_time']:.4f}, "
                        f"Overshoot={best_weights['overshoot']:.4f}, "
                        f"Control={best_weights['control_effort']:.4f}")
            if result['success']:
                logger.info(f"  Best Controller: Kp={result['controller_parameters']['Kp']:.4f}, "
                            f"Ki={result['controller_parameters']['Ki']:.4f}, "
                            f"Kd={result['controller_parameters']['Kd']:.4f}")

        # Initialize high-level GA with better diversity
        ga_instance_high = pygad.GA(
            num_generations=self.high_level_config['generations'],
            num_parents_mating=max(2, self.high_level_config['population_size'] // 5),
            fitness_func=fitness_func_high_level,
            sol_per_pop=self.high_level_config['population_size'],
            num_genes=4,  # 4 weights
            gene_space=[
                {'low': weight_ranges['mse'][0], 'high': weight_ranges['mse'][1]},
                {'low': weight_ranges['settling_time'][0], 'high': weight_ranges['settling_time'][1]},
                {'low': weight_ranges['overshoot'][0], 'high': weight_ranges['overshoot'][1]},
                {'low': weight_ranges['control_effort'][0], 'high': weight_ranges['control_effort'][1]}
            ],
            parent_selection_type="sss",
            keep_parents=1,  # Keep fewer parents for diversity
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=25,  # Mutate 25% of genes
            mutation_probability=0.3,  # 30% mutation probability
            on_generation=on_generation_high_level,
            random_seed=self.high_level_config.get('seed', 42),
            allow_duplicate_genes=True
        )

        # Run high-level optimization
        logger.info("Starting high-level GA optimization...")
        ga_instance_high.run()
        logger.info("High-level GA optimization complete!")
        logger.info(f"Total cache statistics: {self.cache_hits} hits, {self.cache_misses} misses")

        # Get final best solution
        solution, fitness, _ = ga_instance_high.best_solution()

        best_weights = {
            'mse': solution[0],
            'settling_time': solution[1],
            'overshoot': solution[2],
            'control_effort': solution[3]
        }

        # Final evaluation with best weights
        logger.info("=" * 80)
        logger.info("Running final evaluation with best weights...")
        final_result = self._evaluate_weights(best_weights)

        logger.info("=" * 80)
        logger.info("CASCADE GA OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Weights: {best_weights}")
        logger.info(f"Best Baseline Cost: {final_result['baseline_cost']:.4f}")
        logger.info(f"Best GA Cost: {final_result['ga_cost']:.4f}")
        logger.info(f"Best Controller: {final_result['controller_parameters']}")
        logger.info(f"Achieved Metrics: {final_result['achieved_metrics']}")

        return {
            'success': final_result['success'],
            'best_weights': best_weights,
            'controller_parameters': final_result['controller_parameters'],
            'achieved_metrics': final_result['achieved_metrics'],
            'baseline_cost': final_result['baseline_cost'],
            'ga_cost': final_result['ga_cost'],
            'progress': self.progress,
            'warnings': final_result.get('warnings', []),
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
        }

    def _evaluate_weights_and_ranges(
            self,
            weights: Dict[str, float],
            pid_ranges: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate a combination of weights AND PID ranges using low-level GA with caching.

        Args:
            weights: Weight configuration
                Example: {"mse": 5.0, "settling_time": 4.0, ...}
            pid_ranges: PID range configuration (upper bounds, lower is always 0)
                Example: {"Kp": 150.0, "Ki": 25.0, "Kd": 40.0}
                Will be converted to param_ranges: {"Kp": [0.0, 150.0], ...}

        Returns:
            Low-level GA optimization result
        """
        import hashlib

        # Create cache key from both weights and ranges (round to avoid floating point issues)
        weight_key = tuple(round(weights[k], 6) for k in ['mse', 'settling_time', 'overshoot', 'control_effort'])
        range_key = tuple(round(pid_ranges[k], 6) for k in ['Kp', 'Ki', 'Kd'])
        cache_key = weight_key + range_key  # Combined cache key (7-tuple)

        # Check cache
        if cache_key in self.weight_cache:
            self.cache_hits += 1
            return self.weight_cache[cache_key]

        # Cache miss - run low-level GA
        self.cache_misses += 1

        # NEW: This is a high-level evaluation (unique config)
        self.num_high_level_evaluations += 1

        # CRITICAL: Generate unique seed from cache key
        # This ensures each weight+range configuration explores different solution space
        seed_string = str(cache_key)
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

        # Create config with unique seed
        low_level_config_with_seed = self.low_level_config.copy()
        low_level_config_with_seed['seed'] = seed_hash % (2 ** 31)  # Keep within valid range


        low_level_optimizer = GAOptimizer(self.simulator, low_level_config_with_seed)

        param_ranges = {
            "Kp": [0.0, pid_ranges['Kp']],
            "Ki": [0.0, pid_ranges['Ki']],
            "Kd": [0.0, pid_ranges['Kd']]
        }

        result = low_level_optimizer.optimize_pid(
            weights=weights,
            param_ranges=param_ranges,
            fixed_targets=self.fixed_targets
        )

        # Track the earliest point (in global NFE and wall-clock time) at which
        # any low-level run first hit all fixed targets.
        ll_target_nfe = result.get('target_hit_nfe')
        if ll_target_nfe is not None:
            # Translate the run-local NFE into a global (cumulative) NFE by
            # adding the evaluations already completed before this run started.
            global_nfe = self.total_low_level_evaluations + ll_target_nfe
            if self.target_hit_nfe is None or global_nfe < self.target_hit_nfe:
                self.target_hit_nfe = global_nfe
                self.target_hit_time = result.get('target_hit_time')

        # Append per-LL-generation history to the global trace.
        # We do this BEFORE adding to total_low_level_evaluations so that
        # the local NFE values can be translated to global NFE correctly.
        ll_prog = result.get('progress', {})
        local_nfe = ll_prog.get('cumulative_nfe', [])
        local_bl_best = ll_prog.get('best_baseline_cost', [])
        local_scores = ll_prog.get('success_score', [])  # â† NEW

        # Pad success_score with zeros if the LL run produced fewer entries
        # (e.g. very early StopIteration) to keep the three lists aligned.
        n = len(local_nfe)
        local_scores_padded = list(local_scores) + [0] * max(0, n - len(local_scores))  # â† NEW

        nfe_base = self.total_low_level_evaluations

        for loc_nfe, bl, scr in zip(local_nfe, local_bl_best, local_scores_padded):  # â† MOD (added scr)
            global_nfe = nfe_base + int(loc_nfe)

            # Running minimum of baseline cost (unchanged)
            if bl is not None and not (isinstance(bl, float) and np.isnan(bl)):
                if bl < self._ll_running_best:
                    self._ll_running_best = bl

            # Running maximum of success score 
            if isinstance(scr, (int, float)) and not np.isnan(scr):
                if scr > self._ll_running_best_score:
                    self._ll_running_best_score = int(scr)

            self.ll_history_nfe.append(global_nfe)
            self.ll_history_best_baseline_so_far.append(
                self._ll_running_best if self._ll_running_best < float('inf')
                else float('nan')
            )
            self.ll_history_success_score.append(self._ll_running_best_score) 

        # Accumulate low-level evaluations
        self.total_low_level_evaluations += result['num_evaluations']

        # Store in cache
        self.weight_cache[cache_key] = result
        return result

    def optimize(
            self,
            weight_ranges: Dict[str, list],
            pid_range_limits: Dict[str, list],
            fixed_targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run cascade GA optimization.
        High-level GA optimizes weights AND PID ranges.
        Low-level GA optimizes PID parameters within the ranges.

        NEW: The high-level GA now stops early (before all generations are
        exhausted) when any low-level run has satisfied all fixed targets
        (i.e. self.target_hit_nfe is no longer None).  This mirrors the
        score_100 early-halt already present in GAOptimizer.

        Args:
            weight_ranges: Ranges for weight optimization
            pid_range_limits: Limits for PID range optimization
            fixed_targets: Fixed target values for baseline cost calculation

        Returns:
            Optimization results â€” same schema as before plus a new key:
                stop_reason : 'normal' | 'score_100'
        """
        self.fixed_targets = fixed_targets
        self.weight_ranges = weight_ranges
        self.pid_range_limits = pid_range_limits

        logger.info("=" * 80)
        logger.info("CASCADE GA OPTIMIZATION (WITH PID RANGE OPTIMIZATION)")
        logger.info("=" * 80)
        logger.info(f"High-level GA: pop={self.high_level_config['population_size']}, "
                    f"gen={self.high_level_config['generations']}")
        logger.info(f"Low-level GA: pop={self.low_level_config['population_size']}, "
                    f"gen={self.low_level_config['generations']}")
        logger.info(f"Fixed targets: {fixed_targets}")
        logger.info(f"Weight ranges: {weight_ranges}")
        logger.info(f"PID range limits: {pid_range_limits}")

        # -- track why the high-level GA stopped --------------------------
        _hl_stop_reason = ['normal']
        _hl_running_best_score = [0]

        # =====================================================================
        # Fitness function
        # =====================================================================
        def fitness_func_high_level(ga_instance, solution, solution_idx):
            weights = {
                'mse': solution[0],
                'settling_time': solution[1],
                'overshoot': solution[2],
                'control_effort': solution[3]
            }
            pid_ranges = {
                'Kp': solution[4],
                'Ki': solution[5],
                'Kd': solution[6]
            }
            result = self._evaluate_weights_and_ranges(weights, pid_ranges)

            if result['success']:
                return -result['baseline_cost']
            else:
                return -1e6

        # =====================================================================
        # on_generation callback
        # =====================================================================
        def on_generation_high_level(ga_instance):
            """Callback after each high-level generation."""
            gen = ga_instance.generations_completed

            solution, fitness, _ = ga_instance.best_solution()

            best_weights = {
                'mse': solution[0],
                'settling_time': solution[1],
                'overshoot': solution[2],
                'control_effort': solution[3]
            }
            best_pid_ranges = {
                'Kp': solution[4],
                'Ki': solution[5],
                'Kd': solution[6]
            }

            result = self._evaluate_weights_and_ranges(best_weights, best_pid_ranges)

            all_fitness = ga_instance.last_generation_fitness
            mean_baseline = -np.mean(all_fitness)

            # -- Progress tracking ---------------------------------
            self.progress['iteration'].append(gen)
            self.progress['best_baseline_cost'].append(-fitness)
            self.progress['mean_baseline_cost'].append(mean_baseline)
            self.progress['best_weights'].append(best_weights)

            self.progress['weight_mse'].append(best_weights['mse'])
            self.progress['weight_settling_time'].append(best_weights['settling_time'])
            self.progress['weight_overshoot'].append(best_weights['overshoot'])
            self.progress['weight_control_effort'].append(best_weights['control_effort'])

            self.progress['range_Kp'].append(best_pid_ranges['Kp'])
            self.progress['range_Ki'].append(best_pid_ranges['Ki'])
            self.progress['range_Kd'].append(best_pid_ranges['Kd'])

            if result['success']:
                self.progress['best_controller'].append(result['controller_parameters'])
                self.progress['Kp'].append(result['controller_parameters']['Kp'])
                self.progress['Ki'].append(result['controller_parameters']['Ki'])
                self.progress['Kd'].append(result['controller_parameters']['Kd'])

                self.progress['mse'].append(result['achieved_metrics']['mse'])
                self.progress['settling_time'].append(result['achieved_metrics']['settling_time'])
                self.progress['overshoot'].append(result['achieved_metrics']['overshoot'])
                self.progress['control_effort'].append(result['achieved_metrics']['control_effort'])

                self.progress['ga_cost'].append(result['ga_cost'])
                self.progress['success'].append(True)

                hl_scr = self._compute_success_score(  # uses the new method
                    result.get('achieved_metrics', {}),
                    self.fixed_targets,
                )
            else:
                self.progress['best_controller'].append(None)
                self.progress['Kp'].append(np.nan)
                self.progress['Ki'].append(np.nan)
                self.progress['Kd'].append(np.nan)

                self.progress['mse'].append(np.nan)
                self.progress['settling_time'].append(np.nan)
                self.progress['overshoot'].append(np.nan)
                self.progress['control_effort'].append(np.nan)

                self.progress['ga_cost'].append(np.nan)
                self.progress['success'].append(False)

                hl_scr = 0

            self.progress['low_level_evals_cumulative'].append(self.total_low_level_evaluations)
            self.progress['high_level_evals_cumulative'].append(self.num_high_level_evaluations)

            self.progress['success_score'].append(hl_scr)
            if hl_scr > _hl_running_best_score[0]:
                _hl_running_best_score[0] = hl_scr
            self.progress['best_score_so_far'].append(_hl_running_best_score[0])

            # -- Logging -------------------------------------------
            logger.info(f"High-Level Gen {gen}/{self.high_level_config['generations']}: "
                        f"Best Baseline Cost = {-fitness:.4f}, Mean Baseline Cost = {mean_baseline:.4f}")
            logger.info(f"  Cache: {self.cache_hits} hits, {self.cache_misses} misses")
            logger.info(f"  Evaluations: Low-level={self.total_low_level_evaluations}, "
                        f"High-level={self.num_high_level_evaluations}")
            logger.info(f"  Best Weights: MSE={best_weights['mse']:.4f}, "
                        f"Settling={best_weights['settling_time']:.4f}, "
                        f"Overshoot={best_weights['overshoot']:.4f}, "
                        f"Control={best_weights['control_effort']:.4f}")
            logger.info(f"  Best Ranges: Kp=[0,{best_pid_ranges['Kp']:.2f}], "
                        f"Ki=[0,{best_pid_ranges['Ki']:.2f}], "
                        f"Kd=[0,{best_pid_ranges['Kd']:.2f}]")
            if result['success']:
                logger.info(f"  Best Controller: Kp={result['controller_parameters']['Kp']:.4f}, "
                            f"Ki={result['controller_parameters']['Ki']:.4f}, "
                            f"Kd={result['controller_parameters']['Kd']:.4f}")

            # -- propagate low-level early-halt to high-level GA ----------
            #
            # self.target_hit_nfe is set inside _evaluate_weights_and_ranges
            # the moment any low-level run returns with stop_reason='score_100'
            # (all four fixed targets satisfied).  Once that happens there is no
            # point running further high-level generations â€” the current
            # weight+range configuration already yields a compliant controller.
            #
            # Raising StopIteration here mirrors the identical pattern in
            # GAOptimizer.on_generation.  PyGAD catches the exception and stops
            # its main loop; we catch it in the try/except block below.
            if self.target_hit_nfe is not None:
                _hl_stop_reason[0] = 'score_100'
                logger.info(
                    f"  HL STOP (score=100): all targets met by a low-level run "
                    f"at global NFE={self.target_hit_nfe}. "
                    f"Halting high-level GA after gen {gen}."
                )
                return "stop"  # â† ADD this line (PyGAD clean-stop)

        # =====================================================================
        # Build high-level GA instance
        # =====================================================================
        ga_instance_high = pygad.GA(
            num_generations=self.high_level_config['generations'],
            num_parents_mating=max(2, self.high_level_config['population_size'] // 5),
            fitness_func=fitness_func_high_level,
            sol_per_pop=self.high_level_config['population_size'],
            num_genes=7,  # (4 weights + 3 ranges)
            gene_space=[
                # Weight genes (indices 0-3)
                {'low': weight_ranges['mse'][0], 'high': weight_ranges['mse'][1]},
                {'low': weight_ranges['settling_time'][0], 'high': weight_ranges['settling_time'][1]},
                {'low': weight_ranges['overshoot'][0], 'high': weight_ranges['overshoot'][1]},
                {'low': weight_ranges['control_effort'][0], 'high': weight_ranges['control_effort'][1]},
                # PID range genes (indices 4-6)
                {'low': pid_range_limits['Kp'][0], 'high': pid_range_limits['Kp'][1]},
                {'low': pid_range_limits['Ki'][0], 'high': pid_range_limits['Ki'][1]},
                {'low': pid_range_limits['Kd'][0], 'high': pid_range_limits['Kd'][1]},
            ],
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=25,
            mutation_probability=0.3,
            on_generation=on_generation_high_level,
            random_seed=self.high_level_config.get('seed', 42),
            allow_duplicate_genes=True
        )

        # =====================================================================
        # Run high-level optimization
        # =====================================================================
        logger.info("Starting high-level GA optimization...")
        ga_instance_high.run()  # clean â€” no exception raised
        if _hl_stop_reason[0] != 'normal':
            logger.info(
                f"  High-level GA halted early: {_hl_stop_reason[0]} "
                f"(completed {ga_instance_high.generations_completed}/"
                f"{self.high_level_config['generations']} generations)"
            )

        logger.info("High-level GA optimization complete!")
        logger.info(f"Total cache statistics: {self.cache_hits} hits, {self.cache_misses} misses")

        # =====================================================================
        # Extract best solution and run final evaluation
        # =====================================================================
        solution, fitness, _ = ga_instance_high.best_solution()

        best_weights = {
            'mse': solution[0],
            'settling_time': solution[1],
            'overshoot': solution[2],
            'control_effort': solution[3]
        }
        best_pid_ranges = {
            'Kp': solution[4],
            'Ki': solution[5],
            'Kd': solution[6]
        }

        # Final evaluation with best configuration
        logger.info("=" * 80)
        logger.info("Running final evaluation with best weights and ranges...")
        final_result = self._evaluate_weights_and_ranges(best_weights, best_pid_ranges)

        logger.info("=" * 80)
        logger.info("CASCADE GA OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Weights: {best_weights}")
        logger.info(f"Best PID Ranges: {best_pid_ranges}")
        logger.info(f"Best Baseline Cost: {final_result['baseline_cost']:.4f}")
        logger.info(f"Best GA Cost: {final_result['ga_cost']:.4f}")
        logger.info(f"Best Controller: {final_result['controller_parameters']}")
        logger.info(f"Achieved Metrics: {final_result['achieved_metrics']}")
        logger.info(f"Stop reason: {_hl_stop_reason[0]}")  # NEW

        # Dense per-LL-generation history
        self.progress['ll_cumulative_nfe'] = list(self.ll_history_nfe)
        self.progress['ll_best_baseline_so_far'] = list(self.ll_history_best_baseline_so_far)
        self.progress['ll_success_score'] = list(self.ll_history_success_score)  # â† NEW

        return {
            'success': final_result['success'],
            'best_weights': best_weights,
            'best_pid_ranges': best_pid_ranges,
            'controller_parameters': final_result['controller_parameters'],
            'achieved_metrics': final_result['achieved_metrics'],
            'baseline_cost': final_result['baseline_cost'],
            'ga_cost': final_result['ga_cost'],
            'progress': self.progress,
            'warnings': final_result.get('warnings', []),
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses
            },
            # NEW: Add evaluation statistics
            'evaluation_stats': {
                'total_low_level_evaluations': self.total_low_level_evaluations,
                'num_high_level_evaluations': self.num_high_level_evaluations,
                'total_evaluations': self.total_low_level_evaluations
            },
            # First global NFE / wall-clock instant at which a low-level run
            # satisfied all fixed targets (None if never achieved).
            'target_hit_nfe': self.target_hit_nfe,
            'target_hit_time': self.target_hit_time,
            'stop_reason': _hl_stop_reason[0],  # 'normal' | 'score_100'
            'hl_generations_completed': ga_instance_high.generations_completed,
        }

