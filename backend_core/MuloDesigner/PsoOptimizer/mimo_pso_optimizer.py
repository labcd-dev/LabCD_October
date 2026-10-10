import numpy as np
from typing import Dict, Any, List
import concurrent.futures

from backend_core.MuloDesigner.simulator import SystemSimulator
from backend_core.MuloDesigner.equation_editor import replace_last_pid_controller_gains
from backend_core.MuloDesigner.GaAgent.src.logger import get_logger
from backend_core.MuloDesigner.GaAgent.src.callbacks import get_callback

# Inherit from the newly modularized base class
from backend_core.MuloDesigner.PsoOptimizer.pso_optimizer import PSOOptimizer

logger = get_logger(__name__)


def _parallel_evaluate_mimo_fitness(gains_array: np.ndarray, weights_list: List[Dict[str, float]],
                                    system_configs: List[Dict[str, Any]], equation: str) -> tuple:
    if np.all(np.abs(gains_array) < 1e-6):
        return float('inf'), {'success': False, 'metrics': [], 'warnings': ['Trivial all-zero gains rejected']}

    coupled_equation = equation

    for i, config in enumerate(system_configs):
        Kp, Ki, Kd = gains_array[i * 3: i * 3 + 3]
        is_bounded = (config.get('min_ctrl', "") != "" and config.get('max_ctrl', "") != "")

        coupled_equation = replace_last_pid_controller_gains(
            coupled_equation, Kp, Ki, Kd,
            is_bounded, config.get('min_ctrl', ""), config.get('max_ctrl', ""),
            config['controller_index']
        )

    total_cost = 0.0
    channel_metrics = []
    all_warnings = []
    all_success = True

    for i, config in enumerate(system_configs):
        Kp, Ki, Kd = gains_array[i * 3: i * 3 + 3]

        local_simulator = SystemSimulator(config, coupled_equation)
        res = local_simulator.evaluate_pid({'Kp': Kp, 'Ki': Ki, 'Kd': Kd})

        if not res['success']:
            all_success = False
            total_cost += float('inf')
            all_warnings.extend(res.get('warnings', [f'Channel {i} failed']))
            channel_metrics.append({})
            continue

        m = res['metrics']
        channel_metrics.append(m)
        weights = weights_list[i]

        channel_cost = sum(
            weights.get(met, 1.0) * (m[met] ** 2)
            for met in ['mse', 'settling_time', 'overshoot', 'control_effort']
            if met in m
        )
        total_cost += channel_cost

    return total_cost, {'success': all_success, 'metrics': channel_metrics, 'warnings': all_warnings}


class MIMOPSOOptimizer(PSOOptimizer):
    def __init__(self, system_configs: List[Dict[str, Any]], equation: str, config: Dict[str, Any]):
        base_config = config.copy()
        base_config['param_ranges'] = config['param_ranges_list'][0]
        base_config['fixed_targets'] = config['fixed_targets_list'][0]
        base_config['weights'] = config['weights_list'][0]

        # ----------------------------------------------------------------------
        # CRITICAL FIX: RECONSTRUCT BASELINE GAINS
        # The supervisor centers ranges symmetrically around the baseline gains
        # [Kp - rng, Kp + rng]. The exact midpoint is our perfect baseline.
        # We inject this into Particle 0 so the swarm anchors to known stability.
        # ----------------------------------------------------------------------
        baseline_particle = []
        for pr in config['param_ranges_list']:
            kp_min = pr.get('Kp_min', pr.get('Kp', [-1, 1])[0])
            kp_max = pr.get('Kp_max', pr.get('Kp', [-1, 1])[1])
            ki_min = pr.get('Ki_min', pr.get('Ki', [-1, 1])[0])
            ki_max = pr.get('Ki_max', pr.get('Ki', [-1, 1])[1])
            kd_min = pr.get('Kd_min', pr.get('Kd', [-1, 1])[0])
            kd_max = pr.get('Kd_max', pr.get('Kd', [-1, 1])[1])

            baseline_particle.extend([
                (kp_min + kp_max) / 2.0,
                (ki_min + ki_max) / 2.0,
                (kd_min + kd_max) / 2.0
            ])

        base_config['initial_conditions'] = [baseline_particle]

        super().__init__(system_configs[0], equation, base_config)

        self.system_configs = system_configs
        self.num_channels = len(system_configs)
        self.dim = 3 * self.num_channels

        self.param_ranges_list: List[Dict[str, float]] = config['param_ranges_list']
        self.fixed_targets_list: List[Dict[str, float]] = config['fixed_targets_list']
        self.weights_list: List[Dict[str, float]] = config['weights_list']
        self.controller_names: List[str] = config['controller_names']

        # Per-channel running-best trackers for the "so far" line
        self._running_best_per_channel: Dict[str, float] = {
            name: float('inf') for name in self.controller_names
        }
        self._running_best_score_per_channel: Dict[str, float] = {
            name: 0.0 for name in self.controller_names
        }

    def _get_bounds(self) -> np.ndarray:
        bounds_list = []
        for pr in self.param_ranges_list:
            bounds_list.extend([
                [pr.get('Kp_min', pr.get('Kp', [-1, 1])[0]), pr.get('Kp_max', pr.get('Kp', [-1, 1])[1])],
                [pr.get('Ki_min', pr.get('Ki', [-1, 1])[0]), pr.get('Ki_max', pr.get('Ki', [-1, 1])[1])],
                [pr.get('Kd_min', pr.get('Kd', [-1, 1])[0]), pr.get('Kd_max', pr.get('Kd', [-1, 1])[1])]
            ])
        return np.array(bounds_list)

    def _submit_evaluation_future(self, executor: concurrent.futures.ProcessPoolExecutor, position: np.ndarray):
        return executor.submit(
            _parallel_evaluate_mimo_fitness,
            position, self.weights_list, self.system_configs, self.equation
        )

    def compute_baseline_cost(self, metrics_list, fixed_targets_list=None) -> float:
        # Base class may pass self.fixed_targets (a single dict) — normalize.
        if fixed_targets_list is None:
            fixed_targets_list = self.fixed_targets_list
        elif isinstance(fixed_targets_list, dict):
            # Called from base optimize_pid with self.fixed_targets — use the full list.
            fixed_targets_list = self.fixed_targets_list

        if not metrics_list or not fixed_targets_list:
            return float('inf')
        if isinstance(metrics_list, dict):
            # Defensive: single-dict metrics — fall back to per-channel first entry.
            metrics_list = [metrics_list]

        total_cost = 0.0
        for i, metrics in enumerate(metrics_list):
            targets = fixed_targets_list[i] if i < len(fixed_targets_list) else fixed_targets_list[0]
            total_cost += super().compute_baseline_cost(metrics, targets)
        return total_cost

    def _compute_success_score(self, metrics_list, fixed_targets_list=None) -> float:
        if fixed_targets_list is None or isinstance(fixed_targets_list, dict):
            fixed_targets_list = self.fixed_targets_list
        if not metrics_list or not fixed_targets_list:
            return 0.0
        if isinstance(metrics_list, dict):
            metrics_list = [metrics_list]

        score = 0.0
        total_possible = 0.0
        for i, metrics in enumerate(metrics_list):
            targets = fixed_targets_list[i] if i < len(fixed_targets_list) else fixed_targets_list[0]
            score += super()._compute_success_score(metrics, targets)
            for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
                if metric in metrics and metric in targets:
                    total_possible += 25
        return (score / total_possible) * 100.0 if total_possible > 0 else 0.0

    def _dispatch_generation_callback(self, gen: int, cum_nfe: int, elapsed: float,
                                      running_best_baseline: float, bc: float, scr: float,
                                      running_best_score: float, gen_best_m_list: list, gen_best_pos: np.ndarray,
                                      gen_best_baseline_cost: float):
        try:
            _cb = get_callback()
            if _cb is not None:
                for i in range(self.num_channels):
                    m = gen_best_m_list[i] if isinstance(gen_best_m_list, list) and i < len(gen_best_m_list) else {}
                    Kp, Ki, Kd = gen_best_pos[i * 3: i * 3 + 3]

                    # Per-channel instantaneous baseline cost (this generation's best particle)
                    chan_gen_bc = super().compute_baseline_cost(m, self.fixed_targets_list[i])
                    # Per-channel success score for this generation
                    chan_scr = super()._compute_success_score(m, self.fixed_targets_list[i])

                    # Update per-channel running-best trackers
                    chan_name = self.controller_names[i]
                    if np.isfinite(chan_gen_bc):
                        prev_bc = self._running_best_per_channel.get(chan_name, float('inf'))
                        self._running_best_per_channel[chan_name] = min(prev_bc, chan_gen_bc)
                    prev_scr = self._running_best_score_per_channel.get(chan_name, 0.0)
                    self._running_best_score_per_channel[chan_name] = max(prev_scr, chan_scr)

                    running_chan_bc = self._running_best_per_channel.get(chan_name, float('inf'))
                    running_chan_scr = self._running_best_score_per_channel.get(chan_name, 0.0)

                    event = {
                        'event_type': 'generation',
                        'controller_name': chan_name,
                        'attempt': self.current_attempt,
                        'generation': gen,
                        'cumulative_nfe': cum_nfe,
                        'cumulative_wall_time': elapsed,
                        # Solid line: running best-so-far (monotone non-increasing)
                        'best_baseline_so_far': running_chan_bc if np.isfinite(running_chan_bc) else None,
                        # Dotted line: this generation's instantaneous best (fluctuates)
                        'best_baseline_cost': chan_gen_bc if np.isfinite(chan_gen_bc) else None,
                        'success_score': chan_scr,
                        'best_score_so_far': running_chan_scr,
                        'mse': m.get('mse', np.nan),
                        'settling_time': m.get('settling_time', np.nan),
                        'overshoot': m.get('overshoot', np.nan),
                        'control_effort': m.get('control_effort', np.nan),
                        'Kp': float(Kp),
                        'Ki': float(Ki),
                        'Kd': float(Kd),
                        'param_ranges': self.param_ranges_list[i],
                        'fixed_targets': self.fixed_targets_list[i],
                        'weights': self.weights_list[i],
                        'pop_size': self.population_size,
                        'num_gen': self.generations,
                    }
                    _cb(event)
        except Exception as e:
            logger.error(f"MIMO Callback error: {e}")

    def _extract_final_result(self, gbest_position: np.ndarray, gbest_result: Any, gbest_cost: float,
                              progress: dict, stop_reason: str, running_best_score: float,
                              target_hit: dict) -> Dict[str, Any]:
        final_params = []
        for i in range(self.num_channels):
            final_params.append({
                'Kp': float(gbest_position[i * 3]),
                'Ki': float(gbest_position[i * 3 + 1]),
                'Kd': float(gbest_position[i * 3 + 2])
            })

        logger.info(f"✓ MIMO Optimization complete! Achieved Controllers: {final_params}")

        return {
            'success': gbest_result['success'] if gbest_result else False,
            'controller_parameters_list': final_params,
            'achieved_metrics_list': gbest_result['metrics'] if gbest_result and gbest_result['success'] else [],
            'cost': gbest_cost,
            'baseline_cost': self.compute_baseline_cost(gbest_result['metrics'],
                                                        self.fixed_targets_list) if gbest_result and gbest_result[
                'success'] else float('inf'),
            'warnings': gbest_result.get('warnings', []) if gbest_result else [],
            'progress': progress,
            'num_evaluations': self.num_evaluations,
            'stop_reason': stop_reason,
            'final_score': running_best_score,
            'target_hit_nfe': target_hit['nfe'],
            'target_hit_time': target_hit['time'],
            'target_hit_generation': target_hit['generation'],
        }