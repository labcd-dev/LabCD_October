from typing import Any, Dict, List
import copy

from backend_core.MuloDesigner.mulo_tuning_orchestrator import MuloTuningOrchestrator


class MuloSupervisorAgent:

    def __init__(self, run_config: Dict[str, Any], controller_structure: List[Any],
                 system_identification: Dict[str, Any], trimming_result: Dict[str, Any], equation: str):
        # print("supervisor agent")
        self.orchestrator = MuloTuningOrchestrator(run_config, controller_structure, system_identification, trimming_result, equation)
        self.current_state = {}
        self.final_states = []


    def rule_based_control_design(self) -> List[Dict[str, Any]]:
        target_metrics = {
            "mse": 0.01,
            "overshoot": 10,
            "settling_time": 4,
            "baseline_cost": 200,
            # "rise_time": 2,
            # "control_zero_crossings": 1,
            # "steady_state_error": 1
        }

        control_block = self.orchestrator.initialize_control_block()
        initial_pop_size, initial_gen, initial_time = self.get_initial_config()

        for j in range(len(control_block)):
            design_complete = False
            attempt = 0

            self.orchestrator.set_controller_index(j)
            self.set_configuration(initial_pop_size, initial_gen, initial_time)

            print()
            print("cont_index")
            print(j)
            print()

            while not design_complete:
                attempt += 1
                print()
                print("attempt")
                print(attempt)
                print()

                if attempt == 1:
                    # design
                    name = self.generate_controller_name(control_block, j)
                    self.current_state, control_block = self.orchestrator.design_controller(control_block, name)
                # elif attempt == 2:
                #     print("re-try with heavier configuration")
                #     name = self.generate_controller_name(control_block, j, "heavier configuration")
                #     self.use_heavier_config()
                #     self.current_state, control_block = self.orchestrator.design_controller(control_block, name)
                elif attempt == 2:
                    print("re-try with extended search space")
                    name = self.generate_controller_name(control_block, j, "extended search space")
                    control_block = self.extend_search_space(10)
                    # control_block = self.rule_based_extend_search_space(target_metrics, 10/(10**(attempt-3)))
                    self.current_state, control_block = self.orchestrator.design_controller(control_block, name)
                # elif attempt == 4:
                #     print("re-try with extended search space again")
                #     name = self.generate_controller_name(control_block, j, "extended search space again")
                #     control_block = self.extend_search_space(100)
                #     # control_block = self.rule_based_extend_search_space(target_metrics, 100)
                #     self.current_state, control_block = self.orchestrator.design_controller(control_block, name)
                else:
                    self.final_states.append(self.current_state)
                    break

                print()
                import pprint
                pprint.pprint(self.current_state)
                print()

                metrics = self.current_state["best_result"]["achieved_metrics"]
                base_line_cost = self.current_state["best_result"]["best_baseline_cost"]

                case_study = self.orchestrator.get_case_study()
                target = case_study["target"]
                output_channel = case_study["output_channel"]
                trim_point = case_study["trim_ics"][output_channel]
                mse_limit = target_metrics["mse"] * abs(target - trim_point)

                print()
                print("mse limit")
                print(mse_limit)
                print()

                print("true metrics")
                import pprint
                pprint.pprint(metrics)

                # self.set_new_weights(metrics, target_metrics)

                if base_line_cost < target_metrics["baseline_cost"] and abs(metrics["mse"]) < mse_limit:
                    if attempt <= 2:
                        if metrics["settling_time"] < target_metrics["settling_time"] and metrics["overshoot"] < target_metrics["overshoot"]:
                            design_complete = True
                            self.final_states.append(self.current_state)
                    else:
                        design_complete = True
                        self.final_states.append(self.current_state)

        if len(control_block)>1:
            print("re-try with fine tuner")
            channels = self.initialize_channels(control_block)
            self.current_state, control_block = self.orchestrator.fine_tune_control_block(control_block, channels)

        self.set_configuration(initial_pop_size, initial_gen, initial_time)
        self.orchestrator.set_loop_index(self.orchestrator.get_loop_index() + 1)
        self.orchestrator.set_controller_designed(True)

        return self.get_final_states()



    def initialize_channels(self, control_block):
        channels = []
        rng = 5
        fixed_targets = self.orchestrator.get_case_study()["fixed_targets"]

        for idx, cont in enumerate(control_block):
            channels.append({})
            channels[idx]["controller_name"] = self.generate_controller_name(control_block, idx, "Fine Tuner Agent")
            channels[idx]["fixed_targets"] = fixed_targets
            channels[idx]["param_ranges"] = {"Kp": [cont["kp"]-rng, cont["kp"]+rng], "Ki": [cont["ki"]-rng, cont["ki"]+rng], "Kd": [cont["kd"]-rng, cont["kd"]+rng]}
            channels[idx]["simulator_config"] = self.orchestrator.get_simulator_config(cont)

        return channels


    def set_new_weights(self, current_metrics: Dict[str, float], target_metrics: Dict[str, float]):
        run_config = self.orchestrator.get_run_config()

        new_weights = self.update_dynamic_weights(current_metrics, target_metrics)
        run_config["weights"] = new_weights

        self.orchestrator.set_run_config(run_config)


    def update_dynamic_weights(self, current_metrics: Dict[str, float],
                               target_metrics: Dict[str, float],
                               base_weight: float = 1.0,
                               sensitivity: float = 2.0) -> Dict[str, float]:
        """
        Dynamically adjusts optimizer weights based on how far current metrics
        are from the desired target conditions.
        """
        keys = ['mse', 'settling_time', 'overshoot', 'control_effort']
        dynamic_weights = {}

        for key in keys:
            # Get values, defaulting safely to prevent errors
            current_val = current_metrics.get(key, 0.0)

            # Prevent division by zero if target is exactly 0
            target_val = target_metrics.get(key, 1e-6)
            if target_val <= 0:
                target_val = 1e-6

            # Calculate relative distance (only penalize if exceeding target)
            if current_val > target_val:
                relative_error = (current_val - target_val) / target_val
                # Scale the weight up
                dynamic_weights[key] = base_weight + (sensitivity * relative_error)
            else:
                # If the metric is meeting the target, keep it at the base weight
                dynamic_weights[key] = base_weight

        # Optional: Normalize weights so the max weight is always a fixed upper bound (e.g., 10.0)
        # to prevent extreme gradient explosions in your cost function.
        max_w = max(dynamic_weights.values())
        if max_w > 10.0:
            for k in dynamic_weights:
                dynamic_weights[k] = (dynamic_weights[k] / max_w) * 10.0

        return dynamic_weights


    def design_with_changed_coupling_ratio(self, old_state, control_block, name):
        alpha = 0.5

        i = self.orchestrator.get_loop_index()
        controller_block = self.orchestrator.get_controller_structure()[i]["controllers"]
        if len(controller_block) > 1:
            # change cost function
            print("change cost function")

            # self.orchestrator.set_final_state([])

            final_state, control_block = self.orchestrator.design_controller(control_block, name, cost_function="compute_coupling_metrics", coupling_ratio=alpha)
            return final_state , control_block

        return old_state, control_block

    def extend_search_space(self, scale):
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()

        def expand_bounds(val, bounds, scale=100, default=10.0):
            # TODO: This requires more care
            lo, hi = bounds
            if val > hi*0.1:
                return [0, hi * scale]
            elif val < lo*0.1:
                return [lo * scale, 0]
            else:
                return [lo * scale or -default, hi * scale or default]

        # 1. Target ONLY the specific controller being tuned
        cont = controller_structure[loop_index]["controllers"][controller_index]

        r = cont["param_ranges"]

        # 2. Use .get() to safely pull the values without throwing a KeyError
        cont["param_ranges"] = {
            'Kp': expand_bounds(cont.get("kp", 0.0), r['Kp'], scale=scale),
            'Ki': expand_bounds(cont.get("ki", 0.0), r['Ki'], scale=scale),
            'Kd': expand_bounds(cont.get("kd", 0.0), r['Kd'], scale=scale)
        }

        self.orchestrator.set_controller_structure(controller_structure)

        return controller_structure[loop_index]["controllers"]

    def rule_based_extend_search_space(self, target_metrics, scale=10):
        controller_structure = self.orchestrator.get_controller_structure()
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()
        final_state = self.orchestrator.get_final_state()
        metrics = final_state["best_result"]["achieved_metrics"]

        severity = {metric:metrics[metric]/target_metrics[metric] for metric in target_metrics.keys() if metric != "baseline_cost"}
        violated_metrics = {metric: value for metric, value in severity.items() if value > 1.0}
        worst_metric = max(violated_metrics, key=violated_metrics.get)

        print(f"\nDominant metric: {worst_metric}")
        print(f"Severity: {violated_metrics[worst_metric]:.3f}")

        metric_to_gain = {
            "mse": "Kp",
            "settling_time": "Kp",
            "overshoot": "Kd",
        }

        if worst_metric == "control_effort":
            scale = 1/scale

        gain_to_expand = metric_to_gain.get(worst_metric, "Kp")

        cont = controller_structure[loop_index]["controllers"][controller_index]

        ranges = cont["param_ranges"]
        old_bounds = ranges[gain_to_expand]
        lo, hi = old_bounds
        range_width = hi - lo
        new_lo = lo - scale * range_width
        new_hi = hi + scale * range_width
        ranges[gain_to_expand] = [new_lo, new_hi]
        cont["param_ranges"] = ranges

        print(ranges)

        self.orchestrator.set_controller_structure(controller_structure)

        return controller_structure[loop_index]["controllers"]


    def use_heavier_config(self):
        gen_gain = 1.5
        pop_gain = 1.5

        current_pop_size, current_generations, current_time = self.get_initial_config()

        final_state  = self.orchestrator.get_final_state()
        nfe = final_state["feedback_history"][-1]["num_evaluations"]
        if nfe/current_pop_size < 10:
            current_time = 2*current_time

        self.set_configuration(current_pop_size * pop_gain, current_generations * gen_gain, current_time)

    def get_initial_config(self):
        run_config = self.orchestrator.get_run_config()
        pop_size = run_config["population_size"]
        gen = run_config["generations"]
        max_time = run_config["max_wall_clock"]
        return pop_size, gen, max_time

    def set_configuration(self, pop_size, gen, max_time):
        run_config = self.orchestrator.get_run_config()

        run_config["population_size"] = pop_size
        run_config["generations"] = gen
        run_config["max_wall_clock"] = max_time
        run_config["weights"] = {'mse': 1.0, 'settling_time': 1.0, 'overshoot': 1.0, 'control_effort': 0.0}

        self.orchestrator.set_run_config(run_config)

    def generate_controller_name(self, control_block, j, add_info = ""):
        return control_block[j]['controlled_variable'] + " " + add_info

    def get_final_states(self) -> List[Any]:
        return self.final_states

    def get_orchestrator(self):
        return self.orchestrator


