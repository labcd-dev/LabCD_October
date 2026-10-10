from __future__ import annotations

import copy
from enum import Enum
from typing import Any, TypedDict, Dict, List
from langgraph.graph import END, StateGraph

from backend_core.MuloDesigner.mulo_tuning_orchestrator import MuloTuningOrchestrator
from backend_core.MuloDesigner.Agents.mulo_design_agents import Agents


class Decision(str, Enum):
    PROCEED = "Proceed"  # Everything is okay
    INCREASE_OPTIMIZE_BUDGET = "Increase Optimizer Budget"
    EXTEND_SEARCH_SPACE = "Extend Search Space"  # Fine Tuner /
    RESTRICT_SEARCH_SPACE = "Restrict Search Space"
    SELECTIVE_RE_DESIGN = "Selective Re-design"  # Evaluator
    CHANGE_WEIGHTS = "Change Weights"  # Effect of some metrics are very high and we need to decrease its weight
    CHANGE_OBJECTIVE = "Change Objective"  # NEW: Switch between tracking and regulating
    CHANGE_TARGET_METRICS = "Change Target Metrics"  # NEW: Relax or tighten the numeric acceptance thresholds
    TERMINATE = "Terminate"  # We've tried everything and it seems this is the best we can get


class Action(TypedDict, total=False):
    loop_index: int
    controller_index: int
    scale: Dict[str, float]
    new_population_size: int
    new_generation_number: int
    new_max_time: int
    new_weights: Dict[str, float]
    fine_tuner_range: int
    selected_controller: Dict[str, int]
    new_objective: str  # NEW: String payload like "Regulate"
    new_target_metrics: Dict[str, float]  # NEW: Numeric thresholds to relax/tighten for the current controller


class History(TypedDict):
    columns: List[str]
    data: List[List[float]]


class Checkpoint(str, Enum):
    INITIAL_CONTEXT = "Initial Context"
    AFTER_SUPERVISOR_DECISION = "After Supervisor Decision"
    AFTER_FINE_TUNER_DECISION = "After Fine Tuner Decision"
    AFTER_CONTROL_BLOCK_COMPLETED = "After Control Block Completed"
    AFTER_WHOLE_STRUCTURE_COMPLETED = "After Whole Structure Completed"


class UserContext(TypedDict):
    loop_index: int
    controller_index: int
    attempt: int
    checkpoint: Checkpoint
    context: str


class TuningState(TypedDict, total=False):
    user_context: UserContext | None
    user_context_history: List[UserContext]
    loop_index: int
    controller_index: int
    attempt: int
    gains: Dict[str, float]
    achieved_metrics: dict[str, float]
    achieved_cost: float
    target_metrics: dict[str, float]
    optimizer_max_run_time: float  # TODO: If general agent is going to product this must contain gen and pop too
    weights: Dict[str, float]
    param_ranges: Dict[str, float]
    evolution_history: History
    optimizer_agent_decision_trail: List[Any]  # ADDED: To hold GA decision trails
    supervisor_decision: Decision | None
    reason: str
    action: Action | None
    action_history: List[Dict[Decision, Action]]
    error: str | None


class MuloSupervisorAgent:
    """
  Agentic supervisor for one control-design session.

  The MuloTuningOrchestrator is created once in __init__ and survives for
  the complete LangGraph execution. LangGraph owns workflow state and
  routing; the orchestrator owns the actual control-design state and
  tuning operations.
  """
    MAX_GENERATION = 30
    MAX_POPULATION = 50
    MAX_PARAM_RANGES = 1000
    MAX_ATTEMPTS = 5
    MAX_FINE_TUNING_TRIES = 3
    MAX_SELECTIVE_RE_DESIGN = 2

    def __init__(self, run_config: Dict[str, Any], controller_structure: List[Any],
                 system_identification: Dict[str, Any], trimming_result: Dict[str, Any], equation: str):
        # print("supervisor agent")
        self.orchestrator = MuloTuningOrchestrator(run_config, controller_structure, system_identification,
                                                   trimming_result, equation)
        self.control_block = []  # TODO: most be removed and handled by orchestrator
        self.current_state = {}
        self.final_states = []  # TODO: most be removed and handled through graph state
        self._graph = None
        self.initial_run_config = {}
        self.llm_agent = Agents(model_name=run_config["llm_model"])
        self.feedback_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """
    Run the complete agentic tuning workflow.

    A new orchestrator is NOT created during graph execution.
    """
        self._build_graph()

        self.control_block = self.orchestrator.initialize_control_block()

        run_config = self.orchestrator.get_run_config()
        run_config["weights"] = {'mse': 1.0, 'settling_time': 1.0, 'overshoot': 1.0, 'control_effort': 1.0}
        if run_config["control_objective"] == "":
            user_context = UserContext(loop_index=0, controller_index=0, attempt=0, checkpoint=Checkpoint.INITIAL_CONTEXT, context=run_config["control_objective"])
        else:
            user_context = None

        self.initial_run_config = copy.deepcopy(run_config)

        if not self.control_block:
            return [{"supervisor_decision": Decision.TERMINATE, "reason": "Design Process Has Been Completed"}]

        initial_state: TuningState = {
            "loop_index": self.orchestrator.get_loop_index(),
            "controller_index": 0,
            "attempt": 0,
            "optimizer_max_run_time": run_config["max_wall_clock"],
            "user_context": user_context,
            "user_context_history": [user_context] if user_context is not None else [],
            "supervisor_decision": None,
            "error": None,
            "action_history": []
        }

        result = self._graph.invoke(initial_state, config={"recursion_limit": 150})
        self.current_state = result.get("final_state", self.current_state)

        return self.final_states

    def _build_graph(self):
        if self._graph is not None:
            return self._graph

        graph = StateGraph(TuningState)

        graph.add_node("optimize", self._optimize_node)
        graph.add_node("supervisor", self._supervisor_agent_node)
        graph.add_node("increase_optimizer_budget", self._increase_budget_node)
        graph.add_node("change_search_space", self._change_search_space_node)
        graph.add_node("change_weights", self._change_weights_node)
        graph.add_node("change_objective", self._change_objective_node)
        graph.add_node("change_target_metrics", self._change_target_metrics_node)
        graph.add_node("next_controller", self._next_controller_node)
        graph.add_node("fine_tuner_agent", self._fine_tuner_agent_node)
        graph.add_node("fine_tune", self._fine_tune_node)
        graph.add_node("evaluate_agent", self._evaluate_agent_node)
        graph.add_node("selective_re_design", self._selective_redesign_node)
        graph.add_node("finalize_design", self._finalize_design_node)

        graph.set_entry_point("optimize")
        graph.add_conditional_edges(
            "optimize",
            self._route_optimize_skip,
            {
                "skip": "next_controller",
                "supervisor": "supervisor"
            },
        )

        graph.add_conditional_edges(
            "supervisor",
            self._route_supervisor_decision,
            {
                Decision.PROCEED: "next_controller",
                Decision.EXTEND_SEARCH_SPACE: "change_search_space",
                Decision.RESTRICT_SEARCH_SPACE: "change_search_space",
                Decision.INCREASE_OPTIMIZE_BUDGET: "increase_optimizer_budget",
                Decision.CHANGE_WEIGHTS: "change_weights",
                Decision.CHANGE_OBJECTIVE: "change_objective",
                Decision.CHANGE_TARGET_METRICS: "change_target_metrics",
                Decision.TERMINATE: "next_controller",
            },
        )

        graph.add_edge("change_search_space", "optimize")
        graph.add_edge("increase_optimizer_budget", "optimize")
        graph.add_edge("change_weights", "optimize")
        graph.add_edge("change_objective", "optimize")
        graph.add_edge("change_target_metrics", "optimize")

        graph.add_conditional_edges(
            "next_controller",
            self._route_after_next_controller,
            {
                "fine_tuner_agent": "fine_tuner_agent",
                "evaluate_agent": "evaluate_agent",
                "optimize": "optimize",
                "end": "finalize_design"
            },
        )

        graph.add_conditional_edges(
            "fine_tuner_agent",
            self._route_after_fine_tuner_agent,
            {
                "next_controller": "next_controller",
                "fine_tune": "fine_tune",
                "evaluate_agent": "evaluate_agent",
                "end": "finalize_design",
            },
        )
        graph.add_edge("fine_tune", "fine_tuner_agent")
        graph.add_conditional_edges(
            "evaluate_agent",
            self._route_after_evaluate_agent,
            {
                "selective_re_design": "selective_re_design",
                "end": "finalize_design",
            },
        )
        graph.add_edge("selective_re_design", "evaluate_agent")
        graph.add_edge("finalize_design", END)

        self._graph = graph.compile()
        return self._graph

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _optimize_node(self, state: TuningState) -> dict[str, Any]:
        control_block = self.control_block
        cont_index = state["controller_index"]
        attempt = state.get("attempt", 0)
        add_info = f"Try: {attempt} " + state["supervisor_decision"] if state["supervisor_decision"] is not None else ""

        self.orchestrator.set_controller_index(cont_index)

        name = self.generate_controller_name(control_block, cont_index, add_info)

        final_state, control_block = self.orchestrator.design_controller(control_block, name, self.llm_agent)
        self.current_state = final_state

        # TODO: Let's just save gains as a dictionary in control_block
        # TODO: It would be prefect to just pass final state here
        return self.optimizer_state_generation(final_state, attempt + 1)

    def _change_search_space_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=True)

        scale = state["action"]["scale"]

        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()

        def expand_bounds(val, bounds, scale_factor):
            lo, hi = bounds

            if val > hi * 0.1:
                return [0, min(hi * scale_factor, MuloSupervisorAgent.MAX_PARAM_RANGES)]
            if val < lo * 0.1:
                return [max(lo * scale_factor, -MuloSupervisorAgent.MAX_PARAM_RANGES), 0]
            return [max(lo * scale_factor, -MuloSupervisorAgent.MAX_PARAM_RANGES) or -10.0,
                    min(hi * scale_factor, MuloSupervisorAgent.MAX_PARAM_RANGES) or 10.0]

        cont = controller_structure[loop_index]["controllers"][controller_index]
        ranges = cont["param_ranges"]

        cont["param_ranges"] = {
            "Kp": expand_bounds(
                cont.get("kp", 0.0),
                ranges["Kp"],
                scale["kp"],
            ),
            "Ki": expand_bounds(
                cont.get("ki", 0.0),
                ranges["Ki"],
                scale["ki"],
            ),
            "Kd": expand_bounds(
                cont.get("kd", 0.0),
                ranges["Kd"],
                scale["kd"],
            ),
        }

        self.orchestrator.set_controller_structure(controller_structure)
        self.control_block = controller_structure[loop_index]["controllers"]
        return {"param_ranges": cont["param_ranges"]}

    def _increase_budget_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=True)

        action = state["action"]

        self._increase_budget(action)
        return {"optimizer_max_run_time": action["new_max_time"]}

    def _change_weights_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=True)

        new_weights = state["action"]["new_weights"]

        run_config = self.orchestrator.get_run_config()
        run_config["weights"] = new_weights
        self.orchestrator.set_run_config(run_config)

        return {"weights": new_weights}

    def _change_objective_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=False)

        new_obj = state["action"]["new_objective"]

        # 1. Fetch current architecture
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()

        # 2. Inject the new target objective directly into the active controller
        cont = controller_structure[loop_index]["controllers"][controller_index]
        cont["signal_type"] = new_obj

        # 3. Save the modified architecture back to the orchestrator
        self.orchestrator.set_controller_structure(controller_structure)
        self.control_block = controller_structure[loop_index]["controllers"]

        return state

    def _change_target_metrics_node(self, state: TuningState) -> TuningState:
        """
    Overrides the numeric acceptance thresholds (mse, settling_time,
    overshoot, control_effort) used to evaluate the CURRENT controller.

    We mutate two places so the change persists through the rest of this
    controller's optimization loop:
      1. `case_study["fixed_targets"]` – consumed by the tuner on the next
         run (see `MuloTuningOrchestrator.pre_processing`).
      2. `controller_structure[...]["metrics"]` – the source of truth used
         by `pre_processing` if it is ever re-invoked.
    """
        self._manage_optimizer_history(state, retain=True)

        new_targets = state["action"]["new_target_metrics"]
        if not new_targets:
            return {"target_metrics": state.get("target_metrics", {})}

        # 1. Case-study level (feeds the optimizer)
        case_study = self.orchestrator.get_case_study()
        case_study["fixed_targets"] = dict(new_targets)
        self.orchestrator.set_case_study(case_study)

        # 2. Controller-structure level (persistent source of truth)
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()

        cont = controller_structure[loop_index]["controllers"][controller_index]
        cont["metrics"] = dict(new_targets)

        self.orchestrator.set_controller_structure(controller_structure)
        self.control_block = controller_structure[loop_index]["controllers"]

        return {"target_metrics": new_targets}

    def _supervisor_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state)
        optimizer_choice = self.orchestrator.get_run_config()["optimizer_choice"]
        response = self.llm_agent.supervisor_agent(agent_context, optimizer_choice, self.MAX_ATTEMPTS)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"])

        return {
            "supervisor_decision": decision,
            "action": action,
            "reason": response.get("reason", ""),
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _fine_tuner_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state)
        response = self.llm_agent.fine_tuner_agent(self.control_block, agent_context, self.MAX_FINE_TUNING_TRIES)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"])

        if decision == decision.INCREASE_OPTIMIZE_BUDGET:
            self._increase_budget(action)

        return {
            "supervisor_decision": decision,
            "action": action,
            "reason": response.get("reason", ""),
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _evaluate_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state)
        response = self.llm_agent.evaluate_agent(self.orchestrator.get_controller_structure(), agent_context,
                                                 self.MAX_SELECTIVE_RE_DESIGN)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"])

        if decision == decision.INCREASE_OPTIMIZE_BUDGET:
            self._increase_budget(action)

        current_attempt = state.get("attempt", 0)
        if state.get("supervisor_decision") != Decision.SELECTIVE_RE_DESIGN:
            current_attempt = 0

        return {
            "attempt": current_attempt,
            "supervisor_decision": decision,
            "action": action,
            "reason": response.get("reason", ""),
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _fine_tune_node(self, state: TuningState) -> dict[str, Any]:
        channels = []
        control_block = self.control_block
        rng = state["action"]["fine_tuner_range"]
        attempt = state.get("attempt", 0) + 1
        fixed_targets = self.orchestrator.get_case_study()["fixed_targets"]

        for idx, cont in enumerate(control_block):
            channels.append(
                {
                    "controller_name": self.generate_controller_name(
                        control_block, idx, f"Fine Tuner Agent Try: {attempt}  (Range = {rng})"
                    ),
                    "fixed_targets": fixed_targets,
                    "param_ranges": {
                        "Kp": [cont["kp"] - rng, cont["kp"] + rng],
                        "Ki": [cont["ki"] - rng, cont["ki"] + rng],
                        "Kd": [cont["kd"] - rng, cont["kd"] + rng],
                    },
                    "simulator_config": self.orchestrator.get_simulator_config(cont)
                    , })

        final_state, control_block = self.orchestrator.fine_tune_control_block(control_block, channels)

        self.current_state = final_state
        self.control_block = control_block

        return self.optimizer_state_generation(final_state, attempt)

    def _next_controller_node(self, state: TuningState) -> dict[str, Any]:
        self._manage_optimizer_history(state, retain=False)

        self.orchestrator.set_run_config(copy.deepcopy(self.initial_run_config))

        next_controller = state["controller_index"] + 1

        self.orchestrator.set_controller_index(next_controller)

        self.final_states.append(self.current_state)
        self.current_state = {}

        return {
            "controller_index": next_controller,
            "attempt": 0,
            "supervisor_decision": None,
            "optimizer_max_run_time": self.initial_run_config["max_wall_clock"],
            "reason": "",
            "action": None
        }

    def _evaluate_agent_node(self, state: TuningState) -> dict[str, Any]:
        decision = Decision.SELECTIVE_RE_DESIGN
        action = Action(selected_controller={"inner_loop_index": 0, "outer_loop_index": 1, "inner_controller_index": 0,
                                             "outer_controller_index": 0})

        current_attempt = state.get("attempt", 0)
        if state.get("supervisor_decision") != Decision.SELECTIVE_RE_DESIGN:
            current_attempt = 0

        return {
            "attempt": current_attempt,
            "supervisor_decision": decision,
            "action": action,
            "reason": "",
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _selective_redesign_node(self, state: TuningState) -> dict[str, Any]:
        action = state["action"]["selected_controller"]
        attempt = state["attempt"] + 1

        inner_idx = action["inner_controller_index"]
        inner_loop_idx = action["inner_loop_index"]
        inner_block = self.orchestrator.get_controller_structure()[inner_loop_idx]["controllers"][inner_idx]
        add_info = f"By Tuning {inner_block['controlled_variable']} Controller"

        name = self.generate_controller_name(self.control_block, action["outer_controller_index"], add_info)

        final_state, struct = self.orchestrator.selective_redesign_controller(action["inner_loop_index"],
                                                                              action["inner_controller_index"],
                                                                              action["outer_loop_index"],
                                                                              action["outer_controller_index"], name,
                                                                              self.llm_agent)

        self.current_state = final_state
        return self.optimizer_state_generation(final_state, attempt)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_optimize_skip(self, state: TuningState) -> str:
        index = self.orchestrator.get_controller_index()
        if "skip" in self.control_block[index]:
            return "skip"
        return "supervisor"

    def _route_supervisor_decision(self, state: TuningState) -> Decision:
        attempt = state["attempt"]
        if attempt >= MuloSupervisorAgent.MAX_ATTEMPTS:
            return Decision.TERMINATE
        return state["supervisor_decision"]

    def _route_after_next_controller(self, state: TuningState) -> str:
        if len(self.control_block) == 1:
            if self.orchestrator.get_loop_index() >= len(self.orchestrator.get_controller_structure()) - 1:
                return "evaluate_agent"
            return "end"
        elif len(self.control_block) == state["controller_index"]:
            skip_fine_tune = all(["skip" in self.control_block[i] for i in range(len(self.control_block))])
            if skip_fine_tune:
                return "end"
            return "fine_tuner_agent"

        return "optimize"

    def _route_after_fine_tuner_agent(self, state: TuningState) -> str:
        decision = state["supervisor_decision"]
        attempt = state["attempt"]

        if decision == Decision.TERMINATE or attempt >= MuloSupervisorAgent.MAX_FINE_TUNING_TRIES:
            # FIX: Added '- 1' to correctly identify the final loop index
            if self.orchestrator.get_loop_index() >= len(self.orchestrator.get_controller_structure()) - 1:
                return "evaluate_agent"
            return "end"
        return "fine_tune"

    def _route_after_evaluate_agent(self, state: TuningState) -> str:
        decision = state["supervisor_decision"]
        attempt = state["attempt"]

        if attempt >= MuloSupervisorAgent.MAX_SELECTIVE_RE_DESIGN:
            return "end"
        elif decision == Decision.SELECTIVE_RE_DESIGN:
            return "selective_re_design"
        return "end"

    # ------------------------------------------------------------------
    # Agent decision
    # ------------------------------------------------------------------
    def _decide(self, state: TuningState) -> dict[str, Any]:
        metrics = state["achieved_metrics"]
        target_metrics = state["target_metrics"]
        achieved_cost = state["achieved_cost"]
        attempt = int(state["attempt"])

        # Provide a default action dictionary to satisfy the Action TypedDict
        action_params = {
            "scale": {"kp": 10.0, "ki": 10.0, "kd": 10.0},
            "new_population_size": 150,
            "new_generation_number": 75,
            "new_max_time": 120,
            "new_weights": {"mse": 1.0, "settling_time": 1.0, "overshoot": 1.0, "control_effort": 1.0},
            "new_objective": "Pulse"
        }

        # 1. Acceptance Criteria Evaluator
        if achieved_cost < target_metrics.get("baseline_cost", 200) and abs(metrics["mse"]) < target_metrics["mse"]:
            if attempt <= 2:
                # Early attempts require strict adherence to settling time and overshoot limits[cite: 3]
                if metrics["settling_time"] < target_metrics["settling_time"] and metrics["overshoot"] < target_metrics[
                    "overshoot"]:
                    return {
                        "supervisor_decision": Decision.PROCEED.value,
                        "action": action_params,
                        "reason": "All stringent metrics satisfied."
                    }
            else:
                # Later attempts relax the criteria to accept the controller[cite: 3]
                return {
                    "supervisor_decision": Decision.PROCEED.value,
                    "action": action_params,
                    "reason": "Relaxed baseline metrics satisfied after multiple attempts."
                }

        # 2. Retry & Penalty Routing
        if attempt == 1:
            return {
                "supervisor_decision": Decision.EXTEND_SEARCH_SPACE.value,
                "action": action_params,
                "reason": "First attempt failed. Re-try with an extended search space scaled by 10."
            }
        elif attempt == 2:
            return {
                "supervisor_decision": Decision.INCREASE_OPTIMIZE_BUDGET.value,
                "action": action_params,
                "reason": "Second attempt failed. Re-try with a heavier optimizer configuration."
            }
        elif attempt > 4:
            return {
                "supervisor_decision": Decision.TERMINATE.value,
                "action": action_params,
                "reason": "Maximum optimization attempts (4) exceeded without convergence."
            }

        # 3. Default Fallback
        return {
            "supervisor_decision": Decision.EXTEND_SEARCH_SPACE.value,
            "action": action_params,
            "reason": "Continuing search space expansion."
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _increase_budget(self, action):
        config = self.orchestrator.get_run_config()
        default_pop = MuloSupervisorAgent.MAX_POPULATION / 2
        default_gen = MuloSupervisorAgent.MAX_GENERATION / 2

        config["population_size"] = min(action.get("new_population_size", default_pop),
                                        MuloSupervisorAgent.MAX_POPULATION)
        config["generations"] = min(action.get("new_generation_number", default_gen),
                                    MuloSupervisorAgent.MAX_GENERATION)
        config["max_wall_clock"] = action["new_max_time"]

        self.orchestrator.set_run_config(config)

    def _get_agent_context(self, state: TuningState) -> dict:
        """Filters the graph state to provide clean context for the LLMs."""
        excluded_keys = {"supervisor_decision", "reason", "action", "error"}
        return {k: v for k, v in state.items() if k not in excluded_keys}

    def optimizer_state_generation(self, final_state: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        self.feedback_history = final_state.get("agentic_context", {})
        return {
            "attempt": attempt,
            "gains": final_state["best_result"]["gains"],
            "achieved_metrics": final_state["best_result"]["achieved_metrics"],
            "achieved_cost": final_state["best_result"]["achieved_cost"],
            "target_metrics": final_state["tuning_specs"]["target_metrics"],
            "weights": final_state["tuning_specs"]["weights"],
            "param_ranges": final_state["tuning_specs"]["param_ranges"],
            "evolution_history": History(**final_state["evolution_history"]),
            "optimizer_agent_decision_trail": self.feedback_history.get("decision_trial", [])
            # ADDED: Safely pulls GA context if it exists
        }

    def _finalize_design_node(self, state: TuningState) -> TuningState:
        self.orchestrator.set_controller_designed(True)
        self.orchestrator.set_loop_index(
            self.orchestrator.get_loop_index() + 1
        )

        # Restore from the saved baseline rather than fetching the current state
        self.orchestrator.set_run_config(copy.deepcopy(self.initial_run_config))

        return state

    def _manage_optimizer_history(self, state: TuningState, retain: bool = False) -> None:
        """Helper to clear or pass the optimizer's agentic context in the run config."""
        config = self.orchestrator.get_case_study()
        if retain:
            config["feedback_history"] = self.feedback_history
        else:
            config.pop("feedback_history", None)
        self.orchestrator.set_case_study(config)

    def generate_controller_name(self, control_block, j, add_info=""):
        return control_block[j]['controlled_variable'] + " " + add_info

    def get_graph(self):
        self._build_graph()
        return self._graph

    def get_orchestrator(self) -> MuloTuningOrchestrator:
        return self.orchestrator

    def get_final_states(self) -> list[dict[str, Any]]:
        return self.final_states