from __future__ import annotations

import copy
import ctypes
import threading
import time
from enum import Enum
from typing import Any, Callable, TypedDict, Dict, List
from langgraph.graph import END, StateGraph

from backend_core.MuloDesigner.mulo_tuning_orchestrator import MuloTuningOrchestrator
from backend_core.MuloDesigner.simulator import simulate_system_response
from backend_core.MuloDesigner.Agents.mulo_design_agents import Agents, LIVE_AGENT_DECISIONS


class WorkflowRollbackException(BaseException):
    """
    Raised to immediately terminate the currently executing graph step
    and roll back to the latest completed graph-level agentic step.
    Inherits from BaseException so standard `except Exception:` blocks
    inside optimizers or callbacks do not swallow it.
    """
    pass


class WorkflowForceStopException(BaseException):
    """
    Raised when the user forcibly aborts the entire experiment (e.g., New Experiment).
    """
    pass


class Decision(str, Enum):
    PROCEED = "Proceed"  # Everything is okay
    INCREASE_OPTIMIZE_BUDGET = "Increase Optimizer Budget"
    EXTEND_SEARCH_SPACE = "Extend Search Space"  # Fine Tuner /
    RESTRICT_SEARCH_SPACE = "Restrict Search Space"
    SELECTIVE_RE_DESIGN = "Selective Re-design"  # Evaluator
    CHANGE_WEIGHTS = "Change Weights"  # Effect of some metrics are very high and we need to decrease its weight
    CHANGE_OBJECTIVE = "Change Objective"  # NEW: Switch between tracking and regulating
    CHANGE_TARGET_METRICS = "Change Target Metrics"  # NEW: Relax or tighten the numeric acceptance thresholds
    ADD_COST_PENALTY = "Add Cost Penalty"
    TERMINATE = "Terminate"  # We've tried everything and it seems this is the best we can get


class Action(TypedDict, total=False):
    loop_index: int
    controller_index: int
    deciding_agent: str
    comment: str
    reason: str
    scale: Dict[str, float]
    new_population_size: int
    new_generation_number: int
    new_max_time: int
    new_weights: Dict[str, float]
    fine_tuner_range: int
    selected_controller: Dict[str, int]
    new_objective: str  # NEW: String payload like "Regulate"
    new_target_metrics: Dict[str, float]  # NEW: Numeric thresholds to relax/tighten for the current controller
    new_metric: str
    new_target: float


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


class OptimizedControllerState(TypedDict):
    gains: Dict[str, float]
    target_metrics: Dict[str, float]
    achieved_metrics: Dict[str, float]
    achieved_cost: float


class TuningState(TypedDict, total=False):  # TODO: If there was way that I can pass certain fields to only certain agents
    _start_node: str
    user_context: UserContext | None
    user_context_history: List[UserContext]
    loop_index: int
    controller_index: int
    attempt: int
    gains: Dict[str, float]
    achieved_metrics: dict[str, float]
    achieved_cost: float
    target_metrics: dict[str, float]
    optimized_controllers: Dict[str, OptimizedControllerState]
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
    MAX_FINE_TUNING_TRIES = 2
    MAX_SELECTIVE_RE_DESIGN = 1

    AGENTIC_NODES = {"supervisor", "fine_tuner_agent", "evaluate_agent"}

    ALL_GRAPH_NODES = (
        "optimize",
        "supervisor",
        "increase_optimizer_budget",
        "change_search_space",
        "change_weights",
        "change_objective",
        "change_target_metrics",
        "add_cost_penalty",
        "next_controller",
        "evaluate_coupled_baseline",
        "fine_tuner_agent",
        "fine_tune",
        "evaluate_agent",
        "selective_re_design",
        "finalize_design",
    )

    SUPERVISOR_CONTEXT_KEYS = {
        "loop_index", "controller_index", "attempt", "gains",
        "achieved_metrics", "achieved_cost", "target_metrics",
        "optimizer_max_run_time", "weights", "param_ranges",
        "evolution_history", "optimizer_agent_decision_trail", "action_history",
        "user_context", "user_context_history",
    }

    FINE_TUNER_CONTEXT_KEYS = {
        "loop_index", "optimized_controllers", "optimizer_max_run_time", "action_history",
        "user_context", "user_context_history",
    }

    EVALUATE_CONTEXT_KEYS = {
        "attempt", "optimized_controllers", "action_history",
        "user_context", "user_context_history",
    }

    def __init__(self, run_config: Dict[str, Any], controller_structure: Dict[str, Any],
                 trimming_result: Dict[str, Any], equation: str):
        self.orchestrator = MuloTuningOrchestrator(run_config, controller_structure, trimming_result, equation)
        self._graph = None
        self.initial_run_config = {} # TODO: most be removed and handled through Orchestrator
        self.llm_agent = Agents(model_name=run_config["llm_model"])
        self.feedback_history = []

        # Persistent user context across steps and loops
        self.latest_user_context: UserContext | None = None # TODO: most be removed and handled through graph state
        self.user_context_history: List[UserContext] = [] # TODO: most be removed and handled through graph state

        # Thread-safe pause / context injection / rollback state
        self._lock = threading.RLock()
        self._pause_after_step_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._abort_requested = threading.Event()
        self._force_stop_requested = threading.Event()
        self._wait_after_rollback: bool = False
        self._pending_user_prompt: str | None = None

        self._latest_completed_agent_snapshot: Dict[str, Any] | None = None
        self._snapshot_counter: int = 0
        self._current_node_name: str = "Initializing"
        self._workflow_status: str = "idle"
        self._worker_thread_id: int | None = None
        self._in_cancellable_section: bool = False
        self._event_callback: Callable[[Dict[str, Any]], None] | None = None

    # ------------------------------------------------------------------
    # Live Pause, Context Injection & Rollback Public API
    # ------------------------------------------------------------------

    def set_event_callback(self, callback: Callable[[Dict[str, Any]], None] | None) -> None:
        with self._lock:
            self._event_callback = callback

    def _emit_event(self, event: Dict[str, Any]) -> None:
        cb = self._event_callback
        if cb is not None:
            try:
                cb(event)
            except BaseException:
                pass

    def can_rollback_to_agent(self) -> bool:
        """
        Returns True only if at least one graph-level agentic step
        (supervisor, fine_tuner_agent, or evaluate_agent) has completed in the current run.
        """
        with self._lock:
            return self._latest_completed_agent_snapshot is not None

    def get_workflow_status(self) -> Dict[str, Any]:
        """Returns thread-safe telemetry on workflow pause/rollback status and user contexts."""
        with self._lock:
            snap = self._latest_completed_agent_snapshot
            latest_agent_meta = None
            if snap is not None:
                st_snap = snap.get("state", {})
                latest_agent_meta = {
                    "node_name": snap["node_name"],
                    "loop_index": st_snap.get("loop_index", 0),
                    "controller_index": st_snap.get("controller_index", 0),
                    "attempt": st_snap.get("attempt", 0),
                    "checkpoint": self._get_checkpoint_for_node(snap["node_name"], st_snap).value,
                }
            return {
                "status": self._workflow_status,
                "current_node": self._current_node_name,
                "can_rollback": snap is not None,
                "latest_agent": latest_agent_meta,
                "pending_prompt": self._pending_user_prompt,
                "user_context": copy.deepcopy(self.latest_user_context),
                "user_context_history": copy.deepcopy(self.user_context_history),
            }

    def request_pause_after_current_step(self, prompt: str | None = None, pause_and_wait: bool = True) -> None:
        """
        Mode 1 (Graceful): Does NOT interrupt the currently executing node.
        - If `pause_and_wait=True` and no prompt is submitted yet, pauses right after the
          current node finishes so the user can write their prompt and resume.
        - If `prompt` is provided with `pause_and_wait=False`, queues the prompt so it is
          automatically injected right after the current node finishes without stopping.
        """
        with self._lock:
            if prompt is not None and prompt.strip():
                self._pending_user_prompt = prompt.strip()
            if pause_and_wait:
                self._pause_after_step_event.set()
                self._resume_event.clear()
                if self._workflow_status == "running":
                    self._workflow_status = "pausing_after_step"
            else:
                self._pause_after_step_event.clear()
                if self._workflow_status in ("paused", "pausing_after_step"):
                    self._workflow_status = "running"
                self._resume_event.set()

    def request_terminate_and_rollback(self, prompt: str | None = None, wait_for_prompt: bool = True) -> bool:
        """
        Mode 2 (Immediate Terminate & Rollback): Immediately terminates whatever node is
        currently running, rolls back to the state right before the latest completed
        graph-level agentic step (`supervisor`, `fine_tuner_agent`, or `evaluate_agent`),
        and either waits for the user's prompt (`wait_for_prompt=True`) or immediately
        injects `prompt` and re-runs that agentic step and all subsequent steps.
        """
        with self._lock:
            if self._latest_completed_agent_snapshot is None:
                return False

            if prompt is not None and prompt.strip():
                self._pending_user_prompt = prompt.strip()

            self._wait_after_rollback = wait_for_prompt or (self._pending_user_prompt is None)
            self._workflow_status = "terminating"
            self._abort_requested.set()
            self._pause_after_step_event.clear()
            self._resume_event.set()

        # Give cooperative check_abort() inside callbacks up to 0.25s to unwind cleanly first
        deadline = time.time() + 0.25
        while time.time() < deadline:
            with self._lock:
                if not self._in_cancellable_section:
                    return True
            time.sleep(0.02)

        # Fallback: Preemptively raise WorkflowRollbackException in the worker thread
        with self._lock:
            if self._in_cancellable_section and self._worker_thread_id is not None:
                self._in_cancellable_section = False
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._worker_thread_id),
                    ctypes.py_object(WorkflowRollbackException),
                )
        return True

    def resume_with_context(self, prompt: str | None = None) -> None:
        """
        Injects `prompt` (if non-empty) and resumes workflow execution from a paused
        or rolled-back state (or queues the prompt if a Mode 1 step is still finishing).
        """
        with self._lock:
            if prompt is not None and prompt.strip():
                self._pending_user_prompt = prompt.strip()
            self._wait_after_rollback = False
            self._pause_after_step_event.clear()
            if self._workflow_status in ("paused", "paused_after_rollback", "pausing_after_step"):
                self._workflow_status = "running"
            self._resume_event.set()

    def inject_context_while_idle(self, prompt: str, checkpoint: Checkpoint | None = None) -> UserContext | None:
        """
        Allows injecting user context when the graph is between loops (e.g. after a loop completes).
        """
        cleaned = (prompt or "").strip()
        if not cleaned:
            return None
        with self._lock:
            loop_idx = self.orchestrator.get_loop_index()
            cont_idx = self.orchestrator.get_controller_index()
            is_done = loop_idx >= len(self.orchestrator.get_controller_structure())
            cp = checkpoint or (
                Checkpoint.AFTER_WHOLE_STRUCTURE_COMPLETED if is_done else Checkpoint.AFTER_CONTROL_BLOCK_COMPLETED
            )
            new_uc = UserContext(
                loop_index=loop_idx,
                controller_index=cont_idx,
                attempt=0,
                checkpoint=cp,
                context=cleaned,
            )
            self.latest_user_context = new_uc
            self.user_context_history.append(new_uc)
            return new_uc

    def force_stop(self) -> None:
        """Stops the worker thread cleanly if the user resets the experiment."""
        with self._lock:
            self._force_stop_requested.set()
            self._abort_requested.set()
            self._pause_after_step_event.clear()
            self._resume_event.set()
            if self._in_cancellable_section and self._worker_thread_id is not None:
                self._in_cancellable_section = False
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._worker_thread_id),
                    ctypes.py_object(WorkflowForceStopException),
                )

    def check_abort(self) -> None:
        """
        Called cooperatively by optimizer callbacks and at graph node boundaries.
        Immediately raises WorkflowRollbackException if termination/rollback was requested.
        """
        if self._force_stop_requested.is_set():
            with self._lock:
                self._in_cancellable_section = False
            raise WorkflowForceStopException("Experiment reset by user.")
        if self._abort_requested.is_set():
            with self._lock:
                self._in_cancellable_section = False
            raise WorkflowRollbackException("Workflow step terminated for agentic rollback.")

    def _clear_async_exc(self) -> None:
        with self._lock:
            if self._worker_thread_id is not None:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._worker_thread_id),
                    None,
                )

    def _consume_pending_prompt(self) -> str | None:
        with self._lock:
            prompt = self._pending_user_prompt
            self._pending_user_prompt = None
            return prompt

    def _get_checkpoint_for_node(self, node_name: str, state: TuningState) -> Checkpoint:
        if node_name in ("supervisor", "increase_optimizer_budget", "change_search_space",
                         "change_weights", "change_objective", "change_target_metrics", "add_cost_penalty"):
            return Checkpoint.AFTER_SUPERVISOR_DECISION
        if node_name in ("evaluate_coupled_baseline", "fine_tuner_agent", "fine_tune"):
            return Checkpoint.AFTER_FINE_TUNER_DECISION
        if node_name in ("evaluate_agent", "selective_re_design", "next_controller", "finalize_design"):
            if self.orchestrator.get_loop_index() >= len(self.orchestrator.get_controller_structure()) - 1:
                return Checkpoint.AFTER_WHOLE_STRUCTURE_COMPLETED
            return Checkpoint.AFTER_CONTROL_BLOCK_COMPLETED
        if self._latest_completed_agent_snapshot is not None:
            return Checkpoint.AFTER_SUPERVISOR_DECISION
        return Checkpoint.INITIAL_CONTEXT

    def _build_user_context(self, node_name: str, state: TuningState, prompt_text: str) -> UserContext:
        checkpoint_val = self._get_checkpoint_for_node(node_name, state)
        return UserContext(
            loop_index=int(state.get("loop_index", self.orchestrator.get_loop_index())),
            controller_index=int(state.get("controller_index", self.orchestrator.get_controller_index())),
            attempt=int(state.get("attempt", 0)),
            checkpoint=checkpoint_val,
            context=prompt_text,
        )

    def _capture_snapshot(self, node_name: str, state: TuningState) -> Dict[str, Any]:
        with self._lock:
            self._snapshot_counter += 1
            snap_id = self._snapshot_counter
            state_copy = copy.deepcopy(state)
            state_copy["_start_node"] = node_name

            snapshot = {
                "snapshot_id": snap_id,
                "node_name": node_name,
                "state": state_copy,
                "orchestrator": {
                    "run_config": copy.deepcopy(self.orchestrator.get_run_config()),
                    "equation": copy.deepcopy(self.orchestrator.get_equation()),
                    "case_study": copy.deepcopy(self.orchestrator.get_case_study()),
                    "controller_structure": copy.deepcopy(self.orchestrator.get_controller_structure()),
                    "control_block": copy.deepcopy(self.orchestrator.control_block),
                    "loop_index": self.orchestrator.get_loop_index(),
                    "controller_index": self.orchestrator.get_controller_index(),
                    "controller_designed": self.orchestrator.get_controller_designed(),
                    "working_function": self.orchestrator.working_function,
                    "final_state": copy.deepcopy(self.orchestrator.get_final_state()),
                },
                "supervisor_attrs": {
                    "initial_run_config": copy.deepcopy(self.initial_run_config),
                    "feedback_history": copy.deepcopy(self.feedback_history),
                },
                "live_decisions_len": len(LIVE_AGENT_DECISIONS),
            }
            self._emit_event({
                "event_type": "agent_snapshot",
                "snapshot_id": snap_id,
                "node_name": node_name,
            })
            return snapshot

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            orch_snap = snapshot["orchestrator"]
            self.orchestrator.set_run_config(copy.deepcopy(orch_snap["run_config"]))
            self.orchestrator.set_equation(copy.deepcopy(orch_snap["equation"]))
            self.orchestrator.set_case_study(copy.deepcopy(orch_snap["case_study"]))
            self.orchestrator.set_controller_structure(copy.deepcopy(orch_snap["controller_structure"]))
            self.orchestrator.control_block = copy.deepcopy(orch_snap["control_block"])
            self.orchestrator.set_loop_index(orch_snap["loop_index"])
            self.orchestrator.set_controller_index(orch_snap["controller_index"])
            self.orchestrator.set_controller_designed(orch_snap["controller_designed"])
            self.orchestrator.working_function = orch_snap["working_function"]
            self.orchestrator.set_final_state(copy.deepcopy(orch_snap["final_state"]))

            sup_snap = snapshot["supervisor_attrs"]
            self.initial_run_config = copy.deepcopy(sup_snap["initial_run_config"])
            self.feedback_history = copy.deepcopy(sup_snap["feedback_history"])

            live_len = snapshot.get("live_decisions_len", len(LIVE_AGENT_DECISIONS))
            del LIVE_AGENT_DECISIONS[live_len:]

            self._emit_event({
                "event_type": "rollback",
                "snapshot_id": snapshot["snapshot_id"],
                "node_name": snapshot["node_name"],
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, start_from_latest_agent: bool = False) -> None:
        """
        Run the complete agentic tuning workflow.

        A new orchestrator is NOT created during graph execution.
        Supports live Mode 1 (graceful pause & context injection after current step)
        and Mode 2 (immediate termination, rollback to latest completed agentic step,
        context injection, and re-running from that agentic step).
        """
        self._build_graph()

        with self._lock:
            self._worker_thread_id = threading.get_ident()
            self._abort_requested.clear()
            self._force_stop_requested.clear()
            self._pause_after_step_event.clear()
            self._resume_event.set()
            self._workflow_status = "running"

        if start_from_latest_agent and self._latest_completed_agent_snapshot is not None:
            snapshot = self._latest_completed_agent_snapshot
            self._restore_snapshot(snapshot)
            current_invoke_state: TuningState = copy.deepcopy(snapshot["state"])
            current_invoke_state["user_context"] = copy.deepcopy(self.latest_user_context)
            current_invoke_state["user_context_history"] = copy.deepcopy(self.user_context_history)
            prompt_text = self._consume_pending_prompt()
            if prompt_text:
                new_uc = self._build_user_context(snapshot["node_name"], current_invoke_state, prompt_text)
                updated_history = list(current_invoke_state.get("user_context_history", [])) + [new_uc]
                current_invoke_state["user_context"] = new_uc
                current_invoke_state["user_context_history"] = updated_history
                self.latest_user_context = new_uc
                self.user_context_history = copy.deepcopy(updated_history)
            current_invoke_state["_start_node"] = snapshot["node_name"]
        else:
            with self._lock:
                self._latest_completed_agent_snapshot = None

            control_block_exist = self.orchestrator.initialize_control_block()

            run_config = self.orchestrator.get_run_config()
            run_config["weights"] = {'mse': 1.0, 'settling_time': 1.0, 'overshoot': 1.0, 'control_effort': 1.0}

            if not self.user_context_history and run_config.get("control_objective", "") != "":
                initial_uc = UserContext(
                    loop_index=self.orchestrator.get_loop_index(),
                    controller_index=0,
                    attempt=0,
                    checkpoint=Checkpoint.INITIAL_CONTEXT,
                    context=run_config["control_objective"],
                )
                self.latest_user_context = initial_uc
                self.user_context_history = [initial_uc]

            self.initial_run_config = copy.deepcopy(run_config)

            if not control_block_exist:
                with self._lock:
                    self._workflow_status = "completed"
                return [{"supervisor_decision": Decision.TERMINATE, "reason": "Design Process Has Been Completed"}]

            current_invoke_state = {
                "_start_node": "optimize",
                "loop_index": self.orchestrator.get_loop_index(),
                "controller_index": 0,
                "attempt": 0,
                "optimizer_max_run_time": run_config["max_wall_clock"],
                "user_context": copy.deepcopy(self.latest_user_context),
                "user_context_history": copy.deepcopy(self.user_context_history),
                "supervisor_decision": None,
                "error": None,
                "action_history": [],
                "optimized_controllers": {},
            }

        while True:
            try:
                with self._lock:
                    self._in_cancellable_section = True
                result = self._graph.invoke(current_invoke_state, config={"recursion_limit": 150})
                with self._lock:
                    self._in_cancellable_section = False
                    self._workflow_status = "completed"
                    self._current_node_name = "Completed"
                break

            except WorkflowForceStopException:
                with self._lock:
                    self._in_cancellable_section = False
                    self._clear_async_exc()
                    self._workflow_status = "completed"
                break

            except WorkflowRollbackException:
                with self._lock:
                    self._in_cancellable_section = False
                    self._clear_async_exc()
                    self._abort_requested.clear()
                    snapshot = self._latest_completed_agent_snapshot

                if snapshot is None:
                    # Safety fallback if no agentic step has completed yet
                    with self._lock:
                        self._workflow_status = "running"
                    continue

                # 1. Restore orchestrator, supervisor attributes, and state to pre-agent snapshot
                self._restore_snapshot(snapshot)
                restored_state: TuningState = copy.deepcopy(snapshot["state"])
                target_agent_node = snapshot["node_name"]

                # Preserve all historical user contexts accumulated up to now
                if self.user_context_history:
                    restored_state["user_context"] = copy.deepcopy(self.latest_user_context)
                    restored_state["user_context_history"] = copy.deepcopy(self.user_context_history)

                # 2. If user clicked "Terminate & Rollback" first and is now writing their prompt, wait here
                with self._lock:
                    should_wait = self._wait_after_rollback and (self._pending_user_prompt is None)
                    if should_wait:
                        self._workflow_status = "paused_after_rollback"
                        self._current_node_name = f"Rolled back to {target_agent_node}"
                        self._resume_event.clear()

                if should_wait:
                    self._emit_event({
                        "event_type": "workflow_paused",
                        "node_name": target_agent_node,
                        "mode": "rollback",
                    })
                    self._resume_event.wait()
                    if self._force_stop_requested.is_set():
                        break

                # 3. Inject newly submitted UserContext into the restored state
                prompt_text = self._consume_pending_prompt()
                if prompt_text:
                    new_uc = self._build_user_context(target_agent_node, restored_state, prompt_text)
                    updated_history = list(restored_state.get("user_context_history", [])) + [new_uc]
                    restored_state["user_context"] = new_uc
                    restored_state["user_context_history"] = updated_history
                    with self._lock:
                        self.latest_user_context = new_uc
                        self.user_context_history = copy.deepcopy(updated_history)

                # 4. Re-enter the graph directly at the rolled-back agentic step
                restored_state["_start_node"] = target_agent_node
                current_invoke_state = restored_state
                with self._lock:
                    self._workflow_status = "running"
                continue


    def _wrap_node(self, node_name: str, node_fn: Callable[[TuningState], Any]) -> Callable[[TuningState], dict[str, Any]]:
        """
        Wraps each LangGraph node to support:
        1. Pre-agent snapshot capture & post-agent snapshot commit for `supervisor`,
           `fine_tuner_agent`, and `evaluate_agent`.
        2. Cooperative abort checking before and after node execution.
        3. Graceful pause & user context injection immediately after the current node finishes.
        """
        def _wrapped(state: TuningState) -> dict[str, Any]:
            self.check_abort()
            with self._lock:
                self._current_node_name = node_name
            self._emit_event({"event_type": "node_start", "node_name": node_name})

            pre_snapshot = None
            if node_name in self.AGENTIC_NODES:
                pre_snapshot = self._capture_snapshot(node_name, state)

            raw_updates = node_fn(state)
            self.check_abort()

            if raw_updates is state:
                updates: dict[str, Any] = dict(state)
            elif isinstance(raw_updates, dict):
                updates = dict(raw_updates)
            else:
                updates = {}

            # Ensure _start_node resets to default after entry routing
            updates["_start_node"] = "optimize"
            post_state: TuningState = {**state, **updates}

            # Commit pre_snapshot once the graph-level agentic step has completed
            if node_name in self.AGENTIC_NODES and pre_snapshot is not None:
                with self._lock:
                    self._latest_completed_agent_snapshot = pre_snapshot
                self._emit_event({
                    "event_type": "agent_step_completed",
                    "node_name": node_name,
                    "snapshot_id": pre_snapshot["snapshot_id"],
                })

            # Handle Mode 1: Graceful Pause / Inject Context After Current Step
            with self._lock:
                should_pause = self._pause_after_step_event.is_set()
                if should_pause:
                    self._workflow_status = "paused"
                    self._resume_event.clear()

            if should_pause:
                self._emit_event({
                    "event_type": "workflow_paused",
                    "node_name": node_name,
                    "mode": "graceful",
                })
                self._resume_event.wait()
                # Check if the user switched to Mode 2 (Terminate & Rollback) while paused
                self.check_abort()
                with self._lock:
                    self._pause_after_step_event.clear()
                    self._workflow_status = "running"

            prompt_text = self._consume_pending_prompt()
            if prompt_text:
                new_uc = self._build_user_context(node_name, post_state, prompt_text)
                updated_history = list(post_state.get("user_context_history", [])) + [new_uc]
                updates["user_context"] = new_uc
                updates["user_context_history"] = updated_history
                with self._lock:
                    self.latest_user_context = new_uc
                    self.user_context_history = copy.deepcopy(updated_history)
                    self._pause_after_step_event.clear()
                    self._workflow_status = "running"

            return updates

        return _wrapped

    def _route_entry(self, state: TuningState) -> str:
        target = state.get("_start_node", "optimize")
        if target in self.ALL_GRAPH_NODES:
            return target
        return "optimize"

    def _build_graph(self):
        if self._graph is not None:
            return self._graph

        graph = StateGraph(TuningState)

        graph.add_node("__entry_router__", lambda state: {})  # noqa: PyTypeChecker
        graph.add_node("optimize", self._wrap_node("optimize", self._optimize_node))  # noqa: PyTypeChecker
        graph.add_node("supervisor", self._wrap_node("supervisor", self._supervisor_agent_node))  # noqa: PyTypeChecker
        graph.add_node("increase_optimizer_budget", self._wrap_node("increase_optimizer_budget", self._increase_budget_node))  # noqa: PyTypeChecker
        graph.add_node("change_search_space", self._wrap_node("change_search_space", self._change_search_space_node))  # noqa: PyTypeChecker
        graph.add_node("change_weights", self._wrap_node("change_weights", self._change_weights_node))  # noqa: PyTypeChecker
        graph.add_node("change_objective", self._wrap_node("change_objective", self._change_objective_node))  # noqa: PyTypeChecker
        graph.add_node("change_target_metrics", self._wrap_node("change_target_metrics", self._change_target_metrics_node))  # noqa: PyTypeChecker
        graph.add_node("add_cost_penalty", self._wrap_node("add_cost_penalty", self._add_cost_penalty_node))  # noqa: PyTypeChecker
        graph.add_node("next_controller", self._wrap_node("next_controller", self._next_controller_node))  # noqa: PyTypeChecker
        graph.add_node("evaluate_coupled_baseline", self._wrap_node("evaluate_coupled_baseline", self._evaluate_coupled_baseline_node))  # noqa: PyTypeChecker
        graph.add_node("fine_tuner_agent", self._wrap_node("fine_tuner_agent", self._fine_tuner_agent_node))  # noqa: PyTypeChecker
        graph.add_node("fine_tune", self._wrap_node("fine_tune", self._fine_tune_node))  # noqa: PyTypeChecker
        graph.add_node("evaluate_agent", self._wrap_node("evaluate_agent", self._evaluate_agent_node))  # noqa: PyTypeChecker
        graph.add_node("selective_re_design", self._wrap_node("selective_re_design", self._selective_redesign_node))  # noqa: PyTypeChecker
        graph.add_node("finalize_design", self._wrap_node("finalize_design", self._finalize_design_node))  # noqa: PyTypeChecker

        graph.set_entry_point("__entry_router__")
        graph.add_conditional_edges(
            "__entry_router__",
            self._route_entry,
            {node: node for node in self.ALL_GRAPH_NODES},
        )

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
                Decision.ADD_COST_PENALTY: "add_cost_penalty",
                Decision.TERMINATE: "next_controller",
            },
        )

        graph.add_edge("change_search_space", "optimize")
        graph.add_edge("increase_optimizer_budget", "optimize")
        graph.add_edge("change_weights", "optimize")
        graph.add_edge("change_objective", "optimize")
        graph.add_edge("change_target_metrics", "optimize")
        graph.add_edge("add_cost_penalty", "optimize")

        graph.add_conditional_edges(
            "next_controller",
            self._route_after_next_controller,
            {
                "fine_tuner_agent": "evaluate_coupled_baseline",
                "evaluate_agent": "evaluate_agent",
                "optimize": "optimize",
                "end": "finalize_design"
            },
        )

        graph.add_edge("evaluate_coupled_baseline", "fine_tuner_agent")
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
        cont_index = state["controller_index"]
        attempt = state.get("attempt", 0)

        self.orchestrator.set_controller_index(cont_index)

        # Use canonical channel name so all attempts share the same plot
        name = self.generate_controller_name(cont_index)

        # If this is a retry after a Supervisor decision, mark a red dotted boundary line
        if attempt > 0 and state.get("supervisor_decision") is not None:
            decision_val = (
                state["supervisor_decision"].value
                if isinstance(state["supervisor_decision"], Enum)
                else str(state["supervisor_decision"])
            )
            self._emit_supervisor_boundary([name], label=f"Supervisor: {decision_val}")

        final_state = self.orchestrator.design_controller(name, self.llm_agent)

        updates = self.optimizer_state_generation(final_state, attempt + 1, state)
        updates["optimized_controllers"] = self._update_optimized_controllers(
            state, state["loop_index"], cont_index, final_state)

        return updates

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
            "kp": expand_bounds(
                cont.get("kp", 0.0),
                ranges.get("Kp", ranges.get("kp", [-10.0, 10.0])),
                scale["kp"],
            ),
            "ki": expand_bounds(
                cont.get("ki", 0.0),
                ranges.get("Ki", ranges.get("ki", [-10.0, 10.0])),
                scale["ki"],
            ),
            "kd": expand_bounds(
                cont.get("kd", 0.0),
                ranges.get("Kd", ranges.get("kd", [-10.0, 10.0])),
                scale["kd"],
            ),
        }

        self.orchestrator.set_controller_structure(controller_structure)
        return {"param_ranges": cont["param_ranges"]}

    def _increase_budget_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=True)

        action = state["action"]
        supervisor_comment = action.get("comment", "")

        case_study = self.orchestrator.get_case_study()
        case_study["supervisor_comment"] = supervisor_comment
        self.orchestrator.set_case_study(case_study)

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
        supervisor_comment = state["action"].get("comment", "")

        # 1. Fetch current architecture
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        case_study = self.orchestrator.get_case_study()
        loop_index = self.orchestrator.get_loop_index()
        controller_index = self.orchestrator.get_controller_index()

        # 2. Inject the new target objective directly into the active controller
        cont = controller_structure[loop_index]["controllers"][controller_index]
        cont["signal_type"] = new_obj
        case_study["supervisor_comment"] = supervisor_comment

        # 3. Save the modified architecture back to the orchestrator
        self.orchestrator.set_controller_structure(controller_structure)
        self.orchestrator.set_case_study(case_study)

        return state

    def _change_target_metrics_node(self, state: TuningState) -> TuningState:
        """
        Overrides the numeric acceptance thresholds (mse, settling_time,
        overshoot, control_effort) used to evaluate the CURRENT controller.
        """
        self._manage_optimizer_history(state, retain=True)

        new_targets = state["action"]["new_target_metrics"]
        supervisor_comment = state["action"].get("comment", "")
        if not new_targets:
            return {"target_metrics": state.get("target_metrics", {})}

        # 1. Case-study level (feeds the optimizer)
        case_study = self.orchestrator.get_case_study()
        case_study["fixed_targets"] = dict(new_targets)
        case_study["supervisor_comment"] = supervisor_comment
        self.orchestrator.set_case_study(case_study)

        # 2. Controller-structure level (persistent source of truth)
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()

        controller_structure[loop_index]["metrics"] = dict(new_targets)

        self.orchestrator.set_controller_structure(controller_structure)

        return {"target_metrics": new_targets}

    def _add_cost_penalty_node(self, state: TuningState) -> TuningState:
        self._manage_optimizer_history(state, retain=True)

        action = state["action"]
        new_metric = action["new_metric"]
        new_target = action["new_target"]
        supervisor_comment = action.get("comment", "")

        # 1. Update Case Study (Targets & Comment)
        case_study = self.orchestrator.get_case_study()
        case_study["fixed_targets"][new_metric] = new_target
        case_study["supervisor_comment"] = supervisor_comment
        self.orchestrator.set_case_study(case_study)

        # 2. Update Controller Structure (Persistent targets)
        controller_structure = copy.deepcopy(self.orchestrator.get_controller_structure())
        loop_index = self.orchestrator.get_loop_index()

        controller_structure[loop_index]["metrics"][new_metric] = new_target

        self.orchestrator.set_controller_structure(controller_structure)

        # 3. Initialize its weight in run_config if it doesn't exist
        run_config = self.orchestrator.get_run_config()
        if new_metric not in run_config["weights"]:
            run_config["weights"][new_metric] = 1.0
        self.orchestrator.set_run_config(run_config)

        return {
            "target_metrics": case_study["fixed_targets"],
            "weights": run_config["weights"]
        }

    def _evaluate_coupled_baseline_node(self, state: TuningState) -> dict[str, Any]:
        control_block = self.orchestrator.get_active_control_block()
        loop_index = state["loop_index"]
        case_study = self.orchestrator.get_case_study()
        equation = self.orchestrator.get_equation()
        fixed_targets = case_study.get("fixed_targets", {})

        updates = {"optimized_controllers": copy.deepcopy(state.get("optimized_controllers", {}))}

        for idx, cont in enumerate(control_block):
            if "skip" in cont:
                continue

            sim_config = self.orchestrator.get_simulator_config(cont)

            is_bounded = (sim_config["min_ctrl"] != "" and sim_config["max_ctrl"] != "")
            min_ctrl = float(sim_config["min_ctrl"]) if is_bounded else 0.0
            max_ctrl = float(sim_config["max_ctrl"]) if is_bounded else 0.0

            # The inverse equation index targeted by the backend simulator
            target_file_index = len(control_block) - idx - 1

            # Execute physical simulation step to get the fully coupled response
            t, y, ref_val, trim_value, cont_signal, metrics = simulate_system_response(
                equation=equation,
                case_study=case_study,
                controller_structure=cont,
                controller_index=target_file_index,
                Kp=cont["kp"],
                Ki=cont["ki"],
                Kd=cont["kd"],
                is_bounded=is_bounded,
                min_ctrl=min_ctrl,
                max_ctrl=max_ctrl,
                signal_type=sim_config.get("signal_type", "Step"),
                amplitude=sim_config["target"]
            )

            # Compute baseline cost mimicking PSO optimizer logic
            cost = 0.0
            for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
                if metric in metrics and metric in fixed_targets:
                    achieved = metrics[metric]
                    target = fixed_targets[metric]
                    cost += (achieved - target) / max(target, 1e-6)

            # Update the physical control block
            cont["achieved_metrics"] = metrics
            cont["achieved_cost"] = cost

            # Update the LLM's state memory
            agent_friendly_key = f"loop_{loop_index}_controller_{idx}"
            if agent_friendly_key in updates["optimized_controllers"]:
                updates["optimized_controllers"][agent_friendly_key]["achieved_metrics"] = metrics
                updates["optimized_controllers"][agent_friendly_key]["achieved_cost"] = cost
                updates["optimized_controllers"][agent_friendly_key]["gains"] = {
                    "kp": cont["kp"], "ki": cont["ki"], "Kk": cont["kd"]
                }
            else:
                updates["optimized_controllers"][agent_friendly_key] = OptimizedControllerState(
                    gains={"kp": cont["kp"], "ki": cont["ki"], "kd": cont["kd"]},
                    target_metrics=fixed_targets,
                    achieved_metrics=metrics,
                    achieved_cost=cost
                )

        return updates

    def _supervisor_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state, "supervisor")
        optimizer_choice = self.orchestrator.get_run_config()["optimizer_choice"]
        response = self.llm_agent.supervisor_agent(agent_context, optimizer_choice, self.MAX_ATTEMPTS)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"], reason=response["reason"], deciding_agent="Supervisor Agent")

        return {
            "supervisor_decision": decision,
            "action": action,
            "reason": response.get("reason", ""),
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _fine_tuner_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state, "fine_tuner")
        control_block = self.orchestrator.get_active_control_block()
        response = self.llm_agent.fine_tuner_agent(control_block, agent_context, self.MAX_FINE_TUNING_TRIES)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"], reason=response["reason"], deciding_agent="Fine Tuner Agent")

        if decision == decision.INCREASE_OPTIMIZE_BUDGET:
            self._increase_budget(action)

        return {
            "supervisor_decision": decision,
            "action": action,
            "reason": response.get("reason", ""),
            "action_history": state.get("action_history", []) + [{decision: action}],
        }

    def _evaluate_agent_node(self, state: TuningState) -> dict[str, Any]:
        agent_context = self._get_agent_context(state, "evaluate")
        response = self.llm_agent.evaluate_agent(self.orchestrator.get_controller_structure(), agent_context,
                                                 self.MAX_SELECTIVE_RE_DESIGN)

        decision = Decision(response["supervisor_decision"])
        action = Action(**response.get("action", {}), loop_index=state["loop_index"],
                        controller_index=state["controller_index"], reason=response["reason"], deciding_agent="Evaluate Agent")

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
        control_block = self.orchestrator.get_active_control_block()
        rng = state["action"]["fine_tuner_range"]
        attempt = state.get("attempt", 0) + 1
        fixed_targets = self.orchestrator.get_case_study()["fixed_targets"]

        channel_names = []
        for idx, cont in enumerate(control_block):
            chan_name = self.generate_controller_name(idx)
            channel_names.append(chan_name)
            channels.append(
                {
                    "controller_name": chan_name,
                    "fixed_targets": fixed_targets,
                    "param_ranges": {
                        "Kp": [cont["kp"] - rng, cont["kp"] + rng],
                        "Ki": [cont["ki"] - rng, cont["ki"] + rng],
                        "Kd": [cont["kd"] - rng, cont["kd"] + rng],
                    },
                    "simulator_config": self.orchestrator.get_simulator_config(cont),
                }
            )

        # Mark red dotted line before Fine Tuner optimization starts
        self._emit_supervisor_boundary(channel_names, label=f"Fine Tuner (Try {attempt})")

        final_state = self.orchestrator.fine_tune_control_block(channels)

        updates = self.optimizer_state_generation(final_state, attempt, state)
        updates["optimized_controllers"] = self._update_optimized_controllers_fine_tune(state, final_state, channels)
        return updates

    def _next_controller_node(self, state: TuningState) -> dict[str, Any]:
        self._manage_optimizer_history(state, retain=False)

        self.orchestrator.set_run_config(copy.deepcopy(self.initial_run_config))

        next_controller = state["controller_index"] + 1

        self.orchestrator.set_controller_index(next_controller)

        return {
            "controller_index": next_controller,
            "attempt": 0,
            "supervisor_decision": None,
            "optimizer_max_run_time": self.initial_run_config["max_wall_clock"],
            "reason": "",
            "action": None
        }

    def _selective_redesign_node(self, state: TuningState) -> dict[str, Any]:
        action = state["action"]["selected_controller"]
        attempt = state["attempt"] + 1

        inner_idx = int(action["inner_controller_index"])
        inner_loop_idx = int(action["inner_loop_index"])
        outer_idx = int(action["outer_controller_index"])
        outer_loop_idx = int(action["outer_loop_index"])

        controller_structure = self.orchestrator.get_controller_structure()

        outer_key = f"loop_{outer_loop_idx}_controller_{outer_idx}"
        outer_state = state.get("optimized_controllers", {}).get(outer_key, {})
        target_metrics = (
            outer_state.get("target_metrics")
            or controller_structure[outer_loop_idx].get("metrics")
            or self.orchestrator.get_case_study().get("fixed_targets", {})
        )

        case_study = self.orchestrator.get_case_study()
        case_study["fixed_targets"] = dict(target_metrics)
        self.orchestrator.set_case_study(case_study)

        outer_controllers = controller_structure[outer_loop_idx]["controllers"]
        # Use canonical controlled_variable name so it continues on the existing channel graph
        name = outer_controllers[outer_idx]["controlled_variable"]

        self._emit_supervisor_boundary([name], label=f"Evaluate Agent Re-design (Try {attempt})")

        final_state = self.orchestrator.selective_redesign_controller(
            inner_loop_idx, inner_idx, outer_loop_idx, outer_idx, name, self.llm_agent
        )

        updates = self.optimizer_state_generation(final_state, attempt, state)
        updates["optimized_controllers"] = self._update_optimized_controllers_selective(
            state, inner_loop_idx, inner_idx, outer_loop_idx, outer_idx, final_state
        )

        return updates
    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_optimize_skip(self, state: TuningState) -> str:
        index = state["controller_index"]
        if "skip" in self.orchestrator.get_active_control_block()[index]:
            return "skip"
        return "supervisor"

    def _route_supervisor_decision(self, state: TuningState) -> Decision:
        attempt = state["attempt"]
        if attempt >= MuloSupervisorAgent.MAX_ATTEMPTS:
            return Decision.TERMINATE
        return state["supervisor_decision"]

    def _route_after_next_controller(self, state: TuningState) -> str:
        control_block = self.orchestrator.get_active_control_block()
        if len(control_block) == 1:
            if self.orchestrator.get_loop_index() >= len(self.orchestrator.get_controller_structure()) - 1:
                return "evaluate_agent"
            return "end"
        elif len(control_block) == state["controller_index"]:
            skip_fine_tune = all(["skip" in control_block[i] for i in range(len(control_block))])
            if skip_fine_tune:
                return "end"
            return "fine_tuner_agent"

        return "optimize"

    def _route_after_fine_tuner_agent(self, state: TuningState) -> str:
        decision = state["supervisor_decision"]
        attempt = state["attempt"]

        if decision == Decision.TERMINATE or attempt >= MuloSupervisorAgent.MAX_FINE_TUNING_TRIES:
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

        action_params = {
            "scale": {"kp": 10.0, "ki": 10.0, "kd": 10.0},
            "new_population_size": 150,
            "new_generation_number": 75,
            "new_max_time": 120,
            "new_weights": {"mse": 1.0, "settling_time": 1.0, "overshoot": 1.0, "control_effort": 1.0},
            "new_objective": "Pulse"
        }

        if achieved_cost < target_metrics.get("baseline_cost", 200) and abs(metrics["mse"]) < target_metrics["mse"]:
            if attempt <= 2:
                if metrics["settling_time"] < target_metrics["settling_time"] and metrics["overshoot"] < target_metrics["overshoot"]:
                    return {
                        "supervisor_decision": Decision.PROCEED.value,
                        "action": action_params,
                        "reason": "All stringent metrics satisfied."
                    }
            else:
                return {
                    "supervisor_decision": Decision.PROCEED.value,
                    "action": action_params,
                    "reason": "Relaxed baseline metrics satisfied after multiple attempts."
                }

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

        return {
            "supervisor_decision": Decision.EXTEND_SEARCH_SPACE.value,
            "action": action_params,
            "reason": "Continuing search space expansion."
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emit_supervisor_boundary(self, controller_names: List[str], label: str = "") -> None:
        """Marks the current NFE position on the target channel(s) with a red dotted line."""
        for name in controller_names:
            self._emit_event({
                "event_type": "supervisor_boundary",
                "controller_name": name,
                "label": label,
            })

    def _update_optimized_controllers(self, state: TuningState, loop_idx: int, cont_idx: int, final_state: Dict[str, Any]) -> Dict[str, OptimizedControllerState]:
        memory = copy.deepcopy(state.get("optimized_controllers", {}))
        agent_friendly_key = f"loop_{loop_idx}_controller_{cont_idx}"

        if "best_result" in final_state:
            new_cost = final_state["best_result"]["achieved_cost"]
            if agent_friendly_key not in memory or new_cost < memory[agent_friendly_key]["achieved_cost"]:
                memory[agent_friendly_key] = OptimizedControllerState(
                    gains=final_state["best_result"]["gains"],
                    target_metrics=final_state.get("tuning_specs", {}).get("target_metrics", {}),
                    achieved_metrics=final_state["best_result"]["achieved_metrics"],
                    achieved_cost=new_cost
                )
        return memory

    def _update_optimized_controllers_selective(self, state: TuningState, inner_loop_idx: int, inner_idx: int,
                                                outer_loop_idx: int, outer_idx: int, final_state: Dict[str, Any]) -> Dict[str, OptimizedControllerState]:
        memory = copy.deepcopy(state.get("optimized_controllers", {}))
        tuned_key = f"loop_{inner_loop_idx}_controller_{inner_idx}"
        optimized_key = f"loop_{outer_loop_idx}_controller_{outer_idx}"

        if "best_result" in final_state:
            new_cost = final_state["best_result"]["achieved_cost"]
            prev_cost = memory.get(optimized_key, {}).get("achieved_cost", float("inf"))

            if new_cost < prev_cost:
                controller_structure = self.orchestrator.get_controller_structure()

                if tuned_key not in memory:
                    inner_loop = controller_structure[inner_loop_idx]
                    inner_cont = inner_loop["controllers"][inner_idx]
                    memory[tuned_key] = OptimizedControllerState(
                        gains=final_state["best_result"]["gains"],
                        target_metrics=inner_loop.get("metrics", {}),
                        achieved_metrics=inner_cont.get("achieved_metrics", {}),
                        achieved_cost=inner_cont.get("achieved_cost", float("inf")),
                    )
                else:
                    memory[tuned_key]["gains"] = final_state["best_result"]["gains"]

                if optimized_key not in memory:
                    outer_loop = controller_structure[outer_loop_idx]
                    outer_cont = outer_loop["controllers"][outer_idx]
                    memory[optimized_key] = OptimizedControllerState(
                        gains={"kp": outer_cont.get("kp", 0.0), "ki": outer_cont.get("ki", 0.0), "kd": outer_cont.get("kd", 0.0)},
                        target_metrics=outer_loop.get("metrics", {}),
                        achieved_metrics=final_state["best_result"]["achieved_metrics"],
                        achieved_cost=new_cost,
                    )
                else:
                    memory[optimized_key]["achieved_cost"] = new_cost
                    memory[optimized_key]["achieved_metrics"] = final_state["best_result"]["achieved_metrics"]

        return memory

    def _update_optimized_controllers_fine_tune(self, state: TuningState, final_state: Dict[str, Any], channels: List[Any]) -> Dict[str, OptimizedControllerState]:
        memory = copy.deepcopy(state.get("optimized_controllers", {}))

        if "best_result" in final_state:
            new_cost = final_state["best_result"]["achieved_cost"]
            all_gains = final_state["best_result"]["gains"]

            for idx, ch in enumerate(channels):
                agent_friendly_key = f"loop_{state['loop_index']}_controller_{idx}"

                if agent_friendly_key not in memory or new_cost < memory[agent_friendly_key]["achieved_cost"]:
                    controller_gains = all_gains[idx] if isinstance(all_gains, list) else all_gains
                    memory[agent_friendly_key] = OptimizedControllerState(
                        gains=controller_gains,
                        target_metrics=ch["fixed_targets"],
                        achieved_metrics=final_state["best_result"]["achieved_metrics"],
                        achieved_cost=new_cost
                    )
        return memory

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

    def _get_agent_context(self, state: TuningState, agent_type: str = "supervisor") -> dict:
        """Filters and compacts the graph state to provide targeted context for specific LLMs."""
        if agent_type == "supervisor":
            keys_to_keep = self.SUPERVISOR_CONTEXT_KEYS
        elif agent_type == "fine_tuner":
            keys_to_keep = self.FINE_TUNER_CONTEXT_KEYS
        elif agent_type == "evaluate":
            keys_to_keep = self.EVALUATE_CONTEXT_KEYS
        else:
            keys_to_keep = set(state.keys()) - {"_start_node", "supervisor_decision", "reason", "action", "error"}

        filtered_state = {k: v for k, v in state.items() if k in keys_to_keep}

        def compact_context(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, float):
                if abs(obj) > 0 and abs(obj) < 1e-4:
                    return float(f"{obj:.2e}")
                return round(obj, 4)
            elif isinstance(obj, dict):
                return {(k.value if isinstance(k, Enum) else k): compact_context(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [compact_context(v) for v in obj]
            return obj

        return compact_context(filtered_state)

    def optimizer_state_generation(self, final_state: Dict[str, Any], attempt: int, state: TuningState) -> Dict[str, Any]:
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
            "optimizer_agent_decision_trail": self.feedback_history.get("decision_trail", [])
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

    def generate_controller_name(self, j, add_info=""):
        control_block = self.orchestrator.get_active_control_block()
        base_name = control_block[j]["controlled_variable"]
        return f"{base_name} {add_info}".strip() if add_info else base_name

    def get_graph(self):
        self._build_graph()
        return self._graph

    def get_orchestrator(self) -> MuloTuningOrchestrator:
        return self.orchestrator