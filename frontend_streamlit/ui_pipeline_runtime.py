"""Background pipeline execution and event capture shared by the chat workspace."""
from __future__ import annotations

from collections import deque
import datetime as dt
from pathlib import Path
import queue
import sys
import threading
from typing import Any, Dict, Optional

from backend_core.AgentSysID.pipeline import SysIDOptions, SysIDResult, run_pipeline
from backend_core.AgentSysID.utils import reset_stop_flag
from backend_core.AgentSysID.agents.run_evidence import write_json, redact
try:
    from . import ui_activity as activity
except ImportError:
    import ui_activity as activity

class _ThreadRouter:
    """Routes the worker thread's prints to a queue; other threads pass through."""

    def __init__(self, target_thread: int, sink: "queue.Queue[str]", original):
        self._target = target_thread
        self._sink = sink
        self._original = original
        self.captured = deque()
        self.captured_size = 0

    def write(self, text: str) -> int:
        if threading.get_ident() == self._target:
            if text:
                self._sink.put(text)
                self.captured.append(text)
                self.captured_size += len(text)
                while self.captured_size > 2_000_000 and len(self.captured) > 1:
                    self.captured_size -= len(self.captured.popleft())
            return len(text)
        return self._original.write(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

class PipelineRunner:
    """Runs ``run_pipeline`` on a thread, exposing logs, events and the result."""

    def __init__(self, options: SysIDOptions):
        self.options = options
        self.logs: "queue.Queue[str]" = queue.Queue()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.result: Optional[SysIDResult] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None
        self.run_dir: Optional[Path] = None
        self.last_stage = "Preparing run"
        self._review_condition = threading.Condition()
        self._review_sequence = 0
        self._review_decision = None
        self.pending_checkpoint = None

    def _on_event(self, kind: str, payload: Dict[str, Any]) -> None:
        if kind == "run_started":
            self.run_dir = Path(payload["run_dir"])
        elif kind == "stage":
            self.last_stage = payload["name"]
        self.events.put((kind, {**payload, "_timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}))

    def _run(self) -> None:
        original = sys.stdout
        router = _ThreadRouter(threading.get_ident(), self.logs, original)
        sys.stdout = router
        try:
            self.result = run_pipeline(
                self.options,
                on_event=self._on_event,
                install_signal_handler=False,
                human_review=self.request_human_review if getattr(self.options, "human_in_the_loop", False) else None,
            )
        except Exception as exc:  # noqa: BLE001 - surface it in the UI
            import traceback

            self.error = f"{exc}\n\n{traceback.format_exc()}"
            self._on_event("error", {"message": str(exc)})
        finally:
            sys.stdout = original
            run_dir = self.run_dir or (self.result.run_dir if self.result else None)
            if run_dir:
                try:
                    (Path(run_dir) / "runtime_console.log").write_text(redact("".join(router.captured)), encoding="utf-8")
                    write_json(Path(run_dir) / "diagnostic_state.json", {
                        "status": self.result.status if self.result else "failed",
                        "last_stage": self.last_stage, "error": self.error,
                        "message": self.result.message if self.result else "The pipeline raised an exception.",
                    })
                except OSError:
                    pass
            self._on_event("finished", {})

    def start(self) -> None:
        reset_stop_flag()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_human_review(self, phase: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pause the worker until the client chooses how to steer this run."""
        with self._review_condition:
            self._review_sequence += 1
            checkpoint_id = f"{phase}_{self._review_sequence}"
            checkpoint = {"id": checkpoint_id, "phase": phase, **payload}
            self.pending_checkpoint = checkpoint
            self._review_decision = None
            self._on_event("human_checkpoint", checkpoint)
            while self._review_decision is None:
                self._review_condition.wait()
            decision = dict(self._review_decision)
            self.pending_checkpoint = None
            self._review_decision = None
            self._on_event("human_checkpoint_resolved", {
                "id": checkpoint_id, "phase": phase, "action": decision.get("action", "continue")
            })
            return decision

    def respond_to_checkpoint(self, action: str) -> bool:
        """Resolve the currently visible checkpoint; return false if it is stale."""
        with self._review_condition:
            if self.pending_checkpoint is None or self._review_decision is not None:
                return False
            self._review_decision = {"action": str(action)}
            self._review_condition.notify_all()
            return True

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

def drain(runner: "PipelineRunner", state) -> bool:
    """Move everything the worker has produced into session state.

    Returns True when the run just finished, so the caller can navigate.
    """
    chunks = []
    while True:
        try:
            chunks.append(runner.logs.get_nowait())
        except queue.Empty:
            break
    if chunks:
        state["log"] = (state["log"] + "".join(chunks))[-60000:]

    just_finished = False
    while True:
        try:
            kind, payload = runner.events.get_nowait()
        except queue.Empty:
            break
        activity.record(state.setdefault("activity", []), kind, payload)
        if kind == "stage":
            state["stage"] = payload["name"]
        elif kind == "human_checkpoint":
            state["checkpoint"] = payload
        elif kind == "human_checkpoint_resolved":
            state["checkpoint"] = None
            state["checkpoint_action"] = payload.get("action")
        elif kind == "progress":
            state["progress"] = float(payload["value"])
        elif kind == "cycle":
            state["history"].append(payload)
        elif kind == "latency":
            for row in reversed(state["history"]):
                if row.get("cycle") == payload["cycle"]:
                    row["latency"] = payload["latency_ms"]
                    break
        elif kind == "critic":
            state["critic"].append(payload)
        elif kind == "done":
            state["result"] = payload["result"]
            just_finished = True

    if runner.result is not None and state["result"] is None:
        state["result"] = runner.result
        activity.record(state.setdefault("activity", []), "done", {"result": runner.result})
        just_finished = True
    result = state["result"]
    if not runner.running and not state.get("activity_saved"):
        run_dir = result.run_dir if result is not None else runner.run_dir
        if run_dir:
            saved = activity.save(run_dir, state.get("activity", []))
            state["activity_saved"] = True
            state["activity_save_error"] = not saved
    return just_finished
