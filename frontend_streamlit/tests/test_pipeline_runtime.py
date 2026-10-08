import threading

from backend_core.AgentSysID.pipeline import SysIDOptions
from frontend_streamlit import ui_pipeline_runtime as runtime


def test_client_checkpoint_pauses_worker_then_resumes_with_feedback():
    runner = runtime.PipelineRunner(SysIDOptions(data_path="unused.csv", human_in_the_loop=True))
    answer = []
    completed = threading.Event()

    def wait_for_client():
        answer.append(runner.request_human_review("initializer", {"config": {"hidden_layers": [32]}}))
        completed.set()

    worker = threading.Thread(target=wait_for_client)
    worker.start()
    queued_event = runner.events.get(timeout=2)
    assert queued_event[0] == "human_checkpoint"
    runner.events.put(queued_event)
    state = {"log": "", "history": [], "result": None}
    runtime.drain(runner, state)
    assert state["checkpoint"]["phase"] == "initializer"
    assert not completed.is_set()

    assert runner.respond_to_checkpoint("compact") is True
    worker.join(timeout=2)
    assert completed.is_set()
    assert answer == [{"action": "compact"}]

    runtime.drain(runner, state)
    assert state["checkpoint"] is None
    assert state["checkpoint_action"] == "compact"


def test_stale_checkpoint_action_is_ignored():
    runner = runtime.PipelineRunner(SysIDOptions(data_path="unused.csv", human_in_the_loop=True))
    assert runner.respond_to_checkpoint("widen") is False
