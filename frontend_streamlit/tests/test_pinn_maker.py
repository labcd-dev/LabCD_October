import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frontend_streamlit import conversation_core as core
from frontend_streamlit import pinn_maker as maker
from frontend_streamlit import ui_conversation as ui
from backend_core.AgentSysID.model.pinn import load_analytical_xdot


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path / "runs")
    return tmp_path


class FakeClient:
    settings = SimpleNamespace(model="configured-test-model")

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, json.loads(user)))
        return json.dumps(self.response)


def equation_reply(expression="-s0 + a0"):
    return {"status": "ready", "message": "I mapped the supplied first-order equation.",
            "equations": [{"state": "s_position", "expression": expression}], "assumptions": []}


def attached_chat():
    chat = core.new_chat()
    rows = [f"{i * 0.05},{np.exp(-i * 0.05)},{0.2 + 0.01 * i}" for i in range(80)]
    raw = ("time,s_position,a_force\n" + "\n".join(rows) + "\n").encode()
    chat["dataset"] = core.inspect_upload("measurements.csv", raw, chat["id"])
    return chat


def attached_pitch_yaw_chat():
    chat = core.new_chat()
    rows = [
        f"{i * 0.01},{0.01 * i},{-0.02 * i},{0.03 * i},{-0.04 * i},{1.0 if i < 40 else 0.0},{0.5}"
        for i in range(80)
    ]
    raw = ("time,s_pitch,s_yaw,s_dpitch,s_dyaw,a_u1,a_u2\n" + "\n".join(rows) + "\n").encode()
    chat["dataset"] = core.inspect_upload("pitch_yaw.csv", raw, chat["id"])
    return chat


def source_for(chat, source=b"function xdot = model(x,u)\nxdot = -x + u;\nend\n", name="model.m"):
    source = maker.save_source(chat["id"], name, source)
    source["chat_id"] = chat["id"]
    chat["pinn_source"] = source
    return source


def test_detects_equation_preparation_but_not_pinn_explanations():
    assert maker.is_raw_equation_request("xdot_position = -position + force")
    assert maker.is_raw_equation_request("Please prepare this for PINN")
    assert not maker.is_raw_equation_request("Explain how PINN works")
    assert maker.is_raw_equation_request("yes, map it", has_pending_source="needs_clarification")
    assert not maker.is_raw_equation_request("yes, map it", has_pending_source="ready")
    assert not maker.is_raw_equation_request("What should I try next?", has_pending_source="ready")
    assert not maker.is_raw_equation_request("How do I use PINN?", has_pending_source="ready")
    assert not maker.is_raw_equation_request("How do I prepare equations for PINN?", has_pending_source="ready")
    assert not maker.is_raw_equation_request("Why is this PINN equation wrong?", has_pending_source="ready")
    assert maker.is_raw_equation_request("Can you prepare this equation for PINN?", has_pending_source="ready")
    assert not maker.is_raw_equation_request("What should I try next?", has_pending_source="needs_clarification")
    assert maker.is_raw_equation_request("pitch is s0 and yaw is s1", has_pending_source="needs_clarification")
    assert not maker.is_raw_equation_request("Do you use PINN in this run?", has_pending_source=True)
    assert not maker.is_raw_equation_request("Was PINN enabled in this run?", has_pending_source="needs_clarification")
    assert maker.is_pinn_usage_question("Do you use PINN in this run?")
    assert maker.is_pinn_usage_question("Is PINN on?")


def test_empty_provider_explanation_does_not_break_valid_equation(storage):
    chat = attached_chat()
    source_for(chat)
    response = equation_reply()
    response["message"] = ""

    answer = maker.prepare_equation(chat, "Prepare this equation", client=FakeClient(response))

    assert answer["status"] == "ready"
    assert answer["generated_artifact"] is not None
    assert answer["answer"].startswith("I mapped the supplied equations")
    assert "checked that the equations return finite values" in answer["answer"]
    assert "\n\n\n" not in answer["answer"]


def test_empty_clarification_message_gets_a_specific_fallback():
    response = maker._provider_response(json.dumps({
        "status": "clarification", "message": "", "equations": [], "assumptions": [],
    }))
    assert response.message.startswith("Could you clarify how")


def test_safe_expression_rejects_python_and_unavailable_columns():
    with pytest.raises(ValueError, match="safe mathematical"):
        maker._parse_expression("__import__('os').system('whoami')", 1, 1)
    with pytest.raises(ValueError, match="unknown symbol"):
        maker._parse_expression("s0 + a1", 1, 1)
    with pytest.raises(ValueError, match="Powers"):
        maker._parse_expression("s0 ** a0", 1, 1)


def test_m_file_is_mapped_checked_and_used_by_run_options(storage):
    chat = attached_chat()
    raw = b"function xdot = model(x,u)\nxdot = -x + u;\nend\n"
    source = source_for(chat, raw)
    client = FakeClient(equation_reply())

    answer = maker.prepare_equation(chat, "Prepare this MATLAB equation for PINN", client=client)

    artifact = answer["generated_artifact"]
    assert answer["status"] == "ready"
    assert source["status"] == "pending"  # the job snapshot is immutable; UI commits readiness on completion
    assert (storage / "uploads" / chat["id"] / "pinn_sources" / (hashlib.sha256(raw).hexdigest()[:16] + "_model.m")).read_bytes() == raw
    assert client.calls[0][1]["dataset"]["states"][0]["alias"] == "s0"
    assert client.calls[0][1]["dataset"]["inputs"][0]["alias"] == "a0"

    chat["pinn_equation"] = artifact
    chat["settings"]["use_pinn"] = True
    assert maker.is_equation_ready(chat)
    function = load_analytical_xdot(artifact["path"])
    states = torch.tensor([[2.0], [-0.5]])
    actions = torch.tensor([[3.0], [1.0]])
    torch.testing.assert_close(function(states, actions), torch.tensor([[1.0], [1.5]]))
    options = core.make_options(chat)
    assert options.use_pinn is True
    assert options.pinn_equation_file == artifact["path"]


def test_constants_and_math_functions_return_batch_tensors(storage):
    chat = attached_chat()
    source_for(chat, name="equations.txt")
    client = FakeClient(equation_reply("2.0 * cos(s0) + sin(a0)"))
    artifact = maker.prepare_equation(chat, "Use these equations", client=client)["generated_artifact"]
    fn = load_analytical_xdot(artifact["path"])
    output = fn(torch.zeros((4, 1)), torch.zeros((4, 1)))
    assert tuple(output.shape) == (4, 1)
    torch.testing.assert_close(output, torch.full((4, 1), 2.0))


def test_python_pytorch_pinn_assignments_are_safely_translated_without_api(storage):
    chat = attached_pitch_yaw_chat()
    raw = b'''this is my PINN
pitch = states[:, 0]
    yaw = states[:, 1]  # not used in the continuous-time ODEs
    dpitch = states[:, 2]
    dyaw = states[:, 3]

    u1 = actions[:, 0]
    u2 = actions[:, 1]

    # Physical parameters from the MATLAB plant with the stated adjustments.
    m = 1.3872 * 1.08
    g = 9.81
    B_p = 0.8 * 0.92
    B_y = 0.318 * 1.10
    K_pp = 0.2040 * 0.93
    K_yy = 0.0720 * 1.07
    K_py = 0.0068 * 1.09
    K_yp = 0.0219 * 0.91
    J_p = 0.0178 * 1.06
    J_y = 0.0084 * 0.94
    l_cm = 0.186 * 1.05
    J_Tp = J_p + m * l_cm**2
    J_Ty = J_y + m * l_cm**2
    Tp = K_pp * u1 + K_py * u2
    Ty = K_yp * u1 + K_yy * u2

    xdot_pitch = dpitch
    xdot_yaw = dyaw
    xdot_dpitch = (Tp - B_p * dpitch - m * g * l_cm * torch.sin(pitch)) / J_Tp
    xdot_dyaw = (Ty - B_y * dyaw) / J_Ty

    physics_xdot[:, 0] = xdot_pitch
    physics_xdot[:, 1] = xdot_yaw
    physics_xdot[:, 2] = xdot_dpitch
    physics_xdot[:, 3] = xdot_dyaw
'''
    source = source_for(chat, raw, "pitch_yaw_pinn.txt")

    class NeverCallClient:
        settings = SimpleNamespace(model="must-not-be-used")

        def complete(self, *_args):
            raise AssertionError("Python tensor PINN source should use the constrained local translator")

    answer = maker.prepare_equation(chat, "Prepare this Python PINN", source=source, client=NeverCallClient())

    artifact = answer["generated_artifact"]
    assert answer["status"] == "ready"
    assert answer["model"] == "local safe Python equation translator"
    assert "states[:, i]" in answer["answer"]
    assert "uncertainty comments are not sampled" in answer["answer"]
    assert Path(source["path"]).read_bytes() == raw
    function = load_analytical_xdot(artifact["path"])
    state = torch.tensor([[0.2, -0.1, 0.3, -0.4], [-0.3, 0.2, -0.1, 0.25]])
    action = torch.tensor([[1.0, 0.5], [0.0, 0.5]])
    actual = function(state, action)

    m = 1.3872 * 1.08
    g = 9.81
    B_p, B_y = 0.8 * 0.92, 0.318 * 1.10
    K_pp, K_yy = 0.2040 * 0.93, 0.0720 * 1.07
    K_py, K_yp = 0.0068 * 1.09, 0.0219 * 0.91
    J_p, J_y, l_cm = 0.0178 * 1.06, 0.0084 * 0.94, 0.186 * 1.05
    torque_pitch = K_pp * action[:, 0] + K_py * action[:, 1]
    torque_yaw = K_yp * action[:, 0] + K_yy * action[:, 1]
    expected = torch.stack([
        state[:, 2],
        state[:, 3],
        (torque_pitch - B_p * state[:, 2] - m * g * l_cm * torch.sin(state[:, 0])) / (J_p + m * l_cm**2),
        (torque_yaw - B_y * state[:, 3]) / (J_y + m * l_cm**2),
    ], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_python_pinn_translator_rejects_unsafe_statements_and_bad_indices():
    unsafe = "pitch = states[:, 0]\nphysics_xdot[:, 0] = __import__('os').system('whoami')"
    with pytest.raises(ValueError, match="safe PINN math|function outside"):
        maker._python_equations(unsafe, ["s_pitch"], [])

    out_of_range = "pitch = states[:, 1]\nphysics_xdot[:, 0] = pitch"
    with pytest.raises(ValueError, match="confirmed dataset has 1 state"):
        maker._python_equations(out_of_range, ["s_pitch"], [])

    missing = "pitch = states[:, 0]\nphysics_xdot[:, 0] = pitch"
    with pytest.raises(ValueError, match="missing.*state index/indices 1"):
        maker._python_equations(missing, ["s_pitch", "s_yaw"], [])


def test_clarification_does_not_create_or_enable_an_equation(storage):
    chat = attached_chat()
    source_for(chat)
    client = FakeClient({"status": "clarification", "message": "Which column represents angular position?",
                         "equations": [], "assumptions": []})

    answer = maker.prepare_equation(chat, "Prepare this", client=client)

    assert answer["status"] == "clarification"
    assert answer["generated_artifact"] is None
    assert not list((storage / "uploads" / chat["id"] / "pinn_equations").glob("*.py"))
    assert chat["pinn_source"]["status"] == "needs_clarification"


def test_every_state_needs_one_equation_and_nonfinite_rows_are_rejected(storage):
    chat = attached_chat()
    source_for(chat)
    incomplete = FakeClient({"status": "ready", "message": "Ready", "equations": [], "assumptions": []})
    with pytest.raises(ValueError, match="exactly one equation"):
        maker.prepare_equation(chat, "Use this", client=incomplete)

    invalid_math = FakeClient(equation_reply("log(-abs(s0))"))
    with pytest.raises(ValueError, match="non-finite derivative"):
        maker.prepare_equation(chat, "Use this", client=invalid_math)


def test_equation_is_stale_when_dataset_hash_or_file_changes(storage):
    chat = attached_chat()
    source_for(chat)
    answer = maker.prepare_equation(chat, "Use this", client=FakeClient(equation_reply()))
    chat["pinn_equation"] = answer["generated_artifact"]
    assert maker.is_equation_ready(chat)
    Path(answer["generated_artifact"]["path"]).write_text("changed", encoding="utf-8")
    assert not maker.is_equation_ready(chat)


def test_raw_equations_in_chat_start_mapping_and_appear_as_a_source_file(storage, monkeypatch):
    chat = attached_chat()
    core.REGISTRY = core.JobRegistry()
    started = {}

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["snapshot"] = snapshot
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)
    raw = "Use these equations for PINN:\nxdot_position = -position + force"
    ui._submit(chat, raw, [], hooks={})

    assert started["started"]
    assert chat["pinn_source"]["status"] == "processing"
    assert chat["messages"][0]["attachment"]["kind"] == "pinn_source"
    source_item = next(item for item in ui._file_items(chat) if item["kind"] == "code")
    assert Path(source_item["path"]).read_text(encoding="utf-8") == raw


def test_matlab_upload_is_saved_and_waits_for_dataset(storage):
    chat = core.new_chat()

    class Upload:
        name = "plant_model.m"

        @staticmethod
        def getvalue():
            return b"function dx = plant_model(x,u)\ndx = -x + u;\nend\n"

    ui._submit(chat, "", [Upload()], hooks={})

    assert chat["pinn_source"]["name"] == "plant_model.m"
    assert chat["messages"][0]["attachment"]["kind"] == "pinn_source"
    assert "Attach the measurements" in chat["messages"][-1]["content"]
    assert "automatically" in chat["messages"][-1]["content"]
    assert ui._file_items(chat)[0]["kind"] == "code"


def test_saved_matlab_source_prepares_automatically_when_dataset_is_attached(storage, monkeypatch):
    chat = core.new_chat()
    core.REGISTRY = core.JobRegistry()
    source_upload = SimpleNamespace(
        name="plant_model.m",
        getvalue=lambda: b"function dx = plant_model(x,u)\ndx = -x + u;\nend\n")
    ui._submit(chat, "", [source_upload], hooks={})
    assert chat["pinn_source"]["status"] == "pending"
    started = {}

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["snapshot"] = snapshot
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)
    csv = ("time,s_position,a_force\n" + "\n".join(
        f"{index / 10},{index / 20},{index % 7}" for index in range(80))).encode()
    dataset_upload = SimpleNamespace(name="measurements.csv", getvalue=lambda: csv)

    ui._submit(chat, "", [dataset_upload], hooks={})

    assert started["started"]
    assert "confirmed state and input columns" in started["question"]
    assert chat["pinn_source"]["status"] == "processing"
    assert chat["dataset"]["ready"]
    assert chat["messages"][-1]["attachment"]["name"] == "measurements.csv"


def test_saved_source_prepares_automatically_for_pasted_dataset(storage, monkeypatch):
    chat = core.new_chat()
    core.REGISTRY = core.JobRegistry()
    source_for(chat, name="plant_model.m")
    started = {}

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)
    table = "time,s_position,a_force\n" + "\n".join(
        f"{index / 10},{index / 20},{index % 7}" for index in range(80))

    ui._submit(chat, table, [], hooks={})

    assert started["started"]
    assert "confirmed state and input columns" in started["question"]
    assert chat["dataset"]["source_kind"] == "pasted_table"
    assert chat["pinn_source"]["status"] == "processing"


def test_saved_source_prepares_after_client_confirms_column_roles(storage, monkeypatch):
    chat = core.new_chat()
    core.REGISTRY = core.JobRegistry()
    source_for(chat, name="plant_model.m")
    raw = ("t (s),u(k),y(k)\n" + "\n".join(
        f"{index / 10},{index % 7},{index / 20}" for index in range(80))).encode()
    chat["dataset"] = core.inspect_upload("measurements.csv", raw, chat["id"])
    assert chat["dataset"]["column_review_pending"]
    started = {}

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)

    ui._finish_column_review(chat, ["y(k)"], ["u(k)"], time_column="t (s)")

    assert started["started"]
    assert chat["dataset"]["ready"]
    assert chat["dataset"]["columns"] == ["time", "a_u_k", "s_y_k"]
    assert chat["pinn_source"]["status"] == "processing"


def test_yes_column_confirmation_is_not_mistaken_for_pinn_command(storage, monkeypatch):
    chat = core.new_chat()
    core.REGISTRY = core.JobRegistry()
    source_for(chat, name="plant_model.m")
    raw = ("time,input,output\n" + "\n".join(
        f"{index / 10},{index % 7},{index / 20}" for index in range(80))).encode()
    chat["dataset"] = core.inspect_upload("measurements.csv", raw, chat["id"])
    assert chat["dataset"]["column_review_pending"]
    started = {}

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)

    ui._submit(chat, "yes, that's ok", [], hooks={})

    assert started["started"]
    assert chat["dataset"]["columns"] == ["time", "a_input", "s_output"]
    assert chat["pinn_source"]["status"] == "processing"


def test_pinn_usage_question_reads_saved_run_and_does_not_start_equation_job(storage, monkeypatch):
    chat = attached_chat()
    source_for(chat)
    run_dir = storage / "runs" / "completed"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "architecture": "LSTM", "use_pinn": False,
    }), encoding="utf-8")
    chat["run_dir"] = str(run_dir)
    chat["settings"]["use_pinn"] = True  # The saved run manifest is authoritative.
    core.REGISTRY = core.JobRegistry()

    ui._submit(chat, "Do you use PINN in this run?", [], hooks={})

    assert "No." in chat["messages"][-1]["content"]
    assert "data-driven" in chat["messages"][-1]["content"]
    assert "physics loss was used" in chat["messages"][-1]["content"]
    assert chat["id"] not in core.REGISTRY.conversations


def test_ordinary_run_question_does_not_reattach_ready_pinn_source(storage, monkeypatch):
    chat = attached_chat()
    source_for(chat)
    chat["pinn_source"]["status"] = "ready"
    chat["run_dir"] = str(storage / "runs" / "completed")
    core.REGISTRY = core.JobRegistry()
    started = {}

    class CapturedConversationJob:
        running = True
        answer = None
        error = None

        def __init__(self, snapshot, question, purpose="question"):
            started.update(snapshot=snapshot, question=question, purpose=purpose)

        def start(self):
            started["started"] = True

    class UnexpectedPINNJob:
        def __init__(self, *_args):
            raise AssertionError("An ordinary run question must not start PINN preparation")

    monkeypatch.setattr(ui.agent, "ConversationJob", CapturedConversationJob)
    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", UnexpectedPINNJob)

    ui._submit(chat, "What should I try next?", [], hooks={})

    assert started["started"]
    assert started["purpose"] == "question"
    assert started["question"] == "What should I try next?"
    assert "attachment" not in chat["messages"][-1]
