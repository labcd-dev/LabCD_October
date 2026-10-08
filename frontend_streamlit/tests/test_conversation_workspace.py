import json
import queue
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

from frontend_streamlit import conversation_core as core, ui_conversation as ui, ui_pipeline_runtime as runtime

APP = str(Path(__file__).resolve().parents[1] / "agent_sysid_app.py")


@pytest.fixture
def app_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(core, "REGISTRY", core.JobRegistry())
    return tmp_path


def new_app():
    return AppTest.from_file(APP, default_timeout=30).run()


def ready_chat():
    chat = core.new_chat()
    data = ("time,s_position,a_force\n" + "\n".join(f"{i/10},{i/20},{i%7}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("oscillator.csv", data, chat["id"])
    return chat


def flattened_step_response(rows=41):
    values = []
    for index in range(rows):
        state = 1.0 if index < 30 else 0.0
        measured = (0.0000, 0.0029, 0.0113, 0.0243, 0.0415)[index] if index < 5 else 0.05 + index / 50
        values.extend((str(index), f"{index * 0.05:.2f}", f"{state:.3f}", f"{measured:.4f}"))
    return "k t (s) u(k) Input y(k) Output " + " ".join(values)


def test_initial_workspace_has_one_composer_and_no_old_navigation(app_storage):
    app = new_app()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert not app.radio
    assert any("What are we identifying?" in m.value for m in app.markdown)
    assert any("When you’re ready" in m.value and "one measured state" in m.value and
               "headers together" in m.value for m in app.markdown)
    assert app.button(key="conversation_new")


def test_chat_plan_and_missing_file_are_conversational(app_storage):
    app = new_app()
    app.chat_input[0].set_value("show plan").run()
    assert not app.exception
    assert any(b.label == "Start run" and b.disabled for b in app.button)
    app.chat_input[0].set_value("start").run()
    assert any("couldn't find a usable table" in m.value for m in app.markdown)
    chat = app.session_state["conversation"]
    reply = core.read_chat(chat["id"])["messages"][-1]["content"].lower()
    assert "paste a table here" in reply
    assert "system description isn't needed" in reply


def test_uploaded_data_review_and_setup_render_inside_chat(app_storage):
    chat = ready_chat()
    core.begin_setup(chat)
    core.add_message(chat, "assistant", "The measured state rises across the recording.", kind="data_review")
    core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question", stage="trajectory")
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()
    assert not app.exception
    assert any(item.label == "Samples" for item in app.metric)
    assert any(item.label == "More signal checks" for item in app.expander)
    assert any("Which picture matches your data?" in item.value for item in app.markdown)
    assert any("Each separate run starts a new trajectory" in item.value for item in app.markdown)
    assert any("RUN 01" in item.value and "RUN 03" in item.value for item in app.markdown)
    assert any(button.label == "One continuous run" for button in app.button)
    assert any(button.label == "Several stacked runs" for button in app.button)
    next(button for button in app.button if button.label == "One continuous run").click().run()
    assert not app.exception
    assert app.session_state["conversation"]["setup"]["stage"] == "angle"


def test_trajectory_hint_uses_detected_resets_as_a_clue_not_a_decision(app_storage):
    chat = ready_chat()
    chat["dataset"]["analysis"]["time_reset_count"] = 2
    core.begin_setup(chat)
    core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question", stage="trajectory")
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert any("This can indicate stacked runs" in item.value for item in app.caption)
    assert any(button.label == "Several stacked runs" for button in app.button)
    assert any(button.label == "One continuous run" for button in app.button)


def test_unprefixed_output_header_prompts_client_to_confirm_roles(app_storage, monkeypatch):
    data = ("time,input,output\n" + "\n".join(
        f"{index / 10},{index % 5},{index / 20}" for index in range(100))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("measurements.csv", data, chat["id"])
    chat["dataset"] = dataset
    core.add_message(chat, "user", "I've attached my dataset.", attachment=dataset)
    question = core.add_message(chat, "assistant", ui._column_review_prompt(dataset), kind="column_review")

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert any("confirm its columns" in item.value for item in app.markdown)
    state_picker = app.multiselect(key=f"column_review_states_{question['id']}")
    input_picker = app.multiselect(key=f"column_review_inputs_{question['id']}")
    assert state_picker.value == ["output"]
    assert input_picker.value == ["input"]
    assert app.button(key=f"column_review_confirm_{question['id']}").label == "Confirm roles and fix headers"

    class ReviewJob:
        def __init__(self, snapshot, prompt, purpose="question"):
            self.purpose, self.question = purpose, prompt
            self.running, self.error, self.error_detail = True, None, None
            self.answer = None

        def start(self):
            self.answer = {"answer": "The confirmed signals are ready for setup.", "model": "test-model", "evidence": []}
            self.running = False

    monkeypatch.setattr(ui.agent, "ConversationJob", ReviewJob)
    app.button(key=f"column_review_confirm_{question['id']}").click().run()
    assert not app.exception
    corrected = app.session_state["conversation"]["dataset"]
    assert corrected["columns"] == ["time", "a_input", "s_output"]
    assert corrected["ready"]
    assert app.session_state["conversation"]["setup"]["stage"] == "trajectory"


def test_client_confirmation_renames_legacy_headers_and_resumes_data_review(app_storage, monkeypatch):
    data = ("time,input,output\n" + "\n".join(
        f"{index / 10},{index % 5},{index / 20}" for index in range(100))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("measurements.csv", data, chat["id"])
    # Simulate a conversation saved by the previous version, before header
    # clarification existed.
    dataset.pop("column_review", None)
    dataset.pop("column_review_pending", None)
    dataset["issues"] = ["Name measured state columns with the s_ prefix, for example s_position."]
    dataset["ready"] = False
    chat["dataset"] = dataset
    core.add_message(chat, "user", "I've attached my dataset.", attachment=dataset)
    core.add_message(chat, "assistant",
                     "The data is missing a measured state column. Which variable is the output? ")
    original_path = Path(dataset["path"])

    class ReviewJob:
        def __init__(self, snapshot, prompt, purpose="question"):
            self.purpose = purpose
            self.question = prompt
            self.running = True
            self.answer = None
            self.error = None
            self.error_detail = None

        def start(self):
            self.answer = {"answer": "I reviewed the corrected signals.", "model": "test-model", "evidence": []}
            self.running = False

    monkeypatch.setattr(ui.agent, "ConversationJob", ReviewJob)
    app = new_app()
    app.session_state["conversation"] = chat
    app.run().chat_input[0].set_value(
        "Yes, output is the measured state and input is the action.").run()

    assert not app.exception
    corrected = app.session_state["conversation"]["dataset"]
    assert corrected["columns"] == ["time", "a_input", "s_output"]
    assert corrected["states"] == ["s_output"]
    assert corrected["actions"] == ["a_input"]
    assert corrected["ready"]
    assert Path(corrected["path"]).is_file()
    assert original_path.read_bytes() == data
    assert app.session_state["conversation"]["setup"]["stage"] == "trajectory"
    assert any("I prepared the working copy" in item.value for item in app.markdown)
    assert any("I reviewed the corrected signals" in item.value for item in app.markdown)


def test_chat_understands_positional_columns_in_user_clarification(app_storage, monkeypatch):
    data = ("k,t (s),u(k) Input,y(k) Output\n" + "\n".join(
        f"{index},{index * 0.05:.2f},{1.000 if index < 30 else 0.000:.3f},{index / 1000:.4f}"
        for index in range(100))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("step_response.csv", data, chat["id"])
    chat["dataset"] = dataset
    core.add_message(chat, "user", "I've attached my dataset.", attachment=dataset)
    review = core.add_message(chat, "assistant", ui._column_review_prompt(dataset), kind="column_review")

    class ReviewJob:
        def __init__(self, snapshot, prompt, purpose="question"):
            self.purpose, self.question = purpose, prompt
            self.running, self.error, self.error_detail = True, None, None
            self.answer = None

        def start(self):
            self.answer = {"answer": "The corrected signals are ready for setup.", "model": "test-model", "evidence": []}
            self.running = False

    monkeypatch.setattr(ui.agent, "ConversationJob", ReviewJob)
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()
    assert app.selectbox(key=f"column_review_time_{review['id']}").value == "t (s)"
    app.chat_input[0].set_value(
        "No, remove first column; second is time, third is u, fourth is y").run()

    assert not app.exception
    corrected = app.session_state["conversation"]["dataset"]
    assert corrected["columns"] == ["time", "a_u_k_input", "s_y_k_output"]
    assert corrected["column_mapping"] == {
        "k": "removed", "t (s)": "time", "y(k) Output": "s_y_k_output", "u(k) Input": "a_u_k_input"}
    assert corrected["ready"]
    assert app.session_state["conversation"]["setup"]["stage"] == "trajectory"
    assert any("I prepared the working copy" in item.value for item in app.markdown)


def test_uploaded_dataset_card_opens_in_right_file_panel(app_storage):
    chat = ready_chat()
    message = core.add_message(chat, "user", "I've attached my dataset.", attachment=chat["dataset"])
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    file_card = app.button(key=f"open_chat_file_upload_{message['id']}")
    assert file_card.label == "oscillator.csv"
    file_card.click().run()

    assert not app.exception
    assert app.session_state["conversation_files"] is True
    assert app.session_state["conversation_selected_file"] == f"upload:{message['id']}"
    assert any(c.value.startswith("Preview ·") for c in app.caption)


def test_excel_maker_creates_chat_artifact_and_opens_workbook_preview(app_storage, monkeypatch):
    class Client:
        settings = SimpleNamespace(model="configured-current-model")

        def complete(self, system, user):
            return '{"title":"Sensor readings","sheet_name":"Measurements"}'

    class Job:
        def __init__(self, chat, question):
            self.snapshot = chat
            self.question = question
            self.purpose = "excel_maker"
            self.answer = None
            self.error = None
            self.error_detail = None
            self.running = True

        def start(self):
            self.answer = ui.excel_maker.create_workbook(
                self.snapshot, self.question, client=Client(), output_dir=app_storage / "generated")
            self.running = False

    monkeypatch.setattr(ui.excel_maker, "ExcelMakerJob", Job)
    chat = core.new_chat()
    app = new_app()
    app.session_state["conversation"] = chat
    app.run().chat_input[0].set_value(
        "Create an Excel workbook from this raw data:\nName,Score\nAli,2.5\nMina,3.0").run()

    assert not app.exception
    saved = app.session_state["conversation"]
    assistant = saved["messages"][-1]
    artifact = assistant["generated_artifact"]
    assert assistant["kind"] == "excel_maker"
    assert artifact["name"].endswith(".xlsx")
    assert artifact["rows"] == 2
    assert app.session_state["conversation_files"] is True
    assert app.session_state["conversation_selected_file"] == ui._generated_file_id(artifact["path"])
    assert any("Excel Maker created" in item.value for item in app.markdown)
    assert any(button.label == artifact["name"] for button in app.button)
    assert any(frame.value.iloc[0]["Name"] == "Ali" for frame in app.dataframe)


def test_pasted_step_response_flows_to_run_review_and_excel_preview(app_storage, monkeypatch):
    # Keep this workflow test focused on raw-table repair and Excel creation;
    # the setup agent has its own recommendation/approval coverage below.
    monkeypatch.setattr(ui, "_start_run_setup_agent", lambda chat: False)
    started = []
    review_prompts = []

    class Client:
        settings = SimpleNamespace(model="configured-current-model")

        def complete(self, system, user):
            return '{"title":"Step response measurements","sheet_name":"Measurements"}'

    class Job:
        def __init__(self, chat, question):
            self.snapshot, self.question = chat, question
            self.purpose, self.answer = "excel_maker", None
            self.error = self.error_detail = None
            self.running = True

        def start(self):
            self.answer = ui.excel_maker.create_workbook(
                self.snapshot, self.question, client=Client(), output_dir=app_storage / "generated")
            self.running = False

    class Runner:
        def __init__(self, options):
            self.options, self.running, self.result = options, True, None
            self.events, self.logs, self.error = queue.Queue(), queue.Queue(), None
            self.run_dir = None

        def start(self):
            started.append(self.options)

    def complete_data_review(chat, prompt):
        review_prompts.append(prompt)
        chat["setup"] = {"stage": "complete", "dataset_sha256": chat["dataset"]["sha256"]}
        core.save_chat(chat)

    monkeypatch.setattr(ui.excel_maker, "ExcelMakerJob", Job)
    monkeypatch.setattr(ui, "_start_data_review", complete_data_review)
    monkeypatch.setattr(runtime, "PipelineRunner", Runner)
    app = new_app()
    app.chat_input[0].set_value(flattened_step_response()).run()

    chat = app.session_state["conversation"]
    dataset = chat["dataset"]
    preview_message = next(message for message in chat["messages"] if message.get("kind") == "pasted_table_preview")
    review_message = next(message for message in chat["messages"] if message.get("kind") == "column_review")
    assert dataset["source_kind"] == "pasted_table"
    assert dataset["rows"] == 41 and dataset["column_review_pending"]
    assert app.selectbox(key=f"column_review_time_{review_message['id']}").value == "t (s)"
    assert app.multiselect(key=f"column_review_ignored_{review_message['id']}").value == ["k"]
    assert any(frame.value.iloc[0]["t (s)"] == 0.0 for frame in app.dataframe)

    app.chat_input[0].set_value(
        "Remove the first column; second is time, third is u, fourth is y").run()
    corrected = app.session_state["conversation"]["dataset"]
    assert corrected["columns"] == ["time", "a_u_k_input", "s_y_k_output"]
    assert corrected["column_mapping"]["k"] == "removed"
    assert corrected["ready"]
    assert not corrected["issues"]
    assert len(review_prompts) == 1
    assert "at most three short bullets" in review_prompts[0]
    assert "Do not recap architecture" in review_prompts[0]
    assert any("less reliable" in warning for warning in corrected["warnings"])

    app.chat_input[0].set_value("No, remove output; we don't have a separate output").run()
    assert any("does not need a separate output column" in message["content"]
               for message in app.session_state["conversation"]["messages"])
    app.chat_input[0].set_value("I don't have a physical system description").run()
    assert any("description and units are optional" in message["content"]
               for message in app.session_state["conversation"]["messages"])

    app.button(key=f"excel_from_pasted_table_{preview_message['id']}").click().run()
    assert not app.exception
    assistant = app.session_state["conversation"]["messages"][-1]
    artifact = assistant["generated_artifact"]
    assert artifact["rows"] == 41 and artifact["columns"] == 4
    assert artifact["name"].endswith(".xlsx")
    workbook_preview = [frame.value for frame in app.dataframe
                        if "y(k) Output" in frame.value.columns]
    assert workbook_preview and workbook_preview[-1].iloc[0]["y(k) Output"] == 0.0
    assert app.session_state["conversation_files"] is True

    approved_chat = app.session_state["conversation"]
    approved_chat["run_setup_flow"] = {
        "dataset_sha256": approved_chat["dataset"]["sha256"], "stage": "approved"}
    core.save_chat(approved_chat)
    app.run()
    app.chat_input[0].set_value("start").run()
    assert not app.exception
    assert len(started) == 1
    assert started[0].lstm_seq_length == 10
    assert started[0].data_path == corrected["path"]
    assert any("41 samples" in message["content"] and "less reliable" in message["content"]
               for message in app.session_state["conversation"]["messages"])


def test_run_artifact_card_opens_python_source_in_right_file_panel(app_storage):
    run_dir = app_storage / "artifact_run"
    run_dir.mkdir()
    source = run_dir / "predict.py"
    source.write_text("def predict(state, action):\n    return state + action\n")
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed", "cycles_run": 1}))
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    core.add_message(chat, "assistant", "The run completed.", kind="result", run_dir=str(run_dir))

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    next(button for button in app.button if button.label == "predict.py").click().run()

    assert not app.exception
    assert app.session_state["conversation_files"] is True
    assert any("def predict(state, action)" in block.value for block in app.code)


def test_results_zip_is_promoted_and_opens_archive_preview(app_storage):
    run_dir = app_storage / "archive_run"
    run_dir.mkdir()
    archive_path = run_dir / "SystemID_RunResults_archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("run_manifest.json", "{}")
    (run_dir / "predict.py").write_text("def predict(): return 1\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    core.add_message(chat, "assistant", "The run completed.", kind="result", run_dir=str(run_dir))

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    result_file_buttons = [button for button in app.button
                           if button.key and button.key.startswith("open_chat_file_")]
    assert result_file_buttons[0].label == "Results ZIP"
    result_file_buttons[0].click().run()
    assert not app.exception
    assert app.session_state["conversation_files"] is True
    assert any(button.label == "Download file" for button in app.download_button)
    assert any("Archive contents · 1 files" in caption.value for caption in app.caption)


def test_message_prepares_plan_through_current_client(app_storage, monkeypatch):
    def plan(chat, question, client=None):
        return {"message":"Two MLP cycles are ready.", "show_plan":True,
                "settings":core.apply_changes(chat["settings"], {"architecture":"MLP","max_cycles":2})}
    monkeypatch.setattr(core, "plan_reply", plan)
    app = new_app()
    app.chat_input[0].set_value("Use MLP with two cycles").run()
    for job in list(core.REGISTRY.planning.values()):
        job.thread.join(3)
    app.run()
    assert not app.exception
    assert app.session_state["conversation"]["settings"]["architecture"] == "MLP"
    assert any("Two MLP cycles" in m.value for m in app.markdown)


def test_start_runs_pipeline_once_and_restores_completed_conversation(app_storage, monkeypatch):
    calls = []
    class Runner:
        def __init__(self, options):
            self.options, self.running, self.result = options, False, None
            self.events, self.logs, self.error = queue.Queue(), queue.Queue(), None
            self.run_dir = core.OUTPUT_DIR / "run_20260929_120000_test"
        def start(self):
            calls.append(self.options)
            self.run_dir.mkdir(parents=True)
            entry = dict(status="completed", cycles_run=1, best_mse=0.02, latency_ms=0.5, run_dir=str(self.run_dir))
            (self.run_dir / "run_manifest.json").write_text(json.dumps(entry))
            self.result = SimpleNamespace(run_dir=self.run_dir, to_dict=lambda:entry)
    monkeypatch.setattr(runtime, "PipelineRunner", Runner)
    app = new_app()
    chat = ready_chat()
    app.session_state["conversation"] = chat
    app.run().chat_input[0].set_value("start").run()
    assert not app.exception
    assert len(calls) == 1
    assert any("run finished" in m.value for m in app.markdown)
    app.run()
    assert len(calls) == 1
    restored = new_app()
    restored.query_params["chat"] = chat["id"]
    # New AppTest starts a session; remove its initial placeholder before restoring URL.
    del restored.session_state["conversation"]
    restored.run()
    assert restored.session_state["conversation"]["run_dir"] == str(core.OUTPUT_DIR / "run_20260929_120000_test")
    assert any("run finished" in m.value for m in restored.markdown)


def test_completed_single_cycle_run_renders_cycle_details_without_slider(app_storage):
    run_dir = app_storage / "one_cycle_run"
    run_dir.mkdir()
    manifest = {
        "status": "completed", "cycles_run": 1, "best_mse": 0.07, "best_rmse": 0.26,
        "latency_ms": 0.8, "training_seconds": 1.2, "elapsed_seconds": 2.0,
        "performance_history": [{"cycle": 1, "train_mse": 0.05, "val_mse": 0.07,
                                 "training_seconds": 1.2, "config": {"hidden_layers": [8]}}],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    chat["settings"]["max_cycles"] = 1
    core.add_message(chat, "assistant", "One cycle completed.", kind="result", run_dir=str(run_dir))

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert any("Only candidate cycle recorded: 1." in caption.value for caption in app.caption)
    assert any(metric.label == "Search cycles" and metric.value == "1 / 1" for metric in app.metric)


def test_validated_pinn_automatically_checks_inline_setup_control(app_storage):
    chat = ready_chat()
    source = ui.pinn_maker.save_source(
        chat["id"], "plant.m", b"function dx = plant(x,u)\ndx = -x + u;\nend\n")
    source["chat_id"] = chat["id"]
    chat["pinn_source"] = source

    class Client:
        settings = SimpleNamespace(model="configured-test-model")

        @staticmethod
        def complete(system, user):
            return json.dumps({
                "status": "ready", "message": "Mapped the equation.",
                "equations": [{"state": "s_position", "expression": "-s0 + a0"}],
                "assumptions": [],
            })

    answer = ui.pinn_maker.prepare_equation(chat, "Prepare the saved equation", client=Client())
    chat["setup"] = {"stage": "complete"}
    chat["run_setup_flow"] = {
        "dataset_sha256": chat["dataset"]["sha256"], "stage": "model",
        "recommended_architecture": "LSTM", "architecture_reason": "A starting sequence model.",
        "recommended_effort": "regular", "recommended_cycles": 5,
        "effort_reason": "Balanced first search.", "confidence": "low",
    }
    core.add_message(chat, "assistant", "I recommend LSTM for a first comparison.", kind="run_setup")
    chat["settings"]["use_pinn"] = False
    core.REGISTRY.conversations[chat["id"]] = SimpleNamespace(
        running=False, purpose="pinn_maker", question="Prepare the saved equation",
        answer=answer, error=None, error_detail=None)

    app = new_app()
    app.session_state["conversation"] = chat
    suffix = f"{chat['id']}_{chat['dataset']['sha256'][:8]}"
    widget_key = f"setup_pinn_interest_{suffix}"
    app.session_state[widget_key] = "No"
    app.run()

    assert not app.exception
    assert app.session_state[widget_key] == "Yes"
    assert app.session_state["conversation"]["settings"]["use_pinn"] is True
    assert not any(button.label == "Set up run" for button in app.button)
    choice = app.segmented_control(key=widget_key)
    assert choice.value == "Yes"
    assert choice.disabled is False


def test_raw_python_pinn_source_upload_starts_equation_preparation(app_storage, monkeypatch):
    chat = ready_chat()
    started = {}
    monkeypatch.setattr(core, "REGISTRY", core.JobRegistry())

    class CapturedJob:
        purpose = "pinn_maker"
        running = True

        def __init__(self, snapshot, question):
            started["snapshot"] = snapshot
            started["question"] = question

        def start(self):
            started["started"] = True

    monkeypatch.setattr(ui.pinn_maker, "PINNMakerJob", CapturedJob)
    raw = b"xdot_position = -position + force\n"
    uploaded = SimpleNamespace(name="plant_pinn.py", getvalue=lambda: raw)

    ui._submit(chat, "", [uploaded], hooks={})

    assert started["started"]
    assert started["snapshot"]["pinn_source"]["name"] == "plant_pinn.py"
    assert Path(started["snapshot"]["pinn_source"]["path"]).read_bytes() == raw
    assert "Prepare plant_pinn.py for PINN" in started["question"]


def test_run_setup_agent_opens_inline_model_then_approved_effort_steps(app_storage):
    chat = ready_chat()
    chat["setup"] = {"stage": "complete", "dataset_sha256": chat["dataset"]["sha256"]}
    core.add_message(chat, "assistant", "The data assumptions are set.", kind="plan",
                     settings=chat["settings"].copy())
    core.REGISTRY.planning[chat["id"]] = SimpleNamespace(
        purpose="run_setup_recommendation", running=False, error=None, error_detail=None,
        answer={
            "architecture": "MLP", "architecture_reason": "A direct state/input baseline is a sensible first comparison for this recording.",
            "history_steps": 8, "search_effort": "regular", "cycles": 6,
            "effort_reason": "A balanced search gives several candidates without starting with the broadest compute budget.",
            "confidence": "moderate", "model": "configured-test-model",
        })

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert any("Run setup agent recommends MLP" in item.value for item in app.markdown)
    assert any("LEARN THE DYNAMICS" in item.value for item in app.markdown)
    assert app.segmented_control(key=f"setup_architecture_{chat['id']}_{chat['dataset']['sha256'][:8]}").value == "MLP"
    assert app.button(key=f"start_{chat['messages'][0]['id']}").disabled
    assert not any(button.label == "Set up run" for button in app.button)

    suffix = f"{chat['id']}_{chat['dataset']['sha256'][:8]}"
    app.button(key=f"setup_continue_effort_{suffix}").click().run()
    assert not app.exception
    assert app.session_state["conversation"]["run_setup_flow"]["stage"] == "effort"
    assert app.segmented_control(key=f"setup_effort_{suffix}").value == "regular"

    assert any(item.label == "What matters most for this run?" for item in app.selectbox)
    app.selectbox(key=f"setup_goal_{suffix}").set_value("speed").run()
    app.segmented_control(key=f"setup_effort_{suffix}").set_value("fast").run()
    app.button(key=f"setup_approve_{suffix}").click().run()

    assert not app.exception
    flow = app.session_state["conversation"]["run_setup_flow"]
    assert flow["stage"] == "approved"
    assert app.session_state["conversation"]["settings"]["architecture"] == "MLP"
    assert app.session_state["conversation"]["settings"]["optimization_goal"] == "speed"
    assert app.session_state["conversation"]["settings"]["run_mode"] == "fast"
    assert app.session_state["conversation"]["settings"]["max_cycles"] == 6
    assert app.button(key=f"setup_start_{suffix}").disabled is False


def test_human_tuning_checkpoint_cards_render_and_accept_client_guidance():
    class ReviewRunner:
        action = None
        def respond_to_checkpoint(self, action):
            self.action = action
            return True

    script = '''
import streamlit as st
from frontend_streamlit.ui_conversation import _human_checkpoint_card
_human_checkpoint_card({"id": "test-chat"}, st.session_state["task"], st.session_state["state"])
'''
    runner = ReviewRunner()
    task = {"runner": runner}
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["task"] = task
    app.session_state["state"] = {"checkpoint": {
        "id": "initializer_1", "phase": "initializer", "goal": "balanced",
        "config": {"learning_rate": 0.001, "hidden_layers": [32], "dropout_rate": 0.1,
                   "weight_decay": 0.001, "reasoning": "A compact starting model suits this sample size.",
                   "lr_search_min": 0.0001, "lr_search_max": 0.001,
                   "hidden_size_search_min": 16, "hidden_size_search_max": 128,
                   "num_layers_search_min": 1, "num_layers_search_max": 2,
                   "epochs": 80, "batch_size": 32, "early_stop_patience": 20,
                   "activation": "tanh"},
        "setup": {"architecture": "LSTM", "run_mode": "fast", "max_cycles": 7,
                  "max_hours": 0.5, "history_steps": 10, "history_seconds": 0.09,
                  "estimated_parameters": 1700},
        "dataset": {"samples": 120, "train_samples": 84, "validation_samples": 18,
                    "test_samples": 18, "states": ["s_position"], "inputs": ["a_force"],
                    "median_dt": 0.01, "complexity": "Level 2", "complexity_tier": 2,
                    "varying_states": ["s_position"], "varying_inputs": ["a_force"],
                    "derivatives": [], "quality_notes": []},
    }}
    app.run()
    assert not app.exception
    assert any("first candidate, checked against your data" in item.value for item in app.markdown)
    assert app.table
    assert any(button.label == "Smaller model" for button in app.button)
    app.button(key="human_tuning_initializer_1_compact").click().run()
    assert not app.exception
    assert runner.action == "compact"

    runner.action = None
    app.session_state["state"] = {"checkpoint": {
        "id": "early_results_2", "phase": "early_results", "cycle": 2,
        "max_cycles": 6, "best_mse": 0.02,
        "recent_results": [
            {"cycle": 1, "train_mse": 0.03, "val_mse": 0.04, "training_seconds": 1.0},
            {"cycle": 2, "train_mse": 0.02, "val_mse": 0.02, "training_seconds": 1.2},
        ],
        "state_validation_mse": {"s_position": 0.025},
    }}
    app.run()
    assert not app.exception
    assert any("How should the remaining search adapt?" in item.value for item in app.markdown)
    assert app.button(key="human_tuning_early_results_2_widen")


def test_initializer_setup_analysis_connects_budget_and_model_size_to_data():
    rows = ui._initializer_analysis_rows({
        "setup": {"architecture": "LSTM", "run_mode": "fast", "max_cycles": 7,
                  "max_hours": 0.5, "history_steps": 10, "history_seconds": 0.09,
                  "estimated_parameters": 334852},
        "config": {"hidden_layers": [128, 128, 128], "learning_rate": 0.001,
                   "lr_search_min": 0.00005, "lr_search_max": 0.001,
                   "dropout_rate": 0.2, "weight_decay": 0.001},
        "dataset": {"samples": 6500, "train_samples": 5148,
                    "validation_samples": 702, "test_samples": 650,
                    "states": ["s_pitch", "s_yaw", "s_dpitch", "s_dyaw"],
                    "inputs": ["a_left", "a_right"], "median_dt": 0.01,
                    "complexity": "Level 4", "complexity_tier": 4,
                    "varying_states": ["s_pitch", "s_yaw", "s_dpitch", "s_dyaw"],
                    "varying_inputs": ["a_left", "a_right"], "derivatives": []},
    })
    by_area = {row["Review area"]: row for row in rows}

    assert "Fast · up to 7 candidate cycles" in by_area["Search budget"]["Initializer proposal"]
    assert "0.09 s of history" in by_area["Model context"]["Initializer proposal"]
    assert "334,852 trainable parameters" in by_area["Network capacity"]["Initializer proposal"]
    assert "5,148 rows" in by_area["Network capacity"]["Read against the data"]
    assert "held-out test" in by_area["Dataset and split"]["Read against the data"]


def test_completed_run_question_routes_to_conversational_agent(app_storage, monkeypatch):
    calls = []
    class Job:
        running, error, answer = False, None, None
        def __init__(self, chat, question, purpose="question"):
            self.question, self.purpose = question, purpose
            calls.append((chat["run_dir"], question, purpose))
        def start(self):
            self.answer = dict(answer="Checked this run's actual evidence.", model="configured", evidence=[])
    monkeypatch.setattr(ui.agent, "ConversationJob", Job)
    chat = core.new_chat()
    chat["run_dir"] = str(app_storage / "saved_run")
    core.save_chat(chat)
    app = new_app()
    app.session_state["conversation"] = chat
    app.run().chat_input[0].set_value("Why did this model fail?").run()
    assert not app.exception
    assert calls[0][0] == chat["run_dir"]
    assert any("actual evidence" in m.value for m in app.markdown)


def test_conversation_clarification_stays_in_chat_without_run_visuals(app_storage):
    chat = ready_chat()
    core.add_message(chat, "user", "What should I try next?")
    core.REGISTRY = core.JobRegistry()
    core.REGISTRY.conversations[chat["id"]] = SimpleNamespace(
        running=False,
        answer={"status": "clarification", "answer": "Which goal matters most?", "evidence": [],
                "model": "local clarification"},
        question="What should I try next?", purpose="question", error=None, error_detail=None,
    )

    ui._sync(chat, hooks={})

    clarification = chat["messages"][-1]
    assert clarification["kind"] == "clarification"
    assert clarification["content"] == "Which goal matters most?"
    assert "visual_question" not in clarification


def test_identity_question_without_dataset_routes_to_conversational_agent(app_storage, monkeypatch):
    calls = []

    class Job:
        running, error, answer = False, None, None

        def __init__(self, chat, question, purpose="question"):
            self.purpose = purpose
            calls.append((chat["id"], question, purpose))

        def start(self):
            self.answer = dict(answer="I am LabCD, the assistant in LabCD AgentSysID [A1].",
                               model="configured", evidence=[{"id": "A1", "title": "Assistant identity"}])

    monkeypatch.setattr(ui.agent, "ConversationJob", Job)
    chat = core.new_chat()
    core.save_chat(chat)
    app = new_app()
    app.session_state["conversation"] = chat
    app.run().chat_input[0].set_value("What is your name?").run()

    assert not app.exception
    assert calls == [(chat["id"], "What is your name?", "question")]
    assert any("I am LabCD" in item.value for item in app.markdown)


def test_setup_question_is_not_repeated_after_answering_client_question(app_storage):
    chat = ready_chat()
    core.begin_setup(chat)
    prompt = core.setup_prompt(chat)
    question = core.add_message(chat, "assistant", prompt, kind="setup_question", stage="trajectory")
    core.add_message(chat, "user", "What does one continuous experiment mean?")
    core.REGISTRY.conversations[chat["id"]] = SimpleNamespace(
        running=False, purpose="setup_explain", question="What does one continuous experiment mean?",
        answer={"answer": "It means the measurements follow one uninterrupted experiment.",
                "model": "configured-test-model", "evidence": []}, error=None, error_detail=None)

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    saved = app.session_state["conversation"]
    setup_questions = [message for message in saved["messages"] if message.get("kind") == "setup_question"]
    assert not app.exception
    assert len(setup_questions) == 1
    assert setup_questions[0]["id"] == question["id"]
    assert any("one uninterrupted experiment" in item.value for item in app.markdown)
    assert any(button.label == "One continuous run" and not button.disabled for button in app.button)


def test_diagnostic_region_denial_is_distinguished_from_run_failure(app_storage):
    chat = core.new_chat()
    chat["run_dir"] = str(app_storage / "saved_run")
    core.REGISTRY.diagnostics[chat["id"]] = SimpleNamespace(
        running=False, answer=None, error="403 unsupported_country_region_territory")
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()
    assert not app.exception
    assert any("diagnostic request from the current region" in m.value for m in app.markdown)
    assert any("Connection details" in exp.label for exp in app.expander)


def test_other_conversation_cannot_stop_active_run(app_storage, monkeypatch):
    stopped = []
    monkeypatch.setattr(ui, "request_stop", lambda:stopped.append(True))
    core.REGISTRY.training["other"] = {"runner":SimpleNamespace(running=True)}
    app = new_app()
    app.chat_input[0].set_value("stop").run()
    assert not stopped
    assert any("no active training" in m.value for m in app.markdown)


def test_live_training_uses_compact_chat_progress_card(app_storage):
    from time import time

    chat = {"id": "live-progress", "settings": {"max_cycles": 7}}
    state = {
        "stage": "Tuning cycles", "progress": 0.63, "history": [], "log": "",
        "critic": [], "result": None, "checkpoint": None,
        "activity": [
            {"id": "stage:Tuning cycles", "label": "Searching for the best model",
             "state": "complete", "details": {}},
            {"id": "actor:1", "label": "Actor is training cycle 1",
             "state": "running", "details": {}},
        ],
    }
    core.REGISTRY.training[chat["id"]] = {
        "runner": SimpleNamespace(running=True), "started": time() - 90, "state": state,
    }
    script = '''
import streamlit as st
from frontend_streamlit import ui_conversation as ui
ui._working(st.session_state["chat"])
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()

    assert not app.exception
    assert any("Your model is taking shape" in item.value for item in app.markdown)
    assert any("Training candidates and comparing their validation scores" in item.value for item in app.caption)
    assert any("Now · Actor is training cycle 1" in item.value for item in app.caption)
    assert any(item.label == "Live activity · 2 updates" for item in app.status)
    assert any(item.label == "Search cycles" and item.value == "1 / 7" for item in app.metric)
