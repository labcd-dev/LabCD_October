import json
from pathlib import Path

import pytest

from frontend_streamlit import conversation_core as core


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(core, "REGISTRY", core.JobRegistry())
    return tmp_path


def csv_data(rows=100):
    return ("time,s_position,a_force\n" + "\n".join(f"{i/10},{i/20},{i%7}" for i in range(rows))).encode()


def attached_chat():
    chat = core.new_chat()
    chat["dataset"] = core.inspect_upload("measurements.csv", csv_data(), chat["id"])
    return chat


def test_new_conversations_default_to_the_full_fast_search_budget(storage):
    chat = core.new_chat()
    assert chat["settings"]["run_mode"] == "fast"
    assert chat["settings"]["max_cycles"] == 7


def test_client_can_declare_a_periodic_state_when_file_has_no_wrap_jump(storage):
    chat = attached_chat()
    chat["dataset"]["states"] = ["s_pitch", "s_pitch_rate"]
    chat["dataset"]["analysis"] = {
        "detected_wrap_states": [], "angle_named_states": ["s_pitch"]
    }
    chat["setup"] = {"stage": "angle", "dataset_sha256": chat["dataset"]["sha256"]}

    reply = core.advance_setup(
        chat, "pitch is a wrapped angle, even though this recording stays between the boundaries"
    )

    assert reply["kind"] == "plan"
    assert chat["settings"]["angle_indices"] == [0]
    assert chat["settings"]["auto_detect_angles"] is False
    assert "sine/cosine" in reply["content"]
    assert "never crosses" in reply["content"]


def test_pitch_rate_is_not_selected_as_a_wrapped_pitch_state(storage):
    chat = attached_chat()
    chat["dataset"]["states"] = ["s_pitch_rate"]
    chat["dataset"]["analysis"] = {"detected_wrap_states": [], "angle_named_states": []}
    chat["setup"] = {"stage": "angle", "dataset_sha256": chat["dataset"]["sha256"]}

    reply = core.advance_setup(chat, "pitch is wrapped modulo 2 pi")

    assert reply["kind"] == "setup_question"
    assert chat["setup"]["stage"] == "angle_select"
    assert chat["settings"]["angle_indices"] == []


def test_conversation_persists_and_restores_attachment_and_settings(storage):
    chat = attached_chat()
    core.add_message(chat, "user", "Identify this oscillator")
    restored = core.read_chat(chat["id"])
    assert restored["title"] == "Identify this oscillator"
    assert restored["dataset"]["sha256"] == chat["dataset"]["sha256"]
    assert core.list_chats()[0]["id"] == chat["id"]
    assert core.read_chat("../outside") is None
    assert not (storage / "outside.json").exists()


def test_projects_persist_and_removing_a_chat_keeps_run_files(storage):
    project = core.create_project("  Pendulum study  ")
    chat = core.new_chat(project["id"])
    run_dir = storage / "runs" / "completed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.pdf").write_bytes(b"kept")
    chat.update(title="Model comparison", run_dir=str(run_dir))
    core.save_chat(chat)

    assert core.list_projects()[0]["title"] == "Pendulum study"
    assert core.read_chat(chat["id"])["project_id"] == project["id"]
    assert core.list_chats()[0]["project_id"] == project["id"]
    assert core.delete_chat(chat["id"])
    assert core.read_chat(chat["id"]) is None
    assert (run_dir / "report.pdf").read_bytes() == b"kept"
    assert core.read_project(project["id"])["title"] == "Pendulum study"


def test_project_names_and_ids_are_validated(storage):
    with pytest.raises(ValueError, match="Enter a project name"):
        core.create_project("   ")
    with pytest.raises(ValueError, match="80 characters"):
        core.create_project("p" * 81)
    with pytest.raises(ValueError, match="Invalid project identifier"):
        core.new_chat("../outside")
    with pytest.raises(ValueError, match="Invalid project identifier"):
        core.project_path("../outside")


def test_attachments_with_same_name_cannot_overwrite_other_data(storage):
    chat = core.new_chat()
    first = core.inspect_upload("../measurements.csv", csv_data(), chat["id"])
    second = core.inspect_upload("measurements.csv", csv_data(101), chat["id"])
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == csv_data()
    assert first["ready"] and first["rows"] == 100
    assert Path(first["path"]).resolve().is_relative_to(core.UPLOAD_DIR)


@pytest.mark.parametrize("data,issue", [
    (b"time,value\n0,2\n1,3", "10 samples"),
    (csv_data().replace(b"1.0,0.5", b"0.9,0.5"), "Timestamps"),
    (csv_data().replace(b"1.0,0.5", b"1.0,nan"), "missing"),
])
def test_invalid_dataset_cannot_be_started(storage, data, issue):
    chat = core.new_chat()
    chat["dataset"] = core.inspect_upload("data.csv", data, chat["id"])
    assert any(issue in item for item in chat["dataset"]["issues"])
    with pytest.raises(ValueError, match="valid dataset"):
        core.make_options(chat)


@pytest.mark.parametrize("rows,expected_sequence", [(41, 10), (20, 6), (10, 3)])
def test_small_dataset_can_run_with_a_window_that_fits_its_splits(storage, rows, expected_sequence):
    chat = core.new_chat()
    data = ("time,s_position,a_force\n" + "\n".join(
        f"{index / 10},{index / 20},{index % 7}" for index in range(rows))).encode()
    chat["dataset"] = core.inspect_upload("small_measurements.csv", data, chat["id"])
    chat["setup"] = {"stage": "complete", "dataset_sha256": chat["dataset"]["sha256"]}

    assert chat["dataset"]["ready"]
    if rows < 50:
        assert any("less reliable" in warning for warning in chat["dataset"]["warnings"])
    options = core.make_options(chat)

    assert options.lstm_seq_length == expected_sequence
    assert options.rollout_horizon == 1
    assert chat["settings"]["lstm_seq_length"] == expected_sequence


def test_wrong_state_headers_are_asked_and_corrected_after_client_confirmation(storage):
    raw = ("time,input,output\n" + "\n".join(
        f"{index / 10},{index % 5},{index / 20}" for index in range(100))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("measurements.csv", raw, chat["id"])
    chat["dataset"] = dataset
    attachment_message = core.add_message(chat, "user", "I uploaded my measurements.", attachment=dataset)
    review_message = core.add_message(chat, "assistant", "Please confirm the column roles.", kind="column_review")

    assert dataset["column_review_pending"] is True
    assert dataset["column_review"]["suggested_states"] == ["output"]
    assert dataset["column_review"]["suggested_actions"] == ["input"]
    assert dataset["column_review"]["time_required"] is False
    assert dataset["issues"] == []
    assert not dataset["ready"]
    assert core.column_roles_from_answer(dataset, "Yes, output is the measured state and input is the action.") == (
        "time", ["output"], ["input"])

    mapping, corrected = core.apply_column_roles(chat, ["output"], ["input"])
    assert mapping == {"output": "s_output", "input": "a_input"}
    assert corrected["ready"]
    assert corrected["states"] == ["s_output"]
    assert corrected["actions"] == ["a_input"]
    assert corrected["columns"] == ["time", "a_input", "s_output"]
    assert Path(dataset["path"]).read_bytes() == raw
    assert attachment_message["attachment"]["sha256"] == dataset["sha256"]
    assert attachment_message["attachment"]["issues"] == []
    assert attachment_message["attachment"]["column_mapping"] == mapping
    assert chat["messages"][-1]["resolved"] is True
    assert review_message["column_mapping"] == mapping


def test_column_role_answer_requires_an_unambiguous_state_choice(storage):
    raw = ("time,signal_a,signal_b\n" + "\n".join(
        f"{index / 10},{index / 20},{index / 30}" for index in range(100))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("signals.csv", raw, chat["id"])
    assert core.column_roles_from_answer(dataset, "No, those names are wrong.") is None
    assert core.column_roles_from_answer(dataset, "signal_a") is None
    assert core.column_roles_from_answer(
        dataset, "second is the state and third is the input"
    ) == ("time", ["signal_a"], ["signal_b"])


def test_input_is_required_and_each_state_rate_is_in_the_training_data(storage):
    chat = core.new_chat()
    data = ("time,s_position\n" + "\n".join(
        f"{index / 10},{index / 20}" for index in range(20))).encode()
    dataset = core.inspect_upload("no_input.csv", data, chat["id"])

    assert dataset["ready"] is False
    assert dataset["column_review"] is None
    assert any("at least one input/action column" in issue and "needs a time-rate (`xdot_*`) for every state" in issue
               for issue in dataset["issues"])
    planner_prompt = " ".join(core.PLANNER_PROMPT.split())
    assert "at least one input/action column (`a_*`)" in planner_prompt
    assert "LabCD needs a time-rate (`xdot_*`) for every state" in planner_prompt


def test_column_roles_cannot_be_confirmed_without_an_input(storage):
    chat = core.new_chat()
    raw = ("time,output,signal_b\n" + "\n".join(
        f"{index / 10},{index / 20},{index % 3}" for index in range(20))).encode()
    dataset = core.inspect_upload("roles.csv", raw, chat["id"])
    chat["dataset"] = dataset

    assert core.column_roles_from_answer(dataset, "yes") is None
    with pytest.raises(ValueError, match="at least one measured system input or control channel"):
        core.apply_column_roles(chat, ["output"], [], time_column="time")


def test_positional_column_roles_map_time_state_and_action_and_ignore_index(storage):
    raw = ("k,t (s),u(k) Input,y(k) Output\n" + "\n".join(
        f"{index},{index * 0.05:.2f},{1.000 if index < 30 else 0.000:.3f},{index / 1000:.4f}"
        for index in range(41))).encode()
    chat = core.new_chat()
    dataset = core.inspect_upload("step_response.csv", raw, chat["id"])
    chat["dataset"] = dataset

    roles = core.column_roles_from_answer(
        dataset, "Remove the first column; second is time, third is u, fourth is y")
    assert roles == ("t (s)", ["y(k) Output"], ["u(k) Input"])
    assert core.column_roles_from_answer(dataset, "yes thats ok") == roles
    assert core.column_roles_from_answer(dataset, "No, remove output; we don't have a separate output") == roles

    mapping, corrected = core.apply_column_roles(chat, roles[1], roles[2], time_column=roles[0])
    assert mapping == {"k": "removed", "t (s)": "time", "y(k) Output": "s_y_k_output", "u(k) Input": "a_u_k_input"}
    assert corrected["columns"] == ["time", "a_u_k_input", "s_y_k_output"]
    assert corrected["ready"]
    assert not corrected["issues"]
    assert any("less reliable" in warning for warning in corrected["warnings"])
    assert corrected["sample_period"] == pytest.approx(0.05)
    assert Path(dataset["path"]).read_bytes() == raw


def test_options_reject_changed_attachment(storage):
    chat = attached_chat()
    Path(chat["dataset"]["path"]).write_bytes(csv_data(101))
    with pytest.raises(ValueError, match="changed"):
        core.make_options(chat)


def test_plan_maps_to_pipeline_and_locks_requested_values(storage):
    chat = attached_chat()
    chat["settings"] = core.apply_changes(chat["settings"], {"architecture":"MLP", "optimization_goal":"speed", "epochs":25, "max_cycles":2, "hidden_size_min":16, "manual_starting_hidden_layers":[32,16]})
    options = core.make_options(chat)
    assert options.architecture == "MLP" and options.epochs == 25 and options.max_cycles == 2
    assert options.optimization_goal == "speed"
    assert options.human_in_the_loop is True
    assert options.initializer_overrides["epochs"] == 25
    assert options.initializer_overrides["hidden_layers"] == [32,16]
    assert options.api_provider is None and options.llm_model is None
    assert options.customer_description == ""


def test_pcnn_setup_maps_named_exact_constraints_to_state_indices(storage, monkeypatch):
    chat = core.new_chat()
    data = ("time,s_position,s_velocity,a_force\n" + "\n".join(
        f"{i / 10},{i / 20},{i / 30},{i % 7}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("measurements.csv", data, chat["id"])
    equation = storage / "physics_equation.py"
    equation.write_text("def compute_analytical_xdot(states, actions): return states", encoding="utf-8")
    chat["pinn_equation"] = {"path": str(equation)}
    monkeypatch.setattr("frontend_streamlit.pinn_maker.is_equation_ready", lambda _chat: True)
    chat["settings"] = core.apply_changes(chat["settings"], {
        "use_pinn": True, "physics_mode": "pcnn",
        "pcnn_hard_constraint_states": ["s_position"],
    })

    options = core.make_options(chat)

    assert options.physics_mode == "pcnn"
    assert options.pcnn_hard_constraint_states == ["s_position"]
    assert options.pcnn_hard_constraint_indices == [0]
    assert options.pinn_equation_file == str(equation)


@pytest.mark.parametrize("request_text", ["change physics mode to PCNN", "switch to PCNN"])
def test_client_can_change_physics_mode_in_chat(request_text, storage):
    chat = attached_chat()

    changes = core._requested_setting_changes(chat, request_text)

    assert changes["physics_mode"] == "pcnn"
    assert changes["use_pinn"] is True


@pytest.mark.parametrize("question", ["How do I use PCNN?", "Did this run use PCNN?"])
def test_physics_questions_are_not_misread_as_setup_edits(question, storage):
    chat = attached_chat()

    assert core._requested_setting_changes(chat, question) == {}


def test_pcnn_rejects_locking_every_state_equation(storage, monkeypatch):
    chat = attached_chat()
    equation = storage / "physics_equation.py"
    equation.write_text("def compute_analytical_xdot(states, actions): return states", encoding="utf-8")
    chat["pinn_equation"] = {"path": str(equation)}
    monkeypatch.setattr("frontend_streamlit.pinn_maker.is_equation_ready", lambda _chat: True)
    chat["settings"] = core.apply_changes(chat["settings"], {
        "use_pinn": True, "physics_mode": "pcnn",
        "pcnn_hard_constraint_states": ["s_position"],
    })

    with pytest.raises(ValueError, match="at least one state equation left"):
        core.make_options(chat)


def test_manual_run_uses_bounded_per_conversation_presets(storage):
    chat = attached_chat()
    chat["settings"] = core.apply_changes(chat["settings"], {
        "choose_via_llm_initializer": False, "hidden_size_min": 16,
        "hidden_size_max": 32, "num_layers_min": 2, "num_layers_max": 2,
        "learning_rate_min": 0.0001, "learning_rate_max": 0.0002})
    options = core.make_options(chat)
    assert options.manual_starting_hidden_layers == [32, 32]
    assert options.manual_starting_lr == 0.0002
    assert options.initializer_overrides["hidden_layers"] == [32, 32]
    assert chat["settings"]["manual_starting_hidden_layers"] is None


@pytest.mark.parametrize("changes", [
    {"llm_model":"unapproved-model"}, {"data_path":"outside.csv"}, {"epochs":0},
    {"optimization_goal":"guaranteed"},
    {"manual_starting_hidden_layers":[-1]}, {"mse_target":float("nan")},
    {"hidden_size_min":500,"hidden_size_max":32}, {"savgol_window":4},
    {"manual_starting_hidden_layers":[16]}, {"manual_starting_hidden_layers":[32,32,32,32]},
    {"manual_starting_lr":0.5},
    {"manual_split_times":[2,1]}, {"auto_filter_percentiles":[99,1]},
])
def test_planner_cannot_change_paths_model_or_invalid_options(changes):
    with pytest.raises(ValueError):
        core.apply_changes(core.RunSettings().model_dump(), changes)


def test_planner_uses_schema_and_only_applies_valid_changes(storage):
    chat = attached_chat()
    class Client:
        def complete(self, system, user):
            payload = json.loads(user)
            assert "path" not in payload["dataset"]
            assert "settings_schema" in payload
            assert "cannot" in system
            return json.dumps({"message":"Use two MLP cycles.", "changes":{"architecture":"MLP","max_cycles":2}, "show_plan":True})
    result = core.plan_reply(chat, "Try MLP with two cycles", client=Client())
    assert result["settings"]["max_cycles"] == 2
    assert result["settings"]["epochs"] == chat["settings"]["epochs"]
    assert chat["settings"]["architecture"] == "LSTM"


def test_planner_reports_region_denial_without_losing_local_plan(storage, monkeypatch):
    chat = attached_chat()
    def denied(*_args):
        raise RuntimeError("403 unsupported_country_region_territory")
    monkeypatch.setattr(core, "plan_reply", denied)
    job = core.PlanningJob(chat, "Try two cycles")
    job._run()
    assert "region where LabCD is running" in job.error
    assert "show plan" in job.error
    assert "unsupported_country_region_territory" in job.error_detail
    assert core.make_options(chat).data_path == chat["dataset"]["path"]
