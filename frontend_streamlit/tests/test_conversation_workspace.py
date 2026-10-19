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
    app = AppTest.from_file(APP, default_timeout=30).run()
    # Existing workspace tests start after a first-time user closes the welcome modal.
    if app.button(key="labcd_intro_close"):
        app.button(key="labcd_intro_close").click().run()
    return app


def ready_chat():
    chat = core.new_chat()
    data = ("time,s_position,a_force\n" + "\n".join(f"{i/10},{i/20},{i%7}" for i in range(100))).encode()
    chat["dataset"] = core.inspect_upload("oscillator.csv", data, chat["id"])
    return chat


def test_setup_pen_opens_visual_editor(app_storage):
    chat = ready_chat()
    illustration = lambda architecture, selected=False: f'<svg aria-label="{architecture} diagram"></svg>'
    suffix = f"{chat['id']}_{chat['dataset']['sha256'][:8]}"
    script = '''
import streamlit as st
from frontend_streamlit import ui_run_setup_summary as setup_summary
setup_summary.render(st.session_state["chat"], {}, st.session_state["chat"]["settings"], False,
                     lambda architecture, selected=False: f'<svg aria-label="{architecture} diagram"></svg>')
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()

    app.button(key=f"setup_choice_edit_{suffix}").click().run()
    assert not app.exception
    assert any("LABCD · SETUP STUDIO" in item.value for item in app.markdown)
    assert any("LSTM sequence memory" in item.value for item in app.markdown)
    assert app.segmented_control(key=f"setup_popup_model_{suffix}_model").value == "LSTM"


def test_physics_editor_supports_pcnn_and_switching_back_to_pinn(app_storage, monkeypatch):
    from frontend_streamlit import ui_run_setup_summary as setup_summary

    chat = ready_chat()
    chat["settings"].update(use_pinn=True, physics_mode="pcnn")
    chat["pinn_equation"] = {"states": ["s_position"]}
    monkeypatch.setattr(setup_summary.pinn_maker, "is_equation_ready", lambda _chat: True)
    suffix = "physics-mode-test"
    script = '''
import streamlit as st
from frontend_streamlit import ui_run_setup_summary as setup_summary
setup_summary._edit_dialog(st.session_state["chat"], "physics", st.session_state["suffix"],
                           lambda architecture, selected=False: f'<svg aria-label="{architecture} diagram"></svg>')
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.session_state["suffix"] = suffix
    app.run()

    method_key = f"setup_popup_physics_{suffix}_enabled"
    assert app.segmented_control(key=method_key).value == "PCNN"
    assert app.multiselect(key=f"setup_popup_physics_{suffix}_hard_states")
    assert any("physics-constrained network" in item.value for item in app.markdown)
    app.segmented_control(key=method_key).set_value("PINN").run()
    assert not app.exception
    assert any("PINN · soft guidance" in item.value for item in app.markdown)
    app.button(key=f"setup_popup_physics_{suffix}_save").click().run()

    assert not app.exception
    assert app.session_state["chat"]["settings"]["physics_mode"] == "pinn"
    assert app.session_state["chat"]["settings"]["use_pinn"] is True


def test_mlp_setup_can_switch_back_to_lstm_from_history_row(app_storage):
    chat = ready_chat()
    chat["settings"]["architecture"] = "MLP"
    chat["run_setup_flow"] = {
        "recommended_architecture": "MLP",
        "initializer_proposal": {**chat["settings"], "architecture": "MLP"},
    }
    suffix = f"{chat['id']}_{chat['dataset']['sha256'][:8]}"
    script = '''
import streamlit as st
from frontend_streamlit import ui_run_setup_summary as setup_summary
setup_summary.render(st.session_state["chat"], st.session_state["chat"].get("run_setup_flow", {}),
                     st.session_state["chat"]["settings"], False,
                     lambda architecture, selected=False: f'<svg aria-label="{architecture} diagram"></svg>')
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()

    change_model = app.button(key=f"setup_summary_edit_history_{suffix}")
    assert change_model.label == "Change model"
    assert not change_model.disabled
    change_model.click().run()
    assert not app.exception
    model_key = f"setup_popup_model_{suffix}_model"
    assert app.segmented_control(key=model_key).value == "MLP"

    # AppTest closes a dialog on fragment-widget reruns, so exercise the save
    # path in a persistent-dialog harness while retaining the UI assertion above.
    edit_chat = ready_chat()
    edit_chat["settings"]["architecture"] = "MLP"
    edit_suffix = f"{edit_chat['id']}_{edit_chat['dataset']['sha256'][:8]}"
    edit_script = '''
import streamlit as st
from frontend_streamlit import ui_run_setup_summary as setup_summary
setup_summary._edit_dialog(st.session_state["chat"], "model", st.session_state["suffix"],
                           lambda architecture, selected=False: f'<svg aria-label="{architecture} diagram"></svg>')
'''
    editor = AppTest.from_string(edit_script, default_timeout=30)
    editor.session_state["chat"] = edit_chat
    editor.session_state["suffix"] = edit_suffix
    editor.run()
    editor.segmented_control(key=f"setup_popup_model_{edit_suffix}_model").set_value("LSTM").run()
    editor.button(key=f"setup_popup_model_{edit_suffix}_save").click().run()

    assert not editor.exception
    assert editor.session_state["chat"]["settings"]["architecture"] == "LSTM"


def test_each_setup_setting_has_a_distinct_visual(app_storage):
    from frontend_streamlit import ui_run_setup_summary as setup_summary

    chat = ready_chat()
    settings = chat["settings"]
    dataset = {**chat["dataset"], "states": ["s_position"], "inputs": ["a_force"]}
    sections = (
        "model", "history", "physics", "context", "goal", "effort", "cycles",
        "signals", "epochs", "network", "training", "targets", "advanced",
    )
    rendered = {
        section: setup_summary._editor_hero(section, section.title(), settings, dataset, chat, {})
        for section in sections
    }

    assert len(set(rendered.values())) == len(sections)
    for section, visual in rendered.items():
        assert f"setup-edit-visual--{section}" in visual
        assert f'aria-label="' in visual


def test_network_editor_has_one_live_preview_for_the_edited_topology(app_storage):
    from frontend_streamlit import ui_run_setup_summary as setup_summary

    settings = core.RunSettings().model_dump()
    settings["manual_starting_hidden_layers"] = [128, 128, 128]
    widget_prefix = "setup_popup_network_live"
    live_settings = setup_summary._live_network_settings(
        settings,
        widget_prefix,
        {
            f"{widget_prefix}_layers": 2,
            f"{widget_prefix}_width_0": 64,
            f"{widget_prefix}_width_1": 256,
        },
    )
    hero = setup_summary._editor_hero(
        "network", "Network shape", live_settings,
        {"rows": 100, "states": ["s_position"], "inputs": ["a_force"]},
        {}, {},
    )

    assert hero.count("<svg ") == 1
    assert "APPROVED STARTING TOPOLOGY" not in hero
    assert "2 layers · 64 → 256 units" in hero
    assert 'class="setup-edit-nn-width"' in hero
    assert ">64</text>" in hero
    assert ">256</text>" in hero
    assert ">L1</text>" in hero and ">L2</text>" in hero
    assert ">L3</text>" not in hero


def test_network_preview_updates_when_the_client_edits_controls(app_storage):
    chat = ready_chat()
    script = '''
import streamlit as st
from frontend_streamlit import ui_run_setup_summary as setup_summary
setup_summary._edit_dialog(
    st.session_state["chat"], "network", "live",
    lambda architecture, selected=False: "",
)
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()
    assert not app.exception

    app.slider(key="setup_popup_network_live_layers").set_value(2).run()
    assert not app.exception
    app.number_input(key="setup_popup_network_live_width_0").set_value(64).run()
    assert not app.exception

    hero = next(item.value for item in app.markdown if "setup-edit-hero--network" in item.value)
    assert "2 layers · 64 → 128 units" in hero
    assert hero.count("<svg ") == 1
    assert "APPROVED STARTING TOPOLOGY" not in hero


def test_blue_right_aligned_bubble_matches_streamlit_material_user_avatar():
    from frontend_streamlit.ui_conversation_theme import CSS

    assert '[data-testid="stChatMessageAvatarCustom"]' in CSS
    assert '> [data-testid="stChatMessageContent"]' in CSS
    assert 'margin:14px 0 10px auto!important' in CSS
    assert 'width:fit-content; max-width:100%' in CSS
    assert 'min-width:min(100%,128px); margin-left:auto' in CSS
    assert 'linear-gradient(145deg,#182a40 0%,#172538 100%)' in CSS


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
    welcome = next((m.value for m in app.markdown if "conversation-welcome" in m.value), "")
    assert "over time" in welcome
    assert "model" in welcome
    assert "xdot_" not in welcome
    assert not app.text_area
    assert not any("Example measurement sheet" in m.value for m in app.markdown)
    assert len(app.dataframe) == 0
    assert any(button.label == "How do I make an Excel?" for button in app.button)
    assert app.button(key="conversation_new")


def test_excel_how_to_shows_example_sheet_only_after_client_asks(app_storage):
    app = new_app()
    assert len(app.dataframe) == 0

    app.chat_input[0].set_value("How do I make an Excel workbook with LabCD?").run()

    assert not app.exception
    chat = app.session_state["conversation"]
    assert chat["messages"][-1]["kind"] == "excel_help"
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value.columns) == [
        "time", "s_1", "s_2", "a_1", "xdot_1", "xdot_2",
    ]
    assert any("`xdot_1` matches `s_1`" in item.value and "estimates them" in item.value
               for item in app.caption)
    assert any("only an example" in item.value for item in app.caption)
    assert not any(message.get("generated_artifact") for message in chat["messages"])

    shortcut = new_app()
    shortcut.button(key="suggest_table_view").click().run()
    assert not shortcut.exception
    assert shortcut.session_state["conversation"]["messages"][-1]["kind"] == "excel_help"
    assert len(shortcut.dataframe) == 1


def test_first_visit_can_take_the_visual_tutorial_through_all_steps(app_storage):
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.button(key="labcd_intro_close")
    assert app.button(key="labcd_intro_tutorial")
    assert any("From measurements to a model, through chat." in item.value for item in app.markdown)

    app.button(key="labcd_intro_tutorial").click().run()
    assert not app.exception
    assert app.session_state["labcd_onboarding_mode"] == "tutorial"
    expected = [
        "Add measurements",
        "Confirm what the data means",
        "Review the model and search",
        "Run, inspect, and ask",
    ]
    for index, title in enumerate(expected):
        assert any(title in item.value for item in app.markdown)
        assert any("labcd-tour-visual" in item.value for item in app.markdown)
        if index < len(expected) - 1:
            app.button(key="labcd_tour_next").click().run()
            assert not app.exception
    app.button(key="labcd_tour_next").click().run()
    assert not app.exception
    assert app.session_state["labcd_onboarding_seen"] is True


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
    assert any(item.label == "Visual evidence from your data" for item in app.expander)
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


def test_trajectory_picture_click_advances_without_accept_button(app_storage, monkeypatch):
    chat = ready_chat()
    core.begin_setup(chat)
    core.add_message(chat, "assistant", core.setup_prompt(chat), kind="setup_question", stage="trajectory")

    def clicked_picture(**kwargs):
        value = "One continuous run" if "trajectory" in kwargs["key"] else None
        return SimpleNamespace(value=value)

    monkeypatch.setattr(ui.data_ui, "_PICTURE_CHOICE", clicked_picture)
    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert app.session_state["conversation"]["setup"]["stage"] == "angle"
    assert any(message["role"] == "user" and message["content"] == "One continuous run"
               for message in app.session_state["conversation"]["messages"])
    assert not any(button.label.startswith("Continue with One continuous run") for button in app.button)


def test_expired_setup_question_uses_default_and_moves_on_without_client_action(app_storage):
    chat = ready_chat()
    core.begin_setup(chat)
    chat["setup"]["stage"] = "angle"
    question = core.add_message(
        chat, "assistant", core.setup_prompt(chat), kind="setup_question", stage="angle",
    )
    question["asked_at"] = ui.time.time() - ui.QUESTION_SKIP_DELAY_SECONDS - 1
    core.save_chat(chat)

    script = '''
import streamlit as st
from frontend_streamlit import ui_conversation as ui
ui._render_client_question_skip_fragment(st.session_state["chat"], st.session_state["question_id"])
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.session_state["question_id"] = question["id"]
    app.run()

    resolved = app.session_state["chat"]
    saved_question = next(message for message in resolved["messages"] if message["id"] == question["id"])
    assert not app.exception
    assert resolved["setup"]["stage"] == "complete"
    assert resolved["settings"]["auto_detect_angles"] is True
    assert saved_question["question_resolution"]["source"] == "data_default"
    assert not any(message.get("kind") == "question_skipped" for message in resolved["messages"])
    assert not any(message["role"] == "user" and "skip this question" in message["content"].lower()
                   for message in resolved["messages"])
    assert any(message.get("kind") == "plan" and "skip this question and continue" in message["content"]
               for message in resolved["messages"])
    assert not any(item.label == "Use data default" for item in app.button)
    assert not any("Ready when you are" in item.value for item in app.caption)


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


def test_right_file_panel_is_viewport_sized_with_an_independent_scroll_area():
    from frontend_streamlit.ui_conversation_theme import CSS

    assert "bottom:180px!important" in CSS
    assert "height:calc(100dvh - 252px)!important" in CSS
    assert "bottom:150px!important" in CSS
    assert "height:calc(100dvh - 208px)!important" in CSS
    assert "position:sticky!important; top:-8px!important" in CSS
    assert '.st-key-conversation_file_panel [data-testid="stVerticalBlockBorderWrapper"]' not in CSS
    assert "overflow-y:auto!important" in CSS
    assert "overscroll-behavior:contain" in CSS
    assert "height:auto!important; max-height:none!important; min-height:0!important" not in CSS


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


def test_run_files_show_report_zip_then_sandbox_without_separate_code_card(app_storage):
    run_dir = app_storage / "artifact_run"
    deployment = run_dir / "deployment"
    report = run_dir / "report"
    deployment.mkdir(parents=True)
    report.mkdir()
    (deployment / "deployed_controller_plant.py").write_text("# deployable controller\n")
    (deployment / "NN.py").write_text("# inference example\n")
    (report / "System_Report_plant.pdf").write_bytes(b"%PDF-1.4")
    with zipfile.ZipFile(run_dir / "SystemID_RunResults_test.zip", "w") as archive:
        archive.writestr("README.md", "run package")
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed", "cycles_run": 1}))
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    core.add_message(chat, "assistant", "The run completed.", kind="result", run_dir=str(run_dir))

    artifacts = [item for item in ui._file_items(chat) if item["id"].startswith("artifact:")]
    assert [item["label"] for item in artifacts] == ["PDF report", "Results ZIP", "Sandbox"]
    assert all(item["kind"] != "code" for item in artifacts)

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    run_file_buttons = [button for button in app.button
                        if button.key and button.key.startswith("open_chat_file_")]
    assert [button.label for button in run_file_buttons] == ["PDF report", "Results ZIP", "Sandbox"]


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


def test_completed_manifest_recovers_missing_result_message(app_storage):
    run_dir = app_storage / "runs" / "finished_without_chat_update"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "cycles_run": 2, "best_mse": 0.00044,
        "latency_ms": 0.65,
    }), encoding="utf-8")
    chat = core.new_chat()
    chat["run_dir"] = str(run_dir)
    core.add_message(chat, "assistant", "I'm starting the identification run.")

    app = new_app()
    app.session_state["conversation"] = chat
    app.run()

    assert not app.exception
    assert any("The run finished after 2 cycles" in item.value for item in app.markdown)
    assert sum(message.get("kind") == "result" for message in chat["messages"]) == 1

    app.run()
    assert sum(message.get("kind") == "result" for message in chat["messages"]) == 1


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


def test_talk_to_agent_shows_cycle_evidence_messages_and_applies_next_candidate_edits():
    class ReviewRunner:
        action = None
        edits = None
        def respond_to_checkpoint(self, action, edits=None):
            self.action, self.edits = action, edits
            return True

    script = '''
import streamlit as st
from frontend_streamlit.ui_conversation import _talk_to_agent_section
_talk_to_agent_section({"id": "test-chat"}, st.session_state["task"], st.session_state["state"])
'''
    runner = ReviewRunner()
    task = {"runner": runner}
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["task"] = task
    app.session_state["state"] = {"checkpoint": {
        "id": "early_results_2", "phase": "early_results", "cycle": 2,
        "max_cycles": 6, "best_mse": 0.02,
        "recent_results": [
            {"cycle": 1, "train_mse": 0.03, "val_mse": 0.04, "training_seconds": 1.0},
            {"cycle": 2, "train_mse": 0.02, "val_mse": 0.02, "training_seconds": 1.2,
             "state_validation_mse": {"s_position": 0.025}},
        ],
        "next_config": {"learning_rate": 0.001, "hidden_layers": [64, 48],
                        "dropout_rate": 0.1, "weight_decay": 0.001,
                        "batch_size": 64, "patience": 20, "epochs": 120},
        "candidate_bounds": {"learning_rate_min": 0.0001, "learning_rate_max": 0.002,
                             "hidden_size_min": 32, "hidden_size_max": 192,
                             "num_layers_min": 1, "num_layers_max": 3},
        "state_validation_mse": {"s_position": 0.025},
    }, "history": [], "activity": [
        {"id": "critic:2", "label": "Critic reviewed cycle 2 and proposed adjustments",
         "state": "complete", "timestamp": "2026-10-12T12:00:00+00:00",
         "details": {"reasoning": "Validation error improved; retain a compact topology."}},
        {"id": "actor:2", "label": "Actor finished cycle 2", "state": "complete",
         "timestamp": "2026-10-12T11:59:00+00:00", "details": {"val_mse": 0.02}},
    ]}
    app.run()
    assert not app.exception
    assert app.button(key="talk_to_agent_toggle_test-chat")
    assert not app.metric
    app.button(key="talk_to_agent_toggle_test-chat").click().run()
    assert not app.exception
    assert app.button(key="talk_to_agent_close_test-chat")
    assert any(item.label == "Validation MSE" and item.value == "0.02" for item in app.metric)
    assert any("Validation RMSE" in str(item.value) for item in app.dataframe)
    assert any("Validation error improved" in item.value for item in app.text)

    app.button(key="talk_to_agent_close_test-chat").click().run()
    assert not app.exception
    assert app.session_state["talk_to_agent_open_test-chat"] is False
    assert not app.metric
    app.button(key="talk_to_agent_toggle_test-chat").click().run()
    assert not app.exception

    prefix = "next_candidate_test-chat_early_results_2"
    app.number_input(key=f"{prefix}_width_0").set_value(96).run()
    app.number_input(key=f"{prefix}_learning_rate").set_value(0.0005).run()
    app.number_input(key=f"{prefix}_epochs").set_value(2).run()
    app.button(key=f"{prefix}_submit").click().run()
    assert not app.exception
    assert runner.action == "custom"
    assert runner.edits["hidden_layers"] == [96, 48]
    assert runner.edits["learning_rate"] == 0.0005
    assert runner.edits["epochs"] == 2


def test_live_agent_context_carries_the_recorded_agent_findings():
    context = ui._live_agent_context({
        "activity": [{
            "label": "Inspector checked the dataset", "state": "complete",
            "timestamp": "2026-10-13T17:00:00+00:00",
            "details": {"outcome": "reviewed", "response": "No duplicate timestamps were found.",
                        "unrelated_private_field": "not forwarded"},
        }],
        "critic": [{"cycle": 1, "reasoning": "Validation error improved."}],
        "history": [{"cycle": 1, "val_mse": 0.03}],
    }, {"phase": "early_results", "cycle": 1}, 1)

    assert context["recent_agent_activity"][0]["details"]["response"] == "No duplicate timestamps were found."
    assert context["parameter_controls"]["chat_applies_changes"] is False
    assert context["parameter_controls"]["in_progress_candidate_mutable"] is False
    assert context["parameter_controls"]["next_candidate_editor_available"] is True
    assert "epoch limit" in context["parameter_controls"]["next_candidate_parameters"]
    assert "unrelated_private_field" not in context["recent_agent_activity"][0]["details"]
    assert context["critic_reviews"][0]["reasoning"] == "Validation error improved."


def test_talk_to_agent_chat_keeps_live_run_running_and_shows_reply():
    script = '''
import streamlit as st
from types import SimpleNamespace
from frontend_streamlit import ui_conversation as ui

class DemoConversationJob:
    def __init__(self, snapshot, question, *, purpose):
        self.snapshot, self.question, self.purpose = snapshot, question, purpose
        self.answer = None
        self.error = None
        self.running = False
        self.phase = "Preparing evidence"
    def start(self):
        self.answer = {"answer": "Cycle 3 is still running while I answer in this panel.", "evidence": []}

ui.agent.ConversationJob = DemoConversationJob
ui.request_stop = lambda: st.session_state.__setitem__("stop_requested", True)

@st.fragment
def render_agent_panel():
    _task = {"runner": st.session_state["runner"]}
    ui._talk_to_agent_section(
        st.session_state["chat"], _task, st.session_state["run_state"]
    )

render_agent_panel()
'''

    class ActiveRunner:
        running = True
        stop_requested = False
        run_dir = "active-run"

    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["runner"] = ActiveRunner()
    app.session_state["chat"] = {
        "id": "talk-agent-chat-test",
        "messages": [{"role": "user", "content": "Start training"}],
        "settings": {"max_cycles": 7},
    }
    app.session_state["run_state"] = {
        "checkpoint": {
            "id": "early_results_3", "phase": "early_results", "cycle": 3,
            "best_mse": 0.02,
            "recent_results": [
                {"cycle": 2, "train_mse": 0.03, "val_mse": 0.025, "training_seconds": 1.0},
                {"cycle": 3, "train_mse": 0.02, "val_mse": 0.02, "training_seconds": 1.1},
            ],
        },
        "history": [],
        "activity": [],
    }
    app.run()
    assert not app.exception

    app.button(key="talk_to_agent_toggle_talk-agent-chat-test").click().run()
    assert not app.exception
    assert app.button(key="talk_agent_suggest_score_talk-agent-chat-test").label == "Explain this cycle"
    assert app.button(key="talk_agent_suggest_agents_talk-agent-chat-test").label == "Agent decisions"
    assert app.button(key="talk_agent_suggest_tune_talk-agent-chat-test").label == "Tune the next cycle"
    app.text_input(key="talk_agent_chat_talk-agent-chat-test_early_results_3_input").set_value(
        "Can you summarize Cycle 3?"
    ).run()
    next(button for button in app.button if button.label == "Send").click().run()

    assert not app.exception
    assert app.session_state["run_state"]["talk_agent_jobs"]["live_run"].snapshot["live_run_dir"] == "active-run"
    assert app.session_state["runner"].running is True
    assert not app.session_state.get("stop_requested", False)
    assert any("Can you summarize Cycle 3?" in item.value for item in app.markdown)

    # The live fragment polls the background answer; simulate its next tick.
    app.run()
    assert not app.exception
    assert any("Cycle 3 is still running" in item.value for item in app.markdown)
    assert [item["content"] for item in app.session_state["chat"]["run_agent_conversations"]["live_run"]] == [
        "Can you summarize Cycle 3?", "Cycle 3 is still running while I answer in this panel."
    ]


def test_talk_to_agent_is_hidden_during_regular_training():
    script = '''
import streamlit as st
from frontend_streamlit.ui_conversation import _talk_to_agent_section
_talk_to_agent_section({"id": "regular-chat"}, {}, st.session_state["state"])
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["state"] = {
        "checkpoint": None, "history": [], "activity": [],
    }
    app.session_state["talk_to_agent_open_regular-chat"] = True
    app.run()

    assert not app.exception
    assert not any(button.label == "Talk to the agent" for button in app.button)
    assert app.session_state["talk_to_agent_open_regular-chat"] is False


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
    assert any("diagnostic request from the region where LabCD is running" in m.value for m in app.markdown)
    assert any("Connection details" in exp.label for exp in app.expander)


def test_other_conversation_cannot_stop_active_run(app_storage, monkeypatch):
    stopped = []
    monkeypatch.setattr(ui, "request_stop", lambda:stopped.append(True))
    core.REGISTRY.training["other"] = {"runner":SimpleNamespace(running=True)}
    app = new_app()
    app.chat_input[0].set_value("stop").run()
    assert not stopped
    assert any("no active training" in m.value for m in app.markdown)


def test_live_training_uses_compact_chat_progress_card(app_storage, monkeypatch):
    from time import time

    rendered_activity = {}
    monkeypatch.setattr(
        ui.activity, "render",
        lambda items, **kwargs: rendered_activity.update(items=list(items), **kwargs),
    )
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
    assert any("System ID · this run" in item.value for item in app.markdown)
    assert any("Training candidates and comparing their validation scores" in item.value for item in app.caption)
    assert any("Now · Actor is training cycle 1" in item.value for item in app.caption)
    assert not any("Validation MSE by candidate cycle" in item.value for item in app.markdown)
    assert not any("WAITING FOR CYCLE 1 VALIDATION SCORE" in item.value for item in app.markdown)
    assert any("run-learning-snapshot" in item.value for item in app.markdown)
    assert not app.get("progress")
    assert len(app.get("status")) == 1
    assert app.get("status")[0].label == ":shimmer[Thinking]"
    assert [item["label"] for item in rendered_activity["items"]] == [
        "Searching for the best model", "Actor is training cycle 1",
    ]
    assert rendered_activity["heading"] is None
    assert rendered_activity["container_height"] == 280
    assert any(item.label == "Search cycles" and item.value == "1 / 7" for item in app.metric)


def test_live_training_keeps_illustrated_tour_before_cycle_one(app_storage):
    from time import time

    chat = {"id": "before-cycle-one", "settings": {"max_cycles": 7}}
    state = {
        "stage": "Initializer Agent", "progress": 0.45, "history": [], "log": "",
        "critic": [], "result": None, "checkpoint": None,
        "activity": [{"id": "stage:Initializer Agent", "label": "Preparing starting model settings",
                      "state": "running", "details": {}}],
    }
    core.REGISTRY.training[chat["id"]] = {
        "runner": SimpleNamespace(running=True), "started": time(), "state": state,
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
    assert any("run-learning-snapshot" in item.value for item in app.markdown)
    assert not any("Validation MSE by candidate cycle" in item.value for item in app.markdown)
    assert not app.get("vega_lite_chart")


def test_live_training_shows_cycle_block_diagram_and_collapsible_critic_reviews(app_storage, monkeypatch):
    from time import time

    rendered_activity = {}
    monkeypatch.setattr(
        ui.activity, "render",
        lambda items, **kwargs: rendered_activity.update(items=list(items), **kwargs),
    )
    chat = {"id": "critic-progress", "settings": {"max_cycles": 7}}
    state = {
        "stage": "Tuning cycles", "progress": 0.64, "log": "", "result": None,
        "checkpoint": None,
        "history": [
            {"cycle": 1, "train_mse": 0.025, "val_mse": 0.031, "rmse": 0.176,
             "training_seconds": 2.4, "is_best": True},
            {"cycle": 2, "train_mse": 0.018, "val_mse": 0.022, "rmse": 0.148,
             "training_seconds": 2.1, "is_best": True},
        ],
        "critic": [
            {"cycle": 1, "diagnosis": "UNDERFITTING", "status": "NEEDS_IMPROVEMENT",
             "lr_dir": "increase", "lr_step": 0.0002, "hidden_layers": [64, 64],
             "reasoning": "Cycle 1 underfit the validation data; a wider candidate may capture more of the measured dynamics."},
            {"cycle": 2, "diagnosis": "IMPROVING", "status": "CONTINUE",
             "lr_dir": "stay", "lr_step": 0.0, "hidden_layers": [64, 64],
             "reasoning": "Cycle 2 reduced validation error and remains close to its training score, so keep this capacity for another comparison."},
        ],
        "activity": [
            {"id": "stage:Tuning cycles", "label": "Searching for the best model",
             "state": "complete", "details": {}},
            {"id": "actor:3", "label": "Actor is training cycle 3",
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
    diagram = next(item.value for item in app.markdown if "run-process-diagram" in item.value)
    assert "CANDIDATE LOOP" in diagram
    assert "Critic feedback shapes the next candidate" in diagram
    assert "run-search-agent--active" in diagram
    assert any("Validation MSE by candidate cycle" in item.value for item in app.markdown)
    assert not any("run-learning-snapshot" in item.value for item in app.markdown)
    assert app.get("vega_lite_chart")
    assert len(app.get("status")) == 1
    assert app.get("status")[0].label == ":shimmer[Thinking]"
    assert "Critic reviews · 2 cycles" in {item.label for item in app.expander}
    assert [item["label"] for item in rendered_activity["items"]] == [
        "Searching for the best model", "Actor is training cycle 3",
    ]
    assert any("Cycle 1 underfit the validation data" in item.value for item in app.markdown)
    assert any("Cycle 2 reduced validation error" in item.value for item in app.markdown)
    assert any("WHAT THE CRITIC SUGGESTED" in item.value for item in app.markdown)
    assert any(item.label == "Validation MSE" and item.value == "0.031" for item in app.metric)


def test_initializer_review_is_inside_talk_to_agent_before_training(app_storage):
    from time import time

    chat = {"id": "initializer-review", "settings": {"max_cycles": 7}}
    checkpoint = {
        "id": "initializer_1", "phase": "initializer", "goal": "balanced",
        "config": {"hidden_layers": [32, 32], "learning_rate": 0.001,
                   "dropout_rate": 0.1, "weight_decay": 0.001,
                   "reasoning": "A compact first candidate fits this dataset."},
        "setup": {"architecture": "LSTM", "run_mode": "fast", "max_cycles": 7,
                  "history_steps": 5, "max_hours": 0.5},
        "dataset": {"samples": 100, "train_samples": 70, "validation_samples": 15,
                    "test_samples": 15, "states": ["s_position"], "inputs": ["a_force"],
                    "median_dt": 0.1, "varying_states": ["s_position"],
                    "varying_inputs": ["a_force"]},
    }
    state = {
        "stage": "Initializer Agent", "progress": 0.5, "history": [], "log": "",
        "critic": [], "result": None, "checkpoint": checkpoint,
        "activity": [{"id": "stage:Initializer Agent", "label": "Preparing starting model settings",
                      "state": "running", "details": {"name": "Initializer Agent"}}],
    }
    core.REGISTRY.training[chat["id"]] = {
        "runner": SimpleNamespace(running=True), "started": time(), "state": state,
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
    assert app.button(key="talk_to_agent_toggle_initializer-review")
    assert not any("Your model is taking shape" in item.value for item in app.markdown)
    assert any(item.label == "Search cycles" and item.value == "0 / 7" for item in app.metric)
    assert not any(item.label == "Training details" for item in app.expander)

    app.button(key="talk_to_agent_toggle_initializer-review").click().run()
    assert not app.exception
    assert app.button(key="talk_to_agent_close_initializer-review")
    assert any("INITIAL MODEL PROPOSAL · BEFORE TRAINING" in item.value for item in app.caption)
    assert app.button(key="initial_review_initializer-review_initializer_1_approve")

    # Once the checkpoint is resolved, the live candidate-training card appears.
    state["checkpoint"] = None
    state["stage"] = "Tuning cycles"
    app.run()
    assert not app.exception
    assert any("System ID · this run" in item.value for item in app.markdown)
    assert any(item.label == "Search cycles" for item in app.metric)
    assert not any(button.label == "Talk to the agent" for button in app.button)
    assert app.session_state["talk_to_agent_open_initializer-review"] is False


def test_column_role_ui_marks_input_and_state_rates_as_needed(app_storage):
    chat = {
        "id": "required-input-copy",
        "dataset": {"column_review": {
            "options": ["s_position", "command"],
            "suggested_states": ["s_position"], "suggested_actions": [],
            "suggested_ignored": [], "time_required": False,
        }},
    }
    script = '''
import streamlit as st
from frontend_streamlit import ui_conversation as ui
ui._column_review_controls(st.session_state["chat"], "required-input")
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()

    assert not app.exception
    input_picker = app.multiselect(key="column_review_inputs_required-input")
    assert "required" in input_picker.label
    assert any("At least one input is required" in item.value and "We want a time-rate (`xdot_*`) for every state" in item.value
               for item in app.caption)
    # Let the form submit so a missing role produces a clear message instead
    # of trapping the client behind a disabled submit button.
    assert not app.button(key="column_review_confirm_required-input").disabled


def test_column_review_marks_input_and_state_rates_as_needed(app_storage):
    chat = {
        "id": "required-input-copy",
        "dataset": {"column_review": {
            "options": ["s_position", "command"],
            "suggested_states": ["s_position"], "suggested_actions": [],
            "suggested_ignored": [], "time_required": False,
        }},
    }
    script = '''
import streamlit as st
from frontend_streamlit import ui_conversation as ui
ui._column_review_controls(st.session_state["chat"], "required-input")
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["chat"] = chat
    app.run()

    assert not app.exception
    input_picker = app.multiselect(key="column_review_inputs_required-input")
    assert "required" in input_picker.label
    assert any("At least one input is required" in item.value and "We want a time-rate (`xdot_*`) for every state" in item.value
               for item in app.caption)
    # Keep submission possible so a missing required role produces a clear
    # validation message instead of trapping the client in a disabled form.
    assert not app.button(key="column_review_confirm_required-input").disabled
