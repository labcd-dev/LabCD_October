import hashlib

import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from frontend_streamlit import ui_conversation as ui
from frontend_streamlit import ui_model_sandbox as sandbox


def test_profiles_build_sine_step_and_ramp_on_selected_input_only():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    baseline = np.array([2.0, -1.0])

    sine = sandbox._input_profile("Sine", times, baseline, 0, 3.0, 1.0, frequency=1.0)
    step = sandbox._input_profile("Step", times, baseline, 1, 4.0, 1.0, change_time=0.5)
    ramp = sandbox._input_profile("Ramp", times, baseline, 0, 2.0, 1.0)

    np.testing.assert_allclose(sine[:, 0], [2, 5, 2, -1, 2], atol=1e-7)
    np.testing.assert_allclose(sine[:, 1], [-1] * 5)
    np.testing.assert_allclose(step[:, 1], [-1, -1, 3, 3, 3])
    np.testing.assert_allclose(step[:, 0], [2] * 5)
    np.testing.assert_allclose(ramp[:, 0], [2, 2.5, 3, 3.5, 4])
    np.testing.assert_allclose(ramp[:, 1], [-1] * 5)


def test_multiple_profiles_vary_all_or_some_inputs_and_hold_the_rest():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    baseline = np.array([1.0, 2.0, 3.0])

    actions = sandbox._input_profiles(times, baseline, [
        {"input_index": 0, "profile": "Sine", "amplitude": 1.0, "frequency": 0.25},
        {"input_index": 2, "profile": "Ramp", "amplitude": -2.0},
    ])

    np.testing.assert_allclose(actions[:, 0], 1.0 + np.sin(2 * np.pi * 0.25 * times))
    np.testing.assert_allclose(actions[:, 1], [2.0] * len(times))
    np.testing.assert_allclose(actions[:, 2], [3.0, 2.5, 2.0, 1.5, 1.0])


def test_lstm_rollout_appends_each_prediction_to_context():
    class ActionRate:
        def __call__(self, states, actions, dt):
            return actions[:, -1, :], None

    data = {
        "states": np.array([[0.0], [0.0]], dtype=np.float32),
        "actions": np.array([[0.0], [0.0]], dtype=np.float32),
    }
    metadata = {"architecture": "LSTM", "history_steps": 2,
                "integrator": "EULER", "angle_indices": []}
    profile = np.array([[1.0], [2.0], [3.0], [3.0]])

    states = sandbox._rollout(ActionRate(), data, metadata, np.array([0.0]), profile, 1.0)

    np.testing.assert_allclose(states[:, 0], [0.0, 1.0, 3.0, 6.0])


def test_sandbox_reloads_pcnn_equation_and_exact_state_from_run(tmp_path):
    from backend_core.AgentSysID.model.dynamics_model import DynamicsModel

    equation = tmp_path / "physics_equation.py"
    equation.write_text(
        "import torch\n"
        "def compute_analytical_xdot(states, actions):\n"
        "    return torch.stack((states[:, 0] + actions[:, 0], 2 * states[:, 1]), dim=1)\n",
        encoding="utf-8",
    )
    original = DynamicsModel(2, 1, [8], architecture="MLP")
    original.set_scalers([0, 0], [1, 1], [0], [1], [0, 0], [1, 2])
    original.configure_physics(
        "pcnn", lambda states, actions: torch.stack((states[:, 0] + actions[:, 0], 2 * states[:, 1]), dim=1),
        hard_constraint_indices=[0],
    )
    with torch.no_grad():
        original.network[-1].bias.copy_(torch.tensor([0.5, 0.5]))
    weights = tmp_path / "model.pth"
    torch.save(original.state_dict(), weights)

    loaded = sandbox._load_model(
        str(tmp_path), str(weights), weights.stat().st_mtime_ns,
        __import__("json").dumps({
            "states": ["s_x", "s_y"], "actions": ["a_u"], "architecture": "MLP",
            "hidden_layers": [8], "activation": "relu", "dropout_rate": 0.0,
            "angle_indices": [], "physics_mode": "pcnn",
            "physics_equation_file": str(equation), "hard_constraint_indices": [0],
        }, sort_keys=True),
    )
    predicted, _ = loaded(
        torch.tensor([[0.2, 0.3]]), torch.tensor([[0.4]]), torch.tensor([[0.01]])
    )

    torch.testing.assert_close(predicted, torch.tensor([[0.6, 1.6]]))


def test_sandbox_renders_adjustable_duration_and_state_and_input_plots(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_path = tmp_path / "data.csv"
    weights_path = tmp_path / "weights.pth"
    dataset_path.write_text("time,s_position,a_force,a_torque\n0,0,0,0\n", encoding="utf-8")
    weights_path.write_bytes(b"test")
    metadata = {
        "states": ["s_position"], "actions": ["a_force", "a_torque"], "angle_indices": [],
        "architecture": "MLP", "hidden_layers": [8], "activation": "tanh",
        "dropout_rate": 0.0, "history_steps": 1, "sample_period": 0.01,
        "time_column": "time", "integrator": "EULER",
    }
    data = {
        "states": np.array([[0.0], [0.1]], dtype=np.float32),
        "actions": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "last_state": np.array([0.1]), "last_action": np.array([0.0, 0.0]),
        "action_min": np.array([-1.0, -2.0]), "action_max": np.array([1.0, 2.0]),
        "sample_period": 0.01, "sample_count": 101,
    }

    class ConstantRate:
        def __call__(self, state, action, dt):
            return torch.ones((state.shape[0], 1)), None

    monkeypatch.setattr(sandbox, "_run_metadata", lambda _run_dir: (
        {}, {}, dataset_path, weights_path, metadata))
    monkeypatch.setattr(sandbox, "_read_recent_data", lambda *args: data)
    monkeypatch.setattr(sandbox, "_load_model", lambda *args: ConstantRate())
    key = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:10]
    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit import ui_model_sandbox as sandbox
sandbox.render(st.session_state["chat"])
''')
    app.session_state["chat"] = {"run_dir": str(run_dir)}
    app.run()

    assert not app.exception
    assert app.slider(key=f"sandbox_{key}_duration")
    assert app.multiselect(key=f"sandbox_{key}_inputs").value == ["a_force", "a_torque"]
    assert app.segmented_control(key=f"sandbox_{key}_input_0_profile").value == "Sine"
    assert app.segmented_control(key=f"sandbox_{key}_input_1_profile").value == "Sine"
    assert any("Predicted state response" in item.value for item in app.markdown)
    assert any("Applied input profiles" in item.value for item in app.markdown)
    app.multiselect(key=f"sandbox_{key}_inputs").set_value(["a_torque"]).run()
    assert not app.exception
    assert app.multiselect(key=f"sandbox_{key}_inputs").value == ["a_torque"]
    app.segmented_control(key=f"sandbox_{key}_input_1_profile").set_value("Step").run()
    assert not app.exception
    assert any("Applied input profiles" in item.value for item in app.markdown)
    app.multiselect(key=f"sandbox_{key}_inputs").set_value([]).run()
    assert not app.exception
    assert any("No inputs selected" in item.value for item in app.info)


def test_completed_run_adds_sandbox_file_and_opens_preview_on_click(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text('{"status":"completed"}', encoding="utf-8")
    monkeypatch.setattr(ui.hist, "existing_files", lambda _entry: {})
    opened = []
    monkeypatch.setattr(sandbox, "render", lambda chat: opened.append(chat["run_dir"]))
    chat = {"id": "sandbox-file-test", "run_dir": str(run_dir), "messages": []}

    item = sandbox.file_item(chat)
    assert item["label"] == "Sandbox"
    assert item["kind"] == "sandbox"
    assert item["id"].startswith("artifact:")

    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit import ui_conversation as ui
ui._artifacts(st.session_state["chat"], ui._file_items(st.session_state["chat"]))
''')
    app.session_state["chat"] = chat
    app.run()
    assert not app.exception
    next(button for button in app.button if button.label == "Sandbox").click().run()
    assert not app.exception
    assert opened == [str(run_dir)]
    assert app.session_state["conversation_selected_file"] == item["id"]
