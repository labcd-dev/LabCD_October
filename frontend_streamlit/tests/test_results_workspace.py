"""Compare distinct runs, preview real artifacts, and preserve panel choices."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from frontend_streamlit import ui_history as hist
from frontend_streamlit import ui_results_workspace as workspace


def make_run(base, suffix, score=0, mse=0):
    run = base / f"run_20260928_12000{suffix}_oscillator"
    (run / "deployment").mkdir(parents=True)
    (run / "deployment" / "NN.py").write_text(f"# Run {suffix}\nprint('preview only')\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "env_name": "Same name", "success_score": score,
        "best_mse": mse, "best_rmse": 0, "elapsed_seconds": 120,
        "architecture": "MLP", "activation": "relu", "best_config": {"hidden_layers": [16]},
        "abstract": "A report summary", "conclusion": "Held-out tests passed",
    }), encoding="utf-8")
    return run


def test_zero_metrics_and_missing_training_time_remain_distinct():
    rows = {row["Metric"]: row for row in workspace.compare_rows(
        {"success_score": 0, "best_mse": 0, "best_rmse": float("nan")},
        {"success_score": 50, "best_mse": 0.01, "training_seconds": 60})}
    assert rows["Score / 100"]["Run A"] == "0.00"
    assert rows["Score / 100"]["B vs A"] == "+50.00 points"
    assert rows["Validation MSE"]["Run A"] == "0.000e+00"
    assert rows["Validation MSE"]["B vs A"] == "Higher"
    assert rows["Validation RMSE"]["Run A"] == "Not recorded"
    assert rows["Training time"]["Run A"] == "Not recorded"
    assert rows["Training time"]["Run B"] == "1.0 min"


def test_compare_cannot_select_same_run_and_handles_archive_filter(tmp_path):
    first = make_run(tmp_path, "1", 50, 0.01)
    second = make_run(tmp_path, "2", 70, 0.001)
    third = make_run(tmp_path, "3", 90, 0.0001)
    hist.update_metadata(third, archived=True)
    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit.ui_results_workspace import render_compare
render_compare(st.session_state["test_output_dir"])
''')
    app.session_state["test_output_dir"] = str(tmp_path)
    app.run()
    assert not app.exception
    assert len(app.selectbox(key="compare_a").options) == 2
    assert len(app.selectbox(key="compare_b").options) == 1
    assert app.session_state["compare_a"] != app.session_state["compare_b"]
    app.selectbox(key="compare_a").set_value(str(first.resolve())).run()
    assert not app.exception
    assert app.session_state["compare_b"] == str(second.resolve())
    assert app.dataframe[0].value.loc[0, "B vs A"] == "+20.00 points"
    app.toggle(key="compare_archived").set_value(True).run()
    assert not app.exception
    assert len(app.selectbox(key="compare_a").options) == 3
    assert len(app.selectbox(key="compare_b").options) == 2


def test_preview_switches_report_and_code_without_executing_it(tmp_path):
    run = make_run(tmp_path, "1")
    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit.ui_results_workspace import render_panel
render_panel(st.session_state["test_output_dir"])
''')
    app.session_state["test_output_dir"] = str(tmp_path)
    app.run()
    assert not app.exception
    assert app.info[0].value == "No plots are available for this run."
    app.button_group(key="preview_type").set_value("Report").run()
    assert not app.exception
    assert app.expander[0].label == "Report summary"
    assert any("A report summary" in value.value for value in app.markdown)
    app.button_group(key="preview_type").set_value("Code").run()
    assert not app.exception
    assert app.code[0].value.splitlines() == ["# Run 1", "print('preview only')"]
    assert app.session_state["preview_run"] == str(run.resolve())


def test_main_panel_toggle_preserves_run_choice(tmp_path):
    first, second = make_run(tmp_path, "1"), make_run(tmp_path, "2")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30)
    app.session_state["output_dir"] = str(tmp_path)
    app.session_state["section"] = "Compare"
    app.run()
    assert not app.exception
    assert not any(box.key == "preview_run" for box in app.selectbox)
    app.button(key="toggle_results_panel").click().run()
    assert not app.exception
    assert app.session_state["results_panel_open"]
    app.selectbox(key="preview_run").set_value(str(first.resolve())).run()
    app.button_group(key="preview_type").set_value("Code").run()
    app.button(key="toggle_results_panel").click().run()
    assert not app.session_state["results_panel_open"]
    app.button(key="toggle_results_panel").click().run()
    assert not app.exception
    assert app.session_state["preview_run"] == str(first.resolve())
    assert app.button_group(key="preview_type").value == "Code"
    assert app.code[0].value.splitlines()[0] == "# Run 1"


def test_empty_comparison_and_panel_are_usable(tmp_path):
    app = AppTest.from_string('''
import streamlit as st
from frontend_streamlit.ui_results_workspace import render_compare, render_panel
render_compare(st.session_state["test_output_dir"])
render_panel(st.session_state["test_output_dir"])
''')
    app.session_state["test_output_dir"] = str(tmp_path)
    app.run()
    assert not app.exception
    assert "Complete two runs" in app.info[0].value
    assert not app.selectbox
