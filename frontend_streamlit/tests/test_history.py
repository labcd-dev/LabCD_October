"""History organization persists without modifying pipeline artifacts."""

import datetime as dt
import json

from streamlit.testing.v1 import AppTest

from frontend_streamlit import ui_history as hist


def make_run(base, day, suffix="oscillator"):
    run = base / f"run_{day:%Y%m%d}_120000_{suffix}"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "env_name": suffix, "architecture": "LSTM",
        "cycles_run": 2, "success_score": 95,
    }), encoding="utf-8")
    (run / "model.pth").write_bytes(b"test model")
    return run


def test_preferences_survive_reload_and_archive_is_reversible(tmp_path):
    run = make_run(tmp_path, dt.date.today())
    manifest = (run / "run_manifest.json").read_bytes()
    assert hist.update_metadata(run, title="  My oscillator  ")
    assert hist.update_metadata(run, pinned=True)
    assert hist.update_metadata(run, archived=True)
    entry = hist.load_history(tmp_path)[0]
    assert hist.display_name(entry) == "My oscillator"
    assert entry["pinned"] and entry["archived"]
    assert hist.group_history([entry]) == []
    assert hist.group_history([entry], archived=True)[0][1] == [entry]
    assert hist.update_metadata(run, archived=False)
    entry = hist.load_history(tmp_path)[0]
    assert hist.group_history([entry])[0][0] == "Pinned"
    assert (run / "run_manifest.json").read_bytes() == manifest
    assert (run / "model.pth").read_bytes() == b"test model"
    assert not list(run.glob(".history-*.tmp"))


def test_date_groups_search_and_old_pins(tmp_path):
    today = dt.date(2026, 9, 28)
    current = make_run(tmp_path, today)
    make_run(tmp_path, today - dt.timedelta(days=1))
    old = make_run(tmp_path, today - dt.timedelta(days=90))
    assert hist.update_metadata(old, pinned=True, title="Favorite model")
    runs = hist.load_history(tmp_path, limit=None)
    assert [name for name, _ in hist.group_history(runs, today=today)] == ["Pinned", "Today", "Yesterday"]
    assert hist.group_history(runs, query="FAVORITE", today=today)[0][1][0]["run_dir"] == str(old)
    assert len(hist.group_history(runs, query="lstm", today=today)) == 3
    assert hist.update_metadata(current, archived=True)
    assert [name for name, _ in hist.group_history(hist.load_history(tmp_path), today=today)] == ["Pinned", "Yesterday"]


def test_invalid_metadata_does_not_hide_runs(tmp_path):
    run = make_run(tmp_path, dt.date.today())
    for content in ("{broken", "[]", '{"title":42,"pinned":"yes","archived":"false"}'):
        (run / "history_metadata.json").write_text(content, encoding="utf-8")
        entry = hist.load_history(tmp_path)[0]
        assert hist.display_name(entry) == "oscillator"
        assert hist.group_history([entry])
    assert not hist.update_metadata(run, title="   ")
    assert not hist.update_metadata(run, title="x" * 121)
    assert not hist.update_metadata(run, pinned="true")
    assert not hist.update_metadata(run, architecture="elsewhere")
    assert not hist.update_metadata(tmp_path / "missing", title="Missing")


def test_sidebar_select_rename_pin_archive_and_restore(tmp_path):
    run = make_run(tmp_path, dt.date.today())
    script = '''
import streamlit as st
from frontend_streamlit.agent_sysid_app import render_sidebar
st.session_state.setdefault("viewing", None)
render_sidebar(st.session_state["test_output_dir"], False)
'''
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["test_output_dir"] = str(tmp_path)
    app.run()
    assert not app.exception
    row = next(button for button in app.button if (button.key or "").startswith("history_open_"))
    run_key = row.key.removeprefix("history_open_")
    row.click().run()
    assert not app.exception
    assert app.session_state["viewing"]["run_dir"] == str(run)
    assert app.session_state["section"] == "Results"
    app.button(key=f"history_pin_{run_key}").click().run()
    assert not app.exception
    assert hist.load_history(tmp_path)[0]["pinned"]
    app.button(key=f"history_rename_{run_key}").click().run()
    assert not app.exception
    next(widget for widget in app.text_input if widget.label == "Run name").set_value("Spring [test] *v2*")
    next(button for button in app.button if button.label == "Save name").click().run()
    assert not app.exception
    assert hist.display_name(hist.load_history(tmp_path)[0]) == "Spring [test] *v2*"
    assert app.session_state["viewing"]["title"] == "Spring [test] *v2*"
    app.button(key=f"history_archive_{run_key}").click().run()
    assert not app.exception
    assert not any((button.key or "").startswith("history_open_") for button in app.button)
    app.toggle(key="history_archived").set_value(True).run()
    assert not app.exception
    assert app.button(key=f"history_archive_{run_key}").label == "Restore run"
    app.button(key=f"history_archive_{run_key}").click().run()
    assert not app.exception
    app.toggle(key="history_archived").set_value(False).run()
    app.text_input(key="history_search").set_value("Spring").run()
    assert not app.exception
    assert app.button(key=f"history_open_{run_key}")
    assert run.is_dir()
