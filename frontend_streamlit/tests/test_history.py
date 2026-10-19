"""History organization persists without modifying pipeline artifacts."""

import datetime as dt
import json
from pathlib import Path

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


def test_sidebar_select_rename_pin_archive_and_restore(tmp_path, monkeypatch):
    from frontend_streamlit import conversation_core as core
    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path)
    run = make_run(tmp_path, dt.date.today())
    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30).run()
    next(b for b in app.button if (b.key or "").startswith("legacy_run_")).click().run()
    assert not app.exception
    chat = app.session_state["conversation"]
    assert Path(chat["run_dir"]) == run
    next(b for b in app.button if b.label == "Pin conversation").click().run()
    assert core.read_chat(chat["id"])["pinned"]
    app.text_input(key=f"title_{chat['id']}").set_value("Spring [test] *v2*")
    next(b for b in app.button if b.label == "Rename").click().run()
    assert core.read_chat(chat["id"])["title"] == "Spring [test] *v2*"
    next(b for b in app.button if b.label == "Archive conversation").click().run()
    assert core.read_chat(chat["id"])["archived"]
    app.toggle(key="conversation_archived").set_value(True).run()
    app.button(key=f"conversation_open_{chat['id']}").click().run()
    next(b for b in app.button if b.label == "Restore conversation").click().run()
    assert not app.exception
    assert not core.read_chat(chat["id"])["archived"]
    assert run.is_dir()


def test_sidebar_rows_expose_pin_and_archive_actions(tmp_path, monkeypatch):
    from frontend_streamlit import conversation_core as core

    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path)
    saved = core.new_chat()
    saved["title"] = "Saved oscillator"
    core.save_chat(saved)

    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30).run()
    identity = saved["id"]
    assert app.button(key=f"history_pin_{identity}")
    assert app.button(key=f"history_archive_{identity}")

    app.button(key=f"history_pin_{identity}").click().run()
    assert core.read_chat(identity)["pinned"] is True
    app.button(key=f"conversation_open_{identity}").click().run()
    app.button(key=f"history_archive_{identity}").click().run()
    archived = core.read_chat(identity)
    assert archived["archived"] is True
    assert archived["pinned"] is True
    assert app.session_state["conversation"]["id"] != identity


def test_sidebar_projects_create_new_chats_and_move_existing_chats(tmp_path, monkeypatch):
    from frontend_streamlit import conversation_core as core

    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path)
    saved = core.new_chat()
    saved["title"] = "Unsorted experiment"
    core.save_chat(saved)

    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30).run()
    app.text_input(key="project_new_title").set_value("Pendulum study")
    app.button(key="project_create").click().run()
    assert not app.exception
    project = core.list_projects()[0]
    assert project["title"] == "Pendulum study"
    assert app.session_state["conversation"]["project_id"] == project["id"]

    app.selectbox(key=f"history_project_choice_{saved['id']}").set_value(project["id"]).run()
    app.button(key=f"history_move_project_{saved['id']}").click().run()
    assert not app.exception
    assert core.read_chat(saved["id"])["project_id"] == project["id"]
    assert app.button(key=f"project_new_chat_{project['id']}")


def test_permanent_remove_requires_confirmation_and_keeps_run_folder(tmp_path, monkeypatch):
    from frontend_streamlit import conversation_core as core

    monkeypatch.setattr(core, "CHAT_DIR", tmp_path / "chats")
    monkeypatch.setattr(core, "OUTPUT_DIR", tmp_path)
    run = tmp_path / "completed-run"
    run.mkdir()
    (run / "report.pdf").write_bytes(b"saved report")
    saved = core.new_chat()
    saved["title"] = "Completed experiment"
    saved["run_dir"] = str(run)
    core.save_chat(saved)

    app = AppTest.from_file(str(Path(__file__).parents[1] / "agent_sysid_app.py"), default_timeout=30).run()
    app.button(key=f"history_remove_{saved['id']}").click().run()
    assert not app.exception
    assert core.read_chat(saved["id"]) is not None
    app.button(key=f"history_remove_confirm_{saved['id']}").click().run()
    assert not app.exception
    assert core.read_chat(saved["id"]) is None
    assert (run / "report.pdf").read_bytes() == b"saved report"
