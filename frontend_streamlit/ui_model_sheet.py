"""Chat-native display and editing for the first suggested measurement sheet."""
from __future__ import annotations

import copy
import hashlib
import re

import pandas as pd
import streamlit as st

from . import conversation_core as core


def _unit(channel):
    unit = str(channel.get("unit") or "").strip()
    return f" ({unit})" if unit else ""


def _header(value, prefix, fallback):
    text = str(value or "").strip().casefold()
    text = re.sub(r"^(?:time|s_|a_|xdot_)", "", text)
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:48] or fallback
    return f"{prefix}{slug}"


def _records(frame):
    rows = []
    for record in frame.to_dict(orient="records"):
        row = {}
        for name, value in record.items():
            if pd.isna(value):
                row[name] = None
            elif hasattr(value, "item"):
                row[name] = value.item()
            else:
                row[name] = value
        rows.append(row)
    return rows


def _saved_rows(chat, message_id):
    return ((chat.get("starter_sheet_data") or {}).get(message_id) or {}).get("rows") or []


def _store_rows(chat, message_id, frame, visible_columns):
    all_rows = copy.deepcopy(_saved_rows(chat, message_id))
    edited = _records(frame)
    merged = []
    for index, row in enumerate(edited):
        original = all_rows[index] if index < len(all_rows) else {}
        original.update({name: row.get(name) for name in visible_columns})
        merged.append(original)
    saved = chat.setdefault("starter_sheet_data", {}).get(message_id)
    value = {"rows": merged}
    if saved != value:
        chat["starter_sheet_data"][message_id] = value
        core.save_chat(chat)


def _append_channel(chat, message, role):
    schema = copy.deepcopy(message["starter_schema"])
    entries = schema["states"] if role == "state" else schema["inputs"]
    if len(entries) >= (5 if role == "state" else 4):
        st.toast(f"This starter sheet already has the maximum number of {role} columns.")
        return
    used_names = {item["name"] for item in entries}
    next_index = 1
    while f"{role}_{next_index}" in used_names:
        next_index += 1
    base = f"{role}_{next_index}"
    prefix = "s_" if role == "state" else "a_"
    column = _header(base, prefix, base)
    entries.append({"name": column[len(prefix):], "column": column,
                    "label": f"Measured state {next_index}" if role == "state" else f"Applied input {next_index}",
                    "unit": ""})
    if role == "state":
        schema["derivatives"].append({"name": column[len(prefix):],
            "column": f"xdot_{column[len(prefix):]}", "state_column": column,
            "label": f"Rate of change of measured state {len(entries)}", "unit": ""})
    message["starter_schema"] = schema
    core.save_chat(chat)
    st.rerun()


def _remove_channel(chat, message, role, index):
    schema = copy.deepcopy(message["starter_schema"])
    group = "states" if role == "state" else "inputs"
    entries = schema[group]
    if len(entries) <= 1:
        st.toast(f"Keep at least one measured {role} in the sheet.")
        return
    removed = entries.pop(index)
    removed_columns = {removed["column"]}
    if role == "state":
        removed_columns.update(item["column"] for item in schema["derivatives"]
                               if item.get("state_column") == removed["column"])
        schema["derivatives"] = [item for item in schema["derivatives"]
                                 if item.get("state_column") != removed["column"]]
    for row in _saved_rows(chat, message["id"]):
        for column in removed_columns:
            row.pop(column, None)
    message["starter_schema"] = schema
    core.save_chat(chat)
    st.rerun()


def _rename_schema(chat, message):
    schema = copy.deepcopy(message["starter_schema"])
    message_id = message["id"]
    renames = {}
    valid = True
    # Rates stay paired to their state; renaming the state above also renames
    # its xdot_* column automatically.
    state_column, input_column = st.columns(2, gap="medium")
    for container, group, prefix, label, role in (
        (state_column, "states", "s_", "State", "state"),
        (input_column, "inputs", "a_", "Input", "input"),
    ):
        with container:
            st.caption(f"{label}s")
            entries = schema.get(group, [])
            for index, channel in enumerate(entries):
                old = channel["column"]
                widget_id = hashlib.sha1(f"{group}:{old}".encode("utf-8")).hexdigest()[:10]
                field, remove = st.columns([5, 1], vertical_alignment="bottom")
                proposed = field.text_input(
                    f"{label} {index + 1}", value=old,
                    key=f"starter_column_{message_id}_{widget_id}",
                    help=channel.get("label") or old,
                )
                if remove.button(
                    "Remove", icon=":material/close:",
                    key=f"starter_remove_{role}_{message_id}_{index}",
                    disabled=len(entries) <= 1,
                ):
                    _remove_channel(chat, message, role, index)
                new = _header(proposed, prefix, f"{group}_{index + 1}")
                if new != old:
                    renames[old] = new
                    channel["column"] = new
                    channel["name"] = new.removeprefix(prefix)
                    if group == "states":
                        for derivative in schema.get("derivatives", []):
                            if derivative.get("state_column") == old:
                                derivative["state_column"] = new
                                old_default = f"xdot_{old.removeprefix('s_')}"
                                if derivative.get("column") == old_default:
                                    derivative["column"] = f"xdot_{new.removeprefix('s_')}"
                                    derivative["name"] = new.removeprefix("s_")

    headers = ["time", *(item["column"] for item in schema["states"]),
               *(item["column"] for item in schema["inputs"]),
               *(item["column"] for item in schema["derivatives"])]
    if len(headers) != len(set(headers)):
        valid = False
    if renames and valid:
        stored = chat.setdefault("starter_sheet_data", {}).get(message_id, {"rows": []})
        for row in stored.get("rows", []):
            for old, new in renames.items():
                if old in row:
                    row[new] = row.pop(old)
        message["starter_schema"] = schema
        chat["starter_sheet_data"][message_id] = stored
        core.save_chat(chat)
    return schema, valid


def render(chat, message):
    """Render a suggested schema and blank, editable measurement grid."""
    message_id = message["id"]
    if message.get("starter_fallback"):
        st.caption("Generic starting sheet")
    with st.container(border=True, key=f"starter_sheet_card_{message_id}"):
        st.markdown("**Suggested measurement sheet**")
        with st.expander("Edit columns", expanded=False):
            schema, valid_names = _rename_schema(chat, message)
            with st.container(horizontal=True, gap="small"):
                if st.button("Add state", icon=":material/add:", key=f"starter_add_state_{message_id}"):
                    _append_channel(chat, message, "state")
                if st.button("Add input", icon=":material/add:", key=f"starter_add_input_{message_id}"):
                    _append_channel(chat, message, "input")
        states, inputs, derivatives = schema["states"], schema["inputs"], schema["derivatives"]

        if not valid_names:
            st.info("Resolve repeated names in Edit columns to use this sheet.")
            return None

        headers = ["time", *(item["column"] for item in states),
                   *(item["column"] for item in inputs),
                   *(item["column"] for item in derivatives)]

        config = {name: st.column_config.NumberColumn(name, format="%.6g") for name in headers}
        st.caption("Each row is one sample · include time, `s_*` states, `a_*` inputs, and a matching `xdot_*` rate per state.")

        stored_rows = _saved_rows(chat, message_id)
        if stored_rows:
            initial = pd.DataFrame(stored_rows).reindex(columns=headers)
        else:
            initial = pd.DataFrame([{name: None for name in headers} for _ in range(4)], columns=headers)
        key_hash = hashlib.sha1("\0".join(headers).encode()).hexdigest()[:10]
        edited = st.data_editor(
            initial, key=f"starter_sheet_grid_{message_id}_{key_hash}",
            num_rows="dynamic", column_config=config, hide_index=True,
            height=190, width="stretch",
        )
        _store_rows(chat, message_id, edited, headers)
        usable = edited.dropna(how="all").replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
        st.caption("Enter or paste measurements · one row per time sample · at least 10 rows.")
        with st.container(horizontal=True, gap="small"):
            st.download_button(
                "Download blank CSV template",
                data=(pd.DataFrame(columns=headers).to_csv(index=False).encode("utf-8")),
                file_name="labcd_measurement_template.csv", mime="text/csv",
                key=f"starter_download_{message_id}",
            )
            use_sheet = st.button(
                "Use these measurements", icon=":material/check:", type="primary",
                key=f"starter_use_sheet_{message_id}",
                disabled=not valid_names or usable.empty,
            )
        if use_sheet:
            return usable.reindex(columns=headers)
        return None
