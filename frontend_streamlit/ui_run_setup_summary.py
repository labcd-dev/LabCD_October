"""Compact run setup summary with focused edit dialogs."""
from __future__ import annotations

import json
import math
import re
from html import escape

import streamlit as st

from backend_core.AgentSysID import config as cfg

try:
    from . import conversation_core as core, pinn_maker
except ImportError:
    import conversation_core as core
    import pinn_maker


GOALS = {
    "balanced": "Balanced",
    "accuracy": "Best validation fit",
    "speed": "Fast inference",
    "compact": "Smaller model",
}


def _starting_layers(settings: dict) -> list[int]:
    configured = settings.get("manual_starting_hidden_layers")
    low, high = int(settings["hidden_size_min"]), int(settings["hidden_size_max"])
    layers_low, layers_high = int(settings["num_layers_min"]), int(settings["num_layers_max"])
    layer_count = min(max(len(configured) if configured else 3, layers_low), layers_high)
    if configured:
        widths = [min(max(int(width), low), high) for width in configured[:layer_count]]
        widths.extend([widths[-1] if widths else min(max(128, low), high)] * (layer_count - len(widths)))
    else:
        widths = [min(max(128, low), high)] * layer_count
    return widths


def _live_network_settings(settings: dict, widget_prefix: str, widget_state) -> dict:
    """Apply the network editor's current widget values before rendering its hero."""
    live = dict(settings)
    defaults = _starting_layers(live)
    layer_min = int(live["num_layers_min"])
    layer_max = int(live["num_layers_max"])
    width_min = int(live["hidden_size_min"])
    width_max = int(live["hidden_size_max"])
    layer_count = int(widget_state.get(f"{widget_prefix}_layers", len(defaults)))
    layer_count = min(max(layer_count, layer_min), layer_max)
    widths = []
    for index in range(layer_count):
        default = defaults[min(index, len(defaults) - 1)]
        value = int(widget_state.get(f"{widget_prefix}_width_{index}", default))
        widths.append(min(max(value, width_min), width_max))
    live["manual_starting_hidden_layers"] = widths
    return live


def _starting_lr(settings: dict) -> float:
    configured = settings.get("manual_starting_lr")
    if configured is not None:
        return float(configured)
    return min(max(0.001, float(settings["learning_rate_min"])), float(settings["learning_rate_max"]))


def clear_editor_widget_state(chat_id: str) -> None:
    """Drop cached dialog inputs after a setting is changed from chat."""
    chat_marker = f"_{chat_id}_"
    for key in tuple(st.session_state):
        if key.startswith("setup_popup_") and chat_marker in key:
            del st.session_state[key]


def _proposal_settings(flow: dict, settings: dict) -> dict:
    """Return the saved, pre-edit setup baseline for this recommendation."""
    saved = flow.get("initializer_proposal")
    if isinstance(saved, dict):
        return core.RunSettings.model_validate({**settings, **saved}).model_dump()
    # Older saved conversations predate the snapshot. Reconstruct the fields
    # the setup agent actually recommended and keep other current defaults.
    changes = {
        "architecture": flow.get("recommended_architecture", settings["architecture"]),
        "lstm_seq_length": flow.get("recommended_history_steps", settings["lstm_seq_length"]),
        "run_mode": flow.get("recommended_effort", settings["run_mode"]),
        "max_cycles": flow.get("recommended_cycles", settings["max_cycles"]),
    }
    return core.apply_changes(settings, changes)


def _reset_setup_widgets(section: str, prefix: str, proposal: dict, roles: dict,
                         current_settings: dict) -> None:
    """Restore one editor's controls to the original setup proposal."""
    proposal = core.RunSettings.model_validate(proposal).model_dump()
    current = core.RunSettings.model_validate(current_settings).model_dump()

    def set_value(name: str, value) -> None:
        st.session_state[f"{prefix}_{name}"] = value

    if section == "signals":
        set_value("states", list(roles.get("states") or []))
        set_value("actions", list(roles.get("actions") or []))
    elif section == "model":
        set_value("model", proposal["architecture"])
    elif section == "history":
        set_value("steps", int(proposal["lstm_seq_length"]))
    elif section == "physics":
        set_value("enabled", proposal["physics_mode"].upper() if proposal["use_pinn"] else "Off")
        set_value("weight", float(proposal["pinn_loss_weight"]))
        set_value("hard_states", list(proposal.get("pcnn_hard_constraint_states") or []))
        set_value("constraints_initialized", True)
        hard_states = set(proposal.get("pcnn_hard_constraint_states") or [])
        for index, state in enumerate(roles.get("constraint_states") or roles.get("states") or []):
            set_value(f"hard_{index}", state in hard_states)
    elif section == "context":
        set_value("description", proposal["customer_description"])
    elif section == "goal":
        set_value("goal", proposal["optimization_goal"])
    elif section == "effort":
        mode = proposal["run_mode"] if proposal["run_mode"] in {"fast", "regular", "heavy"} else "regular"
        set_value("mode", mode)
    elif section == "cycles":
        cap = int(cfg.run_mode_limits(current["run_mode"])["max_cycles"])
        set_value("count", min(max(int(proposal["max_cycles"]), 1), cap))
    elif section == "network":
        low, high = int(current["hidden_size_min"]), int(current["hidden_size_max"])
        layer_low, layer_high = int(current["num_layers_min"]), int(current["num_layers_max"])
        widths = _starting_layers(proposal)
        layer_count = min(max(len(widths), layer_low), layer_high)
        widths = [min(max(int(widths[min(i, len(widths) - 1)]), low), high)
                  for i in range(layer_count)]
        set_value("layers", layer_count)
        for index in range(8):
            set_value(f"width_{index}", widths[min(index, len(widths) - 1)])
    elif section == "training":
        set_value("learning_rate", min(max(_starting_lr(proposal), current["learning_rate_min"]), current["learning_rate_max"]))
        set_value("epochs", int(proposal["epochs"]))
        set_value("batch", int(proposal["batch_size"]))
        set_value("patience", int(proposal["early_stop_patience"]))
        set_value("dropout", float(proposal["manual_dropout_rate"]))
        set_value("decay", float(proposal["manual_weight_decay"]))
        set_value("activation", proposal["manual_activation"])
    elif section == "targets":
        set_value("mse", float(proposal["mse_target"]))
        set_value("latency", float(proposal["customer_max_latency_ms"]))
    elif section == "advanced":
        for name in (
            "lr_min", "lr_max", "adaptive_reg", "lr_reduce", "lr_floor", "improve",
            "extension", "overfit", "width_min", "width_max", "layers_min", "layers_max",
            "initializer", "derivative_method", "tau", "filter", "lower", "upper",
            "angles_auto", "angle_indices", "multi", "shuffle", "chunk", "savgol_window",
            "savgol_poly", "smd_l1", "smd_l2", "split_times", "integrator", "horizon", "reset",
        ):
            setting_name = {
                "lr_min": "learning_rate_min", "lr_max": "learning_rate_max",
                "adaptive_reg": "adaptive_regularization", "lr_reduce": "lr_reduce_factor",
                "lr_floor": "lr_schedule_min_floor", "improve": "adaptive_improvement_threshold",
                "extension": "epoch_extension_steps", "overfit": "overfit_ratio_limit",
                "width_min": "hidden_size_min", "width_max": "hidden_size_max",
                "layers_min": "num_layers_min", "layers_max": "num_layers_max",
                "initializer": "choose_via_llm_initializer", "tau": "derivative_filter_tau",
                "derivative_method": "derivative_method",
                "filter": "use_state_filter", "angles_auto": "auto_detect_angles",
                "angle_indices": "angle_indices", "multi": "multi_trajectory",
                "shuffle": "shuffle_data", "chunk": "trajectory_chunk_size",
                "savgol_window": "savgol_window", "savgol_poly": "savgol_polyorder",
                "smd_l1": "smd_lambda_1", "smd_l2": "smd_lambda_2",
                "split_times": "manual_split_times", "integrator": "integrator_type",
                "horizon": "rollout_horizon", "reset": "reset_threshold",
            }.get(name)
            if setting_name:
                value = proposal[setting_name]
                if name == "angle_indices":
                    value = ", ".join(map(str, value))
                elif name == "split_times":
                    value = ", ".join(map(str, value))
                elif name == "reset":
                    value = ", ".join(map(str, value or []))
                set_value(name, value)
        low, high = proposal["auto_filter_percentiles"]
        set_value("lower", float(low))
        set_value("upper", float(high))
        set_value("json", json.dumps(proposal, indent=2))

    st.session_state[f"{prefix}_reset_notice"] = True


def _setup_logic(section: str, settings: dict, dataset: dict, flow: dict) -> str:
    """Build a concise, evidence-grounded explanation for this setup section."""
    if flow.get("agent_error"):
        return "The setup agent was unavailable, so these editable starting values use the current defaults. Check them against validation and held-out results after training."
    if section == "model" and flow.get("architecture_reason"):
        text = flow["architecture_reason"]
    elif section == "network":
        reason = flow.get("architecture_reason")
        layout = _network_label(settings)
        first_reason = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(reason or "")))[0]
        text = (f"{first_reason} The first candidate uses {layout}." if first_reason else
                f"The setup agent selected {settings['architecture']} from the measured profile. The first candidate uses {layout}.")
    elif section == "effort" and flow.get("effort_reason"):
        text = flow["effort_reason"]
    elif section == "history":
        steps = int(flow.get("recommended_history_steps", settings["lstm_seq_length"]))
        interval = (dataset.get("analysis") or {}).get("median_dt") or dataset.get("sample_period")
        span = f" (about {(steps - 1) * float(interval):.4g} s at the measured interval)" if interval else ""
        text = f"The setup agent proposed a {steps}-sample history{span} as an initial temporal context; held-out results will show whether it helps."
    elif section == "cycles":
        effort = str(flow.get("recommended_effort", settings["run_mode"]))
        cycles = int(flow.get("recommended_cycles", settings["max_cycles"]))
        text = f"The agent proposed up to {cycles} candidate cycles for {effort} search; its time limit may end the search sooner."
    elif section == "physics":
        cues = ((flow.get("physics_advisory") or {}).get("cues") or [])
        cue = str(cues[0]).strip().rstrip(".") if cues else ""
        mode = str(settings.get("physics_mode", "pinn")).upper()
        if mode == "PCNN":
            states = list(dataset.get("states") or [])
            exact = len(settings.get("pcnn_hard_constraint_states") or [])
            learned = max(0, len(states) - exact)
            learned_word = "state" if learned == 1 else "states"
            exact_word = "state" if exact == 1 else "states"
            text = (f"PCNN enforces {exact} selected {exact_word} exactly; the neural network learns the other {learned} {learned_word} directly from data. Choose only rules you trust to hold for every modeled step.")
        else:
            text = (f"The data check flagged {cue}. PINN adds the checked equation as a soft training penalty; review its assumptions and held-out fit."
                    if cues else "PINN adds the checked equation as a soft training penalty. Review its assumptions and compare held-out results.")
    elif section == "signals":
        text = (f"LabCD predicts derivatives for {len(dataset.get('states') or [])} selected states using "
                f"{len(dataset.get('inputs') or dataset.get('actions') or [])} controls plus time. These roles follow your confirmed data mapping.")
    elif section == "context":
        text = "The agent can use these optional device and unit notes to interpret signals. They do not change the uploaded measurements."
    elif section == "goal":
        preference = GOALS.get(settings["optimization_goal"], "Balanced")
        text = f"The search uses {preference.lower()} as its candidate-ranking preference. It cannot guarantee that the requested target will be met."
    elif section == "training":
        text = (f"The first candidate is capped at {settings['epochs']} epochs, batch size {settings['batch_size']}, "
                f"and patience {settings['early_stop_patience']}; these reviewed values go to the Initializer. "
                "Validation and held-out results show how well they work.")
    elif section == "targets":
        text = (f"The search aims for validation MSE {settings['mse_target']:.4g} and inference under "
                f"{settings['customer_max_latency_ms']:g} ms. These are targets, not performance claims.")
    elif section == "advanced":
        text = (f"The Initializer searches within the selected learning-rate and network bounds; "
                f"{settings['derivative_method'].replace('_', ' ')} estimates derivatives before the {settings['integrator_type']} rollout.")
    else:
        text = "The setup agent uses measured-data patterns to choose a cautious first experiment. Compare held-out results before drawing conclusions."
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(text))) if part.strip()]
    concise = " ".join(sentences[:2])
    if len(concise) > 300:
        concise = concise[:297].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
    return concise


def _preview_settings(section: str, prefix: str, settings: dict, states: list[str] | None = None) -> dict:
    """Reflect unsaved widget choices in the small hero illustration and logic."""
    preview = dict(settings)
    widget_fields = {
        "model": {"architecture": "model"},
        "history": {"lstm_seq_length": "steps"},
        "goal": {"optimization_goal": "goal"},
        "cycles": {"max_cycles": "count"},
        "training": {
            "manual_starting_lr": "learning_rate", "epochs": "epochs", "batch_size": "batch",
            "early_stop_patience": "patience", "manual_dropout_rate": "dropout",
            "manual_weight_decay": "decay", "manual_activation": "activation",
        },
        "targets": {"mse_target": "mse", "customer_max_latency_ms": "latency"},
        "advanced": {
            "learning_rate_min": "lr_min", "learning_rate_max": "lr_max",
            "derivative_method": "derivative_method", "derivative_filter_tau": "tau",
            "use_state_filter": "filter", "auto_detect_angles": "angles_auto",
            "multi_trajectory": "multi", "shuffle_data": "shuffle",
            "trajectory_chunk_size": "chunk", "integrator_type": "integrator",
            "rollout_horizon": "horizon",
        },
    }.get(section, {})
    for field, widget in widget_fields.items():
        session_key = f"{prefix}_{widget}"
        if session_key in st.session_state:
            preview[field] = st.session_state[session_key]
    if section == "physics":
        method = st.session_state.get(
            f"{prefix}_enabled",
            str(settings.get("physics_mode", "pinn")).upper() if settings.get("use_pinn") else "Off",
        )
        preview["use_pinn"] = method != "Off"
        if method in {"PINN", "PCNN"}:
            preview["physics_mode"] = method.lower()
        if states is not None:
            preview["pcnn_hard_constraint_states"] = [
                state for index, state in enumerate(states)
                if st.session_state.get(f"{prefix}_hard_{index}", state in (preview.get("pcnn_hard_constraint_states") or []))
            ]
    elif section == "effort" and f"{prefix}_mode" in st.session_state:
        preview["run_mode"] = st.session_state[f"{prefix}_mode"]
        cap = int(cfg.run_mode_limits(preview["run_mode"])["max_cycles"])
        preview["max_cycles"] = min(int(preview["max_cycles"]), cap)
    elif section == "network":
        layer_low, layer_high = int(settings["num_layers_min"]), int(settings["num_layers_max"])
        layer_count = min(max(int(st.session_state.get(f"{prefix}_layers", len(_starting_layers(settings)))), layer_low), layer_high)
        width_low, width_high = int(settings["hidden_size_min"]), int(settings["hidden_size_max"])
        defaults = _starting_layers(settings)
        preview["manual_starting_hidden_layers"] = [
            min(max(int(st.session_state.get(f"{prefix}_width_{index}", defaults[min(index, len(defaults) - 1)])), width_low), width_high)
            for index in range(layer_count)
        ]
    elif section == "advanced":
        if f"{prefix}_initializer" in st.session_state:
            preview["choose_via_llm_initializer"] = st.session_state[f"{prefix}_initializer"]
        if f"{prefix}_lower" in st.session_state and f"{prefix}_upper" in st.session_state:
            preview["auto_filter_percentiles"] = [st.session_state[f"{prefix}_lower"], st.session_state[f"{prefix}_upper"]]
    return preview


def _network_label(settings: dict) -> str:
    widths = _starting_layers(settings)
    if settings["architecture"] == "LSTM":
        return f"{len(widths)} recurrent layers · " + " → ".join(map(str, widths)) + " units"
    return f"{len(widths)} dense layers · " + " → ".join(map(str, widths)) + " units"


def _training_label(settings: dict) -> str:
    activation = "LSTM built-in gates" if settings["architecture"] == "LSTM" else settings["manual_activation"].upper()
    return (f"LR {_starting_lr(settings):.4g} · {settings['epochs']} epochs · batch {settings['batch_size']} · "
            f"patience {settings['early_stop_patience']} · dropout {settings['manual_dropout_rate']:g} · "
            f"decay {settings['manual_weight_decay']:g} · {activation}")


def _signal_dimensions_visual(state_count: int, action_count: int, feature_count: int,
                              architecture: str) -> str:
    """Show how selected physical channels determine the network dimensions."""
    output_count = int(state_count)
    return f'''<div class="setup-network-preview">
  <div class="setup-network-title">LIVE SYSTEM SIGNAL MAP</div>
  <svg viewBox="0 0 760 164" role="img" aria-label="{feature_count} model input features map to {output_count} state derivatives" xmlns="http://www.w3.org/2000/svg">
    <defs><marker id="signal-map-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#72c9a8"/></marker></defs>
    <rect class="setup-net-ground" x="1" y="1" width="758" height="162" rx="14"/>
    <path class="setup-net-wire" marker-end="url(#signal-map-arrow)" d="M160 63C240 18 370 18 442 58M315 81H442M407 99C421 99 429 99 442 99M559 81H610"/>
    <rect class="setup-net-layer" x="18" y="45" width="142" height="72" rx="12"/>
    <rect class="setup-net-layer" x="174" y="45" width="141" height="72" rx="12"/>
    <rect class="setup-net-layer" x="329" y="45" width="78" height="72" rx="12"/>
    <rect class="setup-net-layer" x="442" y="36" width="117" height="90" rx="14"/>
    <rect x="610" y="45" width="132" height="72" rx="12" fill="#302d48" stroke="#8479c8" stroke-width="1.5"/>
    <text class="setup-net-label" x="89" y="72" text-anchor="middle">STATE OUTPUTS</text>
    <text class="setup-net-count" x="89" y="96" text-anchor="middle">{output_count} measured signals</text>
    <text class="setup-net-label" x="244" y="72" text-anchor="middle">SYSTEM INPUTS</text>
    <text class="setup-net-count" x="244" y="96" text-anchor="middle">{int(action_count)} controls</text>
    <text class="setup-net-label" x="368" y="72" text-anchor="middle">TIME</text>
    <text class="setup-net-count" x="368" y="96" text-anchor="middle">Δt</text>
    <text class="setup-net-label" x="500" y="70" text-anchor="middle">{escape(str(architecture).upper())}</text>
    <text class="setup-net-count" x="500" y="93" text-anchor="middle">{int(feature_count)} input nodes</text>
    <text x="676" y="72" text-anchor="middle" fill="#e1dcff" font-family="ui-monospace,monospace" font-size="11" font-weight="700">PREDICTED</text>
    <text x="676" y="96" text-anchor="middle" fill="#c9c2ff" font-family="ui-monospace,monospace" font-size="10">{output_count} derivatives</text>
  </svg>
</div>'''


def _row(label: str, value: str, section: str, chat: dict, suffix: str,
         illustration, *, disabled: bool = False, on_signals_saved=None,
         on_interaction=None, edit_label: str | None = None,
         editor_section: str | None = None) -> None:
    with st.container(key=f"setup_summary_row_{section}_{suffix}"):
        label_col, value_col, edit_col = st.columns([1.15, 2.2, 1.3], vertical_alignment="center", gap="small")
        label_col.markdown(f"**{label}**")
        value_col.write(value)
        if edit_col.button(edit_label or f"Edit {label}", icon=":material/edit:",
                           key=f"setup_summary_edit_{section}_{suffix}",
                           disabled=disabled, width="stretch"):
            if on_interaction is not None:
                on_interaction()
            _open_editor(chat, editor_section or section, suffix, illustration,
                         on_signals_saved, on_interaction)


def _open_editor(chat: dict, section: str, suffix: str, illustration,
                 on_signals_saved=None, on_interaction=None) -> None:
    prefix = f"setup_popup_{section}_{suffix}_"
    for name in list(st.session_state):
        if name.startswith(prefix):
            del st.session_state[name]
    _edit_dialog(chat, section, suffix, illustration, on_signals_saved, on_interaction)


def _history_visual(steps: int) -> str:
    shown = min(8, max(2, int(steps)))
    blocks = "".join(
        f'<span class="setup-history-step{" current" if index == shown - 1 else ""}">'
        f'<small>t−{shown - index - 1}</small><b>x, u</b></span>'
        for index in range(shown)
    )
    tail = '<span class="setup-history-ellipsis">…</span>' if int(steps) > shown else ""
    return f'<div class="setup-history-visual" aria-label="{int(steps)} recent time steps">{blocks}{tail}<span class="setup-history-arrow">→</span><span class="setup-history-output">LSTM</span></div>'


def _chosen_model(chat: dict, settings: dict, flow: dict, suffix: str, illustration, busy: bool,
                  on_interaction=None) -> None:
    architecture = settings["architecture"]
    is_lstm = architecture == "LSTM"
    widths = _starting_layers(settings)
    units_label = " → ".join(map(str, widths)) + " units"
    title = "Sequence memory" if is_lstm else "Direct mapping"
    description = (
        f"Reads the last {settings['lstm_seq_length']} measurements in order, then predicts state derivatives."
        if is_lstm else
        "Maps the current measured state and inputs directly to state derivatives."
    )
    with st.container(border=True, key=f"setup_chosen_model_{suffix}"):
        copy, diagram = st.columns([1, 1.35], vertical_alignment="center", gap="large")
        copy.markdown('<div class="setup-choice-kicker">CHOSEN STARTING MODEL</div>', unsafe_allow_html=True)
        copy.markdown(f"## {architecture}")
        copy.markdown(f'<div class="setup-choice-subtitle">{title}</div>', unsafe_allow_html=True)
        copy.markdown(f'<div class="setup-choice-description">{description}</div>', unsafe_allow_html=True)
        tags = [
            f'<span>{"RECENT HISTORY" if is_lstm else "CURRENT SAMPLE"}</span>',
            f'<span>{len(widths)} {"RECURRENT" if is_lstm else "DENSE"} LAYERS · {units_label.upper()}</span>',
            '<span>STATE DERIVATIVES</span>',
        ]
        if settings["use_pinn"] and pinn_maker.is_equation_ready(chat):
            tags.append(f'<span>{escape(str(settings.get("physics_mode", "pinn")).upper())}</span>')
        copy.markdown(f'<div class="setup-choice-tags">{"".join(tags)}</div>', unsafe_allow_html=True)
        if flow.get("architecture_reason"):
            with copy.expander("Why this model was suggested"):
                st.write(flow["architecture_reason"])
        if copy.button("Edit chosen model", icon=":material/edit:", key=f"setup_choice_edit_{suffix}",
                       disabled=busy, width="stretch"):
            if on_interaction is not None:
                on_interaction()
            _open_editor(chat, "model", suffix, illustration)
        diagram.markdown(illustration(architecture, selected=True), unsafe_allow_html=True)


def _editor_heading(title: str, current: str, note: str) -> None:
    st.markdown(f'<div class="setup-editor-current"><small>EDITING · {title.upper()}</small>'
                f'<strong>{current}</strong><span>{note}</span></div>', unsafe_allow_html=True)


def _setup_visual(section: str, settings: dict, dataset: dict) -> tuple[str, str, str]:
    """Return a setting-specific diagram, accessible description, and caption."""
    architecture = escape(str(settings.get("architecture", "LSTM")))
    mode = str(settings.get("run_mode", "regular")).lower()
    goal = str(settings.get("optimization_goal", "balanced")).lower()
    cycle_count = max(1, min(7, int(settings.get("max_cycles", 7) or 7)))
    epochs = escape(str(settings.get("max_epochs", settings.get("epochs", 100))))
    widths = " → ".join(map(str, _starting_layers(settings)))
    state_count = len(dataset.get("states") or [])
    input_count = len(dataset.get("inputs") or dataset.get("actions") or [])
    mse_target = escape(str(settings.get("mse_target", "—")))
    latency_target = escape(str(settings.get("customer_max_latency_ms", "—")))

    if section == "model":
        if architecture == "LSTM":
            art = '''<path class="setup-art-wire" d="M60 82H145M184 82H270M310 82H370"/>
              <rect class="setup-art-card" x="38" y="55" width="60" height="54" rx="10"/>
              <rect class="setup-art-card" x="147" y="55" width="62" height="54" rx="10"/>
              <rect class="setup-art-card setup-art-card--active" x="272" y="55" width="48" height="54" rx="10"/>
              <text class="setup-art-label" x="68" y="78" text-anchor="middle">x, u</text><text class="setup-art-muted" x="68" y="94" text-anchor="middle">window</text>
              <text class="setup-art-label" x="178" y="78" text-anchor="middle">LSTM</text><text class="setup-art-muted" x="178" y="94" text-anchor="middle">memory</text>
              <text class="setup-art-label" x="296" y="78" text-anchor="middle">ẋ</text><text class="setup-art-muted" x="296" y="94" text-anchor="middle">output</text>
              <path class="setup-art-loop" d="M177 53C177 24 250 24 250 53"/><path class="setup-edit-forecast" d="M245 47L250 53L255 47"/>'''
            caption = "LSTM carries information forward through the sequence."
            aria = "LSTM sequence memory: measured state and input feed a recurrent model to predict state derivatives"
        else:
            art = '''<path class="setup-art-wire" d="M78 82H342"/>
              <g class="setup-art-nodes"><circle cx="76" cy="56" r="8"/><circle cx="76" cy="82" r="8"/><circle cx="76" cy="108" r="8"/>
              <circle cx="165" cy="48" r="8"/><circle cx="165" cy="71" r="8"/><circle cx="165" cy="94" r="8"/><circle cx="165" cy="117" r="8"/>
              <circle cx="255" cy="60" r="8"/><circle cx="255" cy="83" r="8"/><circle cx="255" cy="106" r="8"/><circle cx="344" cy="82" r="9"/></g>
              <path class="setup-art-flow" d="M86 56L157 48M86 56L157 71M86 82L157 71M86 82L157 94M86 108L157 94M86 108L157 117M173 48L247 60M173 71L247 60M173 71L247 83M173 94L247 83M173 94L247 106M173 117L247 106M263 60L335 82M263 83L335 82M263 106L335 82"/>
              <text class="setup-art-muted" x="76" y="139" text-anchor="middle">STATE + INPUT</text><text class="setup-art-muted" x="344" y="139" text-anchor="middle">ẋ</text>'''
            caption = "MLP maps the current state and inputs directly to derivatives."
            aria = "MLP layers map the current measured state and inputs to state derivatives"
    elif section == "history":
        art = '''<path class="setup-art-axis" d="M43 112H377"/><path class="setup-art-flow" d="M58 81H344"/>
          <g><rect class="setup-art-card" x="48" y="55" width="54" height="52" rx="9"/><rect class="setup-art-card" x="112" y="55" width="54" height="52" rx="9"/>
          <rect class="setup-art-card" x="176" y="55" width="54" height="52" rx="9"/><rect class="setup-art-card setup-art-card--active" x="240" y="55" width="54" height="52" rx="9"/>
          <rect class="setup-art-card" x="304" y="55" width="54" height="52" rx="9"/></g>
          <text class="setup-art-muted" x="75" y="75" text-anchor="middle">t−4</text><text class="setup-art-label" x="75" y="94" text-anchor="middle">x,u</text>
          <text class="setup-art-muted" x="139" y="75" text-anchor="middle">t−3</text><text class="setup-art-label" x="139" y="94" text-anchor="middle">x,u</text>
          <text class="setup-art-muted" x="203" y="75" text-anchor="middle">t−2</text><text class="setup-art-label" x="203" y="94" text-anchor="middle">x,u</text>
          <text class="setup-art-muted" x="267" y="75" text-anchor="middle">t−1</text><text class="setup-art-label" x="267" y="94" text-anchor="middle">x,u</text>
          <text class="setup-art-muted" x="331" y="75" text-anchor="middle">NOW</text><text class="setup-art-label" x="331" y="94" text-anchor="middle">predict</text>
          <circle class="setup-art-pulse" cx="267" cy="55" r="4"/><text class="setup-art-muted" x="210" y="139" text-anchor="middle">RECENT MEASUREMENTS IN ORDER</text>'''
        caption = "The history length controls how many recent steps the LSTM sees."
        aria = "Ordered timeline of recent state and input samples passed into LSTM history"
    elif section == "physics":
        if str(settings.get("physics_mode", "pinn")).lower() == "pcnn":
            art = '''<rect class="setup-art-card" x="24" y="52" width="112" height="60" rx="10"/>
              <text class="setup-art-label" x="80" y="73" text-anchor="middle">HARD RULE</text><text class="setup-art-equation" x="80" y="93" text-anchor="middle">ẋⱼ = fⱼ(x,u)</text>
              <path class="setup-art-wire" d="M136 82H207"/><circle class="setup-art-pulse" cx="208" cy="82" r="7"/>
              <rect class="setup-art-card setup-art-card--active" x="222" y="35" width="174" height="47" rx="10"/><text class="setup-art-label" x="309" y="55" text-anchor="middle">SELECTED STATE</text><text class="setup-art-muted" x="309" y="71" text-anchor="middle">exact equation output</text>
              <rect class="setup-art-card" x="222" y="96" width="174" height="43" rx="10"/><text class="setup-art-label" x="309" y="115" text-anchor="middle">OTHER STATES</text><text class="setup-art-muted" x="309" y="131" text-anchor="middle">learned directly by NN</text>
              <text class="setup-art-muted" x="210" y="158" text-anchor="middle">NO EQUATION BASELINE OR CORRECTION PENALTY</text>'''
            caption = "Selected state derivatives are replaced by exact rules; the NN learns every remaining state from measurements."
            aria = "PCNN routes selected state derivatives through exact constraints and predicts remaining states with the neural network"
        else:
            art = '''<rect class="setup-art-card" x="35" y="37" width="122" height="42" rx="9"/><rect class="setup-art-card" x="35" y="91" width="122" height="42" rx="9"/>
              <text class="setup-art-label" x="50" y="55">MEASUREMENTS</text><path class="setup-edit-signal" d="M51 68C68 56 81 73 98 63S128 57 143 64"/>
              <text class="setup-art-label" x="50" y="109">EQUATION</text><text class="setup-art-equation" x="50" y="126">ẋ = f(x, u)</text>
              <path class="setup-art-wire" d="M157 58H215L250 82M157 112H215L250 82H302"/><circle class="setup-art-pulse" cx="250" cy="82" r="8"/>
              <rect class="setup-art-card setup-art-card--active" x="304" y="57" width="80" height="50" rx="10"/><text class="setup-art-label" x="344" y="78" text-anchor="middle">MODEL</text><text class="setup-art-muted" x="344" y="94" text-anchor="middle">data + physics</text>
              <text class="setup-art-muted" x="215" y="145" text-anchor="middle">DATA + PHYSICS MODEL</text>'''
            caption = "PINN adds the validated equation as a soft training constraint alongside measurements."
            aria = "Measured data and a validated physics equation constrain model training with a soft loss"
    elif section == "context":
        art = '''<rect class="setup-art-card" x="28" y="58" width="83" height="53" rx="10"/><text class="setup-art-label" x="69" y="80" text-anchor="middle">ACTION</text><text class="setup-art-equation" x="69" y="99" text-anchor="middle">u(t)</text>
          <path class="setup-art-flow" d="M111 84H153"/><rect class="setup-art-card setup-art-card--active" x="155" y="48" width="112" height="72" rx="14"/><path class="setup-art-plant" d="M176 91L190 69L205 99L220 65L244 91"/><text class="setup-art-muted" x="211" y="137" text-anchor="middle">PHYSICAL PLANT</text>
          <path class="setup-art-flow" d="M267 84H309"/><rect class="setup-art-card" x="311" y="58" width="82" height="53" rx="10"/><text class="setup-art-label" x="352" y="80" text-anchor="middle">STATE</text><text class="setup-art-equation" x="352" y="99" text-anchor="middle">x(t)</text>
          <circle class="setup-art-pulse" cx="131" cy="84" r="4"/><circle class="setup-art-pulse setup-art-pulse--late" cx="288" cy="84" r="4"/>'''
        caption = "A compact picture of the physical system described by your signals."
        aria = "System context block diagram from control action through a physical plant to measured state"
    elif section == "goal":
        goal_cards = (
            ("accuracy", 19, "ACCURACY", "closer fit"),
            ("balanced", 119, "BALANCED", "fit + speed"),
            ("speed", 219, "SPEED", "fast inference"),
            ("compact", 319, "COMPACT", "smaller model"),
        )
        cards = []
        for key, x, label, sublabel in goal_cards:
            active = " setup-art-card--active" if key == goal else ""
            cards.append(f'<rect class="setup-art-card{active}" x="{x}" y="51" width="82" height="65" rx="11"/>')
            cards.append(f'<circle class="setup-art-target{" setup-art-target--active" if key == goal else ""}" cx="{x + 41}" cy="75" r="13"/><circle class="setup-art-target-dot" cx="{x + 41}" cy="75" r="4"/>')
            cards.append(f'<text class="setup-art-label" x="{x + 41}" y="97" text-anchor="middle">{label}</text><text class="setup-art-muted" x="{x + 41}" y="108" text-anchor="middle">{sublabel}</text>')
        art = "".join(cards) + '<text class="setup-art-muted" x="210" y="144" text-anchor="middle">ONE PRIORITY GUIDES THE SEARCH</text>'
        caption = f"The {escape(goal)} priority changes what candidate trade-offs matter most."
        aria = "Four optimization priorities for accuracy, balanced fit, speed, and compact size, with the selected objective highlighted"
    elif section == "effort":
        active_mode = mode if mode in {"fast", "regular", "heavy"} else "regular"
        mode_data = (("fast", 48, "FAST", "quick scan"), ("regular", 160, "REGULAR", "balanced search"), ("heavy", 272, "HEAVY", "broad search"))
        cards = []
        for key, x, label, note in mode_data:
            active = " setup-art-card--active" if key == active_mode else ""
            cards.append(f'<rect class="setup-art-card{active}" x="{x}" y="47" width="100" height="74" rx="12"/>')
            dots = 3 if key == "fast" else (5 if key == "regular" else 7)
            for dot in range(dots):
                cards.append(f'<circle class="setup-art-search-dot" cx="{x + 23 + dot * 9}" cy="73" r="3.2"/>')
            cards.append(f'<text class="setup-art-label" x="{x + 50}" y="96" text-anchor="middle">{label}</text><text class="setup-art-muted" x="{x + 50}" y="111" text-anchor="middle">{note}</text>')
        art = "".join(cards) + '<text class="setup-art-muted" x="210" y="148" text-anchor="middle">MORE SEARCH = MORE CANDIDATES TO COMPARE</text>'
        caption = f"{escape(active_mode.title())} search is selected; its cycle budget is {cycle_count}."
        aria = "Fast, regular, and heavy search modes with the current run mode highlighted"
    elif section == "cycles":
        cells = []
        cell_width, gap = 42, 9
        total_width = cycle_count * cell_width + (cycle_count - 1) * gap
        start_x = (420 - total_width) / 2
        for index in range(1, cycle_count + 1):
            x = start_x + (index - 1) * (cell_width + gap)
            cells.append(f'<rect class="setup-art-cycle setup-art-cycle--active" x="{x:.1f}" y="59" width="{cell_width}" height="49" rx="9"/><text class="setup-art-label" x="{x + cell_width / 2:.1f}" y="81" text-anchor="middle">{index}</text><text class="setup-art-muted" x="{x + cell_width / 2:.1f}" y="98" text-anchor="middle">TRY</text>')
        bracket_start = start_x
        bracket_end = start_x + total_width
        art = "".join(cells) + f'<path class="setup-art-bracket" d="M{bracket_start:.1f} 122V130H{bracket_end:.1f}V122"/><text class="setup-art-equation" x="210" y="148" text-anchor="middle">{cycle_count} candidate cycle{"s" if cycle_count != 1 else ""}</text>'
        caption = f"The search may evaluate up to {cycle_count} candidate cycles."
        aria = f"Candidate cycle budget showing {cycle_count} planned trials out of seven"
    elif section == "signals":
        roles = [("TIME", "t", 42, "setup-art-time")]
        roles += [("STATE", "x", 154, "setup-art-state")]
        roles += [("INPUT", "u", 266, "setup-art-input")]
        art_parts = ['<path class="setup-art-axis" d="M32 119H390"/>']
        for label, symbol, x, cls in roles:
            art_parts.append(
                f'<rect class="setup-art-column {cls}" x="{x}" y="42" width="82" height="76" rx="10"/>'
                f'<text class="setup-art-muted" x="{x + 41}" y="60" text-anchor="middle">{label}</text>'
                f'<text class="setup-art-equation" x="{x + 41}" y="109" text-anchor="middle">{symbol}</text>'
            )
        # The clock makes clear that t orders samples; the symbols stay at the
        # bottom of each card, below the state and input signal sketches.
        art_parts.append(
            '<circle class="setup-art-clock-face" cx="83" cy="83" r="14"/>'
            '<path class="setup-art-clock-marks" d="M83 70V73M96 83H93M83 96V93M70 83H73"/>'
            '<path class="setup-art-clock-hands" d="M83 83V75M83 83L89 87"/>'
            '<circle class="setup-art-clock-center" cx="83" cy="83" r="1.8"/>'
        )
        # Keep the sketches above the bottom symbols, inside their signal cards.
        art_parts.append('<path class="setup-art-signal-state" d="M160 94C168 90 172 70 183 73S198 96 209 86S221 66 230 70"/><path class="setup-art-signal-input" d="M272 94H282V73H298V94H312V73H328V94H342"/>')
        art_parts.append(f'<text class="setup-art-muted" x="195" y="145" text-anchor="middle">{state_count} STATE{("S" if state_count != 1 else "")} · {input_count} INPUT{("S" if input_count != 1 else "")}</text>')
        art = "".join(art_parts)
        caption = "Time orders the samples; state columns are predicted and input columns drive the system."
        aria = f"Dataset column roles: time, {state_count} measured state columns, and {input_count} required input columns"
    elif section == "epochs":
        art = '''<path class="setup-art-axis" d="M43 119H380M43 119V39"/><path class="setup-art-loss-area" d="M45 49C86 53 86 66 127 69S174 83 211 87S262 99 300 102S344 108 377 111L377 119H45Z"/>
          <path class="setup-art-loss-line" d="M45 49C86 53 86 66 127 69S174 83 211 87S262 99 300 102S344 108 377 111"/>
          <circle class="setup-art-pulse" cx="377" cy="111" r="5"/><text class="setup-art-muted" x="46" y="33">TRAINING ERROR</text><text class="setup-art-equation" x="210" y="145" text-anchor="middle">UP TO ''' + epochs + ''' EPOCHS PER CANDIDATE</text>'''
        caption = f"Each candidate can train for up to {epochs} epochs, subject to stopping rules."
        aria = f"Illustrative training loss curve over up to {epochs} epochs"
    elif section == "network":
        layer_widths = _starting_layers(settings)
        column_widths = [None, *layer_widths, None]
        positions = [27 + index * (366 / (len(column_widths) - 1))
                     for index in range(len(column_widths))]
        node_groups = []
        for layer_index, (x, width) in enumerate(zip(positions, column_widths)):
            role = "input" if layer_index == 0 else "output" if layer_index == len(column_widths) - 1 else "hidden"
            count = 4 if width is None else min(6, max(3, 3 + round(math.log2(max(int(width), 32) / 32))) )
            y_values = [48 + index * (55 / max(1, count - 1)) for index in range(count)]
            for y in y_values:
                node_groups.append(
                    f'<circle class="setup-net-node {role}" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>'
                )
                if layer_index < len(column_widths) - 1:
                    next_x = positions[layer_index + 1]
                    next_width = column_widths[layer_index + 1]
                    next_count = 4 if next_width is None else min(
                        6, max(3, 3 + round(math.log2(max(int(next_width), 32) / 32)))
                    )
                    next_ys = [48 + index * (55 / max(1, next_count - 1)) for index in range(next_count)]
                    node_groups.extend(
                        f'<path class="setup-net-wire" d="M{x + 4:.1f} {y:.1f}L{next_x - 4:.1f} {next_y:.1f}"/>'
                        for next_y in next_ys
                    )
            label = "INPUT" if role == "input" else "OUTPUT" if role == "output" else f"L{layer_index}"
            node_groups.append(
                f'<text class="setup-edit-nn-label" x="{x:.1f}" y="126" text-anchor="middle">{label}</text>'
            )
            if width is not None:
                node_groups.append(
                    f'<text class="setup-edit-nn-width" x="{x:.1f}" y="143" text-anchor="middle">{int(width)}</text>'
                )
        layer_label = escape(widths)
        art = ('<text class="setup-edit-nn-kicker" x="210" y="20" text-anchor="middle">LIVE NETWORK PREVIEW</text>'
               + ''.join(node_groups))
        caption = (f"{architecture} · {len(layer_widths)} layers · "
                   f"{layer_label} units")
        aria = (f"Live {architecture} network with input and output columns and "
                + ", ".join(f"layer {index + 1}: {width} units" for index, width in enumerate(layer_widths)))
    elif section == "training":
        art = '''<rect class="setup-art-card" x="35" y="52" width="94" height="62" rx="11"/><text class="setup-art-label" x="82" y="76" text-anchor="middle">PREDICT</text><text class="setup-art-equation" x="82" y="98" text-anchor="middle">ẋ model</text>
          <path class="setup-art-flow" d="M129 83H174"/><rect class="setup-art-card setup-art-card--active" x="176" y="52" width="94" height="62" rx="11"/><text class="setup-art-label" x="223" y="76" text-anchor="middle">MEASURE</text><text class="setup-art-equation" x="223" y="98" text-anchor="middle">error</text>
          <path class="setup-art-flow" d="M270 83H315"/><rect class="setup-art-card" x="317" y="52" width="72" height="62" rx="11"/><text class="setup-art-label" x="353" y="76" text-anchor="middle">UPDATE</text><text class="setup-art-equation" x="353" y="98" text-anchor="middle">weights</text>
          <path class="setup-art-loop" d="M353 117C353 146 82 146 82 117"/><path class="setup-edit-forecast" d="M88 122L82 117L76 122"/><text class="setup-art-muted" x="220" y="158" text-anchor="middle">REPEAT WITH TRAINING SAMPLES</text>'''
        caption = "Training repeats prediction, error measurement, and weight updates."
        aria = "Training loop from prediction to measuring error to updating weights and repeating"
    elif section == "targets":
        def strictness(value: float, low: float, high: float) -> float:
            log_low, log_high = math.log10(low), math.log10(high)
            scale = (math.log10(min(max(float(value), low), high)) - log_low) / (log_high - log_low)
            return 1.0 - scale

        arc_length = math.pi * 61
        mse_dash = arc_length * strictness(float(settings["mse_target"]), 1e-8, 100000.0)
        latency_dash = arc_length * strictness(float(settings["customer_max_latency_ms"]), 0.001, 60000.0)
        art = f'''<path class="setup-art-gauge-track" d="M53 110A61 61 0 0 1 175 110"/><path class="setup-art-gauge-fit" stroke-dasharray="{mse_dash:.1f} {arc_length:.1f}" d="M53 110A61 61 0 0 1 175 110"/>
          <path class="setup-art-gauge-track" d="M245 110A61 61 0 0 1 367 110"/><path class="setup-art-gauge-speed" stroke-dasharray="{latency_dash:.1f} {arc_length:.1f}" d="M245 110A61 61 0 0 1 367 110"/>
          <text class="setup-art-muted" x="114" y="87" text-anchor="middle">VALIDATION MSE</text><text class="setup-art-equation" x="114" y="105" text-anchor="middle">≤ {mse_target}</text>
          <text class="setup-art-muted" x="306" y="87" text-anchor="middle">LATENCY</text><text class="setup-art-equation" x="306" y="105" text-anchor="middle">≤ {latency_target} ms</text>
          <text class="setup-art-muted" x="210" y="148" text-anchor="middle">FIT QUALITY · INFERENCE SPEED</text>'''
        caption = "The gauges reflect how strict each selected limit is; lower limits are stricter."
        aria = f"Run targets for validation error at most {mse_target} and latency at most {latency_target} milliseconds"
    elif section == "advanced":
        art = '''<path class="setup-art-flow" d="M94 83H132M194 83H232M294 83H332"/>
          <rect class="setup-art-card" x="29" y="54" width="66" height="58" rx="10"/><text class="setup-art-label" x="62" y="78" text-anchor="middle">RAW</text><text class="setup-art-muted" x="62" y="96" text-anchor="middle">samples</text>
          <rect class="setup-art-card" x="133" y="54" width="62" height="58" rx="10"/><text class="setup-art-label" x="164" y="78" text-anchor="middle">FILTER</text><text class="setup-art-muted" x="164" y="96" text-anchor="middle">clean</text>
          <rect class="setup-art-card" x="233" y="54" width="62" height="58" rx="10"/><text class="setup-art-label" x="264" y="78" text-anchor="middle">SPLIT</text><text class="setup-art-muted" x="264" y="96" text-anchor="middle">train/test</text>
          <rect class="setup-art-card setup-art-card--active" x="333" y="54" width="62" height="58" rx="10"/><text class="setup-art-label" x="364" y="78" text-anchor="middle">SOLVE</text><text class="setup-art-muted" x="364" y="96" text-anchor="middle">dynamics</text>
          <circle class="setup-art-pulse" cx="112" cy="83" r="4"/><circle class="setup-art-pulse setup-art-pulse--late" cx="212" cy="83" r="4"/><circle class="setup-art-pulse" cx="312" cy="83" r="4"/>
          <text class="setup-art-muted" x="210" y="144" text-anchor="middle">ADVANCED DATA + SOLVER PIPELINE</text>'''
        caption = "Advanced controls affect how data is prepared and trajectories are evaluated."
        aria = "Advanced data processing pipeline through filtering, train and test split, and trajectory solver"
    else:
        art = '''<path class="setup-edit-grid" d="M35 43H385M35 75H385M35 107H385M35 32V119M120 32V119M205 32V119M290 32V119M375 32V119"/>
          <path class="setup-edit-area" d="M35 104C69 97 80 71 116 78S166 102 204 81S255 48 289 61S343 75 385 48L385 119H35Z"/>
          <path class="setup-edit-signal" d="M35 104C69 97 80 71 116 78S166 102 204 81S255 48 289 61S343 75 385 48"/><circle class="setup-edit-marker" cx="289" cy="61" r="4"/>'''
        caption = "Measured trajectory through the selected model."
        aria = f"Measured trajectory illustration for {architecture}"
    return art, aria, caption


def _editor_hero(section: str, title: str, settings: dict, dataset: dict,
                 chat: dict, flow: dict) -> str:
    """Show a live, setting-specific visual orientation panel."""
    guidance = {
        "model": "Choose how the network reads the measured trajectory.",
        "history": "Set how much recent motion the LSTM can remember.",
        "physics": "Choose a soft PINN loss or PCNN hard constraints for selected state derivatives.",
        "context": "Optional device notes help LabCD explain the signals.",
        "goal": "Tell the search what matters most in the final model.",
        "effort": "Choose how much time and candidate search to spend.",
        "cycles": "Set how many candidate models LabCD should compare.",
        "signals": "Choose which measured columns are system outputs and which are system inputs.",
        "epochs": "Set the training limit for each candidate.",
        "network": "Shape the width and depth of the candidate network.",
        "training": "Tune how each candidate learns from your measurements.",
        "targets": "Set the fit and inference limits that matter to you.",
        "advanced": "Adjust data handling, search bounds, and trajectory solving.",
    }.get(section, "Adjust a starting value before reviewing the candidate network.")
    sample_count = dataset.get("rows") or (dataset.get("analysis") or {}).get("samples")
    samples = f"{int(sample_count):,}" if sample_count else "Ready"
    architecture = escape(str(settings.get("architecture", "LSTM")))
    mode = escape(str(settings.get("run_mode", "regular")).title())
    cycles = escape(str(settings.get("max_cycles", "—")))
    physics_ready = bool(settings.get("use_pinn") and pinn_maker.is_equation_ready(chat))
    physics_mode = str(settings.get("physics_mode", "pinn")).upper()
    if physics_ready and physics_mode == "PCNN":
        physics_status = "PCNN · hard constraints"
    elif physics_ready:
        physics_status = "PINN · soft equation loss"
    else:
        physics_status = "Off · measured data"
    art, svg_label, visual_caption = _setup_visual(section, settings, dataset)
    svg_label = escape(svg_label)
    visual_caption = escape(visual_caption)
    agent_logic = escape(_setup_logic(section, settings, dataset, flow))
    return f'''<section class="setup-edit-hero setup-edit-hero--{section}" aria-label="Setup editor overview">
      <div class="setup-edit-hero-copy">
        <div class="setup-edit-kicker"><span class="setup-edit-spark">✦</span> LABCD · SETUP STUDIO</div>
        <div class="setup-edit-title">{escape(title)}</div>
        <div class="setup-edit-description">{escape(guidance)}</div>
        <div class="setup-edit-badges">
          <span><small>DATA</small><b>{samples} samples</b></span>
          <span><small>MODEL</small><b>{architecture}</b></span>
          <span><small>SEARCH</small><b>{mode} · {cycles} cycles</b></span>
          <span><small>TRAINING</small><b>{escape(physics_status)}</b></span>
        </div>
      </div>
      <div class="setup-edit-visual setup-edit-visual--{section}">
        <svg viewBox="0 0 420 168" role="img" aria-label="{svg_label}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="setup-edit-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#73dfbd" stop-opacity=".22"/><stop offset="1" stop-color="#73dfbd" stop-opacity="0"/></linearGradient>
            <filter id="setup-edit-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>
          <rect class="setup-edit-plot-bg" x=".5" y=".5" width="419" height="167" rx="14"/>
          {art}
        </svg>
        <div class="setup-edit-visual-caption"><span class="setup-edit-live-dot"></span> {visual_caption}</div>
        <div class="setup-edit-agent-logic"><small>AGENT LOGIC</small><span>{agent_logic}</span></div>
      </div>
    </section>'''


@st.dialog("Edit setup", width="large")
def _edit_dialog(chat: dict, section: str, suffix: str, illustration,
                 on_signals_saved=None, on_interaction=None) -> None:
    settings = core.RunSettings.model_validate(chat["settings"]).model_dump()
    flow = chat.get("run_setup_flow") or {}
    dataset = chat.get("dataset") or {}
    active_agent_job = core.REGISTRY.conversations.get(chat.get("id"))
    busy = bool(active_agent_job and getattr(active_agent_job, "running", False))
    key = f"setup_popup_{section}_{suffix}"
    proposal = _proposal_settings(flow, settings)
    proposal_roles = flow.get("initializer_dataset_roles") or {
        "states": list(dataset.get("states") or []),
        "actions": list(dataset.get("inputs") or dataset.get("actions") or []),
    }
    if section == "signals":
        # Widget values survive each Streamlit rerun, so the overview can preview
        # the user's current role selections before they save them.
        dataset = dict(dataset)
        dataset["states"] = st.session_state.get(
            f"{key}_states", dataset.get("states") or []
        )
        dataset["inputs"] = st.session_state.get(
            f"{key}_actions", dataset.get("inputs") or dataset.get("actions") or []
        )
    elif section == "goal":
        selected_goal = st.session_state.get(f"{key}_goal", settings["optimization_goal"])
        if selected_goal in GOALS:
            settings["optimization_goal"] = selected_goal
    elif section == "effort":
        modes = {"fast", "regular", "heavy"}
        selected_mode = st.session_state.get(f"{key}_mode", settings["run_mode"])
        if selected_mode not in modes:
            selected_mode = "regular"
        settings["run_mode"] = selected_mode
        limits = cfg.run_mode_limits(selected_mode)
        settings["max_cycles"] = min(int(settings["max_cycles"]), int(limits["max_cycles"]))
    elif section == "cycles":
        # Read the live slider value before drawing the hero so its cycle cards
        # and count change on the same rerun as the widget.
        limits = cfg.run_mode_limits(settings["run_mode"])
        cap = int(limits["max_cycles"])
        selected_cycles = int(st.session_state.get(f"{key}_count", settings["max_cycles"]))
        settings["max_cycles"] = min(max(selected_cycles, 1), cap)
    elif section == "network":
        # Keep the single network illustration in sync with each edit on the
        # same Streamlit rerun as the layer count or width widget.
        settings = _live_network_settings(settings, key, st.session_state)
    elif section == "targets":
        settings["mse_target"] = float(
            st.session_state.get(f"{key}_mse", settings["mse_target"])
        )
        settings["customer_max_latency_ms"] = float(
            st.session_state.get(f"{key}_latency", settings["customer_max_latency_ms"])
        )
    physics_states = list(dataset.get("states") or [])
    equation_by_state = {
        item.get("state"): item.get("expression", "")
        for item in (chat.get("pinn_equation") or {}).get("equations", [])
        if item.get("state")
    }
    constraint_states = [state for state in physics_states if state in equation_by_state]
    if section == "physics":
        proposal_roles = dict(proposal_roles)
        proposal_roles["constraint_states"] = constraint_states
        recommendations = (chat.get("pinn_equation") or {}).get("pcnn_constraint_recommendations") or []
        recommendations_by_state = {item.get("state"): item for item in recommendations if item.get("state")}
        if not st.session_state.get(f"{key}_constraints_initialized"):
            saved_exact = set(settings.get("pcnn_hard_constraint_states") or []) & set(constraint_states)
            equation = chat.get("pinn_equation") or {}
            review_id = equation.get("proposal_id", equation.get("sha256"))
            previously_reviewed = chat.get("pcnn_constraints_reviewed_for")
            if previously_reviewed == review_id or (not previously_reviewed and saved_exact):
                initial_exact = saved_exact
            else:
                initial_exact = {
                    state for state, item in recommendations_by_state.items()
                    if item.get("recommended_exact") is True and state in constraint_states
                }
            for index, state in enumerate(constraint_states):
                st.session_state[f"{key}_hard_{index}"] = state in initial_exact
            st.session_state[f"{key}_constraints_initialized"] = True
    title = {
        "model": "Starting model", "history": "LSTM history", "physics": "Physics rules · PINN / PCNN",
        "context": "System context", "goal": "Optimization priority", "effort": "Search effort",
        "cycles": "Candidate cycles", "signals": "System inputs and outputs",
        "network": "Network shape", "training": "Training controls",
        "targets": "Run targets", "advanced": "Advanced controls",
    }[section]
    preview_settings = _preview_settings(
        section, key, settings,
        constraint_states if section == "physics" else list(dataset.get("states") or []),
    )
    st.markdown(_editor_hero(section, title, preview_settings, dataset, chat, flow), unsafe_allow_html=True)
    if st.session_state.pop(f"{key}_reset_notice", False):
        st.caption("Restored to the saved Initializer proposal. Save change to apply.")
    updates = {}
    editor_error = None
    if section == "signals":
        columns = [str(name) for name in dataset.get("columns", [])]
        choices = [name for name in columns if name != "time" and not name.startswith("xdot_")]
        current_states = [name for name in dataset.get("states", []) if name in choices]
        current_actions = [
            name for name in (dataset.get("inputs") or dataset.get("actions") or [])
            if name in choices
        ]
        selected_states = st.multiselect(
            "Measured outputs · states",
            choices,
            default=current_states,
            key=f"{key}_states",
            help="Choose the real system measurements whose derivatives LabCD should predict.",
        )
        selected_actions = st.multiselect(
            "System inputs · controls",
            choices,
            default=current_actions,
            key=f"{key}_actions",
            help="Choose the controls or other external inputs that drive the system.",
        )
        if not selected_states:
            editor_error = "Choose at least one measured state output."
        elif not selected_actions:
            editor_error = "Choose at least one system input or control."
        elif set(selected_states) & set(selected_actions):
            editor_error = "A data column cannot be both a state output and an input control."
        elif on_signals_saved is None:
            editor_error = "Signal editing is unavailable in this view. Reopen setup from the current dataset."

        angle_names = {
            current_states[index]
            for index in (settings.get("angle_indices") or [])
            if isinstance(index, int) and 0 <= index < len(current_states)
        }
        angle_features = sum(name in angle_names for name in selected_states)
        model_input_count = len(selected_states) + angle_features + len(selected_actions) + 1
        metrics = st.columns(4)
        metrics[0].metric("State outputs", len(selected_states))
        metrics[1].metric("Control inputs", len(selected_actions))
        metrics[2].metric("Model input nodes", model_input_count)
        metrics[3].metric("Model output nodes", len(selected_states))
        st.markdown(
            _signal_dimensions_visual(
                len(selected_states), len(selected_actions), model_input_count,
                settings["architecture"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="setup-editor-note"><b>How the dimensions are formed</b><br>'
            f'{len(selected_states)} measured states + {angle_features} angle features + '
            f'{len(selected_actions)} controls + 1 time step = {model_input_count} input features. '
            f'The network predicts one derivative for each of the {len(selected_states)} selected states. '
            'These counts follow the chosen data columns automatically.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Time stays the time axis. We want one `xdot_*` rate for every selected state; matching measured rates are kept, and missing rates are estimated from the state data.")
    elif section == "model":
        model = st.segmented_control("Model", ["LSTM", "MLP"],
                                     default=settings["architecture"], key=f"{key}_model",
                                     required=True, width="stretch")
        st.markdown(illustration(model, selected=True), unsafe_allow_html=True)
        if model == "LSTM":
            _editor_heading("Selected network", "LSTM · sequence memory", "Reads recent samples in time order.")
        else:
            _editor_heading("Selected network", "MLP · direct mapping", "Uses the current state and inputs.")
        st.caption("Both options predict state derivatives for the trajectory integrator.")
        updates["architecture"] = model
    elif section == "history":
        steps = st.slider("Past time steps", min_value=2, max_value=500,
                          value=int(settings["lstm_seq_length"]), key=f"{key}_steps")
        st.markdown(_history_visual(steps), unsafe_allow_html=True)
        interval = (dataset.get("analysis") or {}).get("median_dt")
        if interval and interval > 0:
            span = (steps - 1) * interval
            _editor_heading("History window", f"{steps} samples · {span:.4g} s", f"Measured interval: {interval:.4g} s per sample.")
        else:
            _editor_heading("History window", f"{steps} samples", "The dataset sample interval is not available yet.")
        updates["lstm_seq_length"] = int(steps)
    elif section == "physics":
        advisory = flow.get("physics_advisory") or {}
        if advisory.get("ask"):
            st.warning("The setup agent found a measured-data signal to review before applying physics guidance.")
            for cue in advisory.get("cues") or []:
                st.caption(f"Data check · {cue}")
        ready = pinn_maker.is_equation_ready(chat)
        approved = pinn_maker.is_equation_approved(chat)
        if ready and approved:
            complete_equation_set = len(equation_by_state) == len(physics_states)
            method_options = ["Off", "PINN", "PCNN"] if complete_equation_set else ["Off", "PCNN"]
            st.markdown('<div class="setup-editor-lead">PINN adds a soft equation loss. PCNN applies selected derivative rules exactly and lets the neural network learn all remaining states directly.</div>', unsafe_allow_html=True)
            if not complete_equation_set:
                st.caption("This source defines hard-constraint rules for only some states. It can be used with PCNN; PINN requires a complete derivative model.")
            default_method = settings.get("physics_mode", "pinn").upper() if settings["use_pinn"] else "Off"
            if default_method not in method_options:
                default_method = "PCNN" if not complete_equation_set and settings["use_pinn"] else "Off"
            method = st.segmented_control("Physics treatment", method_options,
                                          default=default_method, key=f"{key}_enabled",
                                          required=True, width="stretch")
            enabled = method != "Off"
            updates["use_pinn"] = enabled
            updates["physics_mode"] = method.lower() if enabled else settings.get("physics_mode", "pinn")
            if method == "PINN":
                weight_col, explanation_col = st.columns([1, 1], vertical_alignment="center")
                updates["pinn_loss_weight"] = weight_col.number_input(
                    "Physics loss strength", min_value=0.0, max_value=10000.0,
                    value=float(settings["pinn_loss_weight"]), step=0.05, key=f"{key}_weight",
                    help="Weights the soft mismatch between the network prediction and the complete equation model.")
                explanation_col.markdown('<div class="setup-editor-note"><b>PINN · soft guidance</b><br>The network predicts derivatives; the complete equation adds a weighted training loss.</div>', unsafe_allow_html=True)
            elif method == "PCNN":
                st.markdown("**Choose exact constraints**")
                st.caption("Check only state derivatives that must follow their supplied rule exactly. Every unchecked or unmapped state is predicted directly by the neural network.")
                if recommendations_by_state:
                    if st.button("Use agent suggestions", key=f"{key}_apply_suggestions",
                                 icon=":material/auto_awesome:", width="content"):
                        suggested_exact = {
                            state for state, item in recommendations_by_state.items()
                            if item.get("recommended_exact") is True and state in constraint_states
                        }
                        for index, state in enumerate(constraint_states):
                            st.session_state[f"{key}_hard_{index}"] = state in suggested_exact
                        st.rerun()
                selected = []
                for index, state in enumerate(constraint_states):
                    recommendation = recommendations_by_state.get(state, {})
                    suggested = bool(recommendation.get("recommended_exact"))
                    has_agent_suggestion = recommendation.get("source", "agent") == "agent"
                    suggestion_label = (
                        ("Agent suggests exact" if suggested else "Review before exact use")
                        if has_agent_suggestion else "Client-supplied rule"
                    )
                    with st.container(border=True):
                        rule_col, toggle_col = st.columns([2.3, 1], vertical_alignment="center")
                        rule_col.markdown(f"**{state}** · `{equation_by_state[state]}`")
                        exact = toggle_col.checkbox(
                            "Enforce exactly", key=f"{key}_hard_{index}",
                            help="Checked: this equation replaces the NN output for this state's derivative. Unchecked: the NN predicts this state directly from data.",
                        )
                        if exact:
                            selected.append(state)
                        confidence = (f"{str(recommendation.get('confidence', 'low')).title()} confidence"
                                      if has_agent_suggestion else "reviewed by you")
                        rationale = recommendation.get("rationale") or "Confirm that this rule should hold at every modeled step."
                        rule_col.caption(f"{suggestion_label} · {confidence}. {rationale}")
                omitted_states = [state for state in physics_states if state not in equation_by_state]
                if omitted_states:
                    st.caption("No hard rule supplied for " + ", ".join(omitted_states) + " · neural network outputs")
                st.session_state[f"{key}_hard_states"] = selected
                updates["pcnn_hard_constraint_states"] = selected
                if not selected:
                    editor_error = "Select at least one state derivative to enforce as a hard constraint."
                elif len(selected) >= len(physics_states):
                    editor_error = "Leave at least one state derivative for the neural network to learn."
                exact_word = "state" if len(selected) == 1 else "states"
                learned = max(0, len(physics_states) - len(selected))
                learned_word = "state" if learned == 1 else "states"
                st.caption(f"{len(selected)} exact hard-constrained {exact_word} · {learned} neural-network {learned_word}")
                st.markdown('<div class="setup-editor-note"><b>PCNN · hard constraints</b><br>Selected rules replace those neural outputs exactly. The NN learns every remaining state directly from measurements; there is no equation baseline or correction penalty.</div>', unsafe_allow_html=True)
        elif ready:
            st.info("The physics draft is ready for review in the conversation. Approve it there to unlock its training mode here.")
            st.caption("Until a model or constraint rule is approved, this setup uses measured-data training.")
            updates["use_pinn"] = False
        else:
            mode = str(settings.get("physics_mode", "pinn")).lower()
            lead = (
                "PCNN applies only selected state-derivative rules as exact constraints. Describe which relationships must always hold; the neural network learns the other states directly."
                if mode == "pcnn" else
                "PINN uses a complete reviewed derivative model as a soft training loss. Describe how each state responds to inputs so LabCD can draft a candidate model."
            )
            st.markdown(f'<div class="setup-editor-lead">{lead}</div>', unsafe_allow_html=True)
            description = st.text_area(
                "System and exact-constraint description" if mode == "pcnn" else "Physical system description",
                value=chat.get("pinn_model_description") or settings.get("customer_description", ""),
                max_chars=4000,
                placeholder=("For example: position changes exactly at the measured velocity; acceleration is unknown and should be learned from data."
                             if mode == "pcnn" else "For example: a cart moves on a track; position and velocity respond to measured force, with friction slowing it down."),
                key=f"{key}_physics_description",
                help=("Name the exact relationship and which measured state it constrains. Describe other states as unknown if they should remain neural outputs."
                      if mode == "pcnn" else "Include what each measured state means, what each input does, and any known units or mechanisms."))
            st.caption("Measured data can check whether a rule maps to available signals; it cannot prove that the rule is physically correct.")
            if st.button("Draft PCNN hard-constraint candidates" if mode == "pcnn" else "Draft PINN model",
                         icon=":material/auto_awesome:",
                         key=f"{key}_draft_physics", disabled=busy or not dataset.get("ready") or len(description.strip()) < 12,
                         width="stretch"):
                chat["pinn_model_description"] = description.strip()
                chat["settings"] = core.apply_changes(chat.get("settings") or {},
                                                       {"customer_description": description.strip()})
                core.add_message(chat, "user", description.strip(), kind="physics_model_request",
                                 answer_scope="setup")
                core.save_chat(chat)
                job = pinn_maker.PhysicsModelAgentJob(chat, description.strip())
                core.REGISTRY.conversations[chat["id"]] = job
                job.start()
                if on_interaction is not None:
                    on_interaction()
                st.rerun()
            if not dataset.get("ready"):
                st.caption("Attach a dataset and confirm its time, state, and input columns first.")
            st.caption("No physics rule is active yet; this setup uses measured data until you review and approve a draft.")
            updates["use_pinn"] = False
    elif section == "context":
        _editor_heading("System context", "Optional", "Add device details and units to help interpret signals.")
        updates["customer_description"] = st.text_area(
            "Device, process, and units (optional)", value=settings["customer_description"],
            max_chars=4000, placeholder="For example: motor position in radians; torque in N·m.",
            key=f"{key}_description")
        st.caption("This helps interpret results and is not required for training.")
    elif section == "goal":
        goal_descriptions = {
            "balanced": "Balance validation fit and practical limits.",
            "accuracy": "Prioritize a closer fit on held-out measurements.",
            "speed": "Favor faster predictions at inference time.",
            "compact": "Favor a smaller model footprint.",
        }
        selected_goal = st.segmented_control("What matters most?", list(GOALS),
                                             default=settings["optimization_goal"],
                                             format_func=lambda value: GOALS[value],
                                             key=f"{key}_goal", required=True, width="stretch")
        _editor_heading("Search priority", GOALS[selected_goal], goal_descriptions[selected_goal])
        updates["optimization_goal"] = selected_goal
    elif section == "effort":
        modes = ["fast", "regular", "heavy"]
        current = settings["run_mode"] if settings["run_mode"] in modes else "regular"
        mode = st.segmented_control("Search effort", modes, default=current,
                                    format_func=str.title, required=True, width="stretch", key=f"{key}_mode")
        limits = cfg.run_mode_limits(mode)
        hours = float(limits["max_hours"])
        time_text = f"{hours * 60:g} minutes" if hours < 1 else f"{hours:g} hours"
        profiles = []
        for option in modes:
            profile = cfg.run_mode_limits(option)
            cap_hours = float(profile["max_hours"])
            cap_time = f"{cap_hours * 60:g} min" if cap_hours < 1 else f"{cap_hours:g} hr"
            chosen = " selected" if option == mode else ""
            profiles.append(f'<div class="setup-effort-card{chosen}"><small>{option.upper()} SEARCH</small>'
                            f'<strong>{profile["max_cycles"]} cycles</strong><span>up to {cap_time}</span></div>')
        st.markdown(f'<div class="setup-effort-options">{"".join(profiles)}</div>', unsafe_allow_html=True)
        st.caption(f"The {mode} profile stops at its cycle or time limit, whichever comes first.")
        updates["run_mode"] = mode
        updates["max_cycles"] = min(int(settings["max_cycles"]), int(limits["max_cycles"]))
    elif section == "cycles":
        limits = cfg.run_mode_limits(settings["run_mode"])
        cap = int(limits["max_cycles"])
        updates["max_cycles"] = st.slider("Candidate cycles", min_value=1, max_value=cap,
                                           value=min(int(settings["max_cycles"]), cap), key=f"{key}_count")
        st.markdown(f'<div class="setup-editor-note"><b>{updates["max_cycles"]} of {cap} cycle slots selected</b><br>Each cycle proposes and evaluates one candidate model under the {settings["run_mode"]} search profile.</div>', unsafe_allow_html=True)
    elif section == "network":
        width_min = int(settings["hidden_size_min"])
        width_max = int(settings["hidden_size_max"])
        layer_min = int(settings["num_layers_min"])
        layer_max = int(settings["num_layers_max"])
        proposed_widths = _starting_layers(settings)
        if layer_min == layer_max:
            layer_count = layer_min
            st.caption(f"The current search bounds specify {layer_count} {'recurrent' if settings['architecture'] == 'LSTM' else 'dense'} layer(s).")
        else:
            layer_count = st.slider(
                "Recurrent layers" if settings["architecture"] == "LSTM" else "Dense layers",
                min_value=layer_min, max_value=layer_max,
                value=min(max(len(proposed_widths), layer_min), layer_max), key=f"{key}_layers",
                help="This is the exact starting layer count used for the first candidate.")
        if settings["architecture"] == "LSTM":
            layer_label = "recurrent"
        else:
            layer_label = "dense"
        widths = []
        width_columns = st.columns(min(int(layer_count), 3), gap="small")
        for index in range(int(layer_count)):
            value = proposed_widths[min(index, len(proposed_widths) - 1)]
            with width_columns[index % len(width_columns)]:
                widths.append(int(st.number_input(
                    f"Layer {index + 1} units", min_value=width_min, max_value=width_max,
                    value=min(max(value, width_min), width_max), step=8, key=f"{key}_width_{index}")))
        layer_kind = "LSTM" if settings["architecture"] == "LSTM" else "MLP"
        st.markdown(
            f'<div class="setup-editor-note"><b>{layer_kind} · {" → ".join(map(str, widths))} units</b>'
            f'<br>Each number sets the width of one {layer_label} layer.</div>',
            unsafe_allow_html=True,
        )
        updates["manual_starting_hidden_layers"] = widths
        metric_cols = st.columns(3)
        metric_cols[0].metric("Layers", len(widths))
        metric_cols[1].metric("Layer widths", " → ".join(map(str, widths)))
        metric_cols[2].metric("Topology", "Recurrent" if settings["architecture"] == "LSTM" else "Dense")
    elif section == "training":
        st.markdown('<div class="setup-editor-lead">Set the values used by the first candidate. They will be applied to the Initializer proposal when this setup is approved.</div>', unsafe_allow_html=True)
        low, high = float(settings["learning_rate_min"]), float(settings["learning_rate_max"])
        left, right = st.columns(2, gap="medium")
        updates["manual_starting_lr"] = left.number_input(
            "Starting learning rate", min_value=low, max_value=high,
            value=min(max(_starting_lr(settings), low), high), step=max((high - low) / 100, 0.00000001),
            format="%.7f", key=f"{key}_learning_rate")
        updates["epochs"] = int(right.number_input(
            "Epoch limit per candidate", min_value=1, max_value=5000,
            value=int(settings["epochs"]), step=10, key=f"{key}_epochs"))
        left, right = st.columns(2, gap="medium")
        updates["batch_size"] = int(left.number_input(
            "Batch size", min_value=8, max_value=4096,
            value=int(settings["batch_size"]), step=8, key=f"{key}_batch"))
        updates["early_stop_patience"] = int(right.number_input(
            "Early-stop patience", min_value=1, max_value=1000,
            value=int(settings["early_stop_patience"]), key=f"{key}_patience"))
        left, right = st.columns(2, gap="medium")
        updates["manual_dropout_rate"] = left.slider(
            "Dropout", min_value=0.0, max_value=float(cfg.DROPOUT_RATE_MAX),
            step=0.01, value=float(settings["manual_dropout_rate"]), key=f"{key}_dropout")
        updates["manual_weight_decay"] = right.number_input(
            "Weight decay", min_value=0.0, max_value=float(cfg.WEIGHT_DECAY_MAX),
            step=0.0001, format="%.6f", value=float(settings["manual_weight_decay"]), key=f"{key}_decay")
        if settings["architecture"] == "MLP":
            activations = ["relu", "leaky_relu", "elu", "tanh", "swish"]
            updates["manual_activation"] = st.selectbox(
                "Hidden-layer activation", activations, index=activations.index(settings["manual_activation"]),
                format_func=str.upper, key=f"{key}_activation")
        else:
            updates["manual_activation"] = settings["manual_activation"]
            st.caption("LSTM gates use their built-in activations. Dropout takes effect with two or more recurrent layers.")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Starting LR", f"{updates['manual_starting_lr']:.3g}")
        summary_cols[1].metric("Epoch limit", updates["epochs"])
        summary_cols[2].metric("Batch size", updates["batch_size"])
        summary_cols[3].metric("Patience", updates["early_stop_patience"])
        activation_note = ("LSTM uses its built-in gate activations" if settings["architecture"] == "LSTM"
                           else f"{updates['manual_activation'].upper()} activation")
        st.markdown(
            f'<div class="setup-editor-note"><b>Regularization</b><br>Dropout {updates["manual_dropout_rate"]:.2f} · '
            f'weight decay {updates["manual_weight_decay"]:.5g} · {activation_note}</div>',
            unsafe_allow_html=True)
    elif section == "targets":
        mse_col, latency_col = st.columns(2)
        with mse_col.container(border=True):
            updates["mse_target"] = st.number_input(
                "Validation MSE target", min_value=0.00000001, max_value=100000.0,
                value=float(settings["mse_target"]), format="%.8f", key=f"{key}_mse",
                help="The search target for mean squared error on validation data.")
            st.caption("Lower values ask for a closer validation fit.")
        with latency_col.container(border=True):
            updates["customer_max_latency_ms"] = st.number_input(
                "Inference time budget (ms)", min_value=0.001, max_value=60000.0,
                value=float(settings["customer_max_latency_ms"]), key=f"{key}_latency",
                help="Maximum acceptable prediction time per inference call.")
            st.caption("The measured prediction budget for one call.")
    elif section == "advanced":
        st.markdown('<div class="setup-editor-lead">Tune the training and data handling details. The current values are ready for a first run.</div>', unsafe_allow_html=True)
        bounds_tab, data_tab, solver_tab, full_tab = st.tabs(
            ["Search bounds", "Data", "Integrator", "Full settings"])
        with bounds_tab:
            st.caption("Bounds used by the model search to explore other candidates.")
            low, high = st.columns(2)
            updates["learning_rate_min"] = low.number_input(
                "Minimum learning rate", min_value=0.00000001, max_value=1.0,
                value=float(settings["learning_rate_min"]), format="%.8f", key=f"{key}_lr_min")
            updates["learning_rate_max"] = high.number_input(
                "Maximum learning rate", min_value=0.00000001, max_value=1.0,
                value=float(settings["learning_rate_max"]), format="%.8f", key=f"{key}_lr_max")
            updates["adaptive_regularization"] = st.toggle(
                "Adjust regularization when validation overfits",
                value=bool(settings["adaptive_regularization"]), key=f"{key}_adaptive_reg")
            with st.expander("Learning-rate schedule and overfit threshold"):
                schedule_left, schedule_right = st.columns(2)
                updates["lr_reduce_factor"] = schedule_left.number_input(
                    "Reduce rate by", min_value=0.01, max_value=0.99,
                    value=float(settings["lr_reduce_factor"]), step=0.05, key=f"{key}_lr_reduce")
                updates["lr_schedule_min_floor"] = schedule_right.number_input(
                    "Minimum schedule floor", min_value=0.00000001, max_value=1.0,
                    value=float(settings["lr_schedule_min_floor"]), format="%.8f", key=f"{key}_lr_floor")
                schedule_left, schedule_right = st.columns(2)
                updates["adaptive_improvement_threshold"] = schedule_left.number_input(
                    "Improvement threshold", min_value=0.0, max_value=1.0,
                    value=float(settings["adaptive_improvement_threshold"]), step=0.01, key=f"{key}_improve")
                updates["epoch_extension_steps"] = int(schedule_right.number_input(
                    "Extra epochs per extension", min_value=1, max_value=1000,
                    value=int(settings["epoch_extension_steps"]), key=f"{key}_extension"))
                updates["overfit_ratio_limit"] = st.number_input(
                    "Overfit ratio limit", min_value=1.01, max_value=1000.0,
                    value=float(settings["overfit_ratio_limit"]), step=0.5, key=f"{key}_overfit")
            updates["hidden_size_min"] = int(low.number_input(
                "Minimum hidden width", min_value=4, max_value=2048,
                value=int(settings["hidden_size_min"]), key=f"{key}_width_min"))
            updates["hidden_size_max"] = int(high.number_input(
                "Maximum hidden width", min_value=4, max_value=2048,
                value=int(settings["hidden_size_max"]), key=f"{key}_width_max"))
            low, high = st.columns(2)
            updates["num_layers_min"] = int(low.number_input(
                "Minimum layer count", min_value=1, max_value=8,
                value=int(settings["num_layers_min"]), key=f"{key}_layers_min"))
            updates["num_layers_max"] = int(high.number_input(
                "Maximum layer count", min_value=1, max_value=8,
                value=int(settings["num_layers_max"]), key=f"{key}_layers_max"))
            updates["choose_via_llm_initializer"] = st.toggle(
                "Let the Initializer choose a starting network", value=bool(settings["choose_via_llm_initializer"]),
                key=f"{key}_initializer")
        with data_tab:
            st.caption("Choose how measured trajectories are filtered and interpreted.")
            methods = ["finite_difference", "sliding_mode", "savitzky_golay"]
            updates["derivative_method"] = st.selectbox(
                "Derivative estimation", methods, index=methods.index(settings["derivative_method"]),
                format_func=lambda value: {"finite_difference": "Finite difference", "sliding_mode": "Sliding mode", "savitzky_golay": "Savitzky–Golay"}[value],
                key=f"{key}_derivative_method")
            updates["derivative_filter_tau"] = st.number_input(
                "Derivative filter time constant", min_value=0.0, max_value=100.0,
                value=float(settings["derivative_filter_tau"]), format="%.5f", key=f"{key}_tau")
            updates["use_state_filter"] = st.toggle(
                "Filter state outliers", value=bool(settings["use_state_filter"]), key=f"{key}_filter")
            if updates["use_state_filter"]:
                low, high = settings["auto_filter_percentiles"]
                a, b = st.columns(2)
                updates["auto_filter_percentiles"] = [
                    a.number_input("Lower percentile", min_value=0.0, max_value=100.0,
                                   value=float(low), key=f"{key}_lower"),
                    b.number_input("Upper percentile", min_value=0.0, max_value=100.0,
                                   value=float(high), key=f"{key}_upper"),
                ]
            updates["auto_detect_angles"] = st.toggle(
                "Automatically detect angle states", value=bool(settings["auto_detect_angles"]), key=f"{key}_angles_auto")
            indices = st.text_input("Angle state indices", value=", ".join(map(str, settings["angle_indices"])),
                                    placeholder="e.g. 0, 2", key=f"{key}_angle_indices")
            try:
                updates["angle_indices"] = [int(part.strip()) for part in indices.split(",") if part.strip()]
            except ValueError:
                editor_error = "Angle indices must be whole numbers separated by commas."
            updates["multi_trajectory"] = st.toggle(
                "Treat the file as multiple trajectories", value=bool(settings["multi_trajectory"]), key=f"{key}_multi")
            updates["shuffle_data"] = st.toggle("Shuffle training samples", value=bool(settings["shuffle_data"]), key=f"{key}_shuffle")
            updates["trajectory_chunk_size"] = int(st.number_input(
                "Trajectory chunk size (0 = automatic)", min_value=0, max_value=1000000,
                value=int(settings["trajectory_chunk_size"]), key=f"{key}_chunk"))
            with st.expander("Signal-specific filters"):
                low, high = st.columns(2)
                updates["savgol_window"] = int(low.number_input(
                    "Savitzky–Golay window", min_value=3, max_value=1001, step=2,
                    value=int(settings["savgol_window"]), key=f"{key}_savgol_window"))
                updates["savgol_polyorder"] = int(high.number_input(
                    "Polynomial order", min_value=1, max_value=10,
                    value=int(settings["savgol_polyorder"]), key=f"{key}_savgol_poly"))
                low, high = st.columns(2)
                updates["smd_lambda_1"] = low.number_input(
                    "Sliding-mode lambda 1", min_value=0.000001, max_value=100000.0,
                    value=float(settings["smd_lambda_1"]), key=f"{key}_smd_l1")
                updates["smd_lambda_2"] = high.number_input(
                    "Sliding-mode lambda 2", min_value=0.000001, max_value=100000.0,
                    value=float(settings["smd_lambda_2"]), key=f"{key}_smd_l2")
            split_times = st.text_input(
                "Trajectory split times (optional)", value=", ".join(map(str, settings["manual_split_times"])),
                placeholder="e.g. 2.5, 5.0", key=f"{key}_split_times")
            try:
                updates["manual_split_times"] = [float(part.strip()) for part in split_times.split(",") if part.strip()]
            except ValueError:
                editor_error = "Split times must be numbers separated by commas."
        with solver_tab:
            st.caption("Set how trajectories are advanced and when they reset.")
            updates["integrator_type"] = st.segmented_control(
                "Trajectory integrator", ["EULER", "RK4"], default=settings["integrator_type"],
                format_func=lambda value: "Euler" if value == "EULER" else "Runge–Kutta 4",
                required=True, width="stretch", key=f"{key}_integrator")
            updates["rollout_horizon"] = int(st.slider(
                "Rollout horizon", min_value=1, max_value=100,
                value=int(settings["rollout_horizon"]), key=f"{key}_horizon"))
            threshold = st.text_input(
                "Reset threshold (optional)", value=", ".join(map(str, settings["reset_threshold"] or [])),
                placeholder="e.g. 0.5, 1.2", key=f"{key}_reset")
            updates["reset_threshold"] = None
            if threshold.strip():
                try:
                    updates["reset_threshold"] = [float(part.strip()) for part in threshold.split(",")]
                    if any(not part.strip() for part in threshold.split(",")):
                        raise ValueError
                except ValueError:
                    editor_error = "Reset thresholds must be numbers separated by commas."
        with full_tab:
            raw = st.text_area("Full configuration JSON", value=json.dumps(settings, indent=2),
                               height=220, key=f"{key}_json")
            st.caption("For expert edits. Values in the structured tabs take precedence.")

        if editor_error:
            st.error(editor_error)

    reset, cancel, save = st.columns([1.45, 0.8, 1.35], gap="small")
    reset.button(
        "Reset to Initializer proposal", icon=":material/restart_alt:",
        key=f"{key}_reset", width="stretch",
        help="Restore this section to the setup values saved before any manual edits. Save change to apply.",
        on_click=_reset_setup_widgets,
        args=(section, key, proposal, proposal_roles, settings),
    )
    if cancel.button("Cancel", key=f"{key}_cancel", width="stretch"):
        st.rerun()
    if save.button("Save change", type="primary", icon=":material/check:",
                   key=f"{key}_save", width="stretch", disabled=bool(editor_error)):
        try:
            if section == "signals":
                changed = on_signals_saved(chat, selected_states, selected_actions)
                if not changed:
                    st.toast("Those system signal roles are already selected.")
                st.rerun()
            elif section == "advanced":
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Full settings must be a JSON object.")
                parsed.update(updates)
                updates = parsed
                bounds = updates.get("auto_filter_percentiles")
                if bounds and float(bounds[0]) >= float(bounds[1]):
                    raise ValueError("The lower percentile must be below the upper percentile.")
                chat["settings"] = core.apply_changes(chat["settings"], updates)
                core.save_chat(chat)
                st.rerun()
            else:
                updated_settings = core.apply_changes(chat["settings"], updates)
                if section == "physics" and updates.get("use_pinn") and updates.get("physics_mode") == "pcnn":
                    equation = chat.get("pinn_equation") or {}
                    chat["pcnn_constraints_reviewed_for"] = equation.get("proposal_id", equation.get("sha256"))
                chat["settings"] = updated_settings
                core.save_chat(chat)
                st.rerun()
        except (TypeError, ValueError) as exc:
            st.error(f"Check this setting: {exc}")


def approve_and_start_training(chat: dict) -> bool:
    """Apply the displayed starting values and queue the approved run."""
    dataset = chat.get("dataset") or {}
    if not dataset.get("ready") or chat.get("run_dir"):
        return False
    current = core.RunSettings.model_validate(chat["settings"]).model_dump()
    low, high = float(current["learning_rate_min"]), float(current["learning_rate_max"])
    approved = {
        "manual_starting_lr": min(max(_starting_lr(current), low), high),
        "manual_starting_hidden_layers": [
            min(max(int(width), int(current["hidden_size_min"])), int(current["hidden_size_max"]))
            for width in _starting_layers(current)
        ],
    }
    chat["settings"] = core.apply_changes(chat["settings"], approved)
    flow = chat.get("run_setup_flow") or {}
    flow["stage"] = "approved"
    flow["dataset_sha256"] = dataset.get("sha256")
    flow["preapproved_initializer"] = True
    chat["run_setup_flow"] = flow
    core.save_chat(chat)
    st.session_state["conversation_pending"] = "start"
    return True


def render(chat: dict, flow: dict, settings: dict, busy: bool, illustration,
           on_signals_saved=None, on_interaction=None) -> None:
    dataset = chat.get("dataset") or {}
    suffix = f"{chat['id']}_{str(dataset.get('sha256', ''))[:8]}"
    with st.container(border=True, key=f"run_setup_card_{suffix}"):
        st.caption("RUN SETUP · REVIEW YOUR SETTINGS")
        st.markdown("### Starting setup")
        st.caption("Your chosen network is highlighted below. Use a pen to edit any setting, then approve this setup once to start training.")
        if flow.get("agent_error"):
            st.warning("The setup agent was unavailable. These are editable starting defaults.")
        else:
            st.caption(f"Agent suggestion · {flow.get('recommended_architecture', settings['architecture'])} model · "
                       f"{flow.get('recommended_effort', settings['run_mode']).title()} search")

        _chosen_model(chat, settings, flow, suffix, illustration, busy,
                      on_interaction=on_interaction)

        if settings["use_pinn"] and pinn_maker.is_equation_ready(chat):
            physics = ("PCNN · hard constraints"
                       if str(settings.get("physics_mode", "pinn")).lower() == "pcnn"
                       else "PINN · soft physics loss")
        else:
            physics = "Off · measured data"
        state_names = list(dataset.get("states") or [])
        action_names = list(dataset.get("inputs") or dataset.get("actions") or [])
        state_summary = f"{len(state_names)} measured outputs"
        if state_names:
            state_summary += " · " + ", ".join(state_names[:3])
            if len(state_names) > 3:
                state_summary += ", …"
        action_summary = f"{len(action_names)} system inputs"
        if action_names:
            action_summary += " · " + ", ".join(action_names[:3])
            if len(action_names) > 3:
                action_summary += ", …"
        rows = [
            ("System inputs and outputs", f"{state_summary} · {action_summary}", "signals"),
            ("LSTM history", f"{settings['lstm_seq_length']} time steps" if settings["architecture"] == "LSTM" else "Not used for MLP · choose LSTM to configure", "history"),
            ("Network shape", _network_label(settings), "network"),
            ("Training controls", _training_label(settings), "training"),
            ("Physics method", physics, "physics"),
            ("System context", " ".join(settings["customer_description"].split())[:80] or "Not provided · optional", "context"),
            ("Optimization priority", GOALS.get(settings["optimization_goal"], "Balanced"), "goal"),
            ("Search effort", settings["run_mode"].title(), "effort"),
            ("Candidate cycles", f"Up to {settings['max_cycles']}", "cycles"),
            ("Run targets", f"MSE {settings['mse_target']:.4g} · {settings['customer_max_latency_ms']:g} ms", "targets"),
            ("Advanced controls", f"Data handling · search bounds · integrator", "advanced"),
        ]
        for label, value, section in rows:
            is_mlp_history = section == "history" and settings["architecture"] != "LSTM"
            _row(label, value, section, chat, suffix, illustration,
                 disabled=busy or (section == "signals" and not dataset.get("ready")),
                 edit_label="Change model" if is_mlp_history else None,
                 editor_section="model" if is_mlp_history else None,
                 on_signals_saved=on_signals_saved, on_interaction=on_interaction)

        st.caption("One approval applies the network and training values above, then starts the run.")
        if st.button("Approve setup and start training", type="primary", icon=":material/play_arrow:",
                     key=f"setup_start_{suffix}", disabled=busy or not dataset.get("ready"), width="stretch"):
            if approve_and_start_training(chat):
                st.rerun()
