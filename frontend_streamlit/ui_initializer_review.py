"""One editable, visual approval step for the Initializer's first candidate."""
from __future__ import annotations

import html

import streamlit as st

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.pipeline import estimate_model_parameter_count


def _proposal(config: dict) -> dict:
    widths = [int(value) for value in (config.get("hidden_layers") or [64])]
    return {
        "layers": len(widths),
        "units": widths[0],
        "widths": widths,
        "learning_rate": float(config.get("learning_rate", 0.001)),
        "epochs": int(config.get("epochs", cfg.EPOCHS)),
        "batch_size": int(config.get("batch_size", cfg.BATCH_SIZE)),
        "early_stop_patience": int(config.get("early_stop_patience", cfg.EARLY_STOP_PATIENCE)),
        "activation": str(config.get("activation", "relu")).lower(),
        "dropout_rate": float(config.get("dropout_rate", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }


def _reset_widgets(prefix: str, proposal: dict, maximum_layers: int) -> None:
    for name in (
        "layers", "learning_rate", "epochs", "batch_size",
        "early_stop_patience", "activation", "dropout_rate", "weight_decay",
    ):
        st.session_state[f"{prefix}_{name}"] = proposal[name]
    for index in range(maximum_layers):
        st.session_state[f"{prefix}_width_{index}"] = proposal["widths"][
            min(index, len(proposal["widths"]) - 1)
        ]


def _diagram(architecture: str, widths: list[int], input_count: int, output_count: int,
             *, history: int, activation: str, dropout: float, weight_decay: float,
             learning_rate: float, epochs: int, batch_size: int, patience: int,
             parameter_count: int) -> str:
    """Draw a truthful, compact representation of the architecture in SVG."""
    layer_count = len(widths)
    if layer_count == 1:
        hidden_x = [475]
    else:
        hidden_x = [300 + index * (350 / (layer_count - 1)) for index in range(layer_count)]
    x_positions = [92, *hidden_x, 810]
    # The input and output layers reflect the actual feature and state counts.
    # Hidden layers stay schematic so large widths do not overwhelm the preview.
    node_counts = [
        max(1, int(input_count)),
        *[min(4, max(1, int(width))) for width in widths],
        max(1, int(output_count)),
    ]

    def node_y(count: int) -> list[float]:
        visible = max(1, int(count))
        if visible == 1:
            return [160.0]
        spacing = 138 / (visible - 1)
        return [91 + index * spacing for index in range(visible)]

    def node_radius(count: int) -> float:
        visible = max(1, int(count))
        if visible == 1:
            return 9.0
        spacing = 138 / (visible - 1)
        return min(9.0, max(0.7, spacing * 0.36))

    rows = [node_y(count) for count in node_counts]
    radii = [node_radius(count) for count in node_counts]
    wires = []
    for left in range(len(x_positions) - 1):
        x1, x2 = x_positions[left], x_positions[left + 1]
        for start_y in rows[left]:
            for end_y in rows[left + 1]:
                wires.append(
                    f'<path class="setup-net-wire" d="M{x1 + 11:.1f} {start_y:.1f} '
                    f'C{x1 + 70:.1f} {start_y:.1f} {x2 - 70:.1f} {end_y:.1f} '
                    f'{x2 - 11:.1f} {end_y:.1f}"/>'
                )
    blocks = []
    for index, (x, width) in enumerate(zip(hidden_x, widths), start=1):
        label = f"LSTM {index}" if architecture == "LSTM" else f"Dense {index}"
        recurrence = (
            f'<path class="setup-net-loop" d="M{x + 23:.1f} 87 '
            f'C{x + 62:.1f} 48 {x - 60:.1f} 49 {x - 24:.1f} 87"/>'
            if architecture == "LSTM" else ""
        )
        circles = "".join(
            f'<circle class="setup-net-node hidden" cx="{x:.1f}" cy="{y:.1f}" r="{radii[index]:.1f}"/>'
            for y in rows[index]
        )
        blocks.append(
            f'<rect class="setup-net-layer" x="{x - 45:.1f}" y="69" width="90" height="183" rx="18"/>'
            f'{recurrence}{circles}'
            f'<text class="setup-net-label" x="{x:.1f}" y="285" text-anchor="middle">'
            f'{html.escape(label)}</text>'
            f'<text class="setup-net-count" x="{x:.1f}" y="303" text-anchor="middle">'
            f'{width:,} units</text>'
        )
    inputs = "".join(
        f'<circle class="setup-net-node input" cx="92" cy="{y:.1f}" r="{radii[0]:.1f}"/>'
        for y in rows[0]
    )
    outputs = "".join(
        f'<circle class="setup-net-node output" cx="810" cy="{y:.1f}" r="{radii[-1]:.1f}"/>'
        for y in rows[-1]
    )
    topology = (
        f"{architecture}: {layer_count} recurrent layers, {widths[0]} units each"
        if architecture == "LSTM" else
        f"MLP: {layer_count} dense layers, widths {', '.join(map(str, widths))}"
    )
    effective_dropout = dropout if architecture == "MLP" or layer_count > 1 else 0.0
    subtitle = (
        f"{history} time steps of history · recurrent memory"
        if architecture == "LSTM" else
        f"{activation.upper()} activation · feedforward layers"
    )
    description = html.escape(
        f"{topology}. {input_count} input features map to {output_count} state derivatives. "
        f"Estimated {parameter_count:,} trainable parameters."
    )
    return f'''<div class="setup-network-shell">
  <div class="setup-network-heading">
    <span class="setup-network-kicker">LIVE ARCHITECTURE PREVIEW</span>
    <span class="setup-network-meta">{parameter_count:,} trainable parameters</span>
  </div>
  <svg class="setup-network-svg" viewBox="0 0 900 320" role="img" aria-label="{description}">
    <title>{html.escape(topology)}</title><desc>{description}</desc>
    <rect class="setup-net-ground" x="1" y="1" width="898" height="318" rx="17"/>
    <path class="setup-net-axis" d="M35 160H865"/>
    {''.join(wires)}
    {''.join(blocks)}
    {inputs}{outputs}
    <text class="setup-net-label" x="92" y="285" text-anchor="middle">Input</text>
    <text class="setup-net-count" x="92" y="303" text-anchor="middle">{input_count:,} features</text>
    <text class="setup-net-label" x="810" y="285" text-anchor="middle">Output</text>
    <text class="setup-net-count" x="810" y="303" text-anchor="middle">{output_count:,} derivatives</text>
  </svg>
  <div class="setup-network-foot">
    <span>{html.escape(subtitle)}</span>
    <span>Network output → state derivatives</span>
  </div>
  <div class="setup-network-metrics" aria-label="Live training settings">
    <span><small>LEARNING RATE</small><strong>{learning_rate:.5g}</strong></span>
    <span><small>EPOCH LIMIT</small><strong>{epochs:,}</strong></span>
    <span><small>BATCH SIZE</small><strong>{batch_size:,}</strong></span>
    <span><small>EARLY STOP</small><strong>{patience:,}</strong></span>
    <span><small>EFFECTIVE DROPOUT</small><strong>{effective_dropout:.2f}</strong></span>
    <span><small>WEIGHT DECAY</small><strong>{weight_decay:.5g}</strong></span>
  </div>
</div>'''


def render(checkpoint: dict, prefix: str) -> tuple[str, dict | None] | None:
    """Render the single review card; return a decision only on approval."""
    config = checkpoint.get("config") or {}
    dataset = checkpoint.get("dataset") or {}
    setup = checkpoint.get("setup") or {}
    architecture = str(setup.get("architecture", "LSTM")).upper()
    proposed = _proposal(config)
    width_min = int(config.get("hidden_size_search_min", cfg.HIDDEN_SIZE_MIN))
    width_max = int(config.get("hidden_size_search_max", cfg.HIDDEN_SIZE_MAX))
    layer_min = int(config.get("num_layers_search_min", cfg.NUM_LAYERS_MIN))
    layer_max = int(config.get("num_layers_search_max", cfg.NUM_LAYERS_MAX))
    lr_min = float(config.get("lr_search_min", cfg.LEARNING_RATE_MIN))
    lr_max = float(config.get("lr_search_max", cfg.LEARNING_RATE_MAX))
    width_min, width_max = min(width_min, width_max), max(width_min, width_max)
    layer_min, layer_max = min(layer_min, layer_max), max(layer_min, layer_max)
    lr_min, lr_max = min(lr_min, lr_max), max(lr_min, lr_max)

    defaults = {
        "layers": proposed["layers"],
        "learning_rate": proposed["learning_rate"], "epochs": proposed["epochs"],
        "batch_size": proposed["batch_size"],
        "early_stop_patience": proposed["early_stop_patience"],
        "activation": proposed["activation"],
        "dropout_rate": proposed["dropout_rate"],
        "weight_decay": proposed["weight_decay"],
    }
    for name, value in defaults.items():
        st.session_state.setdefault(f"{prefix}_{name}", value)
    for index in range(layer_max):
        st.session_state.setdefault(
            f"{prefix}_width_{index}", proposed["widths"][min(index, proposed["layers"] - 1)]
        )

    layer_key = f"{prefix}_layers"
    stored_layer_count = int(st.session_state[layer_key])
    layer_count = (
        layer_min if layer_min == layer_max
        else min(max(stored_layer_count, layer_min), layer_max)
    )
    if stored_layer_count != layer_count:
        st.session_state[layer_key] = layer_count
    widths = [int(st.session_state[f"{prefix}_width_{index}"]) for index in range(layer_count)]
    states = list(dataset.get("states") or [])
    actions = list(dataset.get("inputs") or dataset.get("actions") or [])
    angle_names = set(dataset.get("periodic_angle_states") or [])
    angle_indices = [index for index, name in enumerate(states) if name in angle_names]
    feature_count = len(states) + len(angle_indices) + len(actions) + 1
    parameter_count = estimate_model_parameter_count(
        len(states), len(actions), widths, architecture, angle_indices=angle_indices
    )
    history = int(setup.get("history_steps", 1) or 1)
    run_mode = str(setup.get("run_mode", "fast")).title()
    cycles = int(setup.get("max_cycles", 1) or 1)

    st.caption("INITIAL MODEL PROPOSAL · BEFORE TRAINING")
    st.markdown("### See your starting network")
    st.caption("Change the controls below to preview the exact starting shape. Approving it begins candidate training.")
    with st.container(horizontal=True, gap="small", wrap=True):
        st.badge(f"{architecture} model", icon=":material/account_tree:", color="blue")
        st.badge(f"{run_mode} · {cycles} cycles", icon=":material/speed:", color="green")
        st.badge(f"{int(dataset.get('samples', 0) or 0):,} rows", icon=":material/database:", color="gray")
    st.markdown(
        _diagram(
            architecture, widths, feature_count, len(states), history=history,
            activation=str(st.session_state[f"{prefix}_activation"]),
            dropout=float(st.session_state[f"{prefix}_dropout_rate"]),
            weight_decay=float(st.session_state[f"{prefix}_weight_decay"]),
            learning_rate=float(st.session_state[f"{prefix}_learning_rate"]),
            epochs=int(st.session_state[f"{prefix}_epochs"]),
            batch_size=int(st.session_state[f"{prefix}_batch_size"]),
            patience=int(st.session_state[f"{prefix}_early_stop_patience"]),
            parameter_count=parameter_count,
        ),
        unsafe_allow_html=True,
    )
    state_features = f"{len(states)} states"
    if angle_indices:
        state_features += f" + {len(angle_indices)} angle features"
    st.caption(
        f"Input = {state_features} + {len(actions)} controls + time step. "
        f"The network predicts {len(states)} state derivatives; the integrator produces the trajectory."
    )

    st.markdown("#### Network shape")
    if layer_min == layer_max:
        layer_count = layer_min
        st.session_state[f"{prefix}_layers"] = layer_count
        label = "recurrent" if architecture == "LSTM" else "dense"
        st.caption(f"{layer_count} {label} {'layer' if layer_count == 1 else 'layers'} in this search range")
    else:
        layer_count = st.slider(
            "Recurrent layers" if architecture == "LSTM" else "Dense layers",
            min_value=layer_min, max_value=layer_max, key=f"{prefix}_layers",
            help="Each layer is shown in the network preview above.",
        )
    width_columns = st.columns(min(int(layer_count), 3), gap="small")
    widths = []
    for index in range(int(layer_count)):
        with width_columns[index % len(width_columns)]:
            width = st.number_input(
                f"Layer {index + 1} units", min_value=width_min, max_value=width_max,
                step=8, key=f"{prefix}_width_{index}",
            )
            widths.append(int(width))
    if architecture == "LSTM":
        st.caption(f"LSTM layer widths · {' → '.join(map(str, widths))} units.")
    else:
        st.selectbox(
            "Hidden-layer activation", cfg.AVAILABLE_ACTIVATIONS,
            key=f"{prefix}_activation", format_func=lambda value: value.upper(),
        )

    st.markdown("#### Training controls")
    left, right = st.columns(2, gap="medium")
    with left:
        st.number_input(
            "Starting learning rate", min_value=lr_min, max_value=lr_max,
            step=0.00001, format="%.7f", key=f"{prefix}_learning_rate",
        )
        st.number_input(
            "Epoch limit per candidate", min_value=1, max_value=5000,
            step=10, key=f"{prefix}_epochs",
        )
        st.number_input(
            "Batch size", min_value=8, max_value=4096,
            step=8, key=f"{prefix}_batch_size",
        )
    with right:
        st.number_input(
            "Early-stop patience", min_value=1, max_value=1000,
            step=1, key=f"{prefix}_early_stop_patience",
        )
        st.slider(
            "Dropout", min_value=0.0, max_value=float(cfg.DROPOUT_RATE_MAX),
            step=0.01, key=f"{prefix}_dropout_rate",
        )
        st.number_input(
            "Weight decay", min_value=0.0, max_value=float(cfg.WEIGHT_DECAY_MAX),
            step=0.0001, format="%.6f", key=f"{prefix}_weight_decay",
        )
    if architecture == "LSTM":
        st.caption("LSTM gates use their built-in activations. Dropout takes effect when there are at least two recurrent layers.")
    else:
        st.caption("Dropout follows each dense hidden layer. Weight decay applies to training in either architecture.")

    with st.expander("Why these settings? · data and search bounds"):
        reasoning = str(config.get("reasoning", "")).strip()
        if reasoning:
            st.markdown("**Initializer explanation**")
            st.write(reasoning)
        train = int(dataset.get("train_samples", 0) or 0)
        validation = int(dataset.get("validation_samples", 0) or 0)
        test = int(dataset.get("test_samples", 0) or 0)
        st.caption(f"Data split · {train:,} train · {validation:,} validation · {test:,} held out")
        st.caption(
            f"Search bounds · {layer_min}–{layer_max} layers · {width_min}–{width_max} units "
            f"· learning rate {lr_min:.3g}–{lr_max:.3g}"
        )
        for note in dataset.get("quality_notes") or []:
            st.warning(str(note))
        st.caption("This is a starting hypothesis. Validation scores guide the search; the held-out split is checked at the end.")

    edits = {
        "hidden_layers": widths,
        "learning_rate": float(st.session_state[f"{prefix}_learning_rate"]),
        "epochs": int(st.session_state[f"{prefix}_epochs"]),
        "batch_size": int(st.session_state[f"{prefix}_batch_size"]),
        "early_stop_patience": int(st.session_state[f"{prefix}_early_stop_patience"]),
        "activation": str(st.session_state[f"{prefix}_activation"]),
        "dropout_rate": float(st.session_state[f"{prefix}_dropout_rate"]),
        "weight_decay": float(st.session_state[f"{prefix}_weight_decay"]),
    }
    proposed_edits = {key: proposed[key] for key in edits if key != "hidden_layers"}
    proposed_edits["hidden_layers"] = proposed["widths"]
    changed = [key for key in edits if edits[key] != proposed_edits[key]]
    st.caption(
        f"Preview updated · {len(changed)} setting{'s' if len(changed) != 1 else ''} changed"
        if changed else "Showing the Initializer proposal"
    )
    reset_col, approve_col = st.columns([1, 2], gap="small")
    with reset_col:
        st.button(
            "Reset to proposal", icon=":material/restart_alt:", width="stretch",
            key=f"{prefix}_reset", on_click=_reset_widgets,
            args=(prefix, proposed, layer_max),
        )
    with approve_col:
        approved = st.button(
            "Approve setup and start training", type="primary", icon=":material/check_circle:",
            width="stretch", key=f"{prefix}_approve",
        )
    if approved:
        return ("custom", edits) if changed else ("accept", None)
    return None
