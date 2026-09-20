"""
Interactive dataset questionnaire.

Ported from the legacy ``main.py`` preamble. It asks the engineer three things
before the loader touches the file, because each answer changes how the physics
is reconstructed:

  1. a free-text system description (fed to every agent as context)
  2. whether any states are angles that wrap between +pi and -pi
  3. whether the file is one continuous run or several stacked trajectories,
     and if several, whether the split timestamps are known

Answers are written back onto the live ``config`` module, exactly as the
original did, so the loader and agents pick them up.

Headless callers (API / Streamlit / CI) simply skip this module: the config
defaults already describe a single continuous trajectory with no angles.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from backend_core.AgentSysID import config as cfg

_YES = ("yes", "y")
_NO = ("no", "n")
_AUTO = ("auto", "a", "idk")


def _ask(prompt: str, reader: Callable[[str], str]) -> str:
    try:
        return reader(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def run_questionnaire(reader: Optional[Callable[[str], str]] = None) -> dict[str, Any]:
    """
    Run the interactive questionnaire and apply the answers to ``config``.

    Returns a dict summarising what was set (useful for logging and tests).
    ``reader`` defaults to :func:`input` and is injectable for testing.
    """
    reader = reader or input

    print("\n" + "=" * 80)
    print("🛠️  INTERACTIVE DATASET QUESTIONNAIRE")
    print("=" * 80)

    # --- 1. Customer system description ---------------------------------
    print("📝 Enter a short description of your system, data, or physical properties.")
    print("   (e.g., 'This dataset represents a high-speed autonomous sports car...')")
    cust_desc = _ask("   👉 Description (Press Enter to leave blank): ", reader)
    cfg.CUSTOMER_SYSTEM_DESCRIPTION = cust_desc if cust_desc else ""
    print("   ✅ System context saved.")
    print("-" * 80)

    # --- 2. Angular states ------------------------------------------------
    while True:
        print(
            "❓ Question 1: Does your dataset contain any angular states "
            "(e.g., yaw, pitch) that wrap between π and -π?"
        )
        print("   -> Type 'yes' to specify them manually, 'no' if none, or 'auto' for auto-scan.")
        has_angles = _ask("   👉 Your choice (yes / no / auto): ", reader).lower()
        if has_angles in _YES + _NO + _AUTO:
            break
        print("   ⚠️ Invalid input. Please answer 'yes', 'no', or 'auto'.\n")

    if has_angles in _YES:
        while True:
            angle_str = _ask(
                "   👉 Enter state indices (0-based) separated by commas (e.g., 2, 4): ", reader
            )
            try:
                cfg.ANGLE_INDICES = [int(x.strip()) for x in angle_str.split(",") if x.strip()]
                cfg.AUTO_DETECT_ANGLES = False
                print(f"   ✅ Angular indices manually locked to: {cfg.ANGLE_INDICES}")
                break
            except ValueError:
                print("   ⚠️ Invalid format. Please enter numbers separated by commas (e.g., 0, 2).")
    elif has_angles in _NO:
        cfg.ANGLE_INDICES = []
        cfg.AUTO_DETECT_ANGLES = False
        print("   ✅ Configured for NO angular states.")
    else:
        cfg.AUTO_DETECT_ANGLES = True
        cfg.ANGLE_INDICES = []
        print("   🤖 'Auto' selected: Code will analyze dataset and find angles.")
    print("-" * 80)

    # --- 3. Trajectory structure -----------------------------------------
    while True:
        print("❓ Question 2: Is your dataset made of a SINGLE continuous trajectory?")
        is_single = _ask("   👉 Your choice (yes / no): ", reader).lower()
        if is_single in _YES + _NO:
            break
        print("   ⚠️ Invalid input. Please answer 'yes' or 'no'.\n")

    if is_single in _YES:
        cfg.MULTI_TRAJECTORY = False
        cfg.MANUAL_TRAJECTORY_SPLIT_TIMES = []
        print("   ✅ Dataset configured as a single continuous trajectory.")
    else:
        cfg.MULTI_TRAJECTORY = True
        while True:
            print("   ❓ Do you know exact timestamps where new trajectories begin? (yes / auto): ")
            knows_splits = _ask("      👉 Your choice: ", reader).lower()
            if knows_splits in _YES + _NO + _AUTO:
                break

        if knows_splits in _YES:
            while True:
                times_str = _ask(
                    "      👉 Enter timestamps separated by commas (e.g., 12.5, 25.0): ", reader
                )
                try:
                    cfg.MANUAL_TRAJECTORY_SPLIT_TIMES = sorted(
                        float(x.strip()) for x in times_str.split(",") if x.strip()
                    )
                    print(
                        f"      ✅ Configured {len(cfg.MANUAL_TRAJECTORY_SPLIT_TIMES)} manual split points."
                    )
                    break
                except ValueError:
                    print("      ⚠️ Invalid format. Please enter numerical timestamps.")
        else:
            cfg.MANUAL_TRAJECTORY_SPLIT_TIMES = []
            print("      🤖 Auto-detection active for trajectory boundaries.")

    print("=" * 80 + "\n")

    return {
        "customer_description": cfg.CUSTOMER_SYSTEM_DESCRIPTION,
        "angle_indices": list(cfg.ANGLE_INDICES),
        "auto_detect_angles": cfg.AUTO_DETECT_ANGLES,
        "multi_trajectory": cfg.MULTI_TRAJECTORY,
        "manual_split_times": list(cfg.MANUAL_TRAJECTORY_SPLIT_TIMES),
    }


def review_initializer_config(
    setup_config: dict[str, Any],
    reader: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    """
    Show the Initializer Agent's proposal and let the engineer override any of
    it (the legacy hybrid override prompt). Returns the possibly-edited config.
    """
    reader = reader or input

    print("\n" + "=" * 80)
    print("🤖 INITIALIZER AGENT PROPOSED CONFIGURATION:")
    print("=" * 80)
    print(f"  1. Learning Rate (learning_rate)         : {setup_config.get('learning_rate', 0.001):.6f}")
    print(f"  2. Hidden Layers Topology (hidden_layers): {setup_config.get('hidden_layers', [64])}")
    print(f"  3. Activation Function (activation)      : {str(setup_config.get('activation', 'tanh')).upper()}")
    print(f"  4. Dropout Rate (dropout_rate)           : {setup_config.get('dropout_rate', 0.0)}")
    print(f"  5. L2 Weight Decay (weight_decay)        : {setup_config.get('weight_decay', 0.0001)}")
    print(f"  6. State Space Filter (use_state_filter) : {setup_config.get('use_state_filter', False)}")
    print(f"  7. Derivative Filter Tau (tau)           : {setup_config.get('derivative_filter_tau', 0.005)}")
    print(
        f"  8. LR Search Bounds (Min, Max)           : "
        f"[{setup_config.get('lr_search_min', cfg.LEARNING_RATE_MIN)}, "
        f"{setup_config.get('lr_search_max', cfg.LEARNING_RATE_MAX)}]"
    )
    print(
        f"  9. Hidden Size Bounds (Min, Max)         : "
        f"[{setup_config.get('hidden_size_search_min', cfg.HIDDEN_SIZE_MIN)}, "
        f"{setup_config.get('hidden_size_search_max', cfg.HIDDEN_SIZE_MAX)}]"
    )
    print(
        f" 10. Layer Depth Bounds (Min, Max)         : "
        f"[{setup_config.get('num_layers_search_min', cfg.NUM_LAYERS_MIN)}, "
        f"{setup_config.get('num_layers_search_max', cfg.NUM_LAYERS_MAX)}]"
    )
    print(f" 11. Max Epochs (epochs)                   : {setup_config.get('epochs', 300)}")
    print(f" 12. Batch Size (batch_size)               : {setup_config.get('batch_size', 128)}")
    print(f" 13. Early Stop Patience (patience)        : {setup_config.get('early_stop_patience', 25)}")
    print("=" * 80)

    while True:
        want_override = _ask(
            "❓ Do you want to manually override any of these parameters? (yes/no): ", reader
        ).lower()
        if want_override in _YES + _NO:
            break
        print("   ⚠️ Please answer 'yes' or 'no'.")

    if want_override in _NO:
        print("   ✅ All AI-recommended parameters accepted without changes.")
        return setup_config

    print("\n   👉 Enter your manual overrides below.")
    print("      💡 TIPS & EXAMPLES:")
    print("         - For numbers, just type the decimal (e.g., 0.001)")
    print("         - For bounds or layers, type numbers separated by commas (e.g., 32, 256)")
    print("      *(Press Enter without typing anything to keep the AI's choice)*\n")

    def _float_override(key: str, label: str) -> None:
        raw = _ask(f"      - Override {label} [{setup_config.get(key)}]: \n        > ", reader)
        if raw:
            try:
                setup_config[key] = float(raw)
            except ValueError:
                pass

    _float_override("learning_rate", "learning_rate")

    hl_input = _ask(
        f"      - Override hidden_layers [{setup_config.get('hidden_layers')}]: \n        > ", reader
    )
    if hl_input:
        try:
            setup_config["hidden_layers"] = [int(x.strip()) for x in hl_input.split(",") if x.strip()]
        except ValueError:
            pass

    act_input = _ask(
        f"      - Override activation [{setup_config.get('activation')}]: \n        > ", reader
    )
    if act_input and act_input.lower() in cfg.AVAILABLE_ACTIVATIONS:
        setup_config["activation"] = act_input.lower()

    _float_override("dropout_rate", "dropout_rate")
    _float_override("weight_decay", "weight_decay")

    filter_input = _ask(
        f"      - Override use_state_filter [{setup_config.get('use_state_filter')}]: \n        > ", reader
    )
    if filter_input:
        if filter_input.lower() in ("true", "t") + _YES:
            setup_config["use_state_filter"] = True
        elif filter_input.lower() in ("false", "f") + _NO:
            setup_config["use_state_filter"] = False

    if setup_config.get("use_state_filter", False):
        perc_input = _ask(
            f"      - Override auto_filter_percentiles "
            f"{setup_config.get('auto_filter_percentiles', [2, 98])}: \n        > ",
            reader,
        )
        if perc_input:
            try:
                setup_config["auto_filter_percentiles"] = [
                    float(x.strip()) for x in perc_input.split(",") if x.strip()
                ]
            except ValueError:
                pass

    _float_override("derivative_filter_tau", "derivative_filter_tau")

    def _bounds_override(min_key: str, max_key: str, label: str, cast=float) -> None:
        raw = _ask(
            f"      - Override {label} [{setup_config.get(min_key)}, {setup_config.get(max_key)}]: \n        > ",
            reader,
        )
        if raw:
            try:
                vals = [cast(x.strip()) for x in raw.split(",")]
                setup_config[min_key], setup_config[max_key] = min(vals), max(vals)
            except ValueError:
                pass

    _bounds_override("lr_search_min", "lr_search_max", "LR Search Bounds (e.g., 0.00005, 0.01)")
    _bounds_override(
        "hidden_size_search_min", "hidden_size_search_max",
        "Hidden Size Bounds (e.g., 32, 256)", int,
    )
    _bounds_override(
        "num_layers_search_min", "num_layers_search_max",
        "Layer Depth Bounds (e.g., 1, 4)", int,
    )

    print("\n   ✅ Manual overrides applied successfully!")
    return setup_config
