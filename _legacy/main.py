"""
Actor-Critic Hyperparameter Tuning for State-Constrained System Identification
- Dynamically adapts to Excel-based trajectory data
- Initializer Agent / Manual Override: Choose to auto-initialize or manually preset weights
- Automatically selects activation function (LLM + rule-based overrides or manual config)
- Tunes learning rate and hidden layer architecture (size, depth)
- Validates and Trains focusing purely on State Derivative Prediction (X_dot)
- Dual-compatibility for LLM backends: Easily toggle between Groq and OpenRouter
- Saves all output plots explicitly to disk in the current working directory
"""

import signal
import os
import time
import threading       # <--- Add this here!
from tqdm import tqdm  # <--- Add this here!
from config import *
from config import LLM_MODEL  # <--- Add it up here in the global scope!
from framework import *
from model import DynamicsModel
from config import OVERFIT_RATIO_LIMIT
from pdf_generator import generate_final_pdf
from dotenv import load_dotenv
# ============================================================================
#  MAIN LOOP
# ============================================================================
def main():

    # --- 1. RECORD GLOBAL START TIME ---
    global_start_time = time.time()

    # --- NEW: PINN TEMPLATE GENERATOR ---
    if USE_PINN:
        import os
        if not os.path.exists(PINN_EQUATION_FILE):
            print(f"\n⚙️ PINN Mode is ENABLED, but '{PINN_EQUATION_FILE}' was not found.")
            print(f"   Generating a fresh PyTorch physics template...")

            pinn_template = f'''"""
    PHYSICS-INFORMED EQUATION DEFINITIONS
    Define your known analytical derivatives (X_dot) here.
    Use strictly PyTorch math operations (torch.sin, torch.cos, etc.) so the 
    network can maintain its gradient graph.
    """
    import torch

    def compute_analytical_xdot(states, actions):
        """
        states: Tensor of shape [batch_size, state_dim]
        actions: Tensor of shape [batch_size, action_dim]

        Returns:
        physics_xdot: Tensor of shape [batch_size, state_dim]
        """
        batch_size = states.shape[0]
        state_dim = states.shape[1]

        # Initialize the analytical derivative tensor with zeros
        physics_xdot = torch.zeros((batch_size, state_dim), device=states.device, dtype=torch.float32)

        # =========================================================================
        # EXTRACT STATES & ACTIONS (Example for a standard kinematic vehicle model)
        # Edit these indices to match your Excel columns exactly!
        # =========================================================================
        # x   = states[:, 0]
        # y   = states[:, 1]
        # yaw = states[:, 2]
        # v   = states[:, 3]
        # 
        # throttle = actions[:, 0]
        # steering = actions[:, 1]
        #
        # L = 2.5 # Wheelbase in meters
        #
        # =========================================================================
        # DEFINE KNOWN PHYSICS (X_dot)
        # =========================================================================
        # physics_xdot[:, 0] = v * torch.cos(yaw)                       # x_dot
        # physics_xdot[:, 1] = v * torch.sin(yaw)                       # y_dot
        # physics_xdot[:, 2] = (v / L) * torch.tan(steering)            # yaw_dot
        # physics_xdot[:, 3] = throttle                                 # v_dot (Assuming direct acceleration mapping)

        return physics_xdot
    '''


            with open(PINN_EQUATION_FILE, "w", encoding="utf-8") as f:
                f.write(pinn_template)

            print(f"   ✅ Template '{PINN_EQUATION_FILE}' generated successfully.")
            print(
                f"   🛑 Execution paused. Please open the file, define your physical equations, and run this script again.")
            return  # Safely exit so the engineer can write the math

    # =========================================================================
    # --- 2. CONFIGURE RUN LIMITS ACCORDING TO RUN_MODE ---
    # =========================================================================
    mode_str = RUN_MODE.lower()
    if mode_str == "heavy":
        MAX_CYCLES = 40
        max_hours = 4.0
        critic_explore_limit = 15
        memory_capacity = "Unlimited (All Cycles)"
        context_status = "Always Active"
        reasoning_profile = "1 Single Paragraph"
    elif mode_str == "fast":
        MAX_CYCLES = 7
        max_hours = 0.5
        critic_explore_limit = 3
        memory_capacity = "Last 5 Failures"
        context_status = "Disabled"
        reasoning_profile = "2 Concise Sentences"
    else:  # regular
        mode_str = "regular"
        MAX_CYCLES = 20
        max_hours = 1.5
        critic_explore_limit = 8
        memory_capacity = "Last 10 Failures"
        context_status = "Cycles 1–5 Only"
        reasoning_profile = "1 Single Paragraph"

    max_seconds = max_hours * 3600.0

    print("=" * 80)
    print(f"🚀 INITIALIZING FRAMEWORK IN [{mode_str.upper()}] MODE")
    print(f"   ├── Cycle Limit            : {MAX_CYCLES} Cycles")
    print(f"   ├── Time Limit             : {max_hours} Hours ({int(max_hours * 60)} mins)")
    print(
        f"   ├── Critic Search Profile  : Explore (Cycles 1–{critic_explore_limit}) | Fine-Tune (Cycles {critic_explore_limit + 1}+)")
    print(f"   ├── Failure Memory Depth   : {memory_capacity}")
    print(f"   ├── System Context Feed    : {context_status}")
    print(f"   └── Initializer Reasoning  : {reasoning_profile}")
    print("=" * 80)

    reset_stop_flag()
    signal.signal(signal.SIGINT, request_stop)
    print("ℹ️  Click on stop icon at any time to stop training early and get results from the best checkpoint so far.")

    # =========================================================================
    # --- 3. INTERACTIVE DATASET QUESTIONNAIRE ---
    # =========================================================================
    import config
    print("\n" + "=" * 80)
    print("🛠️  INTERACTIVE DATASET QUESTIONNAIRE")
    print("=" * 80)

    # 1. --- CUSTOMER SYSTEM DESCRIPTION ---
    print("📝 Enter a short description of your system, data, or physical properties.")
    print("   (e.g., 'This dataset represents a high-speed autonomous sports car...')")
    cust_desc = input("   👉 Description (Press Enter to leave blank): ").strip()
    config.CUSTOMER_SYSTEM_DESCRIPTION = cust_desc if cust_desc else ""
    print(f"   ✅ System context saved.")
    print("-" * 80)

    # 2. --- QUESTION 1: ANGULAR STATES ---
    while True:
        print(
            "❓ Question 1: Does your dataset contain any angular states (e.g., yaw, pitch) that wrap between π and -π?")
        print("   -> Type 'yes' to specify them manually, 'no' if none, or 'auto' for auto-scan.")
        has_angles = input("   👉 Your choice (yes / no / auto): ").strip().lower()
        if has_angles in ['yes', 'y', 'no', 'n', 'auto', 'a', 'idk']:
            break
        print("   ⚠️ Invalid input. Please answer 'yes', 'no', or 'auto'.\n")

    if has_angles in ['yes', 'y']:
        while True:
            angle_str = input("   👉 Enter state indices (0-based) separated by commas (e.g., 2, 4): ").strip()
            try:
                config.ANGLE_INDICES = [int(x.strip()) for x in angle_str.split(',') if x.strip()]
                config.AUTO_DETECT_ANGLES = False
                print(f"   ✅ Angular indices manually locked to: {config.ANGLE_INDICES}")
                break
            except ValueError:
                print("   ⚠️ Invalid format. Please enter numbers separated by commas (e.g., 0, 2).")
    elif has_angles in ['no', 'n']:
        config.ANGLE_INDICES = []
        config.AUTO_DETECT_ANGLES = False
        print("   ✅ Configured for NO angular states.")
    else:
        config.AUTO_DETECT_ANGLES = True
        config.ANGLE_INDICES = []
        print("   🤖 'Auto' selected: Code will analyze dataset and find angles.")
    print("-" * 80)

    # 3. --- QUESTION 2: TRAJECTORY STRUCTURE ---
    while True:
        print("❓ Question 2: Is your dataset made of a SINGLE continuous trajectory?")
        is_single = input("   👉 Your choice (yes / no): ").strip().lower()
        if is_single in ['yes', 'y', 'no', 'n']:
            break
        print("   ⚠️ Invalid input. Please answer 'yes' or 'no'.\n")

    if is_single in ['yes', 'y']:
        config.MULTI_TRAJECTORY = False
        config.MANUAL_TRAJECTORY_SPLIT_TIMES = []
        print("   ✅ Dataset configured as a single continuous trajectory.")
    else:
        config.MULTI_TRAJECTORY = True
        while True:
            print("   ❓ Do you know exact timestamps where new trajectories begin? (yes / auto): ")
            knows_splits = input("      👉 Your choice: ").strip().lower()
            if knows_splits in ['yes', 'y', 'no', 'n', 'auto', 'a', 'idk']:
                break

        if knows_splits in ['yes', 'y']:
            while True:
                times_str = input("      👉 Enter timestamps separated by commas (e.g., 12.5, 25.0): ").strip()
                try:
                    config.MANUAL_TRAJECTORY_SPLIT_TIMES = sorted(
                        [float(x.strip()) for x in times_str.split(',') if x.strip()])
                    print(f"      ✅ Configured {len(config.MANUAL_TRAJECTORY_SPLIT_TIMES)} manual split points.")
                    break
                except ValueError:
                    print("      ⚠️ Invalid format. Please enter numerical timestamps.")
        else:
            config.MANUAL_TRAJECTORY_SPLIT_TIMES = []
            print("      🤖 Auto-detection active for trajectory boundaries.")
    print("=" * 80 + "\n")

    # =========================================================================
    # 4. LOAD DATASET & RUN DATA INSPECTOR
    # =========================================================================
    # Now that Q1 and Q2 are answered, the Loader knows exactly how to handle the data!
    loader = ExcelDataLoader(EXCEL_FILE_PATH)

    from framework import run_data_inspector_agent
    engineer_notes, cols_to_drop = run_data_inspector_agent(loader)

    # Actually delete the bad columns!
    if cols_to_drop:
        loader.drop_columns(cols_to_drop)

    # Append the engineering context
    if engineer_notes and engineer_notes.lower() != "skip":
        if hasattr(config, 'CUSTOMER_SYSTEM_DESCRIPTION') and config.CUSTOMER_SYSTEM_DESCRIPTION:
            config.CUSTOMER_SYSTEM_DESCRIPTION += f" | Engineer Clarification on Data: {engineer_notes}"
        else:
            config.CUSTOMER_SYSTEM_DESCRIPTION = f"Engineer Clarification on Data: {engineer_notes}"
        print(f"   🧠 Added engineer context to system profile.")

    # 5. NOW EXTRACT TRAJECTORIES (Safely separated & filtered!)
    trajectories = loader.get_trajectories()
    state_dim = loader.state_dim
    action_encoding_dim = loader.action_dim

    # --- UPGRADED: HYBRID DATA SPLIT ---
    print("\n🔀 Splitting data (Hybrid Mode: Shuffled Train/Val, Chronological Test)...")

    # --- NEW: Check if LSTM is active to safely disable shuffling ---
    lstm_active = getattr(config, 'NETWORK_ARCHITECTURE', 'MLP').upper() == "LSTM"
    shuffle_mode = False if lstm_active else True
    if lstm_active:
        print("    ⚠️  LSTM ACTIVE: Data shuffling is DISABLED to preserve contiguous temporal memory.")

    if len(trajectories) > 2:
        # 1. Hold out exactly the last 2 trajectories for pure verification (Test set)
        test_trajs = trajectories[-2:]
        remaining_trajs = trajectories[:-2]

        # 2. Split the rest: ~80% Training / ~20% Validation
        # SHUFFLE is dynamically controlled by the LSTM setting
        from sklearn.model_selection import train_test_split
        train_trajs, val_trajs = train_test_split(remaining_trajs, test_size=0.20, random_state=42,
                                                  shuffle=shuffle_mode)

    elif len(trajectories) == 1:
        print("    ⚠️ Only 1 trajectory detected. Reading Master Configuration...")

        # --- 1. PULL CONFIGURATION TOGGLES ---
        try:
            from config import TRAJECTORY_CHUNK_SIZE, SHUFFLE_DATA
            chunk_size = TRAJECTORY_CHUNK_SIZE
            shuffle_mode = SHUFFLE_DATA

        except ImportError:
            chunk_size = 200  # Fallback if config is missing
            shuffle_mode = True  # Fallback if config is missing
        all_steps = trajectories[0]
        total_len = len(all_steps)

        # --- 2. HOLD OUT THE TEST SET (Always chronological for RK4) ---
        # We always lock the final 10% away so compare_nn_vs_true.py has a smooth timeline
        test_split_idx = int(total_len * 0.90)
        temp_train_val = all_steps[:test_split_idx]
        test_trajs = [all_steps[test_split_idx:]]

        # --- 3. DYNAMIC CHUNKING & SHUFFLING LOGIC ---
        if chunk_size == 0:
            if shuffle_mode:
                print("       -> [MODE ACTIVE]: ZERO CHUNKING, BUT ROW-LEVEL SHUFFLING IS ON!")
                print("       -> [NOTE]: This mode is heavily optimized for MLP architectures.")

                from sklearn.model_selection import train_test_split
                # Shuffle the individual rows (perfect for MLP)
                train_trajs_flat, val_trajs_flat = train_test_split(
                    temp_train_val,
                    test_size=0.15,
                    random_state=42,
                    shuffle=True
                )
                # Re-package them back into lists so the rest of the code understands them
                train_trajs = [train_trajs_flat]
                val_trajs = [val_trajs_flat]

            else:
                print("       -> [MODE ACTIVE]: ZERO CHUNKING (Pure Chronological).")
                print("       -> [WARNING]: SHUFFLE=False. VRAM usage may be very high for LSTMs.")

                # Slice the remaining 90% into Train and Val chronologically
                val_split_idx = int(len(temp_train_val) * 0.88)
                train_trajs = [temp_train_val[:val_split_idx]]
                val_trajs = [temp_train_val[val_split_idx:]]

        else:
            print(f"       -> [MODE ACTIVE]: CHUNKING (Size={chunk_size}).")
            print(f"       -> [MODE ACTIVE]: SHUFFLING = {shuffle_mode}.")

            # Slice into mini-trajectories
            sub_trajectories = []
            for i in range(0, len(temp_train_val), chunk_size):
                chunk = temp_train_val[i: i + chunk_size]
                if len(chunk) >= 25:  # Safe minimum to prevent rollout crashes
                    sub_trajectories.append(chunk)

            # Apply Train/Val split with the exact shuffle toggle requested
            from sklearn.model_selection import train_test_split
            train_trajs, val_trajs = train_test_split(
                sub_trajectories,
                test_size=0.15,
                random_state=42,
                shuffle=shuffle_mode
            )

    else:
        raise ValueError("Not enough trajectories to split! Please generate more data.")

    print(f"    -> Training batches  : {sum(len(t) for t in train_trajs)} rows")
    print(f"    -> Validation batches: {sum(len(t) for t in val_trajs)} rows")

    if len(trajectories) > 2:
        print(f"    -> Testing batches   : {sum(len(t) for t in test_trajs)} rows (2 Trajectories Held Out)")
    else:
        print(f"    -> Testing batches   : {sum(len(t) for t in test_trajs)} rows (Final 10% Held Out)")

    with open(LOG_FILENAME, "w", encoding="utf-8") as log_file:
        log_file.write(f"# ACTOR-CRITIC TUNING PROMPT HISTORY LOG\n")

    print("=" * 80)
    dt_mode_desc = f"Exact dt calculated row-by-row (from column '{TIME_COLUMN}')"

    try:
        from config import NETWORK_ARCHITECTURE, ROLLOUT_HORIZON, LSTM_SEQ_LENGTH, INTEGRATOR_TYPE
        base_arch = NETWORK_ARCHITECTURE.strip().upper()
        horizon = ROLLOUT_HORIZON
        seq_len = LSTM_SEQ_LENGTH
        int_type = INTEGRATOR_TYPE.strip().upper()
    except ImportError:
        base_arch = "MLP"
        horizon = 1
        seq_len = 10
        int_type = "EULER"

    if base_arch == "LSTM":
        arch_display = f"LSTM (Temporal Memory Window: {seq_len} steps)"
    else:
        arch_display = "MLP (Memoryless Instantaneous State)"

    if USE_PINN:
        architecture_mode = f"PINN ({arch_display} + Physics) | Weight: {PINN_LOSS_WEIGHT}"
    else:
        architecture_mode = f"Pure {arch_display} (Data-Driven System ID)"

    if horizon > 1:
        strategy_desc = f"Autoregressive Multi-Step Rollout (Horizon = {horizon} steps)"
        target_desc = "Predicting X_dot & Accumulating Trajectory Drift Penalty"
    else:
        strategy_desc = "Single-Step Supervised Loss"
        target_desc = "Predicting Instantaneous State Derivatives (X_dot)"

    integrator_desc = "4th-Order Runge-Kutta (RK4)" if int_type == "RK4" else "1st-Order Euler"

    print(f"🤖 SYSTEM ID OF DATA SOURCE: '{EXCEL_FILE_PATH}'")
    print(f"⏱️  Time-Step Mode: {dt_mode_desc}")
    print(f"🏗️  Architecture Mode: {architecture_mode}")
    print(f"⚙️  Kinematic Integrator: {integrator_desc}")
    print(f"🧠 Backend Core Driver: {API_PROVIDER.upper()} | Model Assigned: {LLM_MODEL}")
    print(f"📊 Loss Profile: {strategy_desc}")
    print(f"🎯 Target Profile: {target_desc}")
    print("=" * 80)

    # Setup Hyperparameters & Activation Functions (Agent vs. Manual Switch)
    if CHOOSE_VIA_LLM_INITIALIZER:
        print(f"\n🧠 Querying Initializer Agent to establish starting hyperparameters + activation function...")
        initializer = InitializerAgent(loader)

        # ====================================================================
        # ANIMATED LOADING BAR FOR LLM API CALL
        # ====================================================================
        agent_result = [None]
        api_status = [False]

        def fetch_llm():
            agent_result[0] = initializer.determine_initial_setup()
            api_status[0] = True

        llm_thread = threading.Thread(target=fetch_llm)
        llm_thread.start()

        with tqdm(total=100, desc="    ⏳ Awaiting AI response",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}") as pbar:
            current_val = 0.0
            while not api_status[0]:
                step = (99.0 - current_val) * 0.15
                current_val += step
                pbar.update(step)
                time.sleep(0.5)
            pbar.update(100.0 - pbar.n)

        llm_thread.join()
        setup_config = agent_result[0]

        # ====================================================================
        # --- INTERACTIVE HYBRID OVERRIDE PROMPT ---
        # ====================================================================
        import config
        print("\n" + "=" * 80)
        print("🤖 INITIALIZER AGENT PROPOSED CONFIGURATION:")
        print("=" * 80)
        print(f"  1. Learning Rate (learning_rate)       : {setup_config.get('learning_rate', 0.001):.6f}")
        print(f"  2. Hidden Layers Topology (hidden_layers): {setup_config.get('hidden_layers', [64])}")
        print(f"  3. Activation Function (activation)    : {setup_config.get('activation', 'tanh').upper()}")
        print(f"  4. Dropout Rate (dropout_rate)         : {setup_config.get('dropout_rate', 0.0)}")
        print(f"  5. L2 Weight Decay (weight_decay)      : {setup_config.get('weight_decay', 0.0001)}")
        print(f"  6. State Space Filter (use_state_filter): {setup_config.get('use_state_filter', False)}")
        print(f"  7. Derivative Filter Tau (tau)         : {setup_config.get('derivative_filter_tau', 0.005)}")
        print(
            f"  8. LR Search Bounds (Min, Max)         : [{setup_config.get('lr_search_min', config.LEARNING_RATE_MIN)}, {setup_config.get('lr_search_max', config.LEARNING_RATE_MAX)}]")
        print(
            f"  9. Hidden Size Bounds (Min, Max)       : [{setup_config.get('hidden_size_search_min', config.HIDDEN_SIZE_MIN)}, {setup_config.get('hidden_size_search_max', config.HIDDEN_SIZE_MAX)}]")
        print(
            f" 10. Layer Depth Bounds (Min, Max)       : [{setup_config.get('num_layers_search_min', config.NUM_LAYERS_MIN)}, {setup_config.get('num_layers_search_max', config.NUM_LAYERS_MAX)}]")
        print(f"  11. Max Epochs (epochs)                 : {setup_config.get('epochs', 300)}")
        print(f"  12. Batch Size (batch_size)             : {setup_config.get('batch_size', 128)}")
        print(f"  13. Early Stop Patience (patience)     : {setup_config.get('early_stop_patience', 25)}")
        print("=" * 80)

        while True:
            want_override = input(
                "❓ Do you want to manually override any of these parameters? (yes/no): ").strip().lower()
            if want_override in ['yes', 'y', 'no', 'n']:
                break
            print("   ⚠️ Please answer 'yes' or 'no'.")

        if want_override in ['yes', 'y']:
            print("\n   👉 Enter your manual overrides below.")
            print("      💡 TIPS & EXAMPLES:")
            print("         - For numbers, just type the decimal (e.g., 0.001)")
            print("         - For bounds or layers, type numbers separated by commas (e.g., 32, 256)")
            print("      *(Press Enter without typing anything to keep the AI's choice for that parameter)*\n")

            # 1. Learning Rate
            lr_input = input(
                f"      - Override learning_rate [{setup_config.get('learning_rate')}]: \n        > ").strip()
            if lr_input:
                try:
                    setup_config['learning_rate'] = float(lr_input)
                except ValueError:
                    pass

            # 2. Hidden Layers
            hl_input = input(
                f"      - Override hidden_layers [{setup_config.get('hidden_layers')}]: \n        > ").strip()
            if hl_input:
                try:
                    setup_config['hidden_layers'] = [int(x.strip()) for x in hl_input.split(',') if x.strip()]
                except ValueError:
                    pass

            # 3. Activation
            act_input = input(f"      - Override activation [{setup_config.get('activation')}]: \n        > ").strip()
            if act_input:
                act_val = act_input.lower()
                if act_val in config.AVAILABLE_ACTIVATIONS:
                    setup_config['activation'] = act_val

            # 4. Dropout
            drop_input = input(
                f"      - Override dropout_rate [{setup_config.get('dropout_rate')}]: \n        > ").strip()
            if drop_input:
                try:
                    setup_config['dropout_rate'] = float(drop_input)
                except ValueError:
                    pass

            # 5. Weight Decay
            wd_input = input(
                f"      - Override weight_decay [{setup_config.get('weight_decay')}]: \n        > ").strip()
            if wd_input:
                try:
                    setup_config['weight_decay'] = float(wd_input)
                except ValueError:
                    pass

            # 6. State Filter
            filter_input = input(
                f"      - Override use_state_filter [{setup_config.get('use_state_filter')}]: \n        > ").strip()
            if filter_input:
                if filter_input.lower() in ['true', 't', 'yes', 'y']:
                    setup_config['use_state_filter'] = True
                elif filter_input.lower() in ['false', 'f', 'no', 'n']:
                    setup_config['use_state_filter'] = False

            if setup_config.get('use_state_filter', False):
                perc_input = input(
                    f"      - Override auto_filter_percentiles {setup_config.get('auto_filter_percentiles', [2, 98])}: \n        > ").strip()
                if perc_input:
                    try:
                        setup_config['auto_filter_percentiles'] = [float(x.strip()) for x in perc_input.split(',') if
                                                                   x.strip()]
                    except ValueError:
                        pass

            # 7. Filter Tau
            tau_input = input(
                f"      - Override derivative_filter_tau [{setup_config.get('derivative_filter_tau')}]: \n        > ").strip()
            if tau_input:
                try:
                    setup_config['derivative_filter_tau'] = float(tau_input)
                except ValueError:
                    pass

            # 8. LR Bounds
            lr_bounds = input(
                f"      - Override LR Search Bounds [{setup_config.get('lr_search_min')}, {setup_config.get('lr_search_max')}] (e.g., 0.00005, 0.01): \n        > ").strip()
            if lr_bounds:
                try:
                    vals = [float(x.strip()) for x in lr_bounds.split(',')]
                    setup_config['lr_search_min'], setup_config['lr_search_max'] = min(vals), max(vals)
                except ValueError:
                    pass

            # 9. Hidden Size Bounds
            hs_bounds = input(
                f"      - Override Hidden Size Bounds [{setup_config.get('hidden_size_search_min')}, {setup_config.get('hidden_size_search_max')}] (e.g., 32, 256): \n        > ").strip()
            if hs_bounds:
                try:
                    vals = [int(x.strip()) for x in hs_bounds.split(',')]
                    setup_config['hidden_size_search_min'], setup_config['hidden_size_search_max'] = min(vals), max(
                        vals)
                except ValueError:
                    pass

            # 10. Layer Bounds
            ly_bounds = input(
                f"      - Override Layer Depth Bounds [{setup_config.get('num_layers_search_min')}, {setup_config.get('num_layers_search_max')}] (e.g., 1, 4): \n        > ").strip()
            if ly_bounds:
                try:
                    vals = [int(x.strip()) for x in ly_bounds.split(',')]
                    setup_config['num_layers_search_min'], setup_config['num_layers_search_max'] = min(vals), max(vals)
                except ValueError:
                    pass

            print("\n   ✅ Manual overrides applied successfully!")
        else:
            print("   ✅ All AI-recommended parameters accepted without changes.")


        # ====================================================================
        # --- SYNCHRONIZE LOCAL VARIABLES FOR MAIN.PY ---
        # ====================================================================
        chosen_activation = setup_config.get('activation', 'relu')

        # 🧠 DYNAMIC FIX: Pass the boundaries directly to the Actor Agent
        starting_config = {
            "learning_rate": setup_config.get('learning_rate', 0.001),
            "hidden_layers": setup_config.get('hidden_layers', [64]),
            "activation": chosen_activation,
            "dropout_rate": setup_config.get('dropout_rate', 0.0),
            "weight_decay": setup_config.get('weight_decay', 0.0001),
            "lr_search_min": setup_config.get('lr_search_min', config.LEARNING_RATE_MIN),
            "lr_search_max": setup_config.get('lr_search_max', config.LEARNING_RATE_MAX),
            "hidden_size_search_min": setup_config.get('hidden_size_search_min', config.HIDDEN_SIZE_MIN),
            "hidden_size_search_max": setup_config.get('hidden_size_search_max', config.HIDDEN_SIZE_MAX),
            "num_layers_search_min": setup_config.get('num_layers_search_min', config.NUM_LAYERS_MIN),
            "num_layers_search_max": setup_config.get('num_layers_search_max', config.NUM_LAYERS_MAX)
        }

        print("=" * 80 + "\n")

        # ====================================================================
        # DYNAMIC GLOBAL CONFIG OVERRIDE (For the Critic Agent)
        # ====================================================================
        config.DROPOUT_RATE = setup_config.get('dropout_rate', config.DROPOUT_RATE)
        config.WEIGHT_DECAY = setup_config.get('weight_decay', config.WEIGHT_DECAY)
        config.LR_REDUCE_FACTOR = setup_config.get('lr_reduce_factor', config.LR_REDUCE_FACTOR)
        config.USE_STATE_FILTER = setup_config.get('use_state_filter', config.USE_STATE_FILTER)
        config.AUTO_FILTER_PERCENTILES = tuple(
            setup_config.get('auto_filter_percentiles', config.AUTO_FILTER_PERCENTILES))
        config.DERIVATIVE_FILTER_TAU = setup_config.get('derivative_filter_tau', config.DERIVATIVE_FILTER_TAU)

        # 🧠 DYNAMIC FIX: Overwrite the global boundaries permanently
        config.LEARNING_RATE_MIN = starting_config["lr_search_min"]
        config.LEARNING_RATE_MAX = starting_config["lr_search_max"]
        config.HIDDEN_SIZE_MIN = starting_config["hidden_size_search_min"]
        config.HIDDEN_SIZE_MAX = starting_config["hidden_size_search_max"]
        config.NUM_LAYERS_MIN = starting_config["num_layers_search_min"]
        config.NUM_LAYERS_MAX = starting_config["num_layers_search_max"]

        # 1. Force the Loader & Global Config to use the Agent's Reset Threshold
        if 'reset_threshold' in setup_config:
            config.RESET_THRESHOLD = setup_config['reset_threshold']
            loader.reset_threshold = setup_config['reset_threshold']

        # ====================================================================
        # DISPLAY INITIALIZER DASHBOARD
        # ====================================================================
        import textwrap

        # Display customer context if provided
        if hasattr(config, 'CUSTOMER_SYSTEM_DESCRIPTION') and config.CUSTOMER_SYSTEM_DESCRIPTION.strip():
            print(f"\n    📝 CUSTOMER SYSTEM CONTEXT:")
            wrapped_context = textwrap.fill(
                config.CUSTOMER_SYSTEM_DESCRIPTION.strip(),
                width=90,
                initial_indent="        ",
                subsequent_indent="        "
            )
            print(wrapped_context)

        # Multi-paragraph reasoning formatting
        raw_reasoning = setup_config.get('reasoning', 'Heuristic fallback active. No reasoning generated.')
        paragraphs = [p.strip() for p in raw_reasoning.split('\n') if p.strip()]
        formatted_reasoning = "\n\n".join([
            textwrap.fill(p, width=90, initial_indent="        ", subsequent_indent="        ")
            for p in paragraphs
        ])

        print(f"\n    🎯 INITIALIZER AGENT REASONING:\n{formatted_reasoning}")

        # Format Reset Threshold display cleanly (handles list or float)
        thresh_val = config.RESET_THRESHOLD
        if isinstance(thresh_val, (list, tuple)):
            thresh_display = f"[{', '.join([f'{x:.4f}' if isinstance(x, (int, float)) else str(x) for x in thresh_val])}]"
        elif isinstance(thresh_val, (int, float)):
            thresh_display = f"{thresh_val:.4f}"
        else:
            thresh_display = str(thresh_val)

        print(f"\n    🛠️  AGENT-AUTHORIZED SEARCH BOUNDS & STARTING CONFIG:")
        print(f"       ├── Initial Activation    : {chosen_activation.upper()}")
        print(
            f"       ├── Initial Learning Rate : {setup_config.get('learning_rate', 0.001):.6f}  (Search Bounds: [{setup_config.get('lr_search_min', config.LEARNING_RATE_MIN)}, {setup_config.get('lr_search_max', config.LEARNING_RATE_MAX)}])")
        print(f"       ├── Initial Topology      : {setup_config.get('hidden_layers', [64])}")
        print(
            f"       ├── Layer Depth Bounds    : [{setup_config.get('num_layers_search_min', config.NUM_LAYERS_MIN)}, {setup_config.get('num_layers_search_max', config.NUM_LAYERS_MAX)}] layers")
        print(
            f"       ├── Layer Width Bounds    : [{setup_config.get('hidden_size_search_min', config.HIDDEN_SIZE_MIN)}, {setup_config.get('hidden_size_search_max', config.HIDDEN_SIZE_MAX)}] neurons")
        reg_mode = "WORKING (Dynamic)" if getattr(config, 'ADAPTIVE_REGULARIZATION', True) else "CONSTANT (Locked)"
        print(
            f"       ├── Regularization        : Dropout={config.DROPOUT_RATE} | L2 Weight Decay={config.WEIGHT_DECAY}")
        print(f"       ├── Adaptive Reg Status   : [{reg_mode}]")
        print(f"       ├── LR Scheduler Factor   : {config.LR_REDUCE_FACTOR}")
        print(
            f"       ├── State Space Filter    : {'ENABLED' if config.USE_STATE_FILTER else 'DISABLED'} | Percentiles: {list(config.AUTO_FILTER_PERCENTILES)}")
        print(
            f"       └── Kinematic Dynamics    : Reset Threshold={thresh_display} | Simulink Filter Tau={config.DERIVATIVE_FILTER_TAU}")
    else:
        print(f"\n⚙️ Manual presets active. Bypassing Initializer Agent...")
        chosen_activation = MANUAL_ACTIVATION.lower()
        starting_config = {
            "learning_rate": MANUAL_STARTING_LR,
            "hidden_layers": MANUAL_STARTING_HIDDEN_LAYERS.copy()
        }

        # ====================================================================
        # DYNAMIC GLOBAL CONFIG OVERRIDE (Manual values take control)
        # ====================================================================
        import config
        config.DROPOUT_RATE = config.MANUAL_DROPOUT_RATE
        config.WEIGHT_DECAY = config.MANUAL_WEIGHT_DECAY

        # ====================================================================
        # DISPLAY MANUAL DASHBOARD
        # ====================================================================
        print(f"       ├── User Manual Status    : [ACTIVE]")
        print(f"       ├── Preset Activation     : {chosen_activation.upper()}")
        print(f"       ├── Preset Starting LR    : {starting_config['learning_rate']:.6f}")
        print(f"       ├── Preset Topology       : {starting_config['hidden_layers']}")
        print(
            f"       ├── Regularization        : Dropout={config.DROPOUT_RATE} | L2 Weight Decay={config.WEIGHT_DECAY}")
        print(f"       ├── LR Scheduler Factor   : {config.LR_REDUCE_FACTOR}")
        print(
            f"       ├── State Space Filter    : {'ENABLED' if config.USE_STATE_FILTER else 'DISABLED'} | Percentiles: {list(config.AUTO_FILTER_PERCENTILES)}")
        print(
            f"       └── Kinematic Dynamics    : Reset Threshold={config.RESET_THRESHOLD} | Simulink Filter Tau={config.DERIVATIVE_FILTER_TAU}")

    # Pass the mode into the tracker and the critic
    tracker = BestConfigTracker(run_mode=RUN_MODE)
    actor = ActorAgent(chosen_activation, initial_config=starting_config)
    critic = CriticAgent(tracker, run_mode=RUN_MODE)
    explorer = ExplorerAgent(initial_config=starting_config)

    perf_history = []
    best_model = None
    stagnation_count = 0  # <--- NEW: Tracks how long the network has been stuck

    for iteration in range(MAX_CYCLES):

        # --- NEW: STRICT TIME-OUT CHECK ---
        elapsed_seconds = time.time() - global_start_time
        if elapsed_seconds > max_seconds:
            print("\n" + "!" * 80)
            print(f"⏳ TIME LIMIT REACHED: {max_hours} hours elapsed.")
            print("🛑 Gracefully halting tuning loop to compile and deploy the best model found so far...")
            print("!" * 80 + "\n")
            break  # This cleanly exits the loop and sends the script straight to the plotting phase

        cycle_num = iteration + 1
        print(f"\n────────────────────────────────────────────────────────────────────────────────")
        print(f" 🔄  [CYCLE {cycle_num:02d} / {MAX_CYCLES:02d}]  TRAINING INITIALIZED")
        print(f"────────────────────────────────────────────────────────────────────────────────")

        current_lr = actor.current_config['learning_rate']
        current_hl = actor.current_config['hidden_layers']

        print(f"    🛠️  Active Architecture Configurations:")
        print(f"        ├── Learning Rate (η)   : {current_lr:.6f}")
        print(f"        ├── Dropout Rate (p)    : {actor.current_config.get('dropout_rate', MANUAL_DROPOUT_RATE):.3f}")
        print(f"        ├── Weight Decay (L2)   : {actor.current_config.get('weight_decay', MANUAL_WEIGHT_DECAY):.6f}")
        print(f"        ├── Hidden Layers Count : {len(current_hl)} layer(s)")
        print(f"        └── Neurons per Layer   : {current_hl}")

        # Pass both train_trajs and val_trajs into the function
        model_trained, final_train_mse, final_mse, final_rmse, X_val, y_val = train_dynamics_model(
            train_trajs, val_trajs, state_dim, action_encoding_dim,
            hidden_layers=current_hl,
            learning_rate=current_lr,
            epochs=setup_config.get('epochs', config.EPOCHS) if CHOOSE_VIA_LLM_INITIALIZER else config.EPOCHS,

            # ⚠️ CHANGED: Now pulling dynamically from the Actor!
            batch_size=actor.current_config.get('batch_size', config.BATCH_SIZE),
            patience=actor.current_config.get('patience', config.EARLY_STOP_PATIENCE),

            activation=chosen_activation,
            dropout_rate=actor.current_config.get('dropout_rate', config.MANUAL_DROPOUT_RATE),
            weight_decay=actor.current_config.get('weight_decay', config.MANUAL_WEIGHT_DECAY),
            lr_min=config.LR_SCHEDULE_MIN_FLOOR
        )

        # =========================================================
        # CHECK GENERALIZATION GAP (OVERFITTING)
        # =========================================================
        if final_train_mse > 1e-8:
            overfit_ratio = final_mse / final_train_mse
            if overfit_ratio > OVERFIT_RATIO_LIMIT:
                print("\n⚠️ CYCLE REJECTED: Severe Overfitting Detected!")
                print(
                    f"   Validation MSE ({final_mse:.6f}) is {overfit_ratio:.1f}x higher than Training MSE ({final_train_mse:.6f}).")
                print("   The network memorized the dataset instead of learning the physical dynamics.")
                print("   Applying a penalty to this configuration and moving to the next cycle...")

                # Apply a massive artificial penalty so it NEVER gets saved as the best model,
                # and the Critic agent registers this architecture as a complete failure.
                final_mse = 9999.0
                final_rmse = 99.0

        perf_history.append({
            "iteration": iteration,
            "config": actor.current_config.copy(),
            "performance": {"mse": final_mse, "rmse": final_rmse}
        })

        is_new_best = tracker.update(final_mse, actor.current_config, current_rmse=final_rmse)

        # ⚠️ CRITICAL FIX: The logic now correctly tracks and saves the physical PyTorch weights!
        if is_new_best or best_model is None:
            best_model = model_trained

        print(f"\n    📋  [CYCLE {cycle_num:02d} COMPLETED PERFORMANCE REPORT]")
        print(f"        ├── Tested Topology     : Layers={current_hl} | LR={current_lr:.6f}")
        print(f"        ├── Resulting Eval MSE  : {final_mse:.6f}  (RMSE: {final_rmse:.6f})")
        if is_new_best:
            print(f"        ├── ⭐ NEW HISTORICAL BEST TRACKED MODEL OVERTAKEN! ⭐")
        print(f"        └── Current Global Best (MSE)  : {tracker.best_mse:.6f}  (RMSE: {tracker.best_rmse:.6f})")

        if final_mse <= MSE_TARGET:
            print("\n🎯 Target Performance reached! Terminating optimization run sequence.")
            break

        if stop_requested():
            print("\n🛑 Stop requested — skipping further cycles and compiling final results now.")
            break

        print(f"\n    🧠 Querying Critic Agent regarding current hyperparameter performance topology...")

        # 1. Measure the real-world latency
        measured_latency_ms = measure_inference_latency(model_trained, state_dim=state_dim,
                                                        action_dim=action_encoding_dim)
        print(
            f"    ⏱️ Model Inference Latency: {measured_latency_ms:.3f} ms (Customer Limit: {CUSTOMER_MAX_LATENCY_MS} ms)")

        # --- NEW: Append the measured latency to the current cycle's history ---
        perf_history[-1]['performance']['latency'] = measured_latency_ms

        # 2. Pass it to the Critic
        critic_output = critic.evaluate(
            train_mse=final_train_mse,
            val_mse=final_mse,
            current_config=actor.current_config,
            activation=chosen_activation,
            measured_latency=measured_latency_ms,
            max_latency=CUSTOMER_MAX_LATENCY_MS,
            cycle_number=cycle_num,
            val_rmse=final_rmse
        )

        # --- NEW: Save the Critic's reasoning into the memory buffer for the next cycle ---
        tracker.add_reasoning_to_memory(critic_output.get('reasoning', 'No reasoning provided.'))

        print(f"    🎯 CRITIC AGENT RESPONSE COMPILATION:")
        print(f"        ├── Network Diagnosis   : [{critic_output.get('diagnosis', 'UNKNOWN')}]")
        print(f"        ├── Assessment Status   : [{critic_output.get('status')}]")
        print(
            f"        ├── Path Segment Move   : {critic_output.get('lr_dir').upper()} (Delta step adjustment value: {critic_output.get('lr_step')})")
        print(f"        ├── Next Network Shape  : {critic_output.get('hidden_layers')}")
        print(f"        └── Critic Reasoning    : {critic_output.get('reasoning')}")


        # Update stagnation tracking
        if is_new_best:
            stagnation_count = 0  # Reset counter when we hit a breakthrough
        else:
            stagnation_count += 1

        # Agent Routing: Decide whether to Fine-Tune or Explore
        if stagnation_count >= 3:
            print(f"\n    🚨 STAGNATION DETECTED ({stagnation_count} failed cycles). Suspending Actor Agent...")
            print(f"    🌌 Summoning Explorer Agent to force a repulsive architectural shift...")

            new_config, explorer_reasoning = explorer.generate_radical_escape(
                tracker=tracker,
                stuck_config=actor.current_config,
                visited_configs=actor.visited_configs
            )

            print(f"        ├── Target Escape Layers : {new_config['hidden_layers']}")
            print(f"        └── Explorer Reasoning   : {explorer_reasoning}")

            # Manually inject the Explorer's radical topology into the Actor's brain
            actor.current_config = new_config
            actor._add_to_visited(new_config)

            # Reset the counter to give the Actor 3 cycles to locally optimize the new territory
            stagnation_count = 0
        else:
            # Normal Operation: Let the Actor Agent handle precise fine-tuning
            actor.apply_critic_feedback(critic_output, tracker)

    if best_model is not None:
        # Save weights and capture the filename base
        pth_base_name = save_best_model(
            best_model,
            tracker.best_config['hidden_layers'],
            state_dim, action_encoding_dim,
            tracker.best_mse,
            tracker.best_config,
            chosen_activation
        )

        export_standalone_inference_script(
            model=best_model,
            hidden_layers=tracker.best_config['hidden_layers'],
            activation=chosen_activation,
            state_dim=state_dim,
            action_dim=action_encoding_dim,
            pth_filename=pth_base_name,
            env_name=ENV_NAME
        )

        final_latency_ms = measure_inference_latency(best_model, state_dim, action_encoding_dim)

        plot_mse_convergence(perf_history, ENV_NAME)
        plot_nrmse_convergence(perf_history, ENV_NAME)
        plot_hyperparameter_evolution(perf_history)
        plot_layers_neurons_mse_contour(perf_history)
        plot_latency_evolution(perf_history, CUSTOMER_MAX_LATENCY_MS)

        true_traj_data, nn_traj_data = plot_test_dataset_verification(
            best_model, test_trajs, state_dim, action_encoding_dim
        )

        final_score, model_status = calculate_success_score(
            val_mse=tracker.best_mse,
            true_trajectory=true_traj_data,
            nn_trajectory=nn_traj_data,
            complexity_label=getattr(loader, 'complexity_label', 'Stage 3')
        )

        print("\n📝 Querying Report Agent to author the final manuscript text...")
        report_agent = ReportAgent()

        abstract_text, conclusion_text = report_agent.generate_report_text(
            env_name=ENV_NAME,
            state_dim=state_dim,
            action_dim=action_encoding_dim,
            best_config=tracker.best_config,  # ⚠️ FIXED: Reads the true Best Config!
            best_mse=tracker.best_mse,
            best_rmse=tracker.best_rmse,
            latency=final_latency_ms,
            use_pinn=USE_PINN,
            max_latency=CUSTOMER_MAX_LATENCY_MS,
            success_score=final_score,
            model_status=model_status,
            complexity_label=getattr(loader, 'complexity_label', 'Unknown')
        )

        print("📄 Compiling automated engineering manuscript (PDF)...")

        pdf_path = generate_final_pdf(
            env_name=ENV_NAME,
            state_dim=state_dim,
            action_dim=action_encoding_dim,
            best_config=tracker.best_config,  # ⚠️ FIXED: Reads the true Best Config!
            best_mse=tracker.best_mse,
            best_rmse=tracker.best_rmse,
            latency=final_latency_ms,
            use_pinn=USE_PINN,
            timestamp=RUN_TIMESTAMP,
            abstract_text=abstract_text,
            conclusion_text=conclusion_text,
            success_score=final_score,
            model_status=model_status,
            complexity_label=getattr(loader, 'complexity_label', 'Unknown')
        )
        print(f"    ✅ PDF Report successfully generated: {pdf_path}")

        from framework import cost_tracker
        cost_tracker.print_summary(LLM_MODEL)

        from framework import package_final_results_to_zip
        package_final_results_to_zip(
            pdf_filename=pdf_path,
            timestamp=RUN_TIMESTAMP,
            env_name=ENV_NAME
        )

if __name__ == "__main__":
    main()




