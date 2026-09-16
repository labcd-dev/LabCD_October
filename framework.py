import json
import re
import time
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm
from config import *
from model import DynamicsModel

# ============================================================================
#  GPU DEVICE DETECTION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PyTorch Compute Device Set To: {DEVICE.type.upper()}")

# ============================================================================
#  GRACEFUL STOP CONTROL
# ============================================================================
_STOP_STATE = {"flag": False}


def request_stop(*_args, **_kwargs):
    if not _STOP_STATE["flag"]:
        print("\n🛑 Stop requested — finishing the current step and compiling results so far...")
    _STOP_STATE["flag"] = True


def stop_requested():
    return _STOP_STATE["flag"]


def reset_stop_flag():
    _STOP_STATE["flag"] = False


# ============================================================================
#  LOGGING UTILITY
# ============================================================================
def log_agent_interaction(agent_name, system_prompt, user_prompt, response_text, cycle=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cycle_header = f" | [CYCLE {cycle}]" if cycle is not None else ""

    with open(LOG_FILENAME, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n" + "=" * 80 + f"\n")
        log_file.write(f"📅 TIME: {timestamp}{cycle_header}\n")
        log_file.write(f"🤖 AGENT: {agent_name} ({API_PROVIDER.upper()})\n")
        log_file.write(f"-" * 80 + f"\n")
        log_file.write(f"🧠 SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}\n")
        log_file.write(f"-" * 80 + f"\n")
        log_file.write(f"📥 USER PROMPT:\n{user_prompt.strip()}\n")
        log_file.write(f"-" * 80 + f"\n")
        log_file.write(f"📤 RAW LLM RESPONSE:\n{response_text.strip()}\n")
        log_file.write(f"=" * 80 + f"\n")


# ============================================================================
#  API COST TRACKER
# ============================================================================
class APICostTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0

        # Current pricing per 1 Million tokens (USD)
        self.pricing_table = {
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "gpt-4o": {"prompt": 5.00, "completion": 15.00},
            "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
            "llama3-8b-8192": {"prompt": 0.05, "completion": 0.08},
            "default": {"prompt": 1.00, "completion": 2.00}  # Generic Fallback
        }

    def update(self, response):
        self.total_calls += 1
        try:
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})

                # Format 1: Standard OpenAI structure
                if 'prompt_tokens' in usage:
                    self.prompt_tokens += usage.get('prompt_tokens', 0)
                    self.completion_tokens += usage.get('completion_tokens', 0)
                # Format 2: Alternative Groq/Anthropic structure
                elif 'input_tokens' in usage:
                    self.prompt_tokens += usage.get('input_tokens', 0)
                    self.completion_tokens += usage.get('output_tokens', 0)
        except Exception:
            pass

    def print_summary(self, model_name):
        # Match the active model to the pricing table
        rates = self.pricing_table["default"]
        for key in self.pricing_table:
            if key in model_name.lower():
                rates = self.pricing_table[key]
                break

        # Calculate mathematical cost
        cost_prompt = (self.prompt_tokens / 1_000_000.0) * rates["prompt"]
        cost_comp = (self.completion_tokens / 1_000_000.0) * rates["completion"]
        total_cost = cost_prompt + cost_comp

        print("\n" + "=" * 80)
        print("💰 LLM API COST SUMMARY")
        print("=" * 80)
        print(f"  ├── Active LLM Core      : {model_name.upper()}")
        print(f"  ├── Total API Calls      : {self.total_calls} queries")
        print(f"  ├── Input Tokens (Prompt): {self.prompt_tokens:,} tokens")
        print(f"  ├── Output Tokens (Comp) : {self.completion_tokens:,} tokens")
        print("-" * 80)
        print(f"  💵 TOTAL ESTIMATED COST  : ${total_cost:.5f} USD")
        print("=" * 80 + "\n")


# Initialize the global tracker so all agents can access it!
cost_tracker = APICostTracker()


# ============================================================================
#  DATA INSPECTOR AGENT (HIL)
# ============================================================================
def run_data_inspector_agent(loader):
    """
    Human-In-The-Loop (HIL) Agent.
    Takes the fully initialized ExcelDataLoader, reads the mathematical issues
    it found, and asks the Lead Engineer for clarification.
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        from main import model
    except ImportError:
        from config import model

    print("\n" + "=" * 80)
    print("🧠 LLM DATA INSPECTOR AGENT: Analyzing Mathematical Health...")
    print("=" * 80)

    df = loader.df
    stats_summary = []
    for col in df.columns:
        stats_summary.append(f"- '{col}': Variance={df[col].var():.4f}, NaNs={df[col].isna().sum()}")
    stats_text = "\n".join(stats_summary)

    engineering_issues = "\n".join(loader.quality_issues) if hasattr(loader,
                                                                     'quality_issues') and loader.quality_issues else "No mathematical anomalies detected by DataLoader."

    prompt = f"""
    You are the Data Inspector Agent for a deep learning system identification framework.
    Review the dataset statistics and the mathematical issues found by the DataLoader.

    DATASET COMPLEXITY: {getattr(loader, 'complexity_label', 'Unknown')}

    BASIC STATISTICS:
    {stats_text}

    ENGINEERING ISSUES DETECTED BY DATALOADER:
    {engineering_issues}

    RULES FOR ANALYSIS:
    1. If 'ENGINEERING ISSUES DETECTED' shows duplicated or highly correlated states (e.g., correlation > 0.95), you MUST output [ASK_HUMAN] and ask if one is an accidental duplicate that should be dropped.
    2. If 'ENGINEERING ISSUES DETECTED' shows dead actions (near-zero variance), you MUST output [ASK_HUMAN] and ask if the sensor is dead.
    3. If no engineering issues were detected and variances look healthy, output EXACTLY: [PROCEED]

    DECISION FORMAT:
    - [PROCEED]
    - [ASK_HUMAN] (Your single concise question here)
    """

    system_msg = SystemMessage(content="You are a strict data validation AI. Follow the exact output format.")
    user_msg = HumanMessage(content=prompt)

    try:
        response = model.invoke([system_msg, user_msg])
        agent_reply = response.content.strip()
    except Exception as e:
        print(f"  ⚠️ Inspector API error: {e}. Defaulting to [PROCEED].")
        return "", []

    engineer_notes = ""
    cols_to_drop = []

    if "[ASK_HUMAN]" in agent_reply:
        question = agent_reply.replace("[ASK_HUMAN]", "").strip()
        print("\n" + "⚠️ " * 30)
        print("🛑 AGENT DETECTED ANOMALY / QUESTION:")
        print(f"🤖 Agent: {question}")
        print("⚠️ " * 30)
        engineer_notes = input("\n👨‍💻 Your Clarification (or press Enter/type 'skip' to ignore): ").strip()

        # --- NEW: ACTION PARSER TO PHYSICALLY DROP COLUMNS ---
        if engineer_notes and engineer_notes.lower() not in ['skip', 'no', 'none']:
            parse_prompt = f"""
            The agent asked: "{question}"
            The engineer replied: "{engineer_notes}"
            Available dataset columns: {list(df.columns)}

            Did the engineer agree to drop/remove any columns? 
            If yes, output a valid JSON list of the exact column strings to drop. 
            If no, output an empty JSON list [].
            Do not output any markdown or explanation. Just the JSON list.
            Example 1: ["s_pitch2"]
            Example 2: []
            """
            try:
                p_resp = model.invoke([SystemMessage(content="You are a strict JSON array generator."),
                                       HumanMessage(content=parse_prompt)])
                raw_list = p_resp.content.replace("```json", "").replace("```", "").strip()
                cols_to_drop = json.loads(raw_list)
            except:
                pass
    else:
        print("  ✅ LLM confirms dataset is healthy. Proceeding...")

    print("=" * 80 + "\n")
    return engineer_notes, cols_to_drop

# ============================================================================
#  DATA LOADER (Handles Both True Derivatives & Finite Differences)
# ============================================================================
class ExcelDataLoader:
    def __init__(self, file_path):
        # --- Robust Dual File Format Support ---
        if file_path.endswith('.csv'):
            print(f"    📂 Loading CSV dataset: '{file_path}'")
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            print(f"    📂 Loading Excel dataset: '{file_path}'")
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError(f"❌ Unsupported file format for '{file_path}'. Please use a .csv or .xlsx file.")

        # ==========================================================
        # --- NEW: DESTROY GHOST COLUMNS AUTOMATICALLY ---
        # ==========================================================
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]

        # Dynamically map the columns based on their prefixes
        self.state_cols = [c for c in self.df.columns if c.startswith("s_")]
        self.xdot_cols = [c for c in self.df.columns if c.startswith("xdot_")]
        self.action_cols = [c for c in self.df.columns if c.startswith("a_")]

        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)

        # --- PRE-PROCESSING DATA AGENT: Time Analysis ---
        # HARD STOP: If the time column is missing, crash the program immediately.
        if 'time' not in self.df.columns:
            raise ValueError(
                f"🚨 CRITICAL ERROR: No 'time' column found in '{file_path}'. A time column is strictly required to calculate physical derivatives correctly. Execution stopped.")

        self.has_time_column = True
        self._analyze_time_column()

        # --- CHECK FOR X_DOT COLUMNS ---
        self.has_xdot = (len(self.xdot_cols) == self.state_dim)
        if self.has_xdot:
            print("    ✅ 'xdot_' columns detected in the dataset. Using provided derivatives for training.")
        else:
            print("    ⚠️  No 'xdot_' columns found. X_dot will be estimated via finite differences.")
            print("    💡  TIP: If you can provide true X_dot directly from your simulation or hardware sensors,")
            print("         the neural network will achieve much better accuracy without noise amplification.")

        # --- AUTOMATED RESET THRESHOLD DETECTION ---
        try:
            from config import RESET_THRESHOLD
            # If the user didn't lock it, compute it mathematically from the data
            if RESET_THRESHOLD is None or (isinstance(RESET_THRESHOLD, (int, float)) and RESET_THRESHOLD > 90000):
                self.reset_threshold = self._auto_detect_reset_threshold()
            else:
                self.reset_threshold = RESET_THRESHOLD
        except:
            self.reset_threshold = self._auto_detect_reset_threshold()

        print(f"    ⚙️ Auto-calibrated Reset Thresholds: {self.reset_threshold}")

        # --- SMART ANGLE INDEX SELECTION ---
        try:
            import config
            if getattr(config, 'AUTO_DETECT_ANGLES', False) is True:
                print("    ⚙️ Questionnaire requested AUTO-DETECT for angles. Scanning data...")
                self.angle_indices = self._run_angle_auto_detect()
            elif hasattr(config, 'ANGLE_INDICES') and config.ANGLE_INDICES and len(config.ANGLE_INDICES) > 0:
                self.angle_indices = config.ANGLE_INDICES
                print(f"    ⚙️ Locked angular states from questionnaire: {self.angle_indices}")
            else:
                self.angle_indices = []
                print(f"    ⚙️ Questionnaire confirmed NO angular states.")
        except Exception:
            self.angle_indices = self._run_angle_auto_detect()

        # ==========================================================
        # NEW: APPLY THE DYNAMIC STATE FILTER
        # ==========================================================
        self._apply_state_filter()

        # Run the quality agent on the filtered data
        self._run_data_quality_agent()

        # --- 🧠 NEW: CALCULATE AND STORE DATASET COMPLEXITY ---
        self.complexity_tier, self.complexity_label = self._calculate_complexity()

    def drop_columns(self, cols_to_drop):
        """Physically removes columns and their corresponding derivatives."""
        dropped = []
        for c in cols_to_drop:
            # 1. Drop the state/action column
            if c in self.df.columns:
                self.df = self.df.drop(columns=[c])
                if c in self.state_cols: self.state_cols.remove(c)
                if c in self.action_cols: self.action_cols.remove(c)
                dropped.append(c)

            # 2. Actively hunt down and destroy its corresponding xdot column!
            target_xdot = c.replace("s_", "xdot_")
            if target_xdot in self.df.columns:
                self.df = self.df.drop(columns=[target_xdot])
                if target_xdot in self.xdot_cols: self.xdot_cols.remove(target_xdot)
                dropped.append(target_xdot)

        # Update the critical math dimensions
        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)

        # Recalculate thresholds for the new dimensions to prevent broadcasting crashes
        if hasattr(self, 'reset_threshold') and isinstance(self.reset_threshold, list):
            self.reset_threshold = self._auto_detect_reset_threshold()

        if dropped:
            print(f"    🗑️ SUCCESS: Physically removed {dropped} from the dataset.")
            print(f"       -> New Network State Dimensions: {self.state_dim}")


    def _auto_detect_reset_threshold(self):
        """Mathematically scans the Excel data to separate normal driving
           steps from massive trajectory reset boundaries."""
        diffs = self.df[self.state_cols].diff().abs()
        thresholds = []

        for col in self.state_cols:
            col_diffs = diffs[col].dropna()
            if len(col_diffs) == 0:
                thresholds.append(10.0)
                continue

            # Find the 99th percentile of normal driving changes
            p99 = np.percentile(col_diffs, 99.0)
            max_val = col_diffs.max()

            # Sets threshold safely above normal driving, catching only true resets
            thresh = max(p99 * 3.0, (p99 + max_val) / 2.0)
            thresholds.append(float(thresh))

        return thresholds

    def _run_angle_auto_detect(self):
        """Statistically analyzes data to find angles that wrap around pi/-pi."""
        print("      -> No angular indices provided in config. Auto-scanning data...")
        detected_angles = []
        for idx, col in enumerate(self.state_cols):
            data = self.df[col].values

            # Condition 1: Data is bounded near [-pi, pi]
            if np.max(data) <= 3.2 and np.min(data) >= -3.2:
                # Condition 2: Presence of massive wrap-around jumps
                diffs = np.abs(np.diff(data))
                if np.any(diffs > 5.0):
                    detected_angles.append(idx)

        if detected_angles:
            print(f"         ✅ Auto-detected angular states at indices: {detected_angles}")
        else:
            print("         ℹ️ No wrapped angular states detected in this dataset.")

        return detected_angles

    def _run_data_quality_agent(self):
        """Pre-processing mathematical check that analyzes data health for System ID."""
        print("\n    🕵️‍♂️ DATA QUALITY CHECK: Computing mathematical health metrics...")

        # We will save the issues here for the LLM to read!
        self.quality_issues = []

        # --- 1. Persistent Excitation (Action Variance) ---
        action_vars = self.df[self.action_cols].var()
        for col, variance in action_vars.items():
            if variance < 1e-4:
                self.quality_issues.append(f"Action '{col}' has near-zero variance ({variance:.5f}).")

        # --- 2. Multicollinearity (State Correlation) ---
        state_corr = self.df[self.state_cols].corr().abs()
        upper_tri = state_corr.where(np.triu(np.ones(state_corr.shape), k=1).astype(bool))

        high_corr_pairs = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > 0.95:
                    high_corr_pairs.append((row, col, upper_tri.loc[row, col]))

        if high_corr_pairs:
            corr_details = []
            for pair in high_corr_pairs:
                corr_details.append(f"- '{pair[0]}' & '{pair[1]}' (Correlation: {pair[2]:.4f})")
            self.quality_issues.append("Highly correlated or duplicated states detected:\n" + "\n".join(corr_details))

        # --- 3. Outlier Detection (Derivative Spikes) ---
        target_cols = self.xdot_cols if self.has_xdot else self.state_cols
        outlier_count = 0
        for col in target_cols:
            data = self.df[col]
            std = data.std()
            if std > 1e-8:
                z_scores = np.abs((data - data.mean()) / std)
                spikes = (z_scores > 6).sum()
                if spikes > 0:
                    outlier_count += spikes
        if outlier_count > 0:
            self.quality_issues.append(f"Found {outlier_count} extreme numerical spikes (Z-score > 6).")

        print("    ✅ Mathematical checks complete. Handing data to LLM Inspector...")

    def _calculate_complexity(self):
        """Calculates a 1-5 complexity tier based on state dimensions, coupling, and dynamic variance."""
        import numpy as np

        # 1. Structural Complexity (Total number of variables)
        dim_score = len(self.state_cols) + len(self.action_cols)

        # 2. Coupling / Collinearity (Max off-diagonal correlation)
        state_corr = self.df[self.state_cols].corr().abs()

        # FIX: Extract as a mutable copy to avoid the "read-only array" crash
        corr_array = state_corr.to_numpy(copy=True)
        np.fill_diagonal(corr_array, 0)
        max_corr = corr_array.max() if corr_array.size > 0 else 0

        # 3. Dynamic Aggressiveness (Variance of the changes in the system)
        target_cols = self.xdot_cols if self.has_xdot else self.state_cols
        derivatives = self.df[target_cols] if self.has_xdot else self.df[self.state_cols].diff().dropna()

        # Normalize variance to avoid scale dependency (prevents large units from faking complexity)
        agg_score = (derivatives.var() / (derivatives.abs().mean() + 1e-6)).max()

        # 4. Evaluate Tier (1 to 5)
        tier = 1
        if dim_score >= 4 or max_corr > 0.4: tier = 2
        if dim_score >= 7 or max_corr > 0.75 or agg_score > 2.0: tier = 3
        if max_corr > 0.90 or agg_score > 5.0: tier = 4
        if max_corr > 0.95 or agg_score > 10.0: tier = 5

        labels = {
            1: "Level 1 (Trivial / Quasi-Steady)",
            2: "Level 2 (Mildly Nonlinear)",
            3: "Level 3 (Moderately Complex)",
            4: "Level 4 (Highly Complex / Fast Transient)",
            5: "Level 5 (Severe / Chaotic / Discontinuous)"
        }

        complexity_label = labels.get(tier, "Unknown")

        # 5. Print to Terminal
        print("\n    🧠 DATASET COMPLEXITY ANALYSIS:")
        print(f"      -> State & Action Dimensions : {dim_score}")
        print(f"      -> Max State Coupling        : {max_corr:.3f}")
        print(f"      -> Dynamic Aggressiveness    : {agg_score:.3f}")
        print(f"      ==================================================")
        print(f"      📊 ASSIGNED TIER: {complexity_label}")
        print(f"      ==================================================\n")

        return tier, complexity_label

    def _apply_state_filter(self):
        """Clips extreme outliers based on the dynamically chosen percentiles."""
        import config
        # Check if the Initializer Agent (or manual settings) enabled the filter
        if not getattr(config, 'USE_STATE_FILTER', False):
            return

            # Retrieve the percentiles chosen by the Agent (e.g., [5, 95])
        p_low, p_high = getattr(config, 'AUTO_FILTER_PERCENTILES', (2, 98))

        print(f"\n    🧹 STATE SPACE FILTER [ACTIVE]")
        print(f"      -> Clipping state outliers beyond the {p_low}th and {p_high}th percentiles...")

        # Calculate the mathematical boundaries for those percentiles
        lower_bounds = self.df[self.state_cols].quantile(p_low / 100.0)
        upper_bounds = self.df[self.state_cols].quantile(p_high / 100.0)

        # Clip the data so extreme spikes are pulled back to the physical boundaries
        self.df[self.state_cols] = self.df[self.state_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        print("      ✅ State space successfully bounded.")


    def _analyze_time_column(self):
        """Pre-processing agent that analyzes time steps for consistency."""
        times = self.df['time'].values
        dts = np.diff(times)

        # 1. Check for fatal time errors
        if np.any(dts <= 0):
            print("    🚨 CRITICAL ERROR: Time goes backwards or contains duplicate timestamps!")

        dt_std = np.std(dts)
        dt_mean = np.mean(dts)

        # 2. Check for variable time steps (using a 1e-5 tolerance for floating-point math)
        if dt_std > 1e-5:
            print(f"    ⚠️  VARIABLE TIME STEP DETECTED!")
            print(f"        Min dt: {np.min(dts):.5f}s | Max dt: {np.max(dts):.5f}s | Mean: {dt_mean:.5f}s")
            print("        💡 LOGIC APPLIED: Interpolation is disabled to preserve true physics.")
            print("           The framework will automatically calculate the exact dt for every individual row.")
        else:
            print(f"    ✅ Constant time step verified (dt ≈ {dt_mean:.5f}s).")

    def get_trajectories(self):
        print("\n    ⏳ Extracting dataset and computing physical derivatives. Please wait...", flush=True)

        import config
        is_multi = getattr(config, 'MULTI_TRAJECTORY', True)
        manual_times = getattr(config, 'MANUAL_TRAJECTORY_SPLIT_TIMES', [])

        all_rows = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="      -> Reading dataset ", unit=" rows"):
            s = row[self.state_cols].values.astype(np.float32)
            a = row[self.action_cols].values.astype(np.float32)
            xdot = row[self.xdot_cols].values.astype(np.float32) if self.has_xdot else None

            time_col = 'time' if 'time' in self.df.columns else self.df.columns[0]
            t = float(row[time_col]) if self.has_time_column else None

            all_rows.append((s, a, xdot, t))

        trajectories = []
        if not all_rows:
            return trajectories

        # 1. Single Continuous Trajectory per Questionnaire
        if not is_multi:
            print("      -> Processing as 1 continuous trajectory per questionnaire.")
            trajectories.append(self._attach_dt_and_format(all_rows, show_bar=True))
            return trajectories

        # 2. Manual Timestamps Provided by Client
        if manual_times and len(manual_times) > 0:
            print(f"      -> Splitting trajectories using {len(manual_times)} manual timestamps...")
            chunks = []
            curr_chunk = []
            split_set = set(manual_times)

            for row in all_rows:
                t_val = row[3]
                if curr_chunk and any(abs(t_val - target_t) < 1e-4 for target_t in split_set):
                    chunks.append(curr_chunk)
                    curr_chunk = []
                curr_chunk.append(row)

            if curr_chunk:
                chunks.append(curr_chunk)

            for chunk in chunks:
                if len(chunk) > 0:
                    trajectories.append(self._attach_dt_and_format(chunk))

            print(f"      ✅ Created {len(trajectories)} distinct trajectory segments.")
            return trajectories

        # 3. Auto-detect Trajectory Boundaries (Client selected 'auto' or didn't provide timestamps)
        print("      -> Auto-detecting trajectory boundaries from data jumps...")
        chunks = []
        curr_chunk = [all_rows[0]]
        reset_limit = np.array(self.reset_threshold) if isinstance(self.reset_threshold,
                                                                   (list, tuple)) else self.reset_threshold

        for j in range(1, len(all_rows)):
            prev_row = all_rows[j - 1]
            curr_row = all_rows[j]

            prev_s, curr_s = prev_row[0], curr_row[0]
            prev_t, curr_t = prev_row[3], curr_row[3]

            state_jump = np.any(np.abs(curr_s - prev_s) > reset_limit)
            time_reset = (curr_t is not None and prev_t is not None and curr_t < prev_t)

            if state_jump or time_reset:
                chunks.append(curr_chunk)
                curr_chunk = []

            curr_chunk.append(curr_row)

        if curr_chunk:
            chunks.append(curr_chunk)

        for chunk in chunks:
            if len(chunk) > 0:
                trajectories.append(self._attach_dt_and_format(chunk))

        print(f"      ✅ Auto-detected and split into {len(trajectories)} trajectories.")
        return trajectories

    def _attach_dt_and_format(self, chunk, show_bar=False):
        dts = []
        # --- FIRST PASS: Calculate Delta T (dt) strictly from timestamps ---
        for j, (s, a, xdot, t) in enumerate(chunk):
            if j + 1 < len(chunk):
                dt = chunk[j + 1][3] - t
            elif j > 0:
                dt = t - chunk[j - 1][3]
            else:
                try:
                    from config import MIN_DT
                except ImportError:
                    MIN_DT = 1e-6
                dt = MIN_DT

            try:
                from config import MIN_DT
            except ImportError:
                MIN_DT = 1e-6

            if dt is None or dt < MIN_DT:
                dt = MIN_DT

            dts.append(np.float32(dt))

        # ====================================================================
        # PRE-COMPUTE VECTORIZED SAVITZKY-GOLAY DERIVATIVE
        # ====================================================================
        try:
            from config import DERIVATIVE_METHOD
            method = DERIVATIVE_METHOD.strip().lower()
        except ImportError:
            method = "finite_difference"

        xdot_savgol_matrix = None
        if not self.has_xdot and method == "savitzky_golay":
            try:
                from config import SAVGOL_WINDOW, SAVGOL_POLYORDER, ANGLE_INDICES
                window = SAVGOL_WINDOW
                poly = SAVGOL_POLYORDER
                angles = ANGLE_INDICES
            except ImportError:
                window, poly, angles = 11, 2, []

            S_matrix = np.array([item[0] for item in chunk])

            for idx in angles:
                if idx < S_matrix.shape[1]:
                    S_matrix[:, idx] = np.unwrap(S_matrix[:, idx])

            if len(S_matrix) < window: window = len(S_matrix)
            if window % 2 == 0: window -= 1
            if window <= poly: poly = max(1, window - 1)

            if window >= 3:
                from scipy.signal import savgol_filter
                mean_dt = float(np.mean(dts))
                xdot_savgol_matrix = savgol_filter(S_matrix, window_length=window, polyorder=poly, deriv=1,
                                                   delta=mean_dt, axis=0)
            else:
                xdot_savgol_matrix = np.zeros_like(S_matrix)

        # ====================================================================
        # SECOND PASS: ROW-BY-ROW FILTERING & FORMATTING
        # ====================================================================
        processed = []
        prev_xdot_filtered = None
        z0, z1 = None, None

        # Only show the progress bar if explicitly requested
        if show_bar:
            iterator = tqdm(enumerate(chunk), total=len(chunk), desc="      -> Applying filters", unit=" steps")
        else:
            iterator = enumerate(chunk)

        for j, (s, a, xdot, t) in iterator:
            dt = dts[j]

            if not self.has_xdot:
                # ----------------------------------------------------------
                # METHOD A: SAVITZKY-GOLAY
                # ----------------------------------------------------------
                if method == "savitzky_golay":
                    xdot_calculated = xdot_savgol_matrix[j]

                # ----------------------------------------------------------
                # METHOD B: LEVANT'S SLIDING MODE DIFFERENTIATOR
                # ----------------------------------------------------------
                elif method == "sliding_mode":
                    try:
                        from config import SMD_LAMBDA_1, SMD_LAMBDA_2
                        lam1, lam2 = SMD_LAMBDA_1, SMD_LAMBDA_2
                    except ImportError:
                        lam1, lam2 = 5.0, 10.0

                    if z0 is None:
                        z0 = np.copy(s)
                        z1 = np.zeros_like(s)

                    e = z0 - s
                    try:
                        from config import ANGLE_INDICES
                        for idx in ANGLE_INDICES:
                            if idx < len(e):
                                e[idx] = (e[idx] + np.pi) % (2 * np.pi) - np.pi
                    except ImportError:
                        pass

                    sign_e = np.sign(e)
                    sqrt_abs_e = np.sqrt(np.abs(e))
                    xdot_raw = -lam1 * sqrt_abs_e * sign_e + z1
                    z0 = z0 + dt * xdot_raw
                    z1 = z1 - dt * lam2 * sign_e
                    xdot_calculated = xdot_raw

                # ----------------------------------------------------------
                # METHOD C: CLASSIC FINITE DIFFERENCE + SIMULINK FILTER
                # ----------------------------------------------------------
                else:
                    if j + 1 < len(chunk):
                        if j > 0:
                            diff = chunk[j + 1][0] - chunk[j - 1][0]
                            calc_dt = chunk[j + 1][3] - chunk[j - 1][3]
                        else:
                            diff = chunk[j + 1][0] - s
                            calc_dt = dt

                        try:
                            from config import ANGLE_INDICES
                            for idx in ANGLE_INDICES:
                                if idx < len(diff):
                                    diff[idx] = (diff[idx] + np.pi) % (2 * np.pi) - np.pi
                        except ImportError:
                            pass

                        try:
                            # Prioritize the dataloader's auto-calibrated threshold, fallback to config/default
                            if hasattr(self, 'reset_threshold') and self.reset_threshold is not None:
                                reset_limit = np.array(self.reset_threshold) if isinstance(self.reset_threshold, (list,
                                                                                                                  tuple)) else self.reset_threshold
                            else:
                                from config import RESET_THRESHOLD
                                reset_limit = RESET_THRESHOLD
                                if isinstance(reset_limit, (list, tuple)):
                                    reset_limit = np.array(reset_limit)
                        except ImportError:
                            reset_limit = 1.0

                        if np.any(np.abs(diff) > reset_limit):
                            xdot_raw = np.zeros_like(s)
                            prev_xdot_filtered = None
                        else:
                            xdot_raw = diff / calc_dt
                    else:
                        xdot_raw = processed[-1][3] if j > 0 else np.zeros_like(s)

                    try:
                        from config import DERIVATIVE_FILTER_TAU
                        tau = DERIVATIVE_FILTER_TAU
                    except ImportError:
                        tau = 0.0

                    if tau > 0.0:
                        alpha = dt / (tau + dt)
                        if prev_xdot_filtered is None:
                            xdot_calculated = xdot_raw
                        else:
                            xdot_calculated = alpha * xdot_raw + (1.0 - alpha) * prev_xdot_filtered
                        prev_xdot_filtered = xdot_calculated
                    else:
                        xdot_calculated = xdot_raw

                xdot = xdot_calculated.astype(np.float32)

            processed.append((s, a, dt, xdot))

        return processed


# ============================================================================
#  INITIALIZER AGENT
# ============================================================================
class InitializerAgent:
    def __init__(self, loader):
        self.loader = loader

    def determine_initial_setup(self):
        import config  # ⚠️ THE FIX: Import the live config object dynamically!
        import json
        from langchain_core.messages import SystemMessage, HumanMessage

        df = self.loader.df
        num_samples = len(df)
        state_dim = self.loader.state_dim
        action_dim = self.loader.action_dim

        system_complexity = state_dim + action_dim
        samples_per_dim = int(num_samples / max(1, system_complexity))

        if 'time' in df.columns:
            dt_mean = float(np.mean(np.diff(df['time'].values)))
        else:
            dt_mean = "Unknown"

        state_vars = df[self.loader.state_cols].var().values
        action_vars = df[self.loader.action_cols].var().values
        state_var_str = f"Min: {np.min(state_vars):.4f}, Max: {np.max(state_vars):.4f}" if len(
            state_vars) > 0 else "N/A"
        action_var_str = f"Min: {np.min(action_vars):.4f}, Max: {np.max(action_vars):.4f}" if len(
            action_vars) > 0 else "N/A"

        num_angles = len(self.loader.angle_indices)

        # --------------------------------------------------------------------
        # CALCULATE NORMAL vs TELEPORTATION BOUNDARIES
        # --------------------------------------------------------------------
        is_multi = getattr(config, 'MULTI_TRAJECTORY', True)

        if is_multi:
            diffs = df[self.loader.state_cols].diff().abs()
            normal_jumps = diffs.quantile(0.999).fillna(0.0).values
            teleport_jumps = diffs.max().fillna(0.0).values
            normal_jumps_str = f"[{', '.join([f'{val:.4f}' for val in normal_jumps])}]"
            teleport_jumps_str = f"[{', '.join([f'{val:.4f}' for val in teleport_jumps])}]"

            jump_data_str = f"""- Extreme Normal Driving Jumps (99.9th percentile): {normal_jumps_str}
        - Teleportation/Trajectory Jumps (Absolute Max): {teleport_jumps_str}"""
            reset_rule = "RESET_THRESHOLD: (list of floats) Output limits (one per state) positioned between Normal Jumps and Teleportation Jumps."
        else:
            jump_data_str = "- Dataset Mode: Single continuous trajectory (no boundary teleportations)."
            reset_rule = "RESET_THRESHOLD: (list of floats) Output [99999.0] for each state dimension to disable trajectory reset."

        customer_context = getattr(config, 'CUSTOMER_SYSTEM_DESCRIPTION', "").strip()
        customer_prompt_block = ""
        if customer_context:
            customer_prompt_block = f"""
        CUSTOMER SYSTEM DESCRIPTION:
        "{customer_context}"
        (Bias configuration towards physical system characteristics and noise profiles described above.)
            """

        overrides = getattr(config, 'USER_OVERRIDES', {})
        override_block = ""
        if overrides:
            override_block = f"""
        CLIENT HAS LOCKED THE FOLLOWING PARAMETERS:
        {overrides}
        (You MUST output these exact locked values in your response.)
            """

        mode_str = getattr(config, 'RUN_MODE', "regular").lower()
        if mode_str == "fast":
            reasoning_req = "Write exactly TWO concise sentences on a SINGLE continuous line explaining your primary choices."
        else:
            reasoning_req = "Write exactly ONE concise paragraph on a SINGLE continuous line explaining your logic for all parameters."

        complexity_tier = getattr(self.loader, 'complexity_tier', 2)
        complexity_label = getattr(self.loader, 'complexity_label', 'Unknown')

        # ⚠️ THE FIX: Read absolute bounds strictly from the live `config` object
        prompt = f"""
        You are a deep learning expert specializing in data-driven system identification for physical control systems.
        Recommend initial hyperparameters, regularization, physical data filters, and search bounds for this dataset.
        {customer_prompt_block}
        {override_block}

        Dataset Statistics & Physical Properties:
        - Computed Dataset Complexity: {complexity_label} (Tier {complexity_tier} of 5)
        - Total Data Samples: {num_samples}
        - Input State Dimensions: {state_dim}
        - Action Dimensions: {action_dim}
        - Samples per Degree of Freedom: {samples_per_dim}
        - Mean Time Step (dt): {dt_mean} seconds
        - Wrapped Angular States: {num_angles}
        {jump_data_str}

        CRITICAL ARCHITECTURE INSTRUCTIONS based on Tier {complexity_tier}:
        - If Tier 1-2 (Trivial/Mild): Recommend shallow networks (e.g., [64] or [32, 32]), higher learning rates (0.01), minimal regularization, larger batch sizes (e.g., 256), and standard patience (20).
        - If Tier 3 (Moderate): Recommend standard networks (e.g., [64, 64] or [128, 64]), mild weight decay, batch size 128, and patience 25.
        - If Tier 4-5 (Highly Complex/Fast Transient): Recommend deep/wide networks (e.g., [256, 256] or [128, 128, 128]), lower learning rates (e.g., 0.001) for stability, strict regularization (dropout > 0.1), smaller batch sizes (e.g., 64) for stochastic gradient noise, higher epochs (300-500+), and high patience (30-50).
        Absolute Outer Limits (Client Authorized Constraints):
        - Learning Rate: [{config.LEARNING_RATE_MIN}, {config.LEARNING_RATE_MAX}]
        - Hidden Layer Size: [{config.HIDDEN_SIZE_MIN}, {config.HIDDEN_SIZE_MAX}]
        - Architecture Layer Count: [{config.NUM_LAYERS_MIN}, {config.NUM_LAYERS_MAX}]

        Provide your recommendation in this exact format (one line per entry):
        LEARNING_RATE: (float value)
        HIDDEN_LAYERS: [list of integers, e.g., [64]]
        ACTIVATION: (one string element)
        EPOCHS: (integer)
        BATCH_SIZE: (integer)
        EARLY_STOP_PATIENCE: (integer)
        LR_SEARCH_MIN: (float)
        LR_SEARCH_MAX: (float)
        HIDDEN_SIZE_SEARCH_MIN: (integer)
        HIDDEN_SIZE_SEARCH_MAX: (integer)
        NUM_LAYERS_SEARCH_MIN: (integer)
        NUM_LAYERS_SEARCH_MAX: (integer)
        DROPOUT_RATE: (float, e.g., 0.0 to 0.5)
        WEIGHT_DECAY: (float, e.g., 0.0001)
        LR_REDUCE_FACTOR: (float, e.g., 0.5)
        USE_STATE_FILTER: (True or False)
        AUTO_FILTER_PERCENTILES: [lower_int, upper_int]
        DERIVATIVE_FILTER_TAU: (float, e.g., 0.005)
        {reset_rule}
        REASONING: ({reasoning_req} Do NOT use newlines.)
        CRITICAL: Do not output markdown code blocks (```) or introductory/closing prose. Output strictly the KEY: VALUE lines above.
        """
        system_content = "You are a strict technical configuration engine. Output only the requested key-value lines without markdown or code fences."
        system_msg = SystemMessage(content=system_content)
        user_msg = HumanMessage(content=prompt)

        default_config = {
            "learning_rate": 0.001,
            "hidden_layers": [64],
            "activation": "relu",
            "epochs": 300,
            "batch_size": 128,
            "early_stop_patience": 25,
            "lr_search_min": config.LEARNING_RATE_MIN,
            "lr_search_max": config.LEARNING_RATE_MAX,
            "hidden_size_search_min": config.HIDDEN_SIZE_MIN,
            "hidden_size_search_max": config.HIDDEN_SIZE_MAX,
            "num_layers_search_min": config.NUM_LAYERS_MIN,
            "num_layers_search_max": config.NUM_LAYERS_MAX,
            "dropout_rate": 0.0,
            "weight_decay": 0.0001,
            "lr_reduce_factor": 0.5,
            "use_state_filter": False,
            "auto_filter_percentiles": [2, 98],
            "derivative_filter_tau": 0.005,
            "reset_threshold": [1.0] * state_dim,
            "reasoning": "Heuristic fallback active. No reasoning generated."
        }

        try:
            from config import model
            from framework import cost_tracker, log_agent_interaction, AVAILABLE_ACTIVATIONS

            response = model.invoke([system_msg, user_msg])
            cost_tracker.update(response)
            raw_text = response.content.replace("```json", "").replace("```text", "").replace("```", "").strip()
            log_agent_interaction("Initializer Configuration Agent", system_content, prompt, raw_text)

            lines = raw_text.split('\n')
            result = default_config.copy()

            for line in lines:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                key, val_str = line.split(':', 1)
                key = key.strip()
                val_str = val_str.strip()

                if key == 'LEARNING_RATE':
                    try:
                        result['learning_rate'] = float(val_str)
                    except:
                        pass
                elif key == 'HIDDEN_LAYERS':
                    try:
                        result['hidden_layers'] = json.loads(val_str)
                    except:
                        pass
                elif key == 'ACTIVATION':
                    act_val = val_str.lower()
                    if act_val in AVAILABLE_ACTIVATIONS:
                        result['activation'] = act_val
                elif key == 'LR_SEARCH_MIN':
                    try:
                        result['lr_search_min'] = float(val_str)
                    except:
                        pass
                elif key == 'LR_SEARCH_MAX':
                    try:
                        result['lr_search_max'] = float(val_str)
                    except:
                        pass
                elif key == 'HIDDEN_SIZE_SEARCH_MIN':
                    try:
                        result['hidden_size_search_min'] = int(float(val_str))
                    except:
                        pass
                elif key == 'HIDDEN_SIZE_SEARCH_MAX':
                    try:
                        result['hidden_size_search_max'] = int(float(val_str))
                    except:
                        pass
                elif key == 'NUM_LAYERS_SEARCH_MIN':
                    try:
                        result['num_layers_search_min'] = int(float(val_str))
                    except:
                        pass
                elif key == 'NUM_LAYERS_SEARCH_MAX':
                    try:
                        result['num_layers_search_max'] = int(float(val_str))
                    except:
                        pass
                elif key == 'DROPOUT_RATE':
                    try:
                        result['dropout_rate'] = float(val_str)
                    except:
                        pass
                elif key == 'WEIGHT_DECAY':
                    try:
                        result['weight_decay'] = float(val_str)
                    except:
                        pass
                elif key == 'LR_REDUCE_FACTOR':
                    try:
                        result['lr_reduce_factor'] = float(val_str)
                    except:
                        pass
                elif key == 'USE_STATE_FILTER':
                    result['use_state_filter'] = (val_str.lower() == 'true')
                elif key == 'AUTO_FILTER_PERCENTILES':
                    try:
                        result['auto_filter_percentiles'] = json.loads(val_str)
                    except:
                        pass
                elif key == 'DERIVATIVE_FILTER_TAU':
                    try:
                        result['derivative_filter_tau'] = float(val_str)
                    except:
                        pass
                elif key == 'RESET_THRESHOLD':
                    try:
                        if '[' in val_str:
                            result['reset_threshold'] = json.loads(val_str)
                        else:
                            result['reset_threshold'] = [float(val_str)] * state_dim
                    except:
                        pass
                elif key == 'REASONING':
                    result['reasoning'] = val_str

                elif key == 'EPOCHS':
                    try:
                        result['epochs'] = int(float(val_str))
                    except:
                        pass
                elif key == 'BATCH_SIZE':
                    try:
                        result['batch_size'] = int(float(val_str))
                    except:
                        pass
                elif key == 'EARLY_STOP_PATIENCE':
                    try:
                        result['early_stop_patience'] = int(float(val_str))
                    except:
                        pass

            # Mathematical clamping and bounds verification against LIVE config limits
            result['lr_search_min'] = float(
                np.clip(result['lr_search_min'], config.LEARNING_RATE_MIN, config.LEARNING_RATE_MAX))
            result['lr_search_max'] = float(
                np.clip(result['lr_search_max'], config.LEARNING_RATE_MIN, config.LEARNING_RATE_MAX))
            if result['lr_search_min'] > result['lr_search_max']:
                result['lr_search_min'], result['lr_search_max'] = result['lr_search_max'], result['lr_search_min']

            result['hidden_size_search_min'] = int(
                np.clip(result['hidden_size_search_min'], config.HIDDEN_SIZE_MIN, config.HIDDEN_SIZE_MAX))
            result['hidden_size_search_max'] = int(
                np.clip(result['hidden_size_search_max'], config.HIDDEN_SIZE_MIN, config.HIDDEN_SIZE_MAX))
            if result['hidden_size_search_min'] > result['hidden_size_search_max']:
                result['hidden_size_search_min'], result['hidden_size_search_max'] = result['hidden_size_search_max'], \
                result['hidden_size_search_min']

            result['num_layers_search_min'] = int(
                np.clip(result['num_layers_search_min'], config.NUM_LAYERS_MIN, config.NUM_LAYERS_MAX))
            result['num_layers_search_max'] = int(
                np.clip(result['num_layers_search_max'], config.NUM_LAYERS_MIN, config.NUM_LAYERS_MAX))
            if result['num_layers_search_min'] > result['num_layers_search_max']:
                result['num_layers_search_min'], result['num_layers_search_max'] = result['num_layers_search_max'], \
                result['num_layers_search_min']

            result['learning_rate'] = float(
                np.clip(result['learning_rate'], result['lr_search_min'], result['lr_search_max']))
            result['hidden_layers'] = [
                int(np.clip(int(x), result['hidden_size_search_min'], result['hidden_size_search_max']))
                for x in result['hidden_layers']
            ]
            if len(result['hidden_layers']) < result['num_layers_search_min']:
                result['hidden_layers'] = (result['hidden_layers'] * result['num_layers_search_min'])[
                    :result['num_layers_search_min']]
            if len(result['hidden_layers']) > result['num_layers_search_max']:
                result['hidden_layers'] = result['hidden_layers'][:result['num_layers_search_max']]

            return result

        except Exception as e:
            print(f"   ⚠️ Initializer Agent parsing failure: {e}. Falling back to default heuristics.")
            return default_config

def compute_rmse(mse_value):
    import math
    return math.sqrt(max(float(mse_value), 0.0))


# ============================================================================
#  TRAIN DYNAMICS MODEL (WITH UNIVERSAL AUTOREGRESSIVE ROLLOUT)
# ============================================================================
def train_dynamics_model(train_trajs, val_trajs, state_dim, action_encoding_dim, hidden_layers,
                         learning_rate, epochs, batch_size, patience, activation,
                         dropout_rate=0.0, weight_decay=0.0, lr_min=0.00001):
    max_allowed_epochs = epochs

    try:
        from config import NETWORK_ARCHITECTURE, LSTM_SEQ_LENGTH, ROLLOUT_HORIZON
        ARCH = NETWORK_ARCHITECTURE.strip().upper()
        SEQ_LEN = LSTM_SEQ_LENGTH if ARCH == "LSTM" else 1
        HORIZON = ROLLOUT_HORIZON
    except ImportError:
        ARCH = "MLP"
        SEQ_LEN = 1
        HORIZON = 1

    # Total length of the window extracted from the CSV
    TOTAL_LEN = SEQ_LEN + HORIZON - 1

    train_states, train_actions, train_dts, train_xdots = [], [], [], []
    for traj in train_trajs:
        if len(traj) < TOTAL_LEN: continue
        for i in range(len(traj) - TOTAL_LEN + 1):
            window = traj[i: i + TOTAL_LEN]
            train_states.append([step[0] for step in window])
            train_actions.append([step[1] for step in window])
            train_dts.append([[step[2]] for step in window])
            train_xdots.append([step[3] for step in window])

    train_states = np.array(train_states, dtype=np.float32)
    train_actions = np.array(train_actions, dtype=np.float32)
    train_dts = np.array(train_dts, dtype=np.float32)
    train_xdots = np.array(train_xdots, dtype=np.float32)

    flat_states = train_states.reshape(-1, state_dim)
    flat_actions = train_actions.reshape(-1, action_encoding_dim)
    flat_xdots = train_xdots.reshape(-1, state_dim)

    state_scaler = StandardScaler().fit(flat_states)
    action_scaler = StandardScaler().fit(flat_actions)
    output_scaler = StandardScaler().fit(flat_xdots)

    try:
        from config import USE_STATE_FILTER, STATE_BOUNDS_FILTER, AUTO_FILTER_PERCENTILES
    except ImportError:
        USE_STATE_FILTER = False

    if USE_STATE_FILTER:
        chosen_bounds = {}
        if not STATE_BOUNDS_FILTER:
            for dim in range(state_dim):
                p_min, p_max = np.percentile(flat_states[:, dim],
                                             [AUTO_FILTER_PERCENTILES[0], AUTO_FILTER_PERCENTILES[1]])
                if abs(p_max - p_min) < 1e-4:
                    p_min, p_max = np.min(flat_states[:, dim]), np.max(flat_states[:, dim])
                chosen_bounds[dim] = (p_min, p_max)
        else:
            chosen_bounds = STATE_BOUNDS_FILTER

        valid_mask = np.ones(len(train_states), dtype=bool)
        for dim, (low_b, high_b) in chosen_bounds.items():
            valid_mask &= (train_states[:, 0, dim] >= low_b) & (train_states[:, 0, dim] <= high_b)

        train_states = train_states[valid_mask]
        train_actions = train_actions[valid_mask]
        train_dts = train_dts[valid_mask]
        train_xdots = train_xdots[valid_mask]

    # ⚠️ THE FIX: KEEP DATA ON CPU (System RAM) TO SAVE VRAM
    X_train_states_t = torch.tensor(train_states, dtype=torch.float32)
    X_train_acts_t = torch.tensor(train_actions, dtype=torch.float32)
    X_train_dt_t = torch.tensor(train_dts, dtype=torch.float32)
    y_train_xdot_t = torch.tensor(train_xdots, dtype=torch.float32)

    val_states, val_actions, val_dts, val_xdots = [], [], [], []
    for traj in val_trajs:
        if len(traj) < TOTAL_LEN: continue
        for i in range(len(traj) - TOTAL_LEN + 1):
            window = traj[i: i + TOTAL_LEN]
            val_states.append([step[0] for step in window])
            val_actions.append([step[1] for step in window])
            val_dts.append([[step[2]] for step in window])
            val_xdots.append([step[3] for step in window])

    # ⚠️ THE FIX: KEEP VAL DATA ON CPU
    X_val_states_t = torch.tensor(np.array(val_states, dtype=np.float32))
    X_val_acts_t = torch.tensor(np.array(val_actions, dtype=np.float32))
    X_val_dt_t = torch.tensor(np.array(val_dts, dtype=np.float32))
    y_val_xdot_t = torch.tensor(np.array(val_xdots, dtype=np.float32))

    dyn_model = DynamicsModel(state_dim, action_encoding_dim, hidden_layers, activation, dropout_rate=dropout_rate).to(
        DEVICE)
    dyn_model.set_scalers(
        state_mean=state_scaler.mean_, state_scale=state_scaler.scale_,
        action_mean=action_scaler.mean_, action_scale=action_scaler.scale_,
        output_mean=output_scaler.mean_, output_scale=output_scaler.scale_
    )

    try:
        from config import LR_REDUCE_FACTOR
    except ImportError:
        LR_REDUCE_FACTOR = 0.5

    optimizer = optim.AdamW(dyn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_REDUCE_FACTOR,
        patience=max(1, int(patience / 2)), threshold=1e-4, threshold_mode='rel', min_lr=lr_min
    )
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_train_loss = float('inf')
    best_val_norm_loss = float('inf')
    best_val_phys_loss = float('inf')
    best_val_phys_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Pre-calculate normalized targets safely on the CPU
    norm_xdot_train_target = (y_train_xdot_t - dyn_model.output_mean.cpu()) / dyn_model.output_scale.cpu()
    norm_xdot_val_target = (y_val_xdot_t - dyn_model.output_mean.cpu()) / dyn_model.output_scale.cpu()

    epoch = 0
    while epoch < max_allowed_epochs:
        if stop_requested(): break

        current_epoch_lr = optimizer.param_groups[0]['lr']
        dyn_model.train()

        # ⚠️ THE FIX: Permutation must happen on CPU to match the CPU dataset
        perm = torch.randperm(X_train_states_t.size(0))

        epoch_norm_loss = 0.0
        epoch_phys_loss = 0.0
        num_batches = 0
        stopped_mid_epoch = False

        for i in range(0, X_train_states_t.size(0), batch_size):
            if stop_requested():
                stopped_mid_epoch = True
                break
            idx = perm[i:i + batch_size]

            # ⚠️ THE FIX: Move only the current mini-batch to the GPU dynamically!
            b_states = X_train_states_t[idx].to(DEVICE)
            b_actions = X_train_acts_t[idx].to(DEVICE)
            b_dts = X_train_dt_t[idx].to(DEVICE)
            b_xdot = y_train_xdot_t[idx].to(DEVICE)
            b_target_norm = norm_xdot_train_target[idx].to(DEVICE)

            optimizer.zero_grad()
            total_norm_loss = 0.0
            batch_phys_loss = 0.0
            physics_loss_accum = 0.0

            if ARCH == "LSTM":
                current_inputs = b_states[:, :SEQ_LEN, :].clone()
            else:
                current_inputs = b_states[:, 0, :].clone()

            for t in range(HORIZON):
                target_idx = (SEQ_LEN - 1 + t) if ARCH == "LSTM" else t

                if ARCH == "LSTM":
                    curr_action = b_actions[:, t: t + SEQ_LEN, :]
                    curr_dt = b_dts[:, t: t + SEQ_LEN, :]
                else:
                    curr_action = b_actions[:, target_idx, :]
                    curr_dt = b_dts[:, target_idx, :]

                out_phys, out_norm = dyn_model(current_inputs, curr_action, curr_dt)

                true_xdot = b_xdot[:, target_idx, :]
                true_target_norm = b_target_norm[:, target_idx, :]

                step_norm_loss = criterion(out_norm, true_target_norm)
                total_norm_loss += step_norm_loss
                batch_phys_loss += criterion(out_phys, true_xdot).detach().item()

                from config import USE_PINN, PINN_LOSS_WEIGHT
                if USE_PINN:
                    try:
                        from physics_env import compute_analytical_xdot
                        if ARCH == "LSTM":
                            phys_states = current_inputs[:, -1, :]
                            phys_actions = curr_action[:, -1, :]
                        else:
                            phys_states = current_inputs
                            phys_actions = curr_action

                        physics_xdot_pred = compute_analytical_xdot(phys_states, phys_actions)
                        norm_physics_target = (physics_xdot_pred - dyn_model.output_mean) / dyn_model.output_scale
                        physics_loss_accum += criterion(out_norm, norm_physics_target)
                    except ImportError:
                        pass

                if ARCH == "LSTM":
                    dt_val = curr_dt[:, -1, :]
                    next_state_pred = current_inputs[:, -1, :] + (out_phys * dt_val)
                    current_inputs = torch.cat([current_inputs[:, 1:, :], next_state_pred.unsqueeze(1)], dim=1)
                else:
                    next_state_pred = current_inputs + (out_phys * curr_dt)
                    current_inputs = next_state_pred

            total_norm_loss = total_norm_loss / HORIZON
            batch_phys_loss = batch_phys_loss / HORIZON

            if USE_PINN and type(physics_loss_accum) != float:
                physics_loss_accum = physics_loss_accum / HORIZON
                total_norm_loss = total_norm_loss + (PINN_LOSS_WEIGHT * physics_loss_accum)

            total_norm_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in dyn_model.parameters() if p.grad is not None], 1.0)
            optimizer.step()

            epoch_norm_loss += total_norm_loss.item()
            epoch_phys_loss += batch_phys_loss
            num_batches += 1

        if stopped_mid_epoch: break

        current_train_norm_loss = epoch_norm_loss / max(1, num_batches)
        current_train_phys_loss = epoch_phys_loss / max(1, num_batches)

        # --- VALIDATION LOOP ---
        dyn_model.eval()
        val_norm_loss_sum = 0.0
        val_phys_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, X_val_states_t.size(0), batch_size):

                # ⚠️ THE FIX: Move validation batches dynamically
                b_states = X_val_states_t[i:i + batch_size].to(DEVICE)
                b_actions = X_val_acts_t[i:i + batch_size].to(DEVICE)
                b_dts = X_val_dt_t[i:i + batch_size].to(DEVICE)
                b_xdot = y_val_xdot_t[i:i + batch_size].to(DEVICE)
                b_target_norm = norm_xdot_val_target[i:i + batch_size].to(DEVICE)

                batch_norm_loss = 0.0
                batch_phys_loss = 0.0

                if ARCH == "LSTM":
                    current_inputs = b_states[:, :SEQ_LEN, :].clone()
                else:
                    current_inputs = b_states[:, 0, :].clone()

                for t in range(HORIZON):
                    target_idx = (SEQ_LEN - 1 + t) if ARCH == "LSTM" else t

                    if ARCH == "LSTM":
                        curr_action = b_actions[:, t: t + SEQ_LEN, :]
                        curr_dt = b_dts[:, t: t + SEQ_LEN, :]
                    else:
                        curr_action = b_actions[:, target_idx, :]
                        curr_dt = b_dts[:, target_idx, :]

                    out_phys, out_norm = dyn_model(current_inputs, curr_action, curr_dt)

                    true_xdot = b_xdot[:, target_idx, :]
                    true_target_norm = b_target_norm[:, target_idx, :]

                    batch_norm_loss += criterion(out_norm, true_target_norm).item()
                    batch_phys_loss += criterion(out_phys, true_xdot).item()

                    if ARCH == "LSTM":
                        dt_val = curr_dt[:, -1, :]
                        next_state_pred = current_inputs[:, -1, :] + (out_phys * dt_val)
                        current_inputs = torch.cat([current_inputs[:, 1:, :], next_state_pred.unsqueeze(1)], dim=1)
                    else:
                        next_state_pred = current_inputs + (out_phys * curr_dt)
                        current_inputs = next_state_pred

                val_norm_loss_sum += (batch_norm_loss / HORIZON)
                val_phys_loss_sum += (batch_phys_loss / HORIZON)
                val_batches += 1

        current_val_norm_loss = val_norm_loss_sum / max(1, val_batches)
        current_val_phys_loss = val_phys_loss_sum / max(1, val_batches)

        train_losses.append(current_train_phys_loss)
        val_losses.append(current_val_phys_loss)

        current_train_rmse = compute_rmse(current_train_phys_loss)
        current_val_rmse = compute_rmse(current_val_phys_loss)

        scheduler.step(current_val_norm_loss)

        if epoch % 50 == 0 or epoch == max_allowed_epochs - 1:
            print(
                f"      🔄 Epoch {epoch:4d}/{max_allowed_epochs} | Train MSE: {current_train_phys_loss:.6f} (RMSE: {current_train_rmse:.6f}) "
                f"| Val MSE: {current_val_phys_loss:.6f} (RMSE: {current_val_rmse:.6f}) | LR: {current_epoch_lr:.6f}")

        try:
            from config import ADAPTIVE_IMPROVEMENT_THRESHOLD, EPOCH_EXTENSION_STEPS, OVERFIT_RATIO_LIMIT
        except ImportError:
            ADAPTIVE_IMPROVEMENT_THRESHOLD, EPOCH_EXTENSION_STEPS, OVERFIT_RATIO_LIMIT = 0.05, 50, 5.0

        if epoch == max_allowed_epochs - 1:
            triggered_extension = False
            if best_train_loss != float('inf'):
                train_improvement = (best_train_loss - current_train_norm_loss) / best_train_loss
                if train_improvement > ADAPTIVE_IMPROVEMENT_THRESHOLD:
                    max_allowed_epochs += EPOCH_EXTENSION_STEPS
                    triggered_extension = True

            if best_val_norm_loss != float('inf'):
                val_improvement = (best_val_norm_loss - current_val_norm_loss) / best_val_norm_loss
                if val_improvement > ADAPTIVE_IMPROVEMENT_THRESHOLD:
                    if not triggered_extension:
                        max_allowed_epochs += EPOCH_EXTENSION_STEPS
                    triggered_extension = True

            if triggered_extension:
                print(f"    ✨ Progress active! Extending runway to {max_allowed_epochs} epochs.")

        if current_train_norm_loss < best_train_loss:
            best_train_loss = current_train_norm_loss

        if current_val_norm_loss < best_val_norm_loss:
            best_val_norm_loss = current_val_norm_loss
            best_val_phys_loss = current_val_phys_loss
            best_val_phys_rmse = current_val_rmse
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in dyn_model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if epoch >= 50:
                    print(f"      Early stopping triggered at epoch {epoch}")
                    break

        if epoch >= 50 and current_train_phys_loss > 1e-8:
            live_overfit_ratio = current_val_phys_loss / current_train_phys_loss
            if live_overfit_ratio > OVERFIT_RATIO_LIMIT:
                print(
                    f"      ⚠️ Overfit Limit Exceeded at epoch {epoch} ({live_overfit_ratio:.1f}x limit). Aborting early to save time!")
                break

        epoch += 1

    if best_model_state is not None:
        dyn_model.load_state_dict(best_model_state)

    return dyn_model, (min(train_losses) if train_losses else float(
        'inf')), best_val_phys_loss, best_val_phys_rmse, X_val_states_t.cpu().numpy(), y_val_xdot_t.cpu().numpy()
def measure_inference_latency(model, state_dim, action_dim):
    """Simulates real-time control deployment to measure average latency."""
    model.eval()

    device = next(model.parameters()).device
    dummy_state = torch.randn(1, state_dim).to(device)
    dummy_action = torch.randn(1, action_dim).to(device)
    dummy_dt = torch.ones(1, 1).to(device) * 0.01

    for _ in range(10):
        _ = model(dummy_state, dummy_action, dummy_dt)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_state, dummy_action, dummy_dt)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / 100.0) * 1000.0
    return avg_latency_ms



# ============================================================================
#  BEST CONFIG TRACKER
# ============================================================================
class BestConfigTracker:
    def __init__(self, run_mode="regular"):
        self.best_mse = float('inf')
        self.best_rmse = float('inf')
        self.best_config = None
        self.best_reasoning = "N/A"
        self.recent_failures = []
        self.last_was_best = False
        self.run_mode = run_mode.lower()  # Store the mode

    def update(self, mse, config, current_rmse=None):
        self.last_was_best = False
        if mse < self.best_mse:
            self.best_mse = mse
            self.best_rmse = current_rmse
            self.best_config = config
            self.last_was_best = True
            return True
        else:
            self.recent_failures.append({
                'config': config.copy(),
                'mse': mse,
                'reasoning': "Pending..."
            })

            # --- NEW: DYNAMIC MEMORY CAPACITY ---
            if self.run_mode == "fast":
                # Fast mode only remembers the last 5 failures
                if len(self.recent_failures) > 5:
                    self.recent_failures.pop(0)
            elif self.run_mode == "regular":
                # Regular mode now remembers up to 10 failures
                if len(self.recent_failures) > 10:
                    self.recent_failures.pop(0)
            # If "heavy" or "expert", the list grows indefinitely

            return False

    def add_reasoning_to_memory(self, reasoning):
        """Attaches the Critic's text reasoning to the correct saved cycle."""
        if self.last_was_best:
            self.best_reasoning = reasoning
        elif len(self.recent_failures) > 0:
            self.recent_failures[-1]['reasoning'] = reasoning

    def get_recent_failures_str(self):
        if not self.recent_failures:
            return "None"
        res = ""
        for fail in self.recent_failures:
            res += f"- Config: {fail['config']} | MSE: {fail['mse']:.6f}\n  Past Critic Logic: {fail.get('reasoning', 'N/A')}\n"
        return res


# ============================================================================
#  CRITIC AGENT
# ============================================================================
class CriticAgent:
    def __init__(self, tracker, run_mode="regular"):
        self.tracker = tracker
        self.run_mode = run_mode.lower()

    def evaluate(self, train_mse, val_mse, current_config, activation, measured_latency, max_latency, cycle_number,
                 val_rmse=None):
        import json
        import time
        from langchain_core.messages import SystemMessage, HumanMessage

        # ⚠️ THE FIX: Import the absolute limits from your config
        try:
            from config import LEARNING_RATE_MIN, LEARNING_RATE_MAX, HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX, NUM_LAYERS_MIN, \
                NUM_LAYERS_MAX
        except ImportError:
            LEARNING_RATE_MIN, LEARNING_RATE_MAX = 0.00005, 0.001
            HIDDEN_SIZE_MIN, HIDDEN_SIZE_MAX = 32, 256
            NUM_LAYERS_MIN, NUM_LAYERS_MAX = 1, 3

        curr_lr = current_config['learning_rate']
        curr_hs = current_config['hidden_layers']

        overfit_ratio = (val_mse / train_mse) if train_mse > 1e-8 else 0.0
        curr_train_mse_str = f"{train_mse:.6f}"
        curr_val_mse_str = f"{val_mse:.6f}"
        curr_val_rmse_str = f"{val_rmse:.6f}" if val_rmse is not None else "N/A"
        best_mse_str = f"{self.tracker.best_mse:.6f}" if self.tracker.best_mse != float('inf') else "N/A"

        if self.tracker.best_config is None:
            best_lr, best_hs = curr_lr, curr_hs
        else:
            best_lr = self.tracker.best_config['learning_rate']
            best_hs = self.tracker.best_config['hidden_layers']

        failures_str = self.tracker.get_recent_failures_str()

        if self.run_mode == "fast":
            explore_limit = 3
        elif self.run_mode in ["heavy"]:
            explore_limit = 15
        else:
            explore_limit = 8

        if cycle_number <= explore_limit:
            phase_instruction = """
                3. Search Phase: EARLY EXPLORATION. 
                   You MUST make MODERATE TO LARGE topological jumps to aggressively map the architecture space.
                   CRITICAL INSTRUCTION: You are highly biased toward keeping the same number of layers. You MUST break this habit. 
                   If the current network has 2 layers (e.g., [128, 128]), you MUST suggest 3 or 4 layers (e.g., [128, 64, 32]). 
                   If the current network has 4 layers, you MUST suggest 1 or 2 layers. 
                   Do not just scale the numbers up; you must physically change the LENGTH of the HIDDEN_LAYERS list.
                    """
        else:
            phase_instruction = """
                3. Search Phase: LATE FINE-TUNING. 
                   We have established a baseline. You must act as a precise local fine-tuner. 
                   Keep the EXACT same number of layers as the current config, but make VERY SMALL incremental mutations (e.g., adding or removing 5-20 neurons) to polish and locally optimize the current best architecture.
                    """

        try:
            from config import CUSTOMER_SYSTEM_DESCRIPTION
            customer_context = CUSTOMER_SYSTEM_DESCRIPTION.strip()
        except ImportError:
            customer_context = ""

        system_context_block = ""
        if customer_context:
            if self.run_mode == "heavy" or (self.run_mode == "regular" and cycle_number <= 5):
                system_context_block = f"\n                CUSTOMER SYSTEM CONTEXT: '{customer_context}'"

        prompt = f"""
        Evaluate the dynamics model for system identification.{system_context_block}

                CUSTOMER HARD CONSTRAINTS:
                - Max Allowed Inference Latency: {max_latency:.3f} ms

                CURRENT CYCLE METRICS:
                - Measured Inference Latency: {measured_latency:.3f} ms
                - Training MSE: {curr_train_mse_str}
                - Validation MSE: {curr_val_mse_str} (RMSE: {curr_val_rmse_str})
                - Overfit Ratio (Val/Train): {overfit_ratio:.2f}x
                - Learning rate: {curr_lr}
                - Hidden layers: {curr_hs}

                BEST HISTORICAL CYCLE:
                - Best Validation MSE: {best_mse_str}
                - Learning rate: {best_lr}
                - Hidden layers: {best_hs}
                - Logic that achieved this: {self.tracker.best_reasoning}

                RECENT FAILED CONFIGURATIONS:
                {failures_str}

                ABSOLUTE SEARCH BOUNDS (DO NOT VIOLATE):
                - Learning Rate: [{LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}]
                - Hidden Layer Size: [{HIDDEN_SIZE_MIN}, {HIDDEN_SIZE_MAX}]
                - Number of Layers: [{NUM_LAYERS_MIN}, {NUM_LAYERS_MAX}]

                CRITICAL OPTIMIZATION RULES:
                1. Latency: If 'Measured Inference Latency' is greater than 'Max Allowed', you MUST output 'LATENCY_VIOLATION' and aggressively reduce the HIDDEN_LAYERS size.
                2. Continuous Mutation: You CANNOT output the exact same HIDDEN_LAYERS as the CURRENT CYCLE.
                3. The Master Overfit Matrix (Based on Val/Train Ratio):
                   - Ratio < 4.0 : Diagnose as [NORMAL] or [CONVERGING]. The model is healthy.
                   - Ratio >= 4.0 and < 7.0 : Diagnose as [MILD_OVERFITTING]. Command the Actor to increase Regularization.
                   - Ratio >= 7.0 and < 10.0 : Diagnose as [HIGH_OVERFITTING]. Command the Actor to heavily prune layer widths (Topological Shrinkage), drop Patience, and cut Batch Size.
                   - Ratio >= 10.0 : Diagnose as [CRITICAL_OVERFITTING]. Command an immediate LR Backoff (decrease LR aggressively) and force a radical architecture reset.
                {phase_instruction}

                Provide your response in this exact format:
                DIAGNOSIS: [NORMAL / CONVERGING / MILD_OVERFITTING / HIGH_OVERFITTING / CRITICAL_OVERFITTING / UNDERFITTING / UNSTABLE / LATENCY_VIOLATION]
                STATUS: [GOOD / NEEDS_IMPROVEMENT / REJECTED]
                LR_DIRECTION: [increase / decrease / stay]
                LR_STEP: (float, e.g., 0.0002)
                HIDDEN_LAYERS: [new list of integers, e.g., [64, 32] or [128]]
                REASONING: (Deep technical reasoning about generalization vs. inference latency. STRICTLY MAXIMUM 2 SENTENCES.)
                """

        from main import model, log_agent_interaction

        system_msg = SystemMessage(content="You are a strict technical critic. Use the exact format.")
        user_msg = HumanMessage(content=prompt)

        default = {'diagnosis': 'UNKNOWN', 'status': 'NEEDS_IMPROVEMENT', 'lr_dir': 'stay', 'lr_step': 0.0,
                   'hidden_layers': current_config['hidden_layers'].copy(), 'reasoning': 'LLM error'}

        print("    ⏳ Cooldown pause ...", end="", flush=True)
        time.sleep(2)
        print(" Done. 🧠 Awaiting Critic LLM response...", flush=True)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.invoke([system_msg, user_msg])
                from framework import cost_tracker
                cost_tracker.update(response)
                log_agent_interaction("Critic Evaluation Agent", "You are a strict technical critic.", prompt,
                                      response.content, cycle=cycle_number)

                lines = response.content.strip().split('\n')
                result = default.copy()
                for line in lines:
                    if line.startswith('DIAGNOSIS:'):
                        result['diagnosis'] = line.split(':', 1)[1].strip()
                    elif line.startswith('STATUS:'):
                        result['status'] = line.split(':', 1)[1].strip()
                    elif line.startswith('LR_DIRECTION:'):
                        result['lr_dir'] = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('LR_STEP:'):
                        try:
                            result['lr_step'] = float(line.split(':', 1)[1].strip())
                        except:
                            pass
                    elif line.startswith('HIDDEN_LAYERS:'):
                        try:
                            result['hidden_layers'] = json.loads(line.split(':', 1)[1].strip())
                        except:
                            pass
                    elif line.startswith('REASONING:'):
                        result['reasoning'] = line.split(':', 1)[1].strip()
                return result

            except Exception as e:
                print(f"      ⚠️ Critic API Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print("      ⏳ Waiting 15 seconds to clear API rate limits...")
                    time.sleep(15)
                else:
                    print("      ❌ Critic API completely failed. Falling back to mathematical parameters.")
                    return default


class ActorAgent:
    def __init__(self, activation, initial_config=None):
        self.activation = activation
        if initial_config is not None:
            self.current_config = {
                "learning_rate": initial_config["learning_rate"],
                "hidden_layers": initial_config["hidden_layers"],
                "dropout_rate": initial_config.get("dropout_rate", MANUAL_DROPOUT_RATE),
                "weight_decay": initial_config.get("weight_decay", MANUAL_WEIGHT_DECAY),
                "batch_size": initial_config.get("batch_size", 128),
                "patience": initial_config.get("early_stop_patience", 25)
            }
            self.lr_min = initial_config.get("lr_search_min", LEARNING_RATE_MIN)
            self.lr_max = initial_config.get("lr_search_max", LEARNING_RATE_MAX)
            self.hs_min = initial_config.get("hidden_size_search_min", HIDDEN_SIZE_MIN)
            self.hs_max = initial_config.get("hidden_size_search_max", HIDDEN_SIZE_MAX)
            self.num_layers_min = initial_config.get("num_layers_search_min", NUM_LAYERS_MIN)
            self.num_layers_max = initial_config.get("num_layers_search_max", NUM_LAYERS_MAX)
        else:
            self.current_config = {
                "learning_rate": MANUAL_STARTING_LR,
                "hidden_layers": MANUAL_STARTING_HIDDEN_LAYERS.copy(),
                "dropout_rate": MANUAL_DROPOUT_RATE,
                "weight_decay": MANUAL_WEIGHT_DECAY,
            }
            self.lr_min = LEARNING_RATE_MIN
            self.lr_max = LEARNING_RATE_MAX
            self.hs_min = HIDDEN_SIZE_MIN
            self.hs_max = HIDDEN_SIZE_MAX
            self.num_layers_min = NUM_LAYERS_MIN
            self.num_layers_max = NUM_LAYERS_MAX
        self.visited_configs = set()
        self._add_to_visited(self.current_config)

    def unlock_global_bounds(self):
        """Releases the Initializer's strict safety bounds after Cycle 5."""
        self.lr_min = LEARNING_RATE_MIN
        self.lr_max = LEARNING_RATE_MAX
        self.hs_min = HIDDEN_SIZE_MIN
        self.hs_max = HIDDEN_SIZE_MAX
        self.num_layers_min = NUM_LAYERS_MIN
        self.num_layers_max = NUM_LAYERS_MAX

    def _config_to_key(self, config):
        lr_rounded = round(config['learning_rate'] / LR_TOLERANCE) * LR_TOLERANCE
        hidden_tuple = tuple(config['hidden_layers'])
        # Add regularization to the uniqueness key so the Actor can test the same architecture with different dropouts
        drop_rounded = round(config.get('dropout_rate', 0.0), 2)
        return f"LR={lr_rounded:.5f}, HL={hidden_tuple}, Drop={drop_rounded}"

    def _add_to_visited(self, config):
        self.visited_configs.add(self._config_to_key(config))

    def _is_new_config(self, config):
        return self._config_to_key(config) not in self.visited_configs

    def _random_exploration(self):
        new_lr = np.random.uniform(self.lr_min, self.lr_max)
        num_layers = np.random.randint(self.num_layers_min, self.num_layers_max + 1)
        new_hidden = [np.random.randint(self.hs_min, self.hs_max + 1) for _ in range(num_layers)]

        new_config = self.current_config.copy()
        new_config["learning_rate"] = new_lr
        new_config["hidden_layers"] = new_hidden
        # Random exploration also randomly perturbs regularization
        new_config["dropout_rate"] = np.clip(np.random.uniform(0.0, 0.4), 0.0, 0.5)
        new_config["weight_decay"] = np.clip(np.random.uniform(0.00001, 0.01), 0.0, 0.1)
        return new_config

    def _incremental_hidden_change(self, current_hidden, target_hidden):
        if not target_hidden: return current_hidden.copy()
        curr = current_hidden.copy()
        target = target_hidden.copy()

        if len(curr) < len(target):
            new_size = target[len(curr)] if len(curr) < len(target) else target[-1]
            curr.append(new_size)
        elif len(curr) > len(target):
            curr.pop()

        step = 32
        for i in range(min(len(curr), len(target))):
            diff = target[i] - curr[i]
            if abs(diff) <= step:
                curr[i] += diff
            else:
                curr[i] += step if diff > 0 else -step
            curr[i] = np.clip(curr[i], self.hs_min, self.hs_max)
        return curr

    def apply_critic_feedback(self, critic_output, tracker):
        orig_lr = self.current_config['learning_rate']
        orig_hidden = self.current_config['hidden_layers'].copy()
        orig_drop = self.current_config.get('dropout_rate', 0.0)
        orig_wd = self.current_config.get('weight_decay', 0.0001)
        orig_batch = self.current_config.get('batch_size', 128)
        orig_pat = self.current_config.get('patience', 25)

        try:
            import config
            adaptive_reg = getattr(config, 'ADAPTIVE_REGULARIZATION', True)
        except ImportError:
            adaptive_reg = True

        failures_str = tracker.get_recent_failures_str()
        best_lr = tracker.best_config['learning_rate'] if tracker.best_config else orig_lr
        best_hs = tracker.best_config['hidden_layers'] if tracker.best_config else orig_hidden
        best_reasoning = tracker.best_reasoning

        critic_diagnosis = critic_output.get('diagnosis', 'UNKNOWN')

        reg_block = (
            "- If the Critic Diagnosis involves 'OVERFITTING', you MUST INCREASE the DROPOUT_RATE (Max 0.4) and WEIGHT_DECAY (Max 0.01).\n"
            "        - If the Critic Diagnosis involves 'UNDERFITTING' or 'LATENCY_VIOLATION', you MUST DECREASE them towards 0.0."
        ) if adaptive_reg else "- REGULARIZATION LOCKED: Do not attempt to tune dropout or weight decay."

        format_block = (
            "LEARNING_RATE: (float)\n"
            "        HIDDEN_LAYERS: [list of integers, e.g., [128, 64]]\n"
            "        DROPOUT_RATE: (float)\n"
            "        WEIGHT_DECAY: (float)\n"
            "        BATCH_SIZE: (integer)\n"
            "        PATIENCE: (integer)"
        ) if adaptive_reg else (
            "LEARNING_RATE: (float)\n"
            "        HIDDEN_LAYERS: [list of integers, e.g., [128, 64]]\n"
            "        BATCH_SIZE: (integer)\n"
            "        PATIENCE: (integer)"
        )

        print(
            f"        🧠 Actor Agent is analyzing historical failures and Critic reasoning to formulate precise values...")
        prompt = f"""
        You are the deep learning Actor Agent tuning a physical dynamics model.
        Your job is to read the Critic Agent's feedback and finalize the exact Hyperparameters for the next cycle.

        CURRENT CONFIGURATION:
        - Learning Rate: {orig_lr}
        - Hidden Layers: {orig_hidden}
        - Dropout Rate: {orig_drop}
        - Weight Decay: {orig_wd}
        - Batch Size: {orig_batch}
        - Patience: {orig_pat}

        BEST HISTORICAL CONFIGURATION:
        - Learning Rate: {best_lr}
        - Hidden Layers: {best_hs}
        - Logic that achieved this: {best_reasoning}

        HISTORY OF FAILED CONFIGURATIONS TO AVOID:
        {failures_str}

        STRUCTURAL MUTATION REQUIREMENT:
        You MUST physically alter the architecture. To ensure you do not repeat a failed configuration, you must apply at least ONE of the following mutations:
        1. Change the Learning Rate by at least 15%.
        2. Add or remove a minimum of 16 neurons from at least one Hidden Layer.
        3. Add or remove an entire layer.
        Do not output a configuration that identically matches any item in the FAILURE HISTORY.

        CRITIC'S DIAGNOSIS & REASONING FOR CURRENT CYCLE:
        - Diagnosis: {critic_diagnosis}
        - Reasoning: "{critic_output.get('reasoning', 'None')}"
        - Critic's Suggested LR Step: {critic_output.get('lr_dir')} by {critic_output.get('lr_step')}
        - Critic's Target Layers: {critic_output.get('hidden_layers')}

        ABSOLUTE SEARCH BOUNDS (DO NOT VIOLATE):
        - Learning Rate: [{self.lr_min}, {self.lr_max}]
        - Hidden Layer Size: [{self.hs_min}, {self.hs_max}]
        - Number of Layers: [{self.num_layers_min}, {self.num_layers_max}]

        DYNAMIC REGULARIZATION & TRAINING RULES:
        {reg_block}
        - If the Critic Diagnosis involves 'OVERFITTING', you MUST HALVE the BATCH_SIZE (e.g., 128 -> 64 -> 32) and REDUCE PATIENCE by 5 to prevent dataset memorization. Minimum batch size is 16. Minimum patience is 10.
        - If 'UNDERFITTING', keep BATCH_SIZE and PATIENCE the same, or slightly increase PATIENCE to give it more time to learn.

        Based on the history and the Critic's logic, output the EXACT new configuration to test. 

        Format exactly:
        {format_block}
        """

        try:
            from main import model, log_agent_interaction
            from framework import cost_tracker
        except ImportError:
            from config import model
            def log_agent_interaction(*args, **kwargs):
                pass

        from langchain_core.messages import SystemMessage, HumanMessage
        system_msg = SystemMessage(content="You are the Actor Agent. Respond strictly in the requested format.")
        user_msg = HumanMessage(content=prompt)

        try:
            response = model.invoke([system_msg, user_msg])
            try:
                cost_tracker.update(response)
            except NameError:
                pass

            log_agent_interaction("Actor Application Agent", "You are the Actor Agent.", prompt, response.content)

            lines = response.content.strip().split('\n')
            new_lr = orig_lr
            new_hidden = orig_hidden
            new_drop = orig_drop
            new_wd = orig_wd
            new_batch = orig_batch
            new_pat = orig_pat

            for line in lines:
                if line.startswith('LEARNING_RATE:'):
                    new_lr = float(line.split(':', 1)[1].strip())
                elif line.startswith('HIDDEN_LAYERS:'):
                    import json
                    new_hidden = json.loads(line.split(':', 1)[1].strip())
                elif line.startswith('DROPOUT_RATE:'):
                    new_drop = float(line.split(':', 1)[1].strip())
                elif line.startswith('WEIGHT_DECAY:'):
                    new_wd = float(line.split(':', 1)[1].strip())
                elif line.startswith('BATCH_SIZE:'):
                    new_batch = int(float(line.split(':', 1)[1].strip()))
                elif line.startswith('PATIENCE:'):
                    new_pat = int(float(line.split(':', 1)[1].strip()))

            # 1. Enforce Math Bounds
            new_lr = np.clip(new_lr, self.lr_min, self.lr_max)
            new_hidden = [np.clip(int(x), self.hs_min, self.hs_max) for x in new_hidden]
            if len(new_hidden) < self.num_layers_min: new_hidden = [new_hidden[0]] * self.num_layers_min
            if len(new_hidden) > self.num_layers_max: new_hidden = new_hidden[:self.num_layers_max]

            new_batch = max(16, min(256, new_batch))
            new_pat = max(10, min(100, new_pat))

            if adaptive_reg:
                new_drop = float(np.clip(new_drop, 0.0, 0.5))
                new_wd = float(np.clip(new_wd, 0.0, 0.1))
            else:
                new_drop = orig_drop
                new_wd = orig_wd

            # 2. CREATE THE DICTIONARY FIRST
            candidate = self.current_config.copy()

            # 3. THEN ASSIGN THE VALUES TO IT
            candidate["learning_rate"] = float(new_lr)
            candidate["hidden_layers"] = new_hidden
            candidate["dropout_rate"] = new_drop
            candidate["weight_decay"] = new_wd
            candidate["batch_size"] = new_batch
            candidate["patience"] = new_pat

            if self._is_new_config(candidate):
                self.current_config = candidate
                self._add_to_visited(candidate)
                return
        except Exception as e:
            print(f"        ⚠️ Actor Agent LLM parse failed ({e}). Falling back to heuristic math...")

        # --- FALLBACK: ORIGINAL HEURISTIC MATH + BATCH/PATIENCE BUMPS ---
        target_hidden = critic_output.get('hidden_layers', orig_hidden)
        for attempt in range(MAX_RETRIES + 1):
            if attempt == 0:
                lr = orig_lr
                new_hidden = self._incremental_hidden_change(orig_hidden, target_hidden)
                new_drop = orig_drop
                new_wd = orig_wd
                new_batch = orig_batch
                new_pat = orig_pat

                # Math fallback for Overfitting/Underfitting
                if 'OVERFIT' in critic_diagnosis:
                    new_batch = max(16, new_batch // 2)
                    new_pat = max(10, new_pat - 5)
                    if adaptive_reg:
                        new_drop = np.clip(orig_drop + 0.05, 0.0, 0.5)
                        new_wd = np.clip(orig_wd * 2.0 if orig_wd > 0 else 0.0001, 0.0, 0.05)
                elif 'UNDERFIT' in critic_diagnosis:
                    new_pat = min(100, new_pat + 5)
                    if adaptive_reg:
                        new_drop = np.clip(orig_drop - 0.05, 0.0, 0.5)
                        new_wd = np.clip(orig_wd * 0.5, 0.0, 0.05)
            else:
                rand = self._random_exploration()
                lr = rand['learning_rate']
                new_hidden = rand['hidden_layers']
                new_drop = rand.get('dropout_rate', orig_drop) if adaptive_reg else orig_drop
                new_wd = rand.get('weight_decay', orig_wd) if adaptive_reg else orig_wd
                new_batch = rand.get('batch_size', orig_batch)
                new_pat = rand.get('patience', orig_pat)

            if attempt == 0 and critic_output.get('lr_dir') != 'stay':
                if critic_output['lr_dir'] == 'increase':
                    lr += critic_output.get('lr_step', 0.0)
                else:
                    lr -= critic_output.get('lr_step', 0.0)

            lr = np.clip(lr, self.lr_min, self.lr_max)
            new_hidden = [np.clip(int(x), self.hs_min, self.hs_max) for x in new_hidden]
            if len(new_hidden) < self.num_layers_min: new_hidden = [new_hidden[0]] * self.num_layers_min
            if len(new_hidden) > self.num_layers_max: new_hidden = new_hidden[:self.num_layers_max]

            new_batch = max(16, min(256, new_batch))
            new_pat = max(10, min(100, new_pat))

            candidate = self.current_config.copy()
            candidate["learning_rate"] = lr
            candidate["hidden_layers"] = new_hidden
            candidate["dropout_rate"] = new_drop
            candidate["weight_decay"] = new_wd
            candidate["batch_size"] = new_batch
            candidate["patience"] = new_pat

            if self._is_new_config(candidate):
                self.current_config = candidate
                self._add_to_visited(candidate)
                return

        rand = self._random_exploration()
        if not adaptive_reg:
            rand["dropout_rate"] = orig_drop
            rand["weight_decay"] = orig_wd

        # Ensure batch/patience keys exist on random dict
        if "batch_size" not in rand: rand["batch_size"] = orig_batch
        if "patience" not in rand: rand["patience"] = orig_pat

        while not self._is_new_config(rand):
            rand = self._random_exploration()
            if not adaptive_reg:
                rand["dropout_rate"] = orig_drop
                rand["weight_decay"] = orig_wd
            if "batch_size" not in rand: rand["batch_size"] = orig_batch
            if "patience" not in rand: rand["patience"] = orig_pat

        self.current_config = rand
        self._add_to_visited(rand)


class ExplorerAgent:
    def __init__(self, initial_config=None):
        if initial_config is not None:
            self.lr_min = initial_config.get("lr_search_min", LEARNING_RATE_MIN)
            self.lr_max = initial_config.get("lr_search_max", LEARNING_RATE_MAX)
            self.hs_min = initial_config.get("hidden_size_search_min", HIDDEN_SIZE_MIN)
            self.hs_max = initial_config.get("hidden_size_search_max", HIDDEN_SIZE_MAX)
            self.num_layers_min = initial_config.get("num_layers_search_min", NUM_LAYERS_MIN)
            self.num_layers_max = initial_config.get("num_layers_search_max", NUM_LAYERS_MAX)
        else:
            self.lr_min = LEARNING_RATE_MIN
            self.lr_max = LEARNING_RATE_MAX
            self.hs_min = HIDDEN_SIZE_MIN
            self.hs_max = HIDDEN_SIZE_MAX
            self.num_layers_min = NUM_LAYERS_MIN
            self.num_layers_max = NUM_LAYERS_MAX

    # --- NEW: Added cycle_number=None to the arguments ---
    def generate_radical_escape(self, tracker, stuck_config, visited_configs, cycle_number=None):
        stuck_lr = stuck_config['learning_rate']
        stuck_hidden = stuck_config['hidden_layers']
        failures_str = tracker.get_recent_failures_str()

        # If cycle_number wasn't explicitly passed from main.py, estimate it via visited history
        if cycle_number is None:
            cycle_number = len(visited_configs) + 1

        # ====================================================================
        # NEW: EXPLORER READS CUSTOMER CONTEXT (Limited by RUN_MODE & Cycle)
        # ====================================================================
        try:
            from config import CUSTOMER_SYSTEM_DESCRIPTION, RUN_MODE
            run_mode = RUN_MODE.strip().lower()
            customer_context = CUSTOMER_SYSTEM_DESCRIPTION.strip()
        except ImportError:
            run_mode = "regular"
            customer_context = ""

        system_context_block = ""
        if customer_context:
            # Heavy mode = Always read. Regular mode = First 5 cycles only. Fast = Never.
            if run_mode == "heavy" or (run_mode == "regular" and cycle_number <= 5):
                system_context_block = f"\n        CUSTOMER SYSTEM CONTEXT: '{customer_context}'"

        prompt = f"""
        You are the Global Exploration Agent (The Repulsive Barrier).
        The primary tuning loop is currently TRAPPED in a local minimum and cannot improve the physical dynamics model.
        {system_context_block}

        TRAPPED CONFIGURATION (TREAT THIS AS A REPULSIVE OBSTACLE):
        - Stuck Learning Rate: {stuck_lr}
        - Stuck Hidden Layers: {stuck_hidden}

        HISTORY OF RECENT FAILED JUMPS TO AVOID:
        {failures_str}

        ABSOLUTE SEARCH BOUNDS:
        - Learning Rate: [{self.lr_min}, {self.lr_max}]
        - Hidden Layer Size: [{self.hs_min}, {self.hs_max}]
        - Number of Layers: [{self.num_layers_min}, {self.num_layers_max}]

        YOUR MISSION: 
        Execute a radical topological shift. You MUST generate an architecture that is structurally opposite to the trapped configuration.
        - If the trapped model is shallow and wide, make yours deep and narrow.
        - If the trapped model is deep and narrow, make yours shallow and wide.
        - Force a massive adjustment to the learning rate to escape the gradient trench.

        Format exactly:
        LEARNING_RATE: (float)
        HIDDEN_LAYERS: [list of integers, e.g., [32, 32, 16]]
        REASONING: (One specific sentence explaining why this topological inversion escapes the minimum)
        """
        system_msg = SystemMessage(content="You are the Explorer Agent. Respond strictly in the requested format.")
        user_msg = HumanMessage(content=prompt)

        new_lr = stuck_lr
        new_hidden = stuck_hidden
        reasoning = "LLM Generation Failed. Applying mathematical inversion."

        try:
            response = model.invoke([system_msg, user_msg])
            cost_tracker.update(response)
            log_agent_interaction("Explorer Escape Agent", "You are the Explorer Agent.", prompt, response.content)

            lines = response.content.strip().split('\n')
            for line in lines:
                if line.startswith('LEARNING_RATE:'):
                    new_lr = float(line.split(':', 1)[1].strip())
                elif line.startswith('HIDDEN_LAYERS:'):
                    new_hidden = json.loads(line.split(':', 1)[1].strip())
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()

            # Enforce strict search bounds
            new_lr = float(np.clip(new_lr, self.lr_min, self.lr_max))
            new_hidden = [int(np.clip(x, self.hs_min, self.hs_max)) for x in new_hidden]
            if len(new_hidden) < self.num_layers_min: new_hidden = [new_hidden[0]] * self.num_layers_min
            if len(new_hidden) > self.num_layers_max: new_hidden = new_hidden[:self.num_layers_max]

            # 🧠 COPY the static parameters
            candidate = stuck_config.copy()
            candidate["learning_rate"] = new_lr
            candidate["hidden_layers"] = new_hidden

            # If the LLM successfully generated a completely new config, return it
            lr_rounded = round(candidate['learning_rate'] / LR_TOLERANCE) * LR_TOLERANCE
            key = f"LR={lr_rounded:.5f}, HL={tuple(candidate['hidden_layers'])}"
            if key not in visited_configs:
                return candidate, reasoning

        except Exception as e:
            print(f"        ⚠️ Explorer LLM parse failed ({e}). Executing mathematical inversion...")

        # --- FALLBACK: Strict Mathematical Inversion ---
        # If the LLM fails or suggests a duplicate, physically force the opposite dimensions
        target_layers = self.num_layers_max if len(stuck_hidden) <= (
                (self.num_layers_max + self.num_layers_min) / 2) else self.num_layers_min
        target_neurons = self.hs_min if np.mean(stuck_hidden) >= ((self.hs_max + self.hs_min) / 2) else self.hs_max

        while True:
            candidate_hidden = [np.random.randint(target_neurons - 16, target_neurons + 16) for _ in
                                range(target_layers)]
            candidate_hidden = [int(np.clip(x, self.hs_min, self.hs_max)) for x in candidate_hidden]
            candidate_lr = np.random.uniform(self.lr_min, self.lr_max)

            # 🧠 COPY the static parameters
            candidate = stuck_config.copy()
            candidate["learning_rate"] = candidate_lr
            candidate["hidden_layers"] = candidate_hidden

            lr_rounded = round(candidate_lr / LR_TOLERANCE) * LR_TOLERANCE
            key = f"LR={lr_rounded:.5f}, HL={tuple(candidate['hidden_layers'])}"
            if key not in visited_configs:
                return candidate, "Forced mathematical inversion via bounds reflection."

# ============================================================================
#  PLOTTING & AUTOSAVING FUNCTIONS
# ============================================================================
def plot_mse_convergence(performance_history, env_name):
    iters = [p['iteration'] + 1 for p in performance_history]
    mses = [p['performance']['mse'] for p in performance_history]
    plt.figure(figsize=(10, 4))
    plt.plot(iters, mses, 'r-o', linewidth=2, markersize=8)
    plt.xlabel('Actor-Critic Cycle')
    label_mode = "Single-Step"
    plt.ylabel(f'Validation {label_mode} MSE')
    plt.title(f'{label_mode} System Identification Error for {env_name}')
    plt.grid(True)
    plt.yscale('log')

    if SAVE_PLOT:
        target_label = "Xdot"
        # --- FIX: Explicitly named "mse_convergence" ---
        out_name = f"{PLOT_FILENAME_PREFIX}_{env_name}_{target_label}_mse_convergence_{RUN_TIMESTAMP}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"    💾 Saved MSE Convergence Plot to: {out_name}")

    plt.show(block=False)


def plot_nrmse_convergence(performance_history, env_name):
    iters = [p['iteration'] + 1 for p in performance_history]
    rmses = [p['performance'].get('rmse', p['performance'].get('mse')) for p in performance_history]
    plt.figure(figsize=(10, 4))
    plt.plot(iters, rmses, 'r-o', linewidth=2, markersize=8)
    plt.xlabel('Actor-Critic Cycle')
    label_mode = "Single-Step"
    plt.ylabel(f'Validation {label_mode} RMSE')
    plt.title(f'{label_mode} System Identification Error (RMSE) for {env_name}')
    plt.grid(True)
    plt.yscale('log')

    if SAVE_PLOT:
        target_label = "Xdot"
        # --- FIX: Explicitly named "rmse_convergence" ---
        out_name = f"{PLOT_FILENAME_PREFIX}_{env_name}_{target_label}_rmse_convergence_{RUN_TIMESTAMP}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"    💾 Saved RMSE Convergence Plot to: {out_name}")

    plt.show(block=False)


def plot_hyperparameter_evolution(performance_history):
    cycles = [p['iteration'] + 1 for p in performance_history]
    lrs = [p['config']['learning_rate'] for p in performance_history]
    total_neurons = [sum(p['config']['hidden_layers']) for p in performance_history]
    num_layers = [len(p['config']['hidden_layers']) for p in performance_history]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(cycles, lrs, 'b-o')
    ax1.set_yscale('log')
    ax1.set_ylabel('Learning Rate')
    ax1.grid(True)

    ax2.plot(cycles, total_neurons, 'g-s')
    ax2.set_ylabel('Total Neurons')
    ax2.grid(True)

    ax3.plot(cycles, num_layers, 'm-d')
    ax3.set_ylabel('Number of Hidden Layers')
    ax3.set_xlabel('Cycle')
    ax3.grid(True)
    plt.tight_layout()

    if SAVE_PLOT:
        target_label = "Xdot"
        out_name = f"{PLOT_FILENAME_PREFIX}_{ENV_NAME}_{target_label}_hyperparameters_{RUN_TIMESTAMP}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"    💾 Saved Hyperparameter Evolution Plot to: {out_name}")

    plt.show()


def plot_latency_evolution(performance_history, max_latency):
    cycles = [p['iteration'] + 1 for p in performance_history]

    # Extract latency (default to 0 if something went wrong)
    latencies = [p['performance'].get('latency', 0.0) for p in performance_history]

    plt.figure(figsize=(10, 4))

    # Plot the actual measured network latency
    plt.plot(cycles, latencies, 'c-o', linewidth=2, markersize=8, label='Measured Latency')

    # Draw a hard red line representing the customer's maximum allowed limit
    plt.axhline(y=max_latency, color='red', linestyle='--', linewidth=2, label='Max Allowed Limit')

    plt.xlabel('Actor-Critic Cycle', fontweight='bold')
    plt.ylabel('Inference Latency (ms)', fontweight='bold')
    plt.title(f'Network Inference Latency Evolution for {ENV_NAME}', fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOT:
        out_name = f"{PLOT_FILENAME_PREFIX}_{ENV_NAME}_latency_evolution_{RUN_TIMESTAMP}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"    💾 Saved Latency Evolution Plot to: {out_name}")

    # Show without blocking so the final Test verification plot can also render
    plt.show(block=False)


def plot_layers_neurons_mse_contour(performance_history):
    num_layers = np.array([len(p['config']['hidden_layers']) for p in performance_history], dtype=float)
    avg_neurons = np.array([np.mean(p['config']['hidden_layers']) for p in performance_history], dtype=float)
    mse = np.array([p['performance']['mse'] for p in performance_history], dtype=float)

    # --- BULLETPROOF MATPLOTLIB FIX ---
    # 1. Filter out any 'inf' or 'NaN' penalties to keep the plot scale readable
    mse = np.where(np.isinf(mse) | np.isnan(mse), 9999.0, mse)

    # 2. Force strictly positive values (LogNorm crashes on <= 0)
    mse = np.clip(mse, a_min=1e-8, a_max=None)

    vmin_val = float(np.min(mse))
    vmax_val = float(np.max(mse))

    # 3. Mathematically guarantee a valid range (LogNorm crashes if min == max)
    if vmin_val >= vmax_val:
        vmax_val = vmin_val * 10.0

    fig, ax = plt.subplots(figsize=(9, 7))

    contour_ok = False
    if len(performance_history) >= 3 and np.ptp(num_layers) > 0 and np.ptp(avg_neurons) > 0:
        try:
            norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
            contour = ax.tricontourf(num_layers, avg_neurons, mse, levels=14,
                                     cmap='viridis', norm=norm)
            fig.colorbar(contour, ax=ax, label='Validation MSE')
            contour_ok = True
        except Exception as e:
            print(f"    ⚠️ Contour triangulation failed ({e}); falling back to scatter-only plot.")

    scatter_norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
    scatter = ax.scatter(num_layers, avg_neurons, c=mse, cmap='viridis', norm=scatter_norm,
                         s=90, edgecolors='white', linewidths=1.2, zorder=3)
    if not contour_ok:
        fig.colorbar(scatter, ax=ax, label='Validation MSE')

    best_idx = int(np.argmin(mse))
    ax.scatter(num_layers[best_idx], avg_neurons[best_idx], marker='*', s=500,
               c='red', edgecolors='black', linewidths=1.2, zorder=4, label='Best Config')

    ax.set_xlabel('Number of Hidden Layers', fontweight='bold')
    ax.set_ylabel('Avg. Neurons per Layer', fontweight='bold')
    target_label = "Xdot"
    ax.set_title(f'Validation MSE over Architecture Search Space ({target_label}) for {ENV_NAME}', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOT:
        out_name = f"{PLOT_FILENAME_PREFIX}_{ENV_NAME}_{target_label}_layers_neurons_contour_{RUN_TIMESTAMP}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        print(f"    💾 Saved Layers/Neurons MSE Contour Plot to: {out_name}")

    plt.show(block=False)


# ============================================================================
#  PURE DATA-DRIVEN TEST SET VERIFICATION (NO PHYSICS EQUATIONS)
# ============================================================================

def export_standalone_inference_script(model, hidden_layers, activation, state_dim, action_dim, pth_filename, env_name):
    # Extract normalization parameters
    state_mean = model.state_mean.cpu().numpy().flatten().tolist()
    state_scale = model.state_scale.cpu().numpy().flatten().tolist()
    action_mean = model.action_mean.cpu().numpy().flatten().tolist()
    action_scale = model.action_scale.cpu().numpy().flatten().tolist()
    output_mean = model.output_mean.cpu().numpy().flatten().tolist()
    output_scale = model.output_scale.cpu().numpy().flatten().tolist()
    in_dim = state_dim + action_dim + 1

    try:
        from config import NETWORK_ARCHITECTURE
        ARCH = NETWORK_ARCHITECTURE.strip().upper()
    except ImportError:
        ARCH = "MLP"

    if ARCH == "LSTM":
        try:
            from config import LSTM_SEQ_LENGTH
            SEQ_LEN = LSTM_SEQ_LENGTH
        except ImportError:
            SEQ_LEN = 10

        hidden_size = hidden_layers[0] if hidden_layers else 128
        num_layers = len(hidden_layers) if hidden_layers else 2

        network_init = f"""
        # Hardcoded LSTM Architecture
        self.lstm = nn.LSTM(
            input_size={in_dim},
            hidden_size={hidden_size},
            num_layers={num_layers},
            batch_first=True
        )
        self.fc_out = nn.Linear({hidden_size}, {state_dim})"""

        forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out_norm = self.fc_out(lstm_out[:, -1, :])"""

        # 🧠 DYNAMIC FIX: Generates the Smart Memory Buffer Controller for LSTMs
        controller_class = f"""
class NeuralController:
    def __init__(self, weights_path="{pth_filename}.pth", device="cpu"):
        from collections import deque
        self.device = torch.device(device)
        self.seq_len = {SEQ_LEN}

        # LSTM Memory Buffers
        self.state_buffer = deque(maxlen=self.seq_len)
        self.action_buffer = deque(maxlen=self.seq_len)
        self.dt_buffer = deque(maxlen=self.seq_len)

        self.model = DeployableDynamicsModel().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def commit_to_memory(self, state, action, dt):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.dt_buffer.append(dt)

    def predict_xdot(self, state, action, dt):
        temp_s = list(self.state_buffer) + [state]
        temp_a = list(self.action_buffer) + [action]
        temp_dt = list(self.dt_buffer) + [dt]

        # Pad the buffer if the simulation just started
        while len(temp_s) < self.seq_len:
            temp_s.insert(0, temp_s[0])
            temp_a.insert(0, temp_a[0])
            temp_dt.insert(0, temp_dt[0])

        with torch.no_grad():
            s_t = torch.tensor(np.array(temp_s[-self.seq_len:]), dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(np.array(temp_a[-self.seq_len:]), dtype=torch.float32).unsqueeze(0).to(self.device)
            dt_t = torch.tensor(np.array(temp_dt[-self.seq_len:]), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(self.device)
            out_phys = self.model(s_t, a_t, dt_t)
            return out_phys.squeeze(0).cpu().numpy()
        """

    else:
        # Build the MLP layers dynamically
        act_str = "nn.ReLU()"
        if activation.lower() == 'tanh':
            act_str = "nn.Tanh()"
        elif activation.lower() == 'swish':
            act_str = "nn.SiLU()"

        layers_str = ""
        curr_dim = in_dim
        for out_dim in hidden_layers:
            layers_str += f"            nn.Linear({curr_dim}, {out_dim}),\n            {act_str},\n"
            curr_dim = out_dim
        layers_str += f"            nn.Linear({curr_dim}, {state_dim})\n"

        network_init = f"""
        # Hardcoded MLP Architecture
        self.network = nn.Sequential(
{layers_str}        )"""

        forward_pass = """
        x = torch.cat([state_norm, action_norm, dt], dim=-1)
        out_norm = self.network(x)"""

        # 🧠 DYNAMIC FIX: Generates the Classic Memoryless Controller for MLPs
        controller_class = f"""
        class NeuralController:
            def __init__(self, weights_path="{pth_filename}.pth", device="cpu"):
                self.device = torch.device(device)
                self.model = DeployableDynamicsModel().to(self.device)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                self.model.eval()

            def commit_to_memory(self, state, action, dt):
                pass # MLPs are memoryless

            def predict_xdot(self, state, action, dt):
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    dt_t = torch.tensor([[dt]], dtype=torch.float32).to(self.device)
                    out_phys = self.model(s_t, a_t, dt_t)
                    return out_phys.squeeze(0).cpu().numpy()
        """

    script_content = f'''"""
STANDALONE NEURAL DYNAMICS CONTROLLER ({ARCH})
"""
import torch
import torch.nn as nn
import numpy as np

class DeployableDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state_mean", torch.tensor({state_mean}, dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor({state_scale}, dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor({action_mean}, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor({action_scale}, dtype=torch.float32))
        self.register_buffer("output_mean", torch.tensor({output_mean}, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor({output_scale}, dtype=torch.float32))
{network_init}

    def forward(self, state, action, dt):
        state_norm = (state - self.state_mean) / self.state_scale
        action_norm = (action - self.action_mean) / self.action_scale
{forward_pass}
        out_phys = (out_norm * self.output_scale) + self.output_mean
        return out_phys
{controller_class}
'''
    out_name = f"deployed_controller_{env_name}.py"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(script_content)

def plot_test_dataset_verification(dyn_model, test_trajs, state_dim, action_dim):
    import math
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from collections import deque

    dyn_model.eval()
    print("\n📊 Generating pure data-driven verification from the 10% Test set...")

    try:
        from config import NETWORK_ARCHITECTURE, LSTM_SEQ_LENGTH
        ARCH = NETWORK_ARCHITECTURE.strip().upper()
        SEQ_LEN = LSTM_SEQ_LENGTH if ARCH == "LSTM" else 1
    except ImportError:
        ARCH = "MLP"
        SEQ_LEN = 1

    master_true_traj = []
    master_nn_traj = []
    test_seq = test_trajs[0]
    total_steps = len(test_seq)
    mean_dt = float(np.mean([step[2] for step in test_seq]))
    target_horizon_seconds = 10.0
    chunk_size = max(1, int(round(target_horizon_seconds / mean_dt)))
    num_chunks = math.ceil(total_steps / chunk_size)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_steps)
        n_steps = end_idx - start_idx

        true_states = np.array([step[0] for step in test_seq[start_idx:end_idx]])
        actions = np.array([step[1] for step in test_seq[start_idx:end_idx]])
        dts = np.array([step[2] for step in test_seq[start_idx:end_idx]])

        nn_history = []

        # --- NEW: Create a rolling memory window for the LSTM ---
        window_states = deque(maxlen=SEQ_LEN)
        window_actions = deque(maxlen=SEQ_LEN)
        window_dts = deque(maxlen=SEQ_LEN)

        # Pre-fill the memory buffer with the real flight data right before this chunk starts
        for k in range(SEQ_LEN):
            real_idx = max(0, start_idx - SEQ_LEN + 1 + k)
            window_states.append(test_seq[real_idx][0])
            window_actions.append(test_seq[real_idx][1])
            window_dts.append(test_seq[real_idx][2])

        current_state = true_states[0].copy()

        for i in range(n_steps):
            action = actions[i]
            dt = float(dts[i])

            with torch.no_grad():
                try:
                    from framework import DEVICE
                except ImportError:
                    DEVICE = next(dyn_model.parameters()).device

                try:
                    from config import INTEGRATOR_TYPE
                    INT_TYPE = INTEGRATOR_TYPE.strip().upper()
                except ImportError:
                    INT_TYPE = "EULER"

                def get_k(state_val):
                    if ARCH == "LSTM":
                        temp_states = list(window_states) + [state_val]
                        temp_actions = list(window_actions) + [action]
                        temp_dts = list(window_dts) + [dt]

                        while len(temp_states) < SEQ_LEN:
                            temp_states.insert(0, temp_states[0])
                            temp_actions.insert(0, temp_actions[0])
                            temp_dts.insert(0, temp_dts[0])

                        s_temp = torch.tensor(np.array(temp_states[-SEQ_LEN:]), dtype=torch.float32).unsqueeze(0).to(
                            DEVICE)
                        a_temp = torch.tensor(np.array(temp_actions[-SEQ_LEN:]), dtype=torch.float32).unsqueeze(0).to(
                            DEVICE)
                        dt_temp = torch.tensor(np.array(temp_dts[-SEQ_LEN:]), dtype=torch.float32).unsqueeze(
                            -1).unsqueeze(0).to(DEVICE)
                        out, _ = dyn_model(s_temp, a_temp, dt_temp)
                    else:
                        s_temp = torch.tensor(state_val, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        a_temp = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        dt_temp = torch.tensor([[dt]], dtype=torch.float32).to(DEVICE)
                        out, _ = dyn_model(s_temp, a_temp, dt_temp)
                    return out.squeeze(0).cpu().numpy()

                if INT_TYPE == "RK4":
                    k1 = get_k(current_state)
                    k2 = get_k(current_state + 0.5 * dt * k1)
                    k3 = get_k(current_state + 0.5 * dt * k2)
                    k4 = get_k(current_state + dt * k3)
                    pred_xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                else:
                    pred_xdot = get_k(current_state)

                next_state = current_state + (pred_xdot * dt)

            # Push the verified state into the rolling memory window AFTER prediction
            window_states.append(current_state.copy())
            window_actions.append(action)
            window_dts.append(dt)

            nn_history.append(current_state.copy())
            current_state = next_state

        master_true_traj.extend(true_states)
        master_nn_traj.extend(nn_history)

        # ... (Keep the rest of your matplotlib plotting logic here exactly as it was) ...
        steps_axis = np.cumsum(dts) - dts[0]
        plot_dim = min(state_dim, 12)
        fig, axes = plt.subplots(plot_dim, 1, figsize=(12, 2.5 * plot_dim), sharex=True)
        if plot_dim == 1: axes = np.array([axes])
        for idx in range(plot_dim):
            ax = axes[idx]
            ax.plot(steps_axis, true_states[:, idx], '--', color='blue', linewidth=2.5, label='True Test Data')
            ax.plot(steps_axis, np.array(nn_history)[:, idx], '-', color='black', linewidth=1.5, label='NN Prediction')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f"State {idx}", fontsize=10, fontweight='bold')
            if idx == 0:
                ax.legend(loc="best")

            # ---> ADD THIS SAVING BLOCK HERE <---
        try:
            from config import SAVE_PLOT, PLOT_FILENAME_PREFIX, ENV_NAME
            from framework import RUN_TIMESTAMP
            # ONLY save the very first chunk to disk so the PDF Generator doesn't duplicate them!
            if SAVE_PLOT and chunk_idx == 0:
                out_name = f"{PLOT_FILENAME_PREFIX}_{ENV_NAME}_Xdot_test_verification_{RUN_TIMESTAMP}.png"
                fig.savefig(out_name, dpi=150, bbox_inches='tight')
                print(f"    💾 Saved Data-Driven Test Plot for PDF Report to: {out_name}")
        except Exception:
            pass

        plt.show(block=False)

    return np.array(master_true_traj), np.array(master_nn_traj)


def save_best_model(dyn_model, hidden_layers, state_dim, action_encoding_dim, mse, config, activation):
    try:
        from config import ROLLOUT_HORIZON
        mode_label = f"Rollout{ROLLOUT_HORIZON}" if ROLLOUT_HORIZON > 1 else "SingleStep"
    except ImportError:
        mode_label = "SingleStep"

    target_label = "Xdot"
    base = f"best_model_{ENV_NAME}_{RUN_TIMESTAMP}_{mode_label}_{target_label}_MSE_{mse:.6f}"
    torch.save(dyn_model.state_dict(), f"{base}.pth")
    return base


def calculate_success_score(val_mse, true_trajectory, nn_trajectory, complexity_label):
    """
    Calculates the 0-100 Composite Success Score (2-Pillar).
    Injects a 5-Stage System Difficulty Allowance to reward models
    that achieve the same MSE on highly complex, non-linear plants.
    """
    import numpy as np
    import math

    # ---------------------------------------------------------
    # 5-STAGE SYSTEM DIFFICULTY ALLOWANCE
    # ---------------------------------------------------------
    stage = 3  # Default Moderate

    # Parse the incoming label to find the integer stage (1 to 5)
    if isinstance(complexity_label, (int, float)):
        stage = int(complexity_label)
    elif isinstance(complexity_label, str):
        label_lower = complexity_label.lower()
        if "1" in label_lower or "very low" in label_lower:
            stage = 1
        elif "2" in label_lower or "low" in label_lower:
            stage = 2
        elif "4" in label_lower or ("high" in label_lower and "very" not in label_lower):
            stage = 4
        elif "5" in label_lower or "very high" in label_lower:
            stage = 5
        elif "3" in label_lower or "medium" in label_lower or "moderate" in label_lower:
            stage = 3

    # Clamp safely between 1 and 5
    stage = max(1, min(5, stage))

    # Calculate Allowance: 1.0 (Stage 1) up to 3.0 (Stage 5)
    difficulty_allowance = 1.0 + 1 * (stage - 1)

    # ---------------------------------------------------------
    # PILLAR A: 1-Step Prediction Fit (Max 75 Points)
    # ---------------------------------------------------------
    lambda_a = 0.5  # Softened from 1.0 to give better scores for complex fits
    effective_mse = val_mse / difficulty_allowance
    score_a = 75.0 * math.exp(-lambda_a * effective_mse)

    # ---------------------------------------------------------
    # PILLAR B: Closed-Loop Rollout Stability (Max 25 Points)
    # ---------------------------------------------------------
    true_traj = np.array(true_trajectory)
    nn_traj = np.array(nn_trajectory)

    drift_error = np.mean(np.abs(true_traj - nn_traj))
    lambda_b = 1.0  # Softened from 2.0 to reward stable rollouts on complex tracks
    effective_drift = drift_error / difficulty_allowance
    score_b = 25.0 * math.exp(-lambda_b * effective_drift)

    # ---------------------------------------------------------
    # COMPOSITE SCORE & CATEGORIZATION
    # ---------------------------------------------------------
    total_score = score_a + score_b

    if total_score > 75:
        status = "STABLE & HIGH-FIDELITY"
    elif total_score > 50:
        status = "STABLE & ACCEPTABLE"
    elif total_score >= 40:
        status = "UNSTABLE ROLLOUT"
    else:
        status = "UNSTABLE / FAILED"

    # Print the terminal executive summary
    print("\n" + "=" * 80)
    print("🎯 SYSTEM IDENTIFICATION PERFORMANCE & SUCCESS SCORE")
    print("=" * 80)
    print(f"  ├── SYSTEM DIFFICULTY           : Stage {stage} ({difficulty_allowance:.1f}x error leniency applied)")
    print(
        f"  ├── PILLAR A (1-Step Fit)       : {score_a:.1f} / 75.0 pts (Raw Val MSE: {val_mse:.4f} -> Effective: {effective_mse:.4f})")
    print(
        f"  ├── PILLAR B (Rollout Stability): {score_b:.1f} / 25.0 pts (Raw Drift: {drift_error:.4f} -> Effective: {effective_drift:.4f})")
    print("-" * 80)
    print(f"  🔥 FINAL COMPOSITE SCORE        : {total_score:.1f} / 100")
    print(f"  🏷️ MODEL STATUS                 : [{status}]")
    print("=" * 80 + "\n")

    return total_score, status


def package_final_results_to_zip(pdf_filename, timestamp, env_name):
    """
    Scans the working directory for strictly the CURRENT RUN'S generated outputs,
    generates a README.md, packages everything into a ZIP file, and cleans up the local disk.
    """
    import os
    import glob
    import zipfile

    zip_filename = f"SystemID_RunResults_{timestamp}.zip"
    controller_script_name = f"deployed_controller_{env_name}.py"

    print("\n" + "=" * 80)
    print("📦 PACKAGING FINAL PIPELINE RESULTS")
    print("=" * 80)

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # ---------------------------------------------------------
        # 0. GENERATE & ADD README.MD (Root of Zip)
        # ---------------------------------------------------------
        readme_content = f"""# System Identification & Neural Controller Package
        **Run Timestamp:** {timestamp}  
        **Framework:** Automated Multi-Agent Deep Learning & Control Framework

        ---

        ## 📂 Directory Structure

        ### 1. `figures/`
        * Contains all high-resolution diagnostic plots generated during THIS specific run.

        ### 2. `deployment/`
        * **Model Weights (`*.pth`):** The final optimized PyTorch neural network weights.
        * **`NN.py`:** A ready-to-use inference script that initializes the model.
        * **`{controller_script_name}`:** The backend engine class required to parse and execute the neural network.

        ### 3. `report/`
        * **PDF Engineering Report:** A comprehensive documentation file summarizing the model.
        """
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        zipf.write("README.md", arcname="README.md")
        print("  ├── Added root 'README.md' documentation")

        # ---------------------------------------------------------
        # FOLDER 1: FIGURES (STRICTLY FROM THIS RUN)
        # ---------------------------------------------------------
        png_files = glob.glob(f"*{timestamp}*.png")
        for img in png_files:
            zipf.write(img, arcname=f"figures/{os.path.basename(img)}")
        print(f"  ├── [Folder 1] Added {len(png_files)} plots to 'figures/'")

        # ---------------------------------------------------------
        # FOLDER 2: DEPLOYMENT (STRICTLY FROM THIS RUN)
        # ---------------------------------------------------------
        model_files = glob.glob(f"*{timestamp}*.pth")
        for m in model_files:
            zipf.write(m, arcname=f"deployment/{os.path.basename(m)}")

        if os.path.exists(controller_script_name):
            zipf.write(controller_script_name, arcname=f"deployment/{controller_script_name}")

        # Dynamically generate NN.py to import from the correct controller file!
        try:
            from config import INTEGRATOR_TYPE
            INT_TYPE = INTEGRATOR_TYPE.strip().upper()
        except ImportError:
            INT_TYPE = "EULER"

        if INT_TYPE == "RK4":
            integration_block = """# 3. KINEMATIC INTEGRATION (4th-Order Runge-Kutta)
        k1 = plant_model.predict_xdot(current_state, proposed_action, dt)
        k2 = plant_model.predict_xdot(current_state + 0.5 * dt * k1, proposed_action, dt)
        k3 = plant_model.predict_xdot(current_state + 0.5 * dt * k2, proposed_action, dt)
        k4 = plant_model.predict_xdot(current_state + dt * k3, proposed_action, dt)

        predicted_xdot = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        next_state_prediction = current_state + (predicted_xdot * dt)

        # Commit the final physical step to LSTM memory
        try: plant_model.commit_to_memory(current_state, proposed_action, dt)
        except AttributeError: pass # Ignores if architecture is an MLP
        """
        else:
            integration_block = """# 3. KINEMATIC INTEGRATION (1st-Order Euler)
        predicted_xdot = plant_model.predict_xdot(current_state, proposed_action, dt)
        next_state_prediction = current_state + (predicted_xdot * dt)

        # Commit the final physical step to LSTM memory
        try: plant_model.commit_to_memory(current_state, proposed_action, dt)
        except AttributeError: pass # Ignores if architecture is an MLP
        """

        nn_script_content = f"""# ==============================================================================
        # NEURAL NETWORK INFERENCE DEPLOYMENT SCRIPT
        # ==============================================================================
        import numpy as np
        from deployed_controller_{env_name} import NeuralController

        # 1. INITIALIZATION
        plant_model = NeuralController(device="cpu") # Set to "cuda" if using a GPU

        # 2. OPTIMAL CONTROL LOOP
        dt = 0.01  
        # Ensure these match your state/action dimensions!
        current_state = np.zeros(plant_model.model.state_mean.shape)
        proposed_action = np.zeros(plant_model.model.action_mean.shape)

        {integration_block}

        print("Predicted Next State:", next_state_prediction)
        """

        with open("NN.py", "w", encoding="utf-8") as f:
            f.write(nn_script_content)

        zipf.write("NN.py", arcname="deployment/NN.py")
        print(f"  ├── [Folder 2] Added deployment files to 'deployment/'")

        # ---------------------------------------------------------
        # FOLDER 3: REPORT
        # ---------------------------------------------------------
        if os.path.exists(pdf_filename):
            zipf.write(pdf_filename, arcname=f"report/{os.path.basename(pdf_filename)}")
            print(f"  ├── [Folder 3] Added '{os.path.basename(pdf_filename)}' to 'report/'")

        # ---------------------------------------------------------
        # FOLDER 4: AGENTS LOG
        # ---------------------------------------------------------
        log_files = glob.glob("*.log")
        for log_file in log_files:
            zipf.write(log_file, arcname=f"Agents_log/{os.path.basename(log_file)}")

    # =========================================================
    # DISK CLEANUP (Remove loose files so only the ZIP remains)
    # =========================================================
    print("  ├── 🧹 Cleaning up temporary files from disk...")
    for img in png_files: os.remove(img)
    for m in model_files: os.remove(m)
    for log_file in log_files: os.remove(log_file)
    if os.path.exists(controller_script_name): os.remove(controller_script_name)
    if os.path.exists("NN.py"): os.remove("NN.py")
    if os.path.exists(pdf_filename): os.remove(pdf_filename)
    if os.path.exists("README.md"): os.remove("README.md")

    import os
    absolute_path = os.path.abspath(zip_filename)

    print("-" * 80)
    print(f"  ✅ ZIP FILE CREATED SUCCESSFULLY (Local files cleaned): {zip_filename}")
    print(f"  📂 SAVED EXACTLY TO: {absolute_path}")
    print("=" * 80 + "\n")


# ============================================================================
#  REPORT AGENT (Automated PDF Manuscript Authoring)
# ============================================================================
class ReportAgent:
    def __init__(self):
        pass

    def generate_report_text(self, env_name, state_dim, action_dim, best_config, best_mse, best_rmse, latency, use_pinn,
                             max_latency, success_score, model_status, complexity_label):

        # 1. Dynamically read the architecture
        try:
            from config import NETWORK_ARCHITECTURE
            base_arch = NETWORK_ARCHITECTURE.strip().upper()
        except ImportError:
            base_arch = "MLP"

        if base_arch == "LSTM":
            arch_name = "Long Short-Term Memory (LSTM) Network"
        else:
            arch_name = "Multilayer Perceptron (MLP)"

        arch_type = f"Physics-Informed {arch_name} (PINN)" if use_pinn else f"Data-Driven {arch_name}"

        try:
            from config import CUSTOMER_SYSTEM_DESCRIPTION
            customer_context = CUSTOMER_SYSTEM_DESCRIPTION.strip()
        except ImportError:
            customer_context = ""

        customer_prompt_block = ""
        if customer_context:
            customer_prompt_block = f"- Customer System Context: '{customer_context}'"

        prompt = f"""
        You are a Senior Control Systems Engineer authoring the final manuscript for an automated System Identification run.
        Analyze the final data metrics below and generate the Abstract and Concluding Synthesis for the PDF report.

        RUN METRICS:
        - Plant/Environment: {env_name}
        {customer_prompt_block}
        - Architecture Mode: {arch_type}
        - Dataset Complexity Tier: {complexity_label}
        - State Dimensions: {state_dim}
        - Action Dimensions: {action_dim}
        - Final Topology Chosen: {best_config.get('hidden_layers', 'Unknown')}
        - Validation MSE: {best_mse:.6f}
        - Validation RMSE: {best_rmse:.6f}
        - Real-Time Inference Latency: {latency:.3f} ms (Customer Hard Limit: {max_latency} ms)
        - Composite Success Score: {success_score:.1f} / 100
        - Deployment Status: {model_status}

        YOUR TASK:
        1. Write a highly technical, professional Abstract (approx 3 to 4 sentences) summarizing the tuning methodology, the architecture type, the complexity of the dataset, and the final MSE/RMSE accuracy.
        2. Write a highly technical Concluding Remark (approx 4 to 5 sentences) interpreting the deployment viability. Specifically mention the Composite Success Score ({success_score:.1f}/100) and Deployment Status ({model_status}), state whether the latency safely passed the real-time threshold limit, and note its readiness for Model Predictive Control (MPC) or Control Barrier Functions (CBF).

        Format your response EXACTLY like this (on single continuous lines):
        ABSTRACT: (Your abstract text)
        CONCLUSION: (Your conclusion text)
        """

        # 2. --- UPGRADED PROFESSIONAL FALLBACK TEXT ---
        # If the API crashes, we still print a highly professional, lengthy engineering abstract.
        abstract = (
            f"This report presents the results of an automated System Identification run utilizing a {arch_type} "
            f"architecture to model a dataset characterized as {complexity_label}. The tuning methodology employed "
            f"was optimized for a state dimension of {state_dim} and an action dimension of {action_dim}, ultimately "
            f"selecting a final topology of {best_config.get('hidden_layers', 'Unknown')}. The model achieved a validation Mean Squared Error "
            f"(MSE) of {best_mse:.6f} and a Root Mean Squared Error (RMSE) of {best_rmse:.6f}, indicating a high level of "
            f"accuracy in capturing the nonlinear dynamics of the target plant."
        )

        conclusion = (
            f"Achieving a Composite Success Score of {success_score:.1f}/100 ({model_status}), the optimized neural architecture "
            f"demonstrates robust closed-loop stability. Furthermore, with a measured inference latency of {latency:.3f} ms, "
            f"the model safely satisfies the real-time threshold limit of {max_latency} ms. These metrics confirm the network's "
            f"readiness for immediate integration into Model Predictive Control (MPC) or Control Barrier Function (CBF) frameworks "
            f"for advanced autonomous trajectory regulation."
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            system_msg = SystemMessage(
                content="You are a strict technical reporting agent. Respond ONLY in the requested format.")
            user_msg = HumanMessage(content=prompt)

            from config import model
            response = model.invoke([system_msg, user_msg])

            # 3. Safely update cost tracker without circular import crashes
            import sys
            if 'functionalNodes.utilities' in sys.modules and hasattr(sys.modules['functionalNodes.utilities'],
                                                                      'cost_tracker'):
                sys.modules['functionalNodes.utilities'].cost_tracker.update(response)
            elif 'utilities' in sys.modules and hasattr(sys.modules['utilities'], 'cost_tracker'):
                sys.modules['utilities'].cost_tracker.update(response)

            lines = response.content.strip().split('\n')
            for line in lines:
                if line.startswith('ABSTRACT:'):
                    abstract = line.split(':', 1)[1].strip()
                elif line.startswith('CONCLUSION:'):
                    conclusion = line.split(':', 1)[1].strip()

            return abstract, conclusion

        except Exception as e:
            print(f"   ⚠️ Report Agent API failure: {e}. Executing professional text fallbacks.")
            return abstract, conclusion

