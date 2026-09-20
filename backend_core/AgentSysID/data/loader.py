"""
Excel / CSV trajectory loader for AgentSysID.

Expects columns (see DATA_CONTRACT.md):
  - time
  - s_*    (states)
  - a_*    (actions)
  - xdot_* (optional true derivatives)

This is the full ``_legacy/framework.ExcelDataLoader`` pipeline:

  1. dual CSV / Excel ingest + ghost-column removal
  2. time-column consistency analysis (variable dt detection)
  3. auto-calibrated trajectory reset thresholds
  4. angular-state detection (manual / locked / auto-scan)
  5. localized state-space percentile filtering
  6. mathematical data-quality audit (excitation, collinearity, spikes)
  7. 1-5 dataset complexity tiering
  8. trajectory extraction with exact per-row dt and one of three derivative
     estimators (finite difference + Simulink filter, Levant sliding mode,
     Savitzky-Golay)

Trajectories are returned in the package's array-of-columns contract::

    {"states": (T, n_s), "actions": (T, n_a), "xdots": (T, n_s),
     "dts": (T,), "times": (T,)}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend_core.AgentSysID import config as cfg

Trajectory = Dict[str, np.ndarray]

COMPLEXITY_LABELS: Dict[int, str] = {
    1: "Level 1 (Trivial / Quasi-Steady)",
    2: "Level 2 (Mildly Nonlinear)",
    3: "Level 3 (Moderately Complex)",
    4: "Level 4 (Highly Complex / Fast Transient)",
    5: "Level 5 (Severe / Chaotic / Discontinuous)",
}


class ExcelDataLoader:
    """Load and pre-process multi-trajectory system-identification data."""

    def __init__(
        self,
        file_path: str | Path,
        reset_threshold: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        file_path = str(file_path)

        # --- Robust dual file-format support ---------------------------------
        if file_path.endswith(".csv"):
            print(f"    📂 Loading CSV dataset: '{file_path}'")
            self.df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            print(f"    📂 Loading Excel dataset: '{file_path}'")
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError(
                f"❌ Unsupported file format for '{file_path}'. Please use a .csv or .xlsx file."
            )

        self.file_path = file_path

        # --- Destroy ghost columns automatically -----------------------------
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]

        # Dynamically map the columns based on their prefixes
        self.state_cols: List[str] = [c for c in self.df.columns if c.startswith("s_")]
        self.xdot_cols: List[str] = [c for c in self.df.columns if c.startswith("xdot_")]
        self.action_cols: List[str] = [c for c in self.df.columns if c.startswith("a_")]

        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)

        # --- Pre-processing data agent: time analysis ------------------------
        time_col = getattr(cfg, "TIME_COLUMN", "time")
        if time_col not in self.df.columns:
            raise ValueError(
                f"🚨 CRITICAL ERROR: No '{time_col}' column found in '{file_path}'. "
                "A time column is strictly required to calculate physical derivatives "
                "correctly. Execution stopped."
            )
        self.time_col = time_col
        self.has_time_column = True
        self.median_dt = 0.01
        self._analyze_time_column()

        # --- Check for x_dot columns -----------------------------------------
        self.has_xdot = len(self.xdot_cols) == self.state_dim and self.state_dim > 0
        if self.has_xdot:
            print("    ✅ 'xdot_' columns detected in the dataset. Using provided derivatives for training.")
        else:
            print("    ⚠️  No 'xdot_' columns found. X_dot will be estimated via finite differences.")
            print("    💡  TIP: If you can provide true X_dot directly from your simulation or hardware sensors,")
            print("         the neural network will achieve much better accuracy without noise amplification.")

        # --- Automated reset-threshold detection -----------------------------
        if reset_threshold is not None:
            self.reset_threshold: Union[float, List[float]] = (
                list(reset_threshold)
                if isinstance(reset_threshold, (list, tuple, np.ndarray))
                else float(reset_threshold)
            )
        else:
            configured = getattr(cfg, "RESET_THRESHOLD", None)
            # If the user did not lock it, compute it mathematically from the data.
            if configured is None or (
                isinstance(configured, (int, float)) and float(configured) > 90000
            ):
                self.reset_threshold = self._auto_detect_reset_threshold()
            else:
                self.reset_threshold = (
                    list(configured) if isinstance(configured, (list, tuple)) else float(configured)
                )

        print(f"    ⚙️ Auto-calibrated Reset Thresholds: {self.reset_threshold}")

        # --- Smart angle index selection -------------------------------------
        if getattr(cfg, "AUTO_DETECT_ANGLES", False) is True:
            print("    ⚙️ Questionnaire requested AUTO-DETECT for angles. Scanning data...")
            self.angle_indices: List[int] = self._run_angle_auto_detect()
        elif getattr(cfg, "ANGLE_INDICES", None):
            self.angle_indices = list(cfg.ANGLE_INDICES)
            print(f"    ⚙️ Locked angular states from questionnaire: {self.angle_indices}")
        else:
            self.angle_indices = []
            print("    ⚙️ Questionnaire confirmed NO angular states.")

        # --- Apply the dynamic state filter ----------------------------------
        self._apply_state_filter()

        # --- Run the quality agent on the filtered data ----------------------
        self.quality_issues: List[str] = []
        self._run_data_quality_agent()

        # --- Calculate and store dataset complexity --------------------------
        self.complexity_tier, self.complexity_label = self._calculate_complexity()
        # Numeric alias kept for callers that prefer a score over a label.
        self.complexity_score: float = float(self.complexity_tier)

    # ------------------------------------------------------------------
    # Column surgery
    # ------------------------------------------------------------------
    def drop_columns(self, cols_to_drop: Sequence[str]) -> None:
        """Physically remove columns and their corresponding derivatives."""
        dropped: List[str] = []
        for c in cols_to_drop:
            # 1. Drop the state / action column
            if c in self.df.columns:
                self.df = self.df.drop(columns=[c])
                if c in self.state_cols:
                    self.state_cols.remove(c)
                if c in self.action_cols:
                    self.action_cols.remove(c)
                dropped.append(c)

            # 2. Hunt down and destroy its corresponding xdot column
            target_xdot = c.replace("s_", "xdot_")
            if target_xdot in self.df.columns:
                self.df = self.df.drop(columns=[target_xdot])
                if target_xdot in self.xdot_cols:
                    self.xdot_cols.remove(target_xdot)
                dropped.append(target_xdot)

        # Update the critical math dimensions
        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)
        self.has_xdot = len(self.xdot_cols) == self.state_dim and self.state_dim > 0

        # Recalculate thresholds for the new dimensions to avoid broadcast errors
        if isinstance(self.reset_threshold, list):
            self.reset_threshold = self._auto_detect_reset_threshold()

        if dropped:
            print(f"    🗑️ SUCCESS: Physically removed {dropped} from the dataset.")
            print(f"       -> New Network State Dimensions: {self.state_dim}")

    # ------------------------------------------------------------------
    # Mathematical pre-processing
    # ------------------------------------------------------------------
    def _auto_detect_reset_threshold(self) -> List[float]:
        """
        Scan the data to separate normal stepping from trajectory reset
        boundaries (teleportations between stacked runs).
        """
        if not self.state_cols:
            return []

        diffs = self.df[self.state_cols].diff().abs()
        thresholds: List[float] = []

        for col in self.state_cols:
            col_diffs = diffs[col].dropna()
            if len(col_diffs) == 0:
                thresholds.append(10.0)
                continue

            # 99th percentile of normal step-to-step changes
            p99 = float(np.percentile(col_diffs, 99.0))
            max_val = float(col_diffs.max())

            # Safely above normal motion, catching only true resets
            thresh = max(p99 * 3.0, (p99 + max_val) / 2.0)
            thresholds.append(float(thresh))

        return thresholds

    def _run_angle_auto_detect(self) -> List[int]:
        """Statistically analyse data to find angles that wrap around +/- pi."""
        print("      -> No angular indices provided in config. Auto-scanning data...")
        detected_angles: List[int] = []
        for idx, col in enumerate(self.state_cols):
            data = self.df[col].values

            # Condition 1: data is bounded near [-pi, pi]
            if np.max(data) <= 3.2 and np.min(data) >= -3.2:
                # Condition 2: presence of massive wrap-around jumps
                diffs = np.abs(np.diff(data))
                if np.any(diffs > 5.0):
                    detected_angles.append(idx)

        if detected_angles:
            print(f"         ✅ Auto-detected angular states at indices: {detected_angles}")
        else:
            print("         ℹ️ No wrapped angular states detected in this dataset.")

        return detected_angles

    def _run_data_quality_agent(self) -> None:
        """Mathematical health check feeding the LLM Data Inspector."""
        print("\n    🕵️‍♂️ DATA QUALITY CHECK: Computing mathematical health metrics...")

        self.quality_issues = []

        # --- 1. Persistent excitation (action variance) ----------------------
        if self.action_cols:
            action_vars = self.df[self.action_cols].var()
            for col, variance in action_vars.items():
                if variance < 1e-4:
                    self.quality_issues.append(
                        f"Action '{col}' has near-zero variance ({variance:.5f})."
                    )

        # --- 2. Multicollinearity (state correlation) ------------------------
        if len(self.state_cols) > 1:
            state_corr = self.df[self.state_cols].corr().abs()
            upper_tri = state_corr.where(
                np.triu(np.ones(state_corr.shape), k=1).astype(bool)
            )

            high_corr_pairs = []
            for col in upper_tri.columns:
                for row in upper_tri.index:
                    value = upper_tri.loc[row, col]
                    if pd.notna(value) and value > 0.95:
                        high_corr_pairs.append((row, col, float(value)))

            if high_corr_pairs:
                corr_details = [
                    f"- '{a}' & '{b}' (Correlation: {v:.4f})" for a, b, v in high_corr_pairs
                ]
                self.quality_issues.append(
                    "Highly correlated or duplicated states detected:\n" + "\n".join(corr_details)
                )

        # --- 3. Outlier detection (derivative spikes) ------------------------
        target_cols = self.xdot_cols if self.has_xdot else self.state_cols
        outlier_count = 0
        for col in target_cols:
            data = self.df[col]
            std = data.std()
            if std > 1e-8:
                z_scores = np.abs((data - data.mean()) / std)
                spikes = int((z_scores > 6).sum())
                if spikes > 0:
                    outlier_count += spikes
        if outlier_count > 0:
            self.quality_issues.append(
                f"Found {outlier_count} extreme numerical spikes (Z-score > 6)."
            )

        print("    ✅ Mathematical checks complete. Handing data to LLM Inspector...")

    def _calculate_complexity(self) -> Tuple[int, str]:
        """Tier 1-5 complexity from dimensionality, coupling and dynamic aggressiveness."""
        # 1. Structural complexity (total number of variables)
        dim_score = len(self.state_cols) + len(self.action_cols)

        # 2. Coupling / collinearity (max off-diagonal correlation)
        max_corr = 0.0
        if len(self.state_cols) > 1:
            state_corr = self.df[self.state_cols].corr().abs()
            corr_array = state_corr.to_numpy(copy=True)  # mutable copy
            np.fill_diagonal(corr_array, 0)
            corr_array = np.nan_to_num(corr_array, nan=0.0)
            max_corr = float(corr_array.max()) if corr_array.size > 0 else 0.0

        # 3. Dynamic aggressiveness (variance of the system's changes)
        if self.has_xdot:
            derivatives = self.df[self.xdot_cols]
        else:
            derivatives = self.df[self.state_cols].diff().dropna()

        if len(derivatives) == 0 or derivatives.shape[1] == 0:
            agg_score = 0.0
        else:
            # Normalize variance to avoid scale dependency
            ratio = derivatives.var() / (derivatives.abs().mean() + 1e-6)
            agg_score = float(np.nan_to_num(ratio.max(), nan=0.0))

        # 4. Evaluate tier (1 to 5)
        tier = 1
        if dim_score >= 4 or max_corr > 0.4:
            tier = 2
        if dim_score >= 7 or max_corr > 0.75 or agg_score > 2.0:
            tier = 3
        if max_corr > 0.90 or agg_score > 5.0:
            tier = 4
        if max_corr > 0.95 or agg_score > 10.0:
            tier = 5

        complexity_label = COMPLEXITY_LABELS.get(tier, "Unknown")

        print("\n    🧠 DATASET COMPLEXITY ANALYSIS:")
        print(f"      -> State & Action Dimensions : {dim_score}")
        print(f"      -> Max State Coupling        : {max_corr:.3f}")
        print(f"      -> Dynamic Aggressiveness    : {agg_score:.3f}")
        print("      ==================================================")
        print(f"      📊 ASSIGNED TIER: {complexity_label}")
        print("      ==================================================\n")

        return tier, complexity_label

    def _apply_state_filter(self) -> None:
        """Clip extreme outliers based on the dynamically chosen percentiles."""
        if not getattr(cfg, "USE_STATE_FILTER", False):
            return
        if not self.state_cols:
            return

        p_low, p_high = getattr(cfg, "AUTO_FILTER_PERCENTILES", (2, 98))

        print("\n    🧹 STATE SPACE FILTER [ACTIVE]")
        print(f"      -> Clipping state outliers beyond the {p_low}th and {p_high}th percentiles...")

        lower_bounds = self.df[self.state_cols].quantile(float(p_low) / 100.0)
        upper_bounds = self.df[self.state_cols].quantile(float(p_high) / 100.0)

        self.df[self.state_cols] = self.df[self.state_cols].clip(
            lower=lower_bounds, upper=upper_bounds, axis=1
        )

        print("      ✅ State space successfully bounded.")

    def _analyze_time_column(self) -> None:
        """Pre-processing agent that analyses time steps for consistency."""
        times = self.df[self.time_col].values.astype(float)
        if len(times) < 2:
            self.median_dt = 0.01
            return

        dts = np.diff(times)

        # 1. Fatal time errors
        if np.any(dts <= 0):
            print("    🚨 CRITICAL ERROR: Time goes backwards or contains duplicate timestamps!")

        positive = dts[dts > 0]
        self.median_dt = float(np.median(positive)) if len(positive) else 0.01

        dt_std = float(np.std(dts))
        dt_mean = float(np.mean(dts))

        # 2. Variable time steps (1e-5 tolerance for floating-point math)
        if dt_std > 1e-5:
            print("    ⚠️  VARIABLE TIME STEP DETECTED!")
            print(
                f"        Min dt: {np.min(dts):.5f}s | Max dt: {np.max(dts):.5f}s | Mean: {dt_mean:.5f}s"
            )
            print("        💡 LOGIC APPLIED: Interpolation is disabled to preserve true physics.")
            print("           The framework calculates the exact dt for every individual row.")
        else:
            print(f"    ✅ Constant time step verified (dt ≈ {dt_mean:.5f}s).")

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------
    def get_trajectories(self) -> List[Trajectory]:
        """
        Split the dataset into trajectories and attach exact dt plus the
        estimated (or provided) state derivatives.
        """
        if self.state_dim == 0:
            raise ValueError("No state columns (s_*) found in the dataset.")

        print("\n    ⏳ Extracting dataset and computing physical derivatives. Please wait...", flush=True)

        is_multi = getattr(cfg, "MULTI_TRAJECTORY", True)
        manual_times = list(getattr(cfg, "MANUAL_TRAJECTORY_SPLIT_TIMES", []) or [])

        n_rows = len(self.df)
        states = self.df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
        actions = (
            self.df[self.action_cols].to_numpy(dtype=np.float32, copy=True)
            if self.action_cols
            else np.zeros((n_rows, 0), dtype=np.float32)
        )
        xdots = (
            self.df[self.xdot_cols].to_numpy(dtype=np.float32, copy=True)
            if self.has_xdot
            else None
        )
        times = self.df[self.time_col].to_numpy(dtype=np.float64, copy=True)

        # Progress feedback on the ingest pass, as the legacy loader did.
        for _ in tqdm(range(n_rows), desc="      -> Reading dataset ", unit=" rows", leave=False):
            pass

        trajectories: List[Trajectory] = []
        if n_rows == 0:
            return trajectories

        # --- 1. Single continuous trajectory --------------------------------
        if not is_multi:
            print("      -> Processing as 1 continuous trajectory per questionnaire.")
            trajectories.append(
                self._attach_dt_and_format(states, actions, xdots, times, show_bar=True)
            )
            return trajectories

        # --- 2. Manual timestamps provided by the client ---------------------
        if manual_times:
            print(f"      -> Splitting trajectories using {len(manual_times)} manual timestamps...")
            boundaries = [0]
            for j in range(1, n_rows):
                if any(abs(times[j] - target) < 1e-4 for target in manual_times):
                    boundaries.append(j)
            boundaries.append(n_rows)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end > start:
                    trajectories.append(
                        self._attach_dt_and_format(
                            states[start:end],
                            actions[start:end],
                            None if xdots is None else xdots[start:end],
                            times[start:end],
                        )
                    )

            print(f"      ✅ Created {len(trajectories)} distinct trajectory segments.")
            return trajectories

        # --- 3. Auto-detect trajectory boundaries ----------------------------
        print("      -> Auto-detecting trajectory boundaries from data jumps...")
        reset_limit = self._reset_limit_array()

        boundaries = [0]
        for j in range(1, n_rows):
            state_jump = bool(np.any(np.abs(states[j] - states[j - 1]) > reset_limit))
            time_reset = bool(times[j] < times[j - 1])
            if state_jump or time_reset:
                boundaries.append(j)
        boundaries.append(n_rows)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end > start:
                trajectories.append(
                    self._attach_dt_and_format(
                        states[start:end],
                        actions[start:end],
                        None if xdots is None else xdots[start:end],
                        times[start:end],
                    )
                )

        print(f"      ✅ Auto-detected and split into {len(trajectories)} trajectories.")
        return trajectories

    def _reset_limit_array(self) -> np.ndarray | float:
        thresh = self.reset_threshold
        if isinstance(thresh, (list, tuple, np.ndarray)):
            arr = np.asarray(thresh, dtype=np.float64)
            if arr.size == self.state_dim:
                return arr
            # Dimension mismatch (e.g. after a column drop) -> broadcast safely
            return float(arr.max()) if arr.size else 1.0
        return float(thresh)

    # ------------------------------------------------------------------
    def _attach_dt_and_format(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        xdots: Optional[np.ndarray],
        times: np.ndarray,
        show_bar: bool = False,
    ) -> Trajectory:
        """
        Compute the exact dt per row and the derivative targets for one chunk.

        Derivative estimators (only used when the dataset has no xdot_ columns):
          * "savitzky_golay" – vectorised polynomial smoothing differentiator
          * "sliding_mode"   – Levant's robust exact differentiator
          * "finite_difference" (default) – central differences plus a
            first-order Simulink-style low-pass filter (tau)
        """
        n = len(states)
        min_dt = float(getattr(cfg, "MIN_DT", 1e-6))

        # --- FIRST PASS: delta-t strictly from timestamps --------------------
        dts = np.empty(n, dtype=np.float32)
        for j in range(n):
            if j + 1 < n:
                dt = times[j + 1] - times[j]
            elif j > 0:
                dt = times[j] - times[j - 1]
            else:
                dt = min_dt

            if dt is None or dt < min_dt:
                dt = min_dt
            dts[j] = np.float32(dt)

        if xdots is not None:
            return {
                "states": states,
                "actions": actions,
                "xdots": xdots.astype(np.float32),
                "dts": dts,
                "times": times,
            }

        method = str(getattr(cfg, "DERIVATIVE_METHOD", "finite_difference")).strip().lower()
        angles = list(getattr(cfg, "ANGLE_INDICES", []) or []) or list(self.angle_indices)

        # --- Pre-compute vectorised Savitzky-Golay derivative ----------------
        xdot_savgol_matrix: Optional[np.ndarray] = None
        if method == "savitzky_golay":
            window = int(getattr(cfg, "SAVGOL_WINDOW", 15))
            poly = int(getattr(cfg, "SAVGOL_POLYORDER", 2))

            s_matrix = np.array(states, dtype=np.float64, copy=True)
            for idx in angles:
                if idx < s_matrix.shape[1]:
                    s_matrix[:, idx] = np.unwrap(s_matrix[:, idx])

            if n < window:
                window = n
            if window % 2 == 0:
                window -= 1
            if window <= poly:
                poly = max(1, window - 1)

            if window >= 3:
                from scipy.signal import savgol_filter

                mean_dt = float(np.mean(dts))
                xdot_savgol_matrix = savgol_filter(
                    s_matrix,
                    window_length=window,
                    polyorder=poly,
                    deriv=1,
                    delta=mean_dt,
                    axis=0,
                )
            else:
                xdot_savgol_matrix = np.zeros_like(s_matrix)

        # --- SECOND PASS: row-by-row filtering & formatting ------------------
        out_xdots = np.zeros_like(states, dtype=np.float32)
        prev_xdot_filtered: Optional[np.ndarray] = None
        z0: Optional[np.ndarray] = None
        z1: Optional[np.ndarray] = None

        tau = float(getattr(cfg, "DERIVATIVE_FILTER_TAU", 0.0))
        lam1 = float(getattr(cfg, "SMD_LAMBDA_1", 5.0))
        lam2 = float(getattr(cfg, "SMD_LAMBDA_2", 10.0))
        reset_limit = self._reset_limit_array()

        indices: Any = range(n)
        if show_bar:
            indices = tqdm(range(n), total=n, desc="      -> Applying filters", unit=" steps", leave=False)

        for j in indices:
            s = states[j]
            dt = float(dts[j])

            # ---------------- METHOD A: Savitzky-Golay -----------------------
            if method == "savitzky_golay" and xdot_savgol_matrix is not None:
                xdot_calculated = xdot_savgol_matrix[j]

            # ---------------- METHOD B: Levant sliding mode ------------------
            elif method == "sliding_mode":
                if z0 is None:
                    z0 = np.array(s, dtype=np.float64, copy=True)
                    z1 = np.zeros_like(z0)

                e = z0 - s.astype(np.float64)
                for idx in angles:
                    if idx < len(e):
                        e[idx] = (e[idx] + np.pi) % (2 * np.pi) - np.pi

                sign_e = np.sign(e)
                sqrt_abs_e = np.sqrt(np.abs(e))
                xdot_raw = -lam1 * sqrt_abs_e * sign_e + z1
                z0 = z0 + dt * xdot_raw
                z1 = z1 - dt * lam2 * sign_e
                xdot_calculated = xdot_raw

            # ---------------- METHOD C: finite difference + filter -----------
            else:
                if j + 1 < n:
                    if j > 0:
                        diff = (states[j + 1] - states[j - 1]).astype(np.float64)
                        calc_dt = float(times[j + 1] - times[j - 1])
                    else:
                        diff = (states[j + 1] - s).astype(np.float64)
                        calc_dt = dt

                    for idx in angles:
                        if idx < len(diff):
                            diff[idx] = (diff[idx] + np.pi) % (2 * np.pi) - np.pi

                    if calc_dt < min_dt:
                        calc_dt = min_dt

                    if np.any(np.abs(diff) > reset_limit):
                        # Trajectory reset boundary: derivative is meaningless here
                        xdot_raw = np.zeros_like(s, dtype=np.float64)
                        prev_xdot_filtered = None
                    else:
                        xdot_raw = diff / calc_dt
                else:
                    # Last sample: hold the previous derivative
                    xdot_raw = (
                        out_xdots[j - 1].astype(np.float64)
                        if j > 0
                        else np.zeros_like(s, dtype=np.float64)
                    )

                if tau > 0.0:
                    alpha = dt / (tau + dt)
                    if prev_xdot_filtered is None:
                        xdot_calculated = xdot_raw
                    else:
                        xdot_calculated = alpha * xdot_raw + (1.0 - alpha) * prev_xdot_filtered
                    prev_xdot_filtered = xdot_calculated
                else:
                    xdot_calculated = xdot_raw

            out_xdots[j] = np.asarray(xdot_calculated, dtype=np.float32)

        return {
            "states": states,
            "actions": actions,
            "xdots": out_xdots,
            "dts": dts,
            "times": times,
        }


def load_trajectories(
    file_path: str | Path,
) -> Tuple[ExcelDataLoader, List[Trajectory]]:
    """Convenience: construct the loader and return (loader, trajectories)."""
    loader = ExcelDataLoader(file_path)
    trajs = loader.get_trajectories()
    return loader, trajs
