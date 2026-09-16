"""
Excel / CSV trajectory loader for AgentSysID.

Expects columns:
  - time
  - s_*   (states)
  - a_*   (actions)
  - optional xdot_* (true derivatives)

Preserves the original public contract of ExcelDataLoader while cleaning
imports and making paths / config injectable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend_core.AgentSysID import config as cfg


class ExcelDataLoader:
    """Load and pre-process multi-trajectory system-identification data."""

    def __init__(self, file_path: str | Path, reset_threshold: Optional[Union[float, List[float]]] = None):
        file_path = str(file_path)
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

        # Drop ghost Unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]

        self.state_cols = [c for c in self.df.columns if c.startswith("s_")]
        self.xdot_cols = [c for c in self.df.columns if c.startswith("xdot_")]
        self.action_cols = [c for c in self.df.columns if c.startswith("a_")]

        self.state_dim = len(self.state_cols)
        self.action_dim = len(self.action_cols)

        if "time" not in self.df.columns:
            raise ValueError(
                f"🚨 CRITICAL ERROR: No 'time' column found in '{file_path}'. "
                "A time column is strictly required."
            )

        self.has_time_column = True
        self._analyze_time_column()

        self.has_xdot = len(self.xdot_cols) == self.state_dim
        if self.has_xdot:
            print("    ✅ 'xdot_' columns detected. Using provided derivatives.")
        else:
            print("    ⚠️  No 'xdot_' columns found. X_dot will be estimated.")

        self.reset_threshold = (
            reset_threshold if reset_threshold is not None else cfg.RESET_THRESHOLD
        )
        try:
            self._auto_detect_reset_threshold()
        except Exception:
            pass

        self.complexity_label = "Tier-Unknown"
        self.complexity_score = 0.0
        try:
            self._calculate_complexity()
        except Exception:
            pass

        self.issues: List[str] = []
        self.suggestions: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers (simplified from original for maintainability)
    # ------------------------------------------------------------------
    def _analyze_time_column(self) -> None:
        t = self.df["time"].values.astype(float)
        if len(t) < 2:
            self.median_dt = 0.01
            return
        dts = np.diff(t)
        dts = dts[dts > 0]
        self.median_dt = float(np.median(dts)) if len(dts) else 0.01
        print(f"    ⏱  Median dt ≈ {self.median_dt:.6f} s")

    def _auto_detect_reset_threshold(self) -> None:
        """Heuristic: large state jumps mark trajectory boundaries."""
        if not self.state_cols:
            return
        states = self.df[self.state_cols].values.astype(float)
        if len(states) < 3:
            return
        diffs = np.abs(np.diff(states, axis=0))
        # 99.5-th percentile of per-dimension jumps
        thresholds = np.percentile(diffs, 99.5, axis=0) * 3.0
        # Only override if user left the default "disabled" value
        if isinstance(self.reset_threshold, (int, float)) and self.reset_threshold > 1e4:
            self.reset_threshold = thresholds.tolist()
            print(f"    🔄 Auto-detected reset thresholds: {self.reset_threshold}")

    def _calculate_complexity(self) -> None:
        """Simple Tier 1-5 complexity score based on dimensionality & variance."""
        n = self.state_dim + self.action_dim
        if n <= 3:
            self.complexity_label, self.complexity_score = "Tier-1 (Simple)", 1.0
        elif n <= 6:
            self.complexity_label, self.complexity_score = "Tier-2 (Moderate)", 2.0
        elif n <= 10:
            self.complexity_label, self.complexity_score = "Tier-3 (Complex)", 3.0
        elif n <= 16:
            self.complexity_label, self.complexity_score = "Tier-4 (High)", 4.0
        else:
            self.complexity_label, self.complexity_score = "Tier-5 (Extreme)", 5.0
        print(f"    📊 Dataset complexity: {self.complexity_label}")

    def drop_columns(self, cols_to_drop: List[str]) -> None:
        existing = [c for c in cols_to_drop if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)
            self.state_cols = [c for c in self.df.columns if c.startswith("s_")]
            self.action_cols = [c for c in self.df.columns if c.startswith("a_")]
            self.xdot_cols = [c for c in self.df.columns if c.startswith("xdot_")]
            self.state_dim = len(self.state_cols)
            self.action_dim = len(self.action_cols)
            print(f"    🗑  Dropped columns: {existing}")

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """
        Return list of trajectory dicts:
          {"states": (T, n_s), "actions": (T, n_a), "xdots": (T, n_s) or None, "dts": (T,)}
        """
        if self.state_dim == 0:
            raise ValueError("No state columns (s_*) found in the dataset.")

        states = self.df[self.state_cols].values.astype(np.float64)
        actions = (
            self.df[self.action_cols].values.astype(np.float64)
            if self.action_dim
            else np.zeros((len(self.df), 0))
        )
        times = self.df["time"].values.astype(np.float64)

        if self.has_xdot:
            xdots = self.df[self.xdot_cols].values.astype(np.float64)
        else:
            # Finite-difference estimate
            xdots = np.zeros_like(states)
            for i in range(len(states) - 1):
                dt = max(times[i + 1] - times[i], cfg.MIN_DT)
                xdots[i] = (states[i + 1] - states[i]) / dt
            xdots[-1] = xdots[-2] if len(xdots) > 1 else 0.0

        # Split into trajectories by reset threshold / time reset
        trajectories: List[Dict[str, np.ndarray]] = []
        start = 0
        reset_limit = np.asarray(self.reset_threshold)

        for i in range(1, len(states)):
            jump = np.any(np.abs(states[i] - states[i - 1]) > reset_limit)
            time_reset = times[i] < times[i - 1]
            if jump or time_reset:
                trajectories.append(
                    self._format_chunk(states[start:i], actions[start:i], xdots[start:i], times[start:i])
                )
                start = i
        trajectories.append(
            self._format_chunk(states[start:], actions[start:], xdots[start:], times[start:])
        )

        # Drop tiny segments
        trajectories = [t for t in trajectories if len(t["states"]) >= 5]
        print(f"    ✅ Produced {len(trajectories)} trajectory segment(s).")
        return trajectories

    def _format_chunk(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        xdots: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        dts = np.diff(times, append=times[-1] + (times[-1] - times[-2] if len(times) > 1 else cfg.MIN_DT))
        dts = np.maximum(dts, cfg.MIN_DT)
        return {
            "states": states,
            "actions": actions,
            "xdots": xdots,
            "dts": dts,
            "times": times,
        }


def load_trajectories(file_path: str | Path) -> Tuple[ExcelDataLoader, List[Dict[str, np.ndarray]]]:
    """Convenience: construct loader and return (loader, trajectories)."""
    loader = ExcelDataLoader(file_path)
    trajs = loader.get_trajectories()
    return loader, trajs
