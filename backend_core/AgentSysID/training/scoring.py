"""
Composite 0-100 success score (ported from ``_legacy/framework.calculate_success_score``).

Two pillars, scaled by a 5-stage difficulty allowance so that a model hitting a
given MSE on a chaotic plant is not punished like one on a trivial plant:

  * Pillar A (75 pts) – single-step prediction fit from the validation MSE
  * Pillar B (25 pts) – closed-loop rollout stability from the mean absolute
    drift between the true test trajectory and the network's rollout
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np


def parse_complexity_stage(complexity_label: Any) -> int:
    """Map a tier label (or number) onto the 1-5 difficulty stage."""
    stage = 3  # default moderate

    if isinstance(complexity_label, (int, float)) and not isinstance(complexity_label, bool):
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

    return max(1, min(5, stage))


def calculate_success_score(
    val_mse: float,
    true_trajectory: Sequence[Sequence[float]] | np.ndarray,
    nn_trajectory: Sequence[Sequence[float]] | np.ndarray,
    complexity_label: Any,
    verbose: bool = True,
) -> Tuple[float, str]:
    """
    Return ``(total_score, status)`` where status is one of
    STABLE & HIGH-FIDELITY / STABLE & ACCEPTABLE / UNSTABLE ROLLOUT /
    UNSTABLE / FAILED.
    """
    stage = parse_complexity_stage(complexity_label)

    # Allowance: 1.0 (stage 1) up to 5.0 (stage 5)
    difficulty_allowance = 1.0 + 1 * (stage - 1)

    # ---------------------------------------------------------
    # PILLAR A: single-step prediction fit (max 75 points)
    # ---------------------------------------------------------
    lambda_a = 0.5  # softened from 1.0 to reward complex fits
    effective_mse = float(val_mse) / difficulty_allowance
    score_a = 75.0 * math.exp(-lambda_a * effective_mse)

    # ---------------------------------------------------------
    # PILLAR B: closed-loop rollout stability (max 25 points)
    # ---------------------------------------------------------
    true_traj = np.asarray(true_trajectory, dtype=float)
    nn_traj = np.asarray(nn_trajectory, dtype=float)

    if true_traj.size == 0 or nn_traj.size == 0:
        drift_error = 0.0
    else:
        n = min(len(true_traj), len(nn_traj))
        drift_error = float(np.mean(np.abs(true_traj[:n] - nn_traj[:n])))

    lambda_b = 1.0  # softened from 2.0 to reward stable rollouts
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

    if verbose:
        print("\n" + "=" * 80)
        print("🎯 SYSTEM IDENTIFICATION PERFORMANCE & SUCCESS SCORE")
        print("=" * 80)
        print(
            f"  ├── SYSTEM DIFFICULTY           : Stage {stage} "
            f"({difficulty_allowance:.1f}x error leniency applied)"
        )
        print(
            f"  ├── PILLAR A (1-Step Fit)       : {score_a:.1f} / 75.0 pts "
            f"(Raw Val MSE: {val_mse:.4f} -> Effective: {effective_mse:.4f})"
        )
        print(
            f"  ├── PILLAR B (Rollout Stability): {score_b:.1f} / 25.0 pts "
            f"(Raw Drift: {drift_error:.4f} -> Effective: {effective_drift:.4f})"
        )
        print("-" * 80)
        print(f"  🔥 FINAL COMPOSITE SCORE        : {total_score:.1f} / 100")
        print(f"  🏷️ MODEL STATUS                 : [{status}]")
        print("=" * 80 + "\n")

    return total_score, status
