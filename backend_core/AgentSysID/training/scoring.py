"""
Composite 0-100 fit and rollout score (ported from ``_legacy/framework.calculate_success_score``).

Two pillars, scaled by a 5-stage difficulty allowance so that a model hitting a
given MSE on a chaotic plant is not punished like one on a trivial plant:

  * Pillar A (75 pts) – single-step prediction fit from the validation MSE
  * Pillar B (25 pts) – held-out rollout error from the mean absolute
    difference between the true test trajectory and the network's rollout

This heuristic is a comparison score, not a formal stability or safety certificate.
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
    angle_indices: Sequence[int] = (),
) -> Tuple[float | None, str]:
    """
    Return ``(total_score, status)``. The score is unavailable when held-out
    rollout evidence is missing or unusable.
    """
    true_traj = np.asarray(true_trajectory, dtype=float)
    nn_traj = np.asarray(nn_trajectory, dtype=float)
    if not math.isfinite(float(val_mse)) or float(val_mse) < 0:
        if verbose:
            print("Fit/rollout score unavailable: validation MSE is not finite.")
        return None, "VALIDATION UNVERIFIED"
    if (true_traj.ndim != 2 or nn_traj.ndim != 2 or true_traj.size == 0 or
            nn_traj.size == 0 or true_traj.shape[1] == 0 or
            nn_traj.shape[1] != true_traj.shape[1]):
        if verbose:
            print("Rollout score unavailable: no aligned held-out state predictions were saved.")
        return None, "ROLLOUT UNVERIFIED"

    n = min(len(true_traj), len(nn_traj))
    valid = np.isfinite(true_traj[:n]) & np.isfinite(nn_traj[:n])
    if not np.any(valid):
        if verbose:
            print("Rollout score unavailable: held-out predictions contain no finite aligned values.")
        return None, "ROLLOUT UNVERIFIED"

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
    # PILLAR B: held-out rollout error (max 25 points)
    # ---------------------------------------------------------
    differences = nn_traj[:n] - true_traj[:n]
    for index in (() if angle_indices is None else angle_indices):
        if 0 <= int(index) < differences.shape[1]:
            differences[:, int(index)] = (differences[:, int(index)] + np.pi) % (2.0 * np.pi) - np.pi
    drift_error = float(np.mean(np.abs(differences[valid])))

    lambda_b = 1.0  # heuristic weighting for held-out rollout error
    effective_drift = drift_error / difficulty_allowance
    score_b = 25.0 * math.exp(-lambda_b * effective_drift)

    # ---------------------------------------------------------
    # COMPOSITE SCORE & CATEGORIZATION
    # ---------------------------------------------------------
    total_score = score_a + score_b

    if total_score > 75:
        status = "HIGH FIT/ROLLOUT SCORE"
    elif total_score > 50:
        status = "ACCEPTABLE FIT/ROLLOUT SCORE"
    elif total_score >= 40:
        status = "WEAK FIT/ROLLOUT SCORE"
    else:
        status = "LOW FIT/ROLLOUT SCORE"

    if verbose:
        print("\n" + "=" * 80)
        print("🎯 SYSTEM IDENTIFICATION FIT & ROLLOUT SCORE")
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
            f"  ├── PILLAR B (Held-out Rollout Error): {score_b:.1f} / 25.0 pts "
            f"(Raw Drift: {drift_error:.4f} -> Effective: {effective_drift:.4f})"
        )
        print("-" * 80)
        print(f"  🔥 FINAL COMPOSITE SCORE        : {total_score:.1f} / 100")
        print(f"  🏷️ RUN EVIDENCE RATING          : [{status}]")
        print("=" * 80 + "\n")

    return total_score, status
