"""
Hybrid train / validation / test splitting.

Mirrors the legacy ``main.py`` logic:

* **More than two trajectories** – the last two are held out whole as the test
  set (pure verification), the rest are split 80/20 into train/val.
* **Exactly two trajectories** – the last one becomes the test set and the
  first is split chronologically.
* **A single continuous trajectory** – the final 10% is always held out
  chronologically (so the rollout verification has a smooth timeline) and the
  remainder is either

    - kept whole and shuffled at row level (``TRAJECTORY_CHUNK_SIZE = 0`` with
      ``SHUFFLE_DATA = True``; best for MLPs), or
    - kept whole and split chronologically (no chunking, no shuffling), or
    - sliced into mini-trajectories of ``TRAJECTORY_CHUNK_SIZE`` steps and
      split with the requested shuffle flag.

Shuffling is forced off whenever the LSTM architecture is active, to preserve
contiguous temporal memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from backend_core.AgentSysID import config as cfg

Trajectory = Dict[str, np.ndarray]
#: Shortest slice that will not break a rollout window.
MIN_CHUNK_STEPS = 25


def _slice(traj: Trajectory, start: int, end: int) -> Trajectory:
    return {key: np.asarray(value)[start:end] for key, value in traj.items()}


def _take(traj: Trajectory, idx: np.ndarray) -> Trajectory:
    return {key: np.asarray(value)[idx] for key, value in traj.items()}


def split_trajectories(
    trajectories: Sequence[Trajectory],
    architecture: str | None = None,
    chunk_size: int | None = None,
    shuffle: bool | None = None,
    verbose: bool = True,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Return ``(train_trajs, val_trajs, test_trajs)``."""
    if not trajectories:
        raise ValueError("Not enough trajectories to split! Please generate more data.")

    arch = str(architecture or cfg.NETWORK_ARCHITECTURE).strip().upper()
    lstm_active = arch == "LSTM"

    if shuffle is None:
        shuffle = bool(getattr(cfg, "SHUFFLE_DATA", True))
    if chunk_size is None:
        chunk_size = int(getattr(cfg, "TRAJECTORY_CHUNK_SIZE", 0))

    shuffle_mode = False if lstm_active else bool(shuffle)

    if verbose:
        print("\n🔀 Splitting data (Hybrid Mode: Shuffled Train/Val, Chronological Test)...")
        if lstm_active:
            print("    ⚠️  LSTM ACTIVE: Data shuffling is DISABLED to preserve contiguous temporal memory.")

    n_traj = len(trajectories)
    sequence_length = int(getattr(cfg, "LSTM_SEQ_LENGTH", 1)) if lstm_active else 1
    rollout_horizon = max(1, int(getattr(cfg, "ROLLOUT_HORIZON", 1)))
    window_length = max(1, sequence_length + rollout_horizon - 1)

    # --- More than two trajectories --------------------------------------
    if n_traj > 2:
        test_trajs = list(trajectories[-2:])
        remaining = list(trajectories[:-2])
        if len(remaining) < 2:
            # Not enough left to split by trajectory: split the first one in time.
            train_trajs, val_trajs = _chronological_split(
                remaining[0], 0.80, min_samples=window_length
            )
            return [train_trajs], [val_trajs], test_trajs

        train_trajs, val_trajs = train_test_split(
            remaining, test_size=0.20, random_state=42, shuffle=shuffle_mode
        )
        return list(train_trajs), list(val_trajs), test_trajs

    # --- Exactly two trajectories ----------------------------------------
    if n_traj == 2:
        if verbose:
            print("    ⚠️ Only 2 trajectories detected. Holding the second out as the test set.")
        test_trajs = [trajectories[1]]
        train_part, val_part = _chronological_split(
            trajectories[0], 0.85, min_samples=window_length
        )
        return [train_part], [val_part], test_trajs

    # --- A single continuous trajectory ----------------------------------
    if verbose:
        print("    ⚠️ Only 1 trajectory detected. Reading Master Configuration...")

    all_steps = trajectories[0]
    total_len = len(all_steps["states"])
    if total_len < 3 * window_length:
        raise ValueError(
            f"A single trajectory needs at least {3 * window_length} samples for "
            f"train, validation and test windows of length {window_length}. "
            "Use a shorter sequence or rollout window, or provide more measurements."
        )

    # Keep a ten-percent test tail when it is large enough; small recordings
    # borrow rows from the training side so all three partitions contain one
    # complete model window instead of creating an empty validation metric.
    test_rows = max(window_length, total_len - int(total_len * 0.90))
    test_split_idx = total_len - test_rows
    temp_train_val = _slice(all_steps, 0, test_split_idx)
    test_trajs = [_slice(all_steps, test_split_idx, total_len)]

    if chunk_size == 0:
        if shuffle_mode:
            if verbose:
                print("       -> [MODE ACTIVE]: ZERO CHUNKING, BUT ROW-LEVEL SHUFFLING IS ON!")
                print("       -> [NOTE]: This mode is heavily optimized for MLP architectures.")

            n = len(temp_train_val["states"])
            idx = np.arange(n)
            val_rows = min(n - window_length, max(window_length, int(np.ceil(n * 0.15))))
            train_idx, val_idx = train_test_split(
                idx, test_size=val_rows, random_state=42, shuffle=True
            )
            return (
                [_take(temp_train_val, np.sort(train_idx))],
                [_take(temp_train_val, np.sort(val_idx))],
                test_trajs,
            )

        if verbose:
            print("       -> [MODE ACTIVE]: ZERO CHUNKING (Pure Chronological).")
            print("       -> [WARNING]: SHUFFLE=False. VRAM usage may be very high for LSTMs.")

        train_part, val_part = _chronological_split(
            temp_train_val, 0.88, min_samples=window_length
        )
        return [train_part], [val_part], test_trajs

    # Chunked mode
    if verbose:
        print(f"       -> [MODE ACTIVE]: CHUNKING (Size={chunk_size}).")
        print(f"       -> [MODE ACTIVE]: SHUFFLING = {shuffle_mode}.")

    n = len(temp_train_val["states"])
    sub_trajectories: List[Trajectory] = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        if end - i >= MIN_CHUNK_STEPS:
            sub_trajectories.append(_slice(temp_train_val, i, end))

    if len(sub_trajectories) < 2:
        train_part, val_part = _chronological_split(
            temp_train_val, 0.85, min_samples=window_length
        )
        return [train_part], [val_part], test_trajs

    train_trajs, val_trajs = train_test_split(
        sub_trajectories, test_size=0.15, random_state=42, shuffle=shuffle_mode
    )
    return list(train_trajs), list(val_trajs), test_trajs


def _chronological_split(
    traj: Trajectory, train_fraction: float, min_samples: int = 1
) -> Tuple[Trajectory, Trajectory]:
    n = len(traj["states"])
    if n < 2 * min_samples:
        raise ValueError(
            f"Cannot make train and validation windows of length {min_samples} "
            f"from a {n}-sample trajectory. Provide more data or shorten the model window."
        )
    split_idx = max(min_samples, min(n - min_samples, int(n * train_fraction)))
    return _slice(traj, 0, split_idx), _slice(traj, split_idx, n)


def describe_split(
    train_trajs: Sequence[Trajectory],
    val_trajs: Sequence[Trajectory],
    test_trajs: Sequence[Trajectory],
    n_trajectories: int,
) -> None:
    """Print the row counts of each partition."""
    rows = lambda ts: sum(len(t["states"]) for t in ts)  # noqa: E731
    print(f"    -> Training batches  : {rows(train_trajs)} rows")
    print(f"    -> Validation batches: {rows(val_trajs)} rows")
    if n_trajectories > 2:
        print(f"    -> Testing batches   : {rows(test_trajs)} rows (2 Trajectories Held Out)")
    else:
        print(f"    -> Testing batches   : {rows(test_trajs)} rows (Final 10% Held Out)")
