"""Utilities for treating selected state dimensions as periodic angles."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def valid_angle_indices(indices: Iterable[int] | None, state_dim: int) -> tuple[int, ...]:
    """Return unique, in-range state indices while preserving their order."""
    result = []
    for value in (() if indices is None else indices):
        index = int(value)
        if 0 <= index < int(state_dim) and index not in result:
            result.append(index)
    return tuple(result)


def encode_periodic_states(states: np.ndarray, angle_indices: Iterable[int] | None) -> np.ndarray:
    """Replace each periodic scalar angle with its sine/cosine features."""
    values = np.asarray(states)
    if values.ndim < 1:
        raise ValueError("States need a final state-dimension axis.")
    angles = set(valid_angle_indices(angle_indices, values.shape[-1]))
    if not angles:
        return values
    features = []
    for index in range(values.shape[-1]):
        state = values[..., index:index + 1]
        if index in angles:
            features.extend((np.sin(state), np.cos(state)))
        else:
            features.append(state)
    return np.concatenate(features, axis=-1)


def wrap_angle_values(states: np.ndarray, angle_indices: Iterable[int] | None) -> np.ndarray:
    """Return a copy with selected angles represented in [-pi, pi)."""
    source = np.asarray(states)
    dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.float64
    values = np.array(source, dtype=dtype, copy=True)
    if values.ndim < 1:
        raise ValueError("States need a final state-dimension axis.")
    for index in valid_angle_indices(angle_indices, values.shape[-1]):
        values[..., index] = (values[..., index] + np.pi) % (2.0 * np.pi) - np.pi
    return values


def circular_difference(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference, in radians."""
    difference = np.asarray(predicted) - np.asarray(actual)
    return (difference + np.pi) % (2.0 * np.pi) - np.pi
