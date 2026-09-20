"""Compute device detection."""

from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


def describe_device() -> str:
    """Human-readable banner line, printed once by the CLI."""
    if DEVICE.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        return f"🚀 PyTorch Compute Device Set To: CUDA ({name})"
    return "🚀 PyTorch Compute Device Set To: CPU"
