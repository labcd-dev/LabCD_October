"""
Physics-Informed Neural Network (PINN) support.

When ``config.USE_PINN`` is enabled the trainer adds a physics residual to the
loss. The analytical derivatives come from a user-authored module (by default
``physics_env.py`` in the working directory) exposing::

    def compute_analytical_xdot(states, actions) -> torch.Tensor

``ensure_pinn_template`` writes a commented starter file when that module does
not exist yet, so the engineer can fill in the known kinematics.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional

PINN_TEMPLATE = '''"""
PHYSICS-INFORMED EQUATION DEFINITIONS
Define your known analytical derivatives (X_dot) here.
Use strictly PyTorch math operations (torch.sin, torch.cos, ...) so the
network can maintain its gradient graph.
"""
import torch


def compute_analytical_xdot(states, actions):
    """
    states:  Tensor of shape [batch_size, state_dim]
    actions: Tensor of shape [batch_size, action_dim]

    Returns:
    physics_xdot: Tensor of shape [batch_size, state_dim]
    """
    batch_size = states.shape[0]
    state_dim = states.shape[1]

    # Initialize the analytical derivative tensor with zeros
    physics_xdot = torch.zeros((batch_size, state_dim), device=states.device, dtype=torch.float32)

    # =========================================================================
    # EXTRACT STATES & ACTIONS (example: standard kinematic vehicle model)
    # Edit these indices to match your dataset columns exactly!
    # =========================================================================
    # x   = states[:, 0]
    # y   = states[:, 1]
    # yaw = states[:, 2]
    # v   = states[:, 3]
    #
    # throttle = actions[:, 0]
    # steering = actions[:, 1]
    #
    # L = 2.5  # Wheelbase in meters
    #
    # =========================================================================
    # DEFINE KNOWN PHYSICS (X_dot)
    # =========================================================================
    # physics_xdot[:, 0] = v * torch.cos(yaw)              # x_dot
    # physics_xdot[:, 1] = v * torch.sin(yaw)              # y_dot
    # physics_xdot[:, 2] = (v / L) * torch.tan(steering)   # yaw_dot
    # physics_xdot[:, 3] = throttle                        # v_dot

    return physics_xdot
'''


def ensure_pinn_template(pinn_file: str | Path) -> bool:
    """
    Make sure the physics module exists.

    Returns
    -------
    True  – the file already existed and the run may continue.
    False – a fresh template was just written; the engineer must fill in the
            equations before running again (the legacy CLI exits here).
    """
    path = Path(pinn_file)
    if path.exists():
        return True

    print(f"\n⚙️ PINN Mode is ENABLED, but '{path}' was not found.")
    print("   Generating a fresh PyTorch physics template...")
    path.write_text(PINN_TEMPLATE, encoding="utf-8")
    print(f"   ✅ Template '{path}' generated successfully.")
    print("   🛑 Execution paused. Open the file, define your physical equations, and run again.")
    return False


def load_analytical_xdot(pinn_file: str | Path) -> Optional[Callable[..., Any]]:
    """
    Import ``compute_analytical_xdot`` from the physics module.

    Accepts either a module name on ``sys.path`` (``physics_env``) or a path to
    a ``.py`` file. Returns None when the symbol cannot be resolved so the
    trainer can silently fall back to a purely data-driven loss.
    """
    path = Path(pinn_file)

    # 1. Plain module import (matches the legacy `from physics_env import ...`)
    module_name = path.stem or str(pinn_file)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, "compute_analytical_xdot", None)
        if callable(fn):
            return fn
    except Exception:
        pass

    # 2. Direct file load
    if path.is_file():
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                fn = getattr(module, "compute_analytical_xdot", None)
                if callable(fn):
                    return fn
        except Exception as exc:
            print(f"    ⚠️ Could not load PINN equations from '{path}': {exc}")

    return None
