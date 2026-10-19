"""
Validated physics equation support shared by PINN and PCNN.

When ``config.USE_PINN`` is enabled the trainer either adds a PINN residual to
the loss or uses the equation as a PCNN output baseline. Analytical derivatives
come from a validated module exposing::

    def compute_analytical_xdot(states, actions) -> torch.Tensor

``ensure_pinn_template`` writes a commented starter file when that module does
not exist yet, so the engineer can fill in the known kinematics.
"""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
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
    Import ``compute_analytical_xdot`` from the physics module. LabCD-generated
    equations may expose ``build_physics_model`` to provide bounded trainable
    coefficients; legacy equations remain plain callables.

    Accepts either a module name on ``sys.path`` (``physics_env``) or a path to
    a ``.py`` file. Returns None when the symbol cannot be resolved so the
    trainer can silently fall back to a purely data-driven loss.
    """
    path = Path(pinn_file)

    # Resolve explicit files first. Several conversations may each have a
    # different equation file with the same basename; importing by stem could
    # silently reuse a stale module from another run.
    if path.is_file():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            module_name = f"_labcd_physics_{digest}"
            cached = sys.modules.get(module_name)
            if cached is not None:
                builder = getattr(cached, "build_physics_model", None)
                if callable(builder):
                    return builder()
                fn = getattr(cached, "compute_analytical_xdot", None)
                if callable(fn):
                    return fn
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                builder = getattr(module, "build_physics_model", None)
                if callable(builder):
                    return builder()
                fn = getattr(module, "compute_analytical_xdot", None)
                if callable(fn):
                    return fn
        except Exception as exc:
            print(f"    ⚠️ Could not load PINN equations from '{path}': {exc}")

    # A module name remains supported for legacy CLI installations.
    module_name = path.stem or str(pinn_file)
    try:
        module = importlib.import_module(module_name)
        builder = getattr(module, "build_physics_model", None)
        if callable(builder):
            return builder()
        fn = getattr(module, "compute_analytical_xdot", None)
        if callable(fn):
            return fn
    except Exception:
        pass

    return None
