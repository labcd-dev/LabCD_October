#!/usr/bin/env python3
"""
AgentSysID CLI entry point.

Usage (from repository root):
    PYTHONPATH=. python -m backend_core.AgentSysID.run_cli
    PYTHONPATH=. python -m backend_core.AgentSysID.run_cli --data path/to/data.csv --mode fast
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure repo root is on path when run as script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents import (
    ActorAgent,
    CriticAgent,
    ExplorerAgent,
    InitializerAgent,
    ReportAgent,
    run_data_inspector_agent,
)
from backend_core.AgentSysID.data import ExcelDataLoader
from backend_core.AgentSysID.reporting import generate_final_pdf, package_final_results_to_zip
from backend_core.AgentSysID.training import (
    BestConfigTracker,
    compute_rmse,
    measure_inference_latency,
    train_dynamics_model,
)
from backend_core.AgentSysID.utils import (
    cost_tracker,
    request_stop,
    reset_stop_flag,
    setup_logging,
    stop_requested,
)
from backend_core.AgentSysID.utils.device import DEVICE


def _max_cycles(mode: str) -> int:
    mode = (mode or "regular").lower()
    if mode == "fast":
        return 5
    if mode == "heavy":
        return 40
    return 15  # regular


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentSysID – Agentic System Identification")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV/Excel dataset")
    parser.add_argument("--mode", type=str, default=cfg.RUN_MODE, choices=["fast", "regular", "heavy"])
    parser.add_argument("--interactive", action="store_true", help="Enable HIL Data Inspector prompts")
    parser.add_argument("--output-dir", type=str, default="artifacts_sysid", help="Where to write reports")
    args = parser.parse_args(argv)

    data_path = args.data or cfg.EXCEL_FILE_PATH
    if not os.path.isfile(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("   Provide --data path/to/file.csv or place system_data.csv in the working directory.")
        return 1

    reset_stop_flag()
    signal.signal(signal.SIGINT, request_stop)

    log_file = setup_logging("logs")
    print(f"🚀 PyTorch device: {DEVICE.type.upper()}")
    print(f"📂 Dataset: {data_path}")
    print(f"⚙  Run mode: {args.mode}  |  Interactive HIL: {args.interactive}")
    print(f"📝 Agent log: {log_file}")

    # ------------------------------------------------------------------
    # 1. Load & inspect data
    # ------------------------------------------------------------------
    loader = ExcelDataLoader(data_path)
    loader = run_data_inspector_agent(
        loader, interactive=args.interactive, log_filename=str(log_file)
    )
    trajectories = loader.get_trajectories()
    if len(trajectories) < 2:
        print("⚠️  Fewer than 2 trajectories – using a simple 80/20 split of the single segment.")
        # Duplicate for train/val if needed
        if len(trajectories) == 1:
            trajectories = trajectories * 2

    train_trajs, val_trajs = train_test_split(trajectories, test_size=0.2, random_state=42)

    # ------------------------------------------------------------------
    # 2. Initializer
    # ------------------------------------------------------------------
    init_agent = InitializerAgent(loader, log_filename=str(log_file))
    config = init_agent.determine_initial_setup()
    print(f"\n🎯 Initial config from Initializer:\n   {config}")

    # ------------------------------------------------------------------
    # 3. Actor–Critic–Explorer loop
    # ------------------------------------------------------------------
    tracker = BestConfigTracker(run_mode=args.mode)
    critic = CriticAgent(tracker, run_mode=args.mode, log_filename=str(log_file))
    actor = ActorAgent(activation=config.get("activation", "relu"), log_filename=str(log_file))
    explorer = ExplorerAgent(log_filename=str(log_file))

    max_cycles = _max_cycles(args.mode)
    performance_history = []
    best_model = None
    stagnation = 0

    for cycle in range(1, max_cycles + 1):
        if stop_requested():
            print("🛑 Stop requested – exiting loop.")
            break

        print(f"\n{'='*60}\n🔄 CYCLE {cycle}/{max_cycles}\n{'='*60}")
        print(f"   Config: {config}")

        model, train_mse, val_mse = train_dynamics_model(
            train_trajs=train_trajs,
            val_trajs=val_trajs,
            state_dim=loader.state_dim,
            action_dim=loader.action_dim,
            hidden_layers=config["hidden_layers"],
            learning_rate=config["learning_rate"],
            activation=config.get("activation", "relu"),
            dropout_rate=config.get("dropout_rate", 0.0),
            weight_decay=config.get("weight_decay", 0.0),
            epochs=min(cfg.EPOCHS, 100 if args.mode == "fast" else cfg.EPOCHS),
        )

        latency = measure_inference_latency(model, loader.state_dim, loader.action_dim)
        rmse = compute_rmse(val_mse)
        print(f"   Train MSE={train_mse:.6e}  Val MSE={val_mse:.6e}  Latency={latency:.3f} ms")

        improved = tracker.update(val_mse, config, current_rmse=rmse)
        if improved:
            best_model = model
            stagnation = 0
            print("   ⭐ New best configuration.")
        else:
            stagnation += 1

        performance_history.append(
            {"cycle": cycle, "val_mse": val_mse, "train_mse": train_mse, "latency": latency, "config": dict(config)}
        )

        decision = critic.evaluate(
            train_mse=train_mse,
            val_mse=val_mse,
            current_config=config,
            activation=config.get("activation", "relu"),
            measured_latency=latency,
            max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
            cycle_number=cycle,
        )
        print(f"   Critic → {decision['status']}: {decision.get('reason', '')}")

        if decision["status"] == "accept":
            print("✅ Critic accepted the configuration. Stopping search.")
            break

        if stagnation >= 3:
            print("   🧭 Stagnation detected – calling Explorer.")
            config = explorer.propose_escape(performance_history, config, cycle)
            stagnation = 0
        else:
            tracker.add_reasoning_to_memory(decision.get("reason", ""))
            config = actor.propose(
                current_config=config,
                critic_feedback=decision,
                cycle=cycle,
                recent_failures=tracker.get_recent_failures_str(),
            )

    # ------------------------------------------------------------------
    # 4. Report & package
    # ------------------------------------------------------------------
    best_cfg = tracker.best_config or config
    best_mse = tracker.best_mse if tracker.best_mse < float("inf") else float("nan")
    best_rmse = tracker.best_rmse or compute_rmse(best_mse)
    latency = measure_inference_latency(best_model, loader.state_dim, loader.action_dim) if best_model else 0.0

    report_agent = ReportAgent(log_filename=str(log_file))
    abstract = report_agent.generate_report_text(
        env_name=cfg.ENV_NAME,
        state_dim=loader.state_dim,
        action_dim=loader.action_dim,
        best_config=best_cfg,
        best_mse=best_mse,
        best_rmse=best_rmse,
        latency=latency,
        use_pinn=cfg.USE_PINN,
        complexity_label=loader.complexity_label,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = generate_final_pdf(
        env_name=cfg.ENV_NAME,
        abstract=abstract,
        best_config=best_cfg,
        best_mse=best_mse,
        best_rmse=best_rmse,
        latency_ms=latency,
        state_dim=loader.state_dim,
        action_dim=loader.action_dim,
        use_pinn=cfg.USE_PINN,
        output_dir=out_dir,
    )

    extra = []
    if best_model is not None:
        pth = out_dir / f"best_model_{cfg.ENV_NAME}.pth"
        import torch
        torch.save(best_model.state_dict(), pth)
        extra.append(str(pth))

    zip_path = package_final_results_to_zip(
        pdf_filename=pdf_path,
        extra_files=extra,
        output_dir=out_dir,
        env_name=cfg.ENV_NAME,
    )

    cost_tracker.print_summary(cfg.LLM_MODEL)
    print(f"\n✅ AgentSysID finished.")
    print(f"   PDF : {pdf_path}")
    print(f"   ZIP : {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
