"""
LangGraph workflow for the AgentSysID pipeline.

The graph wires the same agents and trainer the CLI uses:

    inspect ──► initialize ──► train_cycle ──► evaluate ──┐
                                   ▲                      │
                                   └──── route (actor / explorer) ◄┘
                                                          │
                                                       finalize ──► END

``run_cli.py`` remains the production entry point (it also owns the
questionnaire and the interactive overrides); this module exposes the same
pipeline as a graph for orchestrators that prefer LangGraph — the FastAPI
adapter in ``backend_api/AgentSysID`` can drive either.

``build_sysid_graph`` returns ``None`` when langgraph is not installed, so
callers can fall back to the sequential runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.graph.state import SysIDState


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_inspect(state: SysIDState) -> Dict[str, Any]:
    """Load the dataset, audit it, and run the HIL Data Inspector."""
    from backend_core.AgentSysID.agents import run_data_inspector_agent
    from backend_core.AgentSysID.data import ExcelDataLoader, split_trajectories

    loader = ExcelDataLoader(state["data_path"])
    engineer_notes, cols_to_drop = run_data_inspector_agent(
        loader, interactive=bool(state.get("interactive", False))
    )
    if cols_to_drop:
        loader.drop_columns(cols_to_drop)

    trajectories = loader.get_trajectories()
    arch = str(state.get("architecture") or cfg.NETWORK_ARCHITECTURE).upper()
    train_trajs, val_trajs, test_trajs = split_trajectories(trajectories, architecture=arch)

    return {
        "_loader": loader,  # carried for downstream nodes
        "state_dim": loader.state_dim,
        "action_dim": loader.action_dim,
        "complexity_tier": loader.complexity_tier,
        "complexity_label": loader.complexity_label,
        "quality_issues": list(loader.quality_issues),
        "engineer_notes": engineer_notes,
        "dropped_columns": list(cols_to_drop),
        "trajectories": trajectories,
        "train_trajs": train_trajs,
        "val_trajs": val_trajs,
        "test_trajs": test_trajs,
        "architecture": arch,
        "cycle": 0,
        "stagnation": 0,
        "performance_history": [],
        "status": "running",
    }


def node_initialize(state: SysIDState) -> Dict[str, Any]:
    """Ask the Initializer for the starting config and the search bounds."""
    from backend_core.AgentSysID.agents import InitializerAgent

    loader = state["_loader"]  # type: ignore[index]
    setup_config = InitializerAgent(loader).determine_initial_setup()

    return {
        "setup_config": setup_config,
        "activation": setup_config.get("activation", "relu"),
        "current_config": {
            "learning_rate": setup_config.get("learning_rate", cfg.MANUAL_STARTING_LR),
            "hidden_layers": setup_config.get("hidden_layers", [64]),
            "dropout_rate": setup_config.get("dropout_rate", cfg.DROPOUT_RATE),
            "weight_decay": setup_config.get("weight_decay", cfg.WEIGHT_DECAY),
            "batch_size": setup_config.get("batch_size", cfg.BATCH_SIZE),
            "patience": setup_config.get("early_stop_patience", cfg.EARLY_STOP_PATIENCE),
        },
        "max_cycles": int(
            state.get("max_cycles") or cfg.run_mode_limits(state.get("run_mode"))["max_cycles"]
        ),
    }


def node_train_cycle(state: SysIDState) -> Dict[str, Any]:
    """Train one candidate architecture and record its performance."""
    from backend_core.AgentSysID.training import (
        measure_inference_latency,
        train_dynamics_model,
    )

    cycle = int(state.get("cycle", 0)) + 1
    config = dict(state["current_config"])

    model, train_mse, val_mse, val_rmse, _, _ = train_dynamics_model(
        state["train_trajs"],
        state["val_trajs"],
        state["state_dim"],
        state["action_dim"],
        hidden_layers=config["hidden_layers"],
        learning_rate=config["learning_rate"],
        epochs=state.get("setup_config", {}).get("epochs", cfg.EPOCHS),
        batch_size=config.get("batch_size", cfg.BATCH_SIZE),
        patience=config.get("patience", cfg.EARLY_STOP_PATIENCE),
        activation=state.get("activation", "relu"),
        dropout_rate=config.get("dropout_rate", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
        architecture=state.get("architecture"),
    )

    if train_mse > 1e-8 and (val_mse / train_mse) > cfg.OVERFIT_RATIO_LIMIT:
        val_mse, val_rmse = 9999.0, 99.0

    latency = measure_inference_latency(model, state["state_dim"], state["action_dim"])

    history = list(state.get("performance_history", []))
    history.append(
        {
            "cycle": cycle,
            "iteration": cycle - 1,
            "config": config,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "rmse": val_rmse,
            "latency": latency,
            "performance": {"mse": val_mse, "rmse": val_rmse, "latency": latency},
        }
    )

    return {
        "cycle": cycle,
        "performance_history": history,
        "_model": model,
        "_train_mse": train_mse,
        "_val_mse": val_mse,
        "_val_rmse": val_rmse,
        "latency_ms": latency,
    }


def node_evaluate(state: SysIDState) -> Dict[str, Any]:
    """Update the tracker, then have the Critic diagnose the cycle."""
    tracker = state.get("_tracker")  # type: ignore[assignment]
    if tracker is None:
        from backend_core.AgentSysID.training import BestConfigTracker

        tracker = BestConfigTracker(run_mode=str(state.get("run_mode", "regular")))

    from backend_core.AgentSysID.agents import CriticAgent

    is_new_best = tracker.update(
        state["_val_mse"], state["current_config"], current_rmse=state["_val_rmse"]  # type: ignore[index]
    )
    best_model = state.get("_best_model") or state.get("_model")
    if is_new_best:
        best_model = state.get("_model")

    critic = CriticAgent(tracker, run_mode=str(state.get("run_mode", "regular")))
    critic_output = critic.evaluate(
        train_mse=state["_train_mse"],  # type: ignore[index]
        val_mse=state["_val_mse"],  # type: ignore[index]
        current_config=state["current_config"],
        activation=state.get("activation", "relu"),
        measured_latency=state["latency_ms"],
        max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
        cycle_number=int(state["cycle"]),
        val_rmse=state["_val_rmse"],  # type: ignore[index]
    )
    tracker.add_reasoning_to_memory(critic_output.get("reasoning", ""))

    return {
        "_tracker": tracker,
        "_best_model": best_model,
        "critic_output": critic_output,
        "best_config": tracker.best_config or {},
        "best_mse": tracker.best_mse,
        "best_rmse": tracker.best_rmse or 0.0,
        "stagnation": 0 if is_new_best else int(state.get("stagnation", 0)) + 1,
    }


def node_route(state: SysIDState) -> Dict[str, Any]:
    """Fine-tune with the Actor, or escape the local minimum with the Explorer."""
    from backend_core.AgentSysID.agents import ActorAgent, ExplorerAgent

    tracker = state["_tracker"]  # type: ignore[index]
    setup_config = state.get("setup_config", {})

    actor = state.get("_actor")
    if actor is None:
        actor = ActorAgent(state.get("activation", "relu"), initial_config=setup_config)
    actor.current_config = dict(state["current_config"])

    if int(state.get("stagnation", 0)) >= 3:
        explorer = ExplorerAgent(initial_config=setup_config)
        new_config, _reasoning = explorer.generate_radical_escape(
            tracker=tracker,
            stuck_config=actor.current_config,
            visited_configs=actor.visited_configs,
            cycle_number=int(state["cycle"]),
        )
        actor.current_config = new_config
        actor._add_to_visited(new_config)
        return {"_actor": actor, "current_config": new_config, "stagnation": 0}

    new_config = actor.apply_critic_feedback(
        state.get("critic_output", {}), tracker, cycle=int(state["cycle"])
    )
    return {"_actor": actor, "current_config": new_config}


def node_finalize(state: SysIDState) -> Dict[str, Any]:
    """Verification rollout, success score, report text, PDF and ZIP."""
    from pathlib import Path

    from backend_core.AgentSysID.agents import ReportAgent
    from backend_core.AgentSysID.reporting import (
        export_standalone_inference_script,
        generate_all_plots,
        generate_final_pdf,
        package_final_results_to_zip,
        plot_test_dataset_verification,
        save_best_model,
        write_nn_inference_helper,
    )
    from backend_core.AgentSysID.training import calculate_success_score, measure_inference_latency
    from backend_core.AgentSysID.utils import setup_run_dir

    model = state.get("_best_model") or state.get("_model")
    if model is None:
        return {"status": "failed"}

    run_dir = Path(state.get("output_dir") or setup_run_dir(env_name=cfg.ENV_NAME))
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = cfg.RUN_TIMESTAMP
    arch = str(state.get("architecture") or cfg.NETWORK_ARCHITECTURE).upper()
    best_cfg = state.get("best_config") or state["current_config"]

    pth = save_best_model(
        model,
        list(best_cfg.get("hidden_layers", [64])),
        state["state_dim"],
        state["action_dim"],
        float(state.get("best_mse", 0.0)),
        best_cfg,
        state.get("activation", "relu"),
        env_name=cfg.ENV_NAME,
        timestamp=stamp,
        output_dir=run_dir,
        rollout_horizon=int(cfg.ROLLOUT_HORIZON),
    )
    export_standalone_inference_script(
        model=model,
        hidden_layers=list(best_cfg.get("hidden_layers", [64])),
        activation=state.get("activation", "relu"),
        state_dim=state["state_dim"],
        action_dim=state["action_dim"],
        pth_stem=pth.name,
        env_name=cfg.ENV_NAME,
        output_dir=run_dir,
        architecture=arch,
        lstm_seq_length=int(cfg.LSTM_SEQ_LENGTH),
    )
    write_nn_inference_helper(cfg.ENV_NAME, run_dir, integrator=str(cfg.INTEGRATOR_TYPE))

    generate_all_plots(
        performance_history=state.get("performance_history", []),
        env_name=cfg.ENV_NAME,
        output_dir=run_dir,
        timestamp=stamp,
        max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
        prefix=cfg.PLOT_FILENAME_PREFIX,
    )

    true_traj, nn_traj = plot_test_dataset_verification(
        model,
        state.get("test_trajs", []),
        state["state_dim"],
        state["action_dim"],
        output_dir=run_dir,
        timestamp=stamp,
        prefix=cfg.PLOT_FILENAME_PREFIX,
        env_name=cfg.ENV_NAME,
        architecture=arch,
    )
    score, status = calculate_success_score(
        val_mse=float(state.get("best_mse", 0.0)),
        true_trajectory=true_traj,
        nn_trajectory=nn_traj,
        complexity_label=state.get("complexity_label", "Unknown"),
    )

    latency = measure_inference_latency(model, state["state_dim"], state["action_dim"])
    abstract, conclusion = ReportAgent().generate_report_text(
        env_name=cfg.ENV_NAME,
        state_dim=state["state_dim"],
        action_dim=state["action_dim"],
        best_config=best_cfg,
        best_mse=float(state.get("best_mse", 0.0)),
        best_rmse=float(state.get("best_rmse", 0.0)),
        latency=latency,
        use_pinn=cfg.USE_PINN,
        max_latency=cfg.CUSTOMER_MAX_LATENCY_MS,
        success_score=score,
        model_status=status,
        complexity_label=state.get("complexity_label", "Unknown"),
        architecture=arch,
    )

    pdf_path = generate_final_pdf(
        env_name=cfg.ENV_NAME,
        state_dim=state["state_dim"],
        action_dim=state["action_dim"],
        best_config=best_cfg,
        best_mse=float(state.get("best_mse", 0.0)),
        best_rmse=float(state.get("best_rmse", 0.0)),
        latency=latency,
        use_pinn=cfg.USE_PINN,
        timestamp=stamp,
        abstract_text=abstract,
        conclusion_text=conclusion,
        success_score=score,
        model_status=status,
        complexity_label=state.get("complexity_label", "Unknown"),
        output_dir=run_dir,
        architecture=arch,
        rollout_horizon=int(cfg.ROLLOUT_HORIZON),
        lstm_seq_length=int(cfg.LSTM_SEQ_LENGTH),
        prefix=cfg.PLOT_FILENAME_PREFIX,
    )
    zip_path = package_final_results_to_zip(run_dir, cfg.ENV_NAME, stamp, pdf_path)

    return {
        "latency_ms": latency,
        "success_score": score,
        "model_status": status,
        "abstract": abstract,
        "conclusion": conclusion,
        "pdf_path": pdf_path,
        "zip_path": zip_path,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def should_continue(state: SysIDState) -> str:
    """Continue tuning, or stop on target / cycle limit / user stop."""
    from backend_core.AgentSysID.utils import stop_requested

    if stop_requested():
        return "finalize"
    if float(state.get("best_mse", float("inf"))) <= cfg.MSE_TARGET:
        return "finalize"
    if int(state.get("cycle", 0)) >= int(state.get("max_cycles", 20)):
        return "finalize"
    return "route"


def build_sysid_graph() -> Optional[Any]:
    """
    Construct and compile the LangGraph StateGraph for the SysID pipeline.

    Returns the compiled graph, or None when langgraph is unavailable so the
    caller can fall back to ``run_cli.main``.
    """
    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except Exception:
        return None

    try:
        graph = StateGraph(SysIDState)

        graph.add_node("inspect", node_inspect)
        graph.add_node("initialize", node_initialize)
        graph.add_node("train_cycle", node_train_cycle)
        graph.add_node("evaluate", node_evaluate)
        graph.add_node("route", node_route)
        graph.add_node("finalize", node_finalize)

        graph.set_entry_point("inspect")
        graph.add_edge("inspect", "initialize")
        graph.add_edge("initialize", "train_cycle")
        graph.add_edge("train_cycle", "evaluate")
        graph.add_conditional_edges(
            "evaluate", should_continue, {"route": "route", "finalize": "finalize"}
        )
        graph.add_edge("route", "train_cycle")
        graph.add_edge("finalize", END)

        return graph.compile()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"   ⚠️ LangGraph wiring failed ({exc}); use run_cli.main instead.")
        return None
