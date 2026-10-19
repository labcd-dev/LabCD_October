"""Readable activity from pipeline events; stored with each run for replay."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path

_FILE = "agent_activity.json"
_STAGES = {
    "Questionnaire": ("Preparing run assumptions", "Run assumptions prepared"),
    "Loading dataset": ("Loading the dataset", "Dataset loaded"),
    "Data Inspector": ("Inspector is reviewing the dataset", "Inspector review finished"),
    "Splitting trajectories": ("Separating training and test trajectories", "Trajectories split for training and testing"),
    "Initializer Agent": ("Preparing starting model settings", "Starting model settings prepared"),
    "Tuning cycles": ("Searching for the best model", "Model search finished"),
    "Held-out verification": ("Testing the model on unseen trajectories", "Held-out verification finished"),
    "Report & packaging": ("Preparing the report and downloads", "Report and downloads prepared"),
}


def record(items: list, kind: str, payload: dict) -> None:
    """Update stable steps without inventing agent actions from console text."""
    stamp = payload.get("_timestamp") or dt.datetime.now(dt.timezone.utc).isoformat()
    details = {k: v for k, v in payload.items() if not k.startswith("_") and k != "result"}

    def put(key, label, state="complete", data=None):
        row = next((r for r in items if r["id"] == key), None)
        if row is None:
            row = {"id": key, "timestamp": stamp, "details": {}}
            items.append(row)
        row.update(label=label, state=state)
        row["details"].update(data if data is not None else details)
        if state != "running":
            row["finished_at"] = stamp
        return row

    if kind == "stage":
        name = payload["name"]
        key = f"stage:{name}"
        for row in items:
            if row["id"].startswith("stage:") and row["id"] != key and row["state"] == "running":
                stage = row["details"]["name"]
                put(row["id"], _STAGES.get(stage, (stage, stage))[1], data={})
        put(key, _STAGES.get(name, (name, name))[0], "running")
    elif kind == "inspector":
        outcome = payload.get("outcome")
        label = {
            "unavailable": "Inspector unavailable — proceeding with loader checks",
            "clarification_skipped": "Inspector flagged a question — automatic run continued",
            "reviewed": "Inspector checked the dataset",
        }.get(outcome, "Inspector review finished")
        put("stage:Data Inspector", label)
    elif kind == "inspector_data":
        row = next((r for r in items if r["id"] == "stage:Data Inspector"), None)
        if row:
            row["details"].update(details)
    elif kind == "initializer":
        label = "Initializer selected starting model settings" if payload.get("mode") == "agent" else "Manual starting model settings applied"
        put("stage:Initializer Agent", label)
    elif kind in ("report_started", "report_done"):
        started = kind == "report_started"
        label = "Report Agent is preparing the report text" if started else "Report Agent prepared the report text"
        put("report", label, "running" if started else "complete", data={})
    elif kind == "human_checkpoint":
        phase = payload.get("phase", "review")
        label = "Initializer is ready for your review" if phase == "initializer" else "Cycle results are ready for your direction"
        put(f"human:{payload.get('id', phase)}", label, "running")
    elif kind == "human_checkpoint_resolved":
        key = f"human:{payload.get('id', payload.get('phase', 'review'))}"
        label = "You approved the starting model" if payload.get("phase") == "initializer" else "You guided the next candidate"
        put(key, label, "complete", data={"phase": payload.get("phase"), "action": payload.get("action")})
    elif kind == "human_guidance_applied":
        if payload.get("phase") == "queued_candidate":
            cycle = int(payload.get("cycle", 0) or 0)
            put(f"guidance:queued_candidate:{cycle}",
                f"Queued settings applied before candidate cycle {cycle}")
            return
        labels = {
            "accept": "Initializer recommendation accepted",
            "custom": "Client approved edited starting settings",
            "compact": "Search steered toward smaller models",
            "regularize": "Search steered toward stronger regularization",
            "widen": "Search widened within the approved limits",
            "continue": "Search continued with the agent’s suggestion",
            "stop": "Client stopped additional candidate search",
        }
        if phase == "early_results" and payload.get("action") == "custom":
            label = "Client set parameters for the next candidate"
        else:
            label = labels.get(payload.get("action"), "Client updated the candidate search")
        put(f"guidance:{payload.get('phase', 'run')}:{len(items)}",
            label, "complete")
    elif kind == "candidate_edit_rejected":
        cycle = int(payload.get("cycle", 0) or 0)
        put(f"guidance:queued_candidate_rejected:{cycle}",
            f"Queued settings for cycle {cycle} were not applied", "error")
    elif kind in ("cycle_started", "cycle", "latency"):
        cycle = payload["cycle"]
        key = f"actor:{cycle}"
        if kind == "latency":
            row = next((r for r in items if r["id"] == key), None)
            if row:
                row["details"].update(details)
        else:
            label = f"Actor is training cycle {cycle}" if kind == "cycle_started" else f"Actor finished cycle {cycle}"
            if kind == "cycle" and payload.get("is_best"):
                label += " — new best model"
            put(key, label, "running" if kind == "cycle_started" else "complete")
    elif kind in ("critic_started", "critic", "explorer_started", "explorer"):
        cycle = payload["cycle"]
        agent = kind.split("_")[0]
        label = f"Critic is reviewing cycle {cycle}" if agent == "critic" else f"Explorer is searching for a new architecture after cycle {cycle}"
        if not kind.endswith("_started"):
            label = f"Critic reviewed cycle {cycle} and proposed adjustments" if agent == "critic" else f"Explorer proposed a new architecture after cycle {cycle}"
        put(f"{agent}:{cycle}", label, "running" if kind.endswith("_started") else "complete")
    elif kind in ("done", "error"):
        result = payload.get("result")
        success = kind == "done" and getattr(result, "status", None) == "completed"
        for row in items:
            if row["state"] == "running":
                stage = row["details"].get("name")
                label = _STAGES.get(stage, (row["label"], row["label"]))[1] if success else f"Step interrupted: {row['label']}"
                put(row["id"], label, "complete" if success else "error", data={})
        if result is not None:
            details = {k: getattr(result, k, None) for k in ("status", "message", "cycles_run", "best_mse", "elapsed_seconds")}
        put("run:finished", "Run completed — results are ready" if success else "Run failed — expand for details", "complete" if success else "error", details)
    elif kind == "stop_requested":
        put("run:stop", "Stop requested — finishing the current step")


def save(run_dir: str | Path, items: list) -> bool:
    """Write atomically so history never reads a partially saved feed."""
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir, prefix=".activity-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(clean(items), handle, ensure_ascii=False, allow_nan=False, default=str)
        os.replace(temporary, Path(run_dir) / _FILE)
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if temporary and temporary.exists():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def load(run_dir: str | Path) -> list:
    try:
        items = json.loads((Path(run_dir) / _FILE).read_text(encoding="utf-8"))
        if not isinstance(items, list):
            return []
        return [row for row in items if isinstance(row, dict) and isinstance(row.get("id"), str)
                and isinstance(row.get("label"), str) and isinstance(row.get("details"), dict)
                and row.get("state") in ("running", "complete", "error")]
    except (OSError, ValueError):
        return []


def render(items: list, *, historical: bool = False, scope: str = "live",
           heading: str | None = "Agent activity feed", container_height: int = 420) -> None:
    import hashlib
    import streamlit as st

    if heading:
        st.markdown(f"##### {heading}")
    if not items:
        st.caption("Activity was not recorded for this older run." if historical else
                   "Start a run to see Inspector findings, training updates, and agent decisions here.")
        return
    st.caption("Expand an update for findings, settings, feedback, and any recorded agent log." +
               (" · Saved with this run." if historical else ""))
    icons = {"running": ":material/hourglass_top:", "complete": ":material/check_circle:", "error": ":material/error:"}
    with st.container(height=container_height, border=True, key=f"agent_activity_feed_{scope}"):
        for row in items:
            key = hashlib.sha1(f"{scope}:{row['id']}".encode()).hexdigest()[:16]
            # Literal text avoids rendering provider output as Markdown/HTML.
            label = row["label"].replace("_", "\\_")
            with st.expander(label, key=f"activity_{key}", type="step", icon=icons[row["state"]]):
                try:
                    stamp = dt.datetime.fromisoformat(row["timestamp"]).astimezone().strftime("%H:%M:%S")
                except (KeyError, TypeError, ValueError):
                    stamp = ""
                status_label = {"running": "Running", "complete": "Done", "error": "Failed"}[row["state"]]
                st.caption(f"{status_label} · {stamp}")
                details = dict(row["details"])
                for field, title in (("response", "Inspector findings"), ("reasoning", "Agent feedback"), ("message", "Message")):
                    value = details.pop(field, None)
                    if value:
                        st.caption(title)
                        st.text(str(value))
                agent_log = details.pop("agent_log", None)
                if agent_log:
                    st.caption("Agent log")
                    entries = agent_log if isinstance(agent_log, list) else [agent_log]
                    for entry in entries:
                        if entry:
                            st.markdown(str(entry), unsafe_allow_html=False)
                if details:
                    st.json(details, expanded=False)
