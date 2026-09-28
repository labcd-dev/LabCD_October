"""
Run history.

Every completed run writes ``run_manifest.json`` into its own folder
(`pipeline.SysIDResult.save_manifest`). This module reads those manifests back
so the UI can list past runs across restarts, the way a cloud app lists past
sessions. It never re-reads the model or the trajectories — only the small
JSON summary and the artefact paths it names.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_METADATA_FILE = "history_metadata.json"


def _load_metadata(run_dir: Path) -> Dict[str, Any]:
    """Read sidebar preferences separately from the pipeline's manifest."""
    try:
        data = json.loads((run_dir / _METADATA_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        title = data.get("title")
        return {
            "title": title.strip()[:120] if isinstance(title, str) else "",
            "pinned": data.get("pinned") is True,
            "archived": data.get("archived") is True,
        }
    except (OSError, ValueError):
        return {}


def update_metadata(run_dir: str | Path, **changes: Any) -> bool:
    """Atomically save a display name, pin or archive flag; never move a run."""
    path = Path(run_dir)
    if not path.is_dir() or set(changes) - {"title", "pinned", "archived"}:
        return False
    if "title" in changes:
        title = changes["title"]
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 120:
            return False
        changes["title"] = title.strip()
    if any(type(changes[key]) is not bool for key in ("pinned", "archived") if key in changes):
        return False
    data = _load_metadata(path)
    data.update(changes)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path, prefix=".history-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path / _METADATA_FILE)
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def display_name(entry: Dict[str, Any]) -> str:
    return entry.get("title") or entry.get("env_name") or entry.get("name", "run")


def group_history(
    runs: List[Dict[str, Any]], *, archived: bool = False, query: str = "",
    today: Optional[datetime.date] = None,
) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Filter first, then keep favorites above runs grouped by local date."""
    today = today or datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    query = query.strip().casefold()
    groups: Dict[str, List[Dict[str, Any]]] = {
        label: [] for label in ("Pinned", "Today", "Yesterday", "Older")
    }
    for entry in runs:
        if bool(entry.get("archived")) != archived:
            continue
        searchable = f"{display_name(entry)} {entry.get('env_name', '')} {entry.get('name', '')} {summarise(entry)}"
        if query and query not in searchable.casefold():
            continue
        when = entry.get("started")
        day = when.date() if isinstance(when, datetime.datetime) else None
        if entry.get("pinned") and not archived:
            group = "Pinned"
        elif day == today:
            group = "Today"
        elif day == yesterday:
            group = "Yesterday"
        else:
            group = "Older"
        groups[group].append(entry)
    return [(label, entries) for label, entries in groups.items() if entries]


def _parse_stamp(run_dir: Path) -> Optional[datetime.datetime]:
    """run_YYYYMMDD_HHMMSS_<env> -> datetime."""
    parts = run_dir.name.split("_")
    if len(parts) >= 3:
        try:
            return datetime.datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def load_history(output_dir: str | Path, limit: Optional[int] = 60) -> List[Dict[str, Any]]:
    """
    Return past runs, newest first.

    A run folder without a manifest (an older run, or one that crashed before
    packaging) still appears, marked incomplete, so nothing silently vanishes
    from the list.
    """
    base = Path(output_dir)
    if not base.is_dir():
        return []

    runs: List[Dict[str, Any]] = []
    for run_dir in base.glob("run_*"):
        if not run_dir.is_dir():
            continue

        stamp = _parse_stamp(run_dir)
        manifest_path = run_dir / "run_manifest.json"
        entry: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "name": run_dir.name,
            "started": stamp,
            "complete": False,
        }

        if manifest_path.is_file():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                entry.update(data)
                entry["complete"] = data.get("status") == "completed"
                entry["run_dir"] = str(run_dir)  # authoritative: where it is now
            except Exception:
                pass
        else:
            # Recover what we can from the folder itself.
            entry["env_name"] = "_".join(run_dir.name.split("_")[3:]) or "run"
            pdfs = list((run_dir / "report").glob("*.pdf")) if (run_dir / "report").is_dir() else []
            zips = list(run_dir.glob("*.zip"))
            entry["pdf_path"] = str(pdfs[0]) if pdfs else None
            entry["zip_path"] = str(zips[0]) if zips else None

        entry.update(_load_metadata(run_dir))
        # The folder identifies the run even when a manifest has a stale name.
        entry["name"] = run_dir.name
        entry["started"] = stamp
        runs.append(entry)

    runs.sort(key=lambda r: (r.get("started") or datetime.datetime.min), reverse=True)
    return runs if limit is None else runs[:limit]


def relative_age(when: Optional[datetime.datetime]) -> str:
    """'just now' / '14m ago' / '3h ago' / '2d ago' / a date."""
    if when is None:
        return "—"
    delta = datetime.datetime.now() - when
    seconds = delta.total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return when.strftime("%d %b")


def summarise(entry: Dict[str, Any]) -> str:
    """A one-line description for the history row."""
    bits = []
    if entry.get("architecture"):
        bits.append(str(entry["architecture"]))
    if entry.get("run_mode"):
        bits.append(str(entry["run_mode"]))
    if entry.get("use_pinn"):
        bits.append("PINN")
    cycles = entry.get("cycles_run")
    if cycles:
        bits.append(f"{cycles} cyc")
    return " · ".join(bits) if bits else "incomplete"


def delete_run(run_dir: str | Path) -> bool:
    """Remove one run folder. Returns True when it is gone."""
    import shutil

    path = Path(run_dir)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        return not path.exists()
    except Exception:
        return False


def existing_files(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    The artefacts of a past run that are still on disk.

    Paths are re-anchored to the run folder, so a history entry keeps working
    after the artifacts directory has been moved or renamed.
    """
    run_dir = Path(entry.get("run_dir", ""))
    out: Dict[str, str] = {}
    if not run_dir.is_dir():
        return out

    candidates = {
        "Results ZIP": sorted(run_dir.glob("SystemID_RunResults_*.zip")),
        "PDF report": sorted((run_dir / "report").glob("*.pdf")),
        "Model weights (.pth)": sorted((run_dir / "deployment").glob("*.pth")),
        "Standalone controller": sorted((run_dir / "deployment").glob("deployed_controller_*.py")),
        "NN.py inference script": sorted((run_dir / "deployment").glob("NN.py")),
        "Agent conversation log": sorted((run_dir / "Agents_log").glob("*")),
    }
    for label, matches in candidates.items():
        files = [m for m in matches if m.is_file()]
        if files:
            out[label] = str(files[0])
    return out


def figures_of(entry: Dict[str, Any]) -> List[str]:
    """Diagnostic figures still on disk, verification plot first."""
    run_dir = Path(entry.get("run_dir", ""))
    fig_dir = run_dir / "figures"
    if not fig_dir.is_dir():
        return []
    figures = sorted(str(p) for p in fig_dir.glob("*.png"))
    verification = [f for f in figures if "test_verification" in f]
    others = [f for f in figures if "test_verification" not in f]
    return verification + others
