"""Bounded, cited evidence for a single run. Never import weights or execute artifacts."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("data_inspector", "initializer", "critic", "actor", "explorer", "report")
MAX_FILE_BYTES = 2_000_000
MAX_CONTEXT_CHARS = 110_000


def redact(text: str) -> str:
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        secret = os.getenv(name, "")
        if len(secret) > 6:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\b(?:sk-|gsk_)[A-Za-z0-9_-]{16,}", "[REDACTED]", text)
    return re.sub(r"(?i)(authorization\s*:\s*bearer\s+|(?:api[_ -]?key|password|access[_ -]?token)\s*[=:]\s*)[^\s,\"'}]+", r"\1[REDACTED]", text)


def clean(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if not re.search(r"(?i)key|secret|password|token", str(k))}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, Path)):
        return redact(str(value))
    if value is None or isinstance(value, (float, int, bool)):
        return value
    return str(value)


def write_json(path: Path, data) -> bool:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".diagnostic-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(clean(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if temporary:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _read(path: Path) -> tuple[str, bool]:
    """Keep both the beginning and the failure-prone tail of very large logs."""
    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            if size <= MAX_FILE_BYTES:
                data = handle.read(MAX_FILE_BYTES)
            else:
                data = handle.read(MAX_FILE_BYTES // 2)
                handle.seek(-MAX_FILE_BYTES // 2, 2)
                data += b"\n[Middle of large file omitted]\n" + handle.read(MAX_FILE_BYTES // 2)
        return redact(data.decode("utf-8", errors="replace").replace("\r\n", "\n")), size > MAX_FILE_BYTES
    except OSError:
        return "", False


def read_json(path: Path, fallback=None):
    text, _ = _read(path)
    try:
        return json.loads(text) if text else fallback
    except (ValueError, TypeError):
        return fallback


def implementation_reference() -> dict:
    # Preserve the real implementation, not only an agent's description of it.
    ranges = {
        "training/scoring.py": [(1, 150)],
        "training/trainer.py": [(1, 18), (505, 562)],
        "data/splitter.py": [(1, 145)],
        "agents/actor.py": [(18, 80), (97, 148)],
        "reporting/plots.py": [(266, 413)],
    }
    out = {}
    for name, spans in ranges.items():
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        out[name] = "\n".join(f"{i+1}: {lines[i]}" for start, end in spans for i in range(start-1, min(end, len(lines))))
    pipeline_lines = (ROOT / "pipeline.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(pipeline_lines) if "# --- Generalization gap check" in line)
    end = next(i for i in range(start, len(pipeline_lines)) if "if final_mse <= cfg.MSE_TARGET" in pipeline_lines[i])
    out["pipeline.py"] = "\n".join(f"{i+1}: {pipeline_lines[i]}" for i in range(start, end+3))
    return out


def snapshot_run(run_dir: Path, options, config, *, phase="initial", data_summary=None) -> bool:
    """Freeze prompts, settings and implementation before they change in another run."""
    path = Path(run_dir) / "run_context.json"
    previous = read_json(path, {})
    if not isinstance(previous, dict):
        previous = {}
    configuration = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith("_")}
    if not previous:
        previous = {
            "schema_version": 1, "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "options": clean(options), "initial_config": clean(configuration),
            "prompts": {name: (ROOT / "prompts" / f"{name}.yaml").read_text(encoding="utf-8") for name in AGENTS},
            "implementation": implementation_reference(),
        }
    previous["phase"] = phase
    previous["effective_config"] = clean(configuration)
    if data_summary is not None:
        previous["data_summary"] = clean(data_summary)
    return write_json(path, previous)


def save_verification(run_dir, true_data, predicted_data, state_names, *, protocol=None) -> bool:
    import numpy as np
    true, predicted = np.asarray(true_data, dtype=float), np.asarray(predicted_data, dtype=float)
    summary = {"true_shape": list(true.shape), "predicted_shape": list(predicted.shape),
               "note": "Held-out state rollout errors, distinct from validation derivative MSE. This does not certify control stability."}
    if protocol is not None:
        summary["protocol"] = clean(protocol)
    if true.ndim == predicted.ndim == 2 and true.shape[1] == predicted.shape[1]:
        n = min(len(true), len(predicted))
        summary["aligned_samples"] = n
        summary["nonfinite_values"] = int((~np.isfinite(true)).sum() + (~np.isfinite(predicted)).sum())
        summary["states"] = []
        for i in range(true.shape[1]):
            valid = np.isfinite(true[:n, i]) & np.isfinite(predicted[:n, i])
            actual, forecast = true[:n, i][valid], predicted[:n, i][valid]
            error = forecast - actual
            scale = float(np.std(actual)) if len(actual) else 0.0
            rmse = float(np.sqrt(np.mean(error**2))) if len(error) else None
            summary["states"].append({"state": state_names[i] if i < len(state_names) else str(i),
                                      "valid_samples": len(actual), "mae": float(np.mean(np.abs(error))) if len(error) else None,
                                      "rmse": rmse, "rmse_over_test_std": rmse/scale if rmse is not None and scale > 0 else None})
    return write_json(Path(run_dir) / "verification_summary.json", summary)


@dataclasses.dataclass
class Source:
    id: str
    title: str
    kind: str
    content: str
    provenance: str


@dataclasses.dataclass
class RunEvidence:
    run_id: str
    manifest: dict
    sources: list[Source]
    coverage: dict

    def to_payload(self):
        return {"run_id": self.run_id, "coverage": self.coverage,
                "sources": [dataclasses.asdict(source) for source in self.sources]}


def collect_evidence(run_dir, question="", *, live_state=None, max_chars=MAX_CONTEXT_CHARS) -> RunEvidence:
    base = Path(run_dir).resolve()
    if not base.is_dir():
        raise ValueError("This run folder is no longer available.")
    sources = []
    coverage = {"missing": [], "notes": [], "agent_turns": 0, "templates": 0, "plot_pixels_read": False}

    def add(identity, title, kind, content, provenance="Saved with this run"):
        if content:
            sources.append(Source(identity, title, kind, redact(content), provenance))

    def add_json(identity, filename, kind):
        path = base / filename
        data = read_json(path) if base in path.resolve().parents else None
        if isinstance(data, (dict, list)):
            add(identity, filename, kind, json.dumps(clean(data), ensure_ascii=False, indent=2))
            return data
        coverage["missing"].append(filename)
        return None

    manifest = add_json("M1", "run_manifest.json", "results") or {}
    if not isinstance(manifest, dict):
        manifest = {}
    # Cycle evidence is separate to allow precise references even with long searches.
    if "performance_history" in manifest:
        sources[0].content = json.dumps(clean({k: v for k, v in manifest.items() if k != "performance_history"}), ensure_ascii=False, indent=2)
    history = manifest.get("performance_history", [])
    if live_state:
        add("F1", "Last observed run state", "runtime", json.dumps(clean(live_state), ensure_ascii=False, indent=2), "Observed by the UI; may be incomplete")
        history = history or live_state.get("history", [])
    if isinstance(history, list):
        for i, row in enumerate(history):
            if isinstance(row, dict):
                add(f"C{i+1}", f"Training cycle {row.get('cycle', i+1)}", "cycle", json.dumps(clean(row), indent=2))
    context_path = base / "run_context.json"
    context = read_json(context_path, {}) if base in context_path.resolve().parents else {}
    if isinstance(context, dict) and context:
        add("R1", "Recorded run settings and data split", "settings", json.dumps({k:v for k,v in context.items() if k not in ("prompts", "implementation")}, indent=2))
        prompts, implementation = context.get("prompts", {}), context.get("implementation", {})
        provenance = "Captured for this run"
    else:
        coverage["missing"].append("run_context.json")
        coverage["notes"].append("Original settings/templates were not captured. Current reference templates and code cannot prove what this older run used; recorded agent turns take priority.")
        prompts = {name: (ROOT / "prompts" / f"{name}.yaml").read_text(encoding="utf-8") for name in AGENTS}
        implementation = implementation_reference()
        provenance = "Current reference; historical version not verified"
    for name, text in prompts.items():
        add(f"P_{name}", f"{name.replace('_', ' ').title()} prompt template", "template", str(text), provenance)
        coverage["templates"] += 1
    for i, (name, text) in enumerate(implementation.items()):
        add(f"I{i+1}", name, "implementation", str(text), provenance)
    add_json("V1", "verification_summary.json", "verification")
    add_json("S1", "diagnostic_state.json", "runtime")
    activity = add_json("E1", "agent_activity.json", "activity")
    log_paths = sorted((base / "Agents_log").glob("*")) + [base / "llm_conversation_history.txt"]
    index = 0
    for path in log_paths:
        # Ignore symlinks outside the selected run and never read manifest-provided paths.
        if not path.is_file() or base not in path.resolve().parents:
            continue
        text, truncated = _read(path)
        if truncated:
            coverage["notes"].append(f"{path.name}: middle of very large log omitted.")
        for block in re.split(r"(?=TIMESTAMP: \[)", text):
            if "AGENT: [" not in block:
                continue
            agent = re.search(r"AGENT: \[([^\]]+)\]", block)
            coverage["agent_turns"] += 1
            # Split long real prompts into individually citable excerpts.
            for start in range(0, len(block), 7000):
                index += 1
                add(f"A{index}", f"{agent.group(1) if agent else 'Agent'} · {path.name} · excerpt {start//7000+1}", "agent_turn", block[start:start+7500])
    if not index:
        coverage["missing"].append("Recorded agent prompts and replies")
    runtime_path = base / "runtime_console.log"
    runtime, truncated = _read(runtime_path) if base in runtime_path.resolve().parents else ("", False)
    if truncated:
        coverage["notes"].append("runtime_console.log: middle of very large log omitted.")
    if runtime:
        for i, start in enumerate(range(0, len(runtime), 6000)):
            add(f"L{i+1}", f"Runtime console · excerpt {i+1}", "console", runtime[start:start+6500])
    else:
        coverage["missing"].append("runtime_console.log")
    knowledge = (ROOT / "prompts" / "diagnostic_knowledge.md").read_text(encoding="utf-8")
    add("K1", "System identification diagnostic guide", "knowledge", knowledge, "Diagnostic reference; not an observation about this run")
    coverage["notes"].append("Plot files and model weights are not interpreted by this text analysis. Use recorded verification metrics; visual conclusions require inspecting the plots.")

    # Prioritize measured data and instructions, then retrieve relevant actual agent turns.
    tokens = set(re.findall(r"[a-z]{3,}", question.lower())) - {"this", "that", "model", "what", "about", "with", "run"}
    required = [s for s in sources if s.kind in ("results", "settings", "verification", "runtime", "knowledge", "template", "implementation")]
    cycles = [s for s in sources if s.kind == "cycle"]
    optional = [s for s in sources if s not in required and s not in cycles]
    def relevance(source):
        text = source.content.lower()
        return sum(text.count(word) for word in tokens) + 15*sum(word in text for word in ("[error]", "traceback", "nan", "overfit", "violation", "ask_human"))
    # Ensure coverage of each agent, the last turns and the last console excerpt.
    representative = {}
    for source in optional:
        group = source.title.split(" · ")[0] if source.kind == "agent_turn" else source.kind
        representative[group] = source
    # A long tuning history must not displace every actual agent prompt.
    priority = required + list(representative.values()) + sorted(cycles, key=relevance, reverse=True) + sorted(optional, key=relevance, reverse=True)
    selected, seen, used = [], set(), 0
    for source in priority:
        if source.id in seen:
            continue
        seen.add(source.id)
        room = max_chars - used
        if room <= 200:
            break
        if len(source.content) > room:
            if source.kind in ("results", "settings", "knowledge"):
                marker = "\n[Excerpt truncated]"
                source = dataclasses.replace(source, content=source.content[:room-len(marker)] + marker)
            else:
                continue
        selected.append(source)
        used += len(source.content)
    coverage.update(sources_available=len(sources), sources_used=len(selected), context_characters=used,
                    fingerprint=hashlib.sha256("\n".join(s.id+s.content for s in selected).encode()).hexdigest())
    if len(selected) < len(sources):
        coverage["notes"].append("The context budget selected relevant excerpts, representative agent turns and measured cycles; not every log excerpt was included.")
    return RunEvidence(base.name, manifest, selected, coverage)
