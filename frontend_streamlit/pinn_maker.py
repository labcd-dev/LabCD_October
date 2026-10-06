"""Safe, data-bound conversion of raw physics equations and MATLAB .m files.

Supported Python tensor assignments are translated locally from a restricted
AST. Other source formats can be interpreted by the configured model, after
which this module validates the map and compiles it itself. Uploaded source
and model-generated text are never executed as Python or MATLAB.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field

from backend_core.AgentSysID.agents.prompt_library import system_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings
from backend_core.AgentSysID.agents.run_evidence import redact

try:
    from . import conversation_core as core
except ImportError:
    import conversation_core as core


MAX_SOURCE_BYTES = 120_000
MAX_EXPRESSION_CHARS = 1200
MAX_AST_NODES = 160
_REQUEST_VERBS = re.compile(r"\b(?:use|prepare|make|create|convert|map|apply|enable|add|try|run)\b", re.I)
_PINN_TERMS = re.compile(r"\b(?:pinn|physics(?:[- ]informed)?|equations?|derivatives?|xdot|d\s*\w+\s*/\s*dt)\b", re.I)
_CONTINUATION = re.compile(r"\b(?:prepare|use|map|convert|equation|pinn|ready|yes|confirm|try|apply|retry)\b", re.I)
_ORDINARY_QUESTION = re.compile(r"^\s*(?:what|why|which|how|show|plot|analy[sz]e|start|run|explain|compare)\b", re.I)
_PINN_USAGE_QUESTION = re.compile(
    r"\b(?:do|does|did)\s+(?:you|we|this(?:\s+run|\s+model)?|the\s+run|the\s+model)\s+"
    r"(?:use|apply|include|train(?:ed)?\s+with)\b.{0,100}\b(?:pinn|physics[- ]informed)\b|"
    r"\b(?:is|are|was|were|has|have)\b.{0,80}\b(?:pinn|physics[- ]informed)\b.{0,60}"
    r"\b(?:used|enabled|included|applied|active)\b.{0,40}\b(?:run|training|model)\b|"
    r"^\s*(?:is|are|was|were|has|have)\b.{0,60}\b(?:pinn|physics[- ]informed)\b.{0,40}"
    r"\b(?:on|off|used|enabled|included|applied|active)\b|"
    r"\b(?:pinn|physics[- ]informed)\b.{0,50}\b(?:in|used in|enabled in)\b.{0,30}"
    r"\b(?:this|the|current)?\s*(?:run|training|model)\b",
    re.I,
)


class EquationTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: str = Field(min_length=1, max_length=160)
    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)


class EquationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ready", "clarification"]
    # Some structured-output providers return an empty explanation even when
    # the actual equation mapping is complete. The application supplies a
    # safe, deterministic client-facing fallback in that case.
    message: str = Field(default="", max_length=5000)
    equations: list[EquationTerm] = Field(default_factory=list, max_length=100)
    assumptions: list[str] = Field(default_factory=list, max_length=30)


_UNARY_FUNCTIONS = {
    "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
    "asin": torch.asin, "acos": torch.acos, "atan": torch.atan,
    "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
    "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "abs": torch.abs,
}
_BINARY_FUNCTIONS = {"minimum": torch.minimum, "maximum": torch.maximum, "atan2": torch.atan2}
_TORCH_FUNCTIONS = {**{key: f"torch.{key}" for key in _UNARY_FUNCTIONS},
                    "minimum": "torch.minimum", "maximum": "torch.maximum", "atan2": "torch.atan2"}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_PYTHON_TENSOR_REFERENCE = re.compile(r"\b(?:states|actions)\s*\[\s*:")
_PYTHON_OUTPUT_REFERENCE = re.compile(r"\bphysics_xdot\s*\[\s*:")


def is_raw_equation_request(question: str, *, has_pending_source: bool | str = False) -> bool:
    """Recognize equation preparation without catching ordinary PINN questions."""
    text = (question or "").strip()
    if not text:
        return False
    if _PINN_USAGE_QUESTION.search(text):
        return False
    if has_pending_source:
        if _CONTINUATION.search(text):
            return True
        if has_pending_source == "needs_clarification" and not _ORDINARY_QUESTION.search(text):
            return True
    has_formula = bool(re.search(
        r"(?:\bxdot[_\w]*\s*=|\bd\s*[_a-zA-Z]\w*\s*/\s*dt\s*=|"
        r"\b[a-zA-Z]\w*\s*dot\s*=|\bderivative\s+of\b.{0,60}=)", text, re.I))
    if has_formula:
        return True
    has_intent = bool(_REQUEST_VERBS.search(text))
    return has_intent and bool(_PINN_TERMS.search(text)) and not bool(
        re.search(r"\b(?:what is|explain|difference|compare|how does)\b", text, re.I)
    )


def is_pinn_usage_question(question: str) -> bool:
    """Recognize questions about whether PINN is active, not equation commands."""
    return bool(_PINN_USAGE_QUESTION.search(question or ""))


def _chat_folder(chat_id: str, subfolder: str) -> Path:
    # chat_path validates the ID before it becomes a path component.
    core.chat_path(chat_id)
    return Path(core.UPLOAD_DIR) / chat_id / subfolder


def save_source(chat_id: str, name: str, raw: bytes) -> dict:
    """Save the uploaded/pasted source byte-for-byte under this conversation."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("Use a non-empty PINN or MATLAB source file smaller than 120 KB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("I couldn't read this source as UTF-8 text. Save the .m file as UTF-8 and try again.") from exc
    if not text.strip():
        raise ValueError("This source file is empty. Add the equations and try again.")
    suffix = Path(name).suffix.lower()
    if suffix not in (".m", ".txt"):
        suffix = ".txt"
    digest = hashlib.sha256(raw).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name)[:100] or f"physics_source{suffix}"
    folder = _chat_folder(chat_id, "pinn_sources")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{digest[:16]}_{safe_name}"
    if not destination.exists():
        destination.write_bytes(raw)
    return {"kind": "pinn_source", "name": Path(name).name, "path": str(destination.resolve()),
            "sha256": digest, "size": len(raw), "status": "pending"}


def _read_source(source: dict) -> str:
    path = Path(source.get("path", ""))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source.get("sha256"):
        raise ValueError("The saved equation source changed. Attach or paste it again before preparing PINN.")
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("The saved equation source is larger than the supported 120 KB limit.")
    return raw.decode("utf-8-sig")


def _parse_expression(expression: str, state_count: int, action_count: int) -> ast.Expression:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("An equation is longer than the supported expression limit.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("An equation could not be read as a mathematical expression. I need one clarification before using it.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("An equation is too complex for the safe PINN expression language.")
    state_names = {f"s{i}" for i in range(state_count)}
    action_names = {f"a{i}" for i in range(action_count)}
    allowed_names = state_names | action_names | {"pi"} | set(_UNARY_FUNCTIONS) | set(_BINARY_FUNCTIONS)
    for node in nodes:
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(float(node.value)):
                raise ValueError("Equations may contain only finite numeric constants.")
            continue
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"The expression uses unknown symbol `{node.id}`. Use the supplied s0/a0 aliases or clarify its mapping.")
            if node.id in state_names and int(node.id[1:]) >= state_count:
                raise ValueError(f"The expression refers to unavailable state alias `{node.id}`.")
            if node.id in action_names and int(node.id[1:]) >= action_count:
                raise ValueError(f"The expression refers to unavailable input alias `{node.id}`.")
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            if isinstance(node.op, ast.Pow) and not (
                isinstance(node.right, ast.Constant) and type(node.right.value) in (int, float)
                and math.isfinite(float(node.right.value)) and abs(float(node.right.value)) <= 8
            ):
                raise ValueError("Powers must use a finite numeric exponent between -8 and 8.")
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            function = node.func.id
            expected = 2 if function in _BINARY_FUNCTIONS else 1 if function in _UNARY_FUNCTIONS else 0
            if not expected or len(node.args) != expected:
                raise ValueError(f"Function `{function}` is not supported with that number of arguments.")
            continue
        if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            continue
        raise ValueError("The equation contains code outside the safe mathematical expression language.")
    return tree


def _python_tensor_source(source_text: str) -> bool:
    """Return whether a source looks like the supported Python tensor form."""
    return bool(_PYTHON_TENSOR_REFERENCE.search(source_text) and
                _PYTHON_OUTPUT_REFERENCE.search(source_text))


def _tensor_column_index(node: ast.AST, tensor_name: str) -> int | None:
    """Read only the exact ``tensor[:, integer]`` indexing used by PINN snippets."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name) or node.value.id != tensor_name:
        return None
    index = node.slice
    if not isinstance(index, ast.Tuple) or len(index.elts) != 2:
        return None
    row, column = index.elts
    if (not isinstance(row, ast.Slice) or row.lower is not None or row.upper is not None or row.step is not None or
            not isinstance(column, ast.Constant) or type(column.value) is not int):
        return None
    return column.value


class _PythonEquationRewriter(ast.NodeTransformer):
    """Expand trusted aliases from a restricted, non-executed Python AST."""
    def __init__(self, values: dict[str, ast.expr], state_count: int, action_count: int):
        self.values = values
        self.state_count = state_count
        self.action_count = action_count

    def visit_Name(self, node: ast.Name):
        if node.id in self.values:
            return copy.deepcopy(self.values[node.id])
        if node.id == "pi":
            return ast.copy_location(ast.Name(id="pi", ctx=ast.Load()), node)
        raise ValueError(f"The Python source uses `{node.id}` before it is defined or mapped to a state/input column.")

    def visit_Subscript(self, node: ast.Subscript):
        for tensor_name, alias, count, label in (
            ("states", "s", self.state_count, "state"),
            ("actions", "a", self.action_count, "input"),
        ):
            if isinstance(node.value, ast.Name) and node.value.id == tensor_name:
                index = _tensor_column_index(node, tensor_name)
                if index is None:
                    raise ValueError(f"Use only `{tensor_name}[:, index]` to select a measured {label} column.")
                if index < 0 or index >= count:
                    raise ValueError(f"The Python source selects {label} index {index}, but the confirmed dataset has {count} {label} column(s).")
                return ast.copy_location(ast.Name(id=f"{alias}{index}", ctx=ast.Load()), node)
        raise ValueError("The Python source contains tensor indexing outside the supported state/input columns.")

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "pi":
            return ast.copy_location(ast.Name(id="pi", ctx=ast.Load()), node)
        raise ValueError("Only approved math functions such as `torch.sin` are supported from Python modules.")

    def visit_Call(self, node: ast.Call):
        function = node.func.id if isinstance(node.func, ast.Name) else None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
            function = node.func.attr
        if function not in _UNARY_FUNCTIONS | _BINARY_FUNCTIONS or node.keywords:
            raise ValueError("The Python source calls a function outside the safe PINN math function list.")
        expected = 2 if function in _BINARY_FUNCTIONS else 1
        if len(node.args) != expected:
            raise ValueError(f"Math function `{function}` needs {expected} argument(s).")
        args = [self.visit(argument) for argument in node.args]
        return ast.copy_location(ast.Call(func=ast.Name(id=function, ctx=ast.Load()), args=args, keywords=[]), node)

    def visit_Constant(self, node: ast.Constant):
        if type(node.value) not in (int, float) or not math.isfinite(float(node.value)):
            raise ValueError("Python PINN equations may contain only finite numeric constants.")
        return node

    def visit_BinOp(self, node: ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ValueError("Use only +, -, *, /, and bounded numeric powers in Python PINN equations.")
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Use only unary + or - in Python PINN equations.")
        return self.generic_visit(node)

    def generic_visit(self, node):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, ast.Add, ast.Sub,
                             ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            return super().generic_visit(node)
        if isinstance(node, ast.expr):
            raise ValueError("The Python source contains syntax outside the safe PINN math language.")
        return super().generic_visit(node)


def _python_equations(source_text: str, states: list[str], actions: list[str]) -> list[EquationTerm] | None:
    """Translate the supported assignment-based Python PINN format without executing it.

    The uploaded source remains untouched. This reads assignments into a small
    allowlisted AST, resolves scalar aliases, and returns expression strings
    that still pass through the regular expression validator and dataset check.
    """
    if not _python_tensor_source(source_text):
        return None

    lines = source_text.splitlines()
    first_equation = next((index for index, line in enumerate(lines)
                           if re.match(r"^\s*[A-Za-z_]\w*\s*=\s*(?:states|actions)\s*\[\s*:", line)), None)
    if first_equation is None:
        raise ValueError("I recognized Python tensor equations, but could not find a state/input assignment. Use `name = states[:, i]` or `name = actions[:, i]`.")
    code_lines = []
    for line in lines[first_equation:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        # These snippets are a flat equation list; discard pasted top-level
        # indentation while preserving whitespace inside expressions.
        code_lines.append(stripped)
    code = "\n".join(code_lines)
    try:
        module = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("I recognized Python-style PINN code, but could not read its assignments. Keep each equation on a valid Python assignment line.") from exc
    if len(list(ast.walk(module))) > MAX_AST_NODES * 12:
        raise ValueError("The Python PINN source is too complex for the safe equation translator.")

    values: dict[str, ast.expr] = {}
    derivatives: dict[int, str] = {}
    rewriter = _PythonEquationRewriter(values, len(states), len(actions))
    for statement in module.body:
        if isinstance(statement, ast.Import):
            if len(statement.names) == 1 and statement.names[0].name == "torch" and statement.names[0].asname is None:
                continue
            raise ValueError("Only an optional plain `import torch` is accepted in Python PINN source.")
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Python PINN source may contain only simple variable assignments and `physics_xdot[:, i] = ...` equations.")
        target = statement.targets[0]
        output_index = _tensor_column_index(target, "physics_xdot")
        if output_index is not None:
            if output_index < 0 or output_index >= len(states):
                raise ValueError(f"The Python source writes derivative index {output_index}, but the dataset has {len(states)} state(s).")
            if output_index in derivatives:
                raise ValueError(f"The Python source assigns derivative index {output_index} more than once.")
            expression = rewriter.visit(copy.deepcopy(statement.value))
            if len(list(ast.walk(expression))) > MAX_AST_NODES:
                raise ValueError("A translated Python equation is too complex for the safe PINN expression language.")
            derivatives[output_index] = ast.unparse(ast.fix_missing_locations(expression))
            continue
        if not isinstance(target, ast.Name) or target.id in {"states", "actions", "physics_xdot", "torch"}:
            raise ValueError("Only scalar aliases and `physics_xdot[:, i]` output assignments are supported in Python PINN source.")
        if target.id in values:
            raise ValueError(f"The Python source assigns `{target.id}` more than once; keep each parameter or alias unambiguous.")
        value = rewriter.visit(copy.deepcopy(statement.value))
        if len(list(ast.walk(value))) > MAX_AST_NODES:
            raise ValueError("A translated Python assignment is too complex for the safe PINN expression language.")
        values[target.id] = ast.fix_missing_locations(value)

    missing = [index for index in range(len(states)) if index not in derivatives]
    if missing:
        aliases = ", ".join(str(index) for index in missing)
        raise ValueError(f"The Python source is missing `physics_xdot[:, i]` equations for state index/indices {aliases}.")
    return [EquationTerm(state=states[index], expression=derivatives[index]) for index in range(len(states))]


def _render(node) -> str:
    """Render an already validated AST as deterministic PyTorch source."""
    if isinstance(node, ast.Expression):
        return _render(node.body)
    if isinstance(node, ast.Constant):
        return repr(float(node.value))
    if isinstance(node, ast.Name):
        return "torch.pi" if node.id == "pi" else node.id
    if isinstance(node, ast.UnaryOp):
        return ("+" if isinstance(node.op, ast.UAdd) else "-") + "(" + _render(node.operand) + ")"
    if isinstance(node, ast.BinOp):
        return f"({_render(node.left)} {_BINOPS[type(node.op)]} {_render(node.right)})"
    if isinstance(node, ast.Call):
        fn = _TORCH_FUNCTIONS[node.func.id]
        return f"{fn}({', '.join(_render(arg) for arg in node.args)})"
    raise ValueError("Unsupported expression node.")


def _generated_module(expressions: list[ast.Expression], state_count: int, action_count: int) -> str:
    lines = [
        '"""Generated locally by LabCD from validated, restricted PINN equations."""',
        "import torch", "", "def compute_analytical_xdot(states, actions):",
        f"    if states.ndim != 2 or states.shape[1] != {state_count}:",
        f"        raise ValueError('Expected a [batch, {state_count}] state tensor')",
        f"    if actions.ndim != 2 or actions.shape[1] != {action_count}:",
        f"        raise ValueError('Expected a [batch, {action_count}] input tensor')",
    ]
    lines.extend(f"    s{i} = states[:, {i}]" for i in range(state_count))
    lines.extend(f"    a{i} = actions[:, {i}]" for i in range(action_count))
    lines.extend(f"    dx{i} = torch.zeros_like(s{i}) + ({_render(expression)})"
                 for i, expression in enumerate(expressions))
    lines.append(f"    return torch.stack([{', '.join(f'dx{i}' for i in range(state_count))}], dim=1)")
    return "\n".join(lines) + "\n"


def _dataset_frame(dataset: dict) -> pd.DataFrame:
    path = Path(dataset.get("path", ""))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset.get("sha256"):
        raise ValueError("The attached dataset changed. Re-attach it before preparing a PINN equation.")
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    columns = [str(column) for column in frame.columns]
    states, actions = list(dataset.get("states") or []), list(dataset.get("actions") or [])
    if not dataset.get("ready") or not states or any(name not in columns for name in states + actions):
        raise ValueError("First attach a dataset and confirm which columns are measured states and inputs. Then I can map the equation safely.")
    values = frame[states + actions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("State and input columns must contain only finite numeric values before preparing a PINN equation.")
    return frame


def _column_summary(frame: pd.DataFrame, columns: list[str], aliases: list[str]) -> list[dict]:
    summary = []
    for column, alias in zip(columns, aliases):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        summary.append({"alias": alias, "column": column, "min": float(np.min(values)),
                        "max": float(np.max(values)), "mean": float(np.mean(values)),
                        "std": float(np.std(values))})
    return summary


def _provider_response(raw: str) -> EquationResponse:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("The configured model returned an unreadable equation proposal. Please retry or clarify the source.") from exc
    response = EquationResponse.model_validate(payload)
    if response.status == "clarification" and response.equations:
        raise ValueError("The equation proposal mixed a question with unverified equations. Please retry so I can resolve it safely.")
    if not response.message.strip():
        response.message = (
            "Could you clarify how the supplied equation variables map to the confirmed state and input columns?"
            if response.status == "clarification" else
            "I mapped the supplied equations to the confirmed columns and checked them against the measured data."
        )
    return response


def _prepare_prompt(chat: dict, question: str, source_text: str, source_name: str,
                    dataset: dict, frame: pd.DataFrame) -> str:
    states, actions = list(dataset["states"]), list(dataset.get("actions") or [])
    transcript = [{"role": item.get("role"), "content": str(item.get("content", ""))[:5000]}
                  for item in chat.get("messages", [])[-12:]
                  if item.get("role") in ("user", "assistant")]
    payload = {
        "task": "Interpret the supplied equation source and map it to the confirmed state/input columns. Do not derive new physics from measurements.",
        "source_name": source_name,
        "source_text_untrusted": source_text,
        "latest_client_request_untrusted": question,
        "recent_conversation_untrusted": transcript,
        "dataset": {
            "name": dataset.get("name"), "sha256": dataset.get("sha256"), "rows": int(dataset["rows"]),
            "states": _column_summary(frame, states, [f"s{i}" for i in range(len(states))]),
            "inputs": _column_summary(frame, actions, [f"a{i}" for i in range(len(actions))]),
            "time_column": "time" if "time" in frame.columns else None,
        },
        "output_schema": {"status": "ready or clarification",
                          "message": "short client-facing explanation; may be empty for ready, but ask one focused question for clarification",
                          "equations": [{"state": "exact state column", "expression": "safe expression"}],
                          "assumptions": ["only explicitly supplied assumptions"]},
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _validate_on_dataset(expressions: list[ast.Expression], frame: pd.DataFrame,
                         states: list[str], actions: list[str]) -> None:
    fn_source = _generated_module(expressions, len(states), len(actions))
    namespace = {"__name__": "_labcd_validated_pinn", "torch": torch}
    # This is app-authored source rendered from the restricted AST, never model/user Python.
    exec(compile(fn_source, "<validated-pinn-equation>", "exec"), namespace)
    function = namespace["compute_analytical_xdot"]
    matrix = frame[states + actions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    for offset in range(0, len(matrix), 4096):
        batch = torch.from_numpy(matrix[offset:offset + 4096])
        state_tensor = batch[:, :len(states)]
        action_tensor = batch[:, len(states):]
        try:
            with torch.no_grad():
                output = function(state_tensor, action_tensor)
        except Exception as exc:
            raise ValueError("The equation failed on measured state/input values. Check units, denominators, and equation mappings.") from exc
        if not torch.is_tensor(output) or tuple(output.shape) != (len(batch), len(states)):
            raise ValueError("The equation did not return one derivative per measured state.")
        if not torch.isfinite(output).all():
            raise ValueError("The equation produces a non-finite derivative for at least one measured row. Check its domain and units.")


def prepare_equation(chat: dict, question: str, *, source: dict | None = None,
                     client=None, output_dir: str | Path | None = None) -> dict:
    """Interpret source, validate exact coverage and write the trainer module."""
    dataset = chat.get("dataset") or {}
    if not dataset.get("ready"):
        raise ValueError("First attach a dataset and confirm its time, state, and input columns. Then I can map this equation for PINN.")
    frame = _dataset_frame(dataset)
    source = source or chat.get("pinn_source")
    if not isinstance(source, dict):
        raise ValueError("Paste or attach the physics equation source first, then ask me to prepare it for PINN.")
    if not Path(source.get("path", "")).resolve().is_relative_to(
            _chat_folder(chat["id"], "pinn_sources").resolve()):
        raise ValueError("The saved equation source is outside this conversation's source folder.")
    source_text = _read_source(source)
    if len(dataset["states"]) > 100 or len(dataset.get("actions") or []) > 100:
        raise ValueError("PINN equation mapping supports up to 100 state and 100 input columns.")
    states, actions = list(dataset["states"]), list(dataset.get("actions") or [])
    local_equations = _python_equations(source_text, states, actions)
    if local_equations is not None:
        response = EquationResponse(
            status="ready",
            message=("I read the Python-style PINN assignments and mapped `states[:, i]` and "
                     "`actions[:, i]` using the confirmed dataset column order. I translated the "
                     "equations and approved math functions without running the uploaded Python."),
            equations=local_equations,
            assumptions=[
                "State and input indices follow the confirmed dataset column order.",
                "Parameter expressions are fixed exactly as written in the source; uncertainty comments are not sampled as ranges.",
            ],
        )
        model_name = "local safe Python equation translator"
    else:
        if client is None:
            client = DiagnosticClient(DiagnosticSettings.defaults())
        prompt = _prepare_prompt(chat, question, source_text, source.get("name", "equation source"), dataset, frame)
        response = _provider_response(client.complete(system_prompt("pinn_maker"), prompt))
        model_name = getattr(getattr(client, "settings", None), "model", "configured model")
    if response.status == "clarification":
        source["status"] = "needs_clarification"
        result = {"answer": response.message, "status": "clarification", "generated_artifact": None,
                  "equations": [], "assumptions": response.assumptions, "evidence": [],
                  "model": model_name}
        return result

    target_order = [equation.state for equation in response.equations]
    if len(target_order) != len(states) or len(set(target_order)) != len(target_order) or set(target_order) != set(states):
        missing = [name for name in states if name not in target_order]
        extra = [name for name in target_order if name not in states]
        detail = (f" Missing: {', '.join(missing)}." if missing else "") + (f" Unknown or repeated: {', '.join(extra)}." if extra else "")
        raise ValueError("PINN needs exactly one equation for every confirmed state." + detail + " Clarify the source and try again.")
    by_state = {equation.state: equation.expression for equation in response.equations}
    parsed = [_parse_expression(by_state[name], len(states), len(actions)) for name in states]
    _validate_on_dataset(parsed, frame, states, actions)
    code = _generated_module(parsed, len(states), len(actions))
    source_digest = source["sha256"]
    generated_identity = hashlib.sha256((source_digest + dataset["sha256"] + json.dumps(by_state, sort_keys=True)).encode()).hexdigest()
    folder = Path(output_dir) if output_dir is not None else _chat_folder(chat["id"], "pinn_equations")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"pinn_equation_{generated_identity[:16]}.py"
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(code.encode()).hexdigest():
        handle, temporary = tempfile.mkstemp(prefix=".pinn-equation-", suffix=".py", dir=folder)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(code)
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    code_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    equations = [{"state": name, "alias": f"s{index}", "expression": by_state[name]}
                 for index, name in enumerate(states)]
    artifact = {"kind": "pinn_equation", "name": destination.name,
                "path": str(destination.resolve()), "sha256": code_digest,
                "source_name": source.get("name"), "source_path": source.get("path"),
                "source_sha256": source_digest, "dataset_sha256": dataset["sha256"],
                "states": states, "actions": actions, "equations": equations,
                "state_aliases": [{"alias": f"s{i}", "column": column} for i, column in enumerate(states)],
                "action_aliases": [{"alias": f"a{i}", "column": column} for i, column in enumerate(actions)],
                "assumptions": response.assumptions}
    details = "\n".join(f"- `{equation['state']}`: `{equation['expression']}`" for equation in equations)
    aliases = ", ".join([f"`s{i}` = `{column}`" for i, column in enumerate(states)] +
                         [f"`a{i}` = `{column}`" for i, column in enumerate(actions)])
    validation_note = ("I mapped one derivative for each measured state and checked that the equations return finite values for every row in the attached dataset. "
                       "Review the variable mapping and units below before starting a PINN run.")
    assumptions_note = ("Assumptions: " + " ".join(response.assumptions)) if response.assumptions else ""
    message = "\n\n".join(part for part in (response.message.strip(), validation_note,
                                            "Variable map: " + aliases, details, assumptions_note) if part)
    return {"answer": message, "status": "ready", "generated_artifact": artifact,
            "equations": equations, "assumptions": response.assumptions, "evidence": [],
            "model": model_name}


def is_equation_ready(chat: dict) -> bool:
    """Confirm generated file, hashes and exact role mapping still match this chat."""
    artifact, dataset = chat.get("pinn_equation"), chat.get("dataset") or {}
    if not isinstance(artifact, dict) or not dataset.get("ready"):
        return False
    try:
        path = Path(artifact["path"])
        if not path.resolve().is_relative_to(_chat_folder(chat["id"], "pinn_equations").resolve()):
            return False
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
            return False
        return (artifact.get("dataset_sha256") == dataset.get("sha256") and
                artifact.get("states") == list(dataset.get("states") or []) and
                artifact.get("actions") == list(dataset.get("actions") or []))
    except (KeyError, OSError, TypeError):
        return False


class PINNMakerJob:
    """Background interpreter job for a physics equation source."""
    def __init__(self, chat: dict, question: str):
        self.snapshot = copy.deepcopy(chat)
        self.question = question
        self.purpose = "pinn_maker"
        self.answer = None
        self.error = None
        self.error_detail = None
        self.phase = "PINN equation · mapping to confirmed states and inputs"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            self.answer = prepare_equation(self.snapshot, self.question)
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:800]
            self.error = str(exc) if isinstance(exc, ValueError) else "I couldn't prepare the PINN equation. Check the source and configured AI connection, then try again."
