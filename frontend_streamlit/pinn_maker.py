"""Safe, data-bound conversion of raw physics equations and MATLAB .m files.

Supported Python tensor assignments are translated locally from a restricted
AST. Other Python and MATLAB source can be interpreted by the configured model,
after which this module validates the map and compiles it itself. Uploaded
source and model-generated text are never executed as Python or MATLAB.
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
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
_REQUEST_VERBS = re.compile(r"\b(?:use|prepare|make|create|convert|map|apply|enable|add|run)\b", re.I)
_PINN_TERMS = re.compile(r"\b(?:pinn|pcnn|physics(?:[- ](?:informed|constrained))?|equations?|derivatives?|xdot|d\s*\w+\s*/\s*dt)\b", re.I)
_CONTINUATION = re.compile(r"\b(?:prepare|use|map|convert|equation|pinn|pcnn|ready|yes|confirm|apply|retry)\b", re.I)
_ORDINARY_QUESTION = re.compile(
    r"^\s*(?:what|why|which|how|where|when|who|can|could|would|should|do|does|did|is|are|was|were|"
    r"have|has|will|show|plot|analy[sz]e|explain|compare)\b", re.I)
_EXPLICIT_PREPARE_VERB = re.compile(r"\b(?:prepare|make|create|convert|map|apply|enable|validate)\b", re.I)
_CLARIFICATION_REPLY = re.compile(
    r"\b(?:s\d+|a\d+|state|input|action|column|equation|derivative|represents?|means|mapped|mapping)\b|"
    r"^\s*(?:yes|no|correct|that's right|that is right|confirmed)\s*[.!]?\s*$", re.I)
_PINN_USAGE_QUESTION = re.compile(
    r"\b(?:do|does|did)\s+(?:you|we|this(?:\s+run|\s+model)?|the\s+run|the\s+model)\s+"
    r"(?:use|apply|include|train(?:ed)?\s+with)\b.{0,100}\b(?:pinn|pcnn|physics[- ](?:informed|constrained))\b|"
    r"\b(?:is|are|was|were|has|have)\b.{0,80}\b(?:pinn|pcnn|physics[- ](?:informed|constrained))\b.{0,60}"
    r"\b(?:used|enabled|included|applied|active)\b.{0,40}\b(?:run|training|model)\b|"
    r"^\s*(?:is|are|was|were|has|have)\b.{0,60}\b(?:pinn|pcnn|physics[- ](?:informed|constrained))\b.{0,40}"
    r"\b(?:on|off|used|enabled|included|applied|active)\b|"
    r"\b(?:pinn|pcnn|physics[- ](?:informed|constrained))\b.{0,50}\b(?:in|used in|enabled in)\b.{0,30}"
    r"\b(?:this|the|current)?\s*(?:run|training|model)\b",
    re.I,
)


class EquationTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: str = Field(min_length=1, max_length=160)
    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)


class PhysicsParameter(BaseModel):
    """One bounded coefficient estimated together with the network weights."""
    model_config = ConfigDict(extra="forbid", strict=True)
    alias: str = Field(pattern=r"^p\d+$", max_length=12)
    name: str = Field(min_length=1, max_length=120)
    initial: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    units: str = Field(default="", max_length=80)
    rationale: str = Field(default="", max_length=500)


class PCNNConstraintRecommendation(BaseModel):
    """A review-only suggestion for a PCNN hard-constrained state derivative."""
    model_config = ConfigDict(extra="forbid", strict=True)
    state: str = Field(min_length=1, max_length=160)
    recommended_exact: bool
    confidence: Literal["low", "medium", "high"] = "low"
    rationale: str = Field(min_length=1, max_length=500)


class EquationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ready", "clarification"]
    # Some structured-output providers return an empty explanation even when
    # the actual equation mapping is complete. The application supplies a
    # safe, deterministic client-facing fallback in that case.
    message: str = Field(default="", max_length=5000)
    equations: list[EquationTerm] = Field(default_factory=list, max_length=100)
    assumptions: list[str] = Field(default_factory=list, max_length=30)
    parameters: list[PhysicsParameter] = Field(default_factory=list, max_length=40)
    pcnn_constraints: list[PCNNConstraintRecommendation] = Field(default_factory=list, max_length=100)


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
    # A saved/ready source is context, not an instruction. Only explicit
    # equation work can prepare it; normal questions must remain ordinary chat.
    explicit_prepare = bool(_EXPLICIT_PREPARE_VERB.search(text) and _PINN_TERMS.search(text))
    if _ORDINARY_QUESTION.search(text):
        direct_request = bool(re.match(r"^\s*(?:can|could|would)\s+you\b", text, re.I) and explicit_prepare)
        if not direct_request:
            return False
    if has_pending_source == "needs_clarification" and "?" not in text:
        if _CLARIFICATION_REPLY.search(text) or _CONTINUATION.search(text):
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
        raise ValueError("Use a non-empty physics-rule source file smaller than 120 KB.")
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
        raise ValueError("The saved physics source changed. Attach or paste it again before preparing this model.")
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("The saved equation source is larger than the supported 120 KB limit.")
    return raw.decode("utf-8-sig")


def _parse_expression(expression: str, state_count: int, action_count: int,
                      parameter_count: int = 0) -> ast.Expression:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("A derivative rule is longer than the supported expression limit.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("A derivative rule could not be read as a mathematical expression. Clarify the rule before using it.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("A derivative rule is too complex for the safe physics expression language.")
    state_names = {f"s{i}" for i in range(state_count)}
    action_names = {f"a{i}" for i in range(action_count)}
    parameter_names = {f"p{i}" for i in range(parameter_count)}
    allowed_names = state_names | action_names | parameter_names | {"pi"} | set(_UNARY_FUNCTIONS) | set(_BINARY_FUNCTIONS)
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


def _normalize_python_paste(source_text: str) -> str:
    """Undo Markdown escaping around pasted Python punctuation for interpretation only."""
    text = (source_text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\\)\\([^\w\s])", r"\1", text)
    return text.replace("`", "")


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
            raise ValueError("The Python source calls a function outside the safe physics math function list.")
        expected = 2 if function in _BINARY_FUNCTIONS else 1
        if len(node.args) != expected:
            raise ValueError(f"Math function `{function}` needs {expected} argument(s).")
        args = [self.visit(argument) for argument in node.args]
        return ast.copy_location(ast.Call(func=ast.Name(id=function, ctx=ast.Load()), args=args, keywords=[]), node)

    def visit_Constant(self, node: ast.Constant):
        if type(node.value) not in (int, float) or not math.isfinite(float(node.value)):
            raise ValueError("Python physics rules may contain only finite numeric constants.")
        return node

    def visit_BinOp(self, node: ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ValueError("Use only +, -, *, /, and bounded numeric powers in Python physics rules.")
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Use only unary + or - in Python physics rules.")
        return self.generic_visit(node)

    def generic_visit(self, node):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, ast.Add, ast.Sub,
                             ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            return super().generic_visit(node)
        if isinstance(node, ast.expr):
            raise ValueError("The Python source contains syntax outside the safe physics math language.")
        return super().generic_visit(node)


def _python_equations(source_text: str, states: list[str], actions: list[str], *,
                      require_all: bool = True) -> list[EquationTerm] | None:
    """Translate supported derivative assignments without executing Python.

    The uploaded source remains untouched. This reads assignments into a small
    allowlisted AST, resolves scalar aliases, and returns expression strings
    that still pass through the regular expression validator and dataset check.
    """
    source_text = _normalize_python_paste(source_text)
    if not _python_tensor_source(source_text):
        return None

    lines = source_text.splitlines()
    first_equation = next((index for index, line in enumerate(lines)
                           if re.match(r"^\s*[A-Za-z_]\w*\s*=\s*(?:states|actions)\s*\[\s*:", line)), None)
    if first_equation is None:
        # The source may be pasted into a single wrapped line. Let the configured
        # equation interpreter read that raw source instead of rejecting it here.
        return None
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
    except (SyntaxError, ValueError):
        # This fast path intentionally supports only a small, unambiguous subset.
        # A syntax/formatting mismatch is not a reason to reject raw PINN code:
        # prepare_equation will send it to the configured interpreter, whose
        # response still passes the same allowlisted expression and data checks.
        return None
    if len(list(ast.walk(module))) > MAX_AST_NODES * 12:
        raise ValueError("The Python physics source is too complex for the safe rule translator.")

    values: dict[str, ast.expr] = {}
    derivatives: dict[int, str] = {}
    rewriter = _PythonEquationRewriter(values, len(states), len(actions))
    for statement in module.body:
        if isinstance(statement, ast.Import):
            if len(statement.names) == 1 and statement.names[0].name == "torch" and statement.names[0].asname is None:
                continue
            raise ValueError("Only a plain optional `import torch` is accepted in Python physics source.")
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Python physics source may contain only simple variable assignments and `physics_xdot[:, i] = ...` derivative rules.")
        target = statement.targets[0]
        output_index = _tensor_column_index(target, "physics_xdot")
        if output_index is not None:
            if output_index < 0 or output_index >= len(states):
                raise ValueError(f"The Python source writes derivative index {output_index}, but the dataset has {len(states)} state(s).")
            if output_index in derivatives:
                raise ValueError(f"The Python source assigns derivative index {output_index} more than once.")
            expression = rewriter.visit(copy.deepcopy(statement.value))
            if len(list(ast.walk(expression))) > MAX_AST_NODES:
                raise ValueError("A translated Python rule is too complex for the safe physics expression language.")
            derivatives[output_index] = ast.unparse(ast.fix_missing_locations(expression))
            continue
        if not isinstance(target, ast.Name) or target.id in {"states", "actions", "physics_xdot", "torch"}:
            raise ValueError("Only scalar aliases and `physics_xdot[:, i]` output assignments are supported in Python physics source.")
        if target.id in values:
            raise ValueError(f"The Python source assigns `{target.id}` more than once; keep each parameter or alias unambiguous.")
        value = rewriter.visit(copy.deepcopy(statement.value))
        if len(list(ast.walk(value))) > MAX_AST_NODES:
            raise ValueError("A translated Python assignment is too complex for the safe physics expression language.")
        values[target.id] = ast.fix_missing_locations(value)

    missing = [index for index in range(len(states)) if index not in derivatives]
    if require_all and missing:
        aliases = ", ".join(str(index) for index in missing)
        raise ValueError(f"The Python source is missing `physics_xdot[:, i]` equations for state index/indices {aliases}.")
    if not derivatives:
        raise ValueError("Add at least one derivative assignment for a state you want PCNN to constrain.")
    return [EquationTerm(state=states[index], expression=derivatives[index])
            for index in range(len(states)) if index in derivatives]


def _render(node, *, parameterized: bool = False) -> str:
    """Render an already validated AST as deterministic PyTorch source."""
    if isinstance(node, ast.Expression):
        return _render(node.body)
    if isinstance(node, ast.Constant):
        return repr(float(node.value))
    if isinstance(node, ast.Name):
        return node.id if parameterized and re.fullmatch(r"p\d+", node.id) else ("torch.pi" if node.id == "pi" else node.id)
    if isinstance(node, ast.UnaryOp):
        return ("+" if isinstance(node.op, ast.UAdd) else "-") + "(" + _render(node.operand, parameterized=parameterized) + ")"
    if isinstance(node, ast.BinOp):
        return f"({_render(node.left, parameterized=parameterized)} {_BINOPS[type(node.op)]} {_render(node.right, parameterized=parameterized)})"
    if isinstance(node, ast.Call):
        fn = _TORCH_FUNCTIONS[node.func.id]
        return f"{fn}({', '.join(_render(arg, parameterized=parameterized) for arg in node.args)})"
    raise ValueError("Unsupported expression node.")


def _generated_module(expressions: list[ast.Expression | None], state_count: int, action_count: int,
                      parameters: list[PhysicsParameter] | None = None) -> str:
    parameters = list(parameters or [])
    if len(expressions) != state_count:
        raise ValueError("The internal constraint map must reserve one slot per confirmed state.")
    defined_indices = [index for index, expression in enumerate(expressions) if expression is not None]
    if not defined_indices:
        raise ValueError("PCNN needs at least one state derivative to enforce as a hard constraint.")
    if not parameters:
        lines = [
            '"""Generated locally by LabCD from validated, restricted physics constraints."""',
            "import torch", "", "def compute_analytical_xdot(states, actions):",
            f"    if states.ndim != 2 or states.shape[1] != {state_count}:",
            f"        raise ValueError('Expected a [batch, {state_count}] state tensor')",
            f"    if actions.ndim != 2 or actions.shape[1] != {action_count}:",
            f"        raise ValueError('Expected a [batch, {action_count}] input tensor')",
        ]
        lines.extend(f"    s{i} = states[:, {i}]" for i in range(state_count))
        lines.extend(f"    a{i} = actions[:, {i}]" for i in range(action_count))
        lines.append(f"    # Only these derivatives have explicit hard-constraint rules: {defined_indices!r}.")
        lines.extend(
            f"    dx{i} = torch.zeros_like(s{i})" if expression is None else
            f"    dx{i} = torch.zeros_like(s{i}) + ({_render(expression)})"
            for i, expression in enumerate(expressions)
        )
        lines.append(f"    return torch.stack([{', '.join(f'dx{i}' for i in range(state_count))}], dim=1)")
        lines.append(f"compute_analytical_xdot.defined_state_indices = {defined_indices!r}")
        return "\n".join(lines) + "\n"

    initials = {item.alias: item.initial for item in parameters}
    initial_logits = []
    for item in parameters:
        if not item.lower < item.upper or not item.lower <= item.initial <= item.upper:
            raise ValueError(f"Parameter `{item.name}` needs an initial value inside its finite bounds.")
        fraction = min(max((item.initial - item.lower) / (item.upper - item.lower), 1e-6), 1.0 - 1e-6)
        initial_logits.append(math.log(fraction / (1.0 - fraction)))

    lines = [
        '"""Generated locally by LabCD from validated, restricted physics constraints."""',
        "import torch", "import torch.nn as nn", "", f"PARAMETER_INITIALS = {initials!r}",
        f"PARAMETER_METADATA = {[item.model_dump() for item in parameters]!r}",
        "", "class _LabCDPhysicsEquation(nn.Module):", "    def __init__(self):",
        "        super().__init__()", f"        self.defined_state_indices = {defined_indices!r}",
        "        self.raw_parameters = nn.ParameterList([",
    ]
    lines.extend(f"            nn.Parameter(torch.tensor({value!r}, dtype=torch.float32))," for value in initial_logits)
    lines.extend([
        "        ])", "", "    def parameter_estimates(self):", "        with torch.no_grad():", "            return {",
    ])
    for index, item in enumerate(parameters):
        lines.append(f"                {item.alias!r}: float(({item.lower!r} + ({item.upper!r} - {item.lower!r}) * torch.sigmoid(self.raw_parameters[{index}])).cpu()),")
    lines.extend(["            }", "", "    def parameter_metadata(self):", "        return PARAMETER_METADATA", "",
                  "    def set_parameter_estimates(self, values):", "        with torch.no_grad():"])
    for index, item in enumerate(parameters):
        lines.append(f"            value = min(max(float(values.get({item.alias!r}, PARAMETER_INITIALS[{item.alias!r}])), {item.lower!r}), {item.upper!r})")
        lines.append(f"            fraction = min(max((value - {item.lower!r}) / ({item.upper!r} - {item.lower!r}), 1e-6), 1.0 - 1e-6)")
        lines.append(f"            self.raw_parameters[{index}].copy_(torch.logit(torch.tensor(fraction, device=self.raw_parameters[{index}].device)))")
    lines.extend([
        "", "    def forward(self, states, actions):",
        f"        if states.ndim != 2 or states.shape[1] != {state_count}:",
        f"            raise ValueError('Expected a [batch, {state_count}] state tensor')",
        f"        if actions.ndim != 2 or actions.shape[1] != {action_count}:",
        f"            raise ValueError('Expected a [batch, {action_count}] input tensor')",
    ])
    lines.extend(f"        s{i} = states[:, {i}]" for i in range(state_count))
    lines.extend(f"        a{i} = actions[:, {i}]" for i in range(action_count))
    for index, item in enumerate(parameters):
        lines.append(f"        p{index} = {item.lower!r} + ({item.upper!r} - {item.lower!r}) * torch.sigmoid(self.raw_parameters[{index}])")
    lines.append(f"        # Only these derivatives have explicit hard-constraint rules: {defined_indices!r}.")
    lines.extend(
        f"        dx{i} = torch.zeros_like(s{i})" if expression is None else
        f"        dx{i} = torch.zeros_like(s{i}) + ({_render(expression, parameterized=True)})"
        for i, expression in enumerate(expressions)
    )
    lines.extend([
        f"        return torch.stack([{', '.join(f'dx{i}' for i in range(state_count))}], dim=1)",
        "", "def build_physics_model():", "    return _LabCDPhysicsEquation()", "",
        "def compute_analytical_xdot(states, actions, parameter_values=None):",
        "    model = build_physics_model().to(states.device)", "    if parameter_values:", "        model.set_parameter_estimates(parameter_values)",
        f"    model.defined_state_indices = {defined_indices!r}",
        "    return model(states, actions)",
    ])
    return "\n".join(lines) + "\n"


def _dataset_frame(dataset: dict) -> pd.DataFrame:
    path = Path(dataset.get("path", ""))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset.get("sha256"):
        raise ValueError("The attached dataset changed. Re-attach it before preparing a physics rule.")
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    columns = [str(column) for column in frame.columns]
    states, actions = list(dataset.get("states") or []), list(dataset.get("actions") or [])
    if not dataset.get("ready") or not states or any(name not in columns for name in states + actions):
        raise ValueError("First attach a dataset and confirm which columns are measured states and inputs. Then I can map the equation safely.")
    values = frame[states + actions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("State and input columns must contain only finite numeric values before preparing physics.")
    return frame


def _column_summary(frame: pd.DataFrame, columns: list[str], aliases: list[str]) -> list[dict]:
    summary = []
    for column, alias in zip(columns, aliases):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        summary.append({"alias": alias, "column": column, "min": float(np.min(values)),
                        "max": float(np.max(values)), "mean": float(np.mean(values)),
                        "std": float(np.std(values))})
    return summary


def _map_column_names_to_aliases(expression: str, states: list[str], actions: list[str]) -> str:
    """Normalize exact measured-column names to the safe sN/aN expression aliases."""
    text = str(expression or "")
    reserved = set(_UNARY_FUNCTIONS) | set(_BINARY_FUNCTIONS) | {"pi", "torch"}
    replacements = [
        (str(name), f"{prefix}{index}")
        for prefix, names in (("s", states), ("a", actions))
        for index, name in enumerate(names)
        if str(name) and str(name) not in reserved
        and not re.fullmatch(r"[sap]\d+", str(name))
    ]
    # Match longest labels first and only whole identifiers so e.g. s_pitch
    # cannot partially rewrite s_dpitch. Substitution is exact and case-sensitive.
    for name, alias in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        text = re.sub(pattern, lambda _match, replacement=alias: replacement, text)
    return text


def _normalize_numbered_model_aliases(expression: str, description: str,
                                      states: list[str], actions: list[str], *,
                                      one_based_states: bool = False,
                                      one_based_actions: bool = False) -> str:
    """Convert client-style 1-based sN/aN references to the app's 0-based aliases.

    Only normalize when an expression contains an alias that cannot exist in the
    confirmed schema (for example s4 with four states). If s0 is present, retain
    the provider's 0-based convention and repair only explicitly declared client
    indices that are out of range.
    """
    text = str(expression or "")
    state_refs = set(re.findall(r"(?<!\w)s(\d+)(?!\w)", text))
    action_refs = set(re.findall(r"(?<!\w)a(\d+)(?!\w)", text))
    state_indexes = {int(value) for value in state_refs}
    action_indexes = {int(value) for value in action_refs}
    state_map: dict[int, int] = {}
    declared_state_map = {int(label[1:]): index for label, index in
                          _numbered_state_assignments(description, states).items()}
    if one_based_states:
        state_map = {index: index - 1 for index in state_indexes
                     if 1 <= index <= len(states)}
    elif state_indexes and max(state_indexes) >= len(states):
        if 0 not in state_indexes and len(declared_state_map) == len(states):
            state_map = {index: declared_state_map[index] for index in state_indexes
                         if index in declared_state_map}
        elif 0 not in state_indexes and max(state_indexes) == len(states):
            state_map = {index: index - 1 for index in state_indexes
                         if 1 <= index <= len(states)}
        else:
            state_map = {index: declared_state_map[index] for index in state_indexes
                         if index >= len(states) and index in declared_state_map}
    elif (state_indexes and 0 not in state_indexes
          and len(declared_state_map) == len(states)):
        state_map = {index: declared_state_map[index] for index in state_indexes
                     if index in declared_state_map}
    for source, target in state_map.items():
        text = re.sub(rf"(?<!\w)s{source}(?!\w)", f"s{target}", text)

    if one_based_actions:
        action_map = {index: index - 1 for index in action_indexes
                      if 1 <= index <= len(actions)}
        for source, target in action_map.items():
            text = re.sub(rf"(?<!\w)a{source}(?!\w)", f"a{target}", text)
    elif action_indexes and max(action_indexes) >= len(actions):
        if 0 not in action_indexes and max(action_indexes) == len(actions):
            action_map = {index: index - 1 for index in action_indexes
                          if 1 <= index <= len(actions)}
        else:
            action_map = {index: index - 1 for index in action_indexes
                          if index == len(actions)}
        for source, target in action_map.items():
            text = re.sub(rf"(?<!\w)a{source}(?!\w)", f"a{target}", text)
    return text


def _model_uses_client_one_based_aliases(response: EquationResponse,
                                         description: str, states: list[str],
                                         actions: list[str]) -> tuple[bool, bool]:
    """Choose an alias convention once for the whole equation set."""
    combined = " ".join(item.expression for item in response.equations)
    state_refs = {int(value) for value in re.findall(r"(?<!\w)s(\d+)(?!\w)", combined)}
    action_refs = {int(value) for value in re.findall(r"(?<!\w)a(\d+)(?!\w)", combined)}
    numbered = _numbered_state_assignments(description, states)
    state_one_based = bool(state_refs) and 0 not in state_refs and (
        (max(state_refs) >= len(states) and max(state_refs) == len(states)) or
        (len(numbered) == len(states) and any(f"s{index}" in numbered for index in state_refs))
    )
    action_one_based = bool(action_refs) and 0 not in action_refs and (
        max(action_refs) >= len(actions) and max(action_refs) == len(actions)
    )
    return state_one_based, action_one_based


def _numbered_state_assignments(description: str, states: list[str]) -> dict[str, int]:
    """Map declared client sN labels to confirmed zero-based state positions."""
    matches = list(re.finditer(
        r"(?<![\w])s(\d+)\s*=\s*(.*?)(?=\s*,?\s*s\d+\s*=|[,;\n]|$)",
        str(description or ""), flags=re.I | re.S,
    ))
    labels = [int(match.group(1)) for match in matches]
    if not matches or len(labels) != len(set(labels)):
        return {}
    base = 0 if 0 in labels else 1
    mapping = {f"s{label}": label - base for label in labels
               if 0 <= label - base < len(states)}
    if len(mapping) != len(matches):
        return {}
    return mapping


def _physics_mode_for_chat(chat: dict) -> str:
    """Use the pending wizard choice while setup is open, saved settings afterward."""
    setup = chat.get("setup") or {}
    pending = setup.get("pending_physics_mode")
    if setup.get("stage") not in (None, "complete") and pending in {"pinn", "pcnn"}:
        return pending
    return str((chat.get("settings") or {}).get("physics_mode") or "pinn").strip().lower()


def _provider_response(raw: str) -> EquationResponse:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("The configured model returned an unreadable equation proposal. Please retry or clarify the source.") from exc
    try:
        response = EquationResponse.model_validate(payload)
    except ValidationError:
        # PCNN exactness suggestions are advisory. A malformed suggestion must
        # not discard otherwise valid equations; retry conservatively with no
        # exact-state recommendations so the UI can keep NN corrections active.
        if not isinstance(payload, dict) or "pcnn_constraints" not in payload:
            raise
        safe_payload = dict(payload)
        safe_payload["pcnn_constraints"] = []
        response = EquationResponse.model_validate(safe_payload)
    if response.status == "clarification" and response.equations:
        raise ValueError("The equation proposal mixed a question with unverified equations. Please retry so I can resolve it safely.")
    if response.status == "clarification" and response.parameters:
        raise ValueError("The equation proposal mixed a question with unreviewed coefficients. Please retry so I can resolve it safely.")
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
    mode = _physics_mode_for_chat(chat)
    transcript = [{"role": item.get("role"), "content": str(item.get("content", ""))[:5000]}
                  for item in chat.get("messages", [])[-12:]
                  if item.get("role") in ("user", "assistant")]
    payload = {
        "physics_mode": mode,
        "task": (
            "Interpret only the derivative rules the client intends to enforce exactly as PCNN hard constraints. Map each rule to its confirmed state and input columns; return only those constrained state derivatives, not a full model for every state. Do not infer extra constraints from measurements."
            if mode == "pcnn" else
            "Interpret the supplied equation source and map a complete derivative equation to every confirmed state for PINN. Do not derive new physics from measurements."
        ),
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
                          "equations": [{"state": "exact constrained state column", "expression": "safe derivative expression"}],
                          "assumptions": ["only explicitly supplied assumptions"]},
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _validate_on_dataset(expressions: list[ast.Expression], frame: pd.DataFrame,
                         states: list[str], actions: list[str],
                         parameters: list[PhysicsParameter] | None = None) -> None:
    fn_source = _generated_module(expressions, len(states), len(actions), parameters)
    namespace = {"__name__": "_labcd_validated_pinn", "torch": torch, "nn": torch.nn}
    # This is app-authored source rendered from the restricted AST, never model/user Python.
    exec(compile(fn_source, "<validated-pinn-equation>", "exec"), namespace)
    function = (namespace["build_physics_model"]() if parameters else
                namespace["compute_analytical_xdot"])
    matrix = frame[states + actions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    candidates = [(function, True)]
    if parameters:
        initial_values = {item.alias: item.initial for item in parameters}
        # Probe each coefficient at its stated bounds on a bounded sample. This
        # catches many singular denominators before the ranges reach training.
        for item in parameters:
            for bound in (item.lower, item.upper):
                trial = namespace["build_physics_model"]()
                values = dict(initial_values)
                values[item.alias] = bound
                trial.set_parameter_estimates(values)
                candidates.append((trial, False))
    for candidate, check_all_rows in candidates:
        sample = matrix if check_all_rows else matrix[:min(len(matrix), 512)]
        for offset in range(0, len(sample), 4096):
            batch = torch.from_numpy(sample[offset:offset + 4096])
            state_tensor = batch[:, :len(states)]
            action_tensor = batch[:, len(states):]
            try:
                with torch.no_grad():
                    output = candidate(state_tensor, action_tensor)
            except Exception as exc:
                raise ValueError("The equation failed on measured state/input values. Check units, denominators, and equation mappings.") from exc
            if not torch.is_tensor(output) or tuple(output.shape) != (len(batch), len(states)):
                raise ValueError("The equation did not return one derivative per measured state.")
            if not torch.isfinite(output).all():
                raise ValueError("The equation produces a non-finite derivative for measured values or a coefficient bound. Check its domain and units.")


def prepare_equation(chat: dict, question: str, *, source: dict | None = None,
                     client=None, output_dir: str | Path | None = None) -> dict:
    """Interpret a complete PINN model or sparse PCNN hard-constraint source."""
    dataset = chat.get("dataset") or {}
    if not dataset.get("ready"):
        raise ValueError("First attach a dataset and confirm its time, state, and input columns. Then I can map the physics source.")
    frame = _dataset_frame(dataset)
    source = source or chat.get("pinn_source")
    if not isinstance(source, dict):
        raise ValueError("Paste or attach the physics source first, then ask me to map it to this dataset.")
    if not Path(source.get("path", "")).resolve().is_relative_to(
            _chat_folder(chat["id"], "pinn_sources").resolve()):
        raise ValueError("The saved equation source is outside this conversation's source folder.")
    source_text = _read_source(source)
    python_source_text = _normalize_python_paste(source_text)
    physics_mode = _physics_mode_for_chat(chat)
    pcnn_mode = physics_mode == "pcnn"
    if len(dataset["states"]) > 100 or len(dataset.get("actions") or []) > 100:
        raise ValueError("Physics source mapping supports up to 100 state and 100 input columns.")
    states, actions = list(dataset["states"]), list(dataset.get("actions") or [])
    local_equations = _python_equations(python_source_text, states, actions, require_all=not pcnn_mode)
    if local_equations is not None:
        response = EquationResponse(
            status="ready",
            message=("I read the supplied state-derivative assignments and mapped `states[:, i]` and "
                     "`actions[:, i]` using the confirmed dataset column order. I translated the "
                     "rules and approved math functions without running the uploaded Python."),
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
        # Use normalized punctuation for pasted Python, while preserving MATLAB
        # source as-authored. The saved original file and its hash are untouched.
        prompt_source = python_source_text if _PYTHON_TENSOR_REFERENCE.search(python_source_text) else source_text
        prompt = _prepare_prompt(chat, question, prompt_source, source.get("name", "equation source"), dataset, frame)
        response = _provider_response(client.complete(system_prompt("pinn_maker"), prompt))
        model_name = getattr(getattr(client, "settings", None), "model", "configured model")
    if response.parameters:
        raise ValueError("Physics source mapping does not support undeclared learnable coefficients. Give each coefficient a fixed value, or describe the system so LabCD can propose bounded coefficients for review.")
    if response.status == "clarification":
        source["status"] = "needs_clarification"
        result = {"answer": response.message, "status": "clarification", "generated_artifact": None,
                  "equations": [], "assumptions": response.assumptions, "evidence": [],
                  "model": model_name}
        return result

    target_order = [equation.state for equation in response.equations]
    complete_pinn_map = (len(target_order) == len(states) and set(target_order) == set(states))
    valid_pcnn_constraints = (bool(target_order) and set(target_order).issubset(set(states)))
    if (not pcnn_mode and not complete_pinn_map) or (pcnn_mode and not valid_pcnn_constraints) or len(set(target_order)) != len(target_order):
        missing = [name for name in states if name not in target_order]
        extra = [name for name in target_order if name not in states]
        detail = (f" Missing: {', '.join(missing)}." if missing else "") + (f" Unknown or repeated: {', '.join(extra)}." if extra else "")
        if pcnn_mode:
            raise ValueError("PCNN needs at least one valid derivative rule for a state you want hard-constrained. Leave other states out so the neural network can learn them." + detail)
        raise ValueError("PINN needs exactly one equation for every confirmed state." + detail + " Clarify the source and try again.")
    by_state = {equation.state: equation.expression for equation in response.equations}
    parsed = [_parse_expression(by_state[name], len(states), len(actions)) if name in by_state else None
              for name in states]
    _validate_on_dataset(parsed, frame, states, actions)
    code = _generated_module(parsed, len(states), len(actions))
    source_digest = source["sha256"]
    generated_identity = hashlib.sha256((source_digest + dataset["sha256"] + json.dumps(by_state, sort_keys=True)).encode()).hexdigest()
    folder = Path(output_dir) if output_dir is not None else _chat_folder(chat["id"], "pinn_equations")
    folder.mkdir(parents=True, exist_ok=True)
    artifact_prefix = "pcnn_constraints" if pcnn_mode else "pinn_equation"
    destination = folder / f"{artifact_prefix}_{generated_identity[:16]}.py"
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
                 for index, name in enumerate(states) if name in by_state]
    artifact = {"kind": "pinn_equation", "physics_mode": physics_mode, "name": destination.name,
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
    validation_note = (
        "I checked the supplied hard-constraint rules against every measured row. Only the listed state derivatives can be constrained; every other state remains a neural-network output. Review the mapping and units before enabling PCNN."
        if pcnn_mode else
        "I mapped one derivative for each measured state and checked that the equations return finite values for every row in the attached dataset. Review the variable mapping and units before starting a PINN run."
    )
    assumptions_note = ("Assumptions: " + " ".join(response.assumptions)) if response.assumptions else ""
    message = "\n\n".join(part for part in (response.message.strip(), validation_note,
                                            "Variable map: " + aliases, details, assumptions_note) if part)
    return {"answer": message, "status": "ready", "generated_artifact": artifact,
            "equations": equations, "assumptions": response.assumptions, "evidence": [],
            "model": model_name}


def _physics_data_checks(frame: pd.DataFrame, states: list[str], actions: list[str]) -> list[str]:
    """Report measured-data limits without using the data to invent a law."""
    checks = []
    if len(frame) < 30:
        checks.append(f"Only {len(frame)} samples are available; coefficient estimates are likely unstable.")
    elif len(frame) < 100:
        checks.append(f"{len(frame)} samples provide a limited basis for identifying unknown coefficients.")
    quiet = []
    for column in actions:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scale = max(1.0, float(np.nanmax(np.abs(values))))
        if float(np.nanstd(values)) <= 1e-6 * scale:
            quiet.append(column)
    if quiet:
        checks.append("Input variation is very small for " + ", ".join(quiet[:4]) + "; their effects may not be identifiable.")
    if len(states) > 1:
        correlations = frame[states].corr(numeric_only=True).abs()
        similar = []
        for i, left in enumerate(states):
            for right in states[i + 1:]:
                value = correlations.loc[left, right]
                if pd.notna(value) and float(value) >= 0.995:
                    similar.append(f"{left} and {right}")
        if similar:
            checks.append("Some states move almost together (" + ", ".join(similar[:3]) + "); separate dynamics may be hard to identify.")
    return checks


def _explicit_kinematic_equations(description: str, states: list[str]) -> list[dict]:
    """Extract direct rate identities from labels or explicit derivative equations."""
    description = str(description or "")
    assignments = list(re.finditer(
        r"(?<![\w])(?P<label>s(?:\d+|_[a-z][a-z0-9_]*))\s*=\s*"
        r"(?P<value>.*?)(?=\s*,?\s*s(?:\d+|_[a-z][a-z0-9_]*)\s*=|[,;\n]|$)",
        description, flags=re.I | re.S,
    ))

    derivative_words = {
        "dot", "rate", "rates", "derivative", "derivatives", "velocity", "velocities",
    }
    filler_words = {
        "a", "an", "the", "of", "is", "state", "signal", "angle", "angular",
    }

    def semantic_role(value: str, *, state_column: bool = False) -> tuple[frozenset[str], bool]:
        source = str(value or "").casefold()
        if state_column:
            source = re.sub(r"^s_?", "", source)
            source = source.replace("_", " ")
        words = re.findall(r"[a-z0-9]+", source)
        if not state_column:
            markers = [index for index, word in enumerate(words) if word in derivative_words]
            if markers:
                words = words[:markers[0] + 1]
        is_rate = any(word in derivative_words for word in words)
        words = [word for word in words if word not in derivative_words]
        if state_column and words:
            if words[0] in {"d", "xdot"}:
                is_rate = True
                words = words[1:]
            elif words[0].startswith("d") and len(words[0]) > 1:
                is_rate = True
                words[0] = words[0][1:]
        core = frozenset(word for word in words if word not in filler_words)
        return core, is_rate

    numeric_labels = [int(match.group("label")[1:]) for match in assignments
                      if match.group("label")[1:].isdigit()]
    index_base = 0 if 0 in numeric_labels else 1
    state_roles = [semantic_role(state, state_column=True) for state in states]
    definitions = []
    for match in assignments:
        label = match.group("label").casefold()
        value = match.group("value").strip()
        core, is_rate = semantic_role(value)
        if not core:
            continue
        exact_column = next((index for index, state in enumerate(states)
                             if str(state).casefold() == label), None)
        if exact_column is not None:
            state_index = exact_column
        elif label[1:].isdigit():
            state_index = int(label[1:]) - index_base
        else:
            matching_columns = [index for index, role in enumerate(state_roles)
                                if role == (core, is_rate)]
            state_index = matching_columns[0] if len(matching_columns) == 1 else -1
        if not 0 <= state_index < len(states):
            continue
        if not is_rate:
            is_rate = state_roles[state_index][1]
        if not core:
            core = state_roles[state_index][0]
        definitions.append({
            "label": label,
            "state_index": state_index,
            "is_rate": is_rate,
            "core": core,
            "source": value,
        })

    relations = []
    for rate in definitions:
        if not rate["is_rate"] or not rate["core"]:
            continue
        bases = [item for item in definitions
                 if not item["is_rate"] and item["core"] == rate["core"]
                 and item["state_index"] != rate["state_index"]]
        if len(bases) != 1:
            continue
        base = bases[0]
        base_index, rate_index = base["state_index"], rate["state_index"]
        if not (0 <= base_index < len(states) and 0 <= rate_index < len(states)):
            continue
        relations.append({
            "state": states[base_index],
            "expression": f"s{rate_index}",
            "rate_state": states[rate_index],
            "client_labels": [base["label"], rate["label"]],
            "source": "client_state_definitions",
        })

    # Accept an explicit ODE form such as `d(s_pitch)/dt = s_dpitch` or the
    # shorthand `d(s_pitch) = s_dpitch`. The older label parser only accepted
    # `s_pitch = pitch` plus a separate `s_dpitch = pitch dot` definition.
    numbered_labels = _numbered_state_assignments(description, states)

    def resolve_state(label: str) -> int | None:
        token = str(label or "").strip().casefold()
        exact = next((index for index, state in enumerate(states)
                      if str(state).casefold() == token), None)
        if exact is not None:
            return exact
        match = re.fullmatch(r"s(\d+)", token)
        if not match:
            return None
        number = int(match.group(1))
        if token in numbered_labels:
            return numbered_labels[token]
        if number == 0:
            return 0 if states else None
        # Client prose most often numbers states from one. Preserve explicit
        # zero-based notation when s0 is used anywhere in the equation text.
        has_zero_alias = bool(re.search(r"(?<!\w)s0(?!\w)", description, re.I))
        index = number if has_zero_alias else number - 1
        return index if 0 <= index < len(states) else None

    derivative_assignments = re.finditer(
        r"(?<![\w])d\s*\(\s*(?P<state>s(?:\d+|_[a-z][a-z0-9_]*))\s*\)\s*"
        r"(?:/\s*dt\s*)?=\s*(?P<value>[^,;\n]+)",
        description, flags=re.I,
    )
    for match in derivative_assignments:
        state_index = resolve_state(match.group("state"))
        value = match.group("value").strip().rstrip(" .")
        rate_index = resolve_state(value)
        if state_index is None or rate_index is None:
            continue
        relation = {
            "state": states[state_index],
            "expression": f"s{rate_index}",
            "rate_state": states[rate_index],
            "client_labels": [match.group("state").casefold(), value.casefold()],
            "source": "explicit_derivative_equation",
        }
        relations = [item for item in relations if item["state"] != relation["state"]]
        relations.append(relation)
    return relations


def _infer_kinematic_equations_from_columns(states: list[str]) -> list[dict]:
    """Find uniquely named coordinate/rate pairs for provisional PINN drafts.

    Column labels can suggest a useful PINN structure, but are never sufficient
    evidence for PCNN hard constraints. The caller must keep these marked as
    assumptions and must not use them in PCNN mode.
    """
    derivative_suffix = re.compile(r"(?:_?(?:dot|rate|velocity|vel|derivative)|_?xdot)$", re.I)

    def role(column: str) -> tuple[str, bool]:
        name = str(column or "").strip().casefold()
        name = re.sub(r"^s_?", "", name)
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        is_rate = False
        suffix_match = derivative_suffix.search(name)
        if suffix_match:
            name = name[:suffix_match.start()].strip("_")
            is_rate = True
        elif name.startswith("xdot_"):
            name = name[5:]
            is_rate = True
        elif name.startswith("d_") and len(name) > 2:
            name = name[2:]
            is_rate = True
        elif name.startswith("d") and len(name) > 1:
            # Common measurement names are `dpitch` and `dyaw`.
            name = name[1:]
            is_rate = True
        return name.strip("_"), is_rate

    roles = [role(state) for state in states]
    pairs = []
    for base_index, (base_name, base_is_rate) in enumerate(roles):
        if base_is_rate or not base_name:
            continue
        matching_rates = [index for index, (rate_name, is_rate) in enumerate(roles)
                          if is_rate and rate_name == base_name and index != base_index]
        if len(matching_rates) != 1:
            continue
        rate_index = matching_rates[0]
        if sum(1 for name, is_rate in roles if not is_rate and name == base_name) != 1:
            continue
        pairs.append({
            "state": states[base_index],
            "expression": f"s{rate_index}",
            "rate_state": states[rate_index],
            "client_labels": [states[base_index], states[rate_index]],
            "source": "column_name_inference",
        })
    return pairs


def _merge_kinematic_candidates(explicit: list[dict], inferred: list[dict]) -> list[dict]:
    """Prefer client-stated pairings and use unique column-name pairs for gaps."""
    merged = {item["state"]: item for item in inferred}
    merged.update({item["state"]: item for item in explicit})
    return list(merged.values())


def _independent_axes_are_explicit(description: str) -> bool:
    return bool(re.search(
        r"\b(?:independent|uncoupled|decoupled)\b|"
        r"\b(?:no|not|doesn['’]?t|doesnt|dont|don't|do not|does not)\b.{0,80}"
        r"\b(?:relation|relationship|coupl(?:e|ed|ing)|depend(?:ent|ence)|related)\b",
        str(description or ""), flags=re.I,
    ))


def _independent_second_order_fallback(
    description: str, states: list[str], actions: list[str], frame: pd.DataFrame,
    kinematic_equations: list[dict],
) -> EquationResponse | None:
    """Build a clearly limited local model when a provider loops on clarification.

    This path only applies to a described two-axis helicopter when both measured
    coordinate/rate pairs are explicitly stated or uniquely suggested by their
    column names, and the table contains a usable time column. It is a reviewable
    local linear hypothesis, not a claim that the hardware follows a verified law.
    """
    described = str(description or "")
    is_two_axis_system = bool(re.search(
        r"\bhelicopter\b|\b2\s*[- ]?\s*dof\b|\btwo\s*[- ]?\s*degree[s]?\s+of\s+freedom\b",
        described, flags=re.I,
    ))
    if not is_two_axis_system or len(states) != 4 or len(actions) < 1 or len(kinematic_equations) != 2:
        return None
    independent_axes = _independent_axes_are_explicit(described)

    paired_indices = []
    for relation in kinematic_equations:
        try:
            angle_index = states.index(relation["state"])
            rate_index = states.index(relation["rate_state"])
        except (KeyError, ValueError):
            return None
        if angle_index == rate_index:
            return None
        paired_indices.append((angle_index, rate_index, relation))
    if len({index for angle, rate, _ in paired_indices for index in (angle, rate)}) != 4:
        return None

    time_column = next((column for column in frame.columns
                        if re.sub(r"[^a-z]", "", str(column).casefold()) in {
                            "t", "time", "timestamp", "times", "timesec",
                            "timesecs", "timeseconds",
                        }), None)
    if time_column is None or len(frame) < 3:
        return None
    time_values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time_values).all() or np.any(np.diff(time_values) <= 0):
        return None

    assumptions = [
        ("The client describes independent axes, so cross-axis state terms are omitted." if independent_axes else
         "Axis coupling was not specified, so cross-axis state terms remain learnable in this local candidate."),
        "The actuator-to-axis mapping was not specified, so both confirmed inputs are included as learnable gains for each axis. Review whether both inputs can affect each axis.",
        "Coefficient bounds are provisional numerical limits scaled from the measured rates, states, inputs, and time column; units were not provided, so review them before approval.",
    ]
    inferred_pairs = [item for item in kinematic_equations
                      if item.get("source") == "column_name_inference"]
    if inferred_pairs:
        labels = ", ".join(
            f"{item['state']} ↔ {item['rate_state']}" for item in inferred_pairs
        )
        assumptions.insert(
            0,
            "The coordinate/rate pairing(s) " + labels +
            " were inferred from column names, not confirmed by the client; verify before approval.",
        )
    equations: list[EquationTerm] = []
    parameters: list[PhysicsParameter] = []
    for angle_index, rate_index, relation in paired_indices:
        equations.append(EquationTerm(
            state=relation["state"], expression=f"s{rate_index}"
        ))

    def coefficient_bound(feature_values: np.ndarray, target_values: np.ndarray) -> float:
        target_scale = max(
            float(np.nanpercentile(np.abs(target_values), 95)),
            float(np.nanstd(target_values)), 1e-8,
        )
        centered = feature_values - float(np.nanmedian(feature_values))
        feature_scale = max(
            float(np.nanstd(feature_values)),
            float(np.nanpercentile(np.abs(centered), 95)), 1e-8,
        )
        if feature_scale <= 1e-8:
            feature_scale = max(abs(float(np.nanmedian(feature_values))), 1.0)
        return min(max(10.0 * target_scale / feature_scale, 1e-6), 1e6)

    for angle_index, rate_index, relation in paired_indices:
        angle_name, rate_name = states[angle_index], states[rate_index]
        rate_values = pd.to_numeric(frame[rate_name], errors="coerce").to_numpy(dtype=float)
        try:
            acceleration = np.gradient(rate_values, time_values)
        except (ValueError, FloatingPointError):
            return None
        if not np.isfinite(acceleration).all():
            return None

        term_specs = [("offset", None, np.ones(len(frame), dtype=float))]
        state_terms = ([(angle_index, states[angle_index]), (rate_index, states[rate_index])]
                       if independent_axes else list(enumerate(states)))
        for state_index, column in state_terms:
            column_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            term_specs.append((f"response to {column}", f"s{state_index}", column_values))
        for action_index, action_name in enumerate(actions):
            action_values = pd.to_numeric(frame[action_name], errors="coerce").to_numpy(dtype=float)
            term_specs.append((f"response to {action_name}", f"a{action_index}", action_values))

        terms = []
        for name, alias, feature_values in term_specs:
            parameter_alias = f"p{len(parameters)}"
            bound = coefficient_bound(feature_values, acceleration)
            parameters.append(PhysicsParameter(
                alias=parameter_alias,
                name=f"{rate_name} {name}",
                initial=0.0,
                lower=-bound,
                upper=bound,
                rationale=(
                    "Provisional finite training range scaled from measured magnitudes; "
                    "the physical units and parameter range were not supplied."
                ),
            ))
            terms.append(parameter_alias if alias is None else f"{parameter_alias}*{alias}")
        equations.append(EquationTerm(state=rate_name, expression=" + ".join(terms)))

    axis_names = [states[angle_index] for angle_index, _rate_index, _relation in paired_indices]
    return EquationResponse(
        status="ready",
        message=(
            f"Drafted a local candidate for {', '.join(axis_names)} using the rate pairings "
            "you specified; coupling and actuator effects follow the stated assumptions for review."
        ),
        equations=equations,
        parameters=parameters,
        assumptions=assumptions,
    )


def _physics_proposal_prompt(chat: dict, description: str, dataset: dict, frame: pd.DataFrame,
                             checks: list[str],
                             kinematic_equations: list[dict] | None = None,
                             column_name_kinematics: list[dict] | None = None) -> str:
    states, actions = list(dataset["states"]), list(dataset.get("actions") or [])
    mode = str((chat.get("setup") or {}).get("pending_physics_mode") or
               (chat.get("settings") or {}).get("physics_mode") or "pinn").strip().lower()
    payload = {
        "physics_mode": mode,
        "task": (
            "Identify only state-derivative rules that are explicitly exact hard constraints in the client's description (clear kinematic identities may qualify). Return equations only for those constrained states; leave all other state outputs to the neural network. Give confidence and a brief rationale for each candidate. If no exact rule is supported, ask one focused clarification instead of inventing constraints."
            if mode == "pcnn" else
            "Draft a candidate continuous-time derivative model for every measured state using the client's description, map it to confirmed measurements, and mark unknown coefficients for bounded learning. This is a soft PINN training guide, not a verified law."
        ),
        "client_description_untrusted": description[:4000],
        "explicit_kinematic_equations_from_description": kinematic_equations or [],
        "provisional_rate_pairings_inferred_from_column_names": (
            column_name_kinematics or [] if mode == "pinn" else []
        ),
        "column_name_pairing_limit": (
            "For PINN only, these unique name-based pairings may support a review-only candidate and must be disclosed as assumptions. They are not client-confirmed and must never become PCNN hard constraints."
        ),
        "client_numbered_state_aliases": {
            label: f"s{index}" for label, index in _numbered_state_assignments(description, states).items()
        },
        "expression_alias_contract": (
            "Use only the supplied zero-based s0..sN and a0..aN aliases in equation expressions. "
            "Client labels such as s1..s4 in the description are descriptive labels; translate them "
            "using client_numbered_state_aliases instead of copying them as expression symbols."
        ),
        "kinematic_equation_instruction": (
            "Include every listed explicit kinematic identity exactly as an equation for its exact state column. "
            "These identities come from the client's stated meanings, not from column names. Do not ask the client to restate them."
        ),
        "recent_client_context_untrusted": [str(item.get("content", ""))[:1500]
            for item in chat.get("messages", [])[-8:] if item.get("role") == "user"],
        "dataset": {
            "name": dataset.get("name"), "rows": int(dataset["rows"]),
            "time_column": "time" if "time" in frame.columns else None,
            "states": _column_summary(frame, states, [f"s{i}" for i in range(len(states))]),
            "inputs": _column_summary(frame, actions, [f"a{i}" for i in range(len(actions))]),
            "data_checks": checks,
        },
        "output_schema": {
            "status": "ready or clarification",
            "message": "short client-facing summary; for clarification ask one focused question",
            "equations": [{"state": "exact state column", "expression": "safe derivative expression using sN, aN, pN"}],
            "parameters": [{"alias": "p0", "name": "coefficient", "initial": 1.0,
                "lower": 0.1, "upper": 10.0, "units": "unit or empty",
                "rationale": "why this coefficient is unknown and why its range is defensible"}],
            "pcnn_constraints": [{"state": "exact constrained state column", "recommended_exact": True,
                "confidence": "low, medium, or high",
                "rationale": "brief reason based on the client's stated mechanism"}],
            "assumptions": ["explicit model assumptions for client review"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _pcnn_explicit_kinematic_fallback(kinematic_equations: list[dict]) -> EquationResponse | None:
    """Turn client-stated state-rate identities into reviewable exact candidates."""
    if not kinematic_equations:
        return None
    equations = [EquationTerm(state=item["state"], expression=item["expression"])
                 for item in kinematic_equations]
    recommendations = [PCNNConstraintRecommendation(
        state=item["state"], recommended_exact=True, confidence="high",
        rationale=f"The client explicitly defines this state as the rate of {item['rate_state']}.",
    ) for item in kinematic_equations]
    assumptions = [f"The client defines d({item['state']})/dt as {item['rate_state']}."
                   for item in kinematic_equations]
    return EquationResponse(
        status="ready",
        message="I mapped the explicitly stated kinematic relationship(s) as PCNN hard-constraint candidates. Other state derivatives remain neural-network outputs.",
        equations=equations, pcnn_constraints=recommendations, assumptions=assumptions,
    )


def _physics_review_fallback(pcnn_mode: bool, description: str, states: list[str],
                             actions: list[str], frame: pd.DataFrame,
                             kinematic_equations: list[dict]) -> EquationResponse | None:
    """Return only a constrained local proposal supported by the client's description."""
    if pcnn_mode:
        # PCNN cannot infer a complete physical model. Recover only explicitly
        # stated kinematic identities; every other derivative stays learned.
        return _pcnn_explicit_kinematic_fallback(kinematic_equations)
    return _independent_second_order_fallback(
        description, states, actions, frame, kinematic_equations
    )


def propose_physics_model(chat: dict, description: str, *, client=None,
                          output_dir: str | Path | None = None) -> dict:
    """Build a review-only physics proposal from client language and measured columns."""
    dataset = chat.get("dataset") or {}
    physics_mode = _physics_mode_for_chat(chat)
    pcnn_mode = physics_mode == "pcnn"
    description = str(description or "").strip()
    if len(description) < 12:
        raise ValueError("Describe what the system is and how its measured states change; add one or two details before drafting physics.")
    if not dataset.get("ready") or dataset.get("column_review_pending"):
        raise ValueError("Confirm the dataset time, state, and input columns before drafting a physics model.")
    frame = _dataset_frame(dataset)
    states, actions = list(dataset["states"]), list(dataset.get("actions") or [])
    if not actions:
        raise ValueError("A physics model needs at least one confirmed system input column.")
    if len(states) > 100 or len(actions) > 100:
        raise ValueError("Physics model drafting supports up to 100 state and 100 input columns.")
    if client is None:
        client = DiagnosticClient(DiagnosticSettings.defaults())
    checks = _physics_data_checks(frame, states, actions)
    kinematic_equations = _explicit_kinematic_equations(description, states)
    column_name_kinematics = _infer_kinematic_equations_from_columns(states)
    pinn_kinematic_candidates = _merge_kinematic_candidates(
        kinematic_equations, column_name_kinematics
    )
    fallback_kinematics = kinematic_equations if pcnn_mode else pinn_kinematic_candidates
    prompt = _physics_proposal_prompt(
        chat, description, dataset, frame, checks, kinematic_equations,
        column_name_kinematics,
    )
    physics_system_prompt = system_prompt("physics_model_agent")
    used_local_fallback = False
    try:
        response = _provider_response(client.complete(physics_system_prompt, prompt))
    except Exception:
        fallback = _physics_review_fallback(
            pcnn_mode, description, states, actions, frame, fallback_kinematics
        )
        if fallback is None:
            raise
        response = fallback
        used_local_fallback = True
    model_name = getattr(getattr(client, "settings", None), "model", "configured model")
    if response.status == "clarification":
        if response.equations or response.parameters or response.pcnn_constraints:
            raise ValueError("The model request mixed a question with unreviewed equations. Please try the description again.")
        if not kinematic_equations:
            fallback = _physics_review_fallback(
                pcnn_mode, description, states, actions, frame, fallback_kinematics
            )
            if fallback is None:
                return {"answer": response.message, "status": "clarification", "generated_artifact": None,
                        "equations": [], "parameters": [], "assumptions": response.assumptions,
                        "data_checks": checks, "evidence": [], "model": model_name}
            response = fallback
            used_local_fallback = True

    if response.status == "clarification":
        independent_axes = _independent_axes_are_explicit(description)
        axis_instruction = (
            "The client explicitly says the axes are independent; do not add cross-axis state terms."
            if independent_axes else
            "Follow the client's stated coupling; do not add cross-axis terms without support in the description."
        )
        completion_task = (
            "The previous reply asked for details that are already stated by the client. "
            "Include the explicitly stated kinematic identities below as exact PCNN candidates. "
            "Do not require equations for the other states; they remain learned by the neural network. "
            "Do not ask the client to restate the identities."
            if pcnn_mode else
            (
                "The previous reply asked for details that are already stated by the client. "
                "Use the explicit kinematic equations below and complete the candidate model for "
                "every other confirmed state. Also consider the provisional coordinate/rate "
                "pairings inferred from column names in the original request, disclose those "
                "pairings as assumptions, and do not treat them as confirmed physical laws. "
                + axis_instruction + " "
                "The input columns are confirmed; where an actuator's axis assignment is unknown, "
                "use bounded learnable input gains and state the mapping as an assumption for "
                "review instead of blocking the draft. Return ready unless a required state or "
                "input column is actually missing."
            )
        )
        completion_prompt = json.dumps({
            "task": completion_task,
            "original_request": json.loads(prompt),
            "explicit_kinematic_equations": kinematic_equations,
            "provisional_rate_pairings_inferred_from_column_names": column_name_kinematics,
            "client_stated_axes_independent": independent_axes,
            "previous_clarification": response.message,
            "required_state_columns": states,
            "confirmed_input_columns": actions,
        }, ensure_ascii=False, allow_nan=False)
        try:
            response = _provider_response(
                client.complete(physics_system_prompt, completion_prompt)
            )
        except Exception:
            fallback = _physics_review_fallback(
                pcnn_mode, description, states, actions, frame, fallback_kinematics
            )
            if fallback is not None:
                response = fallback
                used_local_fallback = True
            else:
                mapped_rates = "; ".join(
                    f"d({item['state']})/dt = {item['rate_state']}"
                    for item in kinematic_equations
                )
                return {
                    "answer": (
                        f"Your description gives these rate equations: {mapped_rates}. "
                        "I couldn't complete the remaining dynamics on the automatic retry. "
                        f"Please confirm how the measured inputs {', '.join(actions)} affect the separate axes."
                    ),
                    "status": "clarification", "generated_artifact": None,
                    "equations": [], "parameters": [], "assumptions": [],
                    "data_checks": checks, "evidence": [], "model": model_name,
                }
        if response.status == "clarification":
            if response.equations or response.parameters or response.pcnn_constraints:
                raise ValueError("The model request mixed a question with unreviewed equations. Please try the description again.")
            fallback = _physics_review_fallback(
                pcnn_mode, description, states, actions, frame, fallback_kinematics
            )
            if fallback is not None:
                response = fallback
                used_local_fallback = True
            else:
                mapped_rates = "; ".join(
                    f"d({item['state']})/dt = {item['rate_state']}"
                    for item in kinematic_equations
                )
                return {
                    "answer": (
                        f"Your description gives these rate equations: {mapped_rates}. "
                        "I still need the remaining state dynamics to complete the draft. "
                        f"Could you explain how the confirmed inputs {', '.join(actions)} act on the independent axes?"
                    ),
                    "status": "clarification", "generated_artifact": None,
                    "equations": [], "parameters": [], "assumptions": response.assumptions,
                    "data_checks": checks, "evidence": [], "model": model_name,
                }

    if kinematic_equations:
        existing_equation_states = {equation.state for equation in response.equations}
        inferred_equations = [
            EquationTerm(state=item["state"], expression=item["expression"])
            for item in kinematic_equations if item["state"] not in existing_equation_states
        ]
        if inferred_equations:
            explicit_by_state = {item["state"]: item for item in kinematic_equations}
            constraints_by_state = {
                item.state: item for item in response.pcnn_constraints
            }
            for equation in inferred_equations:
                relation = explicit_by_state[equation.state]
                constraints_by_state[equation.state] = PCNNConstraintRecommendation(
                    state=equation.state,
                    recommended_exact=True,
                    confidence="high",
                    rationale=(
                        "The client explicitly defines this state as the rate of "
                        f"{relation['rate_state']}."
                    ),
                )
            inferred_assumptions = list(response.assumptions)
            for equation in inferred_equations:
                relation = explicit_by_state[equation.state]
                statement = (
                    f"The client defines d({equation.state})/dt as {relation['rate_state']}."
                )
                if statement not in inferred_assumptions:
                    inferred_assumptions.append(statement)
            ordered_constraints = [
                constraints_by_state[state] for state in states
                if state in constraints_by_state
            ]
            ordered_constraints.extend(
                item for state, item in constraints_by_state.items() if state not in states
            )
            response = response.model_copy(update={
                "equations": [*response.equations, *inferred_equations],
                "pcnn_constraints": ordered_constraints,
                "assumptions": inferred_assumptions,
            })

    target_order = [equation.state for equation in response.equations]
    valid_coverage = (
        bool(target_order) and set(target_order).issubset(set(states))
        if pcnn_mode else
        len(target_order) == len(states) and set(target_order) == set(states)
    )
    if not valid_coverage or len(set(target_order)) != len(target_order):
        counts = {state: target_order.count(state) for state in set(target_order)}
        coverage_issue = {
            "required_state_columns": states,
            "missing_state_columns": [state for state in states if state not in counts],
            "repeated_state_columns": [state for state, count in counts.items() if count > 1],
            "unexpected_state_columns": [state for state in counts if state not in states],
            "returned_state_order": target_order,
        }
        repair_prompt = json.dumps({
            "task": (
                ("Repair the previous proposal. Its PCNN hard-constraint map was empty, repeated, or used unknown states. Return only supported exact state-derivative rules and omit states for neural-network learning. "
                 if pcnn_mode else
                 "Repair the previous proposal. It did not provide exactly one equation for every confirmed state column. ")
                + "Return the full required JSON response, not a patch. Use the client's explicit state meanings and the dataset's exact column "
                "mapping. Preserve clear kinematic identities, use bounded learnable coefficients for "
                "unknown dynamics, and list uncertain input assignments as assumptions for review. "
                "Do not ask for details already supplied. If a complete candidate truly cannot be "
                "supported, return clarification with empty equation, parameter, and constraint lists "
                "and ask one focused question about the specific missing mapping."
            ),
            "original_request": json.loads(prompt),
            "validation_issue": coverage_issue,
            "previous_proposal": response.model_dump(),
        }, ensure_ascii=False, allow_nan=False)
        try:
            response = _provider_response(client.complete(physics_system_prompt, repair_prompt))
        except Exception:
            # Preserve the original proposal so the user gets a focused mapping
            # question below instead of losing the failure to a second-call error.
            pass
        target_order = [equation.state for equation in response.equations]
        coverage_valid = (
            bool(target_order) and set(target_order).issubset(set(states))
            if pcnn_mode else
            len(target_order) == len(states) and set(target_order) == set(states)
        )
        if (response.status == "clarification" or not coverage_valid
                or len(set(target_order)) != len(target_order)):
            fallback = _physics_review_fallback(
                pcnn_mode, description, states, actions, frame, fallback_kinematics
            )
            if fallback is not None:
                response = fallback
                used_local_fallback = True
                target_order = [equation.state for equation in response.equations]
        if response.status == "clarification":
            if response.equations or response.parameters or response.pcnn_constraints:
                raise ValueError("The model request mixed a question with unreviewed equations. Please try the description again.")
            return {"answer": response.message, "status": "clarification", "generated_artifact": None,
                    "equations": [], "parameters": [], "assumptions": response.assumptions,
                    "data_checks": checks, "evidence": [], "model": model_name}

    aliases = [item.alias for item in response.parameters]
    expected_aliases = [f"p{i}" for i in range(len(aliases))]
    if aliases != expected_aliases:
        raise ValueError("The proposed coefficients must use a consecutive p0, p1, … mapping.")
    if len(set(aliases)) != len(aliases):
        raise ValueError("The proposed physics model repeats a coefficient alias.")
    for parameter in response.parameters:
        if not parameter.lower < parameter.upper or not parameter.lower <= parameter.initial <= parameter.upper:
            raise ValueError(f"Check the proposed range for {parameter.name}: the initial value must be inside finite lower and upper bounds.")

    valid_coverage = (
        bool(target_order) and set(target_order).issubset(set(states))
        if pcnn_mode else
        len(target_order) == len(states) and set(target_order) == set(states)
    )
    if not valid_coverage or len(set(target_order)) != len(target_order):
        missing_states = [state for state in states if state not in target_order]
        unexpected_states = list(dict.fromkeys(state for state in target_order if state not in states))
        mismatch = []
        if missing_states:
            mismatch.append("missing confirmed state columns: " + ", ".join(missing_states))
        if unexpected_states:
            mismatch.append("unrecognized state names: " + ", ".join(unexpected_states))
        if not mismatch:
            mismatch.append("a confirmed state was repeated instead of receiving its own rule")
        return {
            "answer": (
                ("I couldn't finish a PCNN hard-constraint map because the draft had " if pcnn_mode else
                 "I couldn't finish a complete PINN equation set because the draft had ")
                + "; ".join(mismatch)
                + (". Please identify a state derivative that should be fixed exactly, or describe its precise physical relationship."
                   if pcnn_mode else
                   ". Please correct the state mapping or describe how those state rates respond to the measured inputs.")
            ),
            "status": "clarification",
            "generated_artifact": None,
            "equations": [],
            "parameters": [],
            "assumptions": [],
            "data_checks": checks,
            "evidence": [],
            "model": model_name,
        }
    constraint_order = [item.state for item in response.pcnn_constraints]
    defined_states = list(dict.fromkeys(target_order))
    valid_pcnn_recommendations = bool(response.pcnn_constraints) and (
        len(constraint_order) == len(defined_states) and len(set(constraint_order)) == len(constraint_order)
        and set(constraint_order) == set(defined_states)
    )
    if valid_pcnn_recommendations:
        recommendations_by_state = {
            item.state: {**item.model_dump(), "source": "agent"}
            for item in response.pcnn_constraints
        }
    else:
        # Missing, malformed, or incomplete advisory output stays conservative:
        # no candidate is preselected as an exact constraint.
        recommendations_by_state = {
            state: {
                "state": state,
                "recommended_exact": False,
                "confidence": "low",
                "rationale": "No exact-constraint recommendation was supplied; review this rule before enforcing it.",
                "source": "safe_default",
            }
            for state in defined_states
        }
    pcnn_recommendations = [recommendations_by_state[state] for state in defined_states]
    one_based_states, one_based_actions = (
        (False, False) if used_local_fallback else
        _model_uses_client_one_based_aliases(response, description, states, actions)
    )
    by_state = {
        item.state: _map_column_names_to_aliases(
            item.expression if used_local_fallback else _normalize_numbered_model_aliases(
                    item.expression, description, states, actions,
                    one_based_states=one_based_states,
                    one_based_actions=one_based_actions,
                ),
            states, actions,
        )
        for item in response.equations
    }
    parsed = [_parse_expression(by_state[name], len(states), len(actions), len(aliases)) if name in by_state else None
              for name in states]
    used_parameters = {node.id for expression in parsed if expression is not None for node in ast.walk(expression)
                       if isinstance(node, ast.Name) and re.fullmatch(r"p\d+", node.id)}
    if used_parameters != set(aliases):
        raise ValueError("Every declared learnable coefficient must appear in an equation, and every pN symbol must be declared.")
    _validate_on_dataset(parsed, frame, states, actions, response.parameters)
    code = _generated_module(parsed, len(states), len(actions), response.parameters)
    source_digest = hashlib.sha256(description.encode("utf-8")).hexdigest()
    generated_identity = hashlib.sha256((source_digest + dataset["sha256"] +
        json.dumps({"equations": by_state, "parameters": [item.model_dump() for item in response.parameters],
                    "pcnn_constraints": pcnn_recommendations}, sort_keys=True)).encode()).hexdigest()
    folder = Path(output_dir) if output_dir is not None else _chat_folder(chat["id"], "pinn_equations")
    folder.mkdir(parents=True, exist_ok=True)
    artifact_prefix = "pcnn_constraints" if pcnn_mode else "pinn_proposal"
    destination = folder / f"{artifact_prefix}_{generated_identity[:16]}.py"
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(code.encode()).hexdigest():
        handle, temporary = tempfile.mkstemp(prefix=".pinn-proposal-", suffix=".py", dir=folder)
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
    equations = [{"state": name, "alias": f"s{index}", "expression": by_state[name]}
                 for index, name in enumerate(states) if name in by_state]
    artifact = {"kind": "pinn_equation", "physics_mode": physics_mode,
                "origin": "ai_proposal", "approved": False,
                "requires_approval": True, "name": destination.name, "path": str(destination.resolve()),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "source_name": "client system description", "source_sha256": source_digest,
                "dataset_sha256": dataset["sha256"], "states": states, "actions": actions,
                "equations": equations, "parameters": [item.model_dump() for item in response.parameters],
                "pcnn_constraint_recommendations": pcnn_recommendations,
                "proposal_id": generated_identity,
                "state_aliases": [{"alias": f"s{i}", "column": column} for i, column in enumerate(states)],
                "action_aliases": [{"alias": f"a{i}", "column": column} for i, column in enumerate(actions)],
                "assumptions": response.assumptions, "data_checks": checks,
                "proposal_message": response.message}
    detail = "\n".join(f"- `{item['state']}`: `{item['expression']}`" for item in equations)
    parameter_text = "\n".join(f"- `{item.alias}` · {item.name}: {item.initial:g} [{item.lower:g}, {item.upper:g}] {item.units}".strip()
                                for item in response.parameters)
    caveat = (
        "These are AI-proposed hard-constraint candidates, not verified laws. Only the listed state derivatives can be fixed; all other state derivatives remain learned by the neural network. Review the rule, units, assumptions, and coefficient bounds before approving."
        if pcnn_mode else
        "This is an AI-proposed model, not a verified physical law. Finite-value and column checks passed; review its equations, units, assumptions, and coefficient bounds before approving it."
    )
    if checks:
        caveat += "\n\nData limits: " + " ".join(checks)
    answer = "\n\n".join(part for part in (response.message.strip(), detail,
        ("Learnable coefficients (estimated during training):\n" + parameter_text) if parameter_text else "",
        ("Assumptions: " + " ".join(response.assumptions)) if response.assumptions else "", caveat) if part)
    return {"answer": answer, "status": "ready", "generated_artifact": artifact,
            "equations": equations, "parameters": artifact["parameters"],
            "assumptions": response.assumptions, "data_checks": checks,
            "evidence": [], "model": model_name}


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


def is_equation_approved(chat: dict) -> bool:
    """Require client approval only for equations generated from descriptions."""
    if not is_equation_ready(chat):
        return False
    artifact = chat.get("pinn_equation") or {}
    return artifact.get("origin") != "ai_proposal" or artifact.get("approved") is True


class PINNMakerJob:
    """Background interpreter job for a physics equation source."""
    def __init__(self, chat: dict, question: str):
        self.snapshot = copy.deepcopy(chat)
        self.question = question
        self.purpose = "pinn_maker"
        self.answer = None
        self.error = None
        self.error_detail = None
        mode = _physics_mode_for_chat(self.snapshot)
        self.phase = ("PCNN constraints · mapping exact state rules" if mode == "pcnn" else
                      "PINN model · mapping derivatives for all states")
        self.started_at = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.started_at = time.monotonic()
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def client_activity_steps(self):
        if self.phase.startswith("PCNN"):
            return ["Read the supplied hard-constraint rules", "Mapped rules to confirmed states and inputs", "Checked each constrained derivative on measured rows"]
        return ["Read the supplied derivative model", "Mapped state and input columns", "Checked all state derivatives on measured rows"]

    def _run(self):
        try:
            self.answer = prepare_equation(self.snapshot, self.question)
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:800]
            self.error = str(exc) if isinstance(exc, ValueError) else "I couldn't map the physics source. Check it and the configured AI connection, then try again."


class PhysicsModelAgentJob:
    """Background job for a review-only model draft from a client description."""
    def __init__(self, chat: dict, description: str):
        self.snapshot = copy.deepcopy(chat)
        self.description = str(description)
        self.question = self.description
        self.purpose = "physics_model_agent"
        self.answer = None
        self.error = None
        self.error_detail = None
        self.pcnn_mode = _physics_mode_for_chat(self.snapshot) == "pcnn"
        self.phase = (
            "Physics Model Agent · identifying exact PCNN constraints"
            if self.pcnn_mode else
            "Physics Model Agent · drafting a PINN model for review"
        )
        self.started_at = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.started_at = time.monotonic()
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def client_activity_steps(self):
        if self.pcnn_mode:
            return ["Read the system description", "Identified explicitly stated exact relationships", "Mapped candidate hard constraints to confirmed signals"]
        return ["Read the system description", "Mapped the confirmed signals", "Checked the candidate equations and data coverage"]

    def _run(self):
        try:
            self.answer = propose_physics_model(self.snapshot, self.description)
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:800]
            self.error = str(exc) if isinstance(exc, ValueError) else "I couldn't draft the physics proposal. Check the configured AI connection and try again."
