"""Excel Maker agent: preserve chat or tabular input and create a downloadable XLSX."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from pydantic import BaseModel, Field

from backend_core.AgentSysID.agents.prompt_library import system_prompt
from backend_core.AgentSysID.agents.run_diagnostic import DiagnosticClient, DiagnosticSettings
from backend_core.AgentSysID.agents.run_evidence import redact

try:
    from . import conversation_core as core
except ImportError:
    import conversation_core as core


_EXCEL_TERMS = r"(?:excel|spreadsheet|workbook|xlsx)"
_EXCEL_ACTIONS = r"(?:make|create|build|generate|export|save|convert|put|turn|give|get|format)"
_IDENTIFIER_HEADER = re.compile(r"(?:^|\b)(?:id|code|zip|postal|phone|telephone|sku|account|serial)(?:\b|$)", re.I)
_STEP_RESPONSE_HEADER = re.compile(
    r"(?<![\w])k\s+t\s*\(\s*s\s*\)\s+u\s*\(\s*k\s*\)\s*input\s+"
    r"y\s*\(\s*k\s*\)\s*output\b", re.I)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def is_excel_request(question: str) -> bool:
    """Recognize workbook creation requests without catching general Excel questions."""
    text = (question or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(r"\bexcel\s+maker\b", text)
        or re.search(rf"\b{_EXCEL_ACTIONS}\b.{{0,70}}\b{_EXCEL_TERMS}\b", text)
        or re.search(rf"\b{_EXCEL_TERMS}\b.{{0,40}}\b(?:this|that|from|please|for\s+me)\b", text)
        or re.search(r"\bexcel\s+(?:this|that|the\s+following)\b", text)
    )


def _json_table(text: str) -> tuple[list[str], list[list[Any]]] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(value, dict) and isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list):
        return [str(item) for item in value["columns"]], [list(row) for row in value["rows"]]
    if isinstance(value, dict) and value and all(isinstance(items, list) for items in value.values()):
        lengths = {len(items) for items in value.values()}
        if len(lengths) == 1:
            headers = list(value)
            return headers, [[value[header][index] for header in headers] for index in range(next(iter(lengths), 0))]
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        headers = list(dict.fromkeys(key for row in value for key in row))
        return headers, [[row.get(header) for header in headers] for row in value]
    if isinstance(value, list) and value and all(isinstance(row, (list, tuple)) for row in value):
        matrix = [list(row) for row in value]
        if matrix and matrix[0]:
            return [str(cell) for cell in matrix[0]], matrix[1:]
    return None


def _split_line(line: str, delimiter: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if delimiter == "|":
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    elif delimiter == "whitespace":
        cells = re.split(r"\s{2,}", stripped)
    else:
        try:
            cells = next(csv.reader([stripped], delimiter=delimiter, skipinitialspace=True, strict=True))
        except (csv.Error, StopIteration):
            return None
        cells = [cell.strip() for cell in cells]
    return cells if len(cells) >= 2 else None


def _delimited_table(text: str) -> tuple[list[str], list[list[Any]]] | None:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not re.fullmatch(r"```[\w-]*", line)]
    candidates: list[tuple[int, list[list[str]]]] = []
    malformed: list[int] = []
    for delimiter in ("\t", ",", ";", "|", "whitespace"):
        groups: list[list[list[str]]] = []
        group: list[list[str]] = []
        width = None
        for line in lines:
            cells = _split_line(line, delimiter)
            is_markdown_rule = delimiter == "|" and cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)
            if is_markdown_rule:
                continue
            if cells and (width is None or len(cells) == width):
                group.append(cells)
                width = len(cells)
            else:
                if len(group) >= 2:
                    groups.append(group)
                    # A row that looks like part of this table but has a
                    # different width must not be silently skipped. Otherwise
                    # the exported workbook could look valid while losing data.
                    if cells and len(cells) >= 2:
                        malformed.append(len(group) * width)
                group = []
                width = None
        if len(group) >= 2:
            groups.append(group)
        for candidate in groups:
            candidates.append((len(candidate) * len(candidate[0]), candidate))
    if not candidates:
        return None
    if malformed and max(malformed) >= max(score for score, _ in candidates):
        raise ValueError("The pasted table has rows with different numbers of cells. Fix the row widths and try again.")
    _, matrix = max(candidates, key=lambda item: (item[0], len(item[1][0])))
    return [str(cell) for cell in matrix[0]], [list(row) for row in matrix[1:]]


def _compact_step_response_table(text: str) -> tuple[list[str], list[list[Any]]] | None:
    """Read the common pasted `k, t(s), u(k), y(k)` table when line breaks vanished."""
    header = _STEP_RESPONSE_HEADER.search(text)
    if not header:
        return None
    tail, cursor, tokens = text[header.end():], 0, []
    while cursor < len(tail):
        separator = re.match(r"[\s,;|]+", tail[cursor:])
        if separator:
            cursor += separator.end()
        match = _NUMBER.match(tail, cursor)
        if not match:
            break
        tokens.append(match.group())
        cursor = match.end()
    if not tokens:
        raise ValueError("I found the step-response headers, but no numeric rows after them. Paste the values under the headers.")
    if len(tokens) % 4:
        raise ValueError("I found the step-response headers, but the numeric values do not form complete four-column rows. Check the pasted table and try again.")
    rows = [tokens[index:index + 4] for index in range(0, len(tokens), 4)]
    return _normalize_table(["k", "t (s)", "u(k) Input", "y(k) Output"], rows)


def parse_raw_table(text: str) -> tuple[list[str], list[list[Any]]] | None:
    """Read JSON, Markdown, CSV, TSV, semicolon, pipe, or aligned text tables."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [text.strip()]
    candidates.extend(block.strip() for block in re.findall(r"```(?:json|csv|tsv|text)?\s*\n(.*?)```", text,
                                                             flags=re.I | re.S))
    for candidate in candidates:
        parsed = _json_table(candidate)
        if parsed and parsed[0] and parsed[1]:
            return _normalize_table(*parsed)
        parsed = _compact_step_response_table(candidate)
        if parsed and parsed[0] and parsed[1]:
            return parsed
        parsed = _delimited_table(candidate)
        if parsed and parsed[0] and parsed[1]:
            return _normalize_table(*parsed)
    return None


def table_csv_bytes(headers: list[str], rows: list[list[Any]]) -> bytes:
    """Serialize parsed chat data to a local CSV without changing its values."""
    headers, rows = _normalize_table(headers, rows)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _normalize_table(headers: list[Any], rows: list[list[Any]]) -> tuple[list[str], list[list[Any]]]:
    if not headers or not rows:
        raise ValueError("Paste a header row and at least one data row.")
    if len(headers) > 200 or len(rows) > 25_000:
        raise ValueError("Excel Maker supports up to 200 columns and 25,000 data rows per chat request.")
    normalized_headers, used = [], set()
    for index, value in enumerate(headers):
        base = str(value).strip() or f"Column {index + 1}"
        candidate, suffix = base, 2
        while candidate.casefold() in used:
            candidate = f"{base} ({suffix})"
            suffix += 1
        used.add(candidate.casefold())
        normalized_headers.append(candidate)
    normalized_rows = []
    for index, row in enumerate(rows):
        if len(row) != len(normalized_headers):
            raise ValueError(f"Data row {index + 1} has {len(row)} cells; expected {len(normalized_headers)}.")
        normalized_rows.append([_typed_value(value, header) for value, header in zip(row, normalized_headers)])
    return normalized_headers, normalized_rows


def _typed_value(value: Any, header: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, date, datetime)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if _IDENTIFIER_HEADER.search(header):
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?", text):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            # Excel cells cannot represent timezone-aware datetimes; keep the
            # original timestamp text so its offset is preserved exactly.
            return text if parsed.tzinfo is not None else parsed
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        try:
            number = float(text)
            if not math.isfinite(number):
                return text
            return int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number
        except ValueError:
            pass
    return text


def _read_file_table(path: str | Path) -> tuple[list[str], list[list[Any]]] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        import pandas as pd
        frame = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_excel(source)
        if frame.empty or len(frame.columns) == 0:
            return None
        frame = frame.astype(object).where(frame.notna(), None)
        return _normalize_table([str(column) for column in frame.columns], frame.values.tolist())
    except (OSError, ValueError, ImportError):
        return None


def _source_table(chat: dict) -> tuple[list[str], list[list[Any]]]:
    for message in reversed(chat.get("messages", [])):
        if message.get("role") != "user":
            continue
        # Keep each turn's text beside its upload: the newest turn should win,
        # while its attachment should take precedence over an older pasted table.
        attachment = message.get("attachment")
        path = attachment.get("path") if isinstance(attachment, dict) else None
        if path:
            attached = _read_file_table(path)
            if attached:
                return attached
            raise ValueError("I couldn't read the latest attachment as a table. Please use a CSV or Excel file with a header row and data rows.")
        parsed = parse_raw_table(str(message.get("content", "")))
        if parsed:
            return parsed
    dataset = chat.get("dataset") or {}
    attached = _read_file_table(dataset.get("path")) if dataset.get("path") else None
    if attached:
        return attached
    raise ValueError("I couldn't find a table in this conversation. Paste a header row plus data rows as CSV, TSV, JSON, or a Markdown table, then ask Excel Maker to create the workbook.")


class WorkbookPlan(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    sheet_name: str = Field(min_length=1, max_length=31)


def _fallback_plan() -> WorkbookPlan:
    return WorkbookPlan(title="Chat data export", sheet_name="Data")


def _make_plan(question: str, headers: list[str], row_count: int, client=None) -> tuple[WorkbookPlan, Any, str | None]:
    try:
        llm = client or DiagnosticClient(DiagnosticSettings.defaults())
        request = json.dumps({"request": "Suggest a clear workbook title using the column names.",
                              "columns": headers, "data_row_count": row_count},
                             ensure_ascii=False)
        raw = llm.complete(system_prompt("excel_maker"), request)
        plan = WorkbookPlan.model_validate_json(raw.strip().removeprefix("```json").removesuffix("```").strip())
        return _clean_plan(plan), llm, None
    except Exception as exc:
        llm = client
        message = redact(f"{type(exc).__name__}: {exc}")[:400]
        return _fallback_plan(), llm, message


def _clean_plan(plan: WorkbookPlan) -> WorkbookPlan:
    title = " ".join(plan.title.split()).strip() or "Chat data export"
    sheet_name = re.sub(r"[:\\/?*\[\]]", " ", plan.sheet_name)
    sheet_name = " ".join(sheet_name.split()).strip("' ")[:31] or "Data"
    return WorkbookPlan(title=title[:120], sheet_name=sheet_name)


def _safe_cell(cell, value: Any) -> None:
    cell.value = value
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        # Store user content as literal text; never turn pasted input into formulas.
        cell.data_type = "s"


def build_workbook(path: str | Path, title: str, sheet_name: str,
                   headers: list[str], rows: list[list[Any]]) -> None:
    """Create one readable worksheet while preserving source values as data."""
    if not rows or not headers:
        raise ValueError("A workbook needs at least one header and one data row.")
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = _clean_plan(WorkbookPlan(title=title[:120], sheet_name=sheet_name[:31])).sheet_name
    worksheet.sheet_view.showGridLines = False
    worksheet.freeze_panes = "A5"
    worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(headers), 2))
    title_cell = worksheet.cell(row=1, column=1, value=title)
    _safe_cell(title_cell, title)
    title_cell.font = Font(name="Aptos Display", size=18, bold=True, color="F8FAFC")
    title_cell.fill = PatternFill("solid", fgColor="101827")
    title_cell.alignment = Alignment(vertical="center")
    worksheet.row_dimensions[1].height = 34
    worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(len(headers), 2))
    subtitle = worksheet.cell(row=2, column=1, value=f"{len(rows):,} data rows · {len(headers)} columns · Excel Maker")
    subtitle.font = Font(name="Aptos", size=10, color="526174")
    worksheet.row_dimensions[2].height = 22

    header_row = 4
    for column, header in enumerate(headers, start=1):
        cell = worksheet.cell(row=header_row, column=column)
        _safe_cell(cell, header)
        cell.font = Font(name="Aptos", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="315A82")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    worksheet.row_dimensions[header_row].height = 25

    for row_number, values in enumerate(rows, start=header_row + 1):
        for column, value in enumerate(values, start=1):
            cell = worksheet.cell(row=row_number, column=column)
            _safe_cell(cell, value)
            if isinstance(value, datetime):
                cell.number_format = "yyyy-mm-dd hh:mm:ss"
            elif isinstance(value, date):
                cell.number_format = "yyyy-mm-dd"
            elif isinstance(value, float):
                cell.number_format = "0.00########"
            cell.alignment = Alignment(vertical="top")

    last_column = len(headers)
    last_row = header_row + len(rows)
    table_ref = f"A{header_row}:{worksheet.cell(row=last_row, column=last_column).coordinate}"
    table = Table(displayName="ChatData", ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False,
                                         showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    worksheet.add_table(table)
    for column, header in enumerate(headers, start=1):
        sample = [str(header)] + [str(row[column - 1] if row[column - 1] is not None else "")
                                  for row in rows[:250]]
        width = min(max(max(len(value) for value in sample) + 2, 12), 42)
        worksheet.column_dimensions[get_column_letter(column)].width = width
    workbook.properties.title = title
    workbook.properties.creator = "LabCD AgentSysID · Excel Maker"
    workbook.save(path)


def _safe_filename(title: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_-")[:70]
    return stem or "chat_data_export"


def create_workbook(chat: dict, question: str, *, client=None, output_dir: str | Path | None = None) -> dict:
    """Use exact conversation data to create an Excel workbook and chat artifact."""
    headers, rows = _source_table(chat)
    plan, llm, plan_error = _make_plan(question, headers, len(rows), client=client)
    folder = Path(output_dir) if output_dir is not None else Path(core.UPLOAD_DIR) / chat["id"] / "generated"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{_safe_filename(plan.title)}_{uuid.uuid4().hex[:8]}.xlsx"
    handle, temporary_name = tempfile.mkstemp(prefix=".excel-maker-", suffix=".xlsx", dir=folder)
    os.close(handle)
    try:
        build_workbook(temporary_name, plan.title, plan.sheet_name, headers, rows)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    artifact = {"path": str(destination.resolve()), "name": destination.name, "title": plan.title,
                "sheet_name": plan.sheet_name, "rows": len(rows), "columns": len(headers),
                "sha256": digest}
    message = (f"Excel Maker created **{destination.name}** with {len(rows):,} data rows and "
               f"{len(headers)} columns. The workbook keeps your supplied table values and includes filters, "
               f"a frozen header, and a readable worksheet layout.")
    if plan_error:
        message += " I used a standard title because the configured AI service could not suggest workbook labels."
    return {"answer": message, "generated_artifact": artifact, "evidence": [],
            "model": getattr(getattr(llm, "settings", None), "model", "local workbook builder")}


class ExcelMakerJob:
    """Background chat job for Excel Maker, compatible with the conversation activity UI."""
    def __init__(self, chat: dict, question: str):
        self.snapshot = copy.deepcopy(chat)
        self.question = question
        self.purpose = "excel_maker"
        self.answer = None
        self.error = None
        self.error_detail = None
        self.phase = "Excel Maker · reading your table"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            self.answer = create_workbook(self.snapshot, self.question)
        except Exception as exc:
            self.error_detail = redact(f"{type(exc).__name__}: {exc}")[:800]
            self.error = str(exc) if isinstance(exc, ValueError) else "Excel Maker couldn't finish creating this workbook. Check the table and try again."
