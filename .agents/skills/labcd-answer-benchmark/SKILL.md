---
name: labcd-answer-benchmark
description: Run and analyze the internal LabCD answer benchmark for question routing, evidence retrieval, setup edits, and optional live answer quality. Use when changing or reviewing LabCD chat answers, setup conversation, run diagnostics, or evidence selection.
---

# LabCD answer benchmark

Use the repository's synthetic benchmark to check LabCD's question-handling structure. Do not use client chats, client files, ratings, or feedback as benchmark data.

## Default evaluation

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --mode offline
```

The offline pass runs the full authored question bank. It checks the actual question router and evidence selector, tests the deterministic chat-to-settings path against the settings schema, and checks local replies that should not call a model. It uses only synthetic setup and run fixtures and makes no model/API requests.

Read the generated `report.md` first, then use `report.json` for exact case IDs and measured fields. Analyze failures by type: route, retrieval/scope, setting dispatch/value, or deterministic reply. A single overall score can hide a broken category, so report per-category coverage and failures.

## Live answer evaluation

Use live evaluation only when the user explicitly asks to measure actual generated answers. It consumes the configured model provider's tokens. The runner requires an explicit limit and samples across categories:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --mode live --limit 20
```

Use `--list-categories` to see categories and `--category <name>` to focus a smaller run. Planner setting edits are tested locally because their parser and schema are deterministic. Live checks report model-generated answers separately from local replies. Provider outages and unsupported-region errors are marked blocked, not as answer-quality failures. Existing reports can be re-scored without provider calls:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --reclassify-report artifacts/labcd_answer_benchmark/run_<timestamp>/report.json
```

Live answer checks include returned status, route, evidence IDs, selected evidence scope, answer length warnings, synthetic per-state ranking, and case-specific expected facts. Inspect each failure before changing prompts or routing.

## Maintenance

- Add questions to `frontend_streamlit/answer_benchmark/cases.json`; keep the fixture synthetic and avoid domain-specific assumptions unless a case explicitly tests them.
- Each group declares a category, scenario, expected route, and required/forbidden evidence kinds. Setting edit entries also declare the exact expected schema fields and values.
- If the product's intended behavior changes, update the benchmark expectation only after confirming the intended behavior from the user-facing code and product design.
- Run the offline benchmark after changes to the benchmark or answer routing. Use live mode on a small, stratified sample after changes to answer prompts or answer composition.
- Do not make prompt edits solely to improve lexical benchmark scores; verify the underlying route, evidence, and user-facing behavior first.
