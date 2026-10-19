# LabCD answer benchmark

This is an internal QA harness for LabCD's question routes, evidence selection, settings edits, and generated answers. It uses only a synthetic measurement profile and a synthetic saved run.

## Run without model calls

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --mode offline
```

The default run executes every question in `cases.json`, checks the production route and evidence selectors, verifies settings edits against the production parser and schema, and exercises local answers that should not reach an AI provider. It makes no model/API calls.

## Ask LabCD to answer a small sample

Live answers use the configured provider and consume tokens. `--limit` is required:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --mode live --limit 20
```

The limit is stratified across question categories. For a focused run, add `--category <category-name>`. Use `--list-categories` to list the available names.

Each invocation writes `report.md` and `report.json` under a timestamped folder in `artifacts/labcd_answer_benchmark/`. The Markdown report summarizes category scores and failures. The JSON preserves each question, checked route, selected evidence IDs/types, live answer, elapsed time, and scoring reasons for follow-up analysis.

Provider outages or unsupported-region errors are reported as **blocked** cases, separate from answer-quality failures. To re-score a saved live report after changing the evaluator, without making more provider calls:

```powershell
.\.venv\Scripts\python.exe -m frontend_streamlit.answer_benchmark.runner --reclassify-report artifacts/labcd_answer_benchmark/run_<timestamp>/report.json
```
