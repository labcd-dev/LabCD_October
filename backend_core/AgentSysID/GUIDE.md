# AgentSysID – Product & UX Guide

## Purpose

AgentSysID automates the discovery of neural dynamics models
(MLP or LSTM) that predict \(\dot{x} = f(x, u)\) from trajectory data.
A multi-agent loop (Initializer → Data Inspector → Critic / Actor / Explorer → Report)
searches architecture and regularisation hyper-parameters while respecting
inference-latency and overfit constraints.

## Inputs

| Item | Requirement |
|------|-------------|
| Trajectory file | CSV or Excel with columns `time`, `s_*` (states), `a_*` (actions). Optional `xdot_*`. |
| Customer description | Free-text context (`CUSTOMER_SYSTEM_DESCRIPTION` / env). |
| Run mode | `fast` (≈5 cycles), `regular` (≈15), `heavy` (≈40). |

## Human-in-the-Loop (Data Inspector)

When `--interactive` (CLI) or the Streamlit checkbox is set, the Data Inspector
may ask the engineer to drop columns or confirm anomalies.  
In CI / headless environments leave interactive mode **off**.

## Outputs

- Best PyTorch weights (`.pth`)
- PDF engineering report (via `reporting/report.py`; migrate to `labcd_pdfmaker` when available)
- Delivery ZIP containing report + weights
- Agent interaction log under `logs/`

## Success criteria

- Validation MSE at or below `MSE_TARGET` (default 5e-5) **or** best effort after max cycles
- Inference latency ≤ `CUSTOMER_MAX_LATENCY_MS`
- Overfit ratio (val/train) < `OVERFIT_RATIO_LIMIT`

## Integration

- **CLI:** `PYTHONPATH=. python -m backend_core.AgentSysID.run_cli --data path.csv`
- **Streamlit:** `python frontend_streamlit/run_agent_sysid_ui.py` (port 8504)
- **HTTP API:** `PYTHONPATH=. uvicorn backend_api.AgentSysID.app:app --port 8006`
  - `POST /api/sysid/jobs`
  - `GET  /api/sysid/jobs/{id}`
  - `GET  /api/sysid/jobs/{id}/results`
  - `POST /api/sysid/jobs/{id}/cancel`

## Migration notes

1. Prompts live under `prompts/*.yaml` – prefer loading them via a shared `PromptLibrary` once `labcd_agents` is installed.
2. PDF generation currently uses `fpdf2`. Replace with `packages/labcd_pdfmaker` following the AgentAdaptive pattern.
3. A full LangGraph `StateGraph` can be wired in `graph/workflow.py` without changing agent class contracts.
