# AgentSysID – Agentic System Identification

LabCD-style package: multi-agent neural system identification (MLP/LSTM).

| Piece | Status |
|-------|--------|
| `backend_core/AgentSysID/` | **Delivered** – CLI + agents + trainer |
| `backend_api/AgentSysID/` | Assignment only → see `ASSIGNMENT.md` |
| `frontend_streamlit/` | Assignment only → see `ASSIGNMENT.md` |
| `frontend_mockup/` | Assignment only → HTML agent-harness mock |

## Quick start (core)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

PYTHONPATH=. python -m backend_core.AgentSysID.run_cli \
  --data backend_core/AgentSysID/data/examples/synthetic_oscillator.csv --mode fast
```

## Layout

```
backend_core/AgentSysID/   # core package (run_cli, agents, model, training, …)
backend_api/AgentSysID/    # ASSIGNMENT.md only
frontend_streamlit/        # ASSIGNMENT.md only
frontend_mockup/           # ASSIGNMENT.md only
_legacy/                   # original flat sources
```

## Docs

- `backend_core/AgentSysID/GUIDE.md` – product rules  
- `backend_core/AgentSysID/data/DATA_CONTRACT.md` – dataset columns  

## Tests

```bash
PYTHONPATH=. pytest backend_core/AgentSysID/tests -v
```
