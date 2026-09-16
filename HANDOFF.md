# Handoff – AgentSysID

**For:** backend API + React / Streamlit / mockup owners  
**Status:** Python **core** delivered under `backend_core/AgentSysID/`. API, Streamlit, and HTML mock are **assignments only** (no implementation in this zip).

## What you receive

| Path | Role |
|------|------|
| `backend_core/AgentSysID/` | Domain logic (agents, trainer, model, data loader, CLI) |
| `backend_core/AgentSysID/GUIDE.md` | Product rules, success criteria |
| `backend_core/AgentSysID/data/DATA_CONTRACT.md` | CSV/Excel column contract |
| `backend_api/AgentSysID/ASSIGNMENT.md` | FastAPI job-adapter brief |
| `frontend_streamlit/ASSIGNMENT.md` | Streamlit reference-UI brief |
| `frontend_mockup/ASSIGNMENT.md` | Static HTML agent-harness mock brief |
| `_legacy/` | Original monolithic sources (reference only) |

## Quick start (core only)

```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY or GROQ_API_KEY

PYTHONPATH=. python -m backend_core.AgentSysID.run_cli \
  --data backend_core/AgentSysID/data/examples/synthetic_oscillator.csv --mode fast
```

## Ownership split

1. **Core (done)** – do not reimplement training/agents in TS or the API layer.
2. **API** – implement per `backend_api/AgentSysID/ASSIGNMENT.md` (job endpoints, same shape as AgentMPC).
3. **Streamlit (optional)** – per `frontend_streamlit/ASSIGNMENT.md`.
4. **HTML mock** – per `frontend_mockup/ASSIGNMENT.md` (design / harness UX target for React).
5. **React** – consume API; use mock + Streamlit assignment as behavioural specs.
