# ASSIGNMENT – AgentSysID Streamlit UI

**Audience:** developer owning the Streamlit reference path (optional vs React).  
**Pattern:** match `frontend_streamlit/agent_mpc_app.py` style in LabCD_NewModules.

## Deliver

`agent_sysid_app.py` (+ optional `run_agent_sysid_ui.py` launcher, port e.g. 8504):

1. File upload (CSV/Excel) — contract in `backend_core/AgentSysID/data/DATA_CONTRACT.md`
2. Knobs: run mode (fast/regular/heavy), architecture (MLP/LSTM), PINN toggle, max latency
3. Start → run `backend_core.AgentSysID` pipeline (background / status)
4. Live stage/progress + agent log snippet
5. Downloads: PDF, ZIP, `.pth`

## Rules

- `PYTHONPATH=.`; import from `backend_core.AgentSysID` only — no duplicated training logic.
- Headless-safe (HIL inspector off by default).
- Keep UI thin; core owns the loop.

## Reference

- Core CLI: `python -m backend_core.AgentSysID.run_cli --data …`
- Visual target: `frontend_mockup/` HTML assignment
