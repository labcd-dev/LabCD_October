# AgentSysID core – handoff

Sequential multi-agent loop (Initializer → Data Inspector → Critic/Actor/Explorer → Report).  
LangGraph scaffold in `graph/` (not wired).

**Entry:** `python -m backend_core.AgentSysID.run_cli`  
**API / UI:** not in this package — see repo-root `backend_api/`, `frontend_streamlit/`, `frontend_mockup/` assignments.

Prompts: `prompts/*.yaml`. Config: `config.py` + `LABCD_SYSID_*` env.  
Monorepo: prefer `labcd_agents` + `labcd_pdfmaker` when available.
