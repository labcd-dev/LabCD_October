# ASSIGNMENT – AgentSysID FastAPI adapter

**Audience:** backend / full-stack developer  
**Pattern:** match `backend_api/AgentMPC/` (or AgentPlant) in LabCD_NewModules.

## Deliver

Thin job-oriented HTTP API over `backend_core.AgentSysID`:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/sysid/jobs` | Start job (`data_path`, `run_mode`, `architecture`, `use_pinn`, …) |
| GET | `/api/sysid/jobs` | List jobs |
| GET | `/api/sysid/jobs/{id}` | Status / stage / progress |
| POST | `/api/sysid/jobs/{id}/cancel` | Cancel |
| GET | `/api/sysid/jobs/{id}/results` | Artefacts (PDF, `.pth`, ZIP paths) |
| GET | `/health` | Liveness |

## Rules

- No heavy logic in routers — call `backend_core.AgentSysID.run_cli` (or a thin service wrapper) in a background thread/worker.
- Prefer persistent job store (Redis/DB) over in-memory when integrating the monorepo.
- Pydantic request/response schemas; absolute imports from repo root; `PYTHONPATH=.`.
- Do **not** reimplement training or agents.

## Reference

- Core: `backend_core/AgentSysID/` + `GUIDE.md` + `data/DATA_CONTRACT.md`
- UI behaviour: `frontend_mockup/` (HTML mock) and this module’s contract above
