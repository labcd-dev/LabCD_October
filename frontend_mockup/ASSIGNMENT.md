# ASSIGNMENT – AgentSysID HTML mockup (agent harness)

**Audience:** frontend / UX implementer  
**Goal:** one self-contained HTML mock that feels like a LabCD **agent harness** for system identification (same visual language as other LabCD mockups if they exist in the monorepo).

## Deliver

Single file (or small static set): e.g. `labcd_studio_sysid.html`

### Screens / panels (typical harness)

1. **Header** – module name “AgentSysID”, case/dataset label, cancel
2. **Data** – upload zone or “loaded plugin/dataset” card; show detected `n_states`, `n_inputs`, column hints (`time`, `s_*`, `a_*`)
3. **Pre-launch config** – run mode, architecture (MLP/LSTM), PINN on/off, max latency, optional customer description
4. **Launch** – primary CTA “Start identification”
5. **Live run** – stage timeline (Inspect → Initialize → Train cycle N → Critic/Actor/Explorer → Report), progress bar, latest agent message / log strip
6. **Results** – best MSE/RMSE, latency, architecture summary; buttons Download PDF / ZIP / weights

### Behaviour (mock only)

- No real backend required: fake progress with JS timers after “Start”
- Disable Launch while “running”; enable downloads when “complete”
- Keep layout clean, dense, engineering-tool aesthetic (not marketing landing page)

### Consistency

- Prefer the same CSS variables / card patterns as other files under monorepo `frontend_mockup/` when present
- Mobile optional; desktop-first is fine

## Out of scope

React, Streamlit, or calling the Python API — this is a **static behavioural mock** for design alignment and full-stack handoff.
