# AgentSysID Streamlit UI — DELIVERED

**Status:** implemented. This file now documents what was built and how to
extend it; the original assignment brief is preserved at the bottom.

## Run it

```bash
PYTHONPATH=. python frontend_streamlit/run_agent_sysid_ui.py        # → http://localhost:8504
PYTHONPATH=. python frontend_streamlit/run_agent_sysid_ui.py --port 8600 --headless

# or directly
PYTHONPATH=. streamlit run frontend_streamlit/agent_sysid_app.py --server.port 8504
```

Requires `streamlit` and `altair` (both in `requirements.txt`). No API key is
needed — every agent falls back to its deterministic mathematical path and the
run still produces the full deliverable.

## Files

| File | Purpose |
|------|---------|
| `agent_sysid_app.py` | The UI |
| `run_agent_sysid_ui.py` | Launcher (sets `PYTHONPATH`, port 8504, dark theme) |
| `../.streamlit/config.toml` | Theme tokens shared with the HTML mock |

## How it stays thin

The UI contains **no training, agent or reporting logic**. Every control maps
to a field on `backend_core.AgentSysID.pipeline.SysIDOptions`, and the run is
`run_pipeline(options, on_event=...)` on a worker thread — the same function
`run_cli.main` calls. Two tests in `backend_core/AgentSysID/tests/test_pipeline.py`
enforce this:

- `test_ui_exposes_every_option_field` — every `SysIDOptions` knob is reachable
  from the UI.
- `test_ui_never_runs_interactively_and_duplicates_no_core_logic` — the UI never
  blocks on `input()` and never imports the trainer or the tracker.

## What is exposed

Everything the interactive CLI asks for:

1. **Dataset** — upload (CSV/XLSX) or the bundled example.
2. **Run budget** — mode (fast / regular / heavy), cycle and epoch overrides,
   output directory.
3. **Questionnaire** — system description, angular states (none / manual /
   auto-detect), trajectory structure (single / stacked, auto or manual split
   timestamps).
4. **Architecture & rollout** — MLP or LSTM, memory window, rollout horizon,
   integrator, chunk size, shuffle.
5. **Derivative estimation** — finite difference / Savitzky-Golay / sliding
   mode, filter tau, window and polyorder, λ1 and λ2, reset threshold.
6. **State-space filter** — on/off and the percentile band.
7. **PINN** — toggle, loss weight, equation file.
8. **Training limits** — batch, patience, MSE target, overfit limit, max
   latency, adaptive regularization, LR schedule, epoch extension.
9. **Search bounds** — the client-authorised LR / width / depth limits.
10. **Initializer** — agent on/off, manual presets, client-locked parameters
    (JSON), and **all fourteen overrides** (blank = keep the agent's choice).
11. **LLM provider** — provider, model, temperature.

## What it shows

- Live stage tracker across the eight pipeline stages, plus a progress bar.
- Per-cycle metrics, a train-vs-validation convergence chart (log scale) and a
  latency chart against the customer limit. The two series colors are validated
  for the dark surface (lightness band, chroma floor, CVD separation,
  normal-vision floor, contrast) — see the `dataviz` colour checks.
- The raw agent log, streamed from the core's stdout.
- Critic decisions and a per-cycle table.
- Final score, deployment status and the complete artefact set as downloads:
  ZIP, PDF, `.pth`, standalone controller, `NN.py`, conversation log — plus the
  diagnostic figures and the held-out verification plot inline.

## Extending it

Add a knob by adding the field to `SysIDOptions` (with `None` as the default so
it leaves `config.py` alone), mapping it in `apply_to_config`, then adding the
widget here. The parity test will fail until the widget exists.

Do **not** read `cfg.X` for a widget default: a run mutates the config module in
place, so a long-lived server would drift to the previous run's values. Use
`D("NAME")`, which reads the pristine snapshot taken at import.

---

## Original assignment brief

**Audience:** developer owning the Streamlit reference path (optional vs React).
**Pattern:** match `frontend_streamlit/agent_mpc_app.py` style in LabCD_NewModules.

### Deliver

`agent_sysid_app.py` (+ optional `run_agent_sysid_ui.py` launcher, port e.g. 8504):

1. File upload (CSV/Excel) — contract in `backend_core/AgentSysID/data/DATA_CONTRACT.md`
2. Knobs: run mode (fast/regular/heavy), architecture (MLP/LSTM), PINN toggle, max latency
3. Start → run `backend_core.AgentSysID` pipeline (background / status)
4. Live stage/progress + agent log snippet
5. Downloads: PDF, ZIP, `.pth`

### Rules

- `PYTHONPATH=.`; import from `backend_core.AgentSysID` only — no duplicated training logic.
- Headless-safe (HIL inspector off by default).
- Keep UI thin; core owns the loop.

### Reference

- Core CLI: `python -m backend_core.AgentSysID.run_cli --data …`
- Visual target: `frontend_mockup/` HTML assignment
