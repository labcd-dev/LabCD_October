# AgentSysID – Product & Engineering Guide

Multi-agent system identification: given logged trajectories of a physical
plant, discover the neural architecture that best predicts its state
derivatives (`x_dot`) under a hard real-time latency budget, then ship the
weights, a standalone controller, a PDF report and a ZIP package.

This guide is the contract every adapter (API, Streamlit, mockup) builds on.
The original flat implementation is preserved under `_legacy/` for reference;
this package is that logic, reorganised.

---

## 1. Pipeline

```
questionnaire ─► ExcelDataLoader ─► Data Inspector (HIL) ─► Initializer
                                                              │
                        ┌─────────────────────────────────────┘
                        ▼
                   Actor ─► train ─► Critic ─► (stagnation?) ─► Explorer
                        ▲                                          │
                        └──────────────────────────────────────────┘
                        │
                        ▼
      held-out rollout verification ─► success score ─► Report Agent ─► PDF ─► ZIP
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Questionnaire | `questionnaire.py` | Angular states, trajectory structure, customer context, Initializer override review |
| Loading | `data/loader.py` | Derivatives, complexity tier, quality audit, trajectory split |
| Splitting | `data/splitter.py` | Hybrid train / val / chronological test |
| Inspection | `agents/data_inspector.py` | HIL anomaly review, physical column drops |
| Initialization | `agents/initializer.py` | Starting config **and** the authorised search bounds |
| Training | `training/trainer.py` | Rollout training, PINN residual, early stop, overfit abort |
| Criticism | `agents/critic.py` | Diagnosis, LR direction, next topology |
| Action | `agents/actor.py` | Bounded, non-repeating mutation |
| Exploration | `agents/explorer.py` | Topological inversion to escape local minima |
| Reporting | `reporting/` | Plots, verification rollout, PDF, deployable export, ZIP |
| Orchestration | `pipeline.py` | `SysIDOptions` + `run_pipeline` — the one implementation |

---

## 2. Run modes

| Mode | Cycles | Time limit | Critic explores until | Failure memory | Customer context |
|------|--------|-----------|----------------------|----------------|------------------|
| `fast` | 7 | 0.5 h | cycle 3 | last 5 | never |
| `regular` | 20 | 1.5 h | cycle 8 | last 10 | cycles 1-5 |
| `heavy` | 40 | 4.0 h | cycle 15 | unlimited | always |

Both limits are enforced: the loop stops at whichever arrives first, and always
compiles the best checkpoint found so far. Ctrl+C is a graceful stop, not an
abort — the current step finishes and the full report is still produced.

---

## 3. The agent contract

Agents speak a strict `KEY: VALUE` line protocol, not free prose, and every
prompt lives in `prompts/*.yaml` so it can be tuned without touching Python.

**Every agent has a deterministic fallback.** If the API is unreachable, rate
limited, or returns something unparseable, the run continues on mathematics:

- **Critic** applies the Master Overfit Matrix directly (ratio thresholds
  4.0 / 7.0 / 10.0, plus a latency-violation branch).
- **Actor** falls back to incremental topology stepping toward the Critic's
  target, then to bounded random exploration.
- **Explorer** forces a mathematical inversion via bounds reflection.
- **Initializer** and **Report Agent** use fully-populated defaults.

A run with no API key still produces a complete, valid deliverable.

### Bounds are a hard contract

The Initializer proposes a search space; it is clamped against the
client-authorised outer limits in `config.py` before anything uses it. The
Actor and Explorer then clamp every candidate against *that* space. A
hallucinated learning rate cannot escape.

### No configuration is ever tested twice

The Actor keys visited configs on `(learning_rate bucketed by LR_TOLERANCE,
hidden_layers, dropout)`. If a proposal repeats, it mutates again.

---

## 4. Data contract

See `data/DATA_CONTRACT.md`. In short: a `time` column, `s_*` states, `a_*`
actions, optional `xdot_*` true derivatives.

**Provide `xdot_*` when you can.** Otherwise derivatives are estimated, which
amplifies sensor noise. Three estimators are available via
`DERIVATIVE_METHOD`:

| Method | Use it when |
|--------|-------------|
| `finite_difference` (default) | Clean data; pair with `DERIVATIVE_FILTER_TAU` for a Simulink-style low-pass |
| `savitzky_golay` | Noisy data with smooth underlying dynamics |
| `sliding_mode` | Levant's robust exact differentiator; noisy data with sharp transitions |

Angular states that wrap at ±π must be declared (`ANGLE_INDICES`) or
auto-detected (`AUTO_DETECT_ANGLES`), otherwise the wrap registers as an
enormous false derivative.

---

## 5. Architecture & loss

- `NETWORK_ARCHITECTURE`: `MLP` (memoryless) or `LSTM` (memory window of
  `LSTM_SEQ_LENGTH` steps).
- `ROLLOUT_HORIZON > 1` trains autoregressively: the network is fed its own
  predictions and accumulates drift penalty.
- Gradients are computed on **normalised** loss; MSE/RMSE are reported in
  **physical** units. Never compare the two.
- `USE_PINN` adds a physics residual weighted by `PINN_LOSS_WEIGHT`, taken from
  `compute_analytical_xdot` in `PINN_EQUATION_FILE`. A template is generated on
  first run and the run pauses so the engineer can fill in the equations.

**Shuffling is forced off for LSTM** so temporal memory stays contiguous.

---

## 6. Acceptance

**A validation split shorter than the rollout window is a hard error.** Every
validation metric would otherwise average 0/0 and the run would report a
perfect MSE of 0.0 from no data, which the Critic would then tune against. The
trainer raises instead, naming the settings to change.

A cycle is rejected outright when the validation loss exceeds
`OVERFIT_RATIO_LIMIT` (default 10x) times the training loss — it is assigned a
penalty MSE so it can never win.

The final model is graded 0-100 by `training/scoring.py`:

- **Pillar A (75 pts)** – single-step fit from validation MSE
- **Pillar B (25 pts)** – closed-loop rollout drift on held-out data

Both are divided by a difficulty allowance derived from the dataset complexity
tier, so the same raw error scores higher on a harder plant.

| Score | Status |
|-------|--------|
| > 75 | STABLE & HIGH-FIDELITY |
| > 50 | STABLE & ACCEPTABLE |
| ≥ 40 | UNSTABLE ROLLOUT |
| < 40 | UNSTABLE / FAILED |

Latency is a hard customer constraint (`CUSTOMER_MAX_LATENCY_MS`), measured on
a real forward pass, not estimated from parameter count.

---

## 7. Outputs

One run writes one self-contained folder:

```
artifacts_sysid/run_<timestamp>_<env>/
├── llm_conversation_history.txt   # every prompt, response and token count
├── figures/                       # 5 diagnostics + held-out verification
├── deployment/                    # .pth, deployed_controller_<env>.py, NN.py
├── report/                        # 9-section PDF manuscript
├── Agents_log/
├── run_manifest.json              # JSON summary; what a UI lists as history
└── SystemID_RunResults_<timestamp>.zip
```

`run_manifest.json` is written at the run root, which the ZIP packager does not
collect — the delivered archive is unchanged. `SysIDResult.save_manifest()`
writes it and `SysIDResult.load_manifest()` reads one back.

`deployed_controller_<env>.py` has **no dependency on this framework** — it
hardcodes the topology and normalisation buffers and needs only torch and
numpy. That is the artefact you hand to the control team.

---

## 8. Integration rules

- Import from `backend_core.AgentSysID` only; never duplicate training logic.
- Run with `PYTHONPATH=.` from the repository root.
- **The CLI is interactive by default**, matching the original `main()`: the
  dataset questionnaire, the HIL Data Inspector and the Initializer config
  review all run unless told otherwise. Adapters must pass `--headless`
  (or `interactive=False` when calling the functions directly), because those
  prompts block on `input()`. A non-TTY stdin is detected and downgraded to
  headless automatically, so a cron job cannot hang.
- Configuration is read live off the module (`config.X`), so overrides applied
  before a stage takes effect in it. Set them before calling, not after.
- **Drive the pipeline through `pipeline.run_pipeline`**, never by copying the
  loop. `SysIDOptions` carries every engineer decision as data; a field left as
  `None` keeps whatever `config.py` holds. `on_event(kind, payload)` streams
  `stage`, `cycle`, `latency`, `critic`, `explorer`, `progress`, `done` and
  `error` for progress UIs. `run_cli.main` and the Streamlit app are both thin
  callers of it, and `graph/workflow.py` exposes the same stages as a LangGraph
  `StateGraph`.
- Pass `install_signal_handler=False` when calling from a worker thread — only
  the main thread may install a SIGINT handler. Use `utils.request_stop()` to
  stop such a run gracefully.
- A long-lived server must not read `config.X` for form defaults: a run mutates
  the config module in place, so snapshot the pristine values at import (the
  Streamlit app does this) or the form will drift to the last run's values.

---

## 9. Extending

**A new agent:** add `prompts/<name>.yaml` with `system` and `user_template`
blocks, a class in `agents/`, and export it from `agents/__init__.py`. Use
`invoke_llm` so cost tracking and conversation logging come for free, and
always write a deterministic fallback.

**A new derivative estimator:** add a branch in
`ExcelDataLoader._attach_dt_and_format` and a case in the parametrised test in
`tests/test_loader.py` that recovers a known analytical derivative.

**A new architecture:** extend `model/dynamics_model.py` and the rollout
branch in `training/trainer.py`, then mirror it in
`reporting/export.py` so the deployable controller stays in sync.

---

## 10. Tests

```bash
PYTHONPATH=. pytest backend_core/AgentSysID/tests -v
```

The suite runs without an API key or a GPU. It covers the derivative
estimators against known analytical derivatives, trajectory splitting, rollout
windowing, the PINN hook, every agent's parsing *and* fallback path, the
scoring pillars, the PDF's nine sections, and a subprocess check that the
exported controller runs standalone.

---

## 11. Ask about a run

In Streamlit, open **Ask run**, or choose a run in history and click
**Ask about this run** in Results. Questions and answers are saved separately
for each run in `run_chat.json`. The results preview can remain open beside
the conversation. Quick questions, model settings, evidence details, export
and clear controls are available in the chat workspace.

`agents/run_diagnostic.py` implements a dedicated `RunDiagnosticAgent` with
two model calls: an investigator audits the run, then an evidence reviewer
challenges the proposed diagnosis. Findings identify recorded observations or
hypotheses, show confidence, and include quotations whose source IDs and exact
text are checked locally. Suggested experiments explain what outcome would
support or falsify a hypothesis. Failed citation locations are sent to the
reviewer; if its answer still fails local format/citation checks, one repair
attempt is allowed. Only source text is accepted, including when harmless
formatting or added sentence-ending punctuation is recovered. A failed review
keeps a validated first pass as an unreviewed draft. Otherwise local checks are
shown with the specific cause, distinguishing rejected answers from API failures.

The agent reads measured results, training cycles, recorded agent prompts and
replies, runtime errors, the activity feed, and a system identification reference
covering excitation, temporal splits, derivative error versus rollout drift,
MLP/LSTM, optimization, physics constraints, and deployment. Sources are bounded
and selected for relevance; omissions and unavailable evidence are disclosed.
It treats other agents' prompts as evidence to audit, never as instructions to
obey. It does not execute generated code, retrain, or change a run's settings.

New pipeline runs save `run_context.json` with original options, initial and
effective configuration, six agent prompt templates, relevant implementation
excerpts, and data-split details. `verification_summary.json` records held-out
state errors separately from derivative MSE, with the recorded chunk horizon,
initialization/warm-up, selected test trajectory, and integration method.
Streamlit also saves a bounded
`runtime_console.log` and `diagnostic_state.json`, including failures. Preflight
failures remain visible in history. Older runs use their actual recorded agent
turns, with today's templates/code explicitly marked as unverified historical
references. Plot pixels and model weights are not analyzed by the text agent;
neither its diagnosis nor the composite score certifies physical stability.

Diagnostics follow the current `config.API_PROVIDER` and `config.LLM_MODEL`
used by the other agents (currently OpenAI / `gpt-4o-mini`), with a separate
client that does not change training settings or their cost tracker. Each
question normally makes two model calls, with at most one additional repair
call for format/citation failures. Calls are billed by the provider using the corresponding existing environment credential
(`OPENAI_API_KEY`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY`). The Diagnostic model
popover shows the current configuration. Provider/model changes for the other
agents automatically apply to new diagnostic questions. Configure the main API
through the existing controls or environment variables:

```text
LABCD_SYSID_API_PROVIDER=openai
LABCD_SYSID_LLM_MODEL=gpt-4o-mini
```

OpenAI requests use `store=False`. Common credentials are redacted from evidence
and saved diagnostic files. Model calls run on a background worker so the UI
remains usable. Diagnose a training run after it finishes, or investigate an
earlier run while training continues.

The offline diagnostic tests cover citation rejection, historical provenance,
prompt handling, bounded retrieval, failure labels, verification metrics, and
run-scoped conversations. Streamlit AppTests also exercise run switching,
submission, clearing, current-run gating, and the Results entry point.
