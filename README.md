# AgentSysID – Agentic System Identification

LabCD-style package: multi-agent neural system identification (MLP/LSTM) with
LangGraph-orchestrated Initializer → Data Inspector (HIL) → Critic / Actor /
Explorer tuning → automated PDF manuscript and deployment package.

| Piece | Status |
|-------|--------|
| `backend_core/AgentSysID/` | **Delivered** – CLI + agents + trainer + reporting |
| `backend_api/AgentSysID/` | Assignment only → see `ASSIGNMENT.md` |
| `frontend_streamlit/` | **Delivered** – Streamlit UI over the same pipeline |
| `frontend_mockup/` | HTML agent-harness mock (design target, aligned to the core) |

## Quick start (core)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY / GROQ_API_KEY

PYTHONPATH=. python -m backend_core.AgentSysID.run_cli \
  --data backend_core/AgentSysID/data/examples/synthetic_oscillator.csv --mode fast
```

The run works without an API key too: every agent falls back to its
deterministic mathematical path and still produces the full deliverable.

### The engineer is in the loop by default

Exactly as the original `main()` did, a normal run **asks you questions before
it trains**:

1. **Dataset questionnaire** — a free-text system description (fed to every
   agent as context), whether any states are angles that wrap at ±π, and
   whether the file is one continuous run or several stacked trajectories.
2. **Data Inspector (HIL)** — if the agent finds a dead sensor or a duplicated
   state, it asks you, and your answer can physically drop those columns.
3. **Initializer review** — the proposed hyper-parameters and search bounds are
   printed, and you may override any of the 14 of them. Press Enter to accept
   the AI's choice for any single parameter.

Pass `--headless` when nobody is at the terminal. A run also detects a
non-TTY stdin (pipes, cron, workers) and switches to headless automatically
rather than hanging.

### CLI options

| Flag | Meaning |
|------|---------|
| `--data PATH` | CSV/Excel dataset (see `data/DATA_CONTRACT.md`) |
| `--mode fast\|regular\|heavy` | Cycle / time / memory budget (7·0.5h, 20·1.5h, 40·4h) |
| `--headless` / `--no-interactive` | Skip all prompts (for API / Streamlit / CI) |
| `--arch MLP\|LSTM` | Override the network architecture for this run |
| `--max-cycles N`, `--epochs N` | Override the tuning and training budgets |
| `--output-dir DIR` | Base artifact directory (default `artifacts_sysid`) |

Ctrl+C is a graceful stop: the current step finishes and the best checkpoint
so far is still packaged.

## The UI

```bash
PYTHONPATH=. python frontend_streamlit/run_agent_sysid_ui.py     # → http://localhost:8504
# or
PYTHONPATH=. streamlit run frontend_streamlit/agent_sysid_app.py --server.port 8504
```

The Streamlit app is the browser equivalent of the terminal run. It exposes
**every** option the CLI asks for — the three questionnaire answers, all
fourteen Initializer overrides (blank = keep the agent's choice), the
architecture / rollout / integrator settings, the derivative estimator, the
state-space filter, PINN, the training limits, the client-authorised search
bounds, the manual presets and the LLM provider — then streams the live stage,
progress, per-cycle metrics, convergence and latency charts, and the raw agent
log, and finally offers the ZIP, PDF, `.pth`, standalone controller, `NN.py`
and conversation log as downloads.

It never duplicates core logic: every control maps to a field on
`SysIDOptions` and the run is `run_pipeline` on a worker thread — the same
function the CLI calls.

`frontend_mockup/labcd_sysid.html` is the chat-style design target for the
React path; its stages, run modes, latency units and score now match the core.

## Outputs

One run writes one self-contained folder:

```
artifacts_sysid/run_<timestamp>_<env>/
├── llm_conversation_history.txt   # every prompt, response, token count
├── figures/                       # MSE, RMSE, hyperparameters, contour,
│                                  # latency, held-out verification rollout
├── deployment/                    # .pth weights, deployed_controller_<env>.py, NN.py
├── report/                        # 9-section PDF engineering manuscript
├── Agents_log/
└── SystemID_RunResults_<timestamp>.zip
```

`deployed_controller_<env>.py` is framework-free: it hardcodes the winning
topology and the normalisation buffers and needs only torch and numpy.

## Layout

```
backend_core/AgentSysID/   # core package
├── run_cli.py             # argv wrapper around pipeline.run_pipeline
├── questionnaire.py       # interactive dataset questions
├── config.py              # every knob, env-overridable, mutated live by agents
├── agents/                # inspector, initializer, critic, actor, explorer, report
├── prompts/               # agent prompts as YAML (tune without touching Python)
├── data/                  # loader (derivatives, complexity, quality) + splitter
├── model/                 # DynamicsModel (MLP/LSTM) + PINN hook
├── training/              # trainer, best-config tracker, success scoring
├── reporting/             # plots, verification rollout, PDF, export, ZIP
├── graph/                 # LangGraph wiring of the same pipeline
└── tests/
├── pipeline.py            # SysIDOptions + run_pipeline — the one implementation
backend_api/AgentSysID/    # ASSIGNMENT.md only
frontend_streamlit/        # agent_sysid_app.py + run_agent_sysid_ui.py
frontend_mockup/           # HTML agent-harness mock (design target)
_legacy/                   # original flat sources (reference)
```

## Docs

- `backend_core/AgentSysID/GUIDE.md` – pipeline, agent contract, product rules
- `backend_core/AgentSysID/data/DATA_CONTRACT.md` – dataset columns

## Tests

```bash
PYTHONPATH=. pytest backend_core/AgentSysID/tests -v
```

120 tests, no API key or GPU required. They cover the derivative estimators,
the rollout trainer, every agent's parsing *and* fallback path, the PDF, the
exported controller, the interactive prompts, and UI/pipeline option parity.
