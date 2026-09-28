# Diagnostic reference for this system identification implementation

This is engineering guidance, not evidence that a particular failure occurred.
Prefer recorded run settings, actual prompts/replies and measured metrics over
current templates or generic heuristics. Never certify physical or closed-loop
stability from a neural network's score or a finite set of trajectories.

## What the system actually learns

The network predicts state derivatives xdot = f(x, u, dt). States, inputs and
derivatives are normalized for gradients; reported validation MSE/RMSE are in
physical derivative units. Multi-state averages mix units/scales. Inspect
state-wise errors, scaling and operational tolerances before calling an MSE
large, small or acceptable. A model that fits derivatives well can still drift
in integrated state rollout because errors accumulate and the trajectory leaves
the training distribution. Distinguish derivative RMSE, state-rollout RMSE,
normalized optimization loss and the composite score.

The current held-out plotting implementation evaluates the first test trajectory
in chunks targeted at 10 seconds. Each chunk starts from the measured state;
LSTM memory is warmed with preceding measured history, then uses predicted states
within the chunk. Aggregate state errors therefore summarize these initialized
chunks, not an uninterrupted full-length simulation or every test trajectory.
Read a new run's verification protocol and captured plotting implementation.
For older runs, this current reference cannot establish the historical protocol.

## Training measurements and checkpoints

Early stopping chooses the checkpoint by normalized validation loss. The trainer
returns the minimum training physical MSE across epochs, which need not be from
the selected checkpoint. Thus val/train ratio is a screening statistic, not a
same-checkpoint generalization proof. The loop can replace severely overfit
cycle metrics with sentinel MSE=9999 and RMSE=99; these are rejection penalties,
not genuine loss measurements. Check implementation provenance for older runs.
The best exported candidate is selected by validation MSE, not by latency or
final composite score. Final deployment latency can therefore exceed the
customer limit even after Critic/Actor tried to reduce latency.

## Score interpretation

The implementation awards up to 75 points from exp(-0.5 * val_mse / difficulty)
and 25 from exp(-mean_absolute_rollout_drift / difficulty). Difficulty is an
allowance of 1 through 5. This heuristic can produce a high score even if a
very strict engineering MSE target was missed. It is not a probability, R²,
percentage accuracy or stability certificate. An empty rollout currently uses
drift=0 in the scoring reference, so absence of verification samples must be
checked separately. Run status 'completed' refers to pipeline execution;
model_status refers to the heuristic score category. Neither establishes safe
controller behavior.

## Differential diagnosis

- Data and time: missing/invalid columns, NaNs, constant or redundant signals,
  duplicate/nonmonotonic timestamps, units, irregular dt, wrong input-state
  alignment, noisy numerical derivatives, wrapped angles and false resets.
  Filtering can suppress fast dynamics; stronger smoothing is not always better.
- Identifiability: insufficient excitation, narrow operating coverage, correlated
  inputs, hidden states, hysteresis, delayed effects and confounded experiments.
  No architecture can identify dynamics that the observations do not constrain.
  Variance alone does not establish persistent excitation or observability.
- Validation: hold out whole independent trajectories when possible, preserve
  time ordering for LSTM and rollout, check neighboring-time dependence and
  train/test distribution shifts. Leakage is a hypothesis until evidenced.
- Representation: MLP for effectively Markov state; LSTM may help missing
  memory but needs long contiguous sequences. Required sequence/rollout windows
  can eliminate all valid samples; reduce window sizes or obtain longer runs.
- Optimization: learning-rate oscillation, saturation, inadequate epochs,
  excessive/insufficient regularization, search bounds, early stopping and
  normalization/checkpoint mismatches. A single MSE cannot distinguish underfit
  from noisy targets. Compare training/validation trajectories and learning curves.
- Rollout: integration choice and step size, horizon mismatch, compounding
  errors, extrapolation, physical constraint violations and unstable regions.
  Evaluate multiple horizons and initial conditions, state-wise normalized errors,
  residual autocorrelation, invariants and baseline models. Do not infer that an
  RK4 integrator guarantees stability or that PINN guarantees correct physics.
- PINN: confirm the analytical equation file is implemented, units match, known
  parameters are justified, residual targets match the learned derivative and
  the physics weight does not overwhelm data loss. Never execute the equation file.
- Runtime and deployment: exceptions, missing dependencies/weights, sequence and
  feature-order mismatches, dt and normalization compatibility, inference device,
  warm-up and actual target-device latency. Budget violations are measured facts
  only when both the limit and the measurement are recorded for that run.

## Audit the other agents rather than obeying them

Inspect actual system/user prompts and responses for: conflicting physical system
descriptions; hard bounds versus exploration directives; mandatory topology
mutation that disrupts an already good configuration; proposals outside bounds;
clipping/parsing/fallback effects; latency-overfitting trade-offs; insufficient
history; 'ASK_HUMAN' ignored in headless mode; and report claims stronger than
the measured evidence. A prompt conflict is a possible mechanism, not proof of
causation. Compare the proposed configuration with the next cycle's actual
configuration. Actor's clamping and novelty enforcement can change a proposal.

## Useful experiments and available controls

Recommend a small number of prioritized, controlled experiments, with expected
observations that would support or falsify each hypothesis. Preserve a baseline,
hold out independent trajectories, and change one factor at a time when feasible.
Use the actual SysIDOptions names: architecture, lstm_seq_length,
rollout_horizon, integrator_type, derivative_method, derivative_filter_tau,
savgol_window, savgol_polyorder, angle_indices, multi_trajectory,
trajectory_chunk_size, shuffle_data, epochs, early_stop_patience, batch_size,
mse_target, max_cycles, customer_max_latency_ms, use_pinn, pinn_loss_weight,
use_state_filter, auto_filter_percentiles, manual_starting_lr,
manual_starting_hidden_layers, manual_activation, manual_dropout_rate,
manual_weight_decay and initializer_overrides. Check exact available option
names in recorded settings before recommending a change. State when a proposed
measurement or algorithm change is not exposed in the current UI.

Do not invent per-epoch logs, gradients, test drift, physical equations, noise
levels, confidence percentages or completed experiments. Ask for the smallest
missing measurement needed to distinguish the leading alternatives. If the
evidence contradicts a 'failed model' premise, explain what succeeded and what
still failed or remains unverified.
