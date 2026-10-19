# LabCD answer benchmark

- Mode: **offline**
- Cases: **210** · passed **182** · failed **28**
- Pass rate: **86.7%**
- Dataset/run inputs: synthetic fixtures only

## Category results

| Category | Passed | Failed |
|---|---:|---:|
| `active_run_chat` | 10 / 10 | 0 |
| `dataset_specific_interpretation` | 9 / 10 | 1 |
| `general_conversation_and_navigation` | 10 / 10 | 0 |
| `general_data_guidance` | 10 / 10 | 0 |
| `general_model_concepts` | 10 / 10 | 0 |
| `general_system_identification` | 9 / 10 | 1 |
| `missing_run_safety` | 8 / 10 | 2 |
| `run_agent_records_and_prompts` | 10 / 10 | 0 |
| `run_context_followups` | 10 / 10 | 0 |
| `run_cycle_specific` | 10 / 10 | 0 |
| `run_settings_and_prompt_snapshot` | 7 / 10 | 3 |
| `run_state_ranking` | 10 / 10 | 0 |
| `run_summary_and_metrics` | 9 / 10 | 1 |
| `settings_chat_edits` | 17 / 20 | 3 |
| `setup_and_initializer_prompts` | 8 / 10 | 2 |
| `setup_architecture_and_pinn` | 3 / 10 | 7 |
| `setup_context_followups` | 10 / 10 | 0 |
| `setup_dataset_and_assumptions` | 6 / 10 | 4 |
| `setup_status_and_recommendation` | 9 / 10 | 1 |
| `setup_value_questions` | 7 / 10 | 3 |

## Failed cases

### general_system_identification-05 · general_system_identification

> What does it mean for a model to predict the next state?

- route: expected general, got run

### dataset_specific_interpretation-09 · dataset_specific_interpretation

> What does the measured profile say about my signals?

- missing evidence kinds: measurement

### setup_status_and_recommendation-07 · setup_status_and_recommendation

> What settings will the first candidate use?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_and_initializer_prompts-09 · setup_and_initializer_prompts

> What prompt does the run setup agent use to suggest an architecture?

- missing evidence IDs: P_initializer

### setup_and_initializer_prompts-10 · setup_and_initializer_prompts

> Where can I see the model recommendation instructions?

- missing evidence IDs: P_initializer

### setup_architecture_and_pinn-01 · setup_architecture_and_pinn

> Should I start with an LSTM or an MLP for these measurements?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-04 · setup_architecture_and_pinn

> What does my setup say about using PINN?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-05 · setup_architecture_and_pinn

> Why is the physics constraint turned off in this setup?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-06 · setup_architecture_and_pinn

> Would an equation be needed before PINN can be enabled?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-07 · setup_architecture_and_pinn

> Has the current setup validated a physics equation?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-08 · setup_architecture_and_pinn

> Can I use an MLP even though LabCD suggested an LSTM?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_architecture_and_pinn-09 · setup_architecture_and_pinn

> What should I consider before choosing PINN for this model?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_dataset_and_assumptions-03 · setup_dataset_and_assumptions

> Which states were marked as possible wrap angles in the starting setup?

- missing evidence kinds: measurement

### setup_dataset_and_assumptions-04 · setup_dataset_and_assumptions

> What trajectory structure is selected in my starting setup?

- missing evidence kinds: measurement

### setup_dataset_and_assumptions-07 · setup_dataset_and_assumptions

> Does my starting setup use automatic or manual angle detection?

- missing evidence kinds: measurement

### setup_dataset_and_assumptions-10 · setup_dataset_and_assumptions

> Can you explain the data assumptions in my starting setup before I train?

- missing evidence kinds: measurement

### setup_value_questions-06 · setup_value_questions

> Which activation function is selected?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_value_questions-08 · setup_value_questions

> What optimization priority is selected?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_value_questions-09 · setup_value_questions

> What trajectory split mode is configured?

- route: expected setup, got general
- missing evidence kinds: setup

### run_summary_and_metrics-03 · run_summary_and_metrics

> What does the held-out verification say about this model?

- route: expected run, got general
- missing evidence kinds: results, verification

### run_settings_and_prompt_snapshot-05 · run_settings_and_prompt_snapshot

> What architecture was recorded in the run context?

- route: expected run, got general
- missing evidence kinds: results, settings

### run_settings_and_prompt_snapshot-07 · run_settings_and_prompt_snapshot

> What trajectory structure did the run use?

- route: expected run, got general
- missing evidence kinds: results, settings

### run_settings_and_prompt_snapshot-09 · run_settings_and_prompt_snapshot

> What is known from the saved settings and what is missing?

- route: expected run, got general
- missing evidence kinds: results, settings

### missing_run_safety-03 · missing_run_safety

> What did the initializer decide in that run?

- route: expected run, got setup
- offline answer path: AssertionError: Offline benchmark attempted an API call

### missing_run_safety-06 · missing_run_safety

> What did the training agents say?

- route: expected run, got general
- offline answer path: AssertionError: Offline benchmark attempted an API call

### settings_chat_edits-02 · settings_chat_edits

> set the learning rate to 0.002

- settings schema rejected the proposed change: ValueError: The manual starting learning rate must fit the selected search range.

### settings_chat_edits-09 · settings_chat_edits

> use 3 layers with 64, 128, and 64 neurons

- request was not dispatched to the settings planner
- requested setting 'manual_starting_hidden_layers' was not extracted

### settings_chat_edits-10 · settings_chat_edits

> make it 4 recurrent layers with 96 neurons each

- requested setting 'manual_starting_hidden_layers' was not extracted
- settings schema rejected the proposed change: ValidationError: 2 validation errors for RunSettings
num_layers_min
  Input should be less than or equal to 8 [type=less_than_equal, input_value=96, input_type=int]
    For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal
num_layers_max
  Input should be less than or equal to 8 [type=less_than_equal, input_value=96, input_type=int]
    For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal
