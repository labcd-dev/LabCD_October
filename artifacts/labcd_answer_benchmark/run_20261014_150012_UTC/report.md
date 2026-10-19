# LabCD answer benchmark

- Mode: **offline**
- Cases: **211** · passed **208** · failed **3**
- Pass rate: **98.6%**
- Dataset/run inputs: synthetic fixtures only

## Category results

| Category | Passed | Failed |
|---|---:|---:|
| `active_run_chat` | 10 / 10 | 0 |
| `dataset_specific_interpretation` | 10 / 10 | 0 |
| `general_conversation_and_navigation` | 10 / 10 | 0 |
| `general_data_guidance` | 10 / 10 | 0 |
| `general_model_concepts` | 10 / 10 | 0 |
| `general_system_identification` | 10 / 10 | 0 |
| `missing_run_safety` | 10 / 10 | 0 |
| `run_agent_records_and_prompts` | 10 / 10 | 0 |
| `run_context_followups` | 10 / 10 | 0 |
| `run_cycle_specific` | 10 / 10 | 0 |
| `run_settings_and_prompt_snapshot` | 10 / 10 | 0 |
| `run_state_ranking` | 10 / 10 | 0 |
| `run_summary_and_metrics` | 10 / 10 | 0 |
| `settings_chat_edits` | 21 / 21 | 0 |
| `setup_and_initializer_prompts` | 9 / 10 | 1 |
| `setup_architecture_and_pinn` | 9 / 10 | 1 |
| `setup_context_followups` | 10 / 10 | 0 |
| `setup_dataset_and_assumptions` | 9 / 10 | 1 |
| `setup_status_and_recommendation` | 10 / 10 | 0 |
| `setup_value_questions` | 10 / 10 | 0 |

## Failed cases

### setup_and_initializer_prompts-09 · setup_and_initializer_prompts

> What prompt does the run setup agent use to suggest an architecture?

- route: expected setup, got run
- missing evidence kinds: setup, template
- missing evidence IDs: P_setup

### setup_architecture_and_pinn-09 · setup_architecture_and_pinn

> What should I consider before choosing PINN for this model?

- route: expected setup, got general
- missing evidence kinds: setup

### setup_dataset_and_assumptions-10 · setup_dataset_and_assumptions

> Can you explain the data assumptions in my starting setup before I train?

- missing evidence kinds: measurement
