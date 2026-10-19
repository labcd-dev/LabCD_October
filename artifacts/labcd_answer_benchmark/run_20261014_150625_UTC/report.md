# LabCD answer benchmark

- Mode: **live**
- Cases: **20** · passed **6** · failed **14**
- Pass rate: **30.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 19 / 19 | 100.0% |
| `evidence_retrieval` | 19 / 19 | 100.0% |
| `settings_edits` | 1 / 1 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `live_answers` | 5 / 19 | 26.3% |

## Category results

| Category | Passed | Failed |
|---|---:|---:|
| `active_run_chat` | 1 / 1 | 0 |
| `dataset_specific_interpretation` | 0 / 1 | 1 |
| `general_conversation_and_navigation` | 1 / 1 | 0 |
| `general_data_guidance` | 0 / 1 | 1 |
| `general_model_concepts` | 0 / 1 | 1 |
| `general_system_identification` | 0 / 1 | 1 |
| `missing_run_safety` | 1 / 1 | 0 |
| `run_agent_records_and_prompts` | 1 / 1 | 0 |
| `run_context_followups` | 0 / 1 | 1 |
| `run_cycle_specific` | 0 / 1 | 1 |
| `run_settings_and_prompt_snapshot` | 0 / 1 | 1 |
| `run_state_ranking` | 1 / 1 | 0 |
| `run_summary_and_metrics` | 0 / 1 | 1 |
| `settings_chat_edits` | 1 / 1 | 0 |
| `setup_and_initializer_prompts` | 0 / 1 | 1 |
| `setup_architecture_and_pinn` | 0 / 1 | 1 |
| `setup_context_followups` | 0 / 1 | 1 |
| `setup_dataset_and_assumptions` | 0 / 1 | 1 |
| `setup_status_and_recommendation` | 0 / 1 | 1 |
| `setup_value_questions` | 0 / 1 | 1 |

## Failed cases

### general_system_identification-01 · general_system_identification

> What is system identification in plain language?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty

### general_model_concepts-01 · general_model_concepts

> What is an MLP and when is it useful?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty

### general_data_guidance-01 · general_data_guidance

> What columns should a dataset for this kind of model contain?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty

### dataset_specific_interpretation-01 · dataset_specific_interpretation

> What state and input columns are in my dataset?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- answer missed expected fact: s_speed
- answer missed expected fact: s_heading
- answer missed expected fact: s_temp
- answer missed expected fact: a_drive
- answer missed expected fact: a_fan

### setup_status_and_recommendation-01 · setup_status_and_recommendation

> What is the current starting setup?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### setup_and_initializer_prompts-01 · setup_and_initializer_prompts

> What prompt does the initializer use?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### setup_architecture_and_pinn-01 · setup_architecture_and_pinn

> Should I start with an LSTM or an MLP for these measurements?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### setup_dataset_and_assumptions-01 · setup_dataset_and_assumptions

> Which state names from the uploaded dataset are listed in my starting setup?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### setup_value_questions-01 · setup_value_questions

> What is my current learning rate?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### run_summary_and_metrics-01 · run_summary_and_metrics

> How did this run go?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### run_cycle_specific-01 · run_cycle_specific

> What happened in cycle 1?

- answer missed expected fact: 0\.0041

Answer excerpt: AI diagnosis is unavailable. These are local evidence checks, not a full answer to your question.

### run_settings_and_prompt_snapshot-01 · run_settings_and_prompt_snapshot

> Which settings were captured when this run started?

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### run_context_followups-01 · run_context_followups

> for cycle 3

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs

### setup_context_followups-01 · setup_context_followups

> yes

- answer status: expected answer, got error
- live model or diagnostic call failed
- answer text is empty
- evidence-scoped answer returned no source IDs
