# LabCD answer benchmark

- Mode: **live**
- Cases: **60** · passed **54** · failed **6** · blocked **0**
- Pass rate on scored cases (60): **90.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 56 / 57 | 98.2% |
| `evidence_retrieval` | 54 / 57 | 94.7% |
| `settings_edits` | 3 / 3 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 51 / 57 | 89.5% |
| `model_generated_answers` | 37 / 39 | 94.9% |

## Category results

| Category | Passed | Failed | Blocked |
|---|---:|---:|---:|
| `active_run_chat` | 3 | 0 | 0 |
| `dataset_specific_interpretation` | 3 | 0 | 0 |
| `general_conversation_and_navigation` | 3 | 0 | 0 |
| `general_data_guidance` | 3 | 0 | 0 |
| `general_model_concepts` | 3 | 0 | 0 |
| `general_system_identification` | 3 | 0 | 0 |
| `missing_run_safety` | 3 | 0 | 0 |
| `run_agent_records_and_prompts` | 3 | 0 | 0 |
| `run_context_followups` | 3 | 0 | 0 |
| `run_cycle_specific` | 3 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 2 | 1 | 0 |
| `run_state_ranking` | 2 | 1 | 0 |
| `run_summary_and_metrics` | 2 | 1 | 0 |
| `settings_chat_edits` | 3 | 0 | 0 |
| `setup_and_initializer_prompts` | 2 | 1 | 0 |
| `setup_architecture_and_pinn` | 3 | 0 | 0 |
| `setup_context_followups` | 2 | 1 | 0 |
| `setup_dataset_and_assumptions` | 2 | 1 | 0 |
| `setup_status_and_recommendation` | 3 | 0 | 0 |
| `setup_value_questions` | 3 | 0 | 0 |

## Failed cases

### run_summary_and_metrics-02 · run_summary_and_metrics

### run_settings_and_prompt_snapshot-02 · run_settings_and_prompt_snapshot

### setup_and_initializer_prompts-03 · setup_and_initializer_prompts

### setup_dataset_and_assumptions-03 · setup_dataset_and_assumptions

### run_state_ranking-03 · run_state_ranking

### setup_context_followups-03 · setup_context_followups
