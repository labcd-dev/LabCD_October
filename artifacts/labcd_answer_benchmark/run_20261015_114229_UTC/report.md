# LabCD answer benchmark

- Mode: **live**
- Cases: **20** · passed **15** · failed **5** · blocked **0**
- Pass rate on scored cases (20): **75.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 19 / 19 | 100.0% |
| `evidence_retrieval` | 15 / 19 | 79.0% |
| `settings_edits` | 1 / 1 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 14 / 19 | 73.7% |
| `model_generated_answers` | 9 / 11 | 81.8% |

## Category results

| Category | Passed | Failed | Blocked |
|---|---:|---:|---:|
| `active_run_chat` | 1 | 0 | 0 |
| `dataset_specific_interpretation` | 1 | 0 | 0 |
| `general_conversation_and_navigation` | 1 | 0 | 0 |
| `general_data_guidance` | 1 | 0 | 0 |
| `general_model_concepts` | 1 | 0 | 0 |
| `general_system_identification` | 1 | 0 | 0 |
| `missing_run_safety` | 1 | 0 | 0 |
| `run_agent_records_and_prompts` | 0 | 1 | 0 |
| `run_context_followups` | 1 | 0 | 0 |
| `run_cycle_specific` | 1 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 0 | 1 | 0 |
| `run_state_ranking` | 0 | 1 | 0 |
| `run_summary_and_metrics` | 0 | 1 | 0 |
| `settings_chat_edits` | 1 | 0 | 0 |
| `setup_and_initializer_prompts` | 1 | 0 | 0 |
| `setup_architecture_and_pinn` | 1 | 0 | 0 |
| `setup_context_followups` | 0 | 1 | 0 |
| `setup_dataset_and_assumptions` | 1 | 0 | 0 |
| `setup_status_and_recommendation` | 1 | 0 | 0 |
| `setup_value_questions` | 1 | 0 | 0 |

## Failed cases

### run_summary_and_metrics-01 · run_summary_and_metrics

### run_state_ranking-01 · run_state_ranking

### run_agent_records_and_prompts-01 · run_agent_records_and_prompts

### run_settings_and_prompt_snapshot-01 · run_settings_and_prompt_snapshot

### setup_context_followups-01 · setup_context_followups
