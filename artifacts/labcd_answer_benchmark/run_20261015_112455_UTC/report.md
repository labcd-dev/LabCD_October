# LabCD answer benchmark

- Mode: **live**
- Cases: **20** · passed **16** · failed **4** · blocked **0**
- Pass rate on scored cases (20): **80.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 19 / 19 | 100.0% |
| `evidence_retrieval` | 18 / 19 | 94.7% |
| `settings_edits` | 1 / 1 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 15 / 19 | 79.0% |
| `model_generated_answers` | 12 / 14 | 85.7% |

## Category results

| Category | Passed | Failed | Blocked |
|---|---:|---:|---:|
| `active_run_chat` | 0 | 1 | 0 |
| `dataset_specific_interpretation` | 1 | 0 | 0 |
| `general_conversation_and_navigation` | 1 | 0 | 0 |
| `general_data_guidance` | 1 | 0 | 0 |
| `general_model_concepts` | 1 | 0 | 0 |
| `general_system_identification` | 1 | 0 | 0 |
| `missing_run_safety` | 1 | 0 | 0 |
| `run_agent_records_and_prompts` | 1 | 0 | 0 |
| `run_context_followups` | 1 | 0 | 0 |
| `run_cycle_specific` | 0 | 1 | 0 |
| `run_settings_and_prompt_snapshot` | 1 | 0 | 0 |
| `run_state_ranking` | 1 | 0 | 0 |
| `run_summary_and_metrics` | 1 | 0 | 0 |
| `settings_chat_edits` | 1 | 0 | 0 |
| `setup_and_initializer_prompts` | 1 | 0 | 0 |
| `setup_architecture_and_pinn` | 0 | 1 | 0 |
| `setup_context_followups` | 1 | 0 | 0 |
| `setup_dataset_and_assumptions` | 0 | 1 | 0 |
| `setup_status_and_recommendation` | 1 | 0 | 0 |
| `setup_value_questions` | 1 | 0 | 0 |

## Failed cases

### setup_architecture_and_pinn-01 · setup_architecture_and_pinn

### setup_dataset_and_assumptions-01 · setup_dataset_and_assumptions

### run_cycle_specific-01 · run_cycle_specific

### active_run_chat-01 · active_run_chat
