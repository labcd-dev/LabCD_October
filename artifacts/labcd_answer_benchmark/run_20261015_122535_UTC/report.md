# LabCD answer benchmark

- Mode: **live**
- Cases: **60** · passed **51** · failed **9** · blocked **0**
- Pass rate on scored cases (60): **85.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 53 / 58 | 91.4% |
| `llm_scope_selection` | 45 / 50 | 90.0% |
| `agent_role_selection` | 6 / 7 | 85.7% |
| `evidence_target_selection` | 14 / 15 | 93.3% |
| `cycle_target_selection` | 2 / 2 | 100.0% |
| `state_ranking_intent` | 4 / 4 | 100.0% |
| `answerer_selection` | 2 / 2 | 100.0% |
| `evidence_retrieval` | 52 / 58 | 89.7% |
| `settings_edits` | 2 / 2 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 49 / 58 | 84.5% |
| `model_generated_answers` | 38 / 45 | 84.4% |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `active_run_chat` | 3 | 0 | 0 | 0 |
| `dataset_specific_interpretation` | 3 | 0 | 0 | 0 |
| `general_conversation_and_navigation` | 3 | 0 | 0 | 0 |
| `general_data_guidance` | 3 | 0 | 0 | 0 |
| `general_model_concepts` | 3 | 0 | 0 | 0 |
| `general_system_identification` | 3 | 0 | 0 | 0 |
| `missing_run_safety` | 3 | 0 | 0 | 0 |
| `run_agent_records_and_prompts` | 2 | 1 | 0 | 0 |
| `run_context_followups` | 3 | 0 | 0 | 0 |
| `run_cycle_specific` | 3 | 0 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 1 | 2 | 0 | 0 |
| `run_state_ranking` | 3 | 0 | 0 | 0 |
| `run_summary_and_metrics` | 3 | 0 | 0 | 0 |
| `semantic_scope_disambiguation` | 1 | 1 | 0 | 0 |
| `settings_chat_edits` | 2 | 0 | 0 | 0 |
| `setup_and_initializer_prompts` | 0 | 3 | 0 | 0 |
| `setup_architecture_and_pinn` | 2 | 1 | 0 | 0 |
| `setup_context_followups` | 2 | 0 | 0 | 0 |
| `setup_dataset_and_assumptions` | 2 | 1 | 0 | 0 |
| `setup_status_and_recommendation` | 3 | 0 | 0 | 0 |
| `setup_value_questions` | 3 | 0 | 0 | 0 |

## Failed cases

### setup_and_initializer_prompts-01 · setup_and_initializer_prompts

### run_settings_and_prompt_snapshot-01 · run_settings_and_prompt_snapshot

### setup_and_initializer_prompts-02 · setup_and_initializer_prompts

### run_settings_and_prompt_snapshot-02 · run_settings_and_prompt_snapshot

### semantic_scope_disambiguation-02 · semantic_scope_disambiguation

### setup_and_initializer_prompts-03 · setup_and_initializer_prompts

### setup_architecture_and_pinn-03 · setup_architecture_and_pinn

### setup_dataset_and_assumptions-03 · setup_dataset_and_assumptions

### run_agent_records_and_prompts-03 · run_agent_records_and_prompts
