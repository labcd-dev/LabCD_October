# LabCD answer benchmark

- Mode: **offline**
- Cases: **221** · passed **209** · failed **2** · blocked **0**
- Pass rate on scored cases (211): **99.1%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 189 / 190 | 99.5% |
| `llm_scope_selection` | 0 / 0 | n/a |
| `agent_role_selection` | 0 / 0 | n/a |
| `evidence_target_selection` | 0 / 0 | n/a |
| `cycle_target_selection` | 0 / 0 | n/a |
| `state_ranking_intent` | 0 / 0 | n/a |
| `answerer_selection` | 0 / 0 | n/a |
| `evidence_retrieval` | 189 / 190 | 99.5% |
| `settings_edits` | 21 / 21 | 100.0% |
| `local_answers` | 16 / 16 | 100.0% |
| `completed_live_responses` | 0 / 0 | n/a |
| `model_generated_answers` | 0 / 0 | n/a |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `active_run_chat` | 10 | 0 | 0 | 0 |
| `dataset_specific_interpretation` | 10 | 0 | 0 | 0 |
| `general_conversation_and_navigation` | 10 | 0 | 0 | 0 |
| `general_data_guidance` | 10 | 0 | 0 | 0 |
| `general_model_concepts` | 10 | 0 | 0 | 0 |
| `general_system_identification` | 10 | 0 | 0 | 0 |
| `missing_run_safety` | 10 | 0 | 0 | 0 |
| `run_agent_records_and_prompts` | 10 | 0 | 0 | 0 |
| `run_context_followups` | 9 | 1 | 0 | 0 |
| `run_cycle_specific` | 10 | 0 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 10 | 0 | 0 | 0 |
| `run_state_ranking` | 10 | 0 | 0 | 0 |
| `run_summary_and_metrics` | 9 | 1 | 0 | 0 |
| `semantic_scope_disambiguation` | 0 | 0 | 0 | 10 |
| `settings_chat_edits` | 21 | 0 | 0 | 0 |
| `setup_and_initializer_prompts` | 10 | 0 | 0 | 0 |
| `setup_architecture_and_pinn` | 10 | 0 | 0 | 0 |
| `setup_context_followups` | 10 | 0 | 0 | 0 |
| `setup_dataset_and_assumptions` | 10 | 0 | 0 | 0 |
| `setup_status_and_recommendation` | 10 | 0 | 0 | 0 |
| `setup_value_questions` | 10 | 0 | 0 | 0 |

## Failed cases

### run_summary_and_metrics-02 · run_summary_and_metrics

### run_context_followups-03 · run_context_followups
