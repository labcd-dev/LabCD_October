# LabCD answer benchmark

- Mode: **live**
- Cases: **60** · passed **60** · failed **0** · blocked **0**
- Pass rate on scored cases (60): **100.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 58 / 58 | 100.0% |
| `llm_scope_selection` | 50 / 50 | 100.0% |
| `agent_role_selection` | 7 / 7 | 100.0% |
| `evidence_target_selection` | 15 / 15 | 100.0% |
| `cycle_target_selection` | 2 / 2 | 100.0% |
| `state_ranking_intent` | 4 / 4 | 100.0% |
| `answerer_selection` | 2 / 2 | 100.0% |
| `evidence_retrieval` | 58 / 58 | 100.0% |
| `settings_edits` | 2 / 2 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 58 / 58 | 100.0% |
| `model_generated_answers` | 47 / 47 | 100.0% |

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
| `run_agent_records_and_prompts` | 3 | 0 | 0 | 0 |
| `run_context_followups` | 3 | 0 | 0 | 0 |
| `run_cycle_specific` | 3 | 0 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 3 | 0 | 0 | 0 |
| `run_state_ranking` | 3 | 0 | 0 | 0 |
| `run_summary_and_metrics` | 3 | 0 | 0 | 0 |
| `semantic_scope_disambiguation` | 2 | 0 | 0 | 0 |
| `settings_chat_edits` | 2 | 0 | 0 | 0 |
| `setup_and_initializer_prompts` | 3 | 0 | 0 | 0 |
| `setup_architecture_and_pinn` | 3 | 0 | 0 | 0 |
| `setup_context_followups` | 2 | 0 | 0 | 0 |
| `setup_dataset_and_assumptions` | 3 | 0 | 0 | 0 |
| `setup_status_and_recommendation` | 3 | 0 | 0 | 0 |
| `setup_value_questions` | 3 | 0 | 0 | 0 |

## Failed cases

No scored benchmark checks failed.
