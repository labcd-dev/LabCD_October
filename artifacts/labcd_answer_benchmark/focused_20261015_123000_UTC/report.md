# LabCD answer benchmark

- Mode: **live**
- Cases: **12** · passed **12** · failed **0** · blocked **0**
- Pass rate on scored cases (12): **100.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 12 / 12 | 100.0% |
| `llm_scope_selection` | 12 / 12 | 100.0% |
| `agent_role_selection` | 7 / 7 | 100.0% |
| `evidence_target_selection` | 10 / 10 | 100.0% |
| `cycle_target_selection` | 3 / 3 | 100.0% |
| `state_ranking_intent` | 0 / 0 | n/a |
| `answerer_selection` | 4 / 4 | 100.0% |
| `evidence_retrieval` | 12 / 12 | 100.0% |
| `settings_edits` | 0 / 0 | n/a |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 12 / 12 | 100.0% |
| `model_generated_answers` | 12 / 12 | 100.0% |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `run_agent_records_and_prompts` | 1 | 0 | 0 | 0 |
| `run_settings_and_prompt_snapshot` | 2 | 0 | 0 | 0 |
| `semantic_scope_disambiguation` | 4 | 0 | 0 | 0 |
| `setup_and_initializer_prompts` | 3 | 0 | 0 | 0 |
| `setup_architecture_and_pinn` | 1 | 0 | 0 | 0 |
| `setup_dataset_and_assumptions` | 1 | 0 | 0 | 0 |

## Failed cases

No scored benchmark checks failed.
