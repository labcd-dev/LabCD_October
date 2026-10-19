# LabCD answer benchmark

- Mode: **live**
- Cases: **10** · passed **9** · failed **1** · blocked **0**
- Pass rate on scored cases (10): **90.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 10 / 10 | 100.0% |
| `llm_scope_selection` | 10 / 10 | 100.0% |
| `agent_role_selection` | 6 / 6 | 100.0% |
| `evidence_target_selection` | 10 / 10 | 100.0% |
| `cycle_target_selection` | 5 / 5 | 100.0% |
| `state_ranking_intent` | 1 / 1 | 100.0% |
| `answerer_selection` | 10 / 10 | 100.0% |
| `evidence_retrieval` | 10 / 10 | 100.0% |
| `settings_edits` | 0 / 0 | n/a |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 9 / 10 | 90.0% |
| `model_generated_answers` | 9 / 9 | 100.0% |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `semantic_scope_disambiguation` | 9 | 1 | 0 | 0 |

## Failed cases

### semantic_scope_disambiguation-09 · semantic_scope_disambiguation
