# LabCD answer benchmark

- Mode: **live**
- Cases: **10** · passed **8** · failed **2** · blocked **0**
- Pass rate on scored cases (10): **80.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 10 / 10 | 100.0% |
| `llm_scope_selection` | 10 / 10 | 100.0% |
| `agent_role_selection` | 5 / 6 | 83.3% |
| `evidence_target_selection` | 10 / 10 | 100.0% |
| `cycle_target_selection` | 4 / 5 | 80.0% |
| `state_ranking_intent` | 1 / 1 | 100.0% |
| `answerer_selection` | 10 / 10 | 100.0% |
| `evidence_retrieval` | 10 / 10 | 100.0% |
| `settings_edits` | 0 / 0 | n/a |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 8 / 10 | 80.0% |
| `model_generated_answers` | 8 / 9 | 88.9% |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `semantic_scope_disambiguation` | 8 | 2 | 0 | 0 |

## Failed cases

### semantic_scope_disambiguation-04 · semantic_scope_disambiguation

### semantic_scope_disambiguation-08 · semantic_scope_disambiguation
