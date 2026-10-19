# LabCD answer benchmark

- Mode: **live**
- Cases: **10** · passed **7** · failed **3** · blocked **0**
- Pass rate on scored cases (10): **70.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 10 / 10 | 100.0% |
| `llm_scope_selection` | 10 / 10 | 100.0% |
| `agent_role_selection` | 4 / 6 | 66.7% |
| `evidence_target_selection` | 10 / 10 | 100.0% |
| `cycle_target_selection` | 4 / 5 | 80.0% |
| `state_ranking_intent` | 1 / 1 | 100.0% |
| `answerer_selection` | 10 / 10 | 100.0% |
| `evidence_retrieval` | 10 / 10 | 100.0% |
| `settings_edits` | 0 / 0 | n/a |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 7 / 10 | 70.0% |
| `model_generated_answers` | 7 / 9 | 77.8% |

## Category results

| Category | Passed | Failed | Blocked | Skipped |
|---|---:|---:|---:|---:|
| `semantic_scope_disambiguation` | 7 | 3 | 0 | 0 |

## Failed cases

### semantic_scope_disambiguation-01 · semantic_scope_disambiguation

### semantic_scope_disambiguation-03 · semantic_scope_disambiguation

### semantic_scope_disambiguation-08 · semantic_scope_disambiguation
