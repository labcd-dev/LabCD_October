# LabCD answer benchmark

- Mode: **live**
- Cases: **20** · passed **4** · failed **0** · blocked **16**
- Pass rate on scored cases (4): **100.0%**
- Dataset/run inputs: synthetic fixtures only

## Score by evaluation layer

| Layer | Passed | Pass rate |
|---|---:|---:|
| `routing` | 19 / 19 | 100.0% |
| `evidence_retrieval` | 19 / 19 | 100.0% |
| `settings_edits` | 1 / 1 | 100.0% |
| `local_answers` | 0 / 0 | n/a |
| `completed_live_responses` | 3 / 3 | 100.0% |
| `model_generated_answers` | 0 / 0 | n/a |

## Category results

| Category | Passed | Failed | Blocked |
|---|---:|---:|---:|
| `active_run_chat` | 0 | 0 | 1 |
| `dataset_specific_interpretation` | 0 | 0 | 1 |
| `general_conversation_and_navigation` | 1 | 0 | 0 |
| `general_data_guidance` | 0 | 0 | 1 |
| `general_model_concepts` | 0 | 0 | 1 |
| `general_system_identification` | 0 | 0 | 1 |
| `missing_run_safety` | 1 | 0 | 0 |
| `run_agent_records_and_prompts` | 0 | 0 | 1 |
| `run_context_followups` | 0 | 0 | 1 |
| `run_cycle_specific` | 0 | 0 | 1 |
| `run_settings_and_prompt_snapshot` | 0 | 0 | 1 |
| `run_state_ranking` | 1 | 0 | 0 |
| `run_summary_and_metrics` | 0 | 0 | 1 |
| `settings_chat_edits` | 1 | 0 | 0 |
| `setup_and_initializer_prompts` | 0 | 0 | 1 |
| `setup_architecture_and_pinn` | 0 | 0 | 1 |
| `setup_context_followups` | 0 | 0 | 1 |
| `setup_dataset_and_assumptions` | 0 | 0 | 1 |
| `setup_status_and_recommendation` | 0 | 0 | 1 |
| `setup_value_questions` | 0 | 0 | 1 |

## Failed cases

No scored benchmark checks failed.

## Unscored provider-blocked cases

- `general_system_identification-01`: Configured AI provider rejected or could not serve the request.
> What is system identification in plain language?


- `general_model_concepts-01`: Configured AI provider rejected or could not serve the request.
> What is an MLP and when is it useful?


- `general_data_guidance-01`: Configured AI provider rejected or could not serve the request.
> What columns should a dataset for this kind of model contain?


- `dataset_specific_interpretation-01`: Configured AI provider rejected or could not serve the request.
> What state and input columns are in my dataset?


- `setup_status_and_recommendation-01`: Configured AI provider rejected or could not serve the request.
> What is the current starting setup?


- `setup_and_initializer_prompts-01`: Configured AI provider rejected or could not serve the request.
> What prompt does the initializer use?


- `setup_architecture_and_pinn-01`: Configured AI provider rejected or could not serve the request.
> Should I start with an LSTM or an MLP for these measurements?


- `setup_dataset_and_assumptions-01`: Configured AI provider rejected or could not serve the request.
> Which state names from the uploaded dataset are listed in my starting setup?


- `setup_value_questions-01`: Configured AI provider rejected or could not serve the request.
> What is my current learning rate?


- `run_summary_and_metrics-01`: Configured AI provider rejected or could not serve the request.
> How did this run go?


- `run_cycle_specific-01`: Provider unavailable; LabCD returned its local evidence fallback.
> What happened in cycle 1?


Answer excerpt: AI diagnosis is unavailable. These are local evidence checks, not a full answer to your question.

- `run_agent_records_and_prompts-01`: Provider unavailable; LabCD returned its local evidence fallback.
> What did the initializer actually propose in this run?


Answer excerpt: AI diagnosis is unavailable. These are local evidence checks, not a full answer to your question.

- `run_settings_and_prompt_snapshot-01`: Configured AI provider rejected or could not serve the request.
> Which settings were captured when this run started?


- `active_run_chat-01`: Provider unavailable; LabCD returned its local evidence fallback.
> How is the current cycle progressing?


Answer excerpt: AI diagnosis is unavailable. These are local evidence checks, not a full answer to your question.

- `run_context_followups-01`: Configured AI provider rejected or could not serve the request.
> for cycle 3


- `setup_context_followups-01`: Configured AI provider rejected or could not serve the request.
> yes
