import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

from backend_core.packages.labcd_agents.src.labcd_agents import LLMFactory
from backend_core.packages.labcd_agents.src.labcd_agents.agent import BaseAgent

# Global thread-safe list to stream real-time decisions to the Streamlit UI
LIVE_AGENT_DECISIONS = []


class Agents(BaseAgent):
    def __init__(self, model_name="gpt-oss-120b", prompt_dir=None):
        super().__init__(model=model_name, temperature=1.0)
        LIVE_AGENT_DECISIONS.clear()
        current_file_dir = Path(__file__).parent.resolve()

        if (current_file_dir / "templates").exists():
            self.prompt_dir = current_file_dir / "templates"
        elif (current_file_dir.parent / "templates").exists():
            self.prompt_dir = current_file_dir.parent / "templates"
        elif prompt_dir:
            self.prompt_dir = Path(prompt_dir).resolve()
        else:
            raise FileNotFoundError("Could not locate the 'templates' directory.")

        self.prompts = self._load_prompts(self.prompt_dir)
        print(f"\n[DEBUG] Successfully loaded templates from: {self.prompt_dir}")
        print(f"[DEBUG] Loaded agents: {list(self.prompts.keys())}\n")

    def _load_prompts(self, directory: Path) -> Dict[str, Any]:
        registry: Dict[str, Any] = {}

        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"CRITICAL: Cannot find templates folder at {directory}")

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in {'.yaml', '.yml'}:
                task_name = file_path.stem
                try:
                    with file_path.open('r', encoding='utf-8') as f:
                        registry[task_name] = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"Failed to parse YAML in {file_path.name}: {e}")

        return registry

    def _get_prompt_string(self, task_name: str, file_key: str = None) -> str:
        task_dict = self.prompts.get(task_name)
        if not task_dict:
            raise KeyError(f"Missing prompt file '{task_name}.yaml'. Found files: {list(self.prompts.keys())}")

        prompt_str = task_dict.get('prompt')
        if not prompt_str:
            raise KeyError(f"The file '{task_name}.yaml' is missing the root 'prompt:' key.")

        return prompt_str

    def constraint_estimator(self, controller_structure, trimming_result, user_context: str = ""):
        prompt_text = self._get_prompt_string('constraint_estimator', 'prompt').format(
            CONTROLLER_STRUCTURE_JSON=json.dumps(controller_structure),
            TRIM_JSON=json.dumps(trimming_result),
            USER_CONTEXT=user_context if user_context else "None"
        )

        response_content = self.call(
            prompt_text=prompt_text,
            system=True,
            is_json=True
        )

        response_content = response_content.replace('"output_variable"', '"output_signal"')
        return response_content


    def constraint_estimator_web(self, controller_structure, trimming_result, target_model, user_context: str = ""):
        prompt_text = self._get_prompt_string('constraint_estimator_web', 'prompt').format(
            CONTROLLER_STRUCTURE_JSON=json.dumps(controller_structure),
            TRIM_JSON=json.dumps(trimming_result),
            USER_CONTEXT=user_context if user_context else "None"
        )
        schema = self.prompts['constraint_estimator_web']['schema']

        target_client = LLMFactory.create(target_model, temperature=self.temperature)

        bound_client = target_client.bind(
            tools=[{"type": "web_search_preview"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_result",
                    "schema": schema,
                    "strict": True
                }
            }
        )

        response_content = self.call(
            prompt_text=prompt_text,
            system=True,
            client=bound_client
        )

        response_content = response_content.replace('"output_variable"', '"output_signal"')
        return response_content

    def supervisor_agent(self, agent_context: dict, optimizer_choice: str, max_attempts: int) -> dict:
        task_name = "pso_agent" if optimizer_choice == "PSO Optimizer" else "supervisor_agent"
        prompt_text = self._get_prompt_string(task_name, task_name).format(
            TUNING_STATE_JSON=json.dumps(agent_context, indent=2),
            MAX_ATTEMPTS=max_attempts
        )

        response_content = self.call(
            prompt_text=prompt_text,
            system=True,
            is_json=True
        )

        decision = json.loads(response_content)
        LIVE_AGENT_DECISIONS.append({"agent": "🧑‍💼 Supervisor Agent", "response": decision})
        return decision

    def fine_tuner_agent(self, control_block: list, agent_context: dict, max_tries: int) -> dict:
        prompt_text = self._get_prompt_string('fine_tuner_agent', 'fine_tuner_agent').format(
            CONTROL_BLOCK_JSON=json.dumps(control_block, indent=2),
            TUNING_STATE_JSON=json.dumps(agent_context, indent=2),
            MAX_TRIES=max_tries
        )

        response_content = self.call(
            prompt_text=prompt_text,
            system=True,
            is_json=True
        )

        decision = json.loads(response_content)
        LIVE_AGENT_DECISIONS.append({"agent": "👌 Fine Tuner Agent", "response": decision})
        return decision

    def evaluate_agent(self, full_architecture: list, agent_context: dict, max_redesigns: int) -> dict:
        prompt_text = self._get_prompt_string('evaluate_agent', 'evaluate_agent').format(
            ARCHITECTURE_JSON=json.dumps(full_architecture, indent=2),
            TUNING_STATE_JSON=json.dumps(agent_context, indent=2),
            MAX_REDESIGNS=max_redesigns
        )

        response_content = self.call(
            prompt_text=prompt_text,
            system=True,
            is_json=True
        )

        decision = json.loads(response_content)
        LIVE_AGENT_DECISIONS.append({"agent": "🔍 Evaluate Agent", "response": decision})
        return decision

    def compute_cost(self, in_tokens: int, out_tokens: int) -> float:
        """Routes the token count to the centralized CostCalculator."""
        return self.cost_calculator.compute_cost(self.model, in_tokens, out_tokens)


    def design_agent(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, float]]:
        """Replaces the LLMAgent.invoke method for GA configuration tasks."""
        response_text, usage = self.invoke_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Calculate cost
        call_cost = self.compute_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )

        # Convert to the legacy dictionary structure expected by graph.py
        usage_compat = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'call_cost': call_cost
        }

        print(f"[DesignAgent] Tokens in={usage_compat['prompt_tokens']}, "
              f"out={usage_compat['completion_tokens']}, Cost=${usage_compat['call_cost']:.6f}")

        response = json.loads(response_text)
        LIVE_AGENT_DECISIONS.append({"agent": "🎧 Optimizer Agent", "response": response})

        return response_text, usage_compat

