"""LLM API cost tracking."""

from __future__ import annotations


class APICostTracker:
    def __init__(self):
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Approximate USD pricing per 1M tokens (update as needed)
        self.pricing_table = {
            "gpt-4o": {"prompt": 2.50, "completion": 10.00},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "llama-3.3-70b": {"prompt": 0.59, "completion": 0.79},
            "llama-3.1-70b": {"prompt": 0.59, "completion": 0.79},
            "mixtral": {"prompt": 0.24, "completion": 0.24},
            "default": {"prompt": 1.00, "completion": 2.00},
        }

    def update(self, response) -> None:
        self.total_calls += 1
        try:
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
                if "prompt_tokens" in usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                elif "input_tokens" in usage:
                    self.prompt_tokens += usage.get("input_tokens", 0)
                    self.completion_tokens += usage.get("output_tokens", 0)
        except Exception:
            pass

    def print_summary(self, model_name: str) -> None:
        rates = self.pricing_table["default"]
        for key in self.pricing_table:
            if key in model_name.lower():
                rates = self.pricing_table[key]
                break

        cost_prompt = (self.prompt_tokens / 1_000_000.0) * rates["prompt"]
        cost_comp = (self.completion_tokens / 1_000_000.0) * rates["completion"]
        total_cost = cost_prompt + cost_comp

        print("\n" + "=" * 80)
        print("💰 LLM API COST SUMMARY")
        print("=" * 80)
        print(f"  ├── Active LLM Core      : {model_name.upper()}")
        print(f"  ├── Total API Calls      : {self.total_calls} queries")
        print(f"  ├── Input Tokens (Prompt): {self.prompt_tokens:,} tokens")
        print(f"  ├── Output Tokens (Comp) : {self.completion_tokens:,} tokens")
        print("-" * 80)
        print(f"  💵 TOTAL ESTIMATED COST  : ${total_cost:.5f} USD")
        print("=" * 80 + "\n")


# Global instance used by agents
cost_tracker = APICostTracker()
