"""LLM API cost tracking (ported from _legacy/framework.APICostTracker)."""

from __future__ import annotations

from typing import Any, Dict


class APICostTracker:
    """Accumulates token usage across every agent call and prices the run."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0

        # Current pricing per 1 Million tokens (USD)
        self.pricing_table: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "gpt-4o": {"prompt": 5.00, "completion": 15.00},
            "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
            "llama3-8b-8192": {"prompt": 0.05, "completion": 0.08},
            "default": {"prompt": 1.00, "completion": 2.00},  # generic fallback
        }

    # ------------------------------------------------------------------
    def update(self, response: Any) -> None:
        """Record one API call, tolerating every provider's usage schema."""
        self.total_calls += 1
        try:
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {}) or {}

                # Format 1: standard OpenAI structure
                if "prompt_tokens" in usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self.completion_tokens += usage.get("completion_tokens", 0) or 0
                # Format 2: alternative Groq / Anthropic structure
                elif "input_tokens" in usage:
                    self.prompt_tokens += usage.get("input_tokens", 0) or 0
                    self.completion_tokens += usage.get("output_tokens", 0) or 0
        except Exception:
            pass

    # ------------------------------------------------------------------
    def rates_for(self, model_name: str) -> Dict[str, float]:
        """Match the active model against the pricing table."""
        rates = self.pricing_table["default"]
        name = (model_name or "").lower()
        for key in self.pricing_table:
            if key != "default" and key in name:
                rates = self.pricing_table[key]
                break
        return rates

    def estimated_cost(self, model_name: str) -> float:
        rates = self.rates_for(model_name)
        cost_prompt = (self.prompt_tokens / 1_000_000.0) * rates["prompt"]
        cost_comp = (self.completion_tokens / 1_000_000.0) * rates["completion"]
        return cost_prompt + cost_comp

    def print_summary(self, model_name: str) -> None:
        total_cost = self.estimated_cost(model_name)

        print("\n" + "=" * 80)
        print("💰 LLM API COST SUMMARY")
        print("=" * 80)
        print(f"  ├── Active LLM Core      : {(model_name or 'unknown').upper()}")
        print(f"  ├── Total API Calls      : {self.total_calls} queries")
        print(f"  ├── Input Tokens (Prompt): {self.prompt_tokens:,} tokens")
        print(f"  ├── Output Tokens (Comp) : {self.completion_tokens:,} tokens")
        print("-" * 80)
        print(f"  💵 TOTAL ESTIMATED COST  : ${total_cost:.5f} USD")
        print("=" * 80 + "\n")


# Global tracker so every agent can contribute to one run total.
cost_tracker = APICostTracker()
