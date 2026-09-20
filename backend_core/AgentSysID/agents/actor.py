"""
Actor Agent.

Turns the Critic's verdict into the exact hyper-parameters for the next cycle,
inside the Initializer's authorised bounds, and guarantees the candidate has
never been tried before (learning rate is bucketed by ``LR_TOLERANCE`` so
cosmetically different configs are not counted as new).

Three layers of defence:
  1. LLM proposal parsed from the strict KEY: VALUE contract
  2. heuristic mutation — incremental topology change toward the Critic's
     target plus the regularization / batch / patience rules
  3. bounded random exploration until an unvisited configuration appears
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from backend_core.AgentSysID import config as cfg
from backend_core.AgentSysID.agents.llm_base import invoke_llm
from backend_core.AgentSysID.agents.prompt_library import block, render, system_prompt
from backend_core.AgentSysID.training.tracker import BestConfigTracker


class ActorAgent:
    def __init__(
        self,
        activation: str,
        initial_config: Optional[Dict[str, Any]] = None,
        log_filename: Optional[str] = None,
    ):
        self.activation = activation
        self.log_filename = log_filename

        if initial_config is not None:
            self.current_config: Dict[str, Any] = {
                "learning_rate": initial_config["learning_rate"],
                "hidden_layers": list(initial_config["hidden_layers"]),
                "dropout_rate": initial_config.get("dropout_rate", cfg.MANUAL_DROPOUT_RATE),
                "weight_decay": initial_config.get("weight_decay", cfg.MANUAL_WEIGHT_DECAY),
                "batch_size": initial_config.get("batch_size", cfg.BATCH_SIZE),
                "patience": initial_config.get("early_stop_patience", cfg.EARLY_STOP_PATIENCE),
            }
            self.lr_min = initial_config.get("lr_search_min", cfg.LEARNING_RATE_MIN)
            self.lr_max = initial_config.get("lr_search_max", cfg.LEARNING_RATE_MAX)
            self.hs_min = initial_config.get("hidden_size_search_min", cfg.HIDDEN_SIZE_MIN)
            self.hs_max = initial_config.get("hidden_size_search_max", cfg.HIDDEN_SIZE_MAX)
            self.num_layers_min = initial_config.get("num_layers_search_min", cfg.NUM_LAYERS_MIN)
            self.num_layers_max = initial_config.get("num_layers_search_max", cfg.NUM_LAYERS_MAX)
        else:
            self.current_config = {
                "learning_rate": cfg.MANUAL_STARTING_LR,
                "hidden_layers": list(cfg.MANUAL_STARTING_HIDDEN_LAYERS),
                "dropout_rate": cfg.MANUAL_DROPOUT_RATE,
                "weight_decay": cfg.MANUAL_WEIGHT_DECAY,
                "batch_size": cfg.BATCH_SIZE,
                "patience": cfg.EARLY_STOP_PATIENCE,
            }
            self.lr_min = cfg.LEARNING_RATE_MIN
            self.lr_max = cfg.LEARNING_RATE_MAX
            self.hs_min = cfg.HIDDEN_SIZE_MIN
            self.hs_max = cfg.HIDDEN_SIZE_MAX
            self.num_layers_min = cfg.NUM_LAYERS_MIN
            self.num_layers_max = cfg.NUM_LAYERS_MAX

        self.visited_configs: set[str] = set()
        self._add_to_visited(self.current_config)

    # ------------------------------------------------------------------
    # Bounds & visited-set bookkeeping
    # ------------------------------------------------------------------
    def unlock_global_bounds(self) -> None:
        """Release the Initializer's strict safety bounds (used after cycle 5)."""
        self.lr_min = cfg.LEARNING_RATE_MIN
        self.lr_max = cfg.LEARNING_RATE_MAX
        self.hs_min = cfg.HIDDEN_SIZE_MIN
        self.hs_max = cfg.HIDDEN_SIZE_MAX
        self.num_layers_min = cfg.NUM_LAYERS_MIN
        self.num_layers_max = cfg.NUM_LAYERS_MAX

    def _config_to_key(self, config: Dict[str, Any]) -> str:
        lr_rounded = round(config["learning_rate"] / cfg.LR_TOLERANCE) * cfg.LR_TOLERANCE
        hidden_tuple = tuple(config["hidden_layers"])
        # Regularization is part of the identity so the same topology can be
        # retried under a different dropout.
        drop_rounded = round(config.get("dropout_rate", 0.0), 2)
        return f"LR={lr_rounded:.5f}, HL={hidden_tuple}, Drop={drop_rounded}"

    def _add_to_visited(self, config: Dict[str, Any]) -> None:
        self.visited_configs.add(self._config_to_key(config))

    def _is_new_config(self, config: Dict[str, Any]) -> bool:
        return self._config_to_key(config) not in self.visited_configs

    # ------------------------------------------------------------------
    # Mutation primitives
    # ------------------------------------------------------------------
    def _random_exploration(self) -> Dict[str, Any]:
        new_lr = np.random.uniform(self.lr_min, self.lr_max)
        num_layers = np.random.randint(self.num_layers_min, self.num_layers_max + 1)
        new_hidden = [
            int(np.random.randint(self.hs_min, self.hs_max + 1)) for _ in range(num_layers)
        ]

        new_config = dict(self.current_config)
        new_config["learning_rate"] = float(new_lr)
        new_config["hidden_layers"] = new_hidden
        new_config["dropout_rate"] = float(np.clip(np.random.uniform(0.0, 0.4), 0.0, 0.5))
        new_config["weight_decay"] = float(np.clip(np.random.uniform(0.00001, 0.01), 0.0, 0.1))
        return new_config

    def _incremental_hidden_change(
        self, current_hidden: List[int], target_hidden: List[int]
    ) -> List[int]:
        """Step the topology toward the Critic's target by at most 32 neurons/layer."""
        if not target_hidden:
            return list(current_hidden)
        curr = list(current_hidden)
        target = list(target_hidden)

        if len(curr) < len(target):
            new_size = target[len(curr)] if len(curr) < len(target) else target[-1]
            curr.append(new_size)
        elif len(curr) > len(target):
            curr.pop()

        step = 32
        for i in range(min(len(curr), len(target))):
            diff = target[i] - curr[i]
            if abs(diff) <= step:
                curr[i] += diff
            else:
                curr[i] += step if diff > 0 else -step
            curr[i] = int(np.clip(curr[i], self.hs_min, self.hs_max))
        return curr

    def _enforce_bounds(self, lr: float, hidden: List[int]) -> tuple[float, List[int]]:
        lr = float(np.clip(lr, self.lr_min, self.lr_max))
        hidden = [int(np.clip(int(x), self.hs_min, self.hs_max)) for x in hidden] or [
            int(self.hs_min)
        ]
        if len(hidden) < self.num_layers_min:
            hidden = [hidden[0]] * self.num_layers_min
        if len(hidden) > self.num_layers_max:
            hidden = hidden[: self.num_layers_max]
        return lr, hidden

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def apply_critic_feedback(
        self,
        critic_output: Dict[str, Any],
        tracker: BestConfigTracker,
        cycle: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mutate ``self.current_config`` into the next configuration to test.

        Returns the new configuration (also stored on the agent).
        """
        orig_lr = self.current_config["learning_rate"]
        orig_hidden = list(self.current_config["hidden_layers"])
        orig_drop = self.current_config.get("dropout_rate", 0.0)
        orig_wd = self.current_config.get("weight_decay", 0.0001)
        orig_batch = self.current_config.get("batch_size", cfg.BATCH_SIZE)
        orig_pat = self.current_config.get("patience", cfg.EARLY_STOP_PATIENCE)

        adaptive_reg = bool(getattr(cfg, "ADAPTIVE_REGULARIZATION", True))
        critic_diagnosis = critic_output.get("diagnosis", "UNKNOWN")

        reg_block = (
            block("actor", "reg_block_adaptive")
            if adaptive_reg
            else block("actor", "reg_block_locked")
        )
        format_block = (
            block("actor", "format_block_adaptive")
            if adaptive_reg
            else block("actor", "format_block_locked")
        )

        print(
            "        🧠 Actor Agent is analyzing historical failures and Critic reasoning "
            "to formulate precise values..."
        )

        prompt = render(
            "actor",
            orig_lr=orig_lr,
            orig_hidden=orig_hidden,
            orig_drop=orig_drop,
            orig_wd=orig_wd,
            orig_batch=orig_batch,
            orig_pat=orig_pat,
            best_lr=tracker.best_config["learning_rate"] if tracker.best_config else orig_lr,
            best_hs=tracker.best_config["hidden_layers"] if tracker.best_config else orig_hidden,
            best_reasoning=tracker.best_reasoning,
            failures_str=tracker.get_recent_failures_str(),
            critic_diagnosis=critic_diagnosis,
            critic_reasoning=critic_output.get("reasoning", "None"),
            critic_lr_dir=critic_output.get("lr_dir"),
            critic_lr_step=critic_output.get("lr_step"),
            critic_hidden=critic_output.get("hidden_layers"),
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            hs_min=self.hs_min,
            hs_max=self.hs_max,
            nl_min=self.num_layers_min,
            nl_max=self.num_layers_max,
            reg_block=reg_block,
            format_block=format_block,
        )

        raw = invoke_llm(
            system_prompt("actor"),
            prompt,
            agent_name="Actor Application Agent",
            cycle=cycle,
            log_filename=self.log_filename,
        )

        if raw.strip():
            try:
                new_lr, new_hidden = orig_lr, orig_hidden
                new_drop, new_wd = orig_drop, orig_wd
                new_batch, new_pat = orig_batch, orig_pat

                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("LEARNING_RATE:"):
                        new_lr = float(line.split(":", 1)[1].strip())
                    elif line.startswith("HIDDEN_LAYERS:"):
                        parsed = json.loads(line.split(":", 1)[1].strip())
                        if isinstance(parsed, list) and parsed:
                            new_hidden = [int(x) for x in parsed]
                    elif line.startswith("DROPOUT_RATE:"):
                        new_drop = float(line.split(":", 1)[1].strip())
                    elif line.startswith("WEIGHT_DECAY:"):
                        new_wd = float(line.split(":", 1)[1].strip())
                    elif line.startswith("BATCH_SIZE:"):
                        new_batch = int(float(line.split(":", 1)[1].strip()))
                    elif line.startswith("PATIENCE:"):
                        new_pat = int(float(line.split(":", 1)[1].strip()))

                # 1. Enforce math bounds
                new_lr, new_hidden = self._enforce_bounds(new_lr, new_hidden)
                new_batch = max(16, min(256, new_batch))
                new_pat = max(10, min(100, new_pat))

                if adaptive_reg:
                    new_drop = float(np.clip(new_drop, cfg.DROPOUT_RATE_MIN, cfg.DROPOUT_RATE_MAX))
                    new_wd = float(np.clip(new_wd, 0.0, 0.1))
                else:
                    new_drop, new_wd = orig_drop, orig_wd

                # 2. Build the candidate on top of the static parameters
                candidate = dict(self.current_config)
                candidate["learning_rate"] = float(new_lr)
                candidate["hidden_layers"] = new_hidden
                candidate["dropout_rate"] = new_drop
                candidate["weight_decay"] = new_wd
                candidate["batch_size"] = new_batch
                candidate["patience"] = new_pat

                if self._is_new_config(candidate):
                    self.current_config = candidate
                    self._add_to_visited(candidate)
                    return self.current_config
            except Exception as exc:  # noqa: BLE001
                print(f"        ⚠️ Actor Agent LLM parse failed ({exc}). Falling back to heuristic math...")

        # --- FALLBACK: heuristic math + batch / patience adjustments --------
        target_hidden = critic_output.get("hidden_layers", orig_hidden)
        for attempt in range(int(cfg.MAX_RETRIES) + 1):
            if attempt == 0:
                lr = orig_lr
                new_hidden = self._incremental_hidden_change(orig_hidden, target_hidden)
                new_drop, new_wd = orig_drop, orig_wd
                new_batch, new_pat = orig_batch, orig_pat

                if "OVERFIT" in critic_diagnosis:
                    new_batch = max(16, new_batch // 2)
                    new_pat = max(10, new_pat - 5)
                    if adaptive_reg:
                        new_drop = float(np.clip(orig_drop + 0.05, 0.0, 0.5))
                        new_wd = float(
                            np.clip(orig_wd * 2.0 if orig_wd > 0 else 0.0001, 0.0, 0.05)
                        )
                elif "UNDERFIT" in critic_diagnosis:
                    new_pat = min(100, new_pat + 5)
                    if adaptive_reg:
                        new_drop = float(np.clip(orig_drop - 0.05, 0.0, 0.5))
                        new_wd = float(np.clip(orig_wd * 0.5, 0.0, 0.05))
            else:
                rand = self._random_exploration()
                lr = rand["learning_rate"]
                new_hidden = rand["hidden_layers"]
                new_drop = rand.get("dropout_rate", orig_drop) if adaptive_reg else orig_drop
                new_wd = rand.get("weight_decay", orig_wd) if adaptive_reg else orig_wd
                new_batch = rand.get("batch_size", orig_batch)
                new_pat = rand.get("patience", orig_pat)

            if attempt == 0 and critic_output.get("lr_dir") != "stay":
                if critic_output.get("lr_dir") == "increase":
                    lr += critic_output.get("lr_step", 0.0)
                else:
                    lr -= critic_output.get("lr_step", 0.0)

            lr, new_hidden = self._enforce_bounds(lr, new_hidden)
            new_batch = max(16, min(256, new_batch))
            new_pat = max(10, min(100, new_pat))

            candidate = dict(self.current_config)
            candidate["learning_rate"] = lr
            candidate["hidden_layers"] = new_hidden
            candidate["dropout_rate"] = new_drop
            candidate["weight_decay"] = new_wd
            candidate["batch_size"] = new_batch
            candidate["patience"] = new_pat

            if self._is_new_config(candidate):
                self.current_config = candidate
                self._add_to_visited(candidate)
                return self.current_config

        # --- LAST RESORT: bounded random search until something is new ------
        def _fresh_random() -> Dict[str, Any]:
            rand = self._random_exploration()
            if not adaptive_reg:
                rand["dropout_rate"] = orig_drop
                rand["weight_decay"] = orig_wd
            rand.setdefault("batch_size", orig_batch)
            rand.setdefault("patience", orig_pat)
            return rand

        rand = _fresh_random()
        guard = 0
        while not self._is_new_config(rand) and guard < 100:
            rand = _fresh_random()
            guard += 1

        self.current_config = rand
        self._add_to_visited(rand)
        return self.current_config

    # ------------------------------------------------------------------
    def propose(
        self,
        current_config: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[Dict[str, Any]] = None,
        cycle: Optional[int] = None,
        tracker: Optional[BestConfigTracker] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """
        Stateless-looking wrapper around :meth:`apply_critic_feedback`.

        Kept so orchestrators (API / Streamlit adapters) can call
        ``actor.propose(...)`` without knowing about the agent's internal state.
        """
        if current_config:
            self.current_config = dict(current_config)
        if tracker is None:
            tracker = BestConfigTracker()
        return self.apply_critic_feedback(critic_feedback or {}, tracker, cycle=cycle)
