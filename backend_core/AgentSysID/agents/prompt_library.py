"""
Prompt library.

Agent prompts live as YAML next to the package (``prompts/*.yaml``) so they can
be reviewed and tuned without touching Python. Each file holds a ``system``
block, a ``user_template`` block, and optionally extra named fragments that the
agent selects at runtime (e.g. the Critic's explore / fine-tune phase blocks).

Templates use ``str.format`` placeholders; ``render`` fills them and tolerates a
missing key rather than crashing a long tuning run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class _SafeDict(dict):
    """Leaves unknown ``{placeholders}`` untouched instead of raising."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> Dict[str, Any]:
    """Load ``prompts/<name>.yaml`` as a dict of named blocks."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a mapping of blocks.")
    return data


def block(name: str, key: str, default: str = "") -> str:
    """Return one named block from a prompt file."""
    return str(load_prompt(name).get(key, default))


def render(name: str, key: str = "user_template", **values: Any) -> str:
    """Render a prompt block with ``values`` substituted."""
    template = block(name, key)
    try:
        return template.format_map(_SafeDict(values))
    except (ValueError, IndexError):
        # Format specs such as {x:.3f} on a non-numeric value: fall back to a
        # plain textual substitution so the run continues.
        out = template
        for k, v in values.items():
            out = out.replace("{" + str(k) + "}", str(v))
        return out


def system_prompt(name: str) -> str:
    """Return the ``system`` block of a prompt file."""
    return block(name, "system").strip()
