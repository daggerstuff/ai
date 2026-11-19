"""Recipe registry loader.

Recipes are reviewed, versioned entries in ``recipes.yaml`` — agents select
them, never define them. Validation happens at load: a malformed recipe
fails fast at service start rather than mid-submit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RECIPES_PATH = Path(__file__).resolve().with_name("recipes.yaml")

KINDS = {"curate_dataset", "train_sft", "eval_paired", "build_release", "other"}
BACKENDS = {"local", "ovhai", "wandb-serverless", "wandb", "r2"}
COST_CLASSES = {"cpu-minute", "cpu-hour", "gpu-hour", "gpu-day"}
REQUIRED_FIELDS = ("id", "kind", "description", "backend", "entrypoint")


@dataclass(frozen=True)
class Recipe:
    id: str
    kind: str
    description: str
    backend: str
    entrypoint: str
    cost_class: str = "cpu-minute"
    est_cost_usd: float = 0.0
    gpu: str = "none"
    timeout: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "backend": self.backend,
            "entrypoint": self.entrypoint,
            "cost_class": self.cost_class,
            "est_cost_usd": self.est_cost_usd,
            "gpu": self.gpu,
            "timeout": self.timeout,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


def load_recipes(path: Path | None = None) -> list[Recipe]:
    source = path or RECIPES_PATH
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    entries = raw.get("recipes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{source} has no recipes")
    recipes: list[Recipe] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"recipe #{index} is not a mapping")
        missing = [name for name in REQUIRED_FIELDS if not entry.get(name)]
        if missing:
            raise ValueError(f"recipe #{index} missing fields: {', '.join(missing)}")
        if entry["kind"] not in KINDS:
            raise ValueError(f"recipe {entry['id']!r}: unknown kind {entry['kind']!r}")
        if entry["backend"] not in BACKENDS:
            raise ValueError(f"recipe {entry['id']!r}: unknown backend {entry['backend']!r}")
        cost_class = entry.get("cost_class", "cpu-minute")
        if cost_class not in COST_CLASSES:
            raise ValueError(f"recipe {entry['id']!r}: unknown cost_class {cost_class!r}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate recipe id {entry['id']!r}")
        seen.add(entry["id"])
        recipes.append(
            Recipe(
                id=entry["id"],
                kind=entry["kind"],
                description=str(entry["description"]).strip(),
                backend=entry["backend"],
                entrypoint=str(entry["entrypoint"]).strip(),
                cost_class=cost_class,
                est_cost_usd=float(entry.get("est_cost_usd", 0.0)),
                gpu=str(entry.get("gpu", "none")),
                timeout=str(entry.get("timeout", "")),
                inputs=dict(entry.get("inputs") or {}),
                outputs=dict(entry.get("outputs") or {}),
            )
        )
    return recipes
