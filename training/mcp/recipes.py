"""Recipe registry loader.

Recipes are reviewed, versioned entries in ``recipes.yaml`` — agents select
them, never define them. Validation happens at load: a malformed recipe
fails fast at service start rather than mid-submit.

P3.3: ``validate_recipe_entry`` exposes the same checks for a *draft* entry so
an advisor can propose a recipe and a human can open the PR that lands it
(see ``training/mcp/RECIPE_WORKFLOW.md``). The MCP never writes ``recipes.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RECIPES_PATH = Path(__file__).resolve().with_name("recipes.yaml")

KINDS = {"curate_dataset", "train_sft", "eval_paired", "build_release", "other"}
BACKENDS = {"local", "ovhai", "wandb-serverless", "wandb", "r2"}
COST_CLASSES = {"cpu-minute", "cpu-hour", "gpu-hour", "gpu-day"}
REQUIRED_FIELDS = ("id", "kind", "description", "backend", "entrypoint")

# Kinds that launch real, billable remote work. Their entrypoints take an
# ``--execute`` flag and default to a self-check/dry run when it is absent, so
# a recipe that omits it would silently "succeed" without launching anything
# while spending the operator's trust (and a daily-cap slot). The registry
# refuses such a recipe rather than letting a submit report a fake success.
LAUNCH_KINDS = frozenset({"train_sft", "eval_paired"})

# P3.4: multi-node / multi-GPU training is deliberately deferred. The decision
# and its rationale live in ``training/mcp/MULTI_NODE_DECISION.md``; the registry
# enforces it so a recipe cannot quietly reintroduce an unsupported shape.
MULTI_NODE_KEYS = ("nodes", "num_nodes", "multi_node", "world_size")


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
    overrides: dict[str, Any] = field(default_factory=dict)
    invariants: dict[str, Any] = field(default_factory=dict)

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
            "overrides": self.overrides,
            "invariants": self.invariants,
        }


def validate_recipe_entry(entry: Any, known_ids: set[str] | None = None) -> list[str]:
    """Return the list of validation errors for one draft recipe entry.

    An empty list means the entry passes exactly the checks ``load_recipes``
    applies at service start. ``known_ids`` lets a caller surface duplicate-id
    conflicts against the current registry.
    """
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["recipe entry is not a mapping"]
    missing = [name for name in REQUIRED_FIELDS if not entry.get(name)]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    recipe_id = entry.get("id")
    if isinstance(recipe_id, str) and known_ids and recipe_id in known_ids:
        errors.append(f"duplicate recipe id {recipe_id!r}")
    kind = entry.get("kind")
    if kind and (not isinstance(kind, str) or kind not in KINDS):
        errors.append(f"unknown kind {kind!r} (expected one of {sorted(KINDS)})")
    backend = entry.get("backend")
    if backend and (not isinstance(backend, str) or backend not in BACKENDS):
        errors.append(f"unknown backend {backend!r} (expected one of {sorted(BACKENDS)})")
    cost_class = entry.get("cost_class", "cpu-minute")
    if not isinstance(cost_class, str) or cost_class not in COST_CLASSES:
        errors.append(f"unknown cost_class {cost_class!r} (expected one of {sorted(COST_CLASSES)})")
    if "est_cost_usd" in entry:
        try:
            float(entry["est_cost_usd"])
        except TypeError, ValueError:
            errors.append(f"est_cost_usd must be numeric, got {entry['est_cost_usd']!r}")
    overrides = entry.get("overrides") or {}
    if not isinstance(overrides, dict):
        errors.append("overrides must be a mapping")
    else:
        for name, spec in overrides.items():
            if not isinstance(spec, dict):
                errors.append(f"override {name!r} must be a mapping")
            elif not spec.get("locked") and "type" not in spec:
                errors.append(f"override {name!r} must declare a type unless locked")
    errors.extend(_multi_node_errors(entry))
    errors.extend(_launch_flag_errors(entry))
    return errors


def _launch_flag_errors(entry: dict[str, Any]) -> list[str]:
    """Refuse a launching recipe whose entrypoint forgot its ``--execute``.

    ``wandb_sft_v15_execute`` and ``ovh_v22_eval.submit run`` both default to a
    dry run when ``--execute`` is absent. A recipe missing that flag would
    submit successfully, report ``succeeded``, and never launch anything — the
    exact fabricated-success failure this control plane exists to prevent.
    """
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind not in LAUNCH_KINDS:
        return []
    entrypoint = entry.get("entrypoint")
    if not isinstance(entrypoint, str):
        return []
    if "--execute" in entrypoint:
        return []
    return [
        f"kind {entry['kind']!r} launches remote work but entrypoint omits --execute "
        "(it would silently dry-run and never start a job)"
    ]


def _multi_node_errors(entry: dict[str, Any]) -> list[str]:
    """P3.4 guard: reject multi-node shapes the control plane does not support."""
    problems: list[str] = []
    for key in MULTI_NODE_KEYS:
        if key not in entry:
            continue
        value = entry[key]
        if key == "multi_node" and bool(value):
            problems.append(f"{key}=true is unsupported (P3.4 defers multi-node)")
        elif key != "multi_node" and isinstance(value, int | float) and value > 1:
            problems.append(f"{key}={value} is unsupported (P3.4 defers multi-node)")
    gpu = str(entry.get("gpu", ""))
    if "multi" in gpu.lower():
        problems.append(f"gpu={gpu!r} looks multi-GPU; P3.4 defers it")
    else:
        # OVH flavours read <model>-<count>-gpu (e.g. l4-1-gpu is one GPU).
        match = re.search(r"(?:^|-)(\d+)-gpu$", gpu.lower())
        if match and int(match.group(1)) > 1:
            problems.append(f"gpu={gpu!r} requests {match.group(1)} GPUs; P3.4 defers multi-GPU")
    if problems:
        problems.append("see training/mcp/MULTI_NODE_DECISION.md")
    return problems


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
        errors = validate_recipe_entry(entry, known_ids=seen)
        if errors:
            raise ValueError(f"recipe #{index} ({entry.get('id', 'no-id')}): {'; '.join(errors)}")
        seen.add(entry["id"])
        recipes.append(
            Recipe(
                id=entry["id"],
                kind=entry["kind"],
                description=str(entry["description"]).strip(),
                backend=entry["backend"],
                entrypoint=str(entry["entrypoint"]).strip(),
                cost_class=entry.get("cost_class", "cpu-minute"),
                est_cost_usd=float(entry.get("est_cost_usd", 0.0)),
                gpu=str(entry.get("gpu", "none")),
                timeout=str(entry.get("timeout", "")),
                inputs=dict(entry.get("inputs") or {}),
                outputs=dict(entry.get("outputs") or {}),
                overrides=dict(entry.get("overrides") or {}),
                invariants=dict(entry.get("invariants") or {}),
            )
        )
    return recipes
