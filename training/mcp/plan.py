"""Plan resolution for training jobs.

``plan_job`` is side-effect free: it validates inputs/overrides against a
registered recipe and returns the exact resolved spec (commands, cost, artifact
paths, warnings). The returned ``plan_hash`` is what ``submit_job`` binds to.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from training.mcp import db, recipes
from training.mcp.config import Settings


class PlanError(Exception):
    pass


def _validate_inputs(recipe: recipes.Recipe, inputs: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name in recipe.inputs:
        if name not in inputs or inputs.get(name) in (None, ""):
            raise PlanError(f"missing required input: {name}")
    for name in inputs:
        if name not in recipe.inputs:
            # Fail closed: a caller that passes an input the recipe does not
            # declare believes it changed the outcome. Ignoring it silently
            # would launch work under a wrong assumption (the suite-version bug
            # class), so the plan is rejected instead.
            raise PlanError(f"unknown input '{name}' is not accepted by recipe {recipe.id}")
    return warnings


def _validate_overrides(recipe: recipes.Recipe, overrides: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name, value in (overrides or {}).items():
        spec = recipe.overrides.get(name)
        if spec is None:
            raise PlanError(f"override '{name}' not allowed by recipe {recipe.id}")
        if spec.get("locked"):
            raise PlanError(f"override '{name}' is locked by recipe {recipe.id}")
        otype = spec.get("type")
        if otype == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise PlanError(f"override '{name}' must be int")
            if "min" in spec and value < spec["min"]:
                raise PlanError(f"override '{name}' below min {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise PlanError(f"override '{name}' above max {spec['max']}")
        elif otype == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PlanError(f"override '{name}' must be number")
        elif otype == "str":
            if not isinstance(value, str):
                raise PlanError(f"override '{name}' must be string")
        elif otype == "bool":
            if not isinstance(value, bool):
                raise PlanError(f"override '{name}' must be a boolean")
    # Defaults for missing non-locked overrides.
    for name, spec in recipe.overrides.items():
        if name not in (overrides or {}) and "default" in spec and not spec.get("locked"):
            overrides[name] = spec["default"]
    return warnings


def _bool_flag(recipe: recipes.Recipe, name: str) -> str:
    """The CLI flag a boolean override named ``name`` substitutes to.

    Derived from the override name (``dry_run`` -> ``--dry-run``) rather than
    hardcoded: the old code turned *every* truthy bool into the literal
    ``--dry-run``, so a recipe with two boolean overrides rendered the same
    flag twice and silently dropped the other. A recipe can still name its
    flag explicitly with ``flag: --verbose``.
    """
    spec = recipe.overrides.get(name) or {}
    declared = spec.get("flag")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return "--" + name.replace("_", "-")


def _resolve_command(recipe: recipes.Recipe, inputs: dict[str, Any], overrides: dict[str, Any]) -> str:
    """Substitute ``<name>`` placeholders from inputs, then overrides.

    A boolean value substitutes as a CLI flag named after the override:
    ``dry_run: True`` becomes ``--dry-run`` and ``False`` becomes an empty
    string, so a boolean override can add or omit a flag without leaving a
    stray placeholder. A placeholder with no value is left intact so the
    caller can warn about it rather than silently dropping an argument.
    """
    cmd = recipe.entrypoint
    values = {**overrides, **inputs}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in values:
            return match.group(0)
        value = values[key]
        if value in (None, ""):
            return ""
        if isinstance(value, bool):
            return _bool_flag(recipe, key) if value else ""
        return str(value)

    resolved = re.sub(r"<([A-Za-z0-9_\-]+)>", repl, cmd)
    return " ".join(resolved.split())


def plan_job(
    settings: Settings, recipe_id: str, inputs: dict[str, Any] | None, overrides: dict[str, Any] | None
) -> dict[str, Any]:
    inputs = dict(inputs or {})
    overrides = dict(overrides or {})
    recipe = None
    for r in recipes.load_recipes():
        if r.id == recipe_id:
            recipe = r
            break
    if recipe is None:
        raise PlanError(f"unknown recipe: {recipe_id}")

    warnings = _validate_inputs(recipe, inputs)
    warnings += _validate_overrides(recipe, overrides)

    # Locked defaults (e.g. derived learning_rate).
    for name, spec in recipe.overrides.items():
        if spec.get("locked") and "default" in spec and name not in overrides:
            overrides[name] = spec["default"]

    command = _resolve_command(recipe, inputs, overrides)
    if "<" in command and ">" in command:
        warnings.append("entrypoint still contains unresolved placeholders")
    for name in recipe.inputs:
        if f"<{name}>" not in recipe.entrypoint:
            warnings.append(
                f"input '{name}' is recorded in the spec but not passed to the entrypoint; "
                "the recipe reads it from its own defaults"
            )
    if recipe.kind == "eval_paired" and not recipe.invariants.get("r2_only"):
        warnings.append("eval recipe missing r2_only invariant")

    spec = {
        "recipe_id": recipe.id,
        "kind": recipe.kind,
        "backend": recipe.backend,
        "command": command,
        "inputs": inputs,
        "overrides": overrides,
        "cost_class": recipe.cost_class,
        "est_cost_usd": recipe.est_cost_usd,
        "gpu": recipe.gpu,
        "timeout": recipe.timeout,
        "outputs": recipe.outputs,
        "invariants": recipe.invariants,
    }
    artifact_paths = dict(recipe.outputs or {})
    spec["artifact_paths"] = artifact_paths

    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    plan_hash = hashlib.sha256((spec_hash + recipe.id).encode()).hexdigest()

    db.cache_plan(settings.db_path, plan_hash, recipe.id, spec)

    return {
        "recipe": recipe.to_dict(),
        "spec": spec,
        "spec_hash": spec_hash,
        "plan_hash": plan_hash,
        "est_cost_usd": recipe.est_cost_usd,
        "cost_class": recipe.cost_class,
        "gpu": recipe.gpu,
        "timeout": recipe.timeout,
        "command": command,
        "artifact_paths": artifact_paths,
        "warnings": warnings,
    }
