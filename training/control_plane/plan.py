"""Plan resolution for training jobs.

``plan_job`` is side-effect free: it validates inputs/overrides against a
registered recipe and returns the exact resolved spec (commands, cost, artifact
paths, warnings). The returned ``plan_hash`` is what ``submit_job`` binds to.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from typing import Any

from training.control_plane import db, recipes
from training.control_plane.config import Settings


class PlanError(Exception):
    pass


#: Recipe inputs are single tokens — tags, digests, file paths. A value with
#: whitespace or shell/CLI metacharacters cannot be anything a reviewed
#: entrypoint expects, so it is rejected at plan time; ``_resolve_command``
#: additionally quotes every substitution.
_INPUT_TOKEN = re.compile(r"^[A-Za-z0-9._@:/+=][A-Za-z0-9._@:/+=-]*$")


def _validate_inputs(recipe: recipes.Recipe, inputs: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name in recipe.inputs:
        if name not in inputs or inputs.get(name) in (None, ""):
            raise PlanError(f"missing required input: {name}")
    for name, value in inputs.items():
        if name not in recipe.inputs:
            # Fail closed: a caller that passes an input the recipe does not
            # declare believes it changed the outcome. Ignoring it silently
            # would launch work under a wrong assumption (the suite-version bug
            # class), so the plan is rejected instead.
            raise PlanError(f"unknown input '{name}' is not accepted by recipe {recipe.id}")
        if not isinstance(value, str) or _INPUT_TOKEN.fullmatch(value) is None:
            raise PlanError(
                f"input '{name}' must be a single token of letters, digits and ._-/@+=: "
                "spaces, quotes and shell characters are rejected"
            )
    return warnings


def _check_bounds(spec: dict[str, Any], name: str, value: float) -> None:
    """Enforce a recipe's min/max on an override, failing closed on a bad bound.

    Recipe bounds are validated at load time, but a bound that is not a
    finite number here must refuse the plan rather than crash the comparison.
    """
    for bound in ("min", "max"):
        if bound not in spec:
            continue
        limit = spec[bound]
        if isinstance(limit, bool) or not isinstance(limit, int | float) or not math.isfinite(limit):
            raise PlanError(f"override '{name}' {bound} must be a finite number")
        if bound == "min" and value < limit:
            raise PlanError(f"override '{name}' below min {limit}")
        if bound == "max" and value > limit:
            raise PlanError(f"override '{name}' above max {limit}")


def _validate_overrides(recipe: recipes.Recipe, overrides: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name, value in (overrides or {}).items():
        spec = recipe.overrides.get(name)
        if spec is None:
            raise PlanError(f"override '{name}' not allowed by recipe {recipe.id}")
        # The plan cache stores the resolved overrides and submit re-plans with
        # them, so a locked override that names its own default must survive
        # that round-trip instead of locking the submit out. The type check
        # keeps 1/True-style equalities from unlocking a bool default with an
        # int value, which would resolve to a different command.
        if spec.get("locked") and (
            "default" not in spec or type(value) is not type(spec["default"]) or value != spec["default"]
        ):
            raise PlanError(f"override '{name}' is locked by recipe {recipe.id}")
        otype = spec.get("type")
        if otype == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise PlanError(f"override '{name}' must be int")
            _check_bounds(spec, name, value)
        elif otype == "float":
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise PlanError(f"override '{name}' must be number")
            if not math.isfinite(value):
                raise PlanError(f"override '{name}' must be a finite number")
            _check_bounds(spec, name, value)
        elif otype == "str":
            if not isinstance(value, str):
                raise PlanError(f"override '{name}' must be string")
        elif otype == "bool" and not isinstance(value, bool):
            raise PlanError(f"override '{name}' must be a boolean")
    return warnings


def effective_overrides(recipe: recipes.Recipe, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """The overrides a plan records: the caller's values plus every recipe default.

    ``plan_job`` stores this expanded dict in the spec, so a submit that repeats
    only the keys the caller supplied has to expand the same way before it can be
    compared with the approved plan.
    """
    expanded = dict(overrides or {})
    for name, spec in recipe.overrides.items():
        if name not in expanded and "default" in spec:
            expanded[name] = spec["default"]
    return expanded


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
        # Quote so a value containing spaces or flags cannot split the
        # argument list or inject an unintended CLI option.
        return shlex.quote(str(value))

    resolved = re.sub(r"<([A-Za-z0-9_\-]+)>", repl, cmd)
    try:
        # Shell-aware normalization: whitespace between arguments collapses,
        # but repeated whitespace inside a quoted value survives, and the
        # quoting itself stays canonical — the same parse the runner performs
        # when it builds the argv.
        return shlex.join(shlex.split(resolved))
    except ValueError:
        # Unbalanced quotes cannot be parsed; fall back to plain whitespace
        # normalization rather than failing the plan.
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
    overrides = effective_overrides(recipe, overrides)

    command = _resolve_command(recipe, inputs, overrides)
    if "<" in command and ">" in command:
        warnings.append("entrypoint still contains unresolved placeholders")
    for name in recipe.inputs:
        if f"<{name}>" not in recipe.entrypoint:
            warnings.append(
                f"input '{name}' is recorded in the spec but not passed to the entrypoint; "
                "the recipe reads it from its own defaults"
            )
    for name in overrides:
        if f"<{name}>" not in recipe.entrypoint:
            warnings.append(
                f"override '{name}' is recorded in the spec but not passed to the entrypoint; "
                "the value does not change the command"
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
