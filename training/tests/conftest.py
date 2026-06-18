"""Pytest configuration shared by ``ai/training/tests``.

This conftest enforces two isolation guarantees so the suite can run as a
whole (``pytest ai/training/tests``) without breaking the production
pilot imports:

1.  Some sibling test files (``test_dpo_integration.py``,
    ``test_grpo_integration.py``) deliberately overwrite
    ``sys.modules["peft"]`` (and ``sys.modules["trl"]``) with a
    ``MagicMock`` at *module import time* so they can simulate trl/peft
    without pulling in the heavy ML deps.  When pytest later walks the
    alphabetically-sorted test directory, ``test_pixelated_production_pilot.py``
    (and ``test_production_pilot.py``) try to do
    ``from training.pixelated_production_pilot import _build_arg_parser``,
    which in turn imports ``transformers``.  ``transformers.trainer_utils``
    calls ``is_peft_available()`` -> ``importlib.util.find_spec("peft")``.
    At that point ``sys.modules["peft"]`` is the ``MagicMock`` and
    ``importlib.util.find_spec`` raises ``ValueError:
    peft.__spec__ is not set`` because ``MagicMock`` deliberately raises
    ``AttributeError`` for dunder attribute access (e.g. ``__spec__``).

    We work around that by stashing the real ``peft`` (and ``trl``,
    ``transformers``) modules on conftest load, then restoring them at
    every ``pytest_collectstart`` boundary so subsequent modules get the
    real package even if earlier modules stomped it.

2.  The ``kernels`` package (only versions <0.13 are pinned in
    ``pyproject.toml``) defensively patches
    ``kernels.layer.layer.LayerRepository.__init__`` and
    ``kernels.layer.func.FuncRepository.__init__`` to default
    ``version=1`` when neither ``revision`` nor ``version`` is supplied.
    This mirrors the same defense applied in
    ``test_dpo_integration.py`` / ``test_grpo_integration.py`` so every
    conftest-managed test benefits from it without each test having to
    replicate the boilerplate.
"""

from __future__ import annotations

import contextlib
import sys
from unittest.mock import MagicMock as _MagicMock

# ---------------------------------------------------------------------------
# Step 1: stash real heavy-ML modules so we can restore them later.
# ---------------------------------------------------------------------------
# IMPORTANT: We explicitly trigger the import of peft/trl/transformers
# here so that the real (non-mocked) packages are guaranteed to be in
# sys.modules by this point. Without forcing the import, peft may not
# have been touched yet, leaving us nothing to restore once the dirty
# sibling test files mutate sys.modules.

STASH_REAL_MODULES = ("peft", "trl", "transformers")


def _stash_if_real(name: str) -> None:
    mod = sys.modules.get(name)
    if mod is None:
        return
    if isinstance(mod, _MagicMock):
        return
    if not hasattr(mod, "__spec__"):
        return
    if mod.__spec__ is None:
        return
    sys.modules[f"__real_{name}__"] = mod


def _ensure_real_stash() -> None:
    """Lazily import real ``peft`` / ``trl`` / ``transformers`` and stash
    them if not already done.  Idempotent — safe to call repeatedly."""
    for name in STASH_REAL_MODULES:
        stash_key = f"__real_{name}__"
        if stash_key in sys.modules:
            continue
        with contextlib.suppress(Exception):
            __import__(name)
        _stash_if_real(name)


# Stash now (during conftest load) AND ensure_lazy for any module that
# was not yet imported by the time the conftest ran.
for _name in STASH_REAL_MODULES:
    _stash_if_real(_name)
_ensure_real_stash()


def _is_module_clobbered(module_name: str) -> bool:
    mod = sys.modules.get(module_name)
    if mod is None:
        return False
    if isinstance(mod, _MagicMock):
        return True
    if not hasattr(mod, "__spec__"):
        return True
    return mod.__spec__ is None


def _restore_real_modules() -> None:
    _ensure_real_stash()
    for name in STASH_REAL_MODULES:
        stash_key = f"__real_{name}__"
        if stash_key not in sys.modules:
            continue
        if not _is_module_clobbered(name):
            continue
        sys.modules[name] = sys.modules[stash_key]


# ---------------------------------------------------------------------------
# Step 2: defensive kernels.LayerRepository/FuncRepository shim.
# ---------------------------------------------------------------------------


class _KernelsShimState:
    """Module-private singleton so we don't need a module-level mutable
    flag (which ruff's PLW0603 discourages)."""

    applied = False


def _safe_patch_repository(cls: type) -> None:
    """Wrap ``cls.__init__`` to default ``version=1`` when neither
    ``revision`` nor ``version`` is supplied.

    Mirrors the same defense applied in
    ``test_dpo_integration.py`` / ``test_grpo_trainer.py``.  Idempotent:
    a callable tagged with ``_kernels_shim_applied`` is treated as
    already-wrapped.
    """
    try:
        orig = cls.__init__
    except (AttributeError, TypeError):
        return
    if not callable(orig):
        return
    if getattr(orig, "_kernels_shim_applied", False):
        return

    def _init(self, *args, **kwargs):
        if "revision" not in kwargs and "version" not in kwargs:
            kwargs["version"] = 1
        return orig(self, *args, **kwargs)

    _init._kernels_shim_applied = True  # type: ignore[attr-defined]
    try:
        cls.__init__ = _init  # type: ignore[misc]
    except (AttributeError, TypeError):
        # Slot descriptors cannot be reassigned.
        return


def _apply_kernels_shim() -> None:
    """Idempotent.  Patch LayerRepository/FuncRepository only — never
    unrelated classes that happen to live in the same module."""
    if _KernelsShimState.applied:
        return
    target_modules = {
        "kernels.layer.layer": ("LayerRepository", "LocalLayerRepository", "LockedLayerRepository"),
        "kernels.layer.func": ("FuncRepository", "LocalFuncRepository", "LockedFuncRepository"),
    }
    for mod_name, candidates in target_modules.items():
        with contextlib.suppress(Exception):
            mod = __import__(mod_name, fromlist=["Repository"])
            for cls_name in candidates:
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    _safe_patch_repository(cls)
    _KernelsShimState.applied = True


with contextlib.suppress(Exception):
    _apply_kernels_shim()


# ---------------------------------------------------------------------------
# Step 3: pytest hooks — restore clobbered modules at every collector
# boundary so subsequent imports see the real package.
# ---------------------------------------------------------------------------


def pytest_collectstart(collector) -> None:  # noqa: ARG001  (required by pytest)
    """Restore clobbered ML modules just before each Module collector begins.

    ``pytest_collectstart`` fires immediately before the Module's
    ``collect()`` is invoked, which is when pytest actually imports the
    test module.  By restoring right before that, we guarantee the
    import-time top-level code of the test module (and any modules it
    imports) sees the genuine ``peft`` / ``trl`` / ``transformers``
    packages — all of which rely on a real ``__spec__`` to function.
    """
    _restore_real_modules()
    _apply_kernels_shim()
