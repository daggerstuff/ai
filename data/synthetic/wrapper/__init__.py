"""Compatibility wrapper for the restored training_corpus package."""

from __future__ import annotations

import importlib
import sys

try:
    _training_corpus = importlib.import_module("training_corpus")
except ModuleNotFoundError:
    # The legacy top-level package is gone; the pal_framework subpackage below
    # is self-contained and must remain importable.
    _training_corpus = None
if _training_corpus is not None:
    _public_names = getattr(_training_corpus, "__all__", None)
    if _public_names is None:
        _public_names = [name for name in dir(_training_corpus) if not name.startswith("_")]

    for name in _public_names:
        globals()[name] = getattr(_training_corpus, name)

    __all__ = list(_public_names)
else:
    __all__ = []

_SUBMODULES = (
    "benchmarks",
    "builder",
    "compare",
    "compose",
    "delta_package",
    "expansion",
    "expansion_authoring",
    "expansion_drafts",
    "expansion_queue",
    "experiments",
    "governance",
    "merge_package",
    "model",
    "normalize",
    "quality",
    "rubrics",
    "seed_package",
    "source_inventory",
    "sources",
    "synthesis",
    "wave1_package",
    "wave2_package",
    "wave3_package",
    "wave4_package",
    "writer",
)

_SOURCE_PACKAGE = "ai.data.synthetic"

for submodule_name in _SUBMODULES:
    try:
        module = importlib.import_module(f"{_SOURCE_PACKAGE}.{submodule_name}")
    except ModuleNotFoundError:
        continue
    sys.modules[f"{__name__}.{submodule_name}"] = module
