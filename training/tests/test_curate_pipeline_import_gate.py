"""Import-gate tests for ``training.curate_pipeline``.

The curation pipeline has optional imports for ``training.annotation.iaa``
(PIX-4344/4345 IAA tier upgrade). That module may be absent from a
worktree, so the pipeline must remain importable and its base tier
classification must still work. These tests pin that behaviour by forcing
the IAA module to be unimportable via ``sys.modules`` mocking.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest


def _block_iaa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``training.annotation.iaa`` (and its parent pkg) unimportable.

    Patches ``sys.modules`` so that importing
    ``training.annotation.iaa`` raises ``ImportError`` even if the real
    module later appears on disk. Each blocked entry gets a ``None``
    sentinel which Python treats as "known-missing" for ``import``.
    """
    monkeypatch.setitem(sys.modules, "training.annotation", None)
    monkeypatch.setitem(sys.modules, "training.annotation.iaa", None)


def _force_reload_curate() -> Any:
    """Drop cached ``training.curate_pipeline`` so the next import re-runs."""
    sys.modules.pop("training.curate_pipeline", None)
    importlib = __import__("importlib")
    importlib.import_module("training.curate_pipeline")

    return sys.modules["training.curate_pipeline"]


def test_curate_pipeline_importable_without_iaa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing curate_pipeline must not raise when annotation.iaa is absent."""
    _block_iaa(monkeypatch)
    module = _force_reload_curate()
    # Optional import fell back to None sentinels, not the real symbols.
    assert module.AnnotationStage is None
    assert module.fleiss_kappa is None
    assert module.label_studio_export_to_iaa is None
    # Core API still present.
    assert callable(module.classify_tier)


def test_classify_tier_falls_through_without_iaa(monkeypatch: pytest.MonkeyPatch) -> None:
    """With IAA unavailable, the T1_GOLD override is skipped -> base tiers."

    - empty record -> T3_BRONZE
    - multi-turn annomi record -> T2_SILVER (not upgraded to T1_GOLD despite
      an annotation_stage/fleiss_kappa that *would* trigger the override
      if the IAA module were present).
    """
    _block_iaa(monkeypatch)
    module = _force_reload_curate()
    classify_tier = module.classify_tier

    # Empty record -> base bronze tier.
    assert classify_tier({"task_type": "", "messages": []}) == "T3_BRONZE"

    # Adversarial safety still wins regardless of IAA presence.
    assert classify_tier({"task_type": "adversarial_safety", "messages": []}) == "T4_SAFETY"

    # Clinically-reviewed -> T1_GOLD via the non-IAA path (unaffected).
    assert classify_tier({"clinical_reviewed": True, "messages": []}) == "T1_GOLD"

    # Multi-turn therapy dialogue -> T2_SILVER.
    messages = [
        {"role": "user", "content": str(i)} for i in range(5)
    ]
    assert (
        classify_tier({"task_type": "", "source": "annomi", "messages": messages})
        == "T2_SILVER"
    )

    # Record carrying IAA upgrade signals must NOT become T1_GOLD when the
    # IAA module is absent (override is gated behind AnnotationStage).
    record_with_iaa_signals = {
        "task_type": "",
        "source": "annomi",
        "messages": messages,
        "annotation_stage": "adjudicated",  # would-be upgrade trigger
        "fleiss_kappa": 0.9,  # would-be upgrade trigger
    }
    assert classify_tier(record_with_iaa_signals) == "T2_SILVER"
