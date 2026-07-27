"""Inference entry-point for the DeepRare multi-agent diagnostic pipeline.

This module provides the production-facing API that the FastAPI/AsyncAPI
layer calls when a clinician submits a patient case for rare-disease
diagnostic reasoning.

Usage
-----
```python
from ai.core.pipelines.inference.rare_disease_pipeline import (
    run_diagnostic_pipeline,
)

result = run_diagnostic_pipeline(patient_case)
top_dx = result["differential"][0]
```
"""

from __future__ import annotations

import logging
from typing import Any

from ai.platform.rare_disease.pipeline import build_default_pipeline
from ai.platform.rare_disease.types import PatientCase

logger = logging.getLogger(__name__)


def run_diagnostic_pipeline(case: PatientCase) -> dict[str, Any]:
    """Run the full DeepRare diagnostic pipeline on a single patient case.

    Parameters
    ----------
    case:
        Structured patient representation (symptoms, tests, family history).

    Returns
    -------
    dict with keys: case_id, differential, convergence_status, iterations,
    total_latency_ms, sub_agent_results.
    """
    logger.info(
        "Starting DeepRare diagnostic pipeline for case %s", case.case_id
    )
    pipeline = build_default_pipeline()
    result = pipeline.run(case)
    logger.info(
        "Pipeline complete — top diagnosis: %s (p=%.4f) in %.1f ms",
        result["differential"][0]["disease_name"]
        if result["differential"]
        else "none",
        result["differential"][0]["posterior_probability"]
        if result["differential"]
        else 0.0,
        result["total_latency_ms"],
    )
    return result


__all__ = ["run_diagnostic_pipeline"]
