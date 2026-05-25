#!/usr/bin/env python3
"""Regression tests for the clinical validity dashboard."""

import os
import sys
import types
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from clinical_validity_dashboard import ClinicalValidityDashboard


def test_calculate_current_metrics_with_recent_data():
    dashboard = ClinicalValidityDashboard(db_path=":memory:")
    now = datetime.now(UTC)

    df = pd.DataFrame(
        [
            {
                "timestamp": now,
                "clinical_validity_score": 0.75,
                "sample_size": 120,
                "pipeline_stage": "validation",
                "data_source": "synthetic",
                "notes": "latest",
            },
            {
                "timestamp": now - timedelta(minutes=30),
                "clinical_validity_score": 0.70,
                "sample_size": 115,
                "pipeline_stage": "validation",
                "data_source": "synthetic",
                "notes": "recent",
            },
            {
                "timestamp": now - timedelta(hours=3),
                "clinical_validity_score": 0.55,
                "sample_size": 110,
                "pipeline_stage": "validation",
                "data_source": "synthetic",
                "notes": "older",
            },
        ]
    )

    metrics = dashboard.calculate_current_metrics(df)

    assert metrics["current_score"] == 0.75
    assert metrics["sample_size"] == 120
    assert metrics["trend_1h"] == pytest.approx(0.05)
    assert metrics["trend_24h"] == pytest.approx(0.20)
    assert metrics["valid_samples_24h"] == 2
    assert metrics["total_samples_24h"] == 3
    assert metrics["clinical_validity_rate"] == pytest.approx(66.66666666666666)
    assert metrics["status"] == "good"


def test_calculate_current_metrics_with_empty_data():
    dashboard = ClinicalValidityDashboard(db_path=":memory:")

    metrics = dashboard.calculate_current_metrics(pd.DataFrame())

    assert metrics == {
        "current_score": 0.0,
        "sample_size": 0,
        "trend_1h": 0.0,
        "trend_24h": 0.0,
        "valid_samples_24h": 0,
        "total_samples_24h": 0,
        "clinical_validity_rate": 0.0,
        "status": "no_data",
    }
