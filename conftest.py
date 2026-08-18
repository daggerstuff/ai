"""Pytest configuration shared by `ai/` tests.

Disable optional safety ML model loading by default during test runs so that
imports remain stable when optional ML dependencies are unavailable or broken.
"""

import os

os.environ.setdefault("AI_DISABLE_SAFETY_ML_MODELS", "1")
os.environ.setdefault("BIAS_DETECTION_DISABLE_SENTRY", "1")
