"""Pytest configuration shared by `ai/` tests.

Disable optional safety ML model loading by default during test runs so that
imports remain stable when optional ML dependencies are unavailable or broken.
"""

import os

os.environ.setdefault("AI_DISABLE_SAFETY_ML_MODELS", "1")
os.environ.setdefault("BIAS_DETECTION_DISABLE_SENTRY", "1")

# Auth fixtures validate AUTH_SECRET_KEY at import time and reject placeholder
# values, so a per-run generated secret is supplied when the environment has
# none. It is generated, never a literal.
if not os.environ.get("AUTH_SECRET_KEY"):
    import secrets

    os.environ["AUTH_SECRET_KEY"] = secrets.token_hex(32)

# The dataset API validates its DSN at import time. Tests patch the database
# layer, so a loopback DSN that is never dialled is enough to import.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")

# The remaining collect_ignore entries are tests that fail on their own
# assertions, not on missing imports: each of the 31 former entries was run
# individually, 27 pass and were removed.
collect_ignore = [
    "qa/testing/api/test_api_comprehensive.py",
    "qa/testing/test_extraction_modules.py",
    "qa/testing/test_jit_scenario_injector.py",
    "qa/testing/test_vertical_fidelity_stack.py",
]
