"""Pytest configuration shared by `ai/` tests.

Disable optional safety ML model loading by default during test runs so that
imports remain stable when optional ML dependencies are unavailable or broken.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("AI_DISABLE_SAFETY_ML_MODELS", "1")
os.environ.setdefault("BIAS_DETECTION_DISABLE_SENTRY", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
elif sys.path[0] != str(_REPO_ROOT):
    sys.path.remove(str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT))

import ai as _ai_package

if _ai_package.__file__ is None:
    raise RuntimeError(
        f"ai resolved to a namespace package ({list(_ai_package.__path__)}); "
        "a test module put a shadowing directory ahead of the repository root on sys.path"
    )

# Auth fixtures validate AUTH_SECRET_KEY at import time and reject placeholder
# values, so a per-run generated secret is supplied when the environment has
# none. It is generated, never a literal.
if not os.environ.get("AUTH_SECRET_KEY"):
    import secrets

    os.environ["AUTH_SECRET_KEY"] = secrets.token_hex(32)

# The dataset API validates its DSN at import time. Tests patch the database
# layer, so a loopback DSN that is never dialled is enough to import.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
