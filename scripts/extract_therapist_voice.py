"""Alias module: :mod:`scripts.extract_therapist_voice` and :mod:`ai.tools.scripts.extract_therapist_voice` are one module.

The implementation lives in ``ai/tools/scripts/extract_therapist_voice.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import extract_therapist_voice as _impl

sys.modules[__name__] = _impl
