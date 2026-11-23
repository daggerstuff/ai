"""Alias module: :mod:`scripts.enhanced_conversion_pipeline` and :mod:`ai.tools.scripts.enhanced_conversion_pipeline` are one module.

The implementation lives in ``ai/tools/scripts/enhanced_conversion_pipeline.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import enhanced_conversion_pipeline as _impl

sys.modules[__name__] = _impl
