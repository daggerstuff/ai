"""Alias module: :mod:`scripts.extraction_config` and :mod:`ai.tools.scripts.extraction_config` are one module.

The implementation lives in ``ai/tools/scripts/extraction_config.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import extraction_config as _impl

sys.modules[__name__] = _impl
