"""Alias module: :mod:`scripts.synthetic_templates` and :mod:`ai.tools.scripts.synthetic_templates` are one module.

The implementation lives in ``ai/tools/scripts/synthetic_templates.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import synthetic_templates as _impl

sys.modules[__name__] = _impl
