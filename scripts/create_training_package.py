"""Alias module: :mod:`scripts.create_training_package` and :mod:`ai.tools.scripts.create_training_package` are one module.

The implementation lives in ``ai/tools/scripts/create_training_package.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import create_training_package as _impl

sys.modules[__name__] = _impl
