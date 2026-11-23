"""Alias module: :mod:`scripts.generate_nightmare_fuel_5k` and :mod:`ai.tools.scripts.generate_nightmare_fuel_5k` are one module.

The implementation lives in ``ai/tools/scripts/generate_nightmare_fuel_5k.py``. Replacing the entry in
``sys.modules`` keeps a single module object, so module-level state (feature flags,
caches) is shared and patching either import path affects the other.
"""

import sys

from ai.tools.scripts import generate_nightmare_fuel_5k as _impl

sys.modules[__name__] = _impl
