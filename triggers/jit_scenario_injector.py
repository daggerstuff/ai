"""Alias module: :mod:`triggers.jit_scenario_injector` is :mod:`ai.configs.jit_scenario_injector`.

The implementation lives in ``ai/configs/jit_scenario_injector.py``. Replacing
the entry in ``sys.modules`` keeps a single module object so the trigger engine,
the configs package and the tests all share one instance and patching either
import path affects the other.
"""

import sys

from ai.configs import jit_scenario_injector as _impl

sys.modules[__name__] = _impl
