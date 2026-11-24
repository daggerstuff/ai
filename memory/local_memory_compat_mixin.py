"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.local_memory_compat_mixin.
"""

import sys

from ai.research import local_memory_compat_mixin as _impl

sys.modules[__name__] = _impl
