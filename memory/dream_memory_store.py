"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.dream_memory_store.
"""

import sys

from ai.research import dream_memory_store as _impl

sys.modules[__name__] = _impl
