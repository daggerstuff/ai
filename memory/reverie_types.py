"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.reverie_types.
"""

import sys

from ai.research import reverie_types as _impl

sys.modules[__name__] = _impl
