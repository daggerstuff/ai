"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.foresight_local_retention.
"""

import sys

from ai.research import foresight_local_retention as _impl

sys.modules[__name__] = _impl
