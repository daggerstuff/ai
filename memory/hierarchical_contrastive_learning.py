"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.hierarchical_contrastive_learning.
"""

import sys

from ai.research import hierarchical_contrastive_learning as _impl

sys.modules[__name__] = _impl
