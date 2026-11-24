"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.local_foresight_memory_record_service.
"""

import sys

from ai.research import local_foresight_memory_record_service as _impl

sys.modules[__name__] = _impl
