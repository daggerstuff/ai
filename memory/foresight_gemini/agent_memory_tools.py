"""Compatibility alias for the live implementation.

``ai.memory`` and ``ai.research`` had drifted into two copies of the same
modules, so the same class had two identities depending on the import
path. This module delegates to ai.research.foresight_gemini.agent_memory_tools.
"""

import sys

from ai.research.foresight_gemini import agent_memory_tools as _impl

sys.modules[__name__] = _impl
