"""Compatibility alias for the live implementation.

``ai.pkg_mera`` exists to keep legacy import paths working; this module
delegates to ai.tools.utilities.pipelines.quality.conversation_quality_pattern_analyzer so there is one
implementation and one module identity behind both paths.
"""

import sys

from ai.tools.utilities.pipelines.quality import conversation_quality_pattern_analyzer as _impl

sys.modules[__name__] = _impl
