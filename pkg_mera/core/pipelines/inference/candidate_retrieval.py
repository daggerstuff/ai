"""Compatibility alias for the live implementation.

``ai.pkg_mera`` exists to keep legacy import paths working; this module
delegates to ai.tools.utilities.pipelines.inference.candidate_retrieval so there is one
implementation and one module identity behind both paths.
"""

import sys

from ai.tools.utilities.pipelines.inference import candidate_retrieval as _impl

sys.modules[__name__] = _impl
