"""Compatibility alias for the live implementation.

``ai.pkg_mera`` exists to keep legacy import paths working; this module
delegates to ai.tools.utilities.pipelines.inference.rare_disease_pipeline so there is one
implementation and one module identity behind both paths.
"""

import sys

from ai.tools.utilities.pipelines.inference import rare_disease_pipeline as _impl

sys.modules[__name__] = _impl
