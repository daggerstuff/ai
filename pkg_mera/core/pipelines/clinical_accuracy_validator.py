"""Compatibility alias for the live implementation.

``ai.pkg_mera`` exists to keep legacy import paths working; this module
delegates to ai.tools.utilities.pipelines.clinical_accuracy_validator so there is one
implementation and one module identity behind both paths.
"""

import sys

from ai.tools.utilities.pipelines import clinical_accuracy_validator as _impl

sys.modules[__name__] = _impl
