"""Training pipelines package.

Bootstraps the repository root onto ``sys.path`` (appended, last-resort) so
absolute ``ai.research.*`` imports resolve when running under the ``ai/``-local
virtualenv (``uv run`` from ``ai/``), whose editable mapping points ``ai`` at
the nested ``ai/ai`` directory instead of this package.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
