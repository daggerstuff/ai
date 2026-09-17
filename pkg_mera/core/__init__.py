"""Compatibility shim: redirect ai.pkg_mera.core.sourcing → ai.sourcing.

Legacy imports from ai.pkg_mera.core.sourcing.journal.cli.config are
redirected to the actual implementation at ai.sourcing.journal.cli.config.
"""

from __future__ import annotations

import sys

_ai_sourcing = __import__("ai.sourcing", fromlist=["journal"])
sys.modules[f"{__name__}.sourcing"] = _ai_sourcing
