"""Alias module: :mod:`api.mcp_server.mcp_stdio_server` is the inference server.

The implementation lives in ``inference/api/mcp_server/mcp_stdio_server.py``.
Replacing the entry in ``sys.modules`` keeps a single module object so the
stdio server, its entry point and the tests all resolve to one implementation.
"""

import sys

from inference.api.mcp_server import mcp_stdio_server as _impl

sys.modules[__name__] = _impl
