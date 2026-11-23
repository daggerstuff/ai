"""Alias module: :mod:`scripts.rclone_boto3_shim` is :mod:`ai.tools.scripts.rclone_boto3_shim`.

The implementation lives in ``ai/tools/scripts/rclone_boto3_shim.py``.
Replacing the entry in ``sys.modules`` keeps a single module object for the
flat import path, the package path and the tests.
"""

import sys

from ai.tools.scripts import rclone_boto3_shim as _impl

sys.modules[__name__] = _impl
