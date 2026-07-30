"""Pixelated AI package.

Backwards-compatibility aliases:

  ai.core     -> ai.pkg_mera.core
  ai.platform -> ai.pkg_mera.platform

The aliases are implemented via a custom ``sys.meta_path`` importer so that
submodules (e.g. ``ai.core.pipelines.foo``) resolve to the exact same module
objects as ``ai.pkg_mera.core.pipelines.foo``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType


def _install_pkg_mera_aliases() -> None:
    """Make ``ai.core``/``ai.platform`` transparent aliases to ``pkg_mera``.

    A meta-path finder/loader pair redirects the entire import tree without
    creating duplicate module objects, which would break ``isinstance`` checks
    when old and new import paths are mixed.
    """
    aliases = {
        "ai.core": "ai.pkg_mera.core",
        "ai.platform": "ai.pkg_mera.platform",
    }

    class _AliasLoader(importlib.abc.Loader):
        def __init__(self, real_name: str) -> None:
            self.real_name = real_name

        def create_module(self, spec):  # noqa: ANN202
            # Warn once when the legacy import path is used.
            warnings.warn(
                f"{spec.name!r} is deprecated; use {self.real_name!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Return the already-imported real module so both the old and new
            # import paths share the exact same module object.
            return importlib.import_module(self.real_name)

        def exec_module(self, module: ModuleType) -> None:  # noqa: ARG002
            # The module was already executed during import_module above.
            pass

    class _AliasFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # noqa: ANN001, ANN202
            for old_prefix, new_prefix in aliases.items():
                if fullname == old_prefix or fullname.startswith(old_prefix + "."):
                    real_name = new_prefix + fullname[len(old_prefix) :]
                    loader = _AliasLoader(real_name)
                    spec = importlib.util.spec_from_loader(fullname, loader)
                    return spec
            return None

    sys.meta_path.insert(0, _AliasFinder())


_install_pkg_mera_aliases()
