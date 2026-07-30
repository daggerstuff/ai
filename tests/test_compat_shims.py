"""Tests for ai.core / ai.platform backwards-compatibility shims.

The pkg_mera refactor moved the implementation to ai.pkg_mera.core and
ai.pkg_mera.platform. These shims ensure code still importing ai.core or
ai.platform continues to work.
"""

import sys
import warnings

import ai


def test_ai_core_resolves_to_pkg_mera_core():
    import ai.core
    import ai.pkg_mera.core

    assert ai.core is ai.pkg_mera.core


def test_ai_platform_resolves_to_pkg_mera_platform():
    import ai.platform
    import ai.pkg_mera.platform

    assert ai.platform is ai.pkg_mera.platform


def test_ai_core_submodule_import():
    from ai.core.utils.s3_dataset_loader import S3DatasetLoader
    from ai.pkg_mera.core.utils.s3_dataset_loader import S3DatasetLoader as NewS3DatasetLoader

    assert S3DatasetLoader is NewS3DatasetLoader


def test_ai_platform_submodule_import():
    from ai.platform.patient_psi.profiles import ClinicalProfile
    from ai.pkg_mera.platform.patient_psi.profiles import ClinicalProfile as NewClinicalProfile

    assert ClinicalProfile is NewClinicalProfile


def test_pkg_mera_core_unchanged():
    import ai.pkg_mera.core

    assert hasattr(ai.pkg_mera.core, "utils")


def test_pkg_mera_platform_unchanged():
    import ai.pkg_mera.platform

    assert hasattr(ai.pkg_mera.platform, "patient_psi")


def test_ai_core_import_emits_deprecation_warning():
    for key in list(sys.modules):
        if key == "ai.core" or key.startswith("ai.core."):
            del sys.modules[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ai.core  # noqa: F401

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("ai.pkg_mera.core" in str(w.message) for w in caught)


def test_ai_platform_import_emits_deprecation_warning():
    for key in list(sys.modules):
        if key == "ai.platform" or key.startswith("ai.platform."):
            del sys.modules[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ai.platform  # noqa: F401

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("ai.pkg_mera.platform" in str(w.message) for w in caught)
