"""Minimal Hugging Face Hub client wrapper used by tests and examples.

This module intentionally performs lazy imports of the external
``huggingface_hub`` package so that unit tests can inject a fake
``huggingface_hub`` module into ``sys.modules`` without requiring the
real dependency to be installed.

The wrapper provides a tiny, well-documented surface that the rest of
the codebase can use without coupling application code to the HF
library API surface.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import List, Optional


__all__ = ["HuggingFaceClient"]


logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Small helper for common Hugging Face Hub operations.

    Behavior is deliberately simple: if a token is provided it will be
    used for all operations; otherwise the client will read
    ``HUGGINGFACE_HUB_TOKEN`` from the environment.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token if token is not None else os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )

    def download_file(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        repo_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Download a single file from a repo using hf_hub_download.

        The function performs a lazy import so tests can provide a fake
        ``huggingface_hub`` module.
        """
        mod = importlib.import_module("huggingface_hub")
        hf_hub_download = getattr(mod, "hf_hub_download", None)
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub.hf_hub_download is not available on the imported module"
            )
        return hf_hub_download(
            repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=cache_dir,
            token=self.token,
            **kwargs,
        )

    def snapshot_download(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Download a repository snapshot (entire repo or revision).

        Uses :func:`huggingface_hub.snapshot_download` via lazy import.
        """
        mod = importlib.import_module("huggingface_hub")
        snapshot_download = getattr(mod, "snapshot_download", None)
        if snapshot_download is None:
            raise RuntimeError(
                "huggingface_hub.snapshot_download is not available on the imported module"
            )
        return snapshot_download(
            repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            token=self.token,
            **kwargs,
        )

    def list_files(
        self, repo_id: str, revision: Optional[str] = None, repo_type: Optional[str] = None
    ) -> List[str]:
        """Return a list of files in the repository using HfApi.list_repo_files.

        The HF API object is created via lazy import of
        ``huggingface_hub.HfApi`` so unit tests can provide a fake
        implementation.
        """
        mod = importlib.import_module("huggingface_hub")
        HfApi = getattr(mod, "HfApi", None)
        if HfApi is None:
            raise RuntimeError(
                "huggingface_hub.HfApi is not available on the imported module"
            )
        api = HfApi()
        return api.list_repo_files(repo_id, repo_type=repo_type, revision=revision)
