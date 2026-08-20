"""DVC versioned dataset loading utilities.

Provides ``load_versioned()`` to pull a specific dataset version from DVC
and verify its integrity before returning parsed JSONL records.

Usage::

    from training.data_versioning import load_versioned

    train = load_versioned("train", version="dataset-v1.0.0")
    val   = load_versioned("val",   version="dataset-v1.0.0")
    test  = load_versioned("test",  version="dataset-v1.0.0")
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

CURATED_DIR = Path("data/curated/sft_chatml")


def _md5_file(path: str | Path) -> str:
    """Compute the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_dvc_md5(dvc_path: str | Path) -> str | None:
    """Read the ``md5`` field from a ``.dvc`` pointer file.

    Returns ``None`` if the file or field is missing.
    """
    p = Path(dvc_path)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    outs = meta.get("outs", [])
    if not outs:
        return None
    return outs[0].get("md5")


def load_versioned(
    split: str,
    version: str = "dataset-v1.0.0",
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load a versioned DVC-tracked JSONL split.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    version:
        Git tag identifying the dataset version (e.g. ``"dataset-v1.0.0"``).
    repo_root:
        Optional path to the repository root (defaults to current working
        directory).  Must be the ``ai`` submodule where DVC is initialised.

    Returns
    -------
    list[dict[str, Any]]
        Parsed JSONL records.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist after ``dvc pull``.
    RuntimeError
        If the MD5 hash of the pulled file does not match the ``.dvc`` pointer.
    subprocess.CalledProcessError
        If ``git checkout`` or ``dvc pull`` fails.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    data_path = root / CURATED_DIR / f"{split}.jsonl"
    dvc_path = root / CURATED_DIR / f"{split}.jsonl.dvc"
    rel_dvc = str(CURATED_DIR / f"{split}.jsonl.dvc")

    # Restore the .dvc pointer for the requested version.
    subprocess.run(
        ["git", "checkout", version, "--", rel_dvc],
        check=True,
        cwd=str(root),
    )

    # Pull the actual data from the DVC remote.
    subprocess.run(
        ["dvc", "pull", str(dvc_path)],
        check=True,
        cwd=str(root),
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path} after dvc pull, not found.")

    # Verify integrity: the file's MD5 must match the .dvc pointer.
    expected_md5 = _read_dvc_md5(dvc_path)
    if expected_md5 is not None:
        actual_md5 = _md5_file(data_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Hash mismatch for {data_path}: "
                f"expected {expected_md5}, got {actual_md5}"
            )

    # Parse and return JSONL records.
    records: list[dict[str, Any]] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_available_versions() -> list[str]:
    """List all ``dataset-*`` git tags in the repository."""
    result = subprocess.run(
        ["git", "tag", "-l", "dataset-v*"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [t for t in result.stdout.strip().splitlines() if t.startswith("dataset-v")]
