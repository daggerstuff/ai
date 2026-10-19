"""Hash-linked, credential-free provenance receipts for synthetic generation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.run-manifest.json")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist non-secret run provenance beside generated records."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def prepare_run_manifest(
    *,
    output_path: Path,
    code_path: Path,
    backend: Any,
    source_artifacts: dict[str, dict[str, str | None]],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Create a unique run receipt or reject a resume with changed compatibility inputs."""
    compatibility = {
        "provider": backend.name,
        "provider_host": urlsplit(backend.url).hostname,
        "configured_model": os.environ.get("NF_MODEL") or None,
        "resolved_request_model": backend.model,
        "code_sha256": sha256_file(code_path),
        "source_artifacts": source_artifacts,
        "generation_config": generation_config,
        "generation_config_sha256": canonical_sha256(generation_config),
    }
    fingerprint = canonical_sha256(compatibility)
    path = manifest_path(output_path)
    has_records = output_path.exists() and output_path.stat().st_size > 0
    if path.exists():
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("compatibility_fingerprint") != fingerprint:
            raise ValueError(
                "generation resume refused: provider/model, source, code, or generation config "
                "differs from run manifest"
            )
    elif has_records:
        raise ValueError(
            "generation resume refused: existing output has no run manifest; choose a new output directory"
        )
    else:
        manifest = {
            "schema_version": "synthetic_generation_run_v1",
            "run_id": str(uuid.uuid4()),
            "compatibility_fingerprint": fingerprint,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "status": "running",
            "compatibility": compatibility,
            "usage": {"status": "not_reported_yet", "scope": "invocation"},
        }
    manifest["status"] = "running"
    manifest["invocation_id"] = str(uuid.uuid4())
    manifest["invocation_started_at_utc"] = datetime.now(UTC).isoformat()
    write_manifest(path, manifest)
    return manifest


def row_provenance(manifest: dict[str, Any], manifest_file: Path, source_key: str) -> dict[str, Any]:
    source = manifest["compatibility"]["source_artifacts"][source_key]
    compatibility = manifest["compatibility"]
    return {
        "run_id": manifest["run_id"],
        "run_manifest": str(manifest_file),
        "provider": compatibility["provider"],
        "provider_host": compatibility["provider_host"],
        "configured_model": compatibility["configured_model"],
        "resolved_request_model": compatibility["resolved_request_model"],
        "source_artifact_path": source["path"],
        "source_artifact_sha256": source["sha256"],
    }


def finalize_run_manifest(manifest: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Store usage for this invocation only; resumed invocation totals are not cumulative."""
    manifest["status"] = "completed"
    completed_at = datetime.now(UTC).isoformat()
    manifest["completed_at_utc"] = completed_at
    manifest["usage"] = {
        "status": "provider_reported_invocation" if usage.get("usage_responses", 0) else "no_usage_reported_invocation",
        "scope": "invocation",
        "cumulative": False,
        "invocation_id": manifest.get("invocation_id"),
        "started_at_utc": manifest.get("invocation_started_at_utc"),
        "completed_at_utc": completed_at,
        "usage_responses": usage.get("usage_responses", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "provider_reported_models": sorted(usage.get("provider_reported_models", [])),
    }
    return manifest
