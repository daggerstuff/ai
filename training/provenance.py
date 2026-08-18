"""Training-record provenance helpers.

The dataset pipeline writes plain JSONL records from several producers.  These
helpers keep provenance shape and license validation consistent without binding
callers to a storage backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

PIPELINE_VERSION = "modern-dataset-provenance-v1"

# Small allow-list of SPDX identifiers used by this repository's training-data
# sources. NOASSERTION is the honest default when upstream licensing is unknown.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC0-1.0",
        "MIT",
        "NOASSERTION",
    }
)


@dataclass(frozen=True)
class ProvenanceOptions:
    """Optional provenance fields that are shared across producers."""

    pipeline_version: str = PIPELINE_VERSION
    license_id: str = "NOASSERTION"
    transformations: tuple[str, ...] = ()


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp suitable for JSONL records."""

    return datetime.now(UTC).isoformat()


def validate_license(license_id: str) -> str:
    """Validate and return a supported SPDX license identifier.

    Raises:
        ValueError: if the identifier is blank or outside the repository allow-list.
    """

    normalized = license_id.strip()
    if normalized not in ALLOWED_LICENSES:
        allowed = ", ".join(sorted(ALLOWED_LICENSES))
        raise ValueError(f"Unsupported license '{license_id}'. Expected one of: {allowed}")
    return normalized


def build_provenance(
    source_url: str,
    source_type: str,
    *,
    acquired_at: str | None = None,
    options: ProvenanceOptions | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated provenance block for a training record."""

    selected = options or ProvenanceOptions()
    normalized_source_url = source_url.strip()
    normalized_source_type = source_type.strip()
    if not normalized_source_url:
        raise ValueError("source_url is required")
    if not normalized_source_type:
        raise ValueError("source_type is required")

    provenance: dict[str, Any] = {
        "source_url": normalized_source_url,
        "source_type": normalized_source_type,
        "acquired_at": acquired_at or utc_now_iso(),
        "pipeline_version": selected.pipeline_version.strip() or PIPELINE_VERSION,
        "license": validate_license(selected.license_id),
        "transformations": [str(item).strip() for item in selected.transformations if str(item).strip()],
    }
    if metadata:
        provenance["metadata"] = dict(metadata)
    return provenance


def attach_provenance(record: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with a validated provenance block attached."""

    if "provenance" in record:
        raise ValueError("record already contains provenance")
    raw_transformations = provenance.get("transformations", ())
    transformations: Iterable[str] = raw_transformations if isinstance(raw_transformations, Iterable) else ()
    validated = build_provenance(
        str(provenance.get("source_url", "")),
        str(provenance.get("source_type", "")),
        acquired_at=str(provenance.get("acquired_at")) if provenance.get("acquired_at") else None,
        options=ProvenanceOptions(
            pipeline_version=str(provenance.get("pipeline_version", PIPELINE_VERSION)),
            license_id=str(provenance.get("license", "NOASSERTION")),
            transformations=tuple(str(item) for item in transformations),
        ),
        metadata=provenance.get("metadata") if isinstance(provenance.get("metadata"), Mapping) else None,
    )
    enriched = dict(record)
    enriched["provenance"] = validated
    return enriched
