from __future__ import annotations

"""Ingestion interface for dataset pipeline

Defines a minimal abstract connector API and a simple registry/factory for connectors.

This is intentionally small and lightweight—connectors should implement the
IngestionConnector ABC and register themselves via the `register_connector` helper.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ingest_utils import read_with_retry, RateLimiter
from .ingestion_deduplication import add_content_check_duplicate


@dataclass
class IngestRecord:
    """Canonical container for an ingested record."""
    id: str
    payload: Any
    metadata: dict[str, Any]


class IngestionError(Exception):
    pass


class IngestionConnector(ABC):
    """Abstract base for ingestion connectors.

    Implementations should be lightweight and focused on one source (S3, GCS,
    YouTube, local filesystem, etc.). The connector is responsible for fetching
    one or more records and returning them as `IngestRecord` instances.
    """

    name: str

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def connect(self) -> None:
        """Establish any needed connections or clients."""

    @abstractmethod
    def fetch(self) -> Iterable[Any]:
        """Yield validated ConversationRecord objects from the source.

        Integrates Phase 02 validation: fetches raw IngestRecord, validates/normalizes
        to canonical schema, computes quality score, and yields if valid.
        Failed records raise IngestionError (for quarantine handling upstream).

        Connectors should implement streaming-style yields so callers can process
        records incrementally and apply backpressure.
        """

    @abstractmethod
    def close(self) -> None:
        """Tear down any connections and free resources."""

    def validate(self, record: IngestRecord) -> bool:
        """Optional source-level validation hook before schema validation.

        Returns True for source-specific checks (e.g., file integrity).
        Schema validation happens in fetch via validate_record.
        Default implementation accepts all records; override when needed.
        """
        return True


# Simple connector registry
_CONNECTOR_REGISTRY: dict[str, type] = {}


def register_connector(name: str, connector_cls: type) -> None:
    """Register a connector class by name.

    Example usage:
        register_connector('local', LocalFileConnector)
    """
    _CONNECTOR_REGISTRY[name] = connector_cls


def get_connector(name: str, *args, **kwargs) -> IngestionConnector:
    cls = _CONNECTOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown connector: {name}")
    return cls(*args, **kwargs)


# Minimal example connector: local filesystem reader
class LocalFileConnector(IngestionConnector):
    """Read files from a local directory and yield IngestRecord per file.

    This example is intentionally simple and synchronous; production connectors
    should add retries/backoff and streaming for large files.
    """

    def __init__(self, directory: str, name: str | None = None, *, retry_options: dict | None = None, rate_limit: dict | None = None):
        super().__init__(name=name)
        self.directory = Path(directory)
        # retry_options passed to read_with_retry (e.g., {'retries': 3, 'backoff_factor': 0.2})
        self.retry_options = retry_options or {"retries": 3, "backoff_factor": 0.2}
        # rate_limit: dict with capacity and refill_rate
        self.rate_limiter = None
        if rate_limit:
            self.rate_limiter = RateLimiter(capacity=rate_limit.get("capacity", 10), refill_rate=rate_limit.get("refill_rate", 1.0))

    def connect(self) -> None:
        if not self.directory.exists():
            raise IngestionError(f"Directory not found: {self.directory}")

    def fetch(self) -> Iterable[IngestRecord]:
        for p in sorted(self.directory.glob("**/*")):
            if p.is_file():
                # Security check: prevent path traversal
                try:
                    # Resolve the path to ensure it's within the allowed directory
                    resolved_path = p.resolve()
                    allowed_base = self.directory.resolve()
                    if not str(resolved_path).startswith(str(allowed_base)):
                        print(f"Security warning: Attempted path traversal detected: {p}")
                        continue
                except Exception:
                    # If path resolution fails, skip the file
                    continue
                    
                # Security check: prevent dangerous file extensions
                ext = p.suffix.lower()
                dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.ps1', '.jar']
                if ext in dangerous_extensions:
                    print(f"Security warning: Skipping dangerous file type: {p}")
                    continue

                try:
                    # honor rate limiting if configured (blocking with timeout)
                    if self.rate_limiter:
                        acquired = self.rate_limiter.acquire(blocking=True, timeout=5)
                        if not acquired:
                            raise IngestionError(f"Rate limiter timeout reading {p}")
                    # read with retry helper; read_with_retry here expects a Path-like and retry options
                    payload = read_with_retry(p, retry_options=self.retry_options)
                    
                    # Deduplication check at ingestion stage
                    if not add_content_check_duplicate(payload):
                        # Content is a duplicate, skip
                        continue
                    
                    rec = IngestRecord(
                        id=str(p.resolve()),
                        payload=payload,
                        metadata={
                            "path": str(p),
                            "size": p.stat().st_size,
                            "source_type": "local_file",  # Add for validation mapping
                        },
                    )
                    # Source-level validation (e.g., file type check)
                    if not self.validate(rec):
                        # Log and quarantine source-level failure
                        from ai.dataset_pipeline import quarantine

                        store = quarantine.get_quarantine_store()
                        errors = [f"Source validation failed for {self.name} connector"]
                        store.quarantine_record(rec, errors)
                        continue  # Skip to next record

                    # Phase 02: Schema validation and normalization
                    from ai.dataset_pipeline import validation

                    try:
                        validated = validation.validate_record(rec)
                        yield validated
                    except validation.ValidationError as ve:
                        # Quarantine schema validation failure and continue
                        from ai.dataset_pipeline import quarantine

                        store = quarantine.get_quarantine_store()
                        errors = [str(ve.errors()) if hasattr(ve, "errors") else str(ve)]
                        store.quarantine_record(rec, errors)
                        # Log (in prod, use structured logger)
                        continue  # Skip to next record
                    except Exception as e:
                        # Quarantine general ingestion errors
                        from ai.dataset_pipeline import quarantine

                        store = quarantine.get_quarantine_store()
                        errors = [f"Unexpected ingestion error: {e}"]
                        store.quarantine_record(rec, errors)
                        # Log
                        continue  # Skip to next record

                except Exception as e:
                    # Outer-level catch: quarantine and continue. Use best-effort because `rec` may not be defined.
                    from ai.dataset_pipeline import quarantine

                    store = quarantine.get_quarantine_store()
                    errors = [f"Unexpected ingestion error: {e}"]
                    try:
                        store.quarantine_record(rec, errors)  # type: ignore[name-defined]
                    except Exception:
                        # If rec isn't defined or quarantine fails, ignore to avoid crashing the connector.
                        pass
                    # Log
                    try:
                        pass  # type: ignore[name-defined]
                    except Exception:
                        str(p)
                    continue

    def close(self) -> None:
        # nothing to close for local
        return None


# Register the example connector
register_connector("local", LocalFileConnector)


__all__ = [
    "IngestRecord",
    "IngestionConnector",
    "IngestionError",
    "LocalFileConnector",
    "get_connector",
    "register_connector",
]
