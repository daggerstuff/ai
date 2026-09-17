from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class RetainScopeConflictError(Exception):
    """Raised when there is a scope conflict during retention operations."""


@dataclass(frozen=True)
class RetainScope:
    """Scope identifiers for a retention operation.

    Mirrors the ``x-memory-*`` header set accepted by the Foresight retain
    route; only provided identifiers are written into item metadata.
    """

    org_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    visibility: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Build scope metadata for retention operations.

        Unscoped retains produce an empty mapping (matching the legacy
        bare-scope behaviour). `visibility` defaults to the platform norm
        "private" when any other scope key is present, so retained items
        never default to an ambiguous visibility.
        """
        metadata: dict[str, Any] = {}
        for key in ("org_id", "project_id", "session_id", "agent_id", "run_id"):
            value = getattr(self, key)
            if value:
                metadata[key] = value
        if metadata or self.visibility:
            metadata["visibility"] = self.visibility or "private"
        return metadata


def scope_metadata(
    scope: RetainScope | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build scope metadata for retention operations.

    Accepts either a :class:`RetainScope` or the legacy keyword form
    (``org_id=..., visibility=...``) used by the MCP retain route.
    """
    if scope is None:
        scope = RetainScope(**kwargs)
    return scope.to_metadata()


def build_scoped_retain_items(
    *,
    items: list[dict[str, Any]],
    user_id: str,
    base_metadata: dict[str, Any] | None,
    ownership_validator: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Build items for retention with scope validation.

    Args:
        items: List of item dictionaries to prepare for retention
        user_id: ID of the user performing the retention
        base_metadata: Base metadata to apply to all items
        ownership_validator: Function that validates ownership of a document ID
            (should raise an exception if validation fails)

    Returns:
        List of prepared item dictionaries that passed ownership validation
    """
    result: list[dict[str, Any]] = []

    for item in items:
        # Create a copy to avoid mutating the original item
        prepared_item = item.copy()

        # Ensure metadata exists and is a dictionary
        metadata: dict[str, Any] = prepared_item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Apply base metadata
        if base_metadata:
            metadata.update(base_metadata)

        # Add user ID to metadata
        metadata["user_id"] = user_id

        # Update the item with prepared metadata
        prepared_item["metadata"] = metadata

        # Validate ownership if document_id is present
        document_id = prepared_item.get("document_id")
        if document_id is not None:
            try:
                ownership_validator(document_id)
            except Exception:
                # If ownership validation fails, skip this item
                # In a more sophisticated implementation, we might log this
                continue

        result.append(prepared_item)

    return result
