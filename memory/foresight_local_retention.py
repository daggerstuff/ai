from __future__ import annotations

from typing import Any, Callable, Dict, List


class RetainScopeConflictError(Exception):
    """Raised when there is a scope conflict during retention operations."""


def scope_metadata() -> Dict[str, Any]:
    """Return default scope metadata for retention operations.
    
    This is a placeholder implementation. In a full implementation, this would
    return metadata about the current scope for retention policies.
    """
    return {}


def build_scoped_retain_items(
    *,
    items: List[Dict[str, Any]],
    user_id: str,
    base_metadata: Dict[str, Any] | None,
    ownership_validator: Callable[[str], None],
) -> List[Dict[str, Any]]:
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
    result: List[Dict[str, Any]] = []
    
    for item in items:
        # Create a copy to avoid mutating the original item
        prepared_item = item.copy()
        
        # Ensure metadata exists and is a dictionary
        metadata: Dict[str, Any] = prepared_item.get("metadata", {})
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