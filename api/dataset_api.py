def validate_identifier(identifier: str) -> str:
    """
    Validates that the identifier contains only alphanumeric characters and underscores.
    Returns the identifier if valid, raises HTTPException otherwise.
    This prevents SQL injection by disallowing special characters.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        raise HTTPException(status_code=400, detail=f"Invalid identifier format: {identifier}")
    return identifier


def enforce_permission(auth_entity: Any, level: PermissionLevel, action: str):
    """
    Centralized scope enforcement with safe access to the 'scopes' list.
    Ensures that missing scopes default to an empty list and return a proper 403.
    """
    # Defensive access to scopes to handle varying auth entity shapes (Review suggestion)
    scopes = []
    if isinstance(auth_entity, dict):
        scopes = auth_entity.get("scopes") or []
    elif hasattr(auth_entity, "scopes"):
        scopes = auth_entity.scopes or []

    if level not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions to {action}. Required scope: {level.value}",
        )
