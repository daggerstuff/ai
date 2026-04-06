def validate_identifier(identifier: str) -> str:
    """
    Validates that the identifier contains only alphanumeric characters and underscores.
    Returns the identifier if valid, raises HTTPException otherwise.
    This prevents SQL injection by disallowing special characters.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        # Avoid echoing the raw identifier back in the error detail to reduce leakage (Review suggestion)
        raise HTTPException(
            status_code=400, detail="Invalid identifier format"
        )
    return identifier
