def main():
    with open("api/dataset_api.py") as f:
        content = f.read()

    # 1. Replace auth_system.validate_api_key with auth_system.authenticate_api_key
    content = content.replace("auth_system.validate_api_key", "auth_system.authenticate_api_key")

    # 2. Update validate_identifier to raise HTTPException
    old_validate_func = '''def validate_identifier(identifier: str) -> str:
    """
    Validates that the identifier contains only alphanumeric characters and underscores.
    Returns the identifier if valid, raises ValueError otherwise.
    This prevents SQL injection by disallowing special characters.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        raise ValueError(f"Invalid identifier format: {identifier}")
    return identifier'''

    new_validate_func = '''def validate_identifier(identifier: str) -> str:
    """
    Validates that the identifier contains only alphanumeric characters and underscores.
    Returns the identifier if valid, raises HTTPException otherwise.
    This prevents SQL injection by disallowing special characters.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        raise HTTPException(status_code=400, detail=f"Invalid identifier format: {identifier}")
    return identifier'''

    content = content.replace(old_validate_func, new_validate_func)

    # 3. Update except ValueError around line 151
    old_except_block = """            try:
                safe_table_name = validate_identifier(table_name)
            except ValueError:
                continue"""

    new_except_block = """            try:
                safe_table_name = validate_identifier(table_name)
            except HTTPException:
                continue"""

    content = content.replace(old_except_block, new_except_block)

    with open("api/dataset_api.py", "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
