from pathlib import Path

def get_project_root() -> Path:
    """Traverses up from the current file to find the project root.

    The project root is identified by the presence of the `pyproject.toml` file.
    It finds the outermost `pyproject.toml` in the directory tree.
    """
    current_path = Path(__file__).resolve() # Resolve to get absolute path and handle symlinks
    
    outermost_root = None

    # Traverse up the directory tree
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            outermost_root = parent # Update to the current parent if pyproject.toml is found
    
    if outermost_root:
        return outermost_root
    else:
        raise FileNotFoundError("pyproject.toml not found in any parent directory.")
