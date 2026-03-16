"""Utility functions for quality module."""
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
