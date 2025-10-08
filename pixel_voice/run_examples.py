#!/usr/bin/env python3
"""
Script to run examples using uv.
"""
import os
import subprocess
import sys
from pathlib import Path


def run_with_uv(script_path: str):
    """Run a script using uv."""
    cmd = ["uv", "run", "python", script_path]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def main():
    """Main function to run examples."""
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    examples = {
        "1": ("API Client Example", "examples/api_client_example.py"),
        "2": ("MCP Client Example", "examples/mcp_client_example.py"),
    }

    print("Pixel Voice Examples (using uv)")
    print("=" * 40)

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")

    choice = input("\nEnter example number (1-2): ").strip()

    if choice in examples:
        name, script_path = examples[choice]
        print(f"\nRunning: {name}")
        print("-" * 40)

        if Path(script_path).exists():
            run_with_uv(script_path)
        else:
            print(f"Error: {script_path} not found")
            sys.exit(1)
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()
