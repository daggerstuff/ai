#!/usr/bin/env python3
"""
Quick script to check if encoding fix is still running and show progress
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))


def check_process():
    """Check if fix_encoding.py is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "fix_encoding.py"], check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            print(f"✅ fix_encoding.py is running (PIDs: {', '.join(pids)})")
            return True
        print("❌ fix_encoding.py is not running")
        return False
    except FileNotFoundError:
        # pgrep not available, try ps
        try:
            result = subprocess.run(["ps", "aux"], check=False, capture_output=True, text=True)
            if "fix_encoding.py" in result.stdout:
                print("✅ fix_encoding.py appears to be running")
                return True
            print("❌ fix_encoding.py is not running")
            return False
        except:
            print("⚠️  Cannot check process status (pgrep/ps not available)")
            return None


def check_results_file():
    """Check if results file exists and show last update"""
    results_path = project_root / "ai/training_ready/data/encoding_fix_results.json"
    if results_path.exists():
        import json

        with open(results_path) as f:
            data = json.load(f)

        timestamp = data.get("timestamp", "unknown")
        summary = data.get("summary", {})

        print("\n📊 Results file found:")
        print(f"   Last update: {timestamp}")
        print(f"   Total processed: {summary.get('total', 0)}")
        print(f"   Fixed: {summary.get('fixed', 0)}")
        print(f"   Skipped: {summary.get('skipped', 0)}")
        print(f"   Failed: {summary.get('failed', 0)}")

        # Show recent files
        results = data.get("results", [])
        if results:
            print("\n📝 Recent files processed:")
            for result in results[-5:]:  # Last 5
                status = "✅" if result.get("success") else "❌"
                file = result.get("file", "unknown")
                print(f"   {status} {file}")

        return True
    print("\n⚠️  No results file found yet (script may still be running)")
    return False


if __name__ == "__main__":
    print("🔍 Checking encoding fix process status...")
    print("=" * 60)

    is_running = check_process()
    has_results = check_results_file()

    if is_running:
        print("\n💡 Tip: The script may be processing large files.")
        print("   Large JSONL files can take several minutes to download and process.")
        print("   Check back in a few minutes or monitor the results file.")
    elif has_results:
        print("\n✅ Process appears to have completed. Check results above.")
