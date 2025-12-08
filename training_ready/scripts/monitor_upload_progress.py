#!/usr/bin/env python3
"""
Monitor Upload Progress

Quick script to check the status of dataset uploads to S3.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def main():
    base_path = Path.cwd()

    # Check HuggingFace downloads
    hf_results_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "hf_download_results.json"
    if hf_results_path.exists():
        with open(hf_results_path, "r") as f:
            hf_data = json.load(f)

        print("📥 HuggingFace Downloads:")
        print(f"  Total: {hf_data['total']}")
        print(f"  ✅ Successful: {hf_data['successful']}")
        print(f"  ❌ Failed: {hf_data['failed']}")
        if hf_data['successful'] > 0:
            print(f"  Success rate: {hf_data['successful'] / hf_data['total'] * 100:.1f}%")
        print()

    # Check local uploads
    local_results_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "local_upload_results.json"
    if local_results_path.exists():
        with open(local_results_path, "r") as f:
            local_data = json.load(f)

        print("📤 Local Uploads:")
        print(f"  Total processed: {local_data['total']}")
        print(f"  ✅ Successful: {local_data['successful']}")
        print(f"  ⏭️  Skipped: {local_data.get('skipped', 0)}")
        print(f"  ❌ Failed: {local_data['failed']}")

        if local_data['total'] > 0:
            success_rate = (local_data['successful'] + local_data.get('skipped', 0)) / local_data['total'] * 100
            print(f"  Success rate: {success_rate:.1f}%")

        # Calculate total size uploaded
        total_size = 0
        for result in local_data.get('results', []):
            if result.get('result', {}).get('success') and 'size' in result.get('result', {}):
                total_size += result['result']['size']

        if total_size > 0:
            print(f"  Total size uploaded: {format_size(total_size)}")

        # Show recent failures if any
        failures = [
            r for r in local_data.get('results', [])
            if not r.get('result', {}).get('success')
        ]
        if failures:
            print(f"\n  ⚠️  Recent failures ({len(failures)}):")
            for fail in failures[:5]:
                error = fail.get('result', {}).get('error', 'Unknown error')
                print(f"    - {Path(fail['local_path']).name}: {error[:60]}")

        print()
    else:
        print("📤 Local Uploads: No results file found (upload not started yet)")
        print()

    # Check catalog for total counts
    catalog_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "dataset_accessibility_catalog.json"
    if catalog_path.exists():
        with open(catalog_path, "r") as f:
            catalog = json.load(f)

        summary = catalog.get('summary', {})
        print("📋 Catalog Summary:")
        print(f"  Total datasets: {summary.get('total', 0)}")
        print(f"  Local-only: {summary.get('local_only', 0)}")
        print(f"  HuggingFace IDs: {summary.get('huggingface_unique_ids', 0)}")
        print()

        # Calculate progress
        if local_results_path.exists():
            local_data = json.load(open(local_results_path))
            local_total = summary.get('local_only', 0)
            local_processed = local_data.get('total', 0)
            local_successful = local_data.get('successful', 0) + local_data.get('skipped', 0)

            if local_total > 0:
                progress_pct = (local_processed / local_total) * 100
                success_pct = (local_successful / local_processed * 100) if local_processed > 0 else 0
                print(f"📊 Upload Progress:")
                print(f"  Processed: {local_processed} / {local_total} ({progress_pct:.1f}%)")
                print(f"  Successful: {local_successful} / {local_processed} ({success_pct:.1f}%)")
                print(f"  Remaining: {local_total - local_processed}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

