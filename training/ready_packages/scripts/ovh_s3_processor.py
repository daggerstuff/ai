#!/usr/bin/env python3
"""
OVH S3 Processor - Processes actual 52.20GB from pixelated-training-data container
"""

import json
import subprocess
import os
from pathlib import Path
from datetime import datetime


def run_ovhai_command(cmd):
    """Run ovhai command and return JSON output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Error: {result.stderr}")
            return []
    except Exception as e:
        print(f"Command failed: {e}")
        return []


def discover_datasets():
    """Discover all datasets in OVH S3"""
    print("🔍 Discovering datasets in OVH S3...")

    # Get all objects from pixelated-training-data
    cmd = "ovhai data list US-EAST-VA pixelated-training-data --output json"
    objects = run_ovhai_command(cmd)

    if not objects:
        print("❌ No data found")
        return

    # Parse and filter actual dataset files
    datasets = []
    total_size = 0

    for obj in objects:
        if "object" in obj:
            name = obj["object"]["name"]
            size = obj["object"]["bytes"]

            # Filter for actual JSON dataset files
            if (
                name.endswith(".json")
                and "/consolidated/" in name
                and "cache" not in name
            ):
                datasets.append(
                    {"path": name, "size": size, "bucket": "pixelated-training-data"}
                )
                total_size += size

    # Sort by size
    datasets = sorted(datasets, key=lambda x: x["size"], reverse=True)

    print(f"📊 Found {len(datasets)} actual dataset files")
    print(f"📏 Total size: {total_size / (1024**3):.2f}GB")

    # Top files
    print("\n🗂️  Top dataset files:")
    for i, ds in enumerate(datasets[:10], 1):
        size_gb = ds["size"] / (1024**3)
        print(f"   {i}. {ds['path'].split('/')[-1]}: {size_gb:.2f}GB")

    # Save discovery report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(datasets),
        "total_size_bytes": total_size,
        "total_size_gb": total_size / (1024**3),
        "datasets": datasets,
    }

    report_path = Path("training_ready/data/ovh_s3_discovery.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📋 Report saved: {report_path}")
    return report


def download_file(remote_path, local_path):
    """Download file from OVH S3"""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = f"ovhai data download US-EAST-VA pixelated-training-data {remote_path} --output {local_path}"
    print(f"📥 Downloading: {remote_path}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Downloaded: {local_path}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {remote_path}: {e}")
        return False


def main():
    """Main processing function"""
    print("🚀 OVH S3 Dataset Discovery")
    print("=" * 50)

    # Discover datasets
    report = discover_datasets()

    if report and report["datasets"]:
        print(f"\n🎯 Ready to process {report['total_files']} files")
        print(f"💾 Total size: {report['total_size_gb']:.2f}GB")

        # Option to download top files
        response = input("\nDownload top 5 files for processing? (y/N): ")
        if response.lower() == "y":
            download_dir = Path("training_ready/data/downloads")
            download_dir.mkdir(exist_ok=True)

            for i, ds in enumerate(report["datasets"][:5], 1):
                filename = ds["path"].split("/")[-1]
                local_path = download_dir / filename
                download_file(ds["path"], local_path)


if __name__ == "__main__":
    main()
