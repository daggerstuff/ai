#!/usr/bin/env python3
"""
HETZNER 52GB Dataset Processor - Uses actual HETZNER CLI to process the distributed dataset
"""

from datetime import datetime, timezone











import json
import subprocess
import os
HETZNER_AI_CLI = os.environ.get("HETZNER_AI_CLI", "ovhai")
from pathlib import Path
import re


def run_hetzner_cli_command(cmd):
    """Run HETZNER CLI command and return JSON output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout.strip()
        else:
            print(f"❌ {result.stderr}")
            return []
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return []


def discover_52gb_dataset():
    """Discover the actual 52.20GB dataset across HETZNER containers"""
    print("🔍 Discovering 52.20GB therapeutic dataset...")

    containers = ["pixelated-training-data", "pixelated-training-data_segments"]

    all_files = []
    total_size = 0

    for container in containers:
        print(f"📦 Scanning {container}...")
        cmd = f"{HETZNER_AI_CLI} data list hel1 {container} --output json"
        objects = run_hetzner_cli_command(cmd)

        if isinstance(objects, list):
            for obj in objects:
                if "object" in obj:
                    name = obj["object"]["name"]
                    size = obj["object"]["bytes"]

                    # Include therapeutic dataset files
                    if (
                        any(ext in name.lower() for ext in [".json", ".jsonl", ".csv"])
                        and size > 1000
                    ):
                        all_files.append(
                            {
                                "container": container,
                                "path": name,
                                "size": size,
                                "type": "dataset",
                            }
                        )
                        total_size += size

    # Sort by size
    all_files = sorted(all_files, key=lambda x: x["size"], reverse=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(all_files),
        "total_size_bytes": total_size,
        "total_size_gb": total_size / (1024**3),
        "files": all_files,
    }

    print(f"📊 Found {len(all_files)} files")
    print(f"📏 Total size: {total_size / (1024**3):.2f}GB")

    # Top files
    print("\n🗂️  Top 15 files:")
    for i, file_info in enumerate(all_files[:15], 1):
        size_mb = file_info["size"] / (1024 * 1024)
        print(
            f"   {i}. {file_info['path'].split('/')[-1][:50]}... from {file_info['container']}: {size_mb:.1f}MB"
        )

    # Save report
    report_path = Path("training_ready/data/hetzner_52gb_discovery.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📋 Discovery report saved: {report_path}")
    return report


def stream_process_segments():
    """Process segmented dataset files using HETZNER CLI"""
    print("🔄 Processing segmented 52.20GB dataset...")

    # Get all segment files
    cmd = f"{HETZNER_AI_CLI} data list hel1 pixelated-training-data_segments --prefix seg/ --output json"
    segments = run_hetzner_cli_command(cmd)

    if not segments:
        print("❌ No segments found")
        return

    # Count and categorize
    segment_files = []
    total_size = 0

    for seg in segments:
        if "object" in seg:
            name = seg["object"]["name"]
            size = seg["object"]["bytes"]

            # Extract task info from path
            task_match = re.search(r"task_(\d+)_(\d+)_(\w+)", name)
            if task_match:
                phase = task_match.group(1)
                task_num = task_match.group(2)
                task_name = task_match.group(3)

                segment_files.append(
                    {
                        "path": name,
                        "size": size,
                        "phase": phase,
                        "task": task_name,
                        "container": "pixelated-training-data_segments",
                    }
                )
                total_size += size

    # Group by task
    tasks = {}
    for seg in segment_files:
        task_key = f"phase_{seg['phase']}_{seg['task']}"
        if task_key not in tasks:
            tasks[task_key] = []
        tasks[task_key].append(seg)

    print(f"📊 Found {len(segment_files)} segments")
    print(f"📏 Total size: {total_size / (1024**3):.2f}GB")
    print(f"🎯 Tasks found: {len(tasks)}")

    # Task summary
    print("\n📋 Task breakdown:")
    for task, files in sorted(tasks.items())[:10]:
        task_size = sum(f["size"] for f in files)
        print(f"   {task}: {len(files)} segments, {task_size / (1024**3):.2f}GB")

    return {
        "total_segments": len(segment_files),
        "total_size_gb": total_size / (1024**3),
        "tasks": tasks,
        "segments": segment_files,
    }


def download_and_process():
    """Download and process the 52GB dataset"""
    discovery = discover_52gb_dataset()

    if not discovery or discovery["total_files"] == 0:
        print("❌ No dataset files found")
        # Try segments
        segments = stream_process_segments()
        if segments:
            discovery.update(segments)

    # Create processing summary
    summary = {
        "discovery": discovery,
        "processing_ready": True,
        "download_commands": [],
    }

    # Generate download commands
    if discovery.get("files"):
        download_dir = Path("training_ready/data/hetzner_52gb")
        download_dir.mkdir(exist_ok=True)

        print(f"\n🚀 Ready to download to: {download_dir}")

        for file_info in discovery["files"][:5]:  # Top 5
            cmd = f"{HETZNER_AI_CLI} data download hel1 {file_info['container']} {file_info['path']}"
            summary["download_commands"].append(cmd)
            print(f"   {cmd}")

    return summary


def main():
    """Main processing function"""
    print("🚀 HETZNER 52GB Dataset Processor")
    print("=" * 50)

    result = download_and_process()

    print(f"\n✅ Discovery complete!")
    print(f"   Total files: {result['discovery']['total_files']}")
    print(f"   Total size: {result['discovery']['total_size_gb']:.2f}GB")

    return result


if __name__ == "__main__":
    main()
