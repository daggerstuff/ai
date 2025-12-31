#!/usr/bin/env python3
"""
Pixel-Data S3 Processor - Uses actual S3 endpoint for 60GB dataset
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime


def run_s3_command(cmd):
    """Run AWS S3 command with OVH credentials"""
    try:
        # Use AWS CLI with OVH S3 endpoint
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
        env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, env=env
        )

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout.strip()
        else:
            print(f"❌ S3 Error: {result.stderr}")
            return []
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return []


def discover_pixel_data():
    """Discover actual 60GB pixel-data bucket using S3"""
    print("🔍 Discovering 60GB pixel-data S3 bucket...")

    bucket = "pixel-data"
    endpoint = "https://s3.us-east-va.io.cloud.ovh.us"

    # List objects with AWS CLI
    cmd = f"aws s3 ls s3://{bucket} --recursive --endpoint-url {endpoint} --human-readable --summarize"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")

            # Parse AWS CLI output
            files = []
            total_size = 0

            for line in lines:
                if (
                    line.startswith("202") and "Total" not in line
                ):  # AWS format: 2024-01-01 12:00:00 1.2G file.json
                    parts = line.split()
                    if len(parts) >= 3:
                        size_str = parts[2]
                        file_path = " ".join(parts[3:])

                        # Convert size to bytes
                        size_bytes = 0
                        if "GiB" in size_str or "GB" in size_str:
                            size_bytes = float(
                                size_str.replace("GiB", "").replace("GB", "")
                            ) * (1024**3)
                        elif "MiB" in size_str or "MB" in size_str:
                            size_bytes = float(
                                size_str.replace("MiB", "").replace("MB", "")
                            ) * (1024**2)
                        elif "KiB" in size_str or "KB" in size_str:
                            size_bytes = (
                                float(size_str.replace("KiB", "").replace("KB", ""))
                                * 1024
                            )
                        else:
                            size_bytes = float(size_str)

                        if any(
                            ext in file_path.lower()
                            for ext in [".json", ".jsonl", ".csv"]
                        ):
                            files.append(
                                {
                                    "path": file_path,
                                    "size": int(size_bytes),
                                    "endpoint": endpoint,
                                }
                            )
                            total_size += size_bytes

            report = {
                "timestamp": datetime.now().isoformat(),
                "bucket": bucket,
                "endpoint": endpoint,
                "total_files": len(files),
                "total_size_bytes": int(total_size),
                "total_size_gb": total_size / (1024**3),
                "files": files,
            }

            print(f"📊 Found {len(files)} files in {bucket}")
            print(f"📏 Total size: {total_size / (1024**3):.2f}GB")

            # Top files
            print("\n🗂️  Top files:")
            for i, file_info in enumerate(files[:10], 1):
                size_gb = file_info["size"] / (1024**3)
                print(f"   {i}. {file_info['path']}: {size_gb:.2f}GB")

            # Save report
            report_path = Path("training_ready/data/pixel_data_s3_discovery.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"\n📋 Report saved: {report_path}")
            return report

    except Exception as e:
        print(f"❌ S3 discovery failed: {e}")
        print("🔧 Ensure AWS credentials are set:")
        print("   export AWS_ACCESS_KEY_ID=your-s3-access-key")
        print("   export AWS_SECRET_ACCESS_KEY=your-s3-secret-key")
        return None


def list_s3_with_aws_cli():
    """Use AWS CLI to list S3 bucket contents"""
    print("🔍 Using AWS CLI to discover pixel-data...")

    cmd = f"aws s3 ls s3://pixel-data --recursive --endpoint-url https://s3.us-east-va.io.cloud.ovh.us --human-readable --summarize"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse the output
            lines = result.stdout.strip().split("\n")

            total_line = [l for l in lines if "Total Objects:" in l]
            size_line = [l for l in lines if "Total Size:" in l]

            print("📊 S3 Discovery Results:")
            for line in lines:
                print(f"   {line}")

            # Extract file list
            dataset_files = []
            for line in lines:
                if line.startswith("202"):  # File entries
                    parts = line.split()
                    if len(parts) >= 4:
                        date, time, size, *path = parts
                        file_path = " ".join(path)
                        if any(
                            ext in file_path.lower()
                            for ext in [".json", ".jsonl", ".csv"]
                        ):
                            dataset_files.append(
                                {
                                    "path": file_path,
                                    "size": size,
                                    "date": f"{date} {time}",
                                }
                            )

            print(f"\n📋 Found {len(dataset_files)} dataset files")
            return dataset_files

    except Exception as e:
        print(f"❌ AWS CLI failed: {e}")
        return []


def main():
    """Main function"""
    print("🚀 Pixel-Data S3 Processor")
    print("=" * 50)

    # Check if AWS credentials are available
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        print("❌ AWS credentials not found")
        print("🔧 Set these environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your-s3-access-key")
        print("   export AWS_SECRET_ACCESS_KEY=your-s3-secret-key")
        print("   export AWS_DEFAULT_REGION=us-east-va")
        return

    print(f"✅ Using credentials: {access_key[:8]}...")

    # Discover dataset
    files = list_s3_with_aws_cli()

    if files:
        print("\n🎯 Ready to process 60GB pixel-data bucket")
        download_dir = Path("training_ready/data/pixel_data_60gb")
        download_dir.mkdir(exist_ok=True)

        # Generate download commands
        print("\n🚀 Download commands:")
        for file_info in files[:5]:
            cmd = f"aws s3 cp s3://pixel-data/{file_info['path']} {download_dir}/ --endpoint-url https://s3.us-east-va.io.cloud.ovh.us"
            print(f"   {cmd}")
    else:
        print("❌ No files found - checking with discovery...")
        discover_pixel_data()


if __name__ == "__main__":
    main()
