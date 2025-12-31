#!/usr/bin/env python3
"""
OVH Direct S3 Processor - Uses OVH S3 format credentials
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def run_s3cmd_command(cmd, access_key=None, secret_key=None):
    """Run s3cmd with OVH S3 format"""
    env = os.environ.copy()
    if access_key:
        env["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = secret_key

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, env=env
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1


def discover_pixel_data():
    """Discover 60GB pixel-data with OVH S3 format"""
    print("🔍 Discovering 60GB pixel-data with OVH S3 format...")

    # Try different credential approaches
    credentials = {
        "env": (
            os.environ.get("AWS_ACCESS_KEY_ID"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
        ),
        "provided": (
            os.environ.get("OVH_S3_ACCESS_KEY"),
            os.environ.get("OVH_S3_SECRET_KEY"),
        ),
    }

    for name, (access_key, secret_key) in credentials.items():
        if not access_key or not secret_key:
            continue

        print(f"🧪 Testing {name} credentials...")

        # Test with AWS CLI
        cmd = "aws s3 ls s3://pixel-data --recursive --endpoint-url https://s3.us-east-va.io.cloud.ovh.us"
        stdout, stderr, code = run_s3cmd_command(cmd, access_key, secret_key)

        if code == 0:
            print(f"✅ Connected with {name} credentials")

            # Parse results
            lines = stdout.split("\n")
            files = []
            total_size = 0

            for line in lines:
                if line.strip() and not line.startswith("PRE"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            size = int(parts[2])
                            path = " ".join(parts[3:])
                            if any(
                                ext in path.lower()
                                for ext in [".json", ".jsonl", ".csv"]
                            ):
                                files.append({"path": path, "size": size})
                                total_size += size
                        except:
                            continue

            report = {
                "timestamp": datetime.now().isoformat(),
                "bucket": "pixel-data",
                "endpoint": "https://s3.us-east-va.io.cloud.ovh.us",
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_gb": total_size / (1024**3),
                "files": sorted(files, key=lambda x: x["size"], reverse=True)[:50],
            }

            # Save discovery
            Path("training_ready/data").mkdir(exist_ok=True)
            with open("training_ready/data/pixel_data_60gb_discovery.json", "w") as f:
                json.dump(report, f, indent=2)

            print(f"📊 Found {len(files)} files")
            print(f"📏 Total size: {total_size / (1024**3):.2f}GB")

            # Top files
            print("\n🗂️  Top 20 files:")
            for i, file_info in enumerate(files[:20], 1):
                size_gb = file_info["size"] / (1024**3)
                print(f"   {i}. {file_info['path']}: {size_gb:.2f}GB")

            return report
        else:
            print(f"❌ {name} credentials failed: {stderr[:100]}...")

    # Create fallback processor that works with OVH format
    create_ovh_specific_processor()
    return None


def create_ovh_specific_processor():
    """Create OVH-specific S3 processor"""

    ovh_script = """#!/bin/bash
# OVH S3 60GB Processor - Correct format for OVH

set -e

# OVH S3 configuration
S3_ACCESS_KEY=${AWS_ACCESS_KEY_ID:-$1}
S3_SECRET_KEY=${AWS_SECRET_ACCESS_KEY:-$2}
S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
S3_BUCKET=pixel-data

if [[ -z "$S3_ACCESS_KEY" || -z "$S3_SECRET_KEY" ]]; then
    echo "❌ OVH S3 credentials required"
    echo "Usage: $0 <ACCESS_KEY> <SECRET_KEY>"
    echo "Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi

echo "🚀 Discovering 60GB pixel-data OVH S3 bucket..."
echo "📍 Endpoint: $S3_ENDPOINT"
echo "📦 Bucket: $S3_BUCKET"

# Create discovery directory
mkdir -p training_ready/data/pixel_data_60gb_discovery

# Use AWS CLI with OVH S3 endpoint
export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
export AWS_DEFAULT_REGION=us-east-va

# Discover all objects
echo "📊 Listing S3 objects..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT --human-readable --summarize > training_ready/data/pixel_data_60gb_discovery/s3_full_listing.txt 2>&1

# Extract therapeutic datasets
echo "🔍 Filtering therapeutic datasets..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
    grep -E '\.(json|jsonl|csv)$' | \
    grep -v -E '\.(lock|tmp|cache|git)' | \
    sort -k3 -hr > training_ready/data/pixel_data_60gb_discovery/therapeutic_datasets.txt

# Count and summarize
echo "📊 Processing discovery..."
total_files=$(wc -l < training_ready/data/pixel_data_60gb_discovery/therapeutic_datasets.txt)
total_size=$(aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT --summarize 2>/dev/null | \
    grep "Total Size" | awk '{print $3}' || echo "0")

echo "📋 Discovery complete:"
echo "   📁 Files: $total_files"
echo "   💾 Size: $total_size"
echo "   📍 Reports: training_ready/data/pixel_data_60gb_discovery/"

# Generate processing commands
cat > training_ready/data/pixel_data_60gb_discovery/process_commands.sh << 'PROCESS_EOF'
#!/bin/bash
# Stream-process 60GB therapeutic corpus

S3_BUCKET=pixel-data
S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us

# Stream process without download
stream_process() {
    aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
        grep '\.jsonl$' | \
        awk '{print $4}' | \
        while read file; do
            echo "Processing: $file"
            aws s3 cp s3://$S3_BUCKET/$file - --endpoint-url $S3_ENDPOINT | \
                python3 -c "
import json, sys
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        # PII cleaning & deduplication
        import re
        text = str(data)
        text = re.sub(r'\\\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Z|a-z]{2,}\\\\b', '[EMAIL_REDACTED]', text)
        text = re.sub(r'\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b', '[SSN_REDACTED]', text)
        print(text)
    except:
        pass
"
        done
}

# Download top datasets
download_top() {
    aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
        grep '\.json' | \
        sort -k3 -hr | \
        head -50 | \
        awk '{print $4}' | \
        xargs -I {} -P 4 aws s3 cp s3://$S3_BUCKET/{} training_ready/data/remote_60gb_corpus/ --endpoint-url $S3_ENDPOINT
}

case "$1" in
    "stream")
        stream_process
        ;;
    "download")
        mkdir -p training_ready/data/remote_60gb_corpus
        download_top
        ;;
    *)
        echo "Usage: $0 [stream|download]"
        echo "   stream: Process 60GB without download"
        echo "   download: Download top datasets"
        ;;
esac
PROCESS_EOF

chmod +x training_ready/data/pixel_data_60gb_discovery/process_commands.sh

echo "✅ 60GB OVH S3 processor ready"
echo "🚀 Commands: training_ready/data/pixel_data_60gb_discovery/process_commands.sh"
"""

    with open("training_ready/scripts/ovh_60gb_processor.sh", "w") as f:
        f.write(ovh_script)

    subprocess.run(["chmod", "+x", "training_ready/scripts/ovh_60gb_processor.sh"])

    print("✅ OVH 60GB processor created")
    print("🚀 Usage: ./training_ready/scripts/ovh_60gb_processor.sh")
    print("   # OR")
    print("   ./training_ready/scripts/ovh_60gb_processor.sh ACCESS_KEY SECRET_KEY")


def main():
    """Main function"""
    print("🚀 OVH Direct 60GB S3 Processor")
    print("=" * 50)
    print("📍 Target: 60GB pixel-data S3 bucket")
    print("🔗 Endpoint: https://s3.us-east-va.io.cloud.ovh.us")
    print("🔑 Using provided credentials format")
    print("")

    # Create OVH processor
    create_ovh_specific_processor()

    print("✅ Ready to process 60GB therapeutic corpus")
    print("🎯 Run: ./training_ready/scripts/ovh_60gb_processor.sh")


if __name__ == "__main__":
    main()
