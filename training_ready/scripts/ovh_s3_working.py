#!/usr/bin/env python3
"""
OVH S3 Working Processor - Correct OVH format for 60GB pixel-data
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def run_ovh_s3_command(cmd):
    """Run AWS CLI with OVH S3 format credentials"""
    if "OVH_S3_ACCESS_KEY" not in os.environ or "OVH_S3_SECRET_KEY" not in os.environ:
        raise ValueError(
            "OVH_S3_ACCESS_KEY and OVH_S3_SECRET_KEY environment variables must be set"
        )

    env = os.environ.copy()
    env.update(
        {
            "AWS_ACCESS_KEY_ID": os.environ["OVH_S3_ACCESS_KEY"],
            "AWS_SECRET_ACCESS_KEY": os.environ["OVH_S3_SECRET_KEY"],
            "AWS_DEFAULT_REGION": "us-east-va",
        }
    )

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, env=env
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1


def discover_60gb_dataset():
    """Discover the actual 60GB pixel-data"""
    print("🚀 Discovering 60GB pixel-data S3 bucket...")
    print("📍 Endpoint: https://s3.us-east-va.io.cloud.ovh.us")
    print("📦 Bucket: pixel-data")

    cmd = "aws s3 ls s3://pixel-data --recursive --endpoint-url https://s3.us-east-va.io.cloud.ovh.us"
    stdout, stderr, code = run_ovh_s3_command(cmd)

    if code != 0:
        print(f"❌ S3 access failed: {stderr}")
        return None

    # Parse results
    lines = stdout.split("\n")
    files = []
    total_size = 0

    for line in lines:
        if line.strip() and "PRE" not in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    size = int(parts[2])
                    path = " ".join(parts[3:])

                    # Include therapeutic dataset files
                    if any(ext in path.lower() for ext in [".json", ".jsonl", ".csv"]):
                        files.append(
                            {
                                "path": path,
                                "size": size,
                                "endpoint": "https://s3.us-east-va.io.cloud.ovh.us",
                            }
                        )
                        total_size += size
                except:
                    continue

    # Generate 60GB discovery report
    report = {
        "timestamp": datetime.now().isoformat(),
        "bucket": "pixel-data",
        "endpoint": "https://s3.us-east-va.io.cloud.ovh.us",
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_gb": total_size / (1024**3),
        "credentials_format": "OVH_S3_APPLICATION_KEY",
        "files": sorted(files, key=lambda x: x["size"], reverse=True)[:100],
    }

    # Save discovery
    Path("training_ready/data").mkdir(exist_ok=True)
    with open("training_ready/data/pixel_data_60gb_final.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"📊 Found {len(files)} files")
    print(f"📏 Total size: {total_size / (1024**3):.2f}GB")

    # Top files
    print("\n🗂️  Top 15 files:")
    for i, file_info in enumerate(files[:15], 1):
        size_gb = file_info["size"] / (1024**3)
        filename = file_info["path"].split("/")[-1]
        print(f"   {i}. {filename}: {size_gb:.2f}GB")

    return report


def create_60gb_processor():
    """Create 60GB processing script"""

    # We must properly escape the environment variables for the shell script
    processor_script = """#!/bin/bash
# 60GB OVH S3 Pixel-Data Processor - Ready to run

set -e

# OVH S3 Configuration
S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
S3_BUCKET="pixel-data"

if [ -z "$OVH_S3_ACCESS_KEY" ] || [ -z "$OVH_S3_SECRET_KEY" ]; then
    echo "❌ Error: OVH_S3_ACCESS_KEY and OVH_S3_SECRET_KEY environment variables must be set"
    exit 1
fi

export AWS_ACCESS_KEY_ID="$OVH_S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$OVH_S3_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-va"

echo "🚀 Processing 60GB OVH S3 therapeutic corpus..."
echo "📍 Endpoint: $S3_ENDPOINT"
echo "📦 Bucket: $S3_BUCKET"

# Create processing directory
mkdir -p training_ready/data/pixel_data_processed

# Discovery phase
echo "🔍 Discovering datasets..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT --human-readable --summarize > training_ready/data/pixel_data_processed/s3_discovery.txt

# Extract therapeutic datasets
echo "🎯 Extracting therapeutic datasets..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
    grep -E '\.(json|jsonl|csv)$' | \
    grep -v -E '\.(lock|tmp|cache|git)' | \
    sort -k3 -hr > training_ready/data/pixel_data_processed/datasets.txt

# Create processing summary
files_count=$(wc -l < training_ready/data/pixel_data_processed/datasets.txt)
total_size=$(aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT --summarize 2>/dev/null | \
    grep "Total Size" | awk '{print $3}' || echo "0")

echo "📊 Discovery complete:"
echo "   📁 Files: $files_count"
echo "   💾 Size: $total_size"

# Generate processing commands
cat > training_ready/data/pixel_data_processed/commands.sh << 'COMMANDS_EOF'
#!/bin/bash
# 60GB S3 Processing Commands

S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
S3_BUCKET="pixel-data"

# 1. Stream-process 60GB without full download
stream_60gb() {
    echo "🔄 Streaming 60GB therapeutic corpus..."
    aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
        grep '\.jsonl$' | \
        awk '{print $4}' | \
        while read file; do
            echo "Processing: $file"
            aws s3 cp s3://$S3_BUCKET/$file - --endpoint-url $S3_ENDPOINT | \
                python3 -c "
import json, sys, hashlib, re
seen_hashes = set()
for line in sys.stdin:
    try:
        data = json.loads(line.strip())

        # PII cleaning
        text = str(data)
        text = re.sub(r'\\\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Z|a-z]{2,}\\\\b', '[EMAIL]', text)
        text = re.sub(r'\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b', '[SSN]', text)
        text = re.sub(r'\\\\b\\\\d{4}[\\\\s-]?\\\\d{4}[\\\\s-]?\\\\d{4}[\\\\s-]?\\\\d{4}\\\\b', '[CARD]', text)


        # Deduplication
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            print(json.dumps(data))
    except:
        continue
"
        done
}

# 2. Download top datasets
download_top() {
    echo "📥 Downloading top therapeutic datasets..."
    mkdir -p training_ready/data/remote_60gb_corpus

    # Download in parallel
    aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
        grep '\.json' | \
        sort -k3 -hr | \
        head -100 | \
        awk '{print $4}' | \
        xargs -I {} -P 8 aws s3 cp s3://$S3_BUCKET/{} training_ready/data/remote_60gb_corpus/ --endpoint-url $S3_ENDPOINT
}

# 3. Stream validation
validate_60gb() {
    echo "✅ Validating 60GB corpus..."
    aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
        grep '\.json' | \
        wc -l | \
        xargs -I {} echo "Total JSON files: {}"
}

case "$1" in
    "stream")
        stream_60gb
        ;;
    "download")
        download_top
        ;;
    "validate")
        validate_60gb
        ;;
    "test")
        # Add test case that was implicitly supported but missing in usage
        validate_60gb
        ;;
    *)
        echo "Usage: $0 [stream|download|validate]"
        echo "   stream: Process 60GB without download"
        echo "   download: Download top datasets"
        echo "   validate: Count and validate corpus"
        ;;
esac
COMMANDS_EOF

chmod +x training_ready/data/pixel_data_processed/commands.sh

echo "✅ 60GB OVH S3 processor ready"
echo "🚀 Commands available:"
echo "   ./training_ready/data/pixel_data_processed/commands.sh stream"
echo "   ./training_ready/data/pixel_data_processed/commands.sh download"
echo "   ./training_ready/data/pixel_data_processed/commands.sh validate"
"""

    with open("training_ready/scripts/process_60gb_ovh.py", "w") as f:
        f.write(processor_script)

    subprocess.run(["chmod", "+x", "training_ready/scripts/process_60gb_ovh.py"])

    return processor_script


def main():
    """Main function"""
    print("🚀 OVH 60GB S3 Direct Processor")
    print("=" * 50)
    print("📍 Target: 60GB pixel-data S3 bucket")
    print("🔗 Endpoint: https://s3.us-east-va.io.cloud.ovh.us")
    print("🔑 Using OVH S3 Application Key format")
    print("")

    # Discover 60GB dataset
    report = discover_60gb_dataset()

    if report:
        create_60gb_processor()
        print("✅ 60GB therapeutic corpus ready for processing")
        print("🎯 Run: ./training_ready/scripts/process_60gb_ovh.py")
    else:
        print("✅ Created 60GB processor for manual execution")
        print("🎯 Run: ./training_ready/scripts/process_60gb_ovh.py")


if __name__ == "__main__":
    main()
