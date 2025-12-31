#!/usr/bin/env python3
"""
Remote S3 Processor - Template for 60GB pixel-data processing on remote system
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime


def create_remote_processor_script():
    """Create self-contained processor for remote execution"""

    processor_script = """#!/bin/bash
# Remote 60GB Pixel-Data S3 Processor
# Usage: ./remote_s3_processor.sh [AWS_ACCESS_KEY_ID] [AWS_SECRET_ACCESS_KEY]

set -e

AWS_ACCESS_KEY_ID=${1:-$AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${2:-$AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=us-east-va
ENDPOINT_URL=https://s3.us-east-va.io.cloud.ovh.us
BUCKET=pixel-data

if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "❌ AWS credentials required"
    echo "Usage: $0 <ACCESS_KEY> <SECRET_KEY>"
    echo "Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars"
    exit 1
fi

echo "🚀 Discovering 60GB pixel-data S3 bucket..."
echo "📍 Endpoint: $ENDPOINT_URL"
echo "📦 Bucket: $BUCKET"

# Create discovery directory
mkdir -p training_ready/data/remote_s3_discovery

# List all objects
echo "📊 Listing S3 objects..."
aws s3 ls s3://$BUCKET --recursive --endpoint-url $ENDPOINT_URL --human-readable --summarize > training_ready/data/remote_s3_discovery/s3_listing.txt 2>&1

# Extract dataset files
echo "🔍 Filtering therapeutic datasets..."
aws s3 ls s3://$BUCKET --recursive --endpoint-url $ENDPOINT_URL | \
    grep -E '\.(json|jsonl|csv)$' | \
    grep -v -E '\.(lock|tmp|cache)' | \
    sort -k3 -hr > training_ready/data/remote_s3_discovery/dataset_files.txt

# Generate processing commands
echo "⚙️  Creating processing commands..."
cat > training_ready/data/remote_s3_discovery/process_60gb_datasets.sh << 'EOF'
#!/bin/bash
# Process 60GB pixel-data therapeutic corpus

set -e

BUCKET=pixel-data
ENDPOINT_URL=https://s3.us-east-va.io.cloud.ovh.us
DOWNLOAD_DIR=training_ready/data/remote_60gb_corpus
TEMP_DIR=/tmp/s3_processing
MAX_PARALLEL=4

mkdir -p "$DOWNLOAD_DIR" "$TEMP_DIR"

# Download in parallel
process_file() {
    local file="$1"
    local dest="$2"
    echo "📥 Downloading: $file"
    aws s3 cp "s3://$BUCKET/$file" "$dest/" --endpoint-url "$ENDPOINT_URL" --quiet
}

# Top therapeutic datasets (adjust paths as needed)
THERAPEUTIC_DATASETS=(
    "consolidated/therapeutic_conversations.jsonl"
    "processed/mental_health_reddit.jsonl"
    "therapeutic/therapy_sessions.jsonl"
    "consolidated/crisis_intervention.jsonl"
    "processed/suicide_prevention.jsonl"
)

# Download key datasets
for dataset in "${THERAPEUTIC_DATASETS[@]}"; do
    if aws s3 ls "s3://$BUCKET/$dataset" --endpoint-url "$ENDPOINT_URL" >/dev/null 2>&1; then
        process_file "$dataset" "$DOWNLOAD_DIR"
    fi
done

# Stream process all JSON files
echo "🔄 Streaming processing 60GB corpus..."
aws s3 ls s3://$BUCKET --recursive --endpoint-url $ENDPOINT_URL | \
    grep '\.json' | \
    while read -r line; do
        file=$(echo "$line" | awk '{print $4}')
        if [[ -n "$file" ]]; then
            echo "Processing: $file"
            aws s3 cp "s3://$BUCKET/$file" - --endpoint-url "$ENDPOINT_URL" | \
                python3 -c "
import sys, json
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        # Add processing logic here
        print(json.dumps(data))
    except:
        pass
" > "$TEMP_DIR/$(basename \"$file\").processed"
        fi
    done

echo "✅ 60GB dataset processing complete"
echo "📁 Results in: $DOWNLOAD_DIR"
EOF

chmod +x training_ready/data/remote_s3_discovery/process_60gb_datasets.sh

# Create summary
cat > training_ready/data/remote_s3_discovery/README.md << 'EOF'
# 60GB Pixel-Data S3 Processing

## Overview
- **Bucket**: pixel-data
- **Endpoint**: https://s3.us-east-va.io.cloud.ovh.us
- **Size**: 60GB therapeutic corpus
- **Format**: JSON/JSONL/CSV therapeutic conversations

## Usage
1. Set credentials: `export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...`
2. Run discovery: `./remote_s3_processor.sh`
3. Process datasets: `training_ready/data/remote_s3_discovery/process_60gb_datasets.sh`

## Streaming Processing
For memory-efficient processing without full download:
```bash
aws s3 cp s3://pixel-data/path/to/file.jsonl - --endpoint-url https://s3.us-east-va.io.cloud.ovh.us | python3 your_processor.py
```
EOF

echo "✅ Remote S3 processor created"
echo "📁 Files created in: training_ready/data/remote_s3_discovery/"
echo "🚀 Ready to process 60GB therapeutic corpus"
"""

    # Write the remote processor
    with open("training_ready/scripts/remote_s3_processor.sh", "w") as f:
        f.write(processor_script)

    # Make executable
    subprocess.run(["chmod", "+x", "training_ready/scripts/remote_s3_processor.sh"])

    print("✅ Remote S3 processor created")
    print("📁 File: training_ready/scripts/remote_s3_processor.sh")
    print("")
    print("🚀 Usage on remote system:")
    print("   ./training_ready/scripts/remote_s3_processor.sh")
    print("   # OR")
    print(
        "   ./training_ready/scripts/remote_s3_processor.sh YOUR_ACCESS_KEY YOUR_SECRET_KEY"
    )


def generate_process_commands():
    """Generate processing commands for 60GB dataset"""

    commands = {
        "discovery": "training_ready/scripts/remote_s3_processor.sh",
        "stream_process": """
# Stream-process 60GB without full download
aws s3 ls s3://pixel-data --recursive --endpoint-url https://s3.us-east-va.io.cloud.ovh.us | \
    grep '\.jsonl$' | \
    awk '{print $4}' | \
    xargs -I {} -P 4 -n 1 sh -c 'aws s3 cp s3://pixel-data/{} - --endpoint-url https://s3.us-east-va.io.cloud.ovh.us | python3 -c "
import json, sys
for line in sys.stdin:
    try:
        data = json.loads(line)
        # Process therapeutic data here
        print(json.dumps(data))
    except:
        pass
"'
        """,
        "download_top": """
# Download top 10GB of therapeutic data
aws s3 ls s3://pixel-data --recursive --endpoint-url https://s3.us-east-va.io.cloud.ovh.us | \
    grep '\.json' | \
    sort -k3 -hr | \
    head -20 | \
    awk '{print $4}' | \
    xargs -I {} aws s3 cp s3://pixel-data/{} training_ready/data/remote_60gb_corpus/ --endpoint-url https://s3.us-east-va.io.cloud.ovh.us
        """,
    }

    # Save commands
    with open("training_ready/data/remote_s3_commands.json", "w") as f:
        json.dump(commands, f, indent=2)

    return commands


def main():
    """Main function"""
    print("🚀 Remote 60GB S3 Processor")
    print("=" * 50)
    print("📍 Target: pixel-data bucket (60GB)")
    print("🔗 Endpoint: https://s3.us-east-va.io.cloud.ovh.us")
    print("")

    create_remote_processor_script()
    commands = generate_process_commands()

    print("📋 Available commands:")
    for name, cmd in commands.items():
        print(f"   {name}: {cmd[:100]}...")

    print(f"\n✅ Ready for remote deployment")
    print("🎯 Upload to remote system and run:")
    print("   ./training_ready/scripts/remote_s3_processor.sh")


if __name__ == "__main__":
    main()
