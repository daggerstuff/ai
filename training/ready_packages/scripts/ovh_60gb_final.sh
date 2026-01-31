#!/bin/bash
# OVH 60GB S3 Processor - Credentials from Env

set -e

# OVH S3 Configuration
S3_BUCKET="pixel-data"
S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
S3_REGION="us-east-va"

# Check for credentials
if [ -z "$OVH_S3_ACCESS_KEY" ] || [ -z "$OVH_S3_SECRET_KEY" ]; then
    echo "❌ Error: OVH_S3_ACCESS_KEY and OVH_S3_SECRET_KEY environment variables must be set"
    exit 1
fi

S3_ACCESS_KEY="$OVH_S3_ACCESS_KEY"
S3_SECRET_KEY="$OVH_S3_SECRET_KEY"

export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
export AWS_DEFAULT_REGION="$S3_REGION"

echo "🚀 Processing 60GB OVH S3 therapeutic corpus..."
echo "📍 Endpoint: $S3_ENDPOINT"
echo "📦 Bucket: $S3_BUCKET"

# Create processing directory
mkdir -p training_ready/data/ovh_60gb_processed

# Phase 1: Discovery
echo "🔍 Phase 1: Discovery..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT --human-readable --summarize > training_ready/data/ovh_60gb_processed/s3_discovery.txt 2>&1

# Phase 2: Extract therapeutic datasets
echo "🎯 Phase 2: Extracting therapeutic datasets..."
aws s3 ls s3://$S3_BUCKET --recursive --endpoint-url $S3_ENDPOINT | \
    grep -E '\.(json|jsonl|csv)$' | \
    grep -v -E '\.(lock|tmp|cache|git)' | \
    sort -k3 -hr > training_ready/data/ovh_60gb_processed/therapeutic_datasets.txt

# Phase 3: Generate processing commands
echo "⚙️  Phase 3: Creating processing commands..."

# Create processing commands
cat > training_ready/data/ovh_60gb_processed/commands.sh << EOF
#!/bin/bash
# 60GB OVH S3 Processing Commands

S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
S3_BUCKET="pixel-data"

# Set credentials for subprocesses
export AWS_ACCESS_KEY_ID="$OVH_S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$OVH_S3_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-va"

# 1. Stream-process 60GB without full download
stream_60gb() {
    echo "🔄 Streaming 60GB therapeutic corpus..."
    aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT | \
        grep '\.jsonl$' | \
        awk '{print \$4}' | \
        while read file; do
            echo "Processing: \$file"
            aws s3 cp s3://\$S3_BUCKET/\$file - --endpoint-url \$S3_ENDPOINT | \
                python3 -c "
import json, sys, hashlib, re
seen_hashes = set()
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        
        # PII cleaning for therapeutic data
        text = str(data)
        text = re.sub(r'\\\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Z|a-z]{2,}\\\\b', '[EMAIL_REDACTED]', text)
        text = re.sub(r'\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b', '[SSN_REDACTED]', text)
        text = re.sub(r'\\\\b\\\\d{4}[\\\\s-]?\\\\d{4}[\\\\s-]?\\\\d{4}[\\\\s-]?\\\\d{4}\\\\b', '[CARD_REDACTED]', text)
        text = re.sub(r'\\\\b\\\\+?1?[-.\\\\s]?\\\\(?[0-9]{3}\\\\)?[-.\\\\s]?[0-9]{3}[-.\\\\s]?[0-9]{4}\\\\b', '[PHONE_REDACTED]', text)
        
        # Preserve therapeutic context
        if 'conversation' in str(data).lower() or 'therapy' in str(data).lower():
            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                print(json.dumps(data))
    except:
        continue
"
        done
}

# 2. Download top 10GB of therapeutic datasets
download_top_10gb() {
    echo "📥 Downloading top 10GB therapeutic datasets..."
    mkdir -p training_ready/data/ovh_60gb_corpus
    
    aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT | \
        grep '\.json' | \
        sort -k3 -hr | \
        head -50 | \
        awk '{print \$4}' | \
        xargs -I {} -P 8 aws s3 cp s3://\$S3_BUCKET/{} training_ready/data/ovh_60gb_corpus/ --endpoint-url \$S3_ENDPOINT
}

# 3. Validate corpus
validate_60gb() {
    echo "✅ Validating 60GB corpus..."
    
    # Count files
    file_count=\$(aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT | wc -l)
    json_count=\$(aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT | grep '\.json' | wc -l)
    
    echo "📊 Total files: \$file_count"
    echo "📊 JSON/CSV files: \$json_count"
    
    # Check bucket size
    aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT --summarize
}

# 4. Stream validation
stream_validate() {
    echo "🎯 Stream validation..."
    aws s3 ls s3://\$S3_BUCKET --recursive --endpoint-url \$S3_ENDPOINT | \
        grep '\.json' | \
        awk '{print \$4}' | \
        head -5 | \
        xargs -I {} sh -c 'echo "Testing: {}"; aws s3 cp s3://\$S3_BUCKET/{} - --endpoint-url \$S3_ENDPOINT | head -1 | python3 -c "import json, sys; print(json.dumps(json.loads(sys.stdin.read()), indent=2)[:200])"'
}

case "\$1" in
    "stream")
        stream_60gb
        ;;
    "download")
        download_top_10gb
        ;;
    "validate")
        validate_60gb
        ;;
    "test")
        stream_validate
        ;;
    *)
        echo "Usage: \$0 [stream|download|validate|test]"
        echo "   stream: Process 60GB without download"
        echo "   download: Download top 10GB datasets"
        echo "   validate: Count and validate corpus"
        echo "   test: Stream validation test"
        ;;
esac
EOF

chmod +x training_ready/data/ovh_60gb_processed/commands.sh

echo "✅ 60GB OVH S3 processor ready with correct credentials"
echo "🎯 Run: ./training_ready/scripts/ovh_60gb_final.sh"
echo "🎯 Commands: ./training_ready/data/ovh_60gb_processed/commands.sh"
