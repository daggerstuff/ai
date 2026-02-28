#!/bin/bash
# Download expanded library using AWS CLI

# Source environment variables carefully
if [ -f "ai/.env" ]; then
    # Parse OVH_S3 specifically to avoid issues with other variables
    export AWS_ACCESS_KEY_ID=$(grep OVH_S3_ACCESS_KEY ai/.env | cut -d'"' -f2)
    export AWS_SECRET_ACCESS_KEY=$(grep OVH_S3_SECRET_KEY ai/.env | cut -d'"' -f2)
    export AWS_DEFAULT_REGION="us-east-va"
fi

BUCKET="pixel-data"
ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"

DOWNLOAD_DIR="ai/training_ready/data/generated"
mkdir -p "$DOWNLOAD_DIR/nightmare_scenarios"
mkdir -p "$DOWNLOAD_DIR/edge_case_expanded"
mkdir -p "/home/vivi/datasets/consolidated/books"
mkdir -p "/home/vivi/datasets/consolidated/transcripts"

echo "🔥 Downloading Nightmare Fuel Scenarios..."
aws s3 sync "s3://$BUCKET/datasets/training_v2/stage3_edge_crisis/" "$DOWNLOAD_DIR/nightmare_scenarios/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*.jsonl"

echo "📝 Downloading Transcripts..."
aws s3 sync "s3://$BUCKET/datasets/consolidated/transcripts/" "/home/vivi/datasets/consolidated/transcripts/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*.md" --include "*.txt"

echo "⚠️ Downloading Extra Crisis/Edge Datasets..."
FILES=(
    "datasets/gdrive/tier3_edge_crisis/crisis_detection_conversations.jsonl"
    "datasets/consolidated/edge_cases/edge_case_output/priority_edge_cases_nvidia.jsonl"
    "datasets/consolidated/conversations/edge_case_dialogues.jsonl"
    "datasets/consolidated/edge_cases/existing_edge_cases.jsonl"
    "datasets/gdrive/raw/dataset_pipeline/crisis_intervention_conversations_dataset.json"
)

for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "   Downloading $filename..."
    aws s3 cp "s3://$BUCKET/$file" "$DOWNLOAD_DIR/edge_case_expanded/$filename" \
        --endpoint-url "$ENDPOINT"
done

echo "📚 Downloading Books..."
BOOKS=(
    "datasets/gdrive/raw/Diagnostic and Statistical Manual of... (Z-Library).pdf"
    "datasets/consolidated/datasets/gifts_of_imperfection-brene-brown.pdf"
    "datasets/consolidated/datasets/myth_of_normal-gabor-mate.pdf"
)

for book in "${BOOKS[@]}"; do
    filename=$(basename "$book")
    echo "   Downloading $filename..."
    aws s3 cp "s3://$BUCKET/$book" "/home/vivi/datasets/consolidated/books/$filename" \
        --endpoint-url "$ENDPOINT"
done

echo "✅ Download complete."
