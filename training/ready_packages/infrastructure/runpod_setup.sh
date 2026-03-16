#!/bin/bash
# Setup script for RunPod / Cloud GPU environments to pull S3 canonical dataset

set -e

# Update and install tools
apt-get update && apt-get install -y awscli jq rclone git build-essential

# Configure Git Repo 
echo "Setting up repository..."
if [ ! -d "/workspace/pixelated" ]; then
    git clone https://github.com/vivi/pixelated.git /workspace/pixelated
fi

cd /workspace/pixelated/ai

# Using UV for environment
echo "Installing Python dependencies using UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Fetch the dataset from OVH / S3 to local NVMe storage before epoch 1
echo "Fetching Dataset from S3..."
aws --endpoint-url https://s3.us-east-va.io.cloud.ovh.us s3 cp s3://pixel-data/final_dataset/ /workspace/data_cache/ --recursive

echo "Setup Complete. You can begin Unsloth LoRA Training using the phase 2 configuration."
