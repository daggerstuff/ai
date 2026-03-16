#!/bin/bash
#
# VPS Setup Script
# One-time setup to prepare the VPS for dataset acquisition
#
# Usage: Run from local machine
#   ./ai/pipelines/orchestrator/scripts/setup_vps.sh

set -euo pipefail

VPS_USER="vivi"
VPS_HOST="146.71.78.184"
SSH_KEY="$HOME/.ssh/planet"
WORKSPACE_DIR="~/pixelated-datasets"

echo "Setting up VPS for dataset acquisition..."
echo ""

# Create directory structure on VPS
echo "Creating directory structure on VPS..."
ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" "mkdir -p $WORKSPACE_DIR && chmod 755 $WORKSPACE_DIR"

# Upload the acquisition script
echo "Uploading vps_dataset_acquisition.sh..."
scp -i "$SSH_KEY" "$(dirname "$0")/vps_dataset_acquisition.sh" "$VPS_USER@$VPS_HOST:$WORKSPACE_DIR/"

# Make it executable
echo "Making script executable..."
ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" "chmod +x $WORKSPACE_DIR/vps_dataset_acquisition.sh"

echo ""
echo "✓ VPS setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run the acquisition script:"
echo "     ssh -i $SSH_KEY $VPS_USER@$VPS_HOST \"cd $WORKSPACE_DIR && nohup ./vps_dataset_acquisition.sh > download_nohup.out 2>&1 &\""
echo ""
echo "  2. Check status:"
echo "     ssh -i $SSH_KEY $VPS_USER@$VPS_HOST \"$WORKSPACE_DIR/vps_dataset_acquisition.sh --status\""
echo ""

