#!/usr/bin/env zsh
# Run the VPS credential setup script remotely via SSH

set -e

VPS_HOST="146.71.78.184"
VPS_USER="vivi"
SSH_KEY="planet"
VPS_SCRIPT="/home/vivi/pixelated/ai/training_ready/scripts/vps_add_ovh_credentials.sh"

echo "🚀 Setting up OVH S3 credentials on VPS..."
echo ""

# First, upload the script if it doesn't exist or is outdated
echo "📤 Ensuring script is on VPS..."
scp -i ~/.ssh/$SSH_KEY \
    "$(dirname $0)/vps_add_ovh_credentials.sh" \
    $VPS_USER@$VPS_HOST:$VPS_SCRIPT

echo ""
echo "🔧 Running credential setup on VPS..."
echo ""

# Run the script on VPS
ssh -i ~/.ssh/$SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
cd /home/vivi/pixelated
chmod +x ai/training_ready/scripts/vps_add_ovh_credentials.sh
./ai/training_ready/scripts/vps_add_ovh_credentials.sh
ENDSSH

echo ""
echo "✅ VPS credentials setup complete!"
echo ""
echo "🧪 Test the connection on VPS:"
echo "   ssh -i ~/.ssh/$SSH_KEY $VPS_USER@$VPS_HOST"
echo "   cd /home/vivi/pixelated"
echo "   set -a && source ai/.env && set +a"
echo "   uv run python3 ai/training_ready/scripts/test_ovh_s3.py"

