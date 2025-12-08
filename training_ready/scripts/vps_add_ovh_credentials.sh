#!/usr/bin/env zsh
# Add OVH S3 credentials to VPS ai/.env file
# Run this on the VPS: /home/vivi/pixelated/ai/training_ready/scripts/vps_add_ovh_credentials.sh

set -e

ENV_FILE="/home/vivi/pixelated/ai/.env"

# Credentials retrieved from OVH CLI
ACCESS_KEY="a0ce13472d2d4ad18501899c066ef04a"
SECRET_KEY="dd21a7515fd849e58f8547fde3882a3f"

echo "🔑 Adding OVH S3 credentials to VPS $ENV_FILE"
echo ""

# Check if file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [[ -f "$(dirname $ENV_FILE)/.env.example" ]]; then
        cp "$(dirname $ENV_FILE)/.env.example" "$ENV_FILE"
        echo "✅ Created .env file from .env.example"
    else
        echo "❌ .env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Check if credentials already exist
if grep -q "OVH_S3_ACCESS_KEY=" "$ENV_FILE" 2>/dev/null; then
    echo "📝 Updating existing OVH_S3_ACCESS_KEY..."
    sed -i "s|OVH_S3_ACCESS_KEY=.*|OVH_S3_ACCESS_KEY=$ACCESS_KEY|" "$ENV_FILE"
else
    echo "➕ Adding OVH_S3_ACCESS_KEY..."
    echo "" >> "$ENV_FILE"
    echo "OVH_S3_ACCESS_KEY=$ACCESS_KEY" >> "$ENV_FILE"
fi

if grep -q "OVH_S3_SECRET_KEY=" "$ENV_FILE" 2>/dev/null; then
    echo "📝 Updating existing OVH_S3_SECRET_KEY..."
    sed -i "s|OVH_S3_SECRET_KEY=.*|OVH_S3_SECRET_KEY=$SECRET_KEY|" "$ENV_FILE"
else
    echo "➕ Adding OVH_S3_SECRET_KEY..."
    echo "OVH_S3_SECRET_KEY=$SECRET_KEY" >> "$ENV_FILE"
fi

# Ensure bucket and endpoint are set
if ! grep -q "OVH_S3_BUCKET=" "$ENV_FILE" 2>/dev/null; then
    echo "➕ Adding OVH_S3_BUCKET..."
    echo "OVH_S3_BUCKET=pixel-data" >> "$ENV_FILE"
fi

if ! grep -q "OVH_S3_ENDPOINT=" "$ENV_FILE" 2>/dev/null; then
    echo "➕ Adding OVH_S3_ENDPOINT..."
    echo "OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us" >> "$ENV_FILE"
fi

if ! grep -q "OVH_S3_REGION=" "$ENV_FILE" 2>/dev/null; then
    echo "➕ Adding OVH_S3_REGION..."
    echo "OVH_S3_REGION=us-east-va" >> "$ENV_FILE"
fi

echo ""
echo "✅ Credentials added/updated successfully!"
echo ""
echo "📋 Summary:"
echo "   OVH_S3_ACCESS_KEY=$ACCESS_KEY"
echo "   OVH_S3_SECRET_KEY=$SECRET_KEY"
echo "   OVH_S3_BUCKET=pixel-data"
echo "   OVH_S3_ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us"
echo "   OVH_S3_REGION=us-east-va"
echo ""
echo "🧪 Test the connection on VPS:"
echo "   cd /home/vivi/pixelated"
echo "   set -a && source ai/.env && set +a"
echo "   uv run python3 ai/training_ready/scripts/test_ovh_s3.py"

