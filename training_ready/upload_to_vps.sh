#!/bin/bash
# Upload tarball to VPS

set -e

VPS_IP="146.71.78.184"
VPS_USER="vivi"
VPS_KEY="${HOME}/.ssh/planet"
TARBALL=$(ls -t /home/vivi/pixelated/training_ready_vps_*.tar.gz | head -1)
VPS_PATH="~/training_ready/"

echo "🚀 Uploading tarball to VPS..."
echo ""
echo "VPS: ${VPS_USER}@${VPS_IP}"
echo "Tarball: ${TARBALL}"
echo "Destination: ${VPS_PATH}"
echo ""

# Check if key exists
if [ ! -f "$VPS_KEY" ]; then
    echo "❌ SSH key not found: $VPS_KEY"
    echo "   Please specify the correct path to your SSH key"
    exit 1
fi

# Check if tarball exists
if [ ! -f "$TARBALL" ]; then
    echo "❌ Tarball not found: $TARBALL"
    exit 1
fi

echo "📦 Uploading..."
scp -i "$VPS_KEY" "$TARBALL" "${VPS_USER}@${VPS_IP}:${VPS_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Upload successful!"
    echo ""
    echo "📋 Next steps on VPS:"
    echo "   ssh -i $VPS_KEY ${VPS_USER}@${VPS_IP}"
    echo "   cd ${VPS_PATH}"
    echo "   tar -xzf training_ready_vps_*.tar.gz"
    echo "   cd ai/training_ready"
    echo "   cat README_VPS.md"
else
    echo ""
    echo "❌ Upload failed. Check:"
    echo "   - SSH key permissions: chmod 600 $VPS_KEY"
    echo "   - VPS connectivity: ping $VPS_IP"
    echo "   - SSH access: ssh -i $VPS_KEY ${VPS_USER}@${VPS_IP}"
    exit 1
fi

