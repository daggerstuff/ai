#!/bin/bash
# Upload tarball to VPS with progress monitoring

set -e

VPS_IP="146.71.78.184"
VPS_USER="vivi"
VPS_KEY="${HOME}/.ssh/planet"
TARBALL=$(ls -t /home/vivi/pixelated/training_ready_vps_*.tar.gz | head -1)
VPS_PATH="~/training_ready/"

echo "🚀 Uploading tarball to VPS with progress..."
echo ""
echo "VPS: ${VPS_USER}@${VPS_IP}"
echo "Tarball: ${TARBALL}"
echo "Size: $(du -h "$TARBALL" | cut -f1)"
echo "Destination: ${VPS_PATH}"
echo ""

# Check if key exists
if [ ! -f "$VPS_KEY" ]; then
    echo "❌ SSH key not found: $VPS_KEY"
    exit 1
fi

# Check if tarball exists
if [ ! -f "$TARBALL" ]; then
    echo "❌ Tarball not found: $TARBALL"
    exit 1
fi

# Create directory on VPS
echo "📁 Creating directory on VPS..."
ssh -i "$VPS_KEY" "${VPS_USER}@${VPS_IP}" "mkdir -p ${VPS_PATH}"

# Check if rsync is available (better progress)
if command -v rsync &> /dev/null; then
    echo "📦 Uploading with rsync (shows progress)..."
    echo ""
    rsync -avz --progress -e "ssh -i $VPS_KEY" \
        "$TARBALL" \
        "${VPS_USER}@${VPS_IP}:${VPS_PATH}"
elif command -v pv &> /dev/null; then
    echo "📦 Uploading with pv (shows progress)..."
    echo ""
    pv "$TARBALL" | ssh -i "$VPS_KEY" "${VPS_USER}@${VPS_IP}" "cat > ${VPS_PATH}$(basename $TARBALL)"
else
    echo "📦 Uploading with scp (verbose mode)..."
    echo "   (Note: Install 'rsync' or 'pv' for better progress display)"
    echo ""
    scp -v -i "$VPS_KEY" "$TARBALL" "${VPS_USER}@${VPS_IP}:${VPS_PATH}"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Upload successful!"
    echo ""
    echo "📋 Verify on VPS:"
    echo "   ssh -i $VPS_KEY ${VPS_USER}@${VPS_IP}"
    echo "   ls -lh ${VPS_PATH}"
    echo ""
    echo "📋 Next steps:"
    echo "   cd ${VPS_PATH}"
    echo "   tar -xzf training_ready_vps_*.tar.gz"
    echo "   cd ai/training_ready"
    echo "   cat README_VPS.md"
else
    echo ""
    echo "❌ Upload failed"
    exit 1
fi

