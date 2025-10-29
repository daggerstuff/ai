#!/bin/bash
# Compress training package for transfer
set -euo pipefail

echo "Compressing training package..."
if ! command -v 7z &> /dev/null; then
    echo "❌ 7z command not found. Please install p7zip."
    exit 1
fi

7z a -t7z -m0=lzma2 -mx=9 -mfb=64 -md=32m -ms=on "therapeutic_ai_training_package_20251028_204104.7z" "therapeutic_ai_training_package_20251028_204104"/

if [ $? -eq 0 ]; then
    echo "✅ Package created: therapeutic_ai_training_package_20251028_204104.7z"
    echo "📦 Size: $(du -h 'therapeutic_ai_training_package_20251028_204104.7z' | cut -f1)"
    echo ""
    echo "To extract on Lightning.ai:"
    echo "  7z x 'therapeutic_ai_training_package_20251028_204104.7z'"
    echo "  cd 'therapeutic_ai_training_package_20251028_204104/'"
    echo "  cat README.md"
else
    echo "❌ Compression failed"
    exit 1
fi
