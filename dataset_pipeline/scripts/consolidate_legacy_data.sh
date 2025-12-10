#!/bin/bash
set -e

# Target Directory
TARGET_DIR="/home/vivi/pixelated/ai/training_data_consolidated/raw/priority"
mkdir -p "$TARGET_DIR"

echo "📂 Starting Consolidation into $TARGET_DIR..."

# Function to safely copy with checksum verification
safe_copy() {
    src="$1"
    dest_name="$2"
    dest="$TARGET_DIR/$dest_name"

    if [ -f "$src" ]; then
        echo "   Found: $src"
        if [ -f "$dest" ]; then
            src_hash=$(sha256sum "$src" | awk '{print $1}')
            dest_hash=$(sha256sum "$dest" | awk '{print $1}')
            if [ "$src_hash" == "$dest_hash" ]; then
                echo "   ✅ Already exists and identical: $dest_name"
            else
                echo "   ⚠️ Conflict! File exists but differs. Saving as ${dest_name}.new"
                cp "$src" "${dest}.new"
            fi
        else
            cp "$src" "$dest"
            echo "   🚀 Copied to: $dest_name"
        fi
    else
        echo "   ❌ Missing: $src"
    fi
}

echo "1️⃣  Consolidating Priority 1 (Therapeutic)..."
safe_copy "/home/vivi/pixelated/ai/lightning/ghost/Wendy/datasets/priority_1/priority_1_FINAL.jsonl" "priority_1.jsonl"

echo "2️⃣  Consolidating Priority 2 (Reasoning)..."
safe_copy "/home/vivi/pixelated/ai/lightning/ghost/Wendy/datasets/priority_2/priority_2_FINAL.jsonl" "priority_2.jsonl"

echo "3️⃣  Consolidating Priority 3 (Specialized)..."
safe_copy "/home/vivi/pixelated/ai/lightning/ghost/Wendy/datasets/priority_3/priority_3_FINAL.jsonl" "priority_3.jsonl"

echo "-----------------------------------"
echo "🎉 Consolidation Complete."
echo "   Target Contents:"
ls -lh "$TARGET_DIR"
