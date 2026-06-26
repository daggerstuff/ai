"""
Cleans up /home/vivi/pixelated/ai/data/ by removing files that are:
- Already captured by the pipeline (and thus in S3)
- Old pipeline artifacts, configs, score CSVs
- Anything not raw source material worth keeping for future runs
"""
import os
import shutil
import glob

DATA_DIR = "/home/vivi/pixelated/ai/data"

# Keep these top-level folders intact — they are raw source voice exports
# worth keeping for future pipeline re-runs with updated extractors
KEEP_PREFIXES = [
    "voice",  # any folder with voice in the name
]

# Patterns to always delete
DELETE_PATTERNS = [
    "**/*_processed*.jsonl",
    "**/*_processed*.json",
    "**/*_nemo.jsonl",          # nemo exports — already in S3
    "**/*_review.csv",          # review metadata, not training data
    "**/*_clinical_scores.csv", # scoring metadata
    "**/nemo_export/**",        # entire nemo export directory
    "**/staged_datasets/**",    # old staged outputs — in S3
    "**/compress/**",           # compressed/temp processing artifacts
    "**/joiner/**",             # planning docs, not datasets
]

def should_keep_folder(folder_name):
    lower = folder_name.lower()
    # Keep raw voice persona folders — they are the unique source material
    return 'voice' in lower

def clean():
    deleted_files = 0
    deleted_bytes = 0
    kept_files = 0

    print(f"Scanning {DATA_DIR} ...")

    # 1. Delete by pattern
    for pattern in DELETE_PATTERNS:
        full_pattern = os.path.join(DATA_DIR, pattern)
        for f in glob.glob(full_pattern, recursive=True):
            if os.path.isfile(f):
                size = os.path.getsize(f)
                os.remove(f)
                deleted_files += 1
                deleted_bytes += size
                print(f"  DEL {f} ({size/1024:.1f} KB)")

    # 2. Walk top-level directories
    for entry in os.scandir(DATA_DIR):
        if not entry.is_dir():
            continue
        
        folder = entry.name
        if should_keep_folder(folder):
            print(f"  KEEP {folder}/ (voice source data)")
            # But still clean up score CSVs inside voice folders
            for f in glob.glob(os.path.join(entry.path, "**/*.csv"), recursive=True):
                if 'clinical_scores' in f or 'review' in f:
                    size = os.path.getsize(f)
                    os.remove(f)
                    deleted_files += 1
                    deleted_bytes += size
            kept_files += 1
            continue
        
        # Remove entire non-voice folders that are pipeline artifacts
        artifact_dirs = ['nemo_export', 'staged_datasets', 'compress', 'joiner']
        if folder in artifact_dirs:
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(entry.path)
                for f in files
            )
            shutil.rmtree(entry.path)
            deleted_bytes += size
            print(f"  DEL DIR {folder}/ ({size/1024/1024:.1f} MB)")
            continue

        kept_files += 1
        print(f"  KEEP {folder}/ (manual review recommended)")

    print(f"\n✅ Cleanup complete!")
    print(f"   Files/dirs deleted: {deleted_files}")
    print(f"   Space freed:        {deleted_bytes/1024/1024:.1f} MB")
    print(f"   Folders kept:       {kept_files}")

if __name__ == "__main__":
    clean()
