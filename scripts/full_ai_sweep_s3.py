import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

# --- Configuration ---
BUCKET_NAME = "pixel-data"
ENDPOINT_URL = "https://hel1.your-objectstorage.com"
REGION_NAME = "hel1"
UPLOAD_PREFIX = "full_ai_sweep/"
MAX_WORKERS = 1  # Sequential for stability as requested
SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

PROJECT_ROOT = Path("/home/vivi/pixelated/ai")

# Load .env from ai/ directory or root
env_paths = [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break


def get_s3_client():
    access_key = os.environ.get("HETZNER_S3_ACCESS_KEY")
    secret_key = os.environ.get("HETZNER_S3_SECRET_KEY")

    if not access_key or not secret_key:
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def scan_for_large_files(root):
    """Finds all unique files > 100MB in the root directory."""
    seen_sizes = {}  # size -> first_path_found
    to_upload = []
    skipped_duplicates = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Skip virtualenvs or git internals if necessary
        if ".venv" in path.parts or ".git" in path.parts:
            continue

        size = path.stat().st_size
        if size > SIZE_THRESHOLD:
            if size in seen_sizes:
                skipped_duplicates.append((path, seen_sizes[size]))
            else:
                seen_sizes[size] = path
                to_upload.append(path)

    return to_upload, skipped_duplicates


def upload_file(s3_client, file_path, root_path):
    # Determine the S3 key
    relative_path = file_path.relative_to(root_path.parent)
    s3_key = f"{UPLOAD_PREFIX}{relative_path}"

    file_size = file_path.stat().st_size
    f"{file_size / (1024 * 1024):.2f} MB"

    max_retries = 5
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            # Check if exists on S3
            try:
                head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                if head["ContentLength"] == file_size:
                    return True
            except Exception:
                pass  # Not on S3 or error checking

            s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
            return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False
    return False


def main():

    to_upload, _duplicates = scan_for_large_files(PROJECT_ROOT)

    sum(f.stat().st_size for f in to_upload)

    if not to_upload:
        return

    s3 = get_s3_client()

    for f in to_upload:
        upload_file(s3, f, PROJECT_ROOT)


if __name__ == "__main__":
    main()
