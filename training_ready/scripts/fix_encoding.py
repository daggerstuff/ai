#!/usr/bin/env python3
"""
Fix Encoding Issues - Detect and fix encoding problems in S3 datasets
Converts files to UTF-8 encoding
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chardet
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# Add project root to path
script_path: Path = Path(__file__).resolve()
project_root: Path = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader  # noqa: E402

logging.getLogger("ai.training_ready.utils.s3_dataset_loader").setLevel(logging.ERROR)

DEFAULT_S3_BUCKET = "pixel-data"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
PROGRESS_INTERVAL = 10000  # Show progress every N lines


class OutputHandler:
    """Handles CLI output with proper formatting"""

    @staticmethod
    def info(message: str, end: str = "\n", flush: bool = False) -> None:
        """Print info message"""
        print(message, end=end, flush=flush)  # noqa: T201

    @staticmethod
    def success(message: str) -> None:
        """Print success message"""
        print(f"     ✅ {message}")  # noqa: T201

    @staticmethod
    def warning(message: str) -> None:
        """Print warning message"""
        print(f"     ⚠️  {message}")  # noqa: T201

    @staticmethod
    def error(message: str) -> None:
        """Print error message"""
        print(f"     ❌ {message}")  # noqa: T201

    @staticmethod
    def header(message: str) -> None:
        """Print header message"""
        print(message)  # noqa: T201

    @staticmethod
    def separator() -> None:
        """Print separator line"""
        print("=" * 80)  # noqa: T201


def detect_encoding(content: bytes) -> tuple[str, float]:
    """
    Detect encoding of content.

    Returns:
        (encoding, confidence) - encoding is guaranteed to be a string
        (defaults to 'utf-8')
    """
    result = chardet.detect(content)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0.0)
    return encoding, confidence


def try_decode(content: bytes, encodings: list[str] | None = None) -> tuple[str | None, str | None]:
    """
    Try to decode content with various encodings.

    Returns:
        (decoded_text, encoding_used) or (None, None) if all fail
    """
    if encodings is None:
        encodings = [
            "utf-8",
            "latin-1",
            "cp1252",
            "iso-8859-1",
            "windows-1252",
            "utf-16",
        ]

    for encoding in encodings:
        try:
            text = content.decode(encoding)
            return text, encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return None, None


def download_with_retry(
    loader: S3DatasetLoader,
    bucket: str,
    key: str,
    output: OutputHandler,
) -> bytes | None:
    """Download file from S3 with retry logic"""
    retry_delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                output.info(f"Retry {attempt}/{MAX_RETRIES}...", end="", flush=True)
            response = loader.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except (EndpointConnectionError, ReadTimeoutError, ClientError) as e:
            if attempt < MAX_RETRIES:
                error_msg = str(e)[:50]
                output.warning(f"Connection error (attempt {attempt}): {error_msg}...")
                output.info(f"Waiting {retry_delay}s...", end="", flush=True)
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return None
    return None


class UploadConfig:
    """Configuration for S3 upload operations"""

    def __init__(
        self,
        loader: S3DatasetLoader,
        bucket: str,
        key: str,
        content: bytes,
        content_type: str,
    ):
        self.loader = loader
        self.bucket = bucket
        self.key = key
        self.content = content
        self.content_type = content_type


def upload_with_retry(config: UploadConfig, output: OutputHandler) -> bool:
    """Upload file to S3 with retry logic"""
    retry_delay = INITIAL_RETRY_DELAY

    for upload_attempt in range(1, MAX_RETRIES + 1):
        try:
            config.loader.s3_client.put_object(
                Bucket=config.bucket,
                Key=config.key,
                Body=config.content,
                ContentType=config.content_type,
            )
            return True
        except (EndpointConnectionError, ReadTimeoutError, ClientError):
            if upload_attempt < MAX_RETRIES:
                output.info("Upload failed, retrying...", end="", flush=True)
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise
    return False


def check_utf8_validity(content: bytes, encoding_used: str) -> bool:
    """Check if content is valid UTF-8"""
    if encoding_used != "utf-8":
        return False
    try:
        content.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def parse_jsonl_lines(
    text: str, output: OutputHandler
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse JSONL lines and return valid entries and invalid lines"""
    lines = text.splitlines()
    valid_entries = []
    invalid_lines = []

    output.info(f"Parsing {len(lines):,} lines...", end="", flush=True)

    for line_num, raw_line in enumerate(lines, 1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        try:
            entry = json.loads(stripped_line)
            valid_entries.append(entry)
        except json.JSONDecodeError as e:
            invalid_lines.append(
                {
                    "line": line_num,
                    "error": str(e),
                    "preview": stripped_line[:100],
                }
            )
            if line_num <= 5:
                output.warning(f"Line {line_num}: JSON decode error - {e}")

        if line_num % PROGRESS_INTERVAL == 0:
            output.info(f" ({line_num:,}/{len(lines):,})...", end="", flush=True)

    output.info(f" ✅ ({len(valid_entries):,} valid entries)")
    if invalid_lines:
        output.warning(f"{len(invalid_lines)} invalid JSON lines found")

    return valid_entries, invalid_lines


def process_jsonl_file(
    loader: S3DatasetLoader,
    s3_path: str,
    dry_run: bool,
    output: OutputHandler,
) -> dict[str, Any]:
    """Process a JSONL file for encoding fixes"""
    bucket, key = loader._parse_s3_path(s3_path)
    filename = Path(key).name

    output.info(f"\n  📝 Processing: {filename}")

    # Download with retry
    content = download_with_retry(loader, bucket, key, output)
    if content is None:
        return {
            "success": False,
            "error": f"Connection failed after {MAX_RETRIES} attempts",
        }

    try:
        # Detect and decode
        detected_encoding, confidence = detect_encoding(content)
        output.info(f"     Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

        text, encoding_used = try_decode(content)
        if text is None or encoding_used is None:
            return {
                "success": False,
                "error": "Could not decode with any known encoding",
                "detected_encoding": detected_encoding,
            }

        # Check if already UTF-8
        if check_utf8_validity(content, encoding_used):
            output.success("Already UTF-8")
            return {
                "success": True,
                "skipped": True,
                "reason": "Already UTF-8",
                "encoding": "utf-8",
            }

        if encoding_used != "utf-8":
            output.warning(f"File is {encoding_used}, converting to UTF-8...")
        else:
            output.warning(f"UTF-8 detection failed, but content decoded as {encoding_used}")

        # Parse JSONL
        valid_entries, invalid_lines = parse_jsonl_lines(text, output)

        if dry_run:
            output.info(f"[DRY RUN] Would convert {len(valid_entries)} entries to UTF-8")
            return {
                "success": True,
                "dry_run": True,
                "original_encoding": encoding_used,
                "entries_count": len(valid_entries),
                "invalid_lines": len(invalid_lines),
            }

        # Re-encode as UTF-8
        output.info("Encoding to UTF-8...", end="", flush=True)
        utf8_content = "\n".join(
            json.dumps(entry, ensure_ascii=False) for entry in valid_entries
        ).encode("utf-8")
        output.info(" ✅")

        # Upload with retry
        output.info("Uploading to S3...", end="", flush=True)
        upload_config = UploadConfig(
            loader=loader,
            bucket=bucket,
            key=key,
            content=utf8_content,
            content_type="application/x-ndjson",
        )
        upload_with_retry(upload_config, output)
        output.info(" ✅")

        output.info(f"Converted {len(valid_entries)} entries to UTF-8")

        return {
            "success": True,
            "dry_run": False,
            "original_encoding": encoding_used,
            "entries_count": len(valid_entries),
            "invalid_lines": len(invalid_lines),
            "bytes_before": len(content),
            "bytes_after": len(utf8_content),
        }

    except Exception as e:
        output.error(f"Error: {e}")
        return {"success": False, "error": str(e)}


def process_json_file(
    loader: S3DatasetLoader,
    s3_path: str,
    dry_run: bool,
    output: OutputHandler,
) -> dict[str, Any]:
    """Process a JSON file for encoding fixes."""
    # ruff: noqa: PLR0911

    def _warn_about_encoding(*, encoding_used: str) -> None:
        if encoding_used != "utf-8":
            output.warning(f"File is {encoding_used}, converting to UTF-8...")
        else:
            output.warning(f"UTF-8 detection failed, but content decoded as {encoding_used}")

    def _parse_json(*, text: str, encoding_used: str) -> dict[str, Any]:
        try:
            return {"success": True, "data": json.loads(text), "encoding_used": encoding_used}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON decode error: {e}", "encoding": encoding_used}

    def _encode_utf8(*, data: Any) -> bytes:
        return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    bucket, key = loader._parse_s3_path(s3_path)
    output.info(f"\n  📝 Processing: {Path(key).name}")

    content = download_with_retry(loader, bucket, key, output)
    if content is None:
        return {"success": False, "error": f"Connection failed after {MAX_RETRIES} attempts"}

    try:
        detected_encoding, confidence = detect_encoding(content)
        output.info(f"     Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

        text, encoding_used = try_decode(content)
        if text is None or encoding_used is None:
            return {
                "success": False,
                "error": "Could not decode with any known encoding",
                "detected_encoding": detected_encoding,
            }

        if check_utf8_validity(content, encoding_used):
            output.success("Already UTF-8")
            return {
                "success": True,
                "skipped": True,
                "reason": "Already UTF-8",
                "encoding": "utf-8",
            }

        _warn_about_encoding(encoding_used=encoding_used)

        parse_result = _parse_json(text=text, encoding_used=encoding_used)
        if not parse_result.get("success"):
            return parse_result

        data = parse_result["data"]
        entries_count = len(data) if isinstance(data, list) else 1

        if dry_run:
            output.info("[DRY RUN] Would convert to UTF-8")
            return {
                "success": True,
                "dry_run": True,
                "original_encoding": encoding_used,
                "entries_count": entries_count,
            }

        output.info("Encoding to UTF-8...", end="", flush=True)
        utf8_content = _encode_utf8(data=data)
        output.info(" ✅")

        output.info("Uploading to S3...", end="", flush=True)
        upload_config = UploadConfig(
            loader=loader,
            bucket=bucket,
            key=key,
            content=utf8_content,
            content_type="application/json",
        )
        upload_with_retry(upload_config, output)
        output.info(" ✅")

        output.info(f"Converted to UTF-8 ({entries_count} entries)")
        return {
            "success": True,
            "dry_run": False,
            "original_encoding": encoding_used,
            "entries_count": entries_count,
            "bytes_before": len(content),
            "bytes_after": len(utf8_content),
        }
    except Exception as e:
        output.error(f"Error: {e}")
        return {"success": False, "error": str(e)}


def find_files_with_encoding_issues() -> list[dict[str, Any]]:
    """
    Find files that likely have encoding issues based on previous scan results.

    NOTE: This function used to consult a local `s3_manifest.json`. We now keep the
    fix workflow fully S3-backed so it can be run from machines that do not have
    local dataset artifacts.
    """
    problematic_files = [
        "datasets/gdrive/processed/phase_1_priority_conversations/"
        "task_5_1_priority_1/priority_1_conversations.jsonl",
        "datasets/gdrive/processed/phase_1_priority_conversations/"
        "task_5_2_priority_2/priority_2_conversations.jsonl",
        "datasets/gdrive/processed/phase_1_priority_conversations/"
        "task_5_3_priority_3/priority_3_conversations.jsonl",
        "datasets/gdrive/processed/phase_1_priority_conversations/"
        "task_5_6_unified_priority/unified_priority_conversations.jsonl",
        "datasets/gdrive/processed/phase_2_professional_datasets/"
        "task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
        "datasets/gdrive/processed/phase_2_professional_datasets/"
        "task_5_11_llama3_mental_counseling/"
        "llama3_mental_counseling_conversations.jsonl",
        "datasets/gdrive/processed/phase_3_cot_reasoning/"
        "task_5_25_tot_reasoning/tot_reasoning_conversations.jsonl",
        "datasets/gdrive/processed/phase_4_reddit_mental_health/"
        "task_5_27_condition_specific/condition_specific_conversations.jsonl",
        "datasets/gdrive/processed/phase_4_reddit_mental_health/"
        "task_5_28_specialized_populations/"
        "specialized_populations_conversations.jsonl",
        "datasets/gdrive/processed/phase_4_reddit_mental_health/"
        "task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
        "datasets/gdrive/processed/phase_4_reddit_mental_health/"
        "task_5_29_temporal_analysis/temporal_analysis_data.jsonl",
        "datasets/gdrive/processed/professional_datasets_final/"
        "soulchat_2_0_complete_no_limits.jsonl",
        "datasets/gdrive/processed/soulchat_complete/soulchat_2_0_complete_no_limits.jsonl",
    ]
    return [{"key": key, "category": "known_problematic", "size": 0} for key in problematic_files]


def list_s3_files_in_prefix(
    *,
    loader: S3DatasetLoader,
    prefix: str,
    category: str,
    output: OutputHandler,
) -> list[dict[str, Any]]:
    """List JSON/JSONL files in S3 under a prefix (no local manifest required)."""
    paginator = loader.s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=loader.bucket, Prefix=prefix)
    results: list[dict[str, Any]] = []

    output.info(f"Listing S3 prefix: s3://{loader.bucket}/{prefix}")

    for page in pages:
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if not isinstance(key, str):
                continue
            if not key.endswith((".json", ".jsonl")):
                continue
            results.append(
                {
                    "key": key,
                    "category": category,
                    "size": int(obj.get("Size", 0) or 0),
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fix encoding issues in S3 datasets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without actually fixing",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Fix specific file (S3 key path, or full s3://bucket/key)",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Fix all .json/.jsonl files under S3 prefix datasets/gdrive/processed/<category>/",
    )
    parser.add_argument(
        "--all-problematic",
        action="store_true",
        help="Fix all known problematic files from scan",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser.parse_args()


def collect_files_to_fix(
    *,
    args: argparse.Namespace,
    loader: S3DatasetLoader,
    output: OutputHandler,
) -> list[dict[str, Any]]:
    """Collect files that need to be fixed based on arguments"""
    files_to_fix: list[dict[str, Any]] = []

    if args.file:
        # Allow either a full s3:// path or a bucket-relative key.
        # Downstream processing normalizes this before download/upload.
        files_to_fix = [{"key": args.file, "category": "manual", "size": 0}]
    elif args.category:
        # Category maps to a known processed prefix in the bucket.
        prefix = f"datasets/gdrive/processed/{args.category.strip().strip('/')}/"
        files_to_fix = list_s3_files_in_prefix(
            loader=loader,
            prefix=prefix,
            category=args.category,
            output=output,
        )
    elif args.all_problematic:
        # Fully S3-backed list: no local manifest required.
        files_to_fix = find_files_with_encoding_issues()

    return files_to_fix


def confirm_proceed(*, dry_run: bool, confirm: bool, output: OutputHandler) -> bool:
    """Confirm before proceeding with fixes"""
    if dry_run or confirm:
        return True

    output.separator()
    response = input("⚠️  Proceed with encoding fixes? (yes/no): ")
    if response.lower() != "yes":
        output.error("Cancelled")
        return False

    return True


def process_files(
    *,
    files_to_fix: list[dict[str, Any]],
    loader: S3DatasetLoader,
    dry_run: bool,
    output: OutputHandler,
) -> list[dict[str, Any]]:
    """Process all files that need encoding fixes"""
    results: list[dict[str, Any]] = []
    total_files = len(files_to_fix)

    for idx, file_info in enumerate(files_to_fix, 1):
        output.info(f"\n[{idx}/{total_files}] ", end="", flush=True)
        raw_key = str(file_info["key"])
        if raw_key.startswith("s3://"):
            s3_path = raw_key
        else:
            s3_path = f"s3://{loader.bucket}/{raw_key.lstrip('/')}"

        if s3_path.endswith(".jsonl"):
            result = process_jsonl_file(loader, s3_path, dry_run, output)
        elif s3_path.endswith(".json"):
            result = process_json_file(loader, s3_path, dry_run, output)
        else:
            result = {"success": False, "error": "Unknown file type"}

        result["file"] = Path(raw_key).name
        result["category"] = file_info["category"]
        results.append(result)

        # Show progress summary
        successful = sum(bool(r.get("success")) for r in results)
        output.info(f"     Progress: {successful}/{idx} successful")

    return results


def print_results(results: list[dict[str, Any]], output: OutputHandler) -> None:
    """Print summary of encoding fix results"""
    output.info("")
    output.separator()
    output.header("📊 ENCODING FIX RESULTS")
    output.separator()

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    skipped = [r for r in successful if r.get("skipped")]
    fixed = [r for r in successful if not r.get("skipped")]

    output.info(f"\n✅ Fixed: {len(fixed)} files")
    output.info(f"⏭️  Skipped (already UTF-8): {len(skipped)} files")
    output.info(f"❌ Failed: {len(failed)} files")

    if fixed:
        output.info("\n📊 Fixed files:")
        for result in fixed:
            encoding = result.get("original_encoding", "unknown")
            entries = result.get("entries_count", 0)
            file_name = result.get("file", "unknown")
            output.info(f"   - {file_name}: {encoding} → UTF-8 ({entries} entries)")

    if failed:
        output.info("\n❌ Failed files:")
        for result in failed:
            file_name = result.get("file", "unknown")
            error = result.get("error", "Unknown error")
            output.info(f"   - {file_name}: {error}")


def save_results(
    *,
    project_root: Path,
    dry_run: bool,
    results: list[dict[str, Any]],
) -> Path:
    """Save encoding fix results to JSON file"""
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    skipped = [r for r in successful if r.get("skipped")]
    fixed = [r for r in successful if not r.get("skipped")]

    results_path = project_root / "ai/training_ready/data/encoding_fix_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dry_run": dry_run,
                "results": results,
                "summary": {
                    "total": len(results),
                    "fixed": len(fixed),
                    "skipped": len(skipped),
                    "failed": len(failed),
                },
            },
            f,
            indent=2,
        )

    return results_path


def main() -> None:
    """Main encoding fix function"""
    args = parse_args()
    output = OutputHandler()

    output.header("🔧 Dataset Encoding Fix")
    output.separator()

    if args.dry_run:
        output.warning("DRY RUN MODE - No files will be modified")
    else:
        output.warning("LIVE MODE - Files will be converted to UTF-8 in S3!")

    # Initialize S3 loader
    loader = S3DatasetLoader(bucket=DEFAULT_S3_BUCKET)

    # Determine files to fix
    files_to_fix = collect_files_to_fix(
        args=args,
        loader=loader,
        output=output,
    )

    if not (args.file or args.category or args.all_problematic):
        output.error("Specify --file, --category, or --all-problematic")
        return

    if not files_to_fix:
        output.error("No files found to fix")
        return

    output.info(f"\n📋 Found {len(files_to_fix)} files to check/fix")

    # Confirm before proceeding
    if not confirm_proceed(dry_run=args.dry_run, confirm=args.confirm, output=output):
        return

    # Process files
    output.info("\n🔧 Processing files...")
    output.info(f"   Total files: {len(files_to_fix)}")
    output.separator()
    results = process_files(
        files_to_fix=files_to_fix,
        loader=loader,
        dry_run=args.dry_run,
        output=output,
    )

    # Print summary
    print_results(results, output)

    # Save results
    results_path = save_results(project_root=project_root, dry_run=args.dry_run, results=results)
    output.info(f"\n💾 Results saved to: {results_path}")
    output.info("\n✅ Encoding fix complete!")


if __name__ == "__main__":
    main()
