#!/usr/bin/env python3
"""
Fix Encoding Issues - Detect and fix encoding problems in S3 datasets
Converts files to UTF-8 encoding
"""

import argparse
import codecs
import contextlib
import json
import logging
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import chardet
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# Add project root to path
script_path: Path = Path(__file__).resolve()
project_root: Path = script_path.parents[3]
sys.path.insert(0, str(project_root))

from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader  # noqa: E402

logging.getLogger("ai.training.ready_packages.utils.s3_dataset_loader").setLevel(logging.ERROR)

DEFAULT_S3_BUCKET = "pixel-data"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
PROGRESS_INTERVAL = 10000  # Show progress every N lines
DETECT_SAMPLE_BYTES = 64 * 1024
INVALID_LINE_SAMPLE_LIMIT = 25


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


def _unique_encodings(encodings: list[str]) -> list[str]:
    """Return encodings list de-duped while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for enc in encodings:
        normalized = enc.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def build_decode_candidates(*, detected_encoding: str) -> list[str]:
    """
    Build a prioritized list of encodings to attempt.

    Notes:
    - Always try utf-8 first.
    - Include the detector suggestion early (chardet often guesses MacRoman/cp1252).
    - Keep latin-1 last since it never fails but may misinterpret characters.
    """
    common = [
        "utf-8",
        detected_encoding,
        "macroman",
        "cp1252",
        "windows-1252",
        "iso-8859-1",
        "latin-1",
        "utf-16",
    ]
    return _unique_encodings(common)


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


def download_sample_with_retry(
    *,
    loader: S3DatasetLoader,
    bucket: str,
    key: str,
    output: OutputHandler,
    sample_bytes: int = DETECT_SAMPLE_BYTES,
) -> bytes | None:
    """Download a small prefix sample from S3 (for encoding detection)."""
    retry_delay = INITIAL_RETRY_DELAY

    # S3 Range header is inclusive.
    range_header = f"bytes=0-{max(sample_bytes - 1, 0)}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                output.info(f"Retry {attempt}/{MAX_RETRIES}...", end="", flush=True)

            try:
                response = loader.s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
                return response["Body"].read()
            except ClientError:
                # Some S3 compatibles may not support Range consistently; fallback.
                response = loader.s3_client.get_object(Bucket=bucket, Key=key)
                body = response["Body"]
                try:
                    return body.read(sample_bytes)
                finally:
                    with contextlib.suppress(Exception):
                        body.close()
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


def iter_s3_jsonl_lines_bytes(
    *,
    loader: S3DatasetLoader,
    bucket: str,
    key: str,
) -> Iterator[bytes]:
    """
    Yield JSONL lines as raw bytes from an S3 object body.

    Uses body.iter_lines() when available; falls back to manual buffering.
    """
    response = loader.s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]

    iter_lines = getattr(body, "iter_lines", None)
    if callable(iter_lines):
        for raw_line in body.iter_lines():
            if raw_line:
                yield raw_line
        return

    # Fallback: manual buffering across chunks.
    buffer = BytesIO()
    for chunk in body.iter_chunks(chunk_size=8192):
        buffer.write(chunk)
        while True:
            buffer.seek(0)
            line = buffer.readline()
            if not line:
                buffer = BytesIO()
                break
            if not line.endswith(b"\n"):
                rest = buffer.read()
                buffer = BytesIO(line + rest)
                break
            if stripped := line.rstrip(b"\r\n"):
                yield stripped

            rest = buffer.read()
            buffer = BytesIO(rest)


def check_jsonl_utf8_streaming(
    *,
    loader: S3DatasetLoader,
    bucket: str,
    key: str,
) -> bool:
    """Check UTF-8 validity without loading entire file into memory."""
    for raw_line in iter_s3_jsonl_lines_bytes(loader=loader, bucket=bucket, key=key):
        raw_line.decode("utf-8")  # strict
    return True


def convert_jsonl_streaming_to_utf8(
    *,
    loader: S3DatasetLoader,
    bucket: str,
    key: str,
    encoding_used: str,
    dry_run: bool,
    output: OutputHandler,
) -> dict[str, Any]:
    """
    Convert a JSONL file from `encoding_used` to UTF-8 in a streaming manner.

    - Never stores full file content in memory.
    - In live mode, writes utf-8 JSONL to a temp file and uploads via multipart.
    """
    invalid_samples: list[dict[str, Any]] = []
    invalid_count = 0
    valid_count = 0
    line_num = 0

    tmp_path: str | None = None
    tmp_file = None

    try:
        if not dry_run:
            tmp_file = tempfile.NamedTemporaryFile("wb", delete=False)
            tmp_path = tmp_file.name

        encoder = codecs.getincrementalencoder("utf-8")()

        for raw_line in iter_s3_jsonl_lines_bytes(loader=loader, bucket=bucket, key=key):
            line_num += 1

            try:
                decoded_line = raw_line.decode(encoding_used)
            except (UnicodeDecodeError, LookupError) as e:
                raise UnicodeDecodeError(
                    encoding_used,
                    raw_line,
                    getattr(e, "start", 0),
                    getattr(e, "end", 1),
                    str(e),
                ) from e

            stripped_line = decoded_line.strip()
            if not stripped_line:
                continue

            try:
                entry = json.loads(stripped_line)
                valid_count += 1

                if not dry_run and tmp_file is not None:
                    out_line = json.dumps(entry, ensure_ascii=False) + "\n"
                    tmp_file.write(encoder.encode(out_line))
            except json.JSONDecodeError as e:
                invalid_count += 1
                if len(invalid_samples) < INVALID_LINE_SAMPLE_LIMIT:
                    invalid_samples.append(
                        {
                            "line": line_num,
                            "error": str(e),
                            "preview": stripped_line[:100],
                        }
                    )
                if line_num <= 5:
                    output.warning(f"Line {line_num}: JSON decode error - {e}")

            if line_num % PROGRESS_INTERVAL == 0:
                output.info(f" ({line_num:,} lines)...", end="", flush=True)

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "original_encoding": encoding_used,
                "entries_count": valid_count,
                "invalid_lines": invalid_count,
                "invalid_line_samples": invalid_samples,
            }

        if tmp_file is None or tmp_path is None:
            return {"success": False, "error": "Internal error: temp file not created"}

        tmp_file.flush()
        tmp_file.close()

        # Upload with retry using multipart transfer manager.
        retry_delay = INITIAL_RETRY_DELAY
        for upload_attempt in range(1, MAX_RETRIES + 1):
            try:
                output.info("Uploading to S3...", end="", flush=True)
                with open(tmp_path, "rb") as f:
                    loader.s3_client.upload_fileobj(
                        Fileobj=f,
                        Bucket=bucket,
                        Key=key,
                        ExtraArgs={"ContentType": "application/x-ndjson"},
                    )
                output.info(" ✅")
                break
            except (EndpointConnectionError, ReadTimeoutError, ClientError):
                if upload_attempt < MAX_RETRIES:
                    output.info("Upload failed, retrying...", end="", flush=True)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise

        bytes_after = Path(tmp_path).stat().st_size
        return {
            "success": True,
            "dry_run": False,
            "original_encoding": encoding_used,
            "entries_count": valid_count,
            "invalid_lines": invalid_count,
            "invalid_line_samples": invalid_samples,
            "bytes_after": bytes_after,
        }
    finally:
        if tmp_file is not None:
            with contextlib.suppress(Exception):
                tmp_file.close()
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink(missing_ok=True)


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

    try:
        # Detect encoding from a small sample (memory safe)
        sample = download_sample_with_retry(loader=loader, bucket=bucket, key=key, output=output)
        if sample is None:
            return {
                "success": False,
                "error": f"Connection failed after {MAX_RETRIES} attempts",
            }

        detected_encoding, confidence = detect_encoding(sample)
        output.info(f"     Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

        # Stream-check UTF-8 validity (doesn't build the whole file in RAM)
        with contextlib.suppress(UnicodeDecodeError):
            check_jsonl_utf8_streaming(loader=loader, bucket=bucket, key=key)
            output.success("Already UTF-8")
            return {
                "success": True,
                "skipped": True,
                "reason": "Already UTF-8",
                "encoding": "utf-8",
            }

        candidates = build_decode_candidates(detected_encoding=detected_encoding)
        output.warning(
            f"File is not valid UTF-8, attempting conversion ({', '.join(candidates)})..."
        )

        last_error: str | None = None
        for encoding_used in candidates:
            if encoding_used.lower() == "utf-8":
                continue
            try:
                result = convert_jsonl_streaming_to_utf8(
                    loader=loader,
                    bucket=bucket,
                    key=key,
                    encoding_used=encoding_used,
                    dry_run=dry_run,
                    output=output,
                )
                result["detected_encoding"] = detected_encoding
                return result
            except UnicodeDecodeError as e:
                last_error = f"{encoding_used}: {e}"
                continue

        return {
            "success": False,
            "error": last_error or "Could not decode with any known encoding",
            "detected_encoding": detected_encoding,
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
