import contextlib
import csv as csv_module
import glob
import json
import os
import subprocess

from dataset_pipeline.extractors.book_extractor import BookExtractor
from dataset_pipeline.extractors.dataset_loader import DatasetLoader
from dataset_pipeline.extractors.s3_streamer import S3Streamer
from dataset_pipeline.processors.chatml_converter import ChatMLConverter
from dataset_pipeline.processors.quality_filter import QualityFilter
from dataset_pipeline.processors.safety_processors import HackathonSafetyProcessor

GDRIVE_REMOTES = ["gdrive:", "drive:"]


def categorize_file(filepath):
    lower_f = filepath.lower()
    if "voice" in lower_f or "transcript" in lower_f or "podcast" in lower_f or "youtube" in lower_f:
        return "voice_training"
    if "reasoning" in lower_f or "cot" in lower_f or "sharegpt" in lower_f:
        return "reasoning_enhancement"
    if "persona" in lower_f or "character" in lower_f or "roleplay" in lower_f:
        return "personality_balancing"
    if lower_f.endswith((".pdf", ".epub")):
        return "psychology_knowledge"
    return "mental_health_conversations"


def should_skip(filepath):
    lower_f = filepath.lower()

    # 1. Dev artifacts and binaries — never useful
    dev_patterns = [
        "node_modules",
        ".git",
        ".local/share/pnpm",
        ".npm/",
        "site-packages",
        ".dist-info",
        "__pycache__",
        "distro-info",
        "/backups/coder",
    ]
    if any(p in lower_f for p in dev_patterns):
        return True

    # 2. Non-data file types
    binary_exts = [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".mp4",
        ".mp3",
        ".wav",
        ".zip",
        ".tar",
        ".gz",
        ".whl",
        ".bin",
        ".exe",
        ".so",
        ".dll",
        ".pyc",
        ".lock",
        ".crt",
        ".key",
        ".pem",
        ".md",
        ".txt",
        ".rst",
        ".html",
        ".htm",
        ".js",
        ".ts",
        ".css",
        ".scss",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".sh",
        ".bash",
        ".env",
        ".log",
        ".xml",
        ".svg",
        ".ico",
    ]
    if any(lower_f.endswith(ext) for ext in binary_exts):
        return True

    # 3. Known junk filenames — pipeline artifacts, event logs, etc.
    junk_names = [
        "log.jsonl",
        "events.jsonl",
        "history.jsonl",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "dataset_info.json",
        "hyperparameters.json",
        "training_config.json",
        "acquisition_summary.json",
        "icecraw-aws.csv",
        "clinical_scores.csv",  # score metadata, not training data
        "_test_report.json",
        "conversation_metadata.csv",
    ]
    fname = lower_f.split("/")[-1]
    if any(fname == j or fname.endswith(j) for j in junk_names):
        return True

    # 4. Skip _processed duplicates — these are re-processed versions of existing files
    # Any file with _processed in the name is a derivative of the original
    if "_processed" in lower_f:
        return True

    # 5. Skip the archive/gdrive mirror when scanning S3 —
    # the live GDrive scan will handle those directly without duplication
    # Exception: chad_drive_imported which is NOT on live gdrive
    if lower_f.startswith("archive/gdrive/") and "chad_drive_imported" not in lower_f:
        return True

    # 6. Skip archive/vps_archaeology — these are old VPS backups of data
    # already present under cleaner paths (datasets/, cot_reasoning/, etc.)
    if lower_f.startswith("archive/vps_archaeology/"):
        return True

    # 7. Skip test/ prefix — test files
    parts = lower_f.split("/")
    return parts[0] == "test"


def stream_local_file(filepath, book_ext):
    lower_f = filepath.lower()
    category = categorize_file(filepath)

    if lower_f.endswith(".jsonl"):
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        yield {
                            "raw_data": json.loads(line),
                            "metadata": {"source_family": category, "file_key": filepath},
                        }
    elif lower_f.endswith(".json"):
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield {"raw_data": item, "metadata": {"source_family": category, "file_key": filepath}}
            elif isinstance(data, dict):
                yield {"raw_data": data, "metadata": {"source_family": category, "file_key": filepath}}
        except Exception:
            pass
    elif lower_f.endswith(".csv"):
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    yield {"raw_data": row, "metadata": {"source_family": category, "file_key": filepath}}
        except Exception:
            pass
    elif lower_f.endswith(".epub"):
        try:
            for chunk in book_ext.extract_epub(filepath):
                yield {
                    "raw_data": {"text": chunk},
                    "metadata": {"source_family": "psychology_knowledge", "file_key": filepath},
                }
        except Exception:
            pass
    elif lower_f.endswith(".pdf"):
        try:
            for chunk in book_ext.extract_pdf(filepath):
                yield {
                    "raw_data": {"text": chunk},
                    "metadata": {"source_family": "psychology_knowledge", "file_key": filepath},
                }
        except Exception:
            pass


def stream_s3_records(streamer, data_ext, book_ext):
    s3_files = list(streamer.list_files(""))

    for f in s3_files:
        if should_skip(f):
            continue
        lower_f = f.lower()
        category = categorize_file(f)

        try:
            if lower_f.endswith((".jsonl", ".json")):
                for record in data_ext.load_jsonl(f, category, category):
                    yield record
            elif lower_f.endswith(".csv"):
                for record in data_ext.load_csv(f, category, category):
                    yield record
            elif lower_f.endswith((".epub", ".pdf")):
                # Use /dev/shm (RAM disk) if available to avoid disk quota
                tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
                temp_path = os.path.join(tmp_dir, os.path.basename(f))
                streamer.download_to_file(f, temp_path)
                try:
                    if lower_f.endswith(".epub"):
                        for chunk in book_ext.extract_epub(temp_path):
                            yield {
                                "raw_data": {"text": chunk},
                                "metadata": {"source_family": "psychology_knowledge", "file_key": f},
                            }
                    else:
                        for chunk in book_ext.extract_pdf(temp_path):
                            yield {
                                "raw_data": {"text": chunk},
                                "metadata": {"source_family": "psychology_knowledge", "file_key": f},
                            }
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception:
            pass


def stream_gdrive_records(book_ext):
    """Scans both gdrive remotes directly via rclone lsjson and streams file contents."""

    for remote in GDRIVE_REMOTES:
        try:
            result = subprocess.run(
                ["rclone", "lsjson", "--recursive", "--files-only", remote], capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                continue

            files = json.loads(result.stdout)

            for file_info in files:
                path = file_info["Path"]
                full_path = f"{remote}{path}"
                lower_path = path.lower()

                if should_skip(lower_path):
                    continue
                if not any(lower_path.endswith(ext) for ext in [".jsonl", ".json", ".csv", ".pdf", ".epub"]):
                    continue

                category = categorize_file(path)

                # Use rclone cat to stream the content without saving to disk
                try:
                    if lower_path.endswith(".jsonl"):
                        result = subprocess.run(["rclone", "cat", full_path], capture_output=True, timeout=120)
                        for line in result.stdout.split(b"\n"):
                            line = line.strip()
                            if line:
                                with contextlib.suppress(Exception):
                                    yield {
                                        "raw_data": json.loads(line),
                                        "metadata": {"source_family": category, "file_key": full_path},
                                    }
                    elif lower_path.endswith(".json"):
                        result = subprocess.run(["rclone", "cat", full_path], capture_output=True, timeout=120)
                        try:
                            data = json.loads(result.stdout)
                            if isinstance(data, list):
                                for item in data:
                                    yield {
                                        "raw_data": item,
                                        "metadata": {"source_family": category, "file_key": full_path},
                                    }
                            elif isinstance(data, dict):
                                yield {"raw_data": data, "metadata": {"source_family": category, "file_key": full_path}}
                        except Exception:
                            pass
                    elif lower_path.endswith(".csv"):
                        result = subprocess.run(["rclone", "cat", full_path], capture_output=True, timeout=120)
                        import io

                        reader = csv_module.DictReader(io.StringIO(result.stdout.decode("utf-8", errors="replace")))
                        for row in reader:
                            yield {"raw_data": row, "metadata": {"source_family": category, "file_key": full_path}}
                    elif lower_path.endswith((".epub", ".pdf")):
                        tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
                        temp_path = os.path.join(tmp_dir, os.path.basename(path))
                        subprocess.run(["rclone", "copy", full_path, tmp_dir], timeout=120)
                        try:
                            if lower_path.endswith(".epub"):
                                for chunk in book_ext.extract_epub(temp_path):
                                    yield {
                                        "raw_data": {"text": chunk},
                                        "metadata": {"source_family": "psychology_knowledge", "file_key": full_path},
                                    }
                            else:
                                for chunk in book_ext.extract_pdf(temp_path):
                                    yield {
                                        "raw_data": {"text": chunk},
                                        "metadata": {"source_family": "psychology_knowledge", "file_key": full_path},
                                    }
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                except Exception:
                    pass
        except Exception:
            pass


def stream_local_records(book_ext):
    local_files = glob.glob("/home/vivi/pixelated/ai/data/**/*", recursive=True)

    for f in local_files:
        if not os.path.isfile(f):
            continue
        if should_skip(f):
            continue
        yield from stream_local_file(f, book_ext)


def all_records(streamer, data_ext, book_ext):
    yield from stream_s3_records(streamer, data_ext, book_ext)
    yield from stream_gdrive_records(book_ext)
    yield from stream_local_records(book_ext)


SHARD_SIZE = 50_000
OUTPUT_PREFIX = "final_dataset/v5_shards"
CHECKPOINT_KEY = "final_dataset/v5_checkpoint.json"


def load_checkpoint(streamer):
    """Load the set of already-processed file keys from S3."""
    try:
        response = streamer.client.get_object(Bucket=streamer.bucket, Key=CHECKPOINT_KEY)
        data = json.loads(response["Body"].read())
        done = set(data.get("completed_files", []))
        shard_num = data.get("next_shard", 0)
        return done, shard_num
    except Exception:
        return set(), 0


def save_checkpoint(streamer, completed_files, next_shard):
    """Save checkpoint to S3."""
    data = json.dumps({"completed_files": list(completed_files), "next_shard": next_shard}).encode("utf-8")
    streamer.client.put_object(Bucket=streamer.bucket, Key=CHECKPOINT_KEY, Body=data)


def upload_shard(streamer, shard_records, shard_num, prefix=OUTPUT_PREFIX):
    """Upload a completed shard to S3."""
    key = f"{prefix}/shard_{shard_num:05d}.jsonl"
    body = "\n".join(json.dumps(r) for r in shard_records).encode("utf-8")
    streamer.client.put_object(Bucket=streamer.bucket, Key=key, Body=body)
    return key


def main():
    streamer = S3Streamer()
    DatasetLoader(streamer)
    book_ext = BookExtractor(streamer)
    converter = ChatMLConverter()
    quality = QualityFilter()
    safety = HackathonSafetyProcessor()

    completed_files, next_shard = load_checkpoint(streamer)

    total_raw = 0
    total_valid = 0
    total_routed_toxic = 0
    current_shard = []
    toxic_review_shard = []
    shard_num = next_shard

    def process_source(source_generator, source_label):
        nonlocal total_raw, total_valid, total_routed_toxic, current_shard, toxic_review_shard, shard_num

        for raw_record in source_generator:
            file_key = raw_record.get("metadata", {}).get("file_key", "")

            # Skip already-checkpointed files
            if file_key in completed_files:
                continue

            total_raw += 1
            if total_raw % 10000 == 0:
                pass

            try:
                chatml = converter.convert(raw_record)

                # PIX-4240: safety pass (PII strip + heuristic toxicity) after
                # conversion. Records flagged for toxic review are routed to a
                # separate shard stream and never enter quality filtering here.
                safety_result = safety.process(chatml)
                chatml = safety_result.cleaned_record

                if safety_result.report.routed_to_toxic_review:
                    total_routed_toxic += 1
                    toxic_review_shard.append(chatml)
                    if len(toxic_review_shard) >= SHARD_SIZE:
                        upload_shard(streamer, toxic_review_shard, shard_num, prefix="final_dataset/v5_toxic_review/")
                        shard_num += 1
                        toxic_review_shard = []
                        save_checkpoint(streamer, completed_files, shard_num)
                    continue

                if quality.passes_filter(chatml):
                    total_valid += 1
                    current_shard.append(chatml)

                    # When shard is full, upload it and checkpoint
                    if len(current_shard) >= SHARD_SIZE:
                        upload_shard(streamer, current_shard, shard_num)
                        shard_num += 1
                        current_shard = []
                        # Save checkpoint after every shard
                        save_checkpoint(streamer, completed_files, shard_num)
            except Exception:
                pass

    # Phase 1: Google Drive (both accounts) — S3 was already extracted
    process_source(stream_gdrive_records(book_ext), "GDrive")
    # Phase 2: Local ai/data
    process_source(stream_local_records(book_ext), "Local")

    # Upload the final partial shard if anything remains
    if current_shard:
        upload_shard(streamer, current_shard, shard_num)
        shard_num += 1

    # PIX-4240: flush the toxic_review tail shard if anything remains
    if toxic_review_shard:
        upload_shard(streamer, toxic_review_shard, shard_num, prefix="final_dataset/v5_toxic_review/")
        shard_num += 1

    # Final checkpoint
    save_checkpoint(streamer, completed_files, shard_num)


if __name__ == "__main__":
    main()
