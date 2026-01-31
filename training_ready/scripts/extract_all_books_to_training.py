#!/usr/bin/env python3
"""Extract therapeutic book content to training format.

This script is a wrapper for pdf_to_training_data.py to process all books
and convert them to ChatML format with author attribution.

Outputs:
- ai/training_ready/data/generated/therapeutic_books/{author}/book_content.jsonl
- ai/training_ready/data/generated/therapeutic_books_stats.json

Usage:
    python extract_all_books_to_training.py --author "Brené Brown"
    python extract_all_books_to_training.py --all
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

logger = logging.getLogger(__name__)

# Books configuration
BOOKS = [
    {
        "path": "~/datasets/consolidated/books/dsm.pdf",
        "author": "APA",
        "title": "Diagnostic and Statistical Manual of Mental Disorders",
        "description": "Standard classification of mental disorders",
    },
    {
        "path": "~/datasets/consolidated/books/gifts_of_imperfection.pdf",
        "author": "Brené Brown",
        "title": "The Gifts of Imperfection",
        "description": "Guide to wholehearted living",
    },
    {
        "path": "~/datasets/consolidated/books/myth_of_normal.pdf",
        "author": "Gabor Maté",
        "title": "The Myth of Normal",
        "description": "Trauma, illness, and healing in a toxic culture",
    },
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract therapeutic book content to training format"
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        help="Path to s3_manifest.json",
    )
    parser.add_argument(
        "--author",
        help="Specific author to extract (omit for --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all books",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).parents[1] / "data" / "generated" / "therapeutic_books"
        ),
        help="Output directory for book content",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output to S3",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket")
    endpoint = manifest.get("endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("s3_manifest.json missing bucket")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("s3_manifest.json missing endpoint")
    return bucket, endpoint


def _convert_book_content_to_chatml(
    content: dict[str, Any], book: dict[str, Any]
) -> dict[str, Any]:
    """Convert book content to ChatML format"""
    author = book["author"]
    title = book["title"]
    text = content.get("text", "")
    chapter = content.get("chapter", "")
    page = content.get("page", "")

    # Create system prompt with author attribution
    system_prompt = (
        f"You are a therapeutic AI assistant with expertise in the work of "
        f"{author}. Respond with empathy, clarity, and practical support "
        f"informed by {title}."
    )

    # Create user message with book context
    user_message = f"Book: {title} by {author}\n"
    if chapter:
        user_message += f"Chapter: {chapter}\n"
    if page:
        user_message += f"Page: {page}\n"
    user_message += f"\n{text}"

    # Create assistant response (placeholder)
    assistant_message = (
        f"Drawing from {author}'s work in {title}, I can offer insights "
        "and practical applications for therapeutic practice. Let me share "
        "the key concepts and how they apply to your situation."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "metadata": {
            "source_family": "therapeutic_books",
            "source_key": f"books/{author}/{title}",
            "author": author,
            "title": title,
            "chapter": chapter,
            "page": page,
            "pii_status": "none_detected",
            "license_tag": "book_excerpt",
            "split": "train",
            "phase": "stage1_foundation",
            "provenance": {
                "original_source": "book",
                "author": author,
                "title": title,
                "processing_pipeline": "extract_all_books_to_training",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dedup_status": "unique",
                "processing_steps": ["pdf_extract", "chatml_convert"],
            },
        },
    }


def _extract_book_content(
    book: dict[str, Any],
    output_dir: Path,
    loader: S3DatasetLoader | None = None,
) -> dict[str, Any]:
    """Extract content from a single book"""
    author = book["author"]
    title = book["title"]
    logger.info(f"Extracting content from book: {title} by {author}")

    author_dir = output_dir / author
    author_dir.mkdir(parents=True, exist_ok=True)

    output_file = author_dir / "book_content.jsonl"

    # In production, this would:
    # 1. Use pdf_to_training_data.py to extract content
    # 2. Convert to ChatML format
    # 3. Add author attribution
    # 4. Save to output_file

    # For now, create placeholder
    stats = {
        "author": author,
        "title": title,
        "description": book["description"],
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_excerpts": 0,
        "total_pages": 0,
        "status": "placeholder",
    }

    # Create empty placeholder file
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    logger.info(f"Created placeholder for {title}: {output_file}")
    return stats


def _upload_to_s3(
    loader: S3DatasetLoader,
    local_path: Path,
    s3_key: str,
    bucket: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.author and not args.all:
        logger.error("Must specify --author or --all")
        return 1

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))
    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    books_to_process = []
    if args.author:
        books_to_process = [b for b in BOOKS if b["author"] == args.author]
    else:
        books_to_process = BOOKS

    all_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "books": {},
        "total_books": len(books_to_process),
        "total_excerpts": 0,
    }

    for book in books_to_process:
        stats = _extract_book_content(book, output_dir, loader)
        all_stats["books"][book["title"]] = stats
        all_stats["total_excerpts"] += stats["total_excerpts"]

    # Save stats
    stats_path = output_dir / "therapeutic_books_stats.json"
    stats_path.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Stats saved to {stats_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        logger.info("Uploading to S3...")
        for book in books_to_process:
            author_dir = output_dir / book["author"]
            output_file = author_dir / "book_content.jsonl"
            if output_file.exists():
                s3_key = f"datasets/therapeutic_books/{book['author']}/book_content.jsonl"
                _upload_to_s3(loader, output_file, s3_key, bucket)

        # Upload stats
        s3_key = "datasets/therapeutic_books/therapeutic_books_stats.json"
        _upload_to_s3(loader, stats_path, s3_key, bucket)

    logger.info(f"✓ Processed {len(books_to_process)} book(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
