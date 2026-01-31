#!/usr/bin/env python3
"""Extract therapeutic book content to training format.

Wrapper for PDFProcessor to process books.

Outputs:
- ai/training_ready/data/generated/therapeutic_books/{author}/book_content.jsonl
- ai/training_ready/data/generated/therapeutic_books_stats.json
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ai.pipelines.orchestrator.processing.pdf_processor import PDFProcessor  # noqa: E402

logger = logging.getLogger(__name__)

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


@dataclass
class ProcessingContext:
    output_dir: Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract therapeutic book content to training format"
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
        default=str(Path(__file__).parents[1] / "data" / "generated" / "therapeutic_books"),
        help="Output directory for book content",
    )
    return parser


def _combine_files(generated_file: str, final_file: Path) -> None:
    gen_path = Path(generated_file)
    mode = "a" if final_file.exists() else "w"
    with (
        open(gen_path, encoding="utf-8") as src,
        open(final_file, mode, encoding="utf-8") as dst,
    ):
        shutil.copyfileobj(src, dst)


def process_book(
    book_config: dict[str, str],
    processor: PDFProcessor,
    ctx: ProcessingContext,
) -> str:
    path_str = book_config["path"]
    pdf_path = Path(path_str).expanduser()

    if not pdf_path.exists():
        logger.warning(f"Book not found at {pdf_path}, skipping.")
        return "skipped"

    logger.info(f"Processing {book_config['title']} by {book_config['author']}...")

    try:
        # Generate JSONL
        generated_file = processor.process_pdf(str(pdf_path), source_name=book_config["title"])
        if generated_file and Path(generated_file).exists():
            # Move to final location
            author_dir = ctx.output_dir / _sanitize_filename(book_config["author"])
            author_dir.mkdir(parents=True, exist_ok=True)

            final_file = author_dir / "book_content.jsonl"
            _combine_files(generated_file, final_file)

            return "success"
        logger.warning(f"No content extracted for {book_config['title']}")
        return "empty"

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return f"error: {e}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Initializing Book Extraction...")

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.author and not args.all:
        logger.error("Must specify --author or --all")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = PDFProcessor(output_dir=str(output_dir / "temp_raw"))

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "books": {},
        "total_books": 0,
    }

    processed_count = 0

    ctx = ProcessingContext(
        output_dir=output_dir,
    )

    for book_config in BOOKS:
        if args.author and args.author.lower() not in book_config["author"].lower():
            continue

        result = process_book(book_config, processor, ctx)
        if result == "success":
            processed_count += 1
        stats["books"][book_config["title"]] = result

    stats["total_books"] = processed_count

    # Save stats
    stats_path = output_dir / "therapeutic_books_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Cleanup temp
    shutil.rmtree(output_dir / "temp_raw", ignore_errors=True)

    logger.info(f"✓ Processed {processed_count} books.")
    return 0


def _sanitize_filename(name: str) -> str:
    return (
        "".join(c for c in name if c.isalnum() or c in (" ", "-", "_"))
        .strip()
        .replace(" ", "_")
        .lower()
    )


if __name__ == "__main__":
    raise SystemExit(main())
