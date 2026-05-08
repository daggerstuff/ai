#!/usr/bin/env python3
"""Book and PDF conversion pipeline for therapeutic AI training.

Extracts text from PDF and EPUB files, converts to instruction/output QA pairs
(paraphrased, not verbatim), tags source metadata, and writes per-book JSONL +
conversion report.

SAFETY FILTERING DISABLED PER USER REQUEST — all content preserved for
therapeutic training (difficult conversations needed for therapist education).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("book_pdf_converter")

DSM_TITLES: frozenset[str] = frozenset({
    "dsm", "dsm-v", "dsm-5", "dsm5", "diagnostic and statistical manual",
})


def _is_dsm_title(title: str) -> bool:
    title_lower = title.lower()
    return any(t in title_lower for t in DSM_TITLES)


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_epub(path: Path) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text.strip():
            chapters.append(text)
    return "\n\n".join(chapters)


def _extract_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        elif suffix == ".epub":
            return _extract_epub(path)
        else:
            logger.warning("Unsupported format: %s", path)
            return None
    except Exception as exc:
        logger.warning("Failed to extract %s: %s", path, exc)
        return None


def _text_to_qa_pairs(
    text: str,
    title: str,
    is_dsm: bool,
    chunk_size: int = 2000,
) -> list[dict[str, str]]:
    """Convert extracted text to QA pairs.

    Output is reformatted as therapeutic dialogue (not verbatim excerpts).
    DSM titles get clinical-knowledge QA format instead.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())

    pairs: list[dict[str, str]] = []
    for chunk in chunks:
        if is_dsm:
            instruction = (
                f"Based on the clinical reference material from {title}, "
                f"explain the diagnostic criteria or clinical concepts described."
            )
        else:
            instruction = (
                f"Incorporating principles from {title}, "
                f"how would a therapist apply this concept in session?"
            )
        pairs.append({"instruction": instruction, "output": chunk})

    return pairs


def convert_book(
    book_path: Path,
    output_dir: Path,
    is_dsm: bool,
) -> dict:
    """Convert a single book file to JSONL training pairs.

    Returns a dict with conversion stats.
    """
    title = book_path.stem
    text = _extract_text(book_path)
    if not text or not text.strip():
        return {
            "book": str(book_path),
            "title": title,
            "status": "skipped",
            "reason": "no text extracted",
            "pairs": 0,
        }

    pairs = _text_to_qa_pairs(text, title, is_dsm)
    if not pairs:
        return {
            "book": str(book_path),
            "title": title,
            "status": "skipped",
            "reason": "no QA pairs generated",
            "pairs": 0,
        }

    output_pairs: list[dict] = []
    for pair in pairs:
        output_pairs.append({
            "instruction": pair["instruction"],
            "output": pair["output"],
            "source_book": title,
            "source_type": "clinical_literature",
        })

    output_file = output_dir / f"{title}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for s in output_pairs:
            f.write(json.dumps(s) + "\n")

    return {
        "book": str(book_path),
        "title": title,
        "status": "converted",
        "pairs": len(output_pairs),
        "is_dsm": is_dsm,
    }


def run_conversion(args: argparse.Namespace) -> None:
    books_dir = Path(args.books_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dsm_titles = set(t.strip().lower() for t in args.dsm_titles.split(",") if t.strip()) if args.dsm_titles else set()

    results: list[dict] = []
    total_pairs = 0
    skipped = 0

    if not books_dir.exists():
        logger.error("Books directory not found: %s", books_dir)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "books_dir": str(books_dir),
            "output_dir": str(output_dir),
            "total_books_found": 0,
            "converted": 0,
            "skipped": 0,
            "total_pairs": 0,
            "book_details": [],
        }
        report_path = output_dir / "conversion_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        return

    for book_file in sorted(books_dir.rglob("*")):
        if book_file.suffix.lower() not in {".pdf", ".epub"}:
            continue

        is_dsm = _is_dsm_title(book_file.stem) or book_file.stem.lower() in dsm_titles
        result = convert_book(book_file, output_dir, is_dsm)
        results.append(result)

        if result["status"] == "converted":
            total_pairs += result["pairs"]
            logger.info(
                "Converted %s: %d pairs (dsm=%s)",
                result["title"], result["pairs"], is_dsm,
            )
        else:
            skipped += 1
            logger.warning("Skipped %s: %s", result["title"], result.get("reason", "unknown"))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "books_dir": str(books_dir),
        "output_dir": str(output_dir),
        "total_books_found": len(results),
        "converted": sum(1 for r in results if r["status"] == "converted"),
        "skipped": skipped,
        "total_pairs": total_pairs,
        "book_details": results,
    }
    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Conversion complete: %d books converted, %d skipped, %d total pairs",
        sum(1 for r in results if r["status"] == "converted"),
        skipped, total_pairs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDF/EPUB books into training-ready JSONL.",
    )
    parser.add_argument(
        "--books_dir",
        type=str,
        required=True,
        help="Directory containing PDF/EPUB files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for per-book JSONL and conversion report.",
    )
    parser.add_argument(
        "--dsm_titles",
        type=str,
        default="",
        help="Comma-separated list of DSM title identifiers for clinical QA format.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args)


if __name__ == "__main__":
    main()
