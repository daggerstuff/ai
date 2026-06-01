#!/usr/bin/env python3
"""Proper Book and PDF conversion pipeline for therapeutic AI training.

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
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("book_pdf_converter")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("NIM_API_KEY", ""))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

HTTP_OK = 200
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

C_YELLOW = "\033[93m"
C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"


def _retry_request(fn, max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(f"  {C_YELLOW}Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay}s...{C_RESET}")
            time.sleep(delay)
    return None


def query_llm(system_prompt: str, user_content: str) -> str:
    if NVIDIA_API_KEY:
        model_id = "minimaxai/minimax-m2.7"
        try:
            def _call_nim():
                url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.3,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code == HTTP_OK:
                    return response.json()["choices"][0]["message"]["content"]
                raise RuntimeError(f"NIM failed ({response.status_code}): {response.text[:200]}")

            res = _retry_request(_call_nim)
            if res:
                return res
        except Exception as e:
            logger.warning(f"NIM failed: {e}")

    if GEMINI_API_KEY:
        model_id = "gemini-2.5-flash"
        try:
            def _call_gemini():
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                    "generationConfig": {"temperature": 0.3},
                }
                response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
                if response.status_code == HTTP_OK:
                    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                raise RuntimeError(f"Gemini failed ({response.status_code}): {response.text[:200]}")

            res = _retry_request(_call_gemini)
            if res:
                return res
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

    return ""


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_epub(path: Path) -> str:
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


def _chunk_text(text: str, chunk_size: int = 4000) -> list[str]:
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
    return chunks


def distill_chunk(chunk: str, title: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a clinical psychology expert. Your task is to extract therapeutic knowledge from the provided "
        "book excerpt and format it as high-quality training data for an AI therapist.\n\n"
        "Generate 3-5 QA pairs based on the excerpt.\n"
        "- 'instruction' MUST be a realistic first-person client statement (1-3 sentences) that reflects a struggle, "
        "question, or emotional state related to the excerpt.\n"
        "- 'output' MUST be a warm, professional, and therapeutically-grounded response (2-5 sentences) that applies "
        "the clinical principles from the book.\n"
        "- Ensure the therapist response is helpful, non-judgmental, and evidence-based.\n"
        "- AVOID generic platitudes. Use the specific insights from the book.\n\n"
        "Output ONLY a valid JSONL format where each line is a JSON object with 'instruction' and 'output' keys. "
        "Do NOT include any other text or Markdown formatting."
    )
    user_content = f"Book Title: {title}\n\nExcerpt:\n{chunk}"

    raw_response = query_llm(system_prompt, user_content)
    if not raw_response:
        return []

    pairs = []
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        try:
            pair = json.loads(line)
            if "instruction" in pair and "output" in pair:
                pairs.append(pair)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", line)
            if match:
                try:
                    pair = json.loads(match.group(0))
                    if "instruction" in pair and "output" in pair:
                        pairs.append(pair)
                except json.JSONDecodeError:
                    continue
    return pairs


def _is_dsm_title(title: str) -> bool:
    title_lower = title.lower()
    return "dsm" in title_lower or "diagnostic and statistical manual" in title_lower


def _text_to_qa_pairs(
    text: str,
    title: str,
    is_dsm: bool,
    chunk_size: int = 2000,
) -> list[dict[str, str]]:
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
    max_chunks: int | None | bool = None,
    is_dsm: bool = False,
) -> dict:
    if isinstance(max_chunks, bool):
        is_dsm = max_chunks
        max_chunks = None

    title = book_path.stem
    logger.info(f"Processing book: {title}")

    text = _extract_text(book_path)
    if not text or not text.strip():
        return {
            "book": str(book_path),
            "title": title,
            "status": "skipped",
            "reason": "no text extracted",
            "pairs": 0,
        }

    chunks = _chunk_text(text)
    if max_chunks:
        chunks = chunks[:max_chunks]
        logger.info(f"Limited to {max_chunks} chunks for testing.")

    all_pairs: list[dict] = []
    output_file = output_dir / f"{title}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            logger.info(f"  Distilling chunk {i+1}/{len(chunks)}...")
            pairs = distill_chunk(chunk, title)
            if not pairs:
                logger.warning(f"    No pairs generated for chunk {i+1}")
                continue

            for pair in pairs:
                enriched_pair = {
                    "instruction": pair["instruction"],
                    "output": pair["output"],
                    "metadata": {
                        "source_book": title,
                        "source_type": "clinical_literature",
                        "distillation_version": "2.0.0",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                }
                all_pairs.append(enriched_pair)
                f.write(json.dumps(enriched_pair) + "\n")

            time.sleep(1)

    return {
        "book": str(book_path),
        "title": title,
        "status": "converted" if all_pairs else "empty",
        "pairs": len(all_pairs),
        "chunks_processed": len(chunks),
    }


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
        "--max_chunks",
        type=int,
        default=None,
        help="Max chunks per book (for testing).",
    )
    return parser


def run_conversion(args: argparse.Namespace) -> None:
    books_dir = Path(args.books_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not books_dir.exists():
        logger.error("Books directory not found: %s", books_dir)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "books_dir": str(books_dir),
            "output_dir": str(output_dir),
            "total_books": 0,
            "total_books_found": 0,
            "converted_books": 0,
            "converted": 0,
            "skipped": 0,
            "total_pairs": 0,
            "total_filtered": 0,
            "details": [],
            "book_details": [],
        }
        with open(output_dir / "conversion_report.json", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        return

    results: list[dict] = []
    total_pairs = 0

    for book_file in sorted(books_dir.rglob("*")):
        if book_file.suffix.lower() not in {".pdf", ".epub"}:
            continue

        result = convert_book(book_file, output_dir, max_chunks=args.max_chunks)
        results.append(result)

        if result["status"] == "converted":
            total_pairs += result["pairs"]
            logger.info(
                f"{C_GREEN}Converted {result['title']}: {result['pairs']} pairs.{C_RESET}"
            )
        else:
            logger.warning(
                f"{C_RED}Failed/Skipped {book_file.name}: {result.get('reason', 'unknown')}{C_RESET}"
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "books_dir": str(books_dir),
        "output_dir": str(output_dir),
        "total_books": len(results),
        "total_books_found": len(results),
        "converted_books": sum(1 for r in results if r["status"] == "converted"),
        "converted": sum(1 for r in results if r["status"] == "converted"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "total_pairs": total_pairs,
        "total_filtered": 0,
        "details": results,
        "book_details": results,
    }
    with open(output_dir / "conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Conversion complete: %d books converted, %d total pairs",
        sum(1 for r in results if r["status"] == "converted"),
        total_pairs,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_conversion(args)


if __name__ == "__main__":
    main()
