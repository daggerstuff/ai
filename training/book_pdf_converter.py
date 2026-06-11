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
import yaml
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import mobi
from pypdf import PdfReader

logging.basicConfig(
    level=logging.DEBUG,
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

# Simple rate limiter: enforce min_interval_seconds between API calls.
_last_api_call: float = 0.0


def _rate_limit():
    global _last_api_call
    elapsed = time.time() - _last_api_call
    min_interval = float(os.environ.get("NIM_MIN_INTERVAL", "18"))
    if elapsed < min_interval:
        sleep_for = min_interval - elapsed
        logger.info(f"  Rate limit pacing: sleeping {sleep_for:.1f}s")
        time.sleep(sleep_for)
    _last_api_call = time.time()


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
        except RuntimeError as e:
            if "429" in str(e):
                delay = 30 * 2 ** (attempt - 1)
                logger.warning(
                    f"  {C_YELLOW}Rate limited (429), attempt {attempt}/{max_retries}. Retrying in {delay}s...{C_RESET}"
                )
                time.sleep(delay)
            else:
                raise
    return None


def query_llm(system_prompt: str, user_content: str) -> str:
    # Try Ollama first if configured - local, no rate limits
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host:
        ollama_model = os.environ.get("OLLAMA_MODEL", "medgemma1.5:latest")
        try:

            def _call_ollama():
                url = f"{ollama_host.rstrip('/')}/api/generate"
                payload = {
                    "model": ollama_model,
                    "prompt": f"System: {system_prompt}\n\nUser: {user_content}",
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
                response = requests.post(url, json=payload, timeout=300)
                if response.status_code == 200:
                    return response.json().get("response", "")
                raise RuntimeError(f"Ollama failed ({response.status_code}): {response.text[:200]}")

            res = _call_ollama()
            if res:
                return res
        except Exception as e:
            logger.warning(f"Ollama failed: {e}")

    if NVIDIA_API_KEY:
        model_id = os.environ.get("NVIDIA_MODEL_ID", "mistralai/mistral-small-4-119b-2603")
        try:

            def _call_nim():
                _rate_limit()
                url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
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

    # Ollama fallback - local inference, no rate limits
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host:
        ollama_model = os.environ.get("OLLAMA_MODEL", "medgemma1.5:latest")
        try:

            def _call_ollama():
                url = f"{ollama_host.rstrip('/')}/api/generate"
                payload = {
                    "model": ollama_model,
                    "prompt": f"System: {system_prompt}\n\nUser: {user_content}",
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
                response = requests.post(url, json=payload, timeout=300)
                if response.status_code == 200:
                    return response.json().get("response", "")
                raise RuntimeError(f"Ollama failed ({response.status_code}): {response.text[:200]}")

            res = _call_ollama()
            if res:
                return res
        except Exception as e:
            logger.warning(f"Ollama failed: {e}")

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


def _extract_azw(path: Path) -> str | None:
    """Extract text from AZW3/MOBI via temporary conversion to EPUB."""
    try:
        tmpdir, epub_path = mobi.extract(str(path))
        result = _extract_epub(Path(epub_path))
        shutil.rmtree(tmpdir, ignore_errors=True)
        return result
    except Exception as exc:
        logger.warning("Failed to extract AZW %s: %s", path, exc)
        return None


def _extract_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        elif suffix == ".epub":
            return _extract_epub(path)
        elif suffix in (".azw3", ".azw", ".mobi"):
            return _extract_azw(path)
        else:
            logger.warning("Unsupported format: %s", path)
            return None
    except Exception:
        # Fallback: try reading as plain text (some "PDFs" are actually text files)
        try:
            text = path.read_text(encoding="utf-8")
            if text and len(text) > 100:
                logger.info("  Recovered %s as plain text (PDF parsing failed)", path.name)
                return text
        except Exception:
            pass
        logger.warning("Failed to extract %s", path)
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
        "You are a clinical psychology expert generating training data for a therapist AI. "
        "Use the book excerpt below to produce grounded, specific therapist responses.\n\n"
        "Generate 3-5 QA pairs.\n"
        "- 'instruction': a first-person client statement (1-3 sentences) rooted in the excerpt's topic.\n"
        "- 'output': a therapist response (2-4 sentences) that:\n"
        "   • References a specific concept, framework, or term from the excerpt — not generic therapy language\n"
        "   • Is direct and specific, not warm-and-fuzzy. No empty validation.\n"
        "   • Varies its sentence structure. NEVER start with 'It sounds like', 'I hear that', "
        "'That must be', 'I can see that', or any variation of therapized agreeableness.\n"
        "   • Gives the client something concrete to work with: a reframe, a distinction, a question to consider.\n\n"
        "BAD output (generic slop — never write this):\n"
        "  \"It sounds like you're really struggling with this. That must be difficult. Let's explore that.\"\n\n"
        "GOOD output (specific, grounded):\n"
        '  "Pete Walker distinguishes the inner critic from genuine self-awareness. '
        "The fact that you're noticing this pattern means you're already building that awareness — "
        'the next step is to name what the critic is actually saying."\n\n'
        'MUST output JSON only. Format: each line is a complete JSON object like {"instruction": "...", "output": "..."}. '
        "CRITICAL: Start each line with { and end with }. Use double quotes for all strings. "
        "Example output lines (copy exactly):\n"
        '{"instruction": "I feel anxious", "output": "What triggers that?"}\n'
        '{"instruction": "I cant sleep", "output": "What time do you try?"}\n'
        "Output ONLY these JSON lines. Nothing else. No explanation."
    )
    user_content = f"Book Title: {title}\n\nExcerpt:\n{chunk}"

    raw_response = query_llm(system_prompt, user_content)
    if not raw_response:
        return []

    # Debug: log first 200 chars of response
    logger.debug(f"LLM response (first 200): {raw_response[:200]}")

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
        except (json.JSONDecodeError, TypeError):
            line_lower = line.lower()
            # Try YAML format (instruction: / output:)
            if "instruction" in line_lower and "output" in line_lower:
                try:
                    data = yaml.safe_load(line)
                    if isinstance(data, dict) and "instruction" in data and "output" in data:
                        pairs.append({"instruction": str(data["instruction"]), "output": str(data["output"])})
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "instruction" in item and "output" in item:
                                pairs.append({"instruction": str(item["instruction"]), "output": str(item["output"])})
                except Exception:
                    pass
            # Try numbered format: "1. Instruction: ... Output: ..." or just "Instruction: ... Output: ..."
            elif re.search(r"Instruction:.*Output:", line):
                try:
                    inst_match = re.search(r"instruction:\s*(.+?)(?:output:|$)", line, re.DOTALL | re.IGNORECASE)
                    out_match = re.search(r"output:\s*(.+)$", line, re.DOTALL | re.IGNORECASE)
                    if inst_match and out_match:
                        instruction = inst_match.group(1).strip()
                        output = out_match.group(1).strip()
                        if instruction and output:
                            pairs.append({"instruction": instruction, "output": output})
                except Exception:
                    pass
            else:
                # Try regex extraction
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
            instruction = f"Incorporating principles from {title}, how would a therapist apply this concept in session?"
        pairs.append({"instruction": instruction, "output": chunk})

    return pairs


def convert_book(
    book_path: Path,
    output_dir: Path,
    max_chunks: int | None | bool = None,
    is_dsm: bool = False,
    use_llm: bool = False,
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

    output_file = output_dir / f"{title}.jsonl"

    # DSM reference works auto-detected: skip LLM path — diagnostic criteria
    # aren't therapy dialogue, and the simple path produces better training data.
    if use_llm and _is_dsm_title(title):
        logger.info(f"  DSM content detected — using simple path (no API calls).")
        is_dsm = True
        use_llm = False

    if use_llm:
        # LLM distillation path — slow, requires API key, higher quality
        # Uses larger chunks (6000 chars) and parallel workers for throughput.
        chunks = _chunk_text(text, chunk_size=6000)
        if max_chunks:
            chunks = chunks[:max_chunks]
            logger.info(f"Limited to {max_chunks} chunks for testing.")

        all_pairs: list[dict] = []
        max_workers = min(int(os.environ.get("LLM_WORKERS", "2")), len(chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_idx = {pool.submit(distill_chunk, c, title): i for i, c in enumerate(chunks)}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    pairs = fut.result()
                except Exception as exc:
                    logger.warning(f"  Chunk {idx + 1}/{len(chunks)} failed: {exc}")
                    continue
                if not pairs:
                    logger.warning(f"    No pairs generated for chunk {idx + 1}")
                    continue

                ts = datetime.now(timezone.utc).isoformat()
                for pair in pairs:
                    enriched_pair = {
                        "instruction": pair["instruction"],
                        "output": pair["output"],
                        "metadata": {
                            "source_book": title,
                            "source_type": "clinical_literature",
                            "distillation_version": "2.0.0",
                            "generated_at": ts,
                        },
                    }
                    all_pairs.append(enriched_pair)

        # Write all pairs at once (order-independent)
        with open(output_file, "w", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        return {
            "book": str(book_path),
            "title": title,
            "status": "converted" if all_pairs else "empty",
            "pairs": len(all_pairs),
            "chunks_processed": len(chunks),
        }

    # Simple path — no API calls, fast, uses raw excerpts with generic instructions
    pairs = _text_to_qa_pairs(text, title, is_dsm, chunk_size=2000)
    if max_chunks:
        pairs = pairs[:max_chunks]

    all_pairs: list[dict] = []
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            enriched_pair = {
                "instruction": pair["instruction"],
                "output": pair["output"],
                "metadata": {
                    "source_book": title,
                    "source_type": "clinical_literature",
                    "distillation_version": "2.0.0",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            all_pairs.append(enriched_pair)
            f.write(json.dumps(enriched_pair) + "\n")

    return {
        "book": str(book_path),
        "title": title,
        "status": "converted" if all_pairs else "empty",
        "pairs": len(all_pairs),
        "chunks_processed": len(pairs),
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
    parser.add_argument(
        "--use_llm",
        action="store_true",
        default=False,
        help="Use LLM distillation (slow, needs API key) instead of fast simple path.",
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
        if book_file.suffix.lower() not in {".pdf", ".epub", ".azw3", ".azw", ".mobi"}:
            continue

        expected_out = output_dir / f"{book_file.stem}.jsonl"
        if expected_out.exists() and expected_out.stat().st_size > 0:
            logger.info(f"{C_GREEN}Skipping {book_file.stem} — output already exists.{C_RESET}")
            results.append(
                {
                    "book": str(book_file),
                    "title": book_file.stem,
                    "status": "skipped",
                    "reason": "already converted",
                    "pairs": 0,
                }
            )
            continue

        result = convert_book(book_file, output_dir, max_chunks=args.max_chunks, use_llm=args.use_llm)
        results.append(result)

        if result["status"] == "converted":
            total_pairs += result["pairs"]
            logger.info(f"{C_GREEN}Converted {result['title']}: {result['pairs']} pairs.{C_RESET}")
        else:
            logger.warning(f"{C_RED}Failed/Skipped {book_file.name}: {result.get('reason', 'unknown')}{C_RESET}")

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
