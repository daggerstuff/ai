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
import contextlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import ebooklib

try:
    import mobi
except ModuleNotFoundError:
    mobi = None  # type: ignore[assignment]
import requests
import yaml
from bs4 import BeautifulSoup
from ebooklib import epub
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
MIN_RECOVERED_TEXT_LENGTH = 100
DEFAULT_NIM_MIN_INTERVAL_SECONDS = 18.0
OLLAMA_TIMEOUT_SECONDS = 300
NIM_TIMEOUT_SECONDS = 180
GEMINI_TIMEOUT_SECONDS = 60

# Simple rate limiter: enforce min_interval_seconds between API calls.
_RATE_LIMIT_STATE = {"last_api_call": 0.0}


def _now_monotonic() -> float:
    return time.time()


def _rate_limit() -> None:
    elapsed = _now_monotonic() - _RATE_LIMIT_STATE["last_api_call"]
    min_interval = float(os.environ.get("NIM_MIN_INTERVAL", str(DEFAULT_NIM_MIN_INTERVAL_SECONDS)))
    if elapsed < min_interval:
        sleep_for = min_interval - elapsed
        logger.info(f"  Rate limit pacing: sleeping {sleep_for:.1f}s")
        time.sleep(sleep_for)
    _RATE_LIMIT_STATE["last_api_call"] = _now_monotonic()


def _retry_request(
    fn: Callable[[], str],
    max_retries: int = MAX_RETRIES,
    base_delay: int = RETRY_BASE_DELAY,
) -> str | None:
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


def _call_ollama(system_prompt: str, user_content: str, host: str, model: str) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": f"System: {system_prompt}\n\nUser: {user_content}",
        "stream": False,
        "options": {"temperature": 0.3},
    }
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    if response.status_code == HTTP_OK:
        return response.json().get("response", "")
    raise RuntimeError(f"Ollama failed ({response.status_code}): {response.text[:200]}")


def _call_nim(system_prompt: str, user_content: str, model_id: str) -> str:
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
    response = requests.post(url, headers=headers, json=payload, timeout=NIM_TIMEOUT_SECONDS)
    if response.status_code == HTTP_OK:
        return response.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"NIM failed ({response.status_code}): {response.text[:200]}")


def _call_gemini(system_prompt: str, user_content: str, model_id: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {"temperature": 0.3},
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    if response.status_code == HTTP_OK:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini failed ({response.status_code}): {response.text[:200]}")


def _try_provider(name: str, call: Callable[[], str], use_retry: bool = False) -> str:
    try:
        response = _retry_request(call) if use_retry else call()
    except Exception as exc:
        logger.warning("%s failed: %s", name, exc)
        return ""
    return response or ""


def query_llm(system_prompt: str, user_content: str) -> str:
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    ollama_model = os.environ.get("OLLAMA_MODEL", "medgemma1.5:latest")
    if ollama_host:
        ollama_response = _try_provider(
            "Ollama",
            lambda: _call_ollama(system_prompt, user_content, ollama_host, ollama_model),
        )
        if ollama_response:
            return ollama_response

    if NVIDIA_API_KEY:
        model_id = os.environ.get("NVIDIA_MODEL_ID", "mistralai/mistral-small-4-119b-2603")
        nim_response = _try_provider("NIM", lambda: _call_nim(system_prompt, user_content, model_id), use_retry=True)
        if nim_response:
            return nim_response

    if GEMINI_API_KEY:
        gemini_response = _try_provider(
            "Gemini",
            lambda: _call_gemini(system_prompt, user_content, "gemini-2.5-flash"),
            use_retry=True,
        )
        if gemini_response:
            return gemini_response

    if ollama_host:
        return _try_provider(
            "Ollama",
            lambda: _call_ollama(system_prompt, user_content, ollama_host, ollama_model),
        )
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
    if mobi is None:
        raise ImportError("mobi is required for AZW3/MOBI extraction but is not installed")
    tmpdir = None
    try:
        tmpdir, epub_path = mobi.extract(str(path))  # type: ignore[union-attr]
        return _extract_epub(Path(epub_path))
    except Exception as exc:
        logger.warning("Failed to extract AZW %s: %s", path, exc)
        return None
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    extractors: dict[str, Callable[[Path], str | None]] = {
        ".pdf": _extract_pdf,
        ".epub": _extract_epub,
        ".txt": lambda text_path: text_path.read_text(encoding="utf-8", errors="replace"),
    }
    azw_suffixes = {".azw3", ".azw", ".mobi"}
    try:
        if suffix in azw_suffixes:
            return _extract_azw(path)
        extractor = extractors.get(suffix)
        if extractor is not None:
            return extractor(path)
        logger.warning("Unsupported format: %s", path)
    except Exception:
        # Fallback: try reading as plain text (some "PDFs" are actually text files)
        try:
            text = path.read_text(encoding="utf-8")
            if text and len(text) > MIN_RECOVERED_TEXT_LENGTH:
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


def _append_yaml_pairs(raw_line: str, pairs: list[dict[str, str]]) -> None:
    data = yaml.safe_load(raw_line)
    if isinstance(data, dict) and "instruction" in data and "output" in data:
        pairs.append({"instruction": str(data["instruction"]), "output": str(data["output"])})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "instruction" in item and "output" in item:
                pairs.append({"instruction": str(item["instruction"]), "output": str(item["output"])})


def _parse_inline_instruction_output(raw_line: str) -> dict[str, str] | None:
    inst_match = re.search(r"instruction:\s*([^,]+?)(?:\s*,\s*\"?output\"?\s*:|$)", raw_line, re.IGNORECASE)
    out_match = re.search(r"output:\s*(.+)$", raw_line, re.IGNORECASE)
    if inst_match and out_match:
        instruction = inst_match.group(1).strip().strip("\"'")
        output = out_match.group(1).strip().strip("\"'")
        if instruction and output:
            return {"instruction": instruction, "output": output}
    return None


def _parse_numbered_instruction_output(raw_line: str) -> dict[str, str] | None:
    inst_match = re.search(r"instruction:\s*(.+?)(?:output:|$)", raw_line, re.DOTALL | re.IGNORECASE)
    out_match = re.search(r"output:\s*(.+)$", raw_line, re.DOTALL | re.IGNORECASE)
    if inst_match and out_match:
        instruction = inst_match.group(1).strip()
        output = out_match.group(1).strip()
        if instruction and output:
            return {"instruction": instruction, "output": output}
    return None


def _extract_json_pair(raw_line: str) -> dict[str, str] | None:
    match = re.search(r"\{.*\}", raw_line)
    if not match:
        return None
    try:
        pair = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "instruction" in pair and "output" in pair:
        return pair
    return None


def _parse_distilled_pair(stripped_line: str, pairs: list[dict[str, str]]) -> None:
    try:
        pair = json.loads(stripped_line)
    except (json.JSONDecodeError, TypeError):
        pair = None
    if isinstance(pair, dict) and "instruction" in pair and "output" in pair:
        pairs.append(pair)
        return

    line_lower = stripped_line.lower()
    if "instruction" in line_lower and "output" in line_lower:
        inline_pair = _parse_inline_instruction_output(stripped_line)
        if inline_pair is not None:
            pairs.append(inline_pair)
            return
        with contextlib.suppress(Exception):
            _append_yaml_pairs(stripped_line, pairs)
        return

    if re.search(r"Instruction:.*Output:", stripped_line):
        with contextlib.suppress(Exception):
            numbered_pair = _parse_numbered_instruction_output(stripped_line)
            if numbered_pair is not None:
                pairs.append(numbered_pair)
        return

    fallback_pair = _extract_json_pair(stripped_line)
    if fallback_pair is not None:
        pairs.append(fallback_pair)


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
        "MUST output JSON only. Format: each line is a complete JSON object like "
        '{"instruction": "...", "output": "..."}. '
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
    for response_line in raw_response.strip().splitlines():
        stripped_line = response_line.strip()
        if not stripped_line or stripped_line.startswith("```"):
            continue
        _parse_distilled_pair(stripped_line, pairs)
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
    max_chunks: int | bool | None = None,
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
        logger.info("  DSM content detected — using LLM distillation (use_llm explicitly set).")
        is_dsm = True
        # Keep use_llm = True to allow LLM distillation for DSM

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

                ts = datetime.now(UTC).isoformat()
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
                    "generated_at": datetime.now(UTC).isoformat(),
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
            "generated_at": datetime.now(UTC).isoformat(),
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
        if book_file.suffix.lower() not in {".pdf", ".epub", ".azw3", ".azw", ".mobi", ".txt"}:
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
        "generated_at": datetime.now(UTC).isoformat(),
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
