#!/usr/bin/env python3
"""Extract BurnoutNarrative records from raw HTML pages.

The burnout datasets on whitebat contain raw HTML pages scraped from
therapist burnout articles. The ingestion pipeline failed because
BurnoutNarrative required source_type, outcome, and outcome_severity fields
that can't be extracted from raw HTML without LLM processing.

This script does the initial extraction: parse HTML → text, create
BurnoutNarrative records with available fields, and leave structured
fields for later LLM enrichment.

Usage:
    uv run python -m ai.pipelines.data_processing.scripts.extract_burnout \
        --input-dir ai/data/raw/burnout \
        --output-dir ai/data/curated/burnout
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_burnout")


class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "nav", "footer", "header", "aside"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
            self._skip = True
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._text.append("\n")
        if tag == "a":
            for attr, val in attrs:
                if attr == "href" and val:
                    pass  # Could capture links

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
            if self._skip_depth == 0:
                self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text)
        # Clean up whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        return raw.strip()


def extract_text_from_html(html_path: Path) -> str:
    """Extract readable text from an HTML file."""
    try:
        content = html_path.read_text(encoding="utf-8", errors="replace")
        extractor = HTMLTextExtractor()
        extractor.feed(content)
        text = extractor.get_text()
        # Filter out very short fragments
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to parse {html_path}: {e}")
        return ""


def infer_source_type(filename: str) -> str | None:
    """Infer source_type from the dataset name/filename."""
    name = filename.lower()
    if "substack" in name:
        return "substack"
    if "medium" in name:
        return "medium"
    if "reddit" in name:
        return "reddit"
    if "blog" in name:
        return "blog"
    # Default: infer from path structure
    return None


def create_burnout_narrative(
    narrative_id: str,
    source_url: str,
    raw_text: str,
    dataset_name: str,
) -> dict:
    """Create a BurnoutNarrative-compatible dict.

    Fields that require LLM extraction (outcome, outcome_severity, failure_modes,
    key_dynamics, etc.) are left as None/empty for later enrichment.
    """
    source_type = infer_source_type(dataset_name)

    return {
        "narrative_id": narrative_id,
        "source_url": source_url,
        "source_type": source_type,
        "author_role": None,
        "years_experience": None,
        "clinical_setting": None,
        "failure_modes": [],
        "key_dynamics": [],
        "trauma_exposure": None,
        "outcome": None,
        "outcome_severity": None,
        "extractable_patterns": [],
        "raw_text": raw_text if len(raw_text) <= 50000 else raw_text[:50000],
    }


def find_html_files(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all HTML files in the burnout input directory.

    Returns list of (dataset_name, html_path) tuples.
    """
    results: list[tuple[str, Path]] = []
    for pattern in ["**/page.html", "**/*.html"]:
        for html_path in input_dir.glob(pattern):
            # Extract dataset name from path structure
            # Expected: burnout_<name>/raw/burnout_<name>/page.html
            parts = html_path.parts
            dataset_name = ""
            for part in parts:
                if part.startswith("burnout_"):
                    dataset_name = part
                    break
            if not dataset_name:
                dataset_name = html_path.stem
            results.append((dataset_name, html_path))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract BurnoutNarrative records from raw HTML.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ai/data/raw/burnout",
        help="Directory containing raw burnout HTML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ai/data/curated/burnout",
        help="Output directory for extracted JSONL",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = find_html_files(input_dir)
    # Deduplicate by dataset_name (keep first occurrence)
    seen_names: set[str] = set()
    deduped_files: list[tuple[str, Path]] = []
    for dataset_name, html_path in html_files:
        if dataset_name not in seen_names:
            seen_names.add(dataset_name)
            deduped_files.append((dataset_name, html_path))
    html_files = deduped_files

    if not html_files:
        logger.error(f"No HTML files found in {input_dir}")
        return 1

    logger.info(f"Found {len(html_files)} unique HTML files to process")

    records: list[dict] = []
    stats = {
        "total_files": len(html_files),
        "extracted": 0,
        "failed": 0,
        "empty_text": 0,
    }

    for dataset_name, html_path in html_files:
        logger.info(f"Processing {dataset_name}...")

        text = extract_text_from_html(html_path)
        if len(text) < 100:
            logger.warning(f"Very short text from {dataset_name}: {len(text)} chars")
            stats["empty_text"] += 1
            continue

        narrative_id = dataset_name
        source_url = f"whitebat:training/pixelated-empathy/output/{dataset_name}/raw/{dataset_name}/page.html"

        record = create_burnout_narrative(
            narrative_id=narrative_id,
            source_url=source_url,
            raw_text=text,
            dataset_name=dataset_name,
        )
        records.append(record)
        stats["extracted"] += 1
        logger.info(f"  Extracted {len(text)} chars from {dataset_name}")

    # Write JSONL output
    output_file = output_dir / "burnout_narratives.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write manifest
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "stats": stats,
        "output_file": str(output_file),
        "total_records": len(records),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Extraction complete: {stats['extracted']} records, {stats['failed']} failed, {stats['empty_text']} empty"
    )
    logger.info(f"Output: {output_file}")
    logger.info(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
