#!/usr/bin/env python3
"""Extract academic research findings for training.

This script uses the JournalResearchIngestor to fetch findings and
converts them to ChatML format.

Outputs:
- ai/training_ready/data/generated/academic_research/findings.jsonl
- ai/training_ready/data/generated/academic_research_stats.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# Add project root to sys.path
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ai.pipelines.orchestrator.sourcing.journal_ingestor import JournalResearchIngestor  # noqa: E402

logger = logging.getLogger(__name__)

RESEARCH_QUERIES = [
    "ADHD interventions and treatment effectiveness",
    "Cognitive behavioral therapy for anxiety",
    "Mindfulness-based stress reduction",
    "Family therapy for adolescent mental health",
    "Trauma-informed care practices",
    "Complex PTSD treatment protocols",
    "Narcissistic abuse recovery",
    "Codependency patterns in families",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract academic research findings for training")
    parser.add_argument(
        "--query",
        help="Specific research query (omit for --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all predefined research queries",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parents[1] / "data" / "generated" / "academic_research"),
        help="Output directory for findings",
    )
    return parser


def _convert_abstract_to_chatml(record: dict[str, Any]) -> dict[str, Any]:
    """Convert ingestion record to ChatML"""
    title = record.get("title", "Unknown Title")
    abstract = record.get("abstract", "")
    source = record.get("source", "Academic")
    query = record.get("query", "")

    system_prompt = (
        "You are a therapeutic AI assistant with expertise in evidence-based practice. "
        "Respond with empathy, clarity, and practical support grounded in research."
    )

    user_message = f"Research findings on {query}:\n\nTitle: {title}\nAbstract: {abstract}"

    # Generate a synthetic insight based on abstract (Placeholder for actual generation)
    # In real pipeline, we might use an LLM here. For now, we wrap the abstract as 'knowledge'.
    assistant_message = (
        f"Based on research from {source}, regarding '{title}':\n\n"
        f"The findings suggest: {abstract[:300]}...\n\n"
        "This indicates significant implications for therapeutic practice in this area."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "metadata": {
            "source_family": "academic_research",
            "source_key": f"academic/{record.get('id', 'unknown')}",
            "query": query,
            "title": title,
            "provenance": {
                "original_source": source,
                "ingestion_id": record.get("id"),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Initializing Academic Research Extraction...")

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.query and not args.all:
        logger.error("Must specify --query or --all")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup temporary directory for raw ingestion
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ingestor = JournalResearchIngestor(output_dir=temp_path)

        queries = [args.query] if args.query else RESEARCH_QUERIES
        raw_output_file = temp_path / "journal_abstracts.jsonl"

        total_ingested = 0

        for q in queries:
            # We treat queries as topics for both DOAJ and ClinicalTrials
            # Simplified mapping
            ingestor.ingest_doaj(query=q, limit=5)
            # ingestor.ingest_clinical_trials(condition=q, limit=5) # Optional/Rate limited

            # Simple sleep/backoff handled by ingestor?

        if raw_output_file.exists():
            # Convert to ChatML
            final_output = output_dir / "findings.jsonl"
            converted_count = 0

            with (
                open(raw_output_file, encoding="utf-8") as fin,
                open(final_output, "w", encoding="utf-8") as fout,
            ):
                for line in fin:
                    try:
                        record = json.loads(line)
                        chatml = _convert_abstract_to_chatml(record)
                        fout.write(json.dumps(chatml, ensure_ascii=False) + "\n")
                        converted_count += 1
                    except json.JSONDecodeError:
                        continue

            total_ingested = converted_count
            logger.info(f"Converted {converted_count} findings to {final_output}")

            # Save stats
            stats = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "queries": queries,
                "total_findings": total_ingested,
            }
            stats_path = output_dir / "academic_research_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        else:
            logger.warning("No findings ingested.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
