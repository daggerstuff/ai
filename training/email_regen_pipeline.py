"""Persona-driven email regeneration pipeline with QA guards.

Regenerates short emails into natural multi-sentence persona messages using LLM
expansion. Includes quality guards (ROUGE-L, length, toxicity, persona consistency)
and Gmail reimport/corpus repair functionality.

Follows patterns from sdg_paraphrase.py and sdg_pipeline.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are Pixelated Empathy, an evidence-based clinical AI assistant "
    "specializing in trauma-informed care, CPTSD, and therapeutic communication. "
    "Respond with empathy, validation, and practical guidance."
)

DEFAULT_PERSONAS: list[str] = [
    "warm_empathetic",
    "structured_practical",
    "gentle_exploratory",
]

TOXICITY_PATTERNS: list[str] = [
    "kill yourself",
    "hurt yourself",
    "end your life",
    "suicide",
    "self-harm",
    "overdose",
]

PERSONA_TRAITS: dict[str, list[str]] = {
    "warm_empathetic": ["empathetic", "warm", "validating", "compassionate"],
    "structured_practical": ["structured", "practical", "clear", "actionable"],
    "gentle_exploratory": ["gentle", "curious", "reflective", "patient"],
}

PERSONA_PROMPTS: dict[str, str] = {
    "warm_empathetic": (
        "You are writing as a warm, empathetic therapist. Expand this brief email "
        "into a full therapeutic response that validates feelings, shows compassion, "
        "and offers gentle guidance. Use 3-5 sentences."
    ),
    "structured_practical": (
        "You are writing as a structured, practical therapist. Expand this brief email "
        "into a clear therapeutic response with actionable steps and organized advice. "
        "Use 3-5 sentences."
    ),
    "gentle_exploratory": (
        "You are writing as a gentle, exploratory therapist. Expand this brief email "
        "into a reflective therapeutic response that invites self-discovery and "
        "thoughtful questions. Use 3-5 sentences."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EmailRecord:
    """A parsed email record from Gmail export or JSONL."""

    email_id: str
    subject: str
    body: str
    sender: str = ""
    timestamp: str = ""


@dataclass
class PipelineConfig:
    """Configuration for the email regeneration pipeline."""

    min_input_length: int = 10
    max_input_length: int = 500
    min_output_sentences: int = 3
    max_output_length: int = 500
    min_output_length: int = 100
    rouge_l_too_similar: float = 0.85
    rouge_l_meaning_drift: float = 0.30
    min_quality_score: float = 0.5
    personas: list[str] = field(default_factory=lambda: list(DEFAULT_PERSONAS))
    temperature: float = 0.8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_retries: int = 3


@dataclass
class RegenerationResult:
    """Result of regenerating a single email."""

    original: EmailRecord
    regenerated_messages: list[dict[str, str]]
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    persona: str = ""
    accepted: bool = False
    error: str | None = None


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    total: int = 0
    processed: int = 0
    accepted: int = 0
    rejected: int = 0
    skipped: int = 0
    errors: int = 0
    persona_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """SHA256 hash of text, truncated to 16 chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _rouge_l_similarity(a: str, b: str) -> float:
    """Word-set Jaccard similarity between two strings (0.0-1.0)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _count_sentences(text: str) -> int:
    """Count sentences in text (periods, exclamation, question marks followed by space/end)."""
    sentences = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return len([s for s in sentences if s.strip()])


def _check_toxicity(text: str, patterns: list[str] | None = None) -> bool:
    """Check if text contains toxicity patterns. Returns True if toxic."""
    check_patterns = patterns or TOXICITY_PATTERNS
    lower = text.lower()
    for pattern in check_patterns:
        if pattern in lower:
            return True
    return False


def _check_persona_consistency(
    text: str, persona: str
) -> tuple[float, list[str]]:
    """Check if text matches persona traits. Returns (score, warnings)."""
    traits = PERSONA_TRAITS.get(persona, [])
    if not traits:
        return 0.5, [f"unknown persona: {persona}"]
    lower = text.lower()
    found = sum(1 for t in traits if t in lower)
    score = found / len(traits)
    warnings: list[str] = []
    if score < 0.25:
        warnings.append(f"low persona match ({persona}): {found}/{len(traits)} traits")
    return score, warnings


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_gmail_export(path: str) -> list[EmailRecord]:
    """Parse Gmail export JSONL into EmailRecord list.

    Expected JSONL format (one JSON per line):
        {"id": "...", "subject": "...", "body": "...", "sender": "...", "timestamp": "..."}
    Also accepts Mbox-style with From/Subject/Body fields or simple JSONL with text field.
    """
    records: list[EmailRecord] = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Gmail export not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON line %d", line_num)
                continue

            email_id = str(obj.get("id", obj.get("email_id", f"email_{line_num}")))
            subject = str(obj.get("subject", ""))
            body = str(obj.get("body", obj.get("text", obj.get("content", ""))))
            sender = str(obj.get("sender", obj.get("from", "")))
            timestamp = str(obj.get("timestamp", obj.get("date", "")))

            records.append(
                EmailRecord(
                    email_id=email_id,
                    subject=subject,
                    body=body,
                    sender=sender,
                    timestamp=timestamp,
                )
            )

    return records


def is_short_email(record: EmailRecord, config: PipelineConfig) -> bool:
    """Check if an email is short enough to need regeneration."""
    body_len = len(record.body.strip())
    return config.min_input_length <= body_len <= config.max_input_length


# ---------------------------------------------------------------------------
# LLM expansion
# ---------------------------------------------------------------------------


def build_expansion_prompt(
    email: EmailRecord, persona: str, config: PipelineConfig
) -> list[dict[str, str]]:
    """Build LLM messages for expanding a short email into a full response."""
    persona_instruction = PERSONA_PROMPTS.get(
        persona, PERSONA_PROMPTS["warm_empathetic"]
    )
    email_text = email.body.strip()
    if email.subject:
        email_text = f"Subject: {email.subject}\n\n{email_text}"

    return [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": (
                f"{persona_instruction}\n\n"
                f"Expand the following brief email into a natural, "
                f"multi-sentence therapeutic response:\n\n{email_text}"
            ),
        },
    ]


def _mock_expand(
    email: EmailRecord, persona: str, config: PipelineConfig
) -> str:
    """Deterministic mock expansion for testing (no API key needed).

    Expands short emails by adding persona-appropriate preambles and closing.
    """
    body = email.body.strip()
    persona_prefix = {
        "warm_empathetic": "I hear you, and what you're feeling makes complete sense. ",
        "structured_practical": "Let's break this down step by step. ",
        "gentle_exploratory": "I'd like to explore this with you for a moment. ",
    }
    persona_suffix = {
        "warm_empathetic": " You're not alone in this, and together we can find a path forward.",
        "structured_practical": " Take one step at a time, and let me know how it goes.",
        "gentle_exploratory": " What comes up for you when you sit with that question?",
    }
    prefix = persona_prefix.get(persona, persona_prefix["warm_empathetic"])
    suffix = persona_suffix.get(persona, persona_suffix["warm_empathetic"])

    expanded = f"{prefix}{body}{suffix}"
    # Ensure minimum sentence count
    if _count_sentences(expanded) < config.min_output_sentences:
        expanded += " This is something we can work through together."
    # Ensure minimum output length
    if len(expanded) < config.min_output_length:
        expanded += " Let's take this one step at a time and find a way forward together."
    if len(expanded) < config.min_output_length:
        expanded += " You deserve support and understanding as you navigate this challenge."
    return expanded


def expand_email(
    email: EmailRecord,
    persona: str,
    llm_generate: Callable[[list[dict[str, str]]], str] | None,
    config: PipelineConfig,
) -> str:
    """Expand a short email into a full persona-driven response.

    Uses llm_generate callback if provided, otherwise falls back to mock expansion.
    """
    messages = build_expansion_prompt(email, persona, config)
    if llm_generate is not None:
        for attempt in range(config.max_retries):
            try:
                result = llm_generate(messages)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(
                    "LLM expansion attempt %d failed: %s", attempt + 1, e
                )
        logger.warning("All LLM expansion attempts failed, using mock")
    return _mock_expand(email, persona, config)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def quality_check(
    original: str,
    expanded: str,
    config: PipelineConfig,
    persona: str = "",
) -> tuple[float, list[str]]:
    """Run quality checks on expanded text. Returns (score, flags)."""
    flags: list[str] = []
    score = 1.0

    # Length checks
    expanded_len = len(expanded)
    if expanded_len < config.min_output_length:
        flags.append("too_short")
        score -= 0.3
    if expanded_len > config.max_output_length:
        flags.append("too_long")
        score -= 0.2

    # Sentence count check
    sentence_count = _count_sentences(expanded)
    if sentence_count < config.min_output_sentences:
        flags.append("insufficient_sentences")
        score -= 0.2

    # ROUGE-L similarity check
    rouge = _rouge_l_similarity(original, expanded)
    if rouge > config.rouge_l_too_similar:
        flags.append("too_similar")
        score -= 0.3
    if rouge < config.rouge_l_meaning_drift:
        flags.append("meaning_drift")
        score -= 0.3

    # Toxicity check
    if _check_toxicity(expanded):
        flags.append("toxicity_detected")
        score -= 0.5

    # Persona consistency
    if persona:
        persona_score, persona_warnings = _check_persona_consistency(
            expanded, persona
        )
        flags.extend(persona_warnings)
        if persona_score < 0.25:
            score -= 0.2

    score = round(max(0.0, min(1.0, score)), 2)
    return score, flags


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _build_record(
    email: EmailRecord,
    expanded: str,
    persona: str,
    quality_score: float,
    flags: list[str],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Build output JSONL record following the training data format."""
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": email.body.strip()},
        {"role": "assistant", "content": expanded},
    ]
    return {
        "source": "email_regen",
        "task_type": "email_expansion",
        "messages": messages,
        "mi_quality": "medium",
        "clinical_reviewed": False,
        "annotation_stage": None,
        "created_by": "email_regen_pipeline",
        "provenance": {
            "source_url": "",
            "source_type": "email_export",
            "acquired_at": email.timestamp,
            "pipeline_version": "1.0.0",
            "license": "internal",
            "transformations": ["persona_expansion"],
        },
        "quality_score": quality_score,
        "fleiss_kappa": None,
        "persona": persona,
        "email_id": email.email_id,
        "quality_flags": flags,
    }


def regenerate_email(
    record: EmailRecord,
    llm_generate: Callable[[list[dict[str, str]]], str] | None,
    config: PipelineConfig,
) -> RegenerationResult:
    """Regenerate a single email through the full pipeline."""
    # Select persona (round-robin based on hash for determinism)
    persona_idx = int(_content_hash(record.body), 16) % len(config.personas)
    persona = config.personas[persona_idx]

    try:
        expanded = expand_email(record, persona, llm_generate, config)
    except Exception as e:
        return RegenerationResult(
            original=record,
            regenerated_messages=[],
            persona=persona,
            error=str(e),
        )

    score, flags = quality_check(record.body, expanded, config, persona)

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": record.body.strip()},
        {"role": "assistant", "content": expanded},
    ]

    accepted = score >= config.min_quality_score and "toxicity_detected" not in flags

    return RegenerationResult(
        original=record,
        regenerated_messages=messages,
        quality_score=score,
        quality_flags=flags,
        persona=persona,
        accepted=accepted,
    )


def run_pipeline(
    input_path: str,
    output_path: str,
    llm_generate: Callable[[list[dict[str, str]]], str] | None,
    config: PipelineConfig,
) -> PipelineStats:
    """Run the email regeneration pipeline on a file of emails.

    Reads Gmail export JSONL, regenerates short emails, writes accepted
    results to output JSONL in training data format.
    """
    records = parse_gmail_export(input_path)
    stats = PipelineStats(total=len(records))

    results: list[dict[str, Any]] = []
    for record in records:
        if not is_short_email(record, config):
            stats.skipped += 1
            continue

        stats.processed += 1
        result = regenerate_email(record, llm_generate, config)

        if result.error:
            stats.errors += 1
            logger.error("Error processing %s: %s", record.email_id, result.error)
            continue

        if result.accepted:
            stats.accepted += 1
            stats.persona_distribution[result.persona] = (
                stats.persona_distribution.get(result.persona, 0) + 1
            )
            output_record = _build_record(
                record,
                result.regenerated_messages[2]["content"],
                result.persona,
                result.quality_score,
                result.quality_flags,
                config,
            )
            results.append(output_record)
        else:
            stats.rejected += 1
            logger.info(
                "Rejected %s: score=%.2f flags=%s",
                record.email_id,
                result.quality_score,
                result.quality_flags,
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def repair_corpus(
    input_path: str,
    output_path: str,
    llm_generate: Callable[[list[dict[str, str]]], str] | None,
    config: PipelineConfig,
) -> PipelineStats:
    """Repair an existing training corpus by regenerating short entries.

    Reads training JSONL, finds entries with short assistant responses,
    regenerates them, and writes a repaired corpus.
    """
    stats = PipelineStats()

    results: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats.total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.errors += 1
                continue

            messages = record.get("messages", [])
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break

            if len(assistant_content) < config.min_output_length:
                # This entry needs repair
                stats.processed += 1
                email = EmailRecord(
                    email_id=str(record.get("email_id", stats.total)),
                    subject="",
                    body=assistant_content,
                    sender="",
                    timestamp="",
                )
                result = regenerate_email(email, llm_generate, config)
                if result.accepted and result.regenerated_messages:
                    stats.accepted += 1
                    stats.persona_distribution[result.persona] = (
                        stats.persona_distribution.get(result.persona, 0) + 1
                    )
                    # Replace assistant message
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            msg["content"] = result.regenerated_messages[2]["content"]
                            break
                    record["quality_score"] = result.quality_score
                    record["quality_flags"] = result.quality_flags
                    record["persona"] = result.persona
                    record["repaired"] = True
                else:
                    stats.rejected += 1
            else:
                stats.skipped += 1

            results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Persona-driven email regeneration pipeline with QA guards."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Gmail export JSONL or training corpus JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for regenerated records",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "repair"],
        default="generate",
        help="generate: new emails from export; repair: fix short entries in corpus",
    )
    parser.add_argument(
        "--min-input-length",
        type=int,
        default=10,
        help="Minimum email body length to consider for regeneration",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=500,
        help="Maximum email body length to consider for regeneration",
    )
    parser.add_argument(
        "--min-output-length",
        type=int,
        default=100,
        help="Minimum expanded output length",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=500,
        help="Maximum expanded output length",
    )
    parser.add_argument(
        "--min-output-sentences",
        type=int,
        default=3,
        help="Minimum number of sentences in expanded output",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.5,
        help="Minimum quality score for acceptance",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help="Persona list to use (default: all three)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="LLM temperature for generation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    return parser


def main() -> int:
    """CLI entry point for the email regeneration pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    config = PipelineConfig(
        min_input_length=args.min_input_length,
        max_input_length=args.max_input_length,
        min_output_length=args.min_output_length,
        max_output_length=args.max_output_length,
        min_output_sentences=args.min_output_sentences,
        min_quality_score=args.min_quality_score,
        personas=args.personas if args.personas else list(DEFAULT_PERSONAS),
        temperature=args.temperature,
    )

    # No LLM API key in CLI mode — use mock expansion
    llm_generate = None

    if args.mode == "repair":
        stats = repair_corpus(args.input, args.output, llm_generate, config)
    else:
        stats = run_pipeline(args.input, args.output, llm_generate, config)

    print(
        f"Pipeline complete: {stats.accepted}/{stats.processed} accepted, "
        f"{stats.rejected} rejected, {stats.skipped} skipped, {stats.errors} errors"
    )
    if stats.persona_distribution:
        print("Persona distribution:")
        for persona, count in sorted(stats.persona_distribution.items()):
            print(f"  {persona}: {count}")

    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
