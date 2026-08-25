"""PII removal for therapeutic training data (three-layer).

Layer 1: regex fast-strip of obvious high-sensitivity entities (email, phone,
    SSN, credit card, IP). Cheap, no model.
Layer 2: Presidio (``AnalyzerEngine`` + ``AnonymizerEngine``) with the full
    entity set from blueprint B.2.2 — ``EMAIL_ADDRESS``, ``PHONE_NUMBER``,
    ``US_SSN``, ``CREDIT_CARD``, ``MEDICAL_LICENSE``, ``IP_ADDRESS``,
    ``PERSON``, ``LOCATION``, ``DATE_TIME``.
Layer 3: LLM pass on borderline spans (Presidio confidence < 0.8) to catch
    indirect-reference PII that rule-based recognizers miss (e.g. "my wife
    Sarah who lives on Elm Street"). Runs only when a NeMo config is supplied;
    otherwise the scrubber degrades gracefully to the two rule-based layers.

Usage::

    uv run python -m training.pii_scrubber                 # scan ChatML files
    uv run python -m training.pii_scrubber --text "..."    # scrub one string
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from training.stage1_filters import _PII_REDACTION, _PII_REGEXES

if TYPE_CHECKING:
    from training.sdg_pipeline import NemoConfig

logger = logging.getLogger("pii_scrubber")

# Presidio entity set (blueprint B.2.2) — broader than the regex Layer 1 set so
# the redactor can also scrub contextual identifiers.
PRESIDIO_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "PERSON",
    "LOCATION",
    "DATE_TIME",
)

# Presidio confidence below which a span is "borderline" and gets the Layer 3
# LLM re-check (blueprint B.2.2).
BORDERLINE_CONFIDENCE = 0.8

LLM_PII_SYSTEM_PROMPT = """You are a privacy redactor for therapeutic AI training data.
Your task is to find and redact personally identifiable information (PII) that
rule-based detectors miss, especially *indirect* references.

Redact the following, replacing each with a bracketed tag:
- [PERSON]: real names, nicknames, initials tied to a person, family members
  ("my wife Sarah", "Dr. Chen")
- [LOCATION]: specific addresses, streets, neighborhoods, clinics, hospitals
- [DATE_TIME]: birth dates, appointment dates, exact timestamps
- [PHONE_NUMBER], [EMAIL_ADDRESS]: any contact info, even partially obfuscated
- [ORGANIZATION]: employers, schools, named workplaces

Do NOT redact generic clinical content, symptoms, diagnoses, or therapeutic
language. Preserve everything else verbatim, including punctuation and line
breaks.

Output ONLY the redacted text — no preamble, no explanation, no code fences."""


@dataclass
class ScrubStats:
    """Per-text PII redaction counters across the three layers."""

    regex_hits: int = 0
    presidio_hits: int = 0
    borderline_hits: int = 0
    llm_hits: int = 0

    @property
    def total_hits(self) -> int:
        return self.regex_hits + self.presidio_hits + self.llm_hits


def strip_pii_regex(text: str) -> tuple[str, int]:
    """Layer 1 regex fast-strip. Returns ``(redacted_text, hit_count)``."""
    if not text:
        return (text, 0)
    redacted = text
    hits = 0
    for pattern in _PII_REGEXES.values():
        redacted, count = pattern.subn(_PII_REDACTION, redacted)
        hits += count
    return (redacted, hits)


def _analyze_presidio(text: str, analyzer: AnalyzerEngine) -> list[Any]:
    """Run Presidio and return recognizer results (empty on import/runtime failure)."""
    try:
        return analyzer.analyze(
            text=text,
            entities=list(PRESIDIO_ENTITIES),
            language="en",
        )
    except Exception as exc:  # Presidio runtime failures degrade to regex-only
        logger.warning("Presidio failed, skipping Layer 2: %s", exc)
        return []


def pii_llm_pass(text: str, nemo_config: NemoConfig) -> str:
    """Layer 3 LLM pass to catch indirect-reference PII.

    Returns the LLM-redacted text, or the original text on any failure.
    """
    if not text or not text.strip():
        return text
    # Lazy import to avoid a hard dependency on sdg_pipeline at module load.
    from training.sdg_pipeline import _call_nemo

    prompt = (
        "Redact any indirect or partially-obscured PII in the text below. "
        "Replace each PII instance with the appropriate bracketed tag. "
        "Return only the redacted text.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        raw = _call_nemo(prompt, nemo_config, system_prompt=LLM_PII_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("PII LLM pass failed: %s", exc)
        return text
    if not raw or not raw.strip():
        return text
    return raw.strip()


def scrub_text(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    nemo_config: NemoConfig | None = None,
) -> tuple[str, ScrubStats]:
    """Three-layer PII scrub. Returns ``(scrubbed_text, ScrubStats)``."""
    stats = ScrubStats()

    # Layer 1: regex fast-strip.
    text, stats.regex_hits = strip_pii_regex(text)
    if not text:
        return (text, stats)

    # Layer 2: Presidio.
    results = _analyze_presidio(text, analyzer)
    if results:
        borderline = [r for r in results if getattr(r, "score", 1.0) < BORDERLINE_CONFIDENCE]
        stats.borderline_hits = len(borderline)
        stats.presidio_hits = len(results)
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        text = anonymized.text

    # Layer 3: LLM pass on borderline spans (only when a NeMo config is supplied).
    if stats.borderline_hits > 0 and nemo_config is not None:
        before = text
        text = pii_llm_pass(text, nemo_config)
        # Heuristic hit count: bracketed tags introduced by the LLM.
        llm_tags = re.findall(r"\[(?:PERSON|LOCATION|DATE_TIME|PHONE_NUMBER|EMAIL_ADDRESS|ORGANIZATION)\]", text)
        stats.llm_hits = len(llm_tags) if text != before else 0

    return (text, stats)


def build_nemo_config_from_env() -> NemoConfig | None:
    """Build a NemoConfig from environment variables, or None if unset."""
    from training.sdg_pipeline import NemoConfig

    endpoint = os.getenv("NEMO_ENDPOINT", "") or os.getenv("NVIDIA_BASE_URL", "")
    api_key = os.getenv("NEMO_API_KEY", "") or os.getenv("NVIDIA_API_KEY", "")
    if not endpoint or not api_key:
        return None
    return NemoConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=os.getenv("NEMO_MODEL", "mistral-nemo"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub PII from therapeutic training text")
    parser.add_argument("--text", type=str, default=None, help="Single text to scrub")
    parser.add_argument("--dir", type=str, default="ai/training/output/books/chatml", help="ChatML directory to scan")
    args = parser.parse_args()

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    nemo_config = build_nemo_config_from_env()
    if nemo_config is None:
        logger.info("No NeMo config; Layer 3 LLM pass disabled.")

    # Single-text mode.
    if args.text is not None:
        scrubbed, stats = scrub_text(args.text, analyzer, anonymizer, nemo_config)
        print(json.dumps({"text": scrubbed, "stats": stats.__dict__}, indent=2))
        return

    # File-scan mode (legacy behavior).
    chatml_files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    if not chatml_files:
        print("No ChatML files found to scan.")
        return

    print(f"Found {len(chatml_files)} files. Starting PII scan...")
    total_pairs = 0
    total_redactions = 0

    for file_path in chatml_files:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        clean_lines = []
        file_redactions = 0

        for line in lines:
            try:
                pair = json.loads(line)
                total_pairs += 1
                messages = pair.get("messages", [])
                for msg in messages:
                    content = msg.get("content", "")
                    if not isinstance(content, str) or not content:
                        continue
                    scrubbed, stats = scrub_text(content, analyzer, anonymizer, nemo_config)
                    if stats.total_hits > 0:
                        msg["content"] = scrubbed
                        file_redactions += stats.total_hits
                clean_lines.append(json.dumps(pair))
            except Exception as e:
                print(f"Error processing line: {e}")

        if file_redactions > 0:
            print(f"Found and scrubbed {file_redactions} PII instances in {os.path.basename(file_path)}")
            total_redactions += file_redactions
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(clean_lines) + "\n")
        else:
            print(f"Clean (0 PII instances): {os.path.basename(file_path)}")

    print("=" * 40)
    print("PII Scan Complete.")
    print(f"Total Conversations Scanned: {total_pairs}")
    print(f"Total Redactions Made: {total_redactions}")


if __name__ == "__main__":
    main()
