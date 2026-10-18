from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

TRUSTED_THERAPEUTIC_CHANNELS: Final[frozenset[str]] = frozenset(
    {
        "BorderlinerNotes",
        "Christopher Germer, Ph.D.",
        "Crappy Childhood Fairy",
        "DoctorRamani",
        "Doc Snipes",
        "Heidi Priebe",
        "Irene Lyon",
        "Jerry Wise",
        "Jim Hopper",
        "Kristin Snowden",
        "MedCircle",
        "Navigating Narcissism",
        "Patrick Teahan",
        "Patrick Teahan ",
        "Phoenix Trauma Center & Dr Scott Giacomucci",
        "Psych2Go",
        "Rebecca C. Mandeville LMFT Scapegoat Abuse Expert",
        "Sounds True",
        "Surviving Narcissism",
        "Therapy Chat Podcast",
        "Therapy Decoded",
        "Therapy in a Nutshell",
        "Therapy in a Nutshell Podcast",
        "Tim Fletcher",
    }
)
BLOCKED_CHANNELS: Final[frozenset[str]] = frozenset(
    {
        "10% Happier",
        "Big Think",
        "Full Story Lane",
        "Inspiration Hub",
        "Jimmy Kimmel Live",
        "Jordan Peterson Motivation Mastery",
        "LastWeekTonight",
        "Limitless Motivation",
        "MindShift Motivation",
        "MSNBC",
        "NA",
        "Narcissist Unveiled",
        "New Life Covenant Church of God",
        "THE MOTIVATIONAL MIND",
        "Theo Von",
        "The Late Show with Stephen Colbert",
        "Veritasium",
    }
)
STRICT_CHANNEL_AUTHORITY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bbody keeps the score\b"),
    re.compile(r"\bdr\.?\b"),
    re.compile(r"\bdoctor\b"),
    re.compile(r"\bexpert\b"),
    re.compile(r"\bpsychiatrist\b"),
    re.compile(r"\bpsychologist\b"),
    re.compile(r"\btherapist\b"),
    re.compile(r"\bbessel\b"),
    re.compile(r"\bdr\.?\s+rick\b"),
    re.compile(r"\bneurosurgeon\b"),
    re.compile(r"\bramani\b"),
)
THERAPEUTIC_KEYWORDS: Final[tuple[str, ...]] = (
    "abuse",
    "adhd",
    "anger",
    "anxiety",
    "attachment",
    "betrayal",
    "body keeps the score",
    "boundar",
    "childhood trauma",
    "codependency",
    "complex trauma",
    "cptsd",
    "depression",
    "dissociation",
    "dysregulation",
    "emotional abuse",
    "emotional neglect",
    "family mobbing",
    "healing",
    "inner child",
    "mindfulness",
    "narciss",
    "nervous system",
    "ptsd",
    "regulat",
    "scapegoat",
    "self worth",
    "shame",
    "somatic",
    "suicid",
    "therapy",
    "trauma",
    "trigger",
)


class ProvenanceMetadata(TypedDict, total=False):
    channel: str
    transcript_file: str


class Provenance(TypedDict, total=False):
    metadata: ProvenanceMetadata


class YouTubeRecord(TypedDict, total=False):
    instruction: str
    language: str
    output: str
    provenance: Provenance
    source_channel: str


@dataclass(frozen=True, slots=True)
class ChannelRubric:
    require_authority_signal: bool = False
    low_confidence_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CurationDecision:
    include: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CurationStats:
    input_dir: str
    output_path: str
    report_path: str
    generated_at: str
    scanned_records: int
    included_records: int
    excluded_records: int
    decision_counts: dict[str, int]
    included_channels: dict[str, int]


STRICT_CHANNEL_RUBRICS: Final[dict[str, ChannelRubric]] = {
    "Common Ego": ChannelRubric(require_authority_signal=True),
    "Dhru Purohit": ChannelRubric(require_authority_signal=True),
    "Dr Rangan Chatterjee": ChannelRubric(require_authority_signal=True),
    "Finding Mastery": ChannelRubric(require_authority_signal=True),
    "Forrest Hanson": ChannelRubric(require_authority_signal=True),
    "How To Academy": ChannelRubric(require_authority_signal=True),
    "Mel Robbins": ChannelRubric(
        require_authority_signal=True,
        low_confidence_terms=("betrayal", "manifest"),
    ),
    "Shawn Stevenson": ChannelRubric(
        require_authority_signal=True,
        low_confidence_terms=("money trauma",),
    ),
    "Tamsen Fadal": ChannelRubric(require_authority_signal=True),
    "TEDx Talks": ChannelRubric(require_authority_signal=True),
    "The Diary Of A CEO": ChannelRubric(
        require_authority_signal=True,
        low_confidence_terms=("authenticity",),
    ),
}


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _therapeutic_signal(value: str) -> bool:
    haystack = _normalized_text(value)
    return any(keyword in haystack for keyword in THERAPEUTIC_KEYWORDS)


def _matches_any_pattern(value: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    haystack = _normalized_text(value)
    return any(pattern.search(haystack) for pattern in patterns)


def _contains_any_term(value: str, terms: tuple[str, ...]) -> bool:
    haystack = _normalized_text(value)
    return any(term in haystack for term in terms)


def _record_channel(record: YouTubeRecord) -> str:
    channel = record.get("source_channel")
    if isinstance(channel, str) and channel.strip():
        return channel
    metadata = record.get("provenance", {}).get("metadata", {})
    fallback_channel = metadata.get("channel")
    return fallback_channel.strip() if isinstance(fallback_channel, str) else ""


def _record_title(record: YouTubeRecord) -> str:
    metadata = record.get("provenance", {}).get("metadata", {})
    transcript_file = metadata.get("transcript_file")
    if not isinstance(transcript_file, str):
        return ""
    return Path(transcript_file).stem


def _parse_record(raw_line: str) -> YouTubeRecord:
    parsed = json.loads(raw_line)
    if not isinstance(parsed, dict):
        raise TypeError("expected JSON object record")

    record: YouTubeRecord = {}
    instruction = parsed.get("instruction")
    if isinstance(instruction, str):
        record["instruction"] = instruction
    language = parsed.get("language")
    if isinstance(language, str):
        record["language"] = language
    output = parsed.get("output")
    if isinstance(output, str):
        record["output"] = output
    source_channel = parsed.get("source_channel")
    if isinstance(source_channel, str):
        record["source_channel"] = source_channel

    provenance_raw = parsed.get("provenance")
    if isinstance(provenance_raw, dict):
        metadata_raw = provenance_raw.get("metadata")
        metadata: ProvenanceMetadata = {}
        if isinstance(metadata_raw, dict):
            channel = metadata_raw.get("channel")
            if isinstance(channel, str):
                metadata["channel"] = channel
            transcript_file = metadata_raw.get("transcript_file")
            if isinstance(transcript_file, str):
                metadata["transcript_file"] = transcript_file
        provenance: Provenance = {}
        if metadata:
            provenance["metadata"] = metadata
        if provenance:
            record["provenance"] = provenance

    return record


def _decide_strict_channel_action(title: str, rubric: ChannelRubric) -> CurationDecision:
    if _contains_any_term(title, rubric.low_confidence_terms):
        return CurationDecision(include=False, reason="strict_channel_low_confidence_title")
    if not _therapeutic_signal(title):
        return CurationDecision(include=False, reason="non_therapeutic_title")
    if not rubric.require_authority_signal:
        return CurationDecision(include=True, reason="therapeutic_title")
    if _matches_any_pattern(title, STRICT_CHANNEL_AUTHORITY_PATTERNS):
        return CurationDecision(include=True, reason="strict_channel_authority_signal")
    return CurationDecision(include=False, reason="strict_channel_needs_authority_signal")


def decide_record_action(record: YouTubeRecord) -> CurationDecision:
    """Classify whether a YouTube training record belongs in the therapy subset."""
    channel = _record_channel(record)
    title = _record_title(record)

    if channel in TRUSTED_THERAPEUTIC_CHANNELS:
        return CurationDecision(include=True, reason="trusted_channel")
    if channel in BLOCKED_CHANNELS:
        return CurationDecision(include=False, reason="blocked_channel")
    rubric = STRICT_CHANNEL_RUBRICS.get(channel)
    if rubric is not None:
        return _decide_strict_channel_action(title, rubric)
    if _therapeutic_signal(title):
        return CurationDecision(include=True, reason="therapeutic_title")
    return CurationDecision(include=False, reason="non_therapeutic_title")


def curate_youtube_output(*, input_dir: Path, output_path: Path, report_path: Path) -> CurationStats:
    """Write a therapy-focused JSONL subset and a JSON report from mixed YouTube output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    decision_counts: Counter[str] = Counter()
    included_channels: Counter[str] = Counter()
    scanned_records = 0
    included_records = 0

    with output_path.open("w", encoding="utf-8") as curated_file:
        for jsonl_path in sorted(input_dir.glob("*.jsonl")):
            with jsonl_path.open(encoding="utf-8") as source_file:
                for line in source_file:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = _parse_record(stripped)
                    scanned_records += 1
                    decision = decide_record_action(record)
                    decision_counts[decision.reason] += 1
                    if not decision.include:
                        continue
                    included_records += 1
                    included_channels[_record_channel(record)] += 1
                    curated_file.write(stripped)
                    curated_file.write("\n")

    stats = CurationStats(
        input_dir=str(input_dir),
        output_path=str(output_path),
        report_path=str(report_path),
        generated_at=datetime.now(UTC).isoformat(),
        scanned_records=scanned_records,
        included_records=included_records,
        excluded_records=scanned_records - included_records,
        decision_counts=dict(sorted(decision_counts.items())),
        included_channels=dict(sorted(included_channels.items())),
    )
    report_path.write_text(json.dumps(asdict(stats), indent=2) + "\n", encoding="utf-8")
    return stats
