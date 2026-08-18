#!/usr/bin/env python3
"""
Generic Therapist Voice Extraction System.

Extracts voice profiles from YouTube therapist transcripts, generates
scored training conversations, and reports clinical validity quality metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extraction_config import (
    CHANNEL_CONFIGS,
    DEFAULT_CONVERSATIONS,
    INGESTED_DIR,
    MAX_MARKED_SENTENCE_LENGTH,
    MIN_COMMON_PHRASE_COUNT,
    MIN_SENTENCE_WORDS,
    OUTPUT_BASE,
    TOPIC_BANK,
    get_config,
    resolve_channel_key,
)
from extraction_io import generate_quality_report, save_channel_output
from extraction_models import ChannelResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_therapist_voice")

try:
    from training.clinical_validity_scorer import ClinicalValidityScorer

    SCORER_AVAILABLE = True
except ImportError:
    ClinicalValidityScorer = None
    SCORER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Voice Profile Extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_voice_profile(texts: list[str], channel_name: str) -> dict:
    """Extract voice patterns from a list of transcript texts."""
    profile: dict[str, Any] = {
        "sentence_starters": Counter(),
        "transition_phrases": Counter(),
        "empathy_markers": Counter(),
        "common_phrases": Counter(),
        "analogies": [],
        "examples": [],
        "teaching_patterns": [],
    }

    all_text = "\n\n".join(texts)

    for text in texts:
        _analyze_single(text, profile)

    _extract_common_phrases(all_text, profile)
    _extract_teaching_style(all_text, profile)

    profile_report = _build_profile_report(channel_name, profile)

    return {
        "profile_raw": {
            "sentence_starters": dict(profile["sentence_starters"].most_common(50)),
            "transition_phrases": dict(profile["transition_phrases"].most_common(30)),
            "empathy_markers": dict(profile["empathy_markers"].most_common(30)),
            "common_phrases": dict(profile["common_phrases"].most_common(100)),
            "analogies": profile["analogies"][:50],
            "examples": profile["examples"][:50],
            "teaching_patterns": profile["teaching_patterns"],
        },
        "report": profile_report,
    }


def _analyze_single(text: str, profile: dict):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence in sentences:
        words = sentence.split()
        if len(words) >= MIN_SENTENCE_WORDS:
            starter = " ".join(words[:2])
            profile["sentence_starters"][starter] += 1

        transitions = [
            "And so",
            "Now",
            "So",
            "But",
            "And then",
            "What happens",
            "Let me",
            "Think about",
            "Imagine",
            "What I find",
            "One of the things",
            "The reality is",
            "What we see",
            "The truth is",
            "Here's the thing",
            "The key is",
            "What I want you to",
            "One of the key",
            "It's important to",
        ]
        for t in transitions:
            if sentence.lower().startswith(t.lower()):
                profile["transition_phrases"][t] += 1

        empathy_patterns = [
            "I understand",
            "I know",
            "I get it",
            "That's painful",
            "That's hard",
            "You might feel",
            "Many people",
            "Some of you",
            "For many",
            "What you're going through",
            "It makes sense that",
            "It's understandable",
            "You're not alone",
            "That must be",
            "I hear you",
            "I can see how",
        ]
        for p in empathy_patterns:
            if p.lower() in sentence.lower():
                profile["empathy_markers"][p] += 1

        if any(m in sentence.lower() for m in ["like a", "as if", "imagine", "think of"]):
            if len(sentence) < MAX_MARKED_SENTENCE_LENGTH:
                profile["analogies"].append(sentence)

        if any(m in sentence.lower() for m in ["let's say", "for example", "think back to"]):
            if len(sentence) < MAX_MARKED_SENTENCE_LENGTH:
                profile["examples"].append(sentence)


def _extract_common_phrases(text: str, profile: dict):
    words = text.lower().split()
    for i in range(len(words) - 2):
        phrase = " ".join(words[i : i + 3])
        profile["common_phrases"][phrase] += 1


def _extract_teaching_style(text: str, profile: dict):
    patterns = [
        "First",
        "Second",
        "Third",
        "What happens is",
        "The reality is",
        "What we find",
        "One of the key",
        "It's important to understand",
        "Let me give you an example",
        "Think about this",
        "What I mean by that",
        "Here's what I want you to",
        "The reason for this",
        "What we know from",
    ]
    for pattern in patterns:
        count = text.lower().count(pattern.lower())
        if count > 0:
            profile["teaching_patterns"].append(
                {
                    "pattern": pattern,
                    "frequency": count,
                }
            )


def _build_profile_report(channel_name: str, profile: dict) -> str:
    report = [f"# {channel_name} Voice Profile\n"]
    len(profile.get("analogies", [])) + len(profile.get("examples", []))
    report.append(f"**Analyzed**: {len(profile.get('sentence_starters', {}))} unique patterns\n\n")

    report.append("## Top Sentence Starters\n")
    for starter, count in profile["sentence_starters"].most_common(20):
        report.append(f'- **"{starter}..."** ({count} times)\n')

    report.append("\n## Transition Phrases\n")
    for phrase, count in profile["transition_phrases"].most_common(15):
        report.append(f'- **"{phrase}"** ({count} times)\n')

    report.append("\n## Empathy & Connection Markers\n")
    for marker, count in profile["empathy_markers"].most_common(15):
        report.append(f'- **"{marker}"** ({count} times)\n')

    report.append("\n## Sample Analogies & Metaphors\n")
    for analogy in profile["analogies"][:10]:
        report.append(f"- {analogy}\n")

    report.append("\n## Sample Examples\n")
    for example in profile["examples"][:10]:
        report.append(f"- {example}\n")

    report.append("\n## Common 3-Word Phrases\n")
    for phrase, count in profile["common_phrases"].most_common(30):
        if count > MIN_COMMON_PHRASE_COUNT:
            report.append(f'- "{phrase}" ({count} times)\n')

    return "".join(report)


# ═══════════════════════════════════════════════════════════════════════════════
#  Conversation Generation
# ═══════════════════════════════════════════════════════════════════════════════


def discover_channels(source_dir: Path = INGESTED_DIR) -> list[str]:
    """Discover channels that have ingested markdown transcripts."""
    found: set[str] = set()
    if not source_dir.exists():
        logger.warning("Source dir %s not found", source_dir)
        return []

    for fname in os.listdir(source_dir):
        if not fname.endswith(".md"):
            continue
        for ck in CHANNEL_CONFIGS:
            if fname.startswith(ck):
                found.add(ck)

    listed = sorted(found)
    logger.info("Discovered %d channels with transcripts: %s", len(listed), listed)
    return listed


def load_channel_transcripts(
    channel_key: str,
    source_dir: Path = INGESTED_DIR,
) -> list[tuple[str, str]]:
    """Load transcripts for a channel. Returns list of (title, content)."""
    results: list[tuple[str, str]] = []
    if not source_dir.exists():
        return results

    for fname in os.listdir(source_dir):
        if not fname.endswith(".md"):
            continue
        if not fname.startswith(channel_key):
            continue
        title = fname.removesuffix(".md")
        with open(source_dir / fname, encoding="utf-8") as f:
            content = f.read()
        results.append((title, content))

    results.sort(key=lambda x: x[0])
    return results


def _strip_transcript_metadata(content: str) -> list[str]:
    lines = content.split("\n")
    cleaned = []
    in_transcript = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Transcript"):
            in_transcript = True
            continue
        if not in_transcript:
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if len(para) < 100:
            continue
        if para.startswith("**") and para.endswith("**"):
            continue
        paragraphs.append(para)
    return paragraphs


_CLIENT_QUESTION_TEMPLATES = [
    "I've been struggling with something related to this and I don't know where to start. Can you help me understand?",
    "This hits close to home. How do I know if this is something I'm dealing with?",
    "I think I might be experiencing some of this. What should I look for?",
    "Can you explain what this looks like in everyday life? I want to understand if it applies to my situation.",
    "What are the signs I should be paying attention to? I feel like I'm missing something obvious.",
    "How does this actually show up in someone's life? I'm trying to understand the practical signs.",
    "I've been wondering about this for a while. What does the research say?",
    "Is this something that can get better on its own, or do I need to actively work on it?",
    "What's the first thing I should understand about this? I feel overwhelmed with information.",
    "Why does this happen? I want to understand the root cause, not just the symptoms.",
]


def generate_conversation_from_transcript(
    title: str,
    content: str,
    config: dict,
) -> dict:
    """Generate a training conversation from a single transcript.

    Parses the raw transcript into coherent multi-turn therapeutic dialogue:
    strips metadata headers, extracts substantial paragraphs, and wraps each
    in a client-question / therapist-response exchange.
    """
    system_prompt = (
        f"You are {config['name']}. {config['description']}. "
        f"Your approach is {config['approach']}. "
        f"Your expertise includes: {', '.join(config['expertise'])}. "
        "Respond to the client's questions using your therapeutic voice and expertise."
    )

    paragraphs = _strip_transcript_metadata(content)

    messages = [{"role": "system", "content": system_prompt}]

    for i, para in enumerate(paragraphs):
        client_q_idx = i % len(_CLIENT_QUESTION_TEMPLATES)
        client_msg = _CLIENT_QUESTION_TEMPLATES[client_q_idx]
        messages.append({"role": "client", "content": client_msg})
        messages.append({"role": "therapist", "content": para.strip()})

    if not messages:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "client", "content": "Can you help me understand this topic?"},
            {"role": "therapist", "content": content},
        ]

    return {
        "conversation_id": f"{config['signature']}_{title}",
        "stage": "stage4_voice_persona",
        "messages": messages,
        "metadata": {
            "source": f"{config['signature']}_transcripts",
            "source_family": "stage4_voice_persona",
            "voice_signature": f"{config['signature']}_v1",
            "persona_id": config["signature"],
            "personality_markers": {
                "style": config["style"],
                "approach": config["approach"],
                "expertise_areas": config["expertise"],
            },
            "generated_at": datetime.now(UTC).isoformat(),
            "paragraph_count": len(paragraphs),
        },
    }


def generate_synthetic_conversations(
    config: dict,
    num_conversations: int,
    profile: dict | None = None,
) -> list[dict]:
    conversations: list[dict] = []
    topics = _get_topics_for_expertise(config.get("expertise", []))

    system_prompt = (
        f"You are {config['name']}. {config['description']}. "
        f"Your approach is {config['approach']}. "
        "Respond with deep empathy, clinical insight, and practical guidance."
    )

    for i in range(num_conversations):
        topic = topics[i % len(topics)]
        conversation = _build_synthetic_dialogue(i, topic, config)
        conversation["messages"].insert(0, {"role": "system", "content": system_prompt})
        conversations.append(conversation)

    return conversations


def _get_topics_for_expertise(expertise: list[str]) -> list[str]:
    topics: list[str] = []
    for area in expertise:
        if "trauma" in area or "PTSD" in area or "CPTSD" in area or "nervous" in area:
            topics.extend(TOPIC_BANK["trauma"])
        if "personality" in area or "narcissis" in area or "BPD" in area:
            topics.extend(TOPIC_BANK["personality_disorders"])
        if "CBT" in area or "DBT" in area or "cbt" in area or "dbt" in area:
            topics.extend(TOPIC_BANK["cbt_dbt"])
        if "attachment" in area:
            topics.extend(TOPIC_BANK["attachment"])
    if not topics:
        topics = TOPIC_BANK["general"]
    return topics


from synthetic_templates import SYNTHETIC_TEMPLATES


def _build_synthetic_dialogue(
    index: int,
    topic: str,
    config: dict,
) -> dict:
    template = random.choice(SYNTHETIC_TEMPLATES)
    conversation = [
        {"role": "client", "content": template["client"]},
        {"role": "therapist", "content": template["therapist"]},
    ]

    return {
        "conversation_id": f"{config['signature']}_synthetic_{index:04d}",
        "stage": "stage4_voice_persona",
        "messages": conversation,
        "metadata": {
            "source": f"{config['signature']}_synthetic",
            "source_family": "stage4_voice_persona",
            "voice_signature": f"{config['signature']}_v1",
            "persona_id": config["signature"],
            "personality_markers": {
                "style": config["style"],
                "approach": config["approach"],
                "expertise_areas": config.get("expertise", []),
            },
            "topic": topic,
            "index": index,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════════════


def score_conversations(conversations: list[dict]) -> tuple[list[float], list[dict]]:
    """Score each conversation's therapist responses using ClinicalValidityScorer.

    Scores each individual message separately and averages them to avoid
    inflating the score via sheer verbosity (e.g. raw 10K-token transcript dumps).
    """
    if not SCORER_AVAILABLE:
        logger.warning("ClinicalValidityScorer not available; skipping scoring")
        return [], []

    assert ClinicalValidityScorer is not None  # narrow type for Pyright

    scores: list[float] = []
    details: list[dict] = []

    for conv in conversations:
        message_scores: list[float] = []
        combined_detail: dict[str, float] = {}
        msg_count = 0

        for msg in conv.get("messages", []):
            if msg.get("role") in ("therapist", "assistant"):
                content = msg.get("content", "")
                if content.strip():
                    msg_count += 1
                    msg_detail = ClinicalValidityScorer.score_detail(content)
                    msg_score = ClinicalValidityScorer.score(content)
                    message_scores.append(msg_score)
                    for dim, val in msg_detail.items():
                        combined_detail.setdefault(dim, 0.0)
                        combined_detail[dim] = max(combined_detail[dim], val)

        if not message_scores:
            scores.append(0.0)
            details.append(dict.fromkeys(ClinicalValidityScorer.WEIGHTS, 0.0))
        else:
            avg_score = sum(message_scores) / len(message_scores)
            scores.append(avg_score)
            details.append(combined_detail)

    return scores, details


def annotate_conversations(conversations: list[dict], scores: list[float], details: list[dict]):
    """Add clinical validity scores to conversation metadata."""
    for conv, score, detail in zip(conversations, scores, details, strict=False):
        conv.setdefault("metadata", {})["clinical_validity"] = {
            "score": round(score, 4),
            "dimensions": {k: round(v, 4) for k, v in detail.items()},
        }


def validate_conversation_quality(conversations: list[dict]) -> dict:
    """Run quality checks on generated conversations and return a report.

    Checks:
      - Each conversation has at least system + client + therapist messages.
      - No conversation has empty content.
      - Message roles are from the allowed set.
      - Score plausibility: warn if all scores are 0.9+ or all 0.0.
    """
    issues: list[str] = []
    allowed_roles = {"system", "client", "therapist", "user", "assistant"}
    empty_count = 0
    role_violations = 0
    low_msg_count = 0
    scores = []

    for conv in conversations:
        msgs = conv.get("messages", [])
        n_msgs = len(msgs)
        roles = {m.get("role") for m in msgs}
        invalid_roles = roles - allowed_roles
        if invalid_roles:
            role_violations += 1
        if n_msgs < 3:
            low_msg_count += 1
        has_empty = any(not (m.get("content") or "").strip() for m in msgs)
        if has_empty:
            empty_count += 1
        cv = conv.get("metadata", {}).get("clinical_validity", {})
        if "score" in cv:
            scores.append(cv["score"])

    if role_violations:
        issues.append(f"{role_violations} conversations have invalid roles")
    if low_msg_count:
        issues.append(f"{low_msg_count} conversations have fewer than 3 messages")
    if empty_count:
        issues.append(f"{empty_count} conversations have empty message content")

    if scores:
        avg = sum(scores) / len(scores)
        if avg > 0.85:
            issues.append(f"High average score ({avg:.3f}) — may indicate verbosity inflation")
        if avg < 0.05:
            issues.append(f"Near-zero average score ({avg:.3f}) — scorer may not be matching")
    else:
        issues.append("No scores found — conversation annotation may not have run")

    return {
        "total": len(conversations),
        "issues": issues,
        "issue_count": len(issues),
        "pass": len(issues) == 0,
        "details": {
            "role_violations": role_violations,
            "low_msg_count": low_msg_count,
            "empty_messages": empty_count,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Processing
# ═══════════════════════════════════════════════════════════════════════════════


def process_channel(
    channel_key: str,
    num_conversations: int = DEFAULT_CONVERSATIONS,
    source_dir: Path = INGESTED_DIR,
    enable_scoring: bool = True,
    force_synthetic: bool = False,
) -> ChannelResult:
    """Process a single channel end-to-end."""
    config = get_config(channel_key)
    if not config:
        logger.warning("Unknown channel: %s (skipping)", channel_key)
        return ChannelResult(name=channel_key)

    logger.info("Processing channel: %s", config["name"])

    transcripts = load_channel_transcripts(channel_key, source_dir)
    texts = [c for _, c in transcripts]
    titles = [t for t, _ in transcripts]

    result = ChannelResult(name=config["name"])
    result.transcripts = texts
    result.transcript_titles = titles

    if texts and not force_synthetic:
        logger.info("  Found %d ingested transcripts", len(texts))
        result.voice_profile = extract_voice_profile(texts, config["name"])

        for title, content in transcripts:
            conv = generate_conversation_from_transcript(title, content, config)
            result.conversations.append(conv)
        logger.info("  Generated %d conversations from transcripts", len(result.conversations))
    else:
        source = "synthetic" if force_synthetic else "no transcripts"
        logger.info("  Using synthetic generation (%s)", source)
        if texts:
            result.voice_profile = extract_voice_profile(texts, config["name"])
            for title, content in transcripts:
                conv = generate_conversation_from_transcript(title, content, config)
                result.conversations.append(conv)

        num_synthetic = max(0, num_conversations - len(result.conversations))
        if num_synthetic > 0:
            syn = generate_synthetic_conversations(config, num_synthetic, result.voice_profile)
            result.conversations.extend(syn)

    if enable_scoring and SCORER_AVAILABLE:
        logger.info("  Scoring %d conversations for clinical validity...", len(result.conversations))
        result.scores, result.score_detail = score_conversations(result.conversations)
        annotate_conversations(result.conversations, result.scores, result.score_detail)
        logger.info(
            "  Mean score: %.4f | Pass rate (≥0.5): %.1f%%",
            result.mean_score,
            result.pass_rate * 100,
        )
    else:
        logger.info("  Scoring skipped")

    quality = validate_conversation_quality(result.conversations)
    if quality["issues"]:
        logger.warning("  Quality issues: %s", "; ".join(quality["issues"]))
    result.validation_report = quality

    return result


def process_all_channels(
    num_conversations: int = DEFAULT_CONVERSATIONS,
    source_dir: Path = INGESTED_DIR,
    enable_scoring: bool = True,
    force_synthetic: bool = False,
) -> list[ChannelResult]:
    """Process all discovered channels."""
    channels = discover_channels(source_dir)
    results: list[ChannelResult] = []
    for ck in channels:
        result = process_channel(ck, num_conversations, source_dir, enable_scoring, force_synthetic)
        results.append(result)
        save_channel_output(ck, result)
    return results


def list_channels():
    """Print all known channels and their status."""

    for ck, _cfg in sorted(CHANNEL_CONFIGS.items()):
        len(load_channel_transcripts(ck, INGESTED_DIR))
        (OUTPUT_BASE / f"{ck.lower()}_voice" / f"{ck.lower()}_voice_profile.json").exists()

    other = []
    if INGESTED_DIR.exists():
        seen = set()
        for fname in os.listdir(INGESTED_DIR):
            if not fname.endswith(".md"):
                continue
            matched = False
            for ck in CHANNEL_CONFIGS:
                if fname.startswith(ck):
                    matched = True
                    break
            if not matched:
                prefix = fname.split("_")[0]
                if prefix not in seen:
                    seen.add(prefix)
                    other.append(prefix)

    if other:
        for _o in sorted(other)[:10]:
            pass
        if len(other) > 10:
            pass



# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract therapist voice profiles and generate scored training conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--channel", "-c", help="Channel name to process")
    parser.add_argument("--all", "-a", action="store_true", help="Process all discovered channels")
    parser.add_argument("--list", "-l", action="store_true", help="List known channels and exit")
    parser.add_argument(
        "--num-conversations",
        "-n",
        type=int,
        default=DEFAULT_CONVERSATIONS,
        help=f"Number of synthetic conversations (default: {DEFAULT_CONVERSATIONS})",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(INGESTED_DIR),
        help=f"Transcript source directory (default: {INGESTED_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_BASE),
        help=f"Output base directory (default: {OUTPUT_BASE})",
    )
    parser.add_argument("--no-score", action="store_true", help="Skip clinical validity scoring")
    parser.add_argument(
        "--force-synthetic",
        "-s",
        action="store_true",
        help="Force synthetic conversation generation even when transcripts exist",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to disk (default: True)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.list:
        list_channels()
        return

    source_dir = Path(args.source_dir)
    output_base = Path(args.output_dir)

    if args.all:
        logger.info("Batch processing all channels...")
        results = process_all_channels(
            num_conversations=args.num_conversations,
            source_dir=source_dir,
            enable_scoring=not args.no_score,
            force_synthetic=args.force_synthetic,
        )

        if args.save:
            report = generate_quality_report(results)
            report_file = output_base / "clinical_validity_quality_report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Saved quality report: %s", report_file)


    elif args.channel:
        ck = resolve_channel_key(args.channel)
        if not ck:
            logger.error("Unknown channel '%s'. Use --list to see available channels.", args.channel)
            sys.exit(1)

        result = process_channel(
            ck,
            num_conversations=args.num_conversations,
            source_dir=source_dir,
            enable_scoring=not args.no_score,
            force_synthetic=args.force_synthetic,
        )

        if args.save:
            save_channel_output(ck, result, output_base)

        if result.scores:
            pass

    else:
        logger.error("Specify --channel or --all. Use --help for details.")


if __name__ == "__main__":
    main()
