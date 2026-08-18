#!/usr/bin/env python3
"""Auto-categorize YouTube training samples by channel name and content keywords."""

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("auto_categorize_youtube")

# Channel-to-category mapping based on primary therapeutic focus
CHANNEL_CATEGORIES: dict[str, str] = {
    "DoctorRamani": "narcissistic_abuse_recovery",
    "Rebecca C. Mandeville LMFT Scapegoat Abuse Expert": "narcissistic_abuse_recovery",
    "Jerry Wise": "narcissistic_abuse_recovery",
    "Kerry McAvoy, PhD": "narcissistic_abuse_recovery",
    "Crappy Childhood Fairy": "narcissistic_abuse_recovery",
    "Lisa A. Romano Breakthrough Life Coach Inc": "narcissistic_abuse_recovery",
    "RichardGrannon": "narcissistic_abuse_recovery",
    "richardgrannon": "narcissistic_abuse_recovery",
    "Surviving Narcissism": "narcissistic_abuse_recovery",
    "Common Ego": "narcissistic_abuse_recovery",
    "Narcissist Unveiled": "narcissistic_abuse_recovery",
    "Navigating Narcissism": "narcissistic_abuse_recovery",
    "Unfilteredd： Narcissistic Families": "narcissistic_abuse_recovery",
    "Natalie Dawson": "narcissistic_abuse_recovery",
    "Prof. Sam Vaknin": "narcissistic_abuse_recovery",
    "Michele Lee Nieves Coaching": "narcissistic_abuse_recovery",
    "hunds-kompetent Karin Actun": "narcissistic_abuse_recovery",
    "BorderlinerNotes": "personality_disorders",
    "Kris Reece": "personality_disorders",
    "Dr. Daniel Fox": "personality_disorders",
    "Dr. Todd Grande": "personality_disorders",
    "Heidi Priebe": "personality_disorders",
    "katimorton": "personality_disorders",
    "Kati Morton": "personality_disorders",
    "Tim Fletcher": "complex_ptsd",
    "Doc Snipes": "complex_ptsd",
    "Irene Lyon": "complex_ptsd",
    "Patrick Teahan ": "complex_ptsd",
    "Phoenix Trauma Center & Dr Scott Giacomucci": "complex_ptsd",
    "Breakthrough Zone": "complex_ptsd",
    "Dr. Scott Eilers": "complex_ptsd",
    "Dr. Kim Sage, Licensed Psychologist ": "complex_ptsd",
    "Jim Hopper": "complex_ptsd",
    "Forrest Hanson": "complex_ptsd",
    "derekscott": "complex_ptsd",
    "Shawn Stevenson": "somatic_therapy",
    "Dhru Purohit": "somatic_therapy",
    "gabormate": "somatic_therapy",
    "Couples Therapy Official": "attachment_disorders",
    "Therapy Chat Podcast": "attachment_disorders",
    "Therapy Decoded": "attachment_disorders",
    "Steph and Craig": "attachment_disorders",
    "Christopher Germer, Ph.D.": "mindfulness_meditation",
    "Eckhart Tolle": "mindfulness_meditation",
    "Sounds True": "mindfulness_meditation",
    "Meadow DeVor": "mindfulness_meditation",
    "10% Happier": "mindfulness_meditation",
    "Psych2Go": "general_mental_health",
    "MedCircle": "general_mental_health",
    "Theo Von": "general_mental_health",
    "Chris Williamson": "general_mental_health",
    "Jay Shetty Podcast": "general_mental_health",
    "Mel Robbins": "general_mental_health",
    "Jordan Peterson Motivation Mastery": "general_mental_health",
    "Limitless Motivation": "general_mental_health",
    "THE MOTIVATIONAL MIND": "general_mental_health",
    "Finding Mastery": "general_mental_health",
    "Inspiration Hub": "general_mental_health",
    "MindShift Motivation": "general_mental_health",
    "TEDx Talks": "general_mental_health",
    "Big Think": "general_mental_health",
    "How To Academy": "general_mental_health",
    "Veritasium": "general_mental_health",
    "Wu Wei Wisdom": "general_mental_health",
    "Kristin Snowden": "general_mental_health",
    "Tamsen Fadal": "general_mental_health",
    "Understood": "general_mental_health",
    "Doug Bopst": "addiction_recovery",
    "NA": "addiction_recovery",
    "Sandstone Care": "addiction_recovery",
    "SCSASmithers": "neurodivergent_mental_health",
    "arielleschwartz": "neurodivergent_mental_health",
    "Arielle_Schwartz": "neurodivergent_mental_health",
    "ARTE": "german_therapy",
    "ARTEde": "german_therapy",
    "DW Deutsch": "german_therapy",
    "SWR Doku": "german_therapy",
    "WDR": "german_therapy",
    "rbb Doku": "german_therapy",
    "Kaltblütig": "german_therapy",
    "Y-Kollektiv": "german_therapy",
    "ZDF MAGAZIN ROYALE": "german_therapy",
    "ZDFheute Nachrichten": "german_therapy",
    "Klein aber Hannah": "german_therapy",
    "RTS - Radio Télévision Suisse": "german_therapy",
    "Caroline Myss": "general_mental_health",
    "Full Story Lane": "general_mental_health",
    "Noetic Films": "general_mental_health",
    "Qbit Films": "general_mental_health",
    "Dr Rangan Chatterjee": "general_mental_health",
    "The Diary Of A CEO": "general_mental_health",
    "Stiff Nipple Epiphany": "general_mental_health",
    "LastWeekTonight": "general_mental_health",
    "The Late Show with Stephen Colbert": "general_mental_health",
    "Jimmy Kimmel Live": "general_mental_health",
    "MSNBC": "general_mental_health",
    "New Life Covenant Church of God": "cultural_religious_contexts",
    " psychologyinseattle": "general_mental_health",
    "psychologyinseattle": "general_mental_health",
    "krisgodinez": "general_mental_health",
    "Kris Godinez": "general_mental_health",
    "Therapy in a Nutshell": "general_mental_health",
    "Therapy in a Nutshell Podcast": "general_mental_health",
}


def _fallback_category(text: str) -> str | None:
    """Keyword-based fallback for channels not in the explicit mapping."""
    text_lower = text.lower()
    keywords = {
        "narcissistic_abuse_recovery": ["narcissist", "gaslight", "scapegoat", "abuser", "toxic"],
        "personality_disorders": ["borderline", "bpd", "personality disorder", "npd"],
        "complex_ptsd": ["trauma", "cptsd", "ptsd", "childhood", "abuse survivor", "dissociation"],
        "somatic_therapy": ["somatic", "body", "nervous system", "tension", "physical"],
        "attachment_disorders": ["attachment", "abandonment", "relationship", "couples", "intimacy"],
        "mindfulness_meditation": ["meditation", "mindfulness", "present moment", "breath"],
        "addiction_recovery": ["addiction", "substance", "sober", "alcohol", "recovery"],
        "eating_disorders": ["eating disorder", "anorexia", "bulimia", "binge", "body image"],
        "ocd_intrusive_thoughts": ["ocd", "intrusive thought", "compulsion", "checking"],
        "neurodivergent_mental_health": ["adhd", "autism", "neurodivergent", "masking", "sensory"],
        "cultural_religious_contexts": ["religious trauma", "spiritual abuse", "faith crisis"],
    }
    scores: dict[str, int] = {}
    for category, words in keywords.items():
        scores[category] = sum(1 for w in words if w in text_lower)
    if scores:
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best
    return None


def _assign_category(sample: dict[str, Any]) -> str:
    """Assign a therapeutic category to a sample."""
    channel = sample.get("source_channel", "")
    if channel in CHANNEL_CATEGORIES:
        return CHANNEL_CATEGORIES[channel]
    combined = sample.get("instruction", "") + " " + sample.get("output", "")
    fallback = _fallback_category(combined)
    if fallback:
        return fallback
    return "general_mental_health"


def run_categorization(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_counts: Counter[str] = Counter()
    total = 0

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        if jsonl_file.name in ("manifest.json", "processing_report.json"):
            continue

        out_file = output_dir / jsonl_file.name
        channel_count = 0

        with open(jsonl_file, encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                category = _assign_category(sample)
                sample["category"] = category
                category_counts[category] += 1
                channel_count += 1
                total += 1
                fout.write(json.dumps(sample, sort_keys=True) + "\n")

        logger.info("Categorized %s: %d samples", jsonl_file.stem, channel_count)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_samples": total,
        "category_distribution": dict(category_counts.most_common()),
        "channel_map": dict(CHANNEL_CATEGORIES.items()),
    }
    report_path = output_dir / "categorization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info("Categorization complete: %d samples across %d categories", total, len(category_counts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-categorize YouTube training samples.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with per-channel JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for categorized JSONL.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_categorization(args)


if __name__ == "__main__":
    main()
