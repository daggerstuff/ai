"""
Therapist voice extraction I/O and report formatting.

Profile rendering (communication patterns, tone characteristics),
channel output saving, and quality report generation.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from extraction_config import OUTPUT_BASE, get_config
from extraction_models import ChannelResult

logger = logging.getLogger("extract_therapist_voice")


# ═══════════════════════════════════════════════════════════════════════════════
#  Profile Rendering Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_communication_patterns(profile_data: dict, config: dict | None) -> list[str]:
    """Derive human-readable communication patterns from extracted profile data."""
    config = config or {}
    patterns = []

    transitions = profile_data.get("transition_phrases", {})
    total_trans = sum(transitions.values())
    if total_trans > 0:
        explanatory = sum(
            transitions.get(t, 0)
            for t in [
                "So",
                "Let me",
                "Here's the thing",
                "The key is",
                "What I want you to",
                "One of the key",
                "It's important to",
                "Think about",
            ]
        )
        analytical = sum(transitions.get(t, 0) for t in ["But", "What happens", "The reality is", "The truth is"])
        analytic_explain_ratio = analytical / max(explanatory, 1)

        if analytic_explain_ratio > 0.6:
            patterns.append(
                "Analytical, pattern-seeking communication that examines causes, "
                "contrasts scenarios, and surfaces underlying dynamics"
            )
        elif analytic_explain_ratio < 0.3:
            patterns.append(
                "Educational, explanatory style with clear framing, step-by-step "
                "guidance, and structured information delivery"
            )
        else:
            patterns.append(
                "Balanced explanatory-analytical style that alternates between teaching concepts and examining patterns"
            )

    empathy_markers = profile_data.get("empathy_markers", {})
    total_emp = sum(empathy_markers.values())
    if total_emp > 10:
        patterns.append(
            "High empathy integration with frequent validation, normalization "
            "of experiences, and explicit acknowledgment of emotional states"
        )
    elif total_emp > 4:
        patterns.append(
            "Moderate empathy woven into content delivery, balancing clinical "
            "information with attunement to felt experience"
        )
    else:
        patterns.append(
            "Empathy expressed primarily through information-sharing and "
            "validating insights rather than explicit emotional mirroring"
        )

    analogies = profile_data.get("analogies", [])
    if len(analogies) > 3:
        patterns.append(
            "Heavy reliance on analogies and metaphors to translate complex concepts into accessible, relatable images"
        )

    examples = profile_data.get("examples", [])
    if len(examples) > 3:
        patterns.append(
            "Frequent use of concrete examples and case illustrations to ground abstract principles in lived experience"
        )

    patterns.append(f"Style: {config.get('style', 'general').replace('_', ' ')}")
    patterns.append(f"Approach: {config.get('approach', 'general').replace('_', ' ')}")

    return patterns


def _derive_tone_characteristics(profile_data: dict, config: dict | None) -> dict:
    """Derive tone characteristics from extracted profile data and channel config."""
    config = config or {}
    empathy_markers = profile_data.get("empathy_markers", {})
    total_emp = sum(empathy_markers.values())
    transitions = profile_data.get("transition_phrases", {})
    total_trans = sum(transitions.values())
    style = config.get("style", "")

    if total_emp > 10:
        empathy_level = "high"
    elif total_emp > 4:
        empathy_level = "moderate"
    else:
        empathy_level = "measured"

    if "clinical" in style or "authoritative" in style or "professional" in style:
        formality = "professional_structured"
    elif "conversational" in style or "casual" in style or "interview" in style:
        formality = "conversational_accessible"
    else:
        formality = "professional_yet_accessible"

    if total_trans > 0:
        fast_pace_markers = sum(transitions.get(t, 0) for t in ["Now", "So"])
        slow_pace_markers = sum(transitions.get(t, 0) for t in ["Let me", "Think about"])
        if fast_pace_markers > slow_pace_markers * 2:
            pacing = "brisk_momentum"
        elif slow_pace_markers > fast_pace_markers * 2:
            pacing = "deliberate_paced"
        else:
            pacing = "measured_rhythm"
    else:
        pacing = "measured_rhythm"

    if "compassionate" in style or "warm" in style or "empathetic" in style:
        emotional_temperature = "warm_supportive"
    elif "authoritative" in style or "direct" in style or "clinical" in style:
        emotional_temperature = "authoritative_grounded"
    elif "contemplative" in style or "mystical" in style or "wisdom" in style:
        emotional_temperature = "calm_contemplative"
    else:
        emotional_temperature = "balanced_engaged"

    return {
        "empathy_level": empathy_level,
        "formality": formality,
        "pacing": pacing,
        "emotional_temperature": emotional_temperature,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Output Saving
# ═══════════════════════════════════════════════════════════════════════════════


def save_channel_output(channel_key: str, result: ChannelResult, output_base: Path = OUTPUT_BASE):
    """Save voice profile, conversations, and quality report for a channel."""
    config = get_config(channel_key)
    sig = config["signature"] if config else channel_key.lower()
    channel_dir = output_base / f"{sig}_voice"
    exports_dir = channel_dir / "exports"
    channel_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    if result.voice_profile:
        profile_data = result.voice_profile.get("profile_raw", {})
        profile_out = {
            "name": config["name"] if config else channel_key,
            "voice_signature": f"{sig}_v1",
            "description": config["description"] if config else "",
            "personality_traits": {
                "primary_style": config["style"] if config else "general",
                "communication_patterns": _derive_communication_patterns(profile_data, config),
                "expertise_areas": config["expertise"] if config else [],
                "tone_characteristics": _derive_tone_characteristics(profile_data, config),
            },
            "training_samples": len(result.transcripts),
            "clinical_validity_score": round(result.mean_score, 4),
            "clinical_validity_pass_rate": round(result.pass_rate, 4),
            "created_at": datetime.now(UTC).isoformat(),
            "version": "2.0",
        }
        if "sentence_starters" in profile_data:
            profile_out["sentence_starters"] = profile_data["sentence_starters"]
        if "transition_phrases" in profile_data:
            profile_out["transition_phrases"] = profile_data["transition_phrases"]
        if "empathy_markers" in profile_data:
            profile_out["empathy_markers"] = profile_data["empathy_markers"]

        profile_file = channel_dir / f"{channel_key.lower()}_voice_profile.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile_out, f, indent=2, ensure_ascii=False)
        logger.info("  Saved profile: %s", profile_file)

        report_content = result.voice_profile.get("report", "")
        if report_content:
            report_file = channel_dir / f"{channel_key.lower()}_voice_analysis.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info("  Saved analysis: %s", report_file)

    if result.conversations:
        conv_file = exports_dir / f"{channel_key.lower()}_conversations.jsonl"
        with open(conv_file, "w", encoding="utf-8") as f:
            for conv in result.conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        logger.info("  Saved %d conversations: %s", len(result.conversations), conv_file)

    if result.scores:
        scores_file = channel_dir / f"{channel_key.lower()}_clinical_scores.csv"
        with open(scores_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conversation_id", "score", "technique", "alliance", "structure", "cultural", "ebp"])
            for conv, score, detail in zip(result.conversations, result.scores, result.score_detail, strict=False):
                cid = conv.get("conversation_id", "")
                writer.writerow(
                    [
                        cid,
                        round(score, 4),
                        round(detail.get("technique", 0), 4),
                        round(detail.get("alliance", 0), 4),
                        round(detail.get("structure", 0), 4),
                        round(detail.get("cultural", 0), 4),
                        round(detail.get("ebp", 0), 4),
                    ]
                )
        logger.info("  Saved scores: %s", scores_file)


def generate_quality_report(results: list[ChannelResult]) -> str:
    """Generate markdown quality report across all processed channels."""
    report: list[str] = [
        "# Clinical Validity Quality Report\n",
        f"**Generated**: {datetime.now(UTC).isoformat()}\n\n",
        "## Per-Channel Quality Summary\n\n",
        "| Channel | Transcripts | Conversations | Mean Score | Pass Rate (≥0.5) | High Quality (≥0.7) |\n",
        "|---------|------------:|--------------:|----------:|-----------------:|--------------------:|\n",
    ]

    all_scores: list[float] = []
    for r in results:
        all_scores.extend(r.scores)
        report.append(
            f"| {r.name} | {len(r.transcripts)} | {len(r.conversations)} | "
            f"{r.mean_score:.4f} | {r.pass_rate:.1%} | {r.high_quality_rate:.1%} |\n"
        )

    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        overall_pass = sum(1 for s in all_scores if s >= 0.5) / len(all_scores)
        overall_high = sum(1 for s in all_scores if s >= 0.7) / len(all_scores)
        report.append(
            f"| **Overall** | | {len(all_scores)} | "
            f"**{overall_mean:.4f}** | **{overall_pass:.1%}** | **{overall_high:.1%}** |\n"
        )

    report.extend(
        [
            "\n## Dimension Averages\n\n",
            "| Channel | Technique | Alliance | Structure | Cultural | EBP |\n",
            "|---------|----------:|---------:|----------:|---------:|----:|\n",
        ]
    )

    for r in results:
        if r.score_detail:
            tech = sum(d.get("technique", 0) for d in r.score_detail) / len(r.score_detail)
            alli = sum(d.get("alliance", 0) for d in r.score_detail) / len(r.score_detail)
            stru = sum(d.get("structure", 0) for d in r.score_detail) / len(r.score_detail)
            cult = sum(d.get("cultural", 0) for d in r.score_detail) / len(r.score_detail)
            ebp = sum(d.get("ebp", 0) for d in r.score_detail) / len(r.score_detail)
            report.append(f"| {r.name} | {tech:.4f} | {alli:.4f} | {stru:.4f} | {cult:.4f} | {ebp:.4f} |\n")

    report.append("\n## Validation Results\n\n")
    total_issues = 0
    for r in results:
        if r.validation_report:
            issues = r.validation_report.get("issues", [])
            if issues:
                total_issues += len(issues)
                report.append(f"- **{r.name}**: {'; '.join(issues)}\n")
    if total_issues == 0:
        report.append("No validation issues detected across all channels.\n")

    return "".join(report)
