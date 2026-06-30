"""Validation benchmark for ClinicalValidityScorer.

Loads all 13 CSV fixture files (691 samples), computes scorer vs human correlation
(Pearson/Spearman >= 0.5), MAE < 0.15, per-dimension agreement >= 60%, and generates
a structured JSON report.

Usage:
    uv run python -m training.benchmark
    uv run python -m training.benchmark --output report.json
    uv run pytest training/tests/test_benchmark.py -v
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from training.clinical_validity_scorer import ClinicalValidityScorer

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "clinical_scores"
TRANSCRIPT_BASE = Path(__file__).parent.parent / "data" / "transcripts" / "transcripts"

CSV_FILES = [
    "couplestherapyofficial_clinical_scores.csv",
    "crappychildhoodfairy_clinical_scores.csv",
    "doctorramani_clinical_scores.csv",
    "dr.danielfox_clinical_scores.csv",
    "medcircle_clinical_scores.csv",
    "navigatingnarcissism_clinical_scores.csv",
    "patrickteahan_clinical_scores.csv",
    "psych2go_clinical_scores.csv",
    "therapychatpodcast_clinical_scores.csv",
    "therapydecoded_clinical_scores.csv",
    "therapyinanutshell_clinical_scores.csv",
    "timfletcher_clinical_scores.csv",
    "wuweiwisdom_clinical_scores.csv",
]

# Human-scored dimensions (dsm5 is not in the CSV, only 5 dimensions)
HUMAN_DIMENSIONS = ["technique", "alliance", "structure", "cultural", "ebp"]

# Agreement tolerance for per-dimension agreement (within ±0.2 = agree)
AGREEMENT_TOLERANCE = 0.2

# Channel name to directory mapping for transcript lookup
# The key is the prefix used in conversation_id, the value is the actual directory name
CHANNEL_PREFIX_MAP = {
    "couples_therapy_official": "Couples Therapy Official",
    "crappy_childhood_fairy": "Crappy Childhood Fairy",
    "doctor_ramani": "DoctorRamani",
    "dr_daniel_fox": "Dr. Daniel Fox",
    "medcircle": "MedCircle",
    "navigating_narcissism": "Navigating Narcissism",
    "patrick_teahan": "Patrick Teahan ",  # Note: trailing space in dir name
    "psych2go": "Psych2Go",
    "therapy_chat_podcast": "Therapy Chat Podcast",
    "therapy_decoded": "Therapy Decoded",
    "therapy_in_a_nutshell": "Therapy in a Nutshell",
    "tim_fletcher": "Tim Fletcher",
    "wu_wei_wisdom": "Wu Wei Wisdom",
}


def _compute_pearson(human_scores: list[float], scorer_scores: list[float]) -> float:
    min_samples = 2
    if len(human_scores) < min_samples:
        return 0.0
    # Check for constant arrays
    if len(set(human_scores)) <= 1 or len(set(scorer_scores)) <= 1:
        return 0.0
    result = scipy_stats.pearsonr(human_scores, scorer_scores)
    return float(result.statistic)


def _compute_spearman(human_scores: list[float], scorer_scores: list[float]) -> float:
    min_samples = 2
    if len(human_scores) < min_samples:
        return 0.0
    # Check for constant arrays
    if len(set(human_scores)) <= 1 or len(set(scorer_scores)) <= 1:
        return 0.0
    result = scipy_stats.spearmanr(human_scores, scorer_scores)
    return float(result.statistic)


def _compute_mae(human_scores: list[float], scorer_scores: list[float]) -> float:
    if not human_scores:
        return 0.0
    return sum(abs(h - s) for h, s in zip(human_scores, scorer_scores, strict=False)) / len(human_scores)


def _compute_agreement(
    human_scores: list[float], scorer_scores: list[float], tolerance: float = AGREEMENT_TOLERANCE
) -> float:
    if not human_scores:
        return 0.0
    matches = sum(1 for h, s in zip(human_scores, scorer_scores, strict=False) if abs(h - s) <= tolerance)
    return matches / len(human_scores)


def _md5_of_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _find_longest_prefix(conv_id: str) -> str | None:
    """Find the longest matching channel prefix for a conversation_id.

    The conversation_id format is: PREFIX_channelFullName_title
    where PREFIX is like 'couples_therapy_official', 'therapy_in_a_nutshell', etc.
    """
    conv_id_lower = conv_id.lower()

    # Try longer prefixes first
    # Sort by length descending
    sorted_prefixes = sorted(CHANNEL_PREFIX_MAP.keys(), key=len, reverse=True)

    for prefix in sorted_prefixes:
        if conv_id_lower.startswith(prefix.lower() + "_"):
            return prefix
        # Also handle the case where prefix itself is the start (without underscore after)
        if conv_id_lower.startswith(prefix.lower()):
            # Make sure it's a proper boundary (followed by _ or end)
            prefix_len = len(prefix)
            if prefix_len == len(conv_id) or conv_id[prefix_len:prefix_len+1] == "_":
                return prefix

    return None


def _normalize_conversation_id(conv_id: str) -> tuple[str | None, str]:
    """Extract channel directory and title from conversation_id.

    Format: PREFIX_channelFullName_title (e.g., couples_therapy_official_CouplesTherapyOfficial_The_Evolution...)

    Returns (channel_directory, title) where channel_directory is the transcript directory name,
    or (None, "") if no matching prefix found.
    """
    prefix = _find_longest_prefix(conv_id)

    if prefix is None:
        return None, ""

    # Title is everything after the prefix
    prefix_len = len(prefix)
    title_start = prefix_len + 1 if conv_id[prefix_len:prefix_len + 1] == "_" else prefix_len

    title = conv_id[title_start:]

    # Get the directory name from the prefix
    channel_dir = CHANNEL_PREFIX_MAP.get(prefix, prefix)

    return channel_dir, title


def _find_transcript_file(channel_dir: str, conv_id: str) -> Path | None:
    """Find the transcript file matching the conversation_id.

    The conversation_id format is: PREFIX_channelDisplayName_title
    where channelDisplayName is like "CouplesTherapyOfficial" but the actual
    transcript file is just "The Evolution..." without the channel display name.

    We extract the title portion (after channel display name) and match it.

    Args:
        channel_dir: Directory name under TRANSCRIPT_BASE
        conv_id: Full conversation_id string

    Returns:
        Path to transcript file if found, None otherwise.
    """
    channel_path = TRANSCRIPT_BASE / channel_dir
    if not channel_path.exists():
        return None

    # Extract just the title portion from the conv_id
    # Format: PREFIX_channelDisplayName_title
    prefix = _find_longest_prefix(conv_id)
    if prefix is None:
        return None

    # Get the part after the prefix
    prefix_len = len(prefix)
    remainder = conv_id[prefix_len:]  # e.g., "_CouplesTherapyOfficial_The_Evolution..."
    if remainder.startswith("_"):
        remainder = remainder[1:]

    # Now remainder is "CouplesTherapyOfficial_The_Evolution..."
    # The first "word" (CamelCase or all caps) is the channel display name
    # We need the rest as the title

    # Split on underscores to find where title starts
    parts = remainder.split("_")
    if len(parts) <= 1:
        return None

    # First part is channel display name (e.g., "CouplesTherapyOfficial")
    # Rest is the title
    title_parts = parts[1:]  # Skip channel display name
    title_candidate = "_".join(title_parts)
    title_normalized = title_candidate.replace("_", " ").replace("-", " ").lower()

    # List transcript files and find best match
    best_match: Path | None = None
    best_score = 0

    stop_words = {
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "with",
        "on", "at", "by", "your", "you", "it", "is", "are", "was", "were",
        "be", "been", "being",
    }

    for f in channel_path.iterdir():
        if not f.is_file() or f.suffix != ".txt":
            continue

        fname_normalized = f.stem.replace("_", " ").replace("-", " ").lower()
        fname_words = set(fname_normalized.split()) - stop_words
        title_words = set(title_normalized.split()) - stop_words

        if not title_words:
            continue

        overlap = len(fname_words & title_words)
        score = overlap / max(len(title_words), 1)

        if score > best_score:
            best_score = score
            best_match = f

    # Accept match if score is reasonable
    min_transcript_match_score = 0.3
    if best_score >= min_transcript_match_score:
        return best_match

    return None


def load_all_csvs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load all 13 CSV fixture files.

    Returns:
        Tuple of (list of row dicts, dict mapping filename -> md5 for determinism check)
    """
    rows: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    missing: list[str] = []

    for name in CSV_FILES:
        path = FIXTURE_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        checksums[name] = _md5_of_file(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize fieldnames (strip whitespace, handle CRLF)
                normalized = {k.strip(): v.strip() if v is not None else "" for k, v in row.items()}
                rows.append(normalized)

    if missing:
        raise FileNotFoundError(f"Missing CSV fixture files: {missing}")
    return rows, checksums


def validate_csv_structure(rows: list[dict[str, Any]]) -> list[str]:
    """Validate that all rows have the expected CSV structure.

    Returns list of error messages (empty if valid).
    """
    errors: list[str] = []
    # Only validate numeric score fields (not conversation_id which is a string)
    numeric_fields = ["score", *HUMAN_DIMENSIONS]
    for i, row in enumerate(rows):
        if "conversation_id" not in row:
            errors.append(f"Row {i}: missing required field 'conversation_id'")
        for field in numeric_fields:
            if field not in row:
                errors.append(f"Row {i}: missing required field '{field}'")
                continue
            try:
                val = float(row[field])
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Row {i}: field '{field}' value {val} out of [0,1] range")
            except ValueError:
                errors.append(f"Row {i}: field '{field}' is not a valid float: {row[field]!r}")
    return errors


def score_row(row: dict[str, Any]) -> dict[str, float] | None:
    """Score a single row by finding its transcript and running the scorer.

    Args:
        row: CSV row with conversation_id

    Returns:
        Dict with scorer scores (overall and per-dimension), or None if transcript not found.
        The overall score is computed using only the 5 human dimensions (technique, alliance,
        structure, cultural, ebp) with equal weights (1/5 each) for fair comparison with human labels.
    """
    conv_id = row.get("conversation_id", "")
    channel_dir, _ = _normalize_conversation_id(conv_id)
    if channel_dir is None:
        return None
    transcript_path = _find_transcript_file(channel_dir, conv_id)

    if transcript_path is None:
        return None

    try:
        text = transcript_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not text:
        return None

    # Run scorer
    detail = ClinicalValidityScorer.score_detail(text)

    # Compute overall using only the 5 human dimensions with equal weights
    # This is for fair comparison with human overall scores which don't include dsm5
    human_dims = ["technique", "alliance", "structure", "cultural", "ebp"]
    overall = sum(detail.get(dim, 0.0) for dim in human_dims) / len(human_dims)

    return {
        "overall": overall,
        "technique": detail.get("technique", 0.0),
        "alliance": detail.get("alliance", 0.0),
        "structure": detail.get("structure", 0.0),
        "cultural": detail.get("cultural", 0.0),
        "ebp": detail.get("ebp", 0.0),
        "dsm5": detail.get("dsm5", 0.0),
    }


def build_report(
    rows: list[dict[str, Any]],
    checksums: dict[str, str],
    scorer_version: str,
) -> dict[str, Any]:
    """Compute all benchmark metrics and build the structured JSON report.

    Only samples with successfully found transcripts are included in the
    correlation and MAE calculations, as we cannot compute scorer scores
    for samples without their source text.

    Args:
        rows: List of CSV rows with human-labeled scores
        checksums: Dict of CSV filename -> md5 for deterministic report
        scorer_version: VERSION string of the scorer

    Returns:
        Structured report dict
    """
    # Only track scored samples (those with transcripts found)
    human_overall: list[float] = []
    scorer_overall: list[float] = []
    per_dimension: dict[str, dict[str, list[float]]] = {dim: {"human": [], "scorer": []} for dim in HUMAN_DIMENSIONS}
    per_channel: dict[str, dict[str, Any]] = {}
    missing_transcripts: list[str] = []
    synthetic_skipped: list[str] = []
    scored_count = 0

    for row in rows:
        conv_id = row.get("conversation_id", "")

        # Skip synthetic samples - they don't have transcripts
        if "synthetic" in conv_id.lower():
            synthetic_skipped.append(conv_id)
            continue

        # Extract human overall score
        try:
            human_score = float(row["score"])
        except (ValueError, KeyError):
            continue

        # Extract channel for per-channel breakdown
        channel_dir, _ = _normalize_conversation_id(conv_id)
        if channel_dir not in per_channel:
            per_channel[channel_dir] = {"human": [], "scorer": [], "dimensions": {dim: {"human": [], "scorer": []} for dim in HUMAN_DIMENSIONS}}

        # Score the row
        scorer_result = score_row(row)

        if scorer_result is None:
            missing_transcripts.append(conv_id)
            # Don't include unscored samples in metrics
            continue

        scored_count += 1
        human_overall.append(human_score)
        scorer_overall.append(scorer_result["overall"])
        per_channel[channel_dir]["human"].append(human_score)
        per_channel[channel_dir]["scorer"].append(scorer_result["overall"])

        # Per-dimension scores
        for dim in HUMAN_DIMENSIONS:
            try:
                human_dim = float(row[dim])
            except (ValueError, KeyError):
                human_dim = 0.0
            per_dimension[dim]["human"].append(human_dim)
            per_dimension[dim]["scorer"].append(scorer_result[dim])
            per_channel[channel_dir]["dimensions"][dim]["human"].append(human_dim)
            per_channel[channel_dir]["dimensions"][dim]["scorer"].append(scorer_result[dim])

    # Compute overall metrics (only on scored samples)
    overall_mae = _compute_mae(human_overall, scorer_overall)
    overall_pearson = _compute_pearson(human_overall, scorer_overall)
    overall_spearman = _compute_spearman(human_overall, scorer_overall)

    # Per-dimension metrics
    dimension_metrics: dict[str, dict[str, Any]] = {}
    for dim in HUMAN_DIMENSIONS:
        human_list = per_dimension[dim]["human"]
        scorer_list = per_dimension[dim]["scorer"]
        agreement = _compute_agreement(human_list, scorer_list)
        dimension_metrics[dim] = {
            "pearson": _compute_pearson(human_list, scorer_list),
            "spearman": _compute_spearman(human_list, scorer_list),
            "mae": _compute_mae(human_list, scorer_list),
            "agreement_rate": agreement,
        }

    # Per-channel metrics
    channel_metrics: dict[str, dict[str, Any]] = {}
    for channel, data in per_channel.items():
        human_list = data["human"]
        scorer_list = data["scorer"]
        if len(human_list) >= 2:
            channel_metrics[channel] = {
                "sample_count": len(human_list),
                "pearson": _compute_pearson(human_list, scorer_list),
                "spearman": _compute_spearman(human_list, scorer_list),
                "mae": _compute_mae(human_list, scorer_list),
                "agreement_rate": _compute_agreement(human_list, scorer_list),
            }
        elif len(human_list) == 1:
            # Single sample - cannot compute correlation
            channel_metrics[channel] = {
                "sample_count": 1,
                "pearson": 0.0,
                "spearman": 0.0,
                "mae": abs(human_list[0] - scorer_list[0]) if scorer_list else 1.0,
                "agreement_rate": 1.0 if human_list and scorer_list and abs(human_list[0] - scorer_list[0]) <= AGREEMENT_TOLERANCE else 0.0,
            }
        else:
            channel_metrics[channel] = {
                "sample_count": 0,
                "pearson": 0.0,
                "spearman": 0.0,
                "mae": 0.0,
                "agreement_rate": 0.0,
            }

    # Build report
    report: dict[str, Any] = {
        "scorer_version": scorer_version,
        "total_sample_count": len(rows),
        "scored_sample_count": scored_count,
        "missing_transcript_count": len(missing_transcripts),
        "csv_checksums": checksums,
        "overall": {
            "pearson_correlation": round(overall_pearson, 4),
            "spearman_correlation": round(overall_spearman, 4),
            "mae": round(overall_mae, 4),
        },
        "per_dimension": {dim: {k: round(v, 4) for k, v in metrics.items()} for dim, metrics in dimension_metrics.items()},
        "per_channel": channel_metrics,
    }

    return report


def run_benchmark(output_path: Path | None = None) -> dict[str, Any]:
    """Run the full benchmark and optionally save to a file.

    Args:
        output_path: Optional path to save the JSON report

    Returns:
        The benchmark report dict
    """
    # Load CSVs
    rows, checksums = load_all_csvs()

    # Validate structure
    errors = validate_csv_structure(rows)
    if errors:
        raise ValueError(f"CSV validation errors: {errors}")

    # Build report
    scorer_version = ClinicalValidityScorer.VERSION
    report = build_report(rows, checksums, scorer_version)

    # Output
    if output_path:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    """CLI entry point for the benchmark."""
    parser = argparse.ArgumentParser(description="Run clinical validity scorer benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--format", type=str, default="json", choices=["json"], help="Output format (default: json)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    try:
        report = run_benchmark(output_path)

        # If no output file specified, print JSON to stdout
        if output_path is None:
            print(json.dumps(report, indent=2))

        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
