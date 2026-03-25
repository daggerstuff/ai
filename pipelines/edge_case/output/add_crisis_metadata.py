#!/usr/bin/env python3
"""
Add crisis_intensity and is_training_edge_case metadata to edge case samples.

Per MasterTrainingPlan.md:
- Edge cases must have is_training_edge_case: true to bypass safety filters
- crisis_intensity metadata required for Stage 3 samples
"""

import json
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "edge_cases_training_format.jsonl"
OUTPUT_FILE = Path(__file__).parent / "edge_cases_with_crisis_metadata.jsonl"

# Crisis intensity mapping by detected pattern
CRISIS_INTENSITY_MAP = {
    "suicidality": "extreme",
    "homicidal_ideation": "extreme",
    "child_abuse_reporting": "extreme",
    "psychotic_episodes": "very_high",
    "domestic_violence": "very_high",
    "sadistic_manipulation": "very_high",
    "trauma_flashbacks": "very_high",
}

DEFAULT_INTENSITY = "high"

def categorize_sample(sample: dict) -> str:
    """
    Detect category from sample content by analyzing user messages.
    Returns category string matching master plan edge case categories.
    """
    # Extract all user message content
    messages = sample.get("messages", [])
    content_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            content_text += " " + msg.get("content", "").lower()

    # Also check metadata for any hints
    metadata = sample.get("metadata", {})
    metadata_str = json.dumps(metadata).lower()
    full_text = content_text + " " + metadata_str

    # Suicide patterns
    if any(pattern in full_text for pattern in [
        "ending his life", "ending her life", "ending their life",
        "suicide", "suicidal", "kill myself", "kill himself", "kill herself",
        "end my life", "end his life", "end her life",
        "thoughts of ending", "expresses thoughts of ending"
    ]):
        return "suicidality"

    # Homicide patterns
    if any(pattern in full_text for pattern in [
        "violent fantasies", "violence toward", "harming others",
        "harming her coworkers", "harming his coworkers",
        "homicidal", "homicide", "kill them", "kill someone",
        "murder", "violent urges"
    ]):
        return "homicidal_ideation"

    # Child abuse patterns
    if any(pattern in full_text for pattern in [
        "child abus", "child molest", "child neglect",
        "abusing child", "abusing children",
        "child welfare", "child protective"
    ]):
        return "child_abuse_reporting"

    # Psychosis patterns
    if any(pattern in full_text for pattern in [
        "government is monitoring", "being monitored", "surveillance",
        "psychotic", "psychosis", "delusion", "delusional",
        "hallucinat", "hearing voices", "paranoid", "paranoia"
    ]):
        return "psychotic_episodes"

    # Domestic violence patterns
    if any(pattern in full_text for pattern in [
        "domestic violence", "partner abus", "spouse abus",
        "abusive partner", "abusive husband", "abusive wife",
        "partner hit", "partner violence", "intimate partner"
    ]):
        return "domestic_violence"

    # Sadistic manipulation patterns
    if any(pattern in full_text for pattern in [
        "sadistic", "sadism", "torture", "cruelty",
        "manipulation", "manipulative", "gaslight"
    ]):
        return "sadistic_manipulation"

    # Trauma/PTSD patterns
    if any(pattern in full_text for pattern in [
        "trauma", "ptsd", "flashback", "triggered",
        "abuse survivor", "survivor of abuse"
    ]):
        return "trauma_flashbacks"

    return "other_edge"

def main():
    print(f"Processing edge cases from {INPUT_FILE}")

    processed = 0
    by_intensity = {"extreme": 0, "very_high": 0, "high": 0}
    by_category = {}

    with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "w") as f_out:
        for line in f_in:
            sample = json.loads(line.strip())

            # Detect category
            category = categorize_sample(sample)
            by_category[category] = by_category.get(category, 0) + 1

            # Get crisis intensity
            intensity = CRISIS_INTENSITY_MAP.get(category, DEFAULT_INTENSITY)
            by_intensity[intensity] += 1

            # Ensure metadata exists
            if "metadata" not in sample:
                sample["metadata"] = {}

            # Add critical flags
            sample["metadata"]["is_training_edge_case"] = True
            sample["metadata"]["crisis_intensity"] = intensity
            sample["metadata"]["edge_case_category"] = category

            # Ensure phase is set
            sample["metadata"]["phase"] = "stage3_edge_stress_test"

            # Write updated sample
            json.dump(sample, f_out)
            f_out.write("\n")

            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed} samples...")

    print(f"\n✅ Complete: {processed} samples processed")
    print(f"\nBy crisis intensity:")
    for intensity, count in sorted(by_intensity.items()):
        print(f"  {intensity}: {count} ({count/processed*100:.1f}%)")

    print(f"\nBy detected category:")
    for category, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"\nTo replace original file:")
    print(f"  mv {OUTPUT_FILE} {INPUT_FILE}")

if __name__ == "__main__":
    main()
