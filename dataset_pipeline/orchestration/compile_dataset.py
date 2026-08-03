import json
import os
import random

from dataset_pipeline.extractors.book_extractor import BookExtractor
from dataset_pipeline.extractors.dataset_loader import DatasetLoader
from dataset_pipeline.extractors.s3_streamer import S3Streamer
from dataset_pipeline.extractors.voice_extractor import VoiceExtractor
from dataset_pipeline.processors.chatml_converter import ChatMLConverter
from dataset_pipeline.processors.quality_filter import QualityFilter


def load_ratios():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset_ratios.json")
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# P0-4 lineage stamps: provenance -> (source, topic_tags,
# therapeutic_modality, is_training_edge_case)
#
# classify_record() (pipelines/orchestration/stage_organizer.py) routes records
# into Stages 1-5 by reading the top-level "source" key and
# "metadata.topic_tags" / "metadata.therapeutic_modality". Without these stamps
# every record fell into Stage 1, and the 35% cap silently dropped ~500K
# records. These maps are pure path/source-family provenance — NO content
# keyword heuristics.
#
# therapeutic_modality is set ONLY where a source directory maps to exactly one
# modality from the stage_organizer _THERAPEUTIC_MODALITIES vocabulary
# (captain-approved curated minimal table): stage2_specialist_addiction -> cbt,
# stage2_specialist_personality -> dbt, tier3_edge_crisis -> trauma_informed.
# All other directories stay unset (no fabrication).
# ---------------------------------------------------------------------------

# file_key prefix -> lineage (longest prefix wins)
STAGE_LINEAGE_BY_PREFIX: dict[str, dict] = {
    # Stage 1 — foundation
    # Directory prefix so all tier1_priority files (merged, priority_2/3,
    # unified, etc.) get stamped — the original single-file key missed the rest.
    "archive/gdrive/tier1_priority/": {
        "source": "mental_health",
        "topic_tags": ["mental_health", "psychology", "support"],
    },
    "datasets/training_v3/stage1_foundation/": {
        "source": "foundation",
        "topic_tags": ["mental_health", "psychology", "support"],
    },
    "cot_reasoning/": {
        "source": "foundation",
        "topic_tags": ["general", "education", "reasoning"],
    },
    # Stage 2 — therapeutic expertise
    "datasets/training_v3/stage2_specialist_addiction/": {
        "source": "therapeutic_expertise",
        "topic_tags": ["therapy", "clinical", "intervention"],
        "therapeutic_modality": "cbt",
    },
    "datasets/training_v3/stage2_specialist_personality/": {
        "source": "therapeutic_expertise",
        "topic_tags": ["therapy", "clinical", "intervention"],
        "therapeutic_modality": "dbt",
    },
    # Stage 3 — edge cases / stress tests
    "datasets/consolidated/edge_cases/": {
        "source": "edge_cases",
        "topic_tags": ["edge_case", "stress_test", "boundary_test"],
    },
    # Stage 4 — voice / persona
    "archive/local_voice_import/": {
        "source": "pixel_voice",
        "topic_tags": ["voice", "persona", "character"],
    },
    "datasets/training_v3/stage4_voice_persona/": {
        "source": "voice_persona",
        "topic_tags": ["voice", "persona", "character", "role_play"],
    },
    # Stage 5 — safety / crisis intervention
    "archive/gdrive/tier3_edge_crisis/": {
        "source": "crisis_intervention",
        "topic_tags": ["crisis", "self_harm", "suicide", "safety"],
        "therapeutic_modality": "trauma_informed",
    },
}

# source_family -> lineage (used when the extractor does not set file_key)
STAGE_LINEAGE_BY_FAMILY: dict[str, dict] = {
    "psychology_knowledge": {
        "source": "psychology_knowledge",
        "topic_tags": ["psychology", "education"],
    },
    "voice_training": {
        "source": "pixel_voice",
        "topic_tags": ["voice", "persona", "character"],
    },
    "reasoning_enhancement": {
        "source": "foundation",
        "topic_tags": ["general", "education", "reasoning"],
    },
}

# These sources are deliberately difficult edge-case training signal and must
# skip QualityFilter deduplication (P0-1), mirroring the edge-case generator.
EDGE_CASE_SOURCES: frozenset[str] = frozenset({"edge_cases", "stress_test", "adversarial"})


def _lineage_for(metadata: dict) -> dict | None:
    """Resolve lineage stamps from record provenance (no content heuristics)."""
    family = metadata.get("source_family", "")
    if family in STAGE_LINEAGE_BY_FAMILY:
        return STAGE_LINEAGE_BY_FAMILY[family]

    file_key = metadata.get("file_key", "")
    for prefix, lineage in sorted(STAGE_LINEAGE_BY_PREFIX.items(), key=len, reverse=True):
        if file_key.startswith(prefix):
            return lineage
    return None


def _stamp_lineage(chatml: dict, metadata: dict) -> dict:
    """Attach source/topic_tags/therapeutic_modality lineage stamps to a ChatML record."""
    lineage = _lineage_for(metadata)
    if lineage is None:
        return chatml

    chatml["source"] = lineage["source"]
    metadata_stamped = chatml.setdefault("metadata", {})
    metadata_stamped["topic_tags"] = lineage["topic_tags"]
    modality = lineage.get("therapeutic_modality")
    if modality is not None:
        metadata_stamped["therapeutic_modality"] = modality
    if lineage["source"] in EDGE_CASE_SOURCES:
        chatml["is_training_edge_case"] = True
    return chatml


def main():
    streamer = S3Streamer()
    voice_ext = VoiceExtractor(streamer)
    book_ext = BookExtractor(streamer)
    data_ext = DatasetLoader(streamer)

    converter = ChatMLConverter()
    quality = QualityFilter()
    ratios = load_ratios()

    # Categorized buckets
    dataset_buckets = {
        "psychology_knowledge": [],
        "voice_training": [],
        "mental_health_conversations": [],
        "reasoning_enhancement": [],
        "personality_balancing": [],
        "edge_cases": [],
        "crisis_intervention": [],
    }

    for record in voice_ext.extract_all():
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["voice_training"].append(chatml)

    for record in book_ext.extract_all():
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["psychology_knowledge"].append(chatml)

    # Mental Health Conversations (tier1_priority dir + stage1 + stage2 dirs)
    # NOTE: list_files(prefix) concatenates the yielded basename onto the prefix,
    # so passing a full file path double-concatenates and yields broken keys
    # (0 records). Load by directory prefix instead.
    mh_prefixes = [
        "archive/gdrive/tier1_priority/",
        "datasets/training_v3/stage1_foundation/",
        "datasets/training_v3/stage2_specialist_addiction/",
        "datasets/training_v3/stage2_specialist_personality/",
    ]
    for f in mh_prefixes:
        for record in data_ext.load_jsonl(f, "mental_health_conversations", "mental_health"):
            chatml = converter.convert(record)
            _stamp_lineage(chatml, record.get("metadata", {}))
            if quality.passes_filter(chatml):
                dataset_buckets["mental_health_conversations"].append(chatml)

    # Reasoning Enhancement (cot_reasoning)
    for record in data_ext.load_jsonl("cot_reasoning/", "reasoning_enhancement", "cot"):
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["reasoning_enhancement"].append(chatml)

    # Personality Balancing (stage4_voice_persona from HF)
    for record in data_ext.load_jsonl(
        "datasets/training_v3/stage4_voice_persona/", "personality_balancing", "synthetic_persona"
    ):
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["personality_balancing"].append(chatml)

    # Stage 3 — Edge cases / stress tests
    # Real S3 layout: datasets/consolidated/edge_cases/edge_case_output/ holds
    # augmented_prompts_10k.json (~10K prompt records) + priority_edge_case_prompts.json.
    for record in data_ext.load_jsonl("datasets/consolidated/edge_cases/edge_case_output/", "edge_cases", "edge_case"):
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["edge_cases"].append(chatml)

    # Stage 5 — Crisis intervention / safety content
    for record in data_ext.load_jsonl("archive/gdrive/tier3_edge_crisis/", "crisis_intervention", "crisis"):
        chatml = converter.convert(record)
        _stamp_lineage(chatml, record.get("metadata", {}))
        if quality.passes_filter(chatml):
            dataset_buckets["crisis_intervention"].append(chatml)

    target_size = 100000
    final_dataset = []

    for category, ratio in ratios.items():
        bucket_data = dataset_buckets.get(category, [])
        target_count = int(target_size * ratio)

        if len(bucket_data) == 0:
            continue

        if len(bucket_data) > target_count:
            sampled = random.sample(bucket_data, target_count)
        else:
            # If we don't have enough, just take everything we have
            sampled = bucket_data

        final_dataset.extend(sampled)

    random.shuffle(final_dataset)

    output_key = "final_dataset/final_training_dataset.jsonl"

    # Write to S3
    streamer.write_jsonl(output_key, final_dataset)


if __name__ == "__main__":
    main()
