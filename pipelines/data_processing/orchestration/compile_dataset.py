import json
import os
import random

from pipelines.data_processing.extractors.book_extractor import BookExtractor
from pipelines.data_processing.extractors.dataset_loader import DatasetLoader
from pipelines.data_processing.extractors.s3_streamer import S3Streamer
from pipelines.data_processing.extractors.voice_extractor import VoiceExtractor
from pipelines.data_processing.processors.chatml_converter import ChatMLConverter
from pipelines.data_processing.processors.quality_filter import QualityFilter


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
    """Resolve lineage stamps from record provenance (no content heuristics).

    Order: longest file_key prefix match first (most specific), then
    source_family fallback. Checking family first would shadow more
    specific path stamps for records that carry both fields.
    """
    file_key = metadata.get("file_key", "")
    for prefix, lineage in sorted(
        STAGE_LINEAGE_BY_PREFIX.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if file_key.startswith(prefix):
            return lineage

    family = metadata.get("source_family", "")
    return STAGE_LINEAGE_BY_FAMILY.get(family)


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



def _convert_and_stamp(
    record: dict,
    converter: ChatMLConverter,
    quality: QualityFilter,
) -> dict | None:
    """Convert a raw record to ChatML, stamp lineage, and filter by quality.

    Returns the stamped chatml record if it passes quality filter, else None.
    """
    chatml = converter.convert(record)
    _stamp_lineage(chatml, record.get("metadata", {}))
    if quality.passes_filter(chatml):
        return chatml
    return None


def _sample_dataset(
    dataset_buckets: dict[str, list[dict]],
    ratios: dict[str, float],
    target_size: int,
) -> list[dict]:
    """Sample each bucket to its target ratio and return shuffled final dataset."""
    final_dataset: list[dict] = []
    for category, ratio in ratios.items():
        bucket_data = dataset_buckets.get(category, [])
        target_count = int(target_size * ratio)
        if not bucket_data:
            continue
        if len(bucket_data) > target_count:
            sampled = random.sample(bucket_data, target_count)
        else:
            sampled = bucket_data
        final_dataset.extend(sampled)
    random.shuffle(final_dataset)
    return final_dataset



def main():
    streamer = S3Streamer()
    voice_ext = VoiceExtractor(streamer)
    book_ext = BookExtractor(streamer)
    data_ext = DatasetLoader(streamer)

    converter = ChatMLConverter()
    quality = QualityFilter()
    ratios = load_ratios()

    dataset_buckets = _collect_buckets(voice_ext, book_ext, data_ext, converter, quality)
    final_dataset = _sample_dataset(dataset_buckets, ratios, target_size=100000)
    streamer.write_jsonl("final_dataset/final_training_dataset.jsonl", final_dataset)


def _collect_buckets(
    voice_ext: VoiceExtractor,
    book_ext: BookExtractor,
    data_ext: DatasetLoader,
    converter: ChatMLConverter,
    quality: QualityFilter,
) -> dict[str, list[dict]]:
    """Extract records from all sources into categorized buckets."""
    buckets: dict[str, list[dict]] = {
        "psychology_knowledge": [],
        "voice_training": [],
        "mental_health_conversations": [],
        "reasoning_enhancement": [],
        "personality_balancing": [],
        "edge_cases": [],
        "crisis_intervention": [],
    }

    for record in voice_ext.extract_all():
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["voice_training"].append(chatml)

    for record in book_ext.extract_all():
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["psychology_knowledge"].append(chatml)

    # Mental Health Conversations (tier1_priority dir + stage1 + stage2 dirs)
    mh_prefixes = [
        "archive/gdrive/tier1_priority/",
        "datasets/training_v3/stage1_foundation/",
        "datasets/training_v3/stage2_specialist_addiction/",
        "datasets/training_v3/stage2_specialist_personality/",
    ]
    for prefix in mh_prefixes:
        for record in data_ext.load_jsonl(prefix, "mental_health_conversations", "mental_health"):
            chatml = _convert_and_stamp(record, converter, quality)
            if chatml:
                buckets["mental_health_conversations"].append(chatml)

    for record in data_ext.load_jsonl("cot_reasoning/", "reasoning_enhancement", "cot"):
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["reasoning_enhancement"].append(chatml)

    for record in data_ext.load_jsonl(
        "datasets/training_v3/stage4_voice_persona/", "personality_balancing", "synthetic_persona"
    ):
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["personality_balancing"].append(chatml)

    for record in data_ext.load_jsonl(
        "datasets/consolidated/edge_cases/edge_case_output/", "edge_cases", "edge_case"
    ):
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["edge_cases"].append(chatml)

    for record in data_ext.load_jsonl(
        "archive/gdrive/tier3_edge_crisis/", "crisis_intervention", "crisis"
    ):
        chatml = _convert_and_stamp(record, converter, quality)
        if chatml:
            buckets["crisis_intervention"].append(chatml)

    return buckets


if __name__ == "__main__":
    main()
