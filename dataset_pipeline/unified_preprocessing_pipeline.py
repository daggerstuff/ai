"""
Unified Preprocessing Pipeline for Pixelated Empathy AI Training

This module orchestrates the integration of all data sources into a unified training dataset:
- ULTIMATE_FINAL_DATASET.jsonl (2.6GB, 608,497 conversations)
- Psychology knowledge base (4,867 concepts)
- YouTube transcripts from expert creators
- Crisis intervention scenarios
- Therapeutic counseling conversations
- Medical consultation dialogues
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
import hashlib
from datetime import datetime, timezone
import re
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CRISIS_ESCALATION_LEVELS = {"high", "very_high"}

@dataclass
class DataSource:
    """Represents a data source with metadata"""
    name: str
    path: str
    format: str
    size_bytes: int
    record_count: Optional[int] = None
    quality_score: Optional[float] = None
    source_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: Optional[str] = None


@dataclass
class StagePolicy:
    """Defines guard-rail expectations per training stage."""
    name: str
    min_empathy: float
    min_safety: float
    allow_crisis_override: bool = False
    requires_voice_signature: bool = False
    dedup_priority: int = 1


def get_default_stage_policies() -> Dict[str, StagePolicy]:
    """Return baseline policy definitions for all stages."""
    return {
        "stage1_foundation": StagePolicy(
            name="stage1_foundation",
            min_empathy=0.55,
            min_safety=0.7,
            dedup_priority=1
        ),
        "stage2_therapeutic_expertise": StagePolicy(
            name="stage2_therapeutic_expertise",
            min_empathy=0.5,
            min_safety=0.68,
            dedup_priority=2
        ),
        "stage3_edge_stress_test": StagePolicy(
            name="stage3_edge_stress_test",
            min_empathy=0.35,
            min_safety=0.55,
            allow_crisis_override=True,
            dedup_priority=3
        ),
        "stage4_voice_persona": StagePolicy(
            name="stage4_voice_persona",
            min_empathy=0.6,
            min_safety=0.75,
            requires_voice_signature=True,
            dedup_priority=4
        )
    }


class StageCatalog:
    """Loads manifest metadata and infers stage assignments."""

    def __init__(self, manifest_path: Path = Path("ai/data/master_dataset_manifest.json")):
        self.manifest_path = manifest_path
        self.stage_map: Dict[str, str] = {}
        self.stage_priorities: Dict[str, int] = {
            "stage1_foundation": 1,
            "stage2_therapeutic_expertise": 2,
            "stage3_edge_stress_test": 3,
            "stage4_voice_persona": 4
        }
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return

        try:
            with open(self.manifest_path, "r") as manifest_file:
                data = json.load(manifest_file)
        except Exception as exc:
            logger.warning(f"Unable to load manifest for stage catalog: {exc}")
            return

        datasets = data.get("datasets", {})
        for section in datasets.values():
            if not isinstance(section, dict):
                continue
            for key, dataset in section.items():
                if not isinstance(dataset, dict):
                    continue
                stage = dataset.get("stage")
                if not stage:
                    continue
                self.stage_map[key.lower()] = stage
                for field_name in ("path", "gdrive_path"):
                    if raw_path := dataset.get(field_name):
                        stem = Path(str(raw_path)).stem.lower()
                        self.stage_map[stem] = stage

    def lookup(self, source_name: str, path: Optional[str], source_type: Optional[str]) -> str:
        """Return the best stage label for the provided source metadata."""
        candidates = [source_name or "", source_type or ""]
        if path:
            candidates.extend([path, Path(path).stem])

        for candidate in candidates:
            if not candidate:
                continue
            if stage := self.stage_map.get(candidate.lower()):
                return stage

        return self._infer_fallback(source_name or "", source_type or "")

    def _infer_fallback(self, source_name: str, source_type: str) -> str:
        """Heuristic stage inference when manifest metadata is missing."""
        text = f"{source_name} {source_type}".lower()
        if any(token in text for token in ["edge_case", "reddit", "suicide", "kaggle_tf", "nightmare"]):
            return "stage3_edge_stress_test"
        if any(token in text for token in ["voice", "tim_fletcher", "persona", "transcript"]):
            return "stage4_voice_persona"
        if any(token in text for token in ["cot", "reasoning", "memo", "knowledge"]):
            return "stage2_therapeutic_expertise"
        return "stage1_foundation"

    def get_priority(self, stage: str) -> int:
        return self.stage_priorities.get(stage, 1)

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    target_quality_threshold: float = 0.8
    deduplication_enabled: bool = True
    validation_enabled: bool = True
    safety_filtering_enabled: bool = True
    psychology_integration_enabled: bool = True
    youtube_rag_integration_enabled: bool = True
    crisis_scenario_weight: float = 1.5
    therapeutic_conversation_weight: float = 1.0
    knowledge_base_weight: float = 1.2
    stage_policy_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class UnifiedPreprocessingPipeline:
    """Main preprocessing pipeline orchestrator"""

    PROGRESS_LOG_INTERVAL = 10000

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.data_sources: List[DataSource] = []
        self.processed_records = 0
        self.quality_filtered_records = 0
        self.safety_filtered_records = 0
        self.final_dataset_path = None
        self.stage_catalog = StageCatalog()
        self.stage_policies = self._build_stage_policies()
        self._pii_patterns = [
            re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
            re.compile(r"\b\(?\d{3}\)?\s*\d{3}[-.\s]?\d{4}\b"),
            re.compile(r"\b\d{9}\b"),
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        ]

    def _is_crisis_override_active(self, metadata: Dict[str, Any], policy: StagePolicy) -> bool:
        crisis_flag = metadata.get('crisis_intensity')
        if not crisis_flag or not policy.allow_crisis_override:
            return False
        if isinstance(crisis_flag, str):
            crisis_flag = crisis_flag.lower()
        return crisis_flag in CRISIS_ESCALATION_LEVELS

    def _build_stage_policies(self) -> Dict[str, StagePolicy]:
        policies = get_default_stage_policies()
        for stage, overrides in self.config.stage_policy_overrides.items():
            base_policy = policies.get(stage)
            if base_policy:
                merged = {**base_policy.__dict__, **overrides}
            else:
                merged = {"name": stage, **overrides}
            policies[stage] = StagePolicy(**merged)
        return policies

    def get_stage_policy(self, stage: str) -> StagePolicy:
        return self.stage_policies.get(stage, self.stage_policies["stage1_foundation"])

    def register_data_source(self, source: DataSource):
        """Register a data source for processing"""
        if not source.stage:
            source.stage = self.stage_catalog.lookup(source.name, source.path, source.source_type)
        source.metadata.setdefault("stage", source.stage)
        self.data_sources.append(source)
        logger.info(f"Registered data source: {source.name} ({source.format})")

    def discover_data_sources(self):
        """Discover and register all available data sources"""
        # Main datasets
        datasets_dir = Path("ai/training_data_consolidated/final_datasets")
        if datasets_dir.exists():
            for file_path in datasets_dir.glob("*.*"):
                if file_path.suffix in ['.jsonl', '.json']:
                    size = file_path.stat().st_size
                    source = DataSource(
                        name=file_path.stem,
                        path=str(file_path),
                        format=file_path.suffix.lstrip('.'),
                        size_bytes=size,
                        source_type="training_dataset"
                    )
                    self.register_data_source(source)

        # Psychology knowledge base
        psych_dir = Path("ai/training_data_consolidated/psychology_knowledge")
        if psych_dir.exists():
            for file_path in psych_dir.glob("*.json"):
                size = file_path.stat().st_size
                source = DataSource(
                    name=f"psychology_{file_path.stem}",
                    path=str(file_path),
                    format="json",
                    size_bytes=size,
                    source_type="knowledge_base"
                )
                self.register_data_source(source)

        # YouTube transcripts
        transcripts_dir = Path("ai/training_data_consolidated/transcripts")
        if transcripts_dir.exists():
            transcript_count = len(list(transcripts_dir.glob("*.md")))
            if transcript_count > 0:
                source = DataSource(
                    name="youtube_transcripts",
                    path=str(transcripts_dir),
                    format="markdown",
                    size_bytes=sum(f.stat().st_size for f in transcripts_dir.glob("*.md")),
                    record_count=transcript_count,
                    source_type="youtube_transcripts"
                )
                self.register_data_source(source)

        # Conversations
        conversations_dir = Path("ai/training_data_consolidated/conversations")
        if conversations_dir.exists():
            for file_path in conversations_dir.glob("*.jsonl"):
                size = file_path.stat().st_size
                source = DataSource(
                    name=f"conversations_{file_path.stem}",
                    path=str(file_path),
                    format="jsonl",
                    size_bytes=size,
                    source_type="conversations"
                )
                self.register_data_source(source)

        logger.info(f"Discovered {len(self.data_sources)} data sources")

    def validate_data_source(self, source: DataSource) -> bool:
        """Validate a data source"""
        try:
            if not os.path.exists(source.path):
                logger.warning(f"Data source path does not exist: {source.path}")
                return False

            if source.format == "jsonl":
                # Check if it's a valid JSONL file
                with open(source.path, 'r') as f:
                    if line := f.readline():
                        json.loads(line)
            elif source.format == "json":
                # Check if it's a valid JSON file
                with open(source.path, 'r') as f:
                    json.load(f)

            return True
        except Exception as e:
            logger.error(f"Validation failed for {source.name}: {str(e)}")
            return False

    def process_dataset(self, source: DataSource) -> List[Dict[str, Any]]:
        """Process a single dataset"""
        logger.info(f"Processing dataset: {source.name}")

        try:
            if source.format == "jsonl":
                return self._process_jsonl(source)
            if source.format == "json":
                return self._process_json(source)
            logger.warning(f"Unsupported format '{source.format}' for {source.name}")
            return []
        except Exception as e:
            logger.error(f"Error processing {source.name}: {str(e)}")
            return []

    def _process_jsonl(self, source: DataSource) -> List[Dict[str, Any]]:
        """Process a JSONL format dataset"""
        records = []
        processed_count = 0

        with open(source.path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    if processed_record := self._process_single_record(record, source):
                        records.append(processed_record)
                        processed_count += 1
                        if processed_count % self.PROGRESS_LOG_INTERVAL == 0:
                            logger.info(f"Processed {processed_count} records from {source.name}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} of {source.name}: {str(e)}")

        logger.info(f"Completed processing {source.name}: {len(records)} valid records")
        return records

    def _process_json(self, source: DataSource) -> List[Dict[str, Any]]:
        """Process a JSON format dataset"""
        with open(source.path, 'r') as f:
            data = json.load(f)

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'conversations' in data:
            items = data['conversations']

        records = []
        for item in items:
            if processed_record := self._process_single_record(item, source):
                records.append(processed_record)

        logger.info(f"Completed processing {source.name}: {len(records)} valid records")
        return records

    def _process_single_record(self, record: Dict[str, Any], source: DataSource) -> Optional[Dict[str, Any]]:
        """Process a single record: enhance and validate"""
        enhanced = self.enhance_record(record, source)
        return enhanced if self.validate_record(enhanced) else None

    def enhance_record(self, record: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Enhance a record with metadata and source information"""
        # Add source tracking
        record['_source'] = source.name
        record['_source_type'] = source.source_type
        record['_processed_at'] = datetime.now(timezone.utc).isoformat()

        # Add quality scoring if not present
        if 'metadata' not in record:
            record['metadata'] = {}

        if 'quality_score' not in record.get('metadata', {}):
            record['metadata']['quality_score'] = self.estimate_quality_score(record)

        # Ensure stage metadata is populated
        self.resolve_stage_for_record(record, source)

        return record

    def resolve_stage_for_record(self, record: Dict[str, Any], source: DataSource) -> str:
        """Populate and return the stage associated with this record."""
        metadata = record.setdefault('metadata', {})
        stage = (
            metadata.get('stage') or
            source.metadata.get('stage') or
            source.stage or
            self.stage_catalog.lookup(source.name, source.path, source.source_type)
        )
        metadata['stage'] = stage
        return stage

    def estimate_quality_score(self, record: Dict[str, Any]) -> float:
        """Estimate quality score for a record"""
        score = 0.5  # Base score

        # Check for content length
        content_length = 0
        if 'text' in record:
            content_length = len(record['text'])
        elif 'messages' in record:
            for msg in record['messages']:
                if 'content' in msg:
                    content_length += len(msg['content'])

        if content_length > 100:
            score += 0.2
        if content_length > 500:
            score += 0.1

        # Check for proper structure
        if 'messages' in record and len(record['messages']) >= 2:
            score += 0.2

        # Check for metadata
        if 'metadata' in record:
            score += 0.1

        return min(score, 1.0)

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a single record"""
        if not record:
            return False

        metadata = record.get('metadata', {})
        stage = metadata.get('stage', 'stage1_foundation')
        policy = self.get_stage_policy(stage)

        # Basic validation
        if self.config.validation_enabled:
            # Check for required fields
            if 'messages' not in record and 'text' not in record:
                return False

            # Check content quality
            if 'text' in record:
                content_length = len(record['text'])
            else:
                content_length = sum(
                    len(msg['content']) for msg in record['messages'] if 'content' in msg
                )

            if content_length < 10:  # Minimum content length
                return False

        # Quality filtering
        if self.config.target_quality_threshold > 0:
            quality_score = record.get('metadata', {}).get('quality_score', 0.5)
            if quality_score < self.config.target_quality_threshold:
                return False

        empathy_score = metadata.get('empathy_score', 0.5)
        crisis_override_active = self._is_crisis_override_active(metadata, policy)
        if empathy_score < policy.min_empathy and not crisis_override_active:
            return False

        safety_score = metadata.get('safety_score', 0.7)
        if safety_score < policy.min_safety and not crisis_override_active:
            return False

        return bool(
            not policy.requires_voice_signature or metadata.get('voice_signature')
        )

    def deduplicate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate records"""
        if not self.config.deduplication_enabled:
            return records

        hash_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        duplicates_removed = 0
        replacements = 0

        for record in records:
            # Create content hash for deduplication
            content_to_hash = ""
            if 'text' in record:
                content_to_hash = record['text']
            elif 'messages' in record:
                content_to_hash = "".join([msg.get('content', '') for msg in record['messages']])

            if content_to_hash:
                content_hash = hashlib.md5(content_to_hash.encode()).hexdigest()
                stage = record.get('metadata', {}).get('stage', 'stage1_foundation')
                priority = self.stage_catalog.get_priority(stage)
                existing = hash_map.get(content_hash)

                if existing is None:
                    hash_map[content_hash] = {"record": record, "priority": priority}
                elif priority > existing["priority"]:
                    hash_map[content_hash] = {"record": record, "priority": priority}
                    replacements += 1
                else:
                    duplicates_removed += 1

        logger.info(
            "Removed %s duplicate records (%s higher-priority replacements)",
            duplicates_removed,
            replacements
        )
        return [entry["record"] for entry in hash_map.values()]

    def apply_safety_filtering(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply safety filtering to records"""
        if not self.config.safety_filtering_enabled:
            return records

        safe_records = []
        unsafe_filtered = 0

        for record in records:
            metadata = record.get('metadata', {})
            stage = metadata.get('stage', 'stage1_foundation')
            policy = self.get_stage_policy(stage)

            content = self._collect_record_content(record)
            safety_score = metadata.get('safety_score', 0.7)

            if self._contains_pii(content):
                unsafe_filtered += 1
                continue

            crisis_override_active = self._is_crisis_override_active(metadata, policy)

            if policy.allow_crisis_override:
                # Drop non-crisis records that fail safety thresholds even in lenient mode
                if safety_score < policy.min_safety and not crisis_override_active:
                    unsafe_filtered += 1
                    continue
            elif safety_score < policy.min_safety or self._contains_disallowed_keywords(content):
                unsafe_filtered += 1
                continue

            safe_records.append(record)

        self.safety_filtered_records += unsafe_filtered
        logger.info(f"Filtered {unsafe_filtered} unsafe records (stage-aware)")
        return safe_records

    def _collect_record_content(self, record: Dict[str, Any]) -> str:
        if 'text' in record:
            return str(record['text']).lower()
        if 'messages' in record:
            return " ".join([msg.get('content', '') for msg in record['messages']]).lower()
        return ""

    def _contains_disallowed_keywords(self, content: str) -> bool:
        unsafe_keywords = ['explicit', 'nsfw', 'inappropriate']
        return any(keyword in content for keyword in unsafe_keywords)

    def _contains_pii(self, content: str) -> bool:
        if not content:
            return False
        return any(pattern.search(content) for pattern in self._pii_patterns)

    def integrate_psychology_knowledge(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Integrate psychology knowledge base concepts into records"""
        if not self.config.psychology_integration_enabled:
            return records

        # Load psychology knowledge base
        psych_knowledge = {}
        psych_dir = Path("ai/training_data_consolidated/psychology_knowledge")
        if psych_dir.exists():
            for file_path in psych_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            psych_knowledge |= data
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'concept_id' in item:
                                    psych_knowledge[item['concept_id']] = item
                except Exception as e:
                    logger.warning(f"Failed to load psychology knowledge from {file_path}: {str(e)}")

        if not psych_knowledge:
            logger.warning("No psychology knowledge base found for integration")
            return records

        # Apply psychology concepts to records
        enhanced_records = []
        for record in records:
            # Add psychology metadata
            if 'metadata' not in record:
                record['metadata'] = {}

            record['metadata']['psychology_concepts'] = self.extract_psychology_concepts(record, psych_knowledge)
            enhanced_records.append(record)

        logger.info(f"Integrated psychology knowledge into {len(enhanced_records)} records")
        return enhanced_records

    def extract_psychology_concepts(self, record: Dict[str, Any], psych_knowledge: Dict[str, Any]) -> List[str]:
        """Extract relevant psychology concepts from a record"""
        concepts = []
        content = ""

        if 'text' in record:
            content = record['text'].lower()
        elif 'messages' in record:
            content = " ".join([msg.get('content', '').lower() for msg in record['messages']])

        # Simple keyword matching for psychology concepts
        for concept_id, concept_data in psych_knowledge.items():
            concept_terms = []
            if isinstance(concept_data, dict):
                concept_terms = [concept_data.get('category', ''), concept_data.get('content', '')]
            elif isinstance(concept_data, str):
                concept_terms = [concept_data]

            for term in concept_terms:
                if term and term.lower() in content:
                    concepts.append(concept_id)
                    break

        return list(set(concepts))[:10]  # Limit to top 10 concepts

    def execute_pipeline(self) -> str:
        """Execute the complete preprocessing pipeline"""
        logger.info("Starting unified preprocessing pipeline execution")

        # Discover data sources
        self.discover_data_sources()

        if not self.data_sources:
            raise ValueError("No data sources found for processing")

        # Process all data sources
        all_records = []
        for source in self.data_sources:
            if self.validate_data_source(source):
                records = self.process_dataset(source)
                all_records.extend(records)
                self.processed_records += len(records)
                logger.info(f"Added {len(records)} records from {source.name}")
            else:
                logger.warning(f"Skipping invalid data source: {source.name}")

        logger.info(f"Total records processed: {len(all_records)}")

        # Apply preprocessing steps
        if self.config.deduplication_enabled:
            all_records = self.deduplicate_records(all_records)

        if self.config.safety_filtering_enabled:
            all_records = self.apply_safety_filtering(all_records)

        if self.config.psychology_integration_enabled:
            all_records = self.integrate_psychology_knowledge(all_records)

        # Final validation
        final_records = [record for record in all_records if self.validate_record(record)]

        logger.info(f"Final dataset contains {len(final_records)} records")

        # Save final dataset
        output_dir = Path("ai/dataset_pipeline/final_output")
        output_dir.mkdir(exist_ok=True)

        final_dataset_path = output_dir / f"unified_training_dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(final_dataset_path, 'w') as f:
            for record in final_records:
                f.write(json.dumps(record) + '\n')

        self.final_dataset_path = str(final_dataset_path)

        # Generate summary report
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sources_processed": len(self.data_sources),
            "total_records_processed": self.processed_records,
            "final_record_count": len(final_records),
            "deduplication_enabled": self.config.deduplication_enabled,
            "safety_filtering_enabled": self.config.safety_filtering_enabled,
            "psychology_integration_enabled": self.config.psychology_integration_enabled,
            "final_dataset_path": self.final_dataset_path,
            "final_dataset_size_bytes": final_dataset_path.stat().st_size if final_dataset_path.exists() else 0
        }

        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Pipeline execution completed. Final dataset saved to: {self.final_dataset_path}")
        return self.final_dataset_path

# Convenience functions
def create_default_pipeline() -> UnifiedPreprocessingPipeline:
    """Create a pipeline with default configuration"""
    config = ProcessingConfig(
        target_quality_threshold=0.7,
        deduplication_enabled=True,
        validation_enabled=True,
        safety_filtering_enabled=True,
        psychology_integration_enabled=True,
        youtube_rag_integration_enabled=True
    )
    return UnifiedPreprocessingPipeline(config)

def run_pipeline() -> str:
    """Run the complete preprocessing pipeline"""
    pipeline = create_default_pipeline()
    return pipeline.execute_pipeline()

if __name__ == "__main__":
    # Example usage
    try:
        final_dataset_path = run_pipeline()
        print(f"Pipeline completed successfully. Dataset saved to: {final_dataset_path}")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise
