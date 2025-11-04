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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class UnifiedPreprocessingPipeline:
    """Main preprocessing pipeline orchestrator"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.data_sources: List[DataSource] = []
        self.processed_records = 0
        self.quality_filtered_records = 0
        self.safety_filtered_records = 0
        self.final_dataset_path = None

    def register_data_source(self, source: DataSource):
        """Register a data source for processing"""
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
                    line = f.readline()
                    if line:
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

        records = []
        processed_count = 0

        try:
            if source.format == "jsonl":
                with open(source.path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            record = json.loads(line.strip())
                            record = self.enhance_record(record, source)
                            if self.validate_record(record):
                                records.append(record)
                                processed_count += 1

                                if processed_count % 10000 == 0:
                                    logger.info(f"Processed {processed_count} records from {source.name}")

                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} of {source.name}: {str(e)}")
                            continue

            elif source.format == "json":
                with open(source.path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            item = self.enhance_record(item, source)
                            if self.validate_record(item):
                                records.append(item)
                                processed_count += 1
                    elif isinstance(data, dict) and 'conversations' in data:
                        for item in data['conversations']:
                            item = self.enhance_record(item, source)
                            if self.validate_record(item):
                                records.append(item)
                                processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {source.name}: {str(e)}")

        logger.info(f"Completed processing {source.name}: {len(records)} valid records")
        return records

    def enhance_record(self, record: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Enhance a record with metadata and source information"""
        # Add source tracking
        record['_source'] = source.name
        record['_source_type'] = source.source_type
        record['_processed_at'] = datetime.utcnow().isoformat()

        # Add quality scoring if not present
        if 'metadata' not in record:
            record['metadata'] = {}

        if 'quality_score' not in record.get('metadata', {}):
            record['metadata']['quality_score'] = self.estimate_quality_score(record)

        return record

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

        # Basic validation
        if self.config.validation_enabled:
            # Check for required fields
            if 'messages' not in record and 'text' not in record:
                return False

            # Check content quality
            content_length = 0
            if 'text' in record:
                content_length = len(record['text'])
            elif 'messages' in record:
                for msg in record['messages']:
                    if 'content' in msg:
                        content_length += len(msg['content'])

            if content_length < 10:  # Minimum content length
                return False

        # Quality filtering
        if self.config.target_quality_threshold > 0:
            quality_score = record.get('metadata', {}).get('quality_score', 0.5)
            if quality_score < self.config.target_quality_threshold:
                return False

        return True

    def deduplicate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate records"""
        if not self.config.deduplication_enabled:
            return records

        seen_hashes = set()
        unique_records = []
        duplicates_removed = 0

        for record in records:
            # Create content hash for deduplication
            content_to_hash = ""
            if 'text' in record:
                content_to_hash = record['text']
            elif 'messages' in record:
                content_to_hash = "".join([msg.get('content', '') for msg in record['messages']])

            if content_to_hash:
                content_hash = hashlib.md5(content_to_hash.encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_records.append(record)
                else:
                    duplicates_removed += 1

        logger.info(f"Removed {duplicates_removed} duplicate records")
        return unique_records

    def apply_safety_filtering(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply safety filtering to records"""
        if not self.config.safety_filtering_enabled:
            return records

        safe_records = []
        unsafe_filtered = 0

        for record in records:
            # Basic safety checks
            is_safe = True

            # Check for explicit content markers
            unsafe_keywords = ['explicit', 'nsfw', 'inappropriate']
            content = ""
            if 'text' in record:
                content = record['text'].lower()
            elif 'messages' in record:
                content = " ".join([msg.get('content', '').lower() for msg in record['messages']])

            for keyword in unsafe_keywords:
                if keyword in content:
                    is_safe = False
                    break

            if is_safe:
                safe_records.append(record)
            else:
                unsafe_filtered += 1

        logger.info(f"Filtered {unsafe_filtered} unsafe records")
        return safe_records

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
                            psych_knowledge.update(data)
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
        final_records = []
        for record in all_records:
            if self.validate_record(record):
                final_records.append(record)

        logger.info(f"Final dataset contains {len(final_records)} records")

        # Save final dataset
        output_dir = Path("ai/dataset_pipeline/final_output")
        output_dir.mkdir(exist_ok=True)

        final_dataset_path = output_dir / f"unified_training_dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(final_dataset_path, 'w') as f:
            for record in final_records:
                f.write(json.dumps(record) + '\n')

        self.final_dataset_path = str(final_dataset_path)

        # Generate summary report
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
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