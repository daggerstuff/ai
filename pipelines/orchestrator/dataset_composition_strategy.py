"""
Dataset Composition and Balancing Strategy for Pixelated Empathy AI Training

This module implements the balanced composition strategy for the unified training dataset:
- 30% Standard therapeutic conversations
- 25% Edge case scenarios (challenging situations)
- 20% Voice-derived dialogues (personality-consistent)
- 15% Psychology knowledge integration
- 10% Dual persona training examples
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging

from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetComposer:
    """Handles dataset composition and balancing"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.composition_ratios = {
            'standard_therapeutic': 0.30,
            'edge_case_scenarios': 0.25,
            'voice_derived_dialogues': 0.20,
            'psychology_knowledge': 0.15,
            'dual_persona_examples': 0.10
        }
        self.quality_threshold = self.config.get('quality_threshold', 0.7)

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file"""
        records = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(records)} records from {dataset_path}")
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {str(e)}")
        return records

    def filter_by_quality(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter records by quality threshold"""
        filtered_records = []
        low_quality_count = 0

        for record in records:
            quality_score = record.get('metadata', {}).get('quality_score', 0.5)
            if quality_score >= self.quality_threshold:
                filtered_records.append(record)
            else:
                low_quality_count += 1

        logger.info(f"Quality filtering removed {low_quality_count} records (threshold: {self.quality_threshold})")
        return filtered_records

    def deduplicate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate records based on content hash"""
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

        logger.info(f"Deduplication removed {duplicates_removed} duplicate records")
        return unique_records

    def categorize_records(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize records by source type for composition strategy"""
        categorized = defaultdict(list)

        for record in records:
            source_type = record.get('_source_type', 'unknown')

            # Map source types to composition categories
            if source_type == 'training_dataset':
                # Further categorize based on metadata
                if 'edge_case' in record.get('_source', '').lower():
                    categorized['edge_case_scenarios'].append(record)
                elif 'voice' in record.get('_source', '').lower():
                    categorized['voice_derived_dialogues'].append(record)
                else:
                    categorized['standard_therapeutic'].append(record)
            elif source_type == 'knowledge_base':
                categorized['psychology_knowledge'].append(record)
            elif source_type == 'youtube_transcripts':
                categorized['voice_derived_dialogues'].append(record)
            elif source_type == 'conversations':
                # Check for dual persona indicators
                if record.get('metadata', {}).get('dual_persona', False):
                    categorized['dual_persona_examples'].append(record)
                else:
                    categorized['standard_therapeutic'].append(record)
            else:
                categorized['standard_therapeutic'].append(record)

        return dict(categorized)

    def balance_dataset(self, categorized_records: Dict[str, List[Dict[str, Any]]],
                       target_size: int) -> List[Dict[str, Any]]:
        """Balance dataset according to composition ratios"""
        balanced_records = []

        # Calculate target counts for each category
        target_counts = {}
        for category, ratio in self.composition_ratios.items():
            target_counts[category] = int(target_size * ratio)

        logger.info(f"Target composition: {target_counts}")

        # Sample from each category
        for category, target_count in target_counts.items():
            if category in categorized_records:
                available_records = categorized_records[category]
                if len(available_records) >= target_count:
                    # Randomly sample the required number
                    sampled_records = random.sample(available_records, target_count)
                    balanced_records.extend(sampled_records)
                    logger.info(f"Sampled {target_count} records from {category}")
                else:
                    # Use all available records
                    balanced_records.extend(available_records)
                    logger.warning(f"Only {len(available_records)} available for {category}, using all")
            else:
                logger.warning(f"No records found for category {category}")

        logger.info(f"Balanced dataset contains {len(balanced_records)} records")
        return balanced_records

    def generate_composition_report(self, original_records: List[Dict[str, Any]],
                                  balanced_records: List[Dict[str, Any]],
                                  categorized_records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate detailed composition report"""
        report = {
            'original_dataset_stats': {
                'total_records': len(original_records),
                'quality_filtered': len(original_records) - len(self.filter_by_quality(original_records)),
                'deduplicated': len(self.filter_by_quality(original_records)) - len(self.deduplicate_records(self.filter_by_quality(original_records)))
            },
            'composition_breakdown': {},
            'final_dataset_stats': {
                'total_records': len(balanced_records),
                'quality_scores': {
                    'min': min([r.get('metadata', {}).get('quality_score', 0) for r in balanced_records]),
                    'max': max([r.get('metadata', {}).get('quality_score', 0) for r in balanced_records]),
                    'avg': sum([r.get('metadata', {}).get('quality_score', 0) for r in balanced_records]) / len(balanced_records)
                }
            }
        }

        # Add composition breakdown
        total_balanced = len(balanced_records)
        for category, records in categorized_records.items():
            actual_count = len([r for r in balanced_records if r.get('_source_type_category') == category])
            target_ratio = self.composition_ratios.get(category, 0)
            actual_ratio = actual_count / total_balanced if total_balanced > 0 else 0

            report['composition_breakdown'][category] = {
                'available_records': len(records),
                'target_ratio': target_ratio,
                'actual_count': actual_count,
                'actual_ratio': actual_ratio,
                'difference': abs(target_ratio - actual_ratio)
            }

        return report

    def execute_composition_strategy(self, input_dataset_path: str,
                                   output_dataset_path: str = None) -> Tuple[str, Dict[str, Any]]:
        """Execute the complete dataset composition strategy"""
        logger.info("Starting dataset composition strategy execution")

        # Load dataset
        records = self.load_dataset(input_dataset_path)

        # Apply quality filtering
        filtered_records = self.filter_by_quality(records)

        # Apply deduplication
        unique_records = self.deduplicate_records(filtered_records)

        # Categorize records
        categorized_records = self.categorize_records(unique_records)

        # Balance dataset according to composition strategy
        target_size = len(unique_records)  # Maintain approximate size
        balanced_records = self.balance_dataset(categorized_records, target_size)

        # Enhance records with composition metadata
        for i, record in enumerate(balanced_records):
            if '_source_type_category' not in record:
                # Determine which category this record belongs to
                for category, category_records in categorized_records.items():
                    if record in category_records:
                        record['_source_type_category'] = category
                        break

        # Generate composition report
        report = self.generate_composition_report(records, balanced_records, categorized_records)

        # Save balanced dataset
        if not output_dataset_path:
            output_dir = get_dataset_pipeline_output_root() / "final_output"
            output_dir.mkdir(exist_ok=True)
            output_dataset_path = output_dir / "balanced_training_dataset.jsonl"

        with open(output_dataset_path, 'w') as f:
            for record in balanced_records:
                f.write(json.dumps(record) + '\n')

        logger.info(f"Balanced dataset saved to {output_dataset_path}")

        # Save composition report
        report_path = Path(str(output_dataset_path).replace('.jsonl', '_composition_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Composition report saved to {report_path}")

        return str(output_dataset_path), report

# Convenience functions
def create_default_composer() -> DatasetComposer:
    """Create a composer with default configuration"""
    config = {
        'quality_threshold': 0.7
    }
    return DatasetComposer(config)

def run_composition_strategy(input_dataset_path: str) -> Tuple[str, Dict[str, Any]]:
    """Run the complete composition strategy"""
    composer = create_default_composer()
    return composer.execute_composition_strategy(input_dataset_path)

if __name__ == "__main__":
    # Example usage
    try:
        # This would typically be run after the unified preprocessing pipeline
        # final_dataset_path = "ai/pipelines/orchestrator/final_output/unified_training_dataset_*.jsonl"
        # balanced_dataset_path, report = run_composition_strategy(final_dataset_path)
        # print(f"Composition strategy completed. Balanced dataset: {balanced_dataset_path}")
        pass
    except Exception as e:
        print(f"Composition strategy failed: {str(e)}")
        raise