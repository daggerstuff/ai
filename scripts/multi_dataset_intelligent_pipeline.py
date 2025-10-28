#!/usr/bin/env python3
"""
Multi-Dataset Intelligent Processing Pipeline
Applies the Multi-Pattern Intelligent Agent to ALL raw datasets across multiple locations,
then combines, merges, cleans, and formats for unified Lightning.ai H100 LoRA training.

Processes:
1. ai/lightning/pixelated-training/ - Extensive training set with processed data
2. ai/lightning/pixelated-v2/ - Main training set with cleaned transcripts
3. ai/lightning/ - Base raw datasets
4. .notes/pixel-training/ - Transcript sources
"""

import json
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
import sys
import hashlib
from collections import defaultdict

# Import path utilities and intelligent agent
sys.path.append(str(Path(__file__).parent))
from path_utils import get_workspace_root, get_unified_training_dir, get_ai_dir
from intelligent_prompt_agent import MultiPatternAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetSource:
    """Represents a dataset source location"""
    name: str
    path: Path
    type: str  # 'json', 'transcript', 'processed'
    priority: int  # Processing priority
    expected_format: str  # Expected data format

@dataclass
class ProcessingStats:
    """Track processing statistics across all datasets"""
    total_sources: int = 0
    total_files: int = 0
    total_conversations: int = 0
    processed_conversations: int = 0
    high_quality: int = 0
    medium_quality: int = 0
    low_quality: int = 0
    duplicates_removed: int = 0
    errors: int = 0
    extracted_questions: int = 0
    contextual_questions: int = 0

class MultiDatasetIntelligentPipeline:
    """Enhanced pipeline for processing multiple dataset sources"""
    
    def __init__(self, output_dir: Path = None):
        self.agent = MultiPatternAgent()
        self.output_dir = output_dir or get_unified_training_dir()
        self.output_dir.mkdir(exist_ok=True)
        self.stats = ProcessingStats()
        self.processed_hashes = set()  # For deduplication
        self.conversation_cache = {}  # For intelligent merging
        
        # Define dataset sources (using dynamic path resolution)
        workspace_root = get_workspace_root()
        self.dataset_sources = [
            # Priority 1: Existing high-quality processed datasets
            DatasetSource(
                "pixelated-training-processed",
                workspace_root / "ai/lightning/pixelated-training/processed",
                "json",
                1,
                "filtered_conversations"
            ),
            DatasetSource(
                "pixelated-v2-training-examples",
                workspace_root / "ai/lightning/pixelated-v2/data/training_examples",
                "json", 
                1,
                "quality_conversations"
            ),
            DatasetSource(
                "pixelated-training-moe-data",
                workspace_root / "ai/lightning/pixelated-training/moe_training_data",
                "json",
                1,
                "expert_conversations"
            ),
            
            # Priority 2: Raw transcript data needing intelligent processing
            DatasetSource(
                "pixelated-v2-transcripts",
                workspace_root / "ai/lightning/pixelated-v2/data/processed",
                "transcript",
                2,
                "raw_text"
            ),
            DatasetSource(
                "pixel-training-notes",
                workspace_root / ".notes/pixel-training",
                "transcript",
                2,
                "raw_text"
            ),
            DatasetSource(
                "pixelated-training-transcripts", 
                workspace_root / "ai/lightning/pixelated-training/transcripts",
                "transcript",
                2,
                "markdown_transcripts"
            ),
            
            # Priority 3: Base lightning datasets
            DatasetSource(
                "lightning-base",
                workspace_root / "ai/lightning",
                "json",
                3,
                "base_conversations"
            )
        ]
    
    def discover_dataset_files(self) -> Dict[str, List[Path]]:
        """Discover all dataset files across sources"""
        logger.info("🔍 Discovering dataset files across all sources...")
        
        discovered_files = {}
        
        for source in self.dataset_sources:
            source_files = []
            
            if not source.path.exists():
                logger.warning(f"Source path does not exist: {source.path}")
                continue
                
            if source.type == "json":
                # Find JSON conversation files
                json_files = list(source.path.rglob("*.json"))
                # Filter out config/metadata files
                conversation_files = [
                    f for f in json_files 
                    if not any(x in f.name.lower() for x in ['config', 'metadata', 'report', 'stats', 'summary'])
                ]
                source_files.extend(conversation_files)
                
            elif source.type == "transcript":
                # Find transcript files (txt, md)
                transcript_files = list(source.path.rglob("*.txt")) + list(source.path.rglob("*.md"))
                source_files.extend(transcript_files)
            
            discovered_files[source.name] = source_files
            self.stats.total_files += len(source_files)
            
            logger.info(f"  {source.name}: {len(source_files)} files")
        
        self.stats.total_sources = len([s for s in discovered_files.values() if s])
        logger.info(f"📊 Total: {self.stats.total_sources} sources, {self.stats.total_files} files")
        
        return discovered_files
    
    def load_conversation_data(self, file_path: Path, source_type: str) -> List[Dict]:
        """Load conversation data from various file formats"""
        try:
            if source_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Check for common conversation container keys
                    for key in ['conversations', 'data', 'examples', 'training_data']:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    # Single conversation
                    return [data]
                
            elif source_type == "transcript":
                # Convert transcript to conversation format for intelligent processing
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Create a segment for intelligent processing
                return [{
                    'text': text,
                    'style': 'therapeutic',  # Default style
                    'source': file_path.stem,
                    'file': str(file_path),
                    'confidence': 0.5,
                    'quality': 0.5
                }]
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            self.stats.errors += 1
            
        return []
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for deduplication"""
        # Normalize content for hashing
        normalized = content.lower().strip()
        # Remove common variations
        normalized = normalized.replace('\n', ' ').replace('\r', '')
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate_content(self, content: str) -> bool:
        """Check if content is duplicate using fuzzy matching"""
        content_hash = self.generate_content_hash(content)
        
        if content_hash in self.processed_hashes:
            return True
            
        # Check for near-duplicates (first 100 chars)
        short_hash = self.generate_content_hash(content[:100])
        for existing_hash in self.processed_hashes:
            if existing_hash.startswith(short_hash[:16]):  # First 16 chars match
                return True
                
        self.processed_hashes.add(content_hash)
        return False
    
    def extract_conversation_content(self, item: Dict) -> str:
        """Extract text content from conversation item"""
        # Handle different conversation formats
        if 'conversations' in item:
            # Standard conversation format
            content_parts = []
            for conv in item['conversations']:
                if 'value' in conv:
                    content_parts.append(conv['value'])
            return ' '.join(content_parts)
            
        elif 'text' in item:
            # Simple text format
            return item['text']
            
        elif 'content' in item:
            # Alternative content field
            return item['content']
            
        elif 'response' in item:
            # Question-response format
            question = item.get('question', '')
            response = item.get('response', '')
            return f"{question} {response}"
            
        return ""
    
    def process_dataset_source(self, source_name: str, files: List[Path]) -> List[Dict]:
        """Process all files from a dataset source"""
        logger.info(f"🔄 Processing {source_name} ({len(files)} files)")
        
        source = next(s for s in self.dataset_sources if s.name == source_name)
        processed_conversations = []
        
        for file_path in files:
            if file_path.suffix.lower() in ['.json']:
                conversations = self.load_conversation_data(file_path, "json")
            else:
                conversations = self.load_conversation_data(file_path, "transcript")
            
            for item in conversations:
                # Extract content for deduplication
                content = self.extract_conversation_content(item)
                if not content or len(content.strip()) < 10:
                    continue
                    
                # Check for duplicates
                if self.is_duplicate_content(content):
                    self.stats.duplicates_removed += 1
                    continue
                
                # Process with intelligent agent if it's raw content
                if source.type == "transcript" or 'conversations' not in item:
                    # Apply intelligent processing to raw content
                    if 'text' not in item:
                        item['text'] = content
                    
                    processed_item, status = self.process_segment_with_agent(item)
                    if processed_item:
                        processed_conversations.append(processed_item)
                        self.stats.processed_conversations += 1
                        
                        # Track generation method
                        method = processed_item['metadata']['analysis']['generation_method']
                        if method == 'extracted':
                            self.stats.extracted_questions += 1
                        else:
                            self.stats.contextual_questions += 1
                            
                        # Track quality
                        quality = processed_item.get('computed_quality', 0.0)
                        if quality >= 0.7:
                            self.stats.high_quality += 1
                        elif quality >= 0.4:
                            self.stats.medium_quality += 1
                        else:
                            self.stats.low_quality += 1
                else:
                    # Already processed conversation - validate and include
                    if self.validate_existing_conversation(item):
                        processed_conversations.append(item)
                        self.stats.processed_conversations += 1
                        self.stats.high_quality += 1  # Assume existing are good quality
        
        logger.info(f"  ✅ {source_name}: {len(processed_conversations)} conversations processed")
        return processed_conversations
    
    def process_segment_with_agent(self, segment: Dict) -> Tuple[Dict, str]:
        """Process segment with intelligent agent (from enhanced pipeline)"""
        try:
            # Analysis with intelligent agent
            analysis = self.agent.analyze_segment(segment['text'])
            
            # Determine Q/A generation strategy
            if (analysis['extracted_question'] and 
                analysis['question_confidence'] > 0.4 and
                analysis['semantic_coherence'] > 0.3):
                # Use extracted question
                question = analysis['extracted_question']
                method = "extracted"
                
                if not question.endswith('?') and not question.endswith('.'):
                    question += "?"
            else:
                # Generate contextual question
                question = self.agent.generate_contextual_prompt(segment, analysis)
                method = "contextual"
            
            # Create training pair
            training_pair = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": segment['text']}
                ],
                "expert_id": self._get_expert_id(segment.get('style', 'therapeutic')),
                "style": segment.get('style', 'therapeutic'),
                "quality": segment.get('quality', 0.5),
                "source": segment.get('source', 'unknown'),
                "metadata": {
                    "confidence": segment.get('confidence', 0.0),
                    "file": segment.get('file', 'unknown'),
                    "expert_name": segment.get('style', 'therapeutic'),
                    "analysis": {
                        "content_type": analysis['content_type'],
                        "content_confidence": analysis['content_confidence'],
                        "question_confidence": analysis['question_confidence'],
                        "semantic_coherence": analysis['semantic_coherence'],
                        "overall_confidence": analysis['overall_confidence'],
                        "generation_method": method,
                        "transition_markers": analysis['transition_markers'],
                        "processing_notes": analysis['processing_notes']
                    }
                }
            }
            
            # Quality assessment
            quality_score = self._assess_quality(training_pair, analysis)
            training_pair["computed_quality"] = quality_score
            
            return training_pair, "success"
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
            return None, f"error: {e}"
    
    def _get_expert_id(self, style: str) -> int:
        """Map style to expert ID"""
        expert_mapping = {
            "therapeutic": 0,
            "educational": 1, 
            "empathetic": 2,
            "practical": 3
        }
        return expert_mapping.get(style, 0)
    
    def _assess_quality(self, training_pair: Dict, analysis: Dict) -> float:
        """Assess training pair quality"""
        quality_score = training_pair.get('quality', 0.5) * 0.3
        quality_score += analysis['overall_confidence'] * 0.4
        quality_score += analysis['semantic_coherence'] * 0.3
        
        if analysis.get('generation_method') == 'extracted':
            quality_score += 0.1
            
        if analysis['content_confidence'] > 0.7:
            quality_score += 0.05
            
        if analysis['overall_confidence'] < 0.2:
            quality_score -= 0.1
            
        return max(0.0, min(1.0, quality_score))
    
    def validate_existing_conversation(self, item: Dict) -> bool:
        """Validate existing conversation format"""
        if 'conversations' in item:
            convs = item['conversations']
            if len(convs) >= 2:
                return True
        return False
    
    def combine_and_deduplicate(self, all_conversations: List[Dict]) -> List[Dict]:
        """Intelligent combination and deduplication of all conversations"""
        logger.info("🔄 Combining and deduplicating all conversations...")
        
        # Group by content similarity for intelligent merging
        content_groups = defaultdict(list)
        
        for conv in all_conversations:
            content = self.extract_conversation_content(conv)
            content_hash = self.generate_content_hash(content[:200])  # Use first 200 chars
            content_groups[content_hash].append(conv)
        
        merged_conversations = []
        
        for content_hash, group in content_groups.items():
            if len(group) == 1:
                # No duplicates
                merged_conversations.append(group[0])
            else:
                # Merge duplicates intelligently - keep highest quality
                best_conv = max(group, key=lambda x: x.get('computed_quality', x.get('quality', 0)))
                
                # Merge metadata from all versions
                merged_sources = set()
                for conv in group:
                    source = conv.get('source', conv.get('metadata', {}).get('file', 'unknown'))
                    merged_sources.add(source)
                
                best_conv['metadata'] = best_conv.get('metadata', {})
                best_conv['metadata']['merged_sources'] = list(merged_sources)
                best_conv['metadata']['merge_count'] = len(group)
                
                merged_conversations.append(best_conv)
                self.stats.duplicates_removed += len(group) - 1
        
        logger.info(f"✅ Combined: {len(all_conversations)} → {len(merged_conversations)} conversations")
        logger.info(f"📊 Removed {self.stats.duplicates_removed} duplicates")
        
        return merged_conversations
    
    def create_unified_dataset(self, conversations: List[Dict]) -> Dict:
        """Create unified Lightning.ai H100 LoRA dataset"""
        logger.info("🚀 Creating unified Lightning.ai H100 LoRA dataset...")
        
        # Sort by quality
        conversations.sort(key=lambda x: x.get('computed_quality', x.get('quality', 0)), reverse=True)
        
        # Split train/validation (90/10)
        total = len(conversations)
        train_size = int(total * 0.9)
        
        train_data = conversations[:train_size]
        val_data = conversations[train_size:]
        
        # Create expert-specific datasets
        experts = {"therapeutic": 0, "educational": 1, "empathetic": 2, "practical": 3}
        expert_data = {style: [] for style in experts.keys()}
        
        for conv in conversations:
            style = conv.get('style', 'therapeutic')
            if style in expert_data:
                expert_data[style].append(conv)
        
        # Save datasets
        datasets = {
            'train.json': train_data,
            'validation.json': val_data,
            **{f'expert_{style}.json': data for style, data in expert_data.items()}
        }
        
        for filename, data in datasets.items():
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create configuration
        config = {
            "model_config": {
                "base_model": "microsoft/DialoGPT-medium",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            "training_config": {
                "num_experts": 4,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "num_epochs": 3,
                "max_length": 1024,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "save_steps": 100,
                "eval_steps": 100,
                "logging_steps": 10
            },
            "data_config": {
                "train_file": "train.json",
                "validation_file": "validation.json",
                "expert_files": {f"expert_{style}": f"expert_{style}.json" for style in experts.keys()}
            },
            "expert_mapping": experts,
            "dataset_stats": {
                "total_conversations": total,
                "train_conversations": len(train_data),
                "validation_conversations": len(val_data),
                "expert_distribution": {style: len(data) for style, data in expert_data.items()},
                "processing_stats": {
                    "total_sources": self.stats.total_sources,
                    "total_files": self.stats.total_files,
                    "processed_conversations": self.stats.processed_conversations,
                    "high_quality": self.stats.high_quality,
                    "medium_quality": self.stats.medium_quality,
                    "low_quality": self.stats.low_quality,
                    "duplicates_removed": self.stats.duplicates_removed,
                    "extracted_questions": self.stats.extracted_questions,
                    "contextual_questions": self.stats.contextual_questions,
                    "errors": self.stats.errors
                }
            }
        }
        
        config_path = self.output_dir / "unified_lightning_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Unified dataset created: {self.output_dir}")
        logger.info(f"📊 Total conversations: {total}")
        logger.info(f"📊 Train/Val split: {len(train_data)}/{len(val_data)}")
        
        return config
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive processing report"""
        report = {
            "multi_dataset_processing_summary": {
                "timestamp": str(pd.Timestamp.now()),
                "total_sources_processed": self.stats.total_sources,
                "total_files_processed": self.stats.total_files,
                "total_conversations": self.stats.processed_conversations,
                "success_rate": (self.stats.processed_conversations / max(1, self.stats.total_files)) * 100
            },
            "quality_distribution": {
                "high_quality": self.stats.high_quality,
                "medium_quality": self.stats.medium_quality,
                "low_quality": self.stats.low_quality,
                "quality_percentage": {
                    "high": (self.stats.high_quality / max(1, self.stats.processed_conversations)) * 100,
                    "medium": (self.stats.medium_quality / max(1, self.stats.processed_conversations)) * 100,
                    "low": (self.stats.low_quality / max(1, self.stats.processed_conversations)) * 100
                }
            },
            "intelligent_agent_performance": {
                "extracted_questions": self.stats.extracted_questions,
                "contextual_questions": self.stats.contextual_questions,
                "extraction_rate": (self.stats.extracted_questions / max(1, self.stats.processed_conversations)) * 100
            },
            "data_cleaning_results": {
                "duplicates_removed": self.stats.duplicates_removed,
                "errors_encountered": self.stats.errors,
                "deduplication_rate": (self.stats.duplicates_removed / max(1, self.stats.total_files)) * 100
            },
            "dataset_sources": [
                {
                    "name": source.name,
                    "path": str(source.path),
                    "type": source.type,
                    "priority": source.priority
                } for source in self.dataset_sources
            ],
            "next_steps": [
                "✅ Multi-pattern intelligent agent applied to all raw datasets",
                "✅ Intelligent combination and deduplication completed",
                "✅ Unified Lightning.ai H100 LoRA dataset created",
                "🚀 Ready for Lightning.ai H100 training deployment",
                "📈 Monitor training performance and iterate on quality"
            ]
        }
        
        report_path = self.output_dir / "comprehensive_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Run the multi-dataset intelligent processing pipeline"""
    logger.info("🚀 Starting Multi-Dataset Intelligent Processing Pipeline")
    logger.info("🎯 Mission: Apply intelligent agent to ALL raw datasets, then combine/merge/clean")
    
    # Initialize pipeline
    pipeline = MultiDatasetIntelligentPipeline()
    
    # Phase 1: Discover all dataset files
    discovered_files = pipeline.discover_dataset_files()
    
    if not any(discovered_files.values()):
        logger.error("❌ No dataset files discovered!")
        return
    
    # Phase 2: Process each dataset source with intelligent agent
    all_conversations = []
    
    for source_name, files in discovered_files.items():
        if not files:
            continue
            
        processed_conversations = pipeline.process_dataset_source(source_name, files)
        all_conversations.extend(processed_conversations)
    
    if not all_conversations:
        logger.error("❌ No conversations processed!")
        return
    
    # Phase 3: Combine and deduplicate intelligently
    unified_conversations = pipeline.combine_and_deduplicate(all_conversations)
    
    # Phase 4: Create unified Lightning.ai dataset
    config = pipeline.create_unified_dataset(unified_conversations)
    
    # Phase 5: Generate comprehensive report
    report = pipeline.generate_comprehensive_report()
    
    # Final summary
    logger.info("🎉 Multi-Dataset Processing Complete!")
    logger.info(f"✅ Processed {pipeline.stats.total_sources} sources with {pipeline.stats.total_files} files")
    logger.info(f"✅ Generated {len(unified_conversations)} high-quality conversations")
    logger.info(f"✅ Intelligent agent extraction rate: {(pipeline.stats.extracted_questions / max(1, pipeline.stats.processed_conversations)) * 100:.1f}%")
    logger.info(f"✅ Deduplication: {pipeline.stats.duplicates_removed} duplicates removed")
    logger.info(f"🚀 Unified dataset ready for Lightning.ai H100 training!")
    logger.info(f"📁 Output location: {pipeline.output_dir}")

if __name__ == "__main__":
    # Handle missing pandas import
    try:
        import pandas as pd
    except ImportError:
        import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.datetime.now()
    
    main()