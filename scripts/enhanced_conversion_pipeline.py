#!/usr/bin/env python3
"""
Enhanced Therapeutic Training Data Conversion Pipeline
Integrates the Intelligent Multi-Pattern Agent with LoRA format conversion.

Replaces the original system that created 100% generic questions with
intelligent content analysis and contextually appropriate Q/A generation.
"""

import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys

# Import our intelligent agent
sys.path.append(str(Path(__file__).parent))
from intelligent_prompt_agent import MultiPatternAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversionStats:
    """Track conversion statistics"""
    total_segments: int = 0
    processed: int = 0
    high_quality: int = 0
    medium_quality: int = 0
    low_quality: int = 0
    errors: int = 0
    interview_extracted: int = 0
    contextual_generated: int = 0

class EnhancedConversionPipeline:
    """Enhanced pipeline using intelligent pattern analysis"""
    
    def __init__(self, min_confidence_threshold: float = 0.3):
        self.agent = MultiPatternAgent()
        self.min_confidence_threshold = min_confidence_threshold
        self.stats = ConversionStats()
        
    def process_segment(self, segment: Dict) -> Tuple[Dict, str]:
        """Process a single segment with intelligent analysis"""
        try:
            # Step 1: Analyze segment with intelligent agent
            analysis = self.agent.analyze_segment(segment['text'])
            
            # Step 2: Determine Q/A generation strategy
            if (analysis['extracted_question'] and 
                analysis['question_confidence'] > 0.4 and
                analysis['semantic_coherence'] > 0.3):
                # Use extracted question from content
                question = analysis['extracted_question']
                method = "extracted"
                self.stats.interview_extracted += 1
                
                # Clean up question formatting
                if not question.endswith('?') and not question.endswith('.'):
                    question += "?"
                    
                logger.debug(f"Extracted question: {question}")
                
            else:
                # Generate contextual question
                question = self.agent.generate_contextual_prompt(segment, analysis)
                method = "contextual"
                self.stats.contextual_generated += 1
                
                logger.debug(f"Generated contextual question: {question}")
            
            # Step 3: Create training pair
            training_pair = {
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt", 
                        "value": segment['text']
                    }
                ],
                "expert_id": self._get_expert_id(segment['style']),
                "style": segment['style'],
                "quality": segment.get('quality', 0.5),
                "source": segment.get('source', 'unknown'),
                "metadata": {
                    "confidence": segment.get('confidence', 0.0),
                    "file": segment.get('file', 'unknown'),
                    "expert_name": segment['style'],
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
            
            # Step 4: Quality assessment
            quality_score = self._assess_quality(training_pair, analysis)
            training_pair["computed_quality"] = quality_score
            
            if quality_score >= 0.7:
                self.stats.high_quality += 1
            elif quality_score >= 0.4:
                self.stats.medium_quality += 1
            else:
                self.stats.low_quality += 1
            
            self.stats.processed += 1
            return training_pair, "success"
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Error processing segment: {e}")
            return None, f"error: {e}"
    
    def _get_expert_id(self, style: str) -> int:
        """Map style to expert ID for MoE architecture"""
        expert_mapping = {
            "therapeutic": 0,
            "educational": 1, 
            "empathetic": 2,
            "practical": 3
        }
        return expert_mapping.get(style, 0)
    
    def _assess_quality(self, training_pair: Dict, analysis: Dict) -> float:
        """Assess the quality of the generated training pair"""
        quality_score = 0.0
        
        # Base quality from original segment
        base_quality = training_pair.get('quality', 0.5)
        quality_score += base_quality * 0.3
        
        # Analysis confidence factors
        quality_score += analysis['overall_confidence'] * 0.4
        quality_score += analysis['semantic_coherence'] * 0.3
        
        # Method bonus
        if analysis.get('generation_method') == 'extracted':
            quality_score += 0.1  # Bonus for extracted questions
            
        # Content type confidence
        if analysis['content_confidence'] > 0.7:
            quality_score += 0.05
            
        # Penalty for low confidence
        if analysis['overall_confidence'] < 0.2:
            quality_score -= 0.1
            
        return max(0.0, min(1.0, quality_score))
    
    def process_segments_file(self, input_path: Path, output_path: Path) -> Dict:
        """Process a complete segments file"""
        logger.info(f"Processing {input_path.name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        training_pairs = []
        file_stats = ConversionStats()
        file_stats.total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(segments)} segments")
                
            training_pair, status = self.process_segment(segment)
            
            if training_pair:
                training_pairs.append(training_pair)
                file_stats.processed += 1
                
                # Update file-specific quality stats
                quality = training_pair.get('computed_quality', 0.0)
                if quality >= 0.7:
                    file_stats.high_quality += 1
                elif quality >= 0.4:
                    file_stats.medium_quality += 1
                else:
                    file_stats.low_quality += 1
                    
                # Track generation methods
                method = training_pair['metadata']['analysis']['generation_method']
                if method == 'extracted':
                    file_stats.interview_extracted += 1
                else:
                    file_stats.contextual_generated += 1
            else:
                file_stats.errors += 1
                logger.warning(f"  Failed to process segment {i}: {status}")
        
        # Save training pairs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Completed {input_path.name}: {file_stats.processed}/{file_stats.total_segments} segments")
        logger.info(f"  Quality: High={file_stats.high_quality}, Medium={file_stats.medium_quality}, Low={file_stats.low_quality}")
        logger.info(f"  Methods: Extracted={file_stats.interview_extracted}, Contextual={file_stats.contextual_generated}")
        
        return {
            "file": input_path.name,
            "total": file_stats.total_segments,
            "processed": file_stats.processed,
            "success_rate": file_stats.processed / file_stats.total_segments if file_stats.total_segments > 0 else 0,
            "quality_distribution": {
                "high": file_stats.high_quality,
                "medium": file_stats.medium_quality, 
                "low": file_stats.low_quality
            },
            "generation_methods": {
                "extracted": file_stats.interview_extracted,
                "contextual": file_stats.contextual_generated
            },
            "errors": file_stats.errors
        }
    
    def create_lightning_dataset(self, training_pairs: List[Dict], output_dir: Path):
        """Create Lightning.ai H100 LoRA format dataset"""
        logger.info("Creating Lightning.ai H100 LoRA dataset")
        
        # Split into train/validation (90/10)
        total_pairs = len(training_pairs)
        train_size = int(total_pairs * 0.9)
        
        train_data = training_pairs[:train_size]
        val_data = training_pairs[train_size:]
        
        # Save main datasets
        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
            
        with open(output_dir / "validation.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # Create expert-specific datasets
        experts = {"therapeutic": 0, "educational": 1, "empathetic": 2, "practical": 3}
        expert_data = {style: [] for style in experts.keys()}
        
        for pair in training_pairs:
            style = pair['style']
            if style in expert_data:
                expert_data[style].append(pair)
        
        for style, data in expert_data.items():
            filename = output_dir / f"expert_{style}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create Lightning.ai configuration
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
                "expert_files": {
                    f"expert_{style}": f"expert_{style}.json" 
                    for style in experts.keys()
                }
            },
            "expert_mapping": experts,
            "dataset_stats": {
                "total_conversations": total_pairs,
                "train_conversations": len(train_data),
                "validation_conversations": len(val_data),
                "expert_distribution": {
                    style: len(data) for style, data in expert_data.items()
                }
            }
        }
        
        with open(output_dir / "lightning_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        return config

    def generate_quality_report(self, all_stats: List[Dict], output_dir: Path):
        """Generate comprehensive quality report"""
        report = {
            "conversion_summary": {
                "total_files": len(all_stats),
                "total_segments": sum(s['total'] for s in all_stats),
                "total_processed": sum(s['processed'] for s in all_stats),
                "overall_success_rate": sum(s['processed'] for s in all_stats) / sum(s['total'] for s in all_stats),
                "total_errors": sum(s['errors'] for s in all_stats)
            },
            "quality_analysis": {
                "high_quality": sum(s['quality_distribution']['high'] for s in all_stats),
                "medium_quality": sum(s['quality_distribution']['medium'] for s in all_stats),
                "low_quality": sum(s['quality_distribution']['low'] for s in all_stats)
            },
            "generation_methods": {
                "extracted_questions": sum(s['generation_methods']['extracted'] for s in all_stats),
                "contextual_questions": sum(s['generation_methods']['contextual'] for s in all_stats)
            },
            "file_details": all_stats,
            "improvements_over_original": [
                "✅ Intelligent content type detection with confidence scoring",
                "✅ Actual question extraction from interview/podcast content", 
                "✅ Semantic coherence validation between questions and responses",
                "✅ Context-aware prompt generation instead of generic templates",
                "✅ Response boundary detection using transition markers",
                "✅ Multi-pattern analysis handling diverse content formats",
                "✅ Quality assessment based on analysis confidence",
                "✅ Comprehensive metadata for training optimization"
            ]
        }
        
        with open(output_dir / "conversion_quality_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {output_dir / 'conversion_quality_report.json'}")
        return report

def main():
    """Run the enhanced conversion pipeline"""
    logger.info("🚀 Starting Enhanced Therapeutic Training Data Conversion Pipeline")
    
    # Setup paths
    segments_dir = Path("/root/pixelated/data/training_segments")
    lora_dir = Path("/root/pixelated/data/lora_training")
    lightning_dir = Path("/root/pixelated/data/lightning_h100")
    
    # Create output directories
    lora_dir.mkdir(exist_ok=True)
    lightning_dir.mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = EnhancedConversionPipeline(min_confidence_threshold=0.3)
    
    # Process all segment files
    all_training_pairs = []
    all_file_stats = []
    
    segment_files = list(segments_dir.glob("*.json"))
    if not segment_files:
        logger.warning(f"No segment files found in {segments_dir}")
        logger.info("Creating sample data for testing...")
        
        # Create sample data for testing
        sample_segments = [
            {
                "text": "When someone has narcissistic traits, they often use manipulation and gaslighting to control others. This creates significant trauma for their victims who may struggle with self-doubt and confusion.",
                "style": "therapeutic",
                "confidence": 0.85,
                "quality": 0.75,
                "source": "tim_fletcher",
                "file": "narcissism_and_trauma.txt"
            },
            {
                "text": "Interviewer: How can somebody begin to heal from childhood trauma. Tim Fletcher: Well, that's a huge question because unfortunately, most people don't realize the depth of their childhood wounds.",
                "style": "therapeutic", 
                "confidence": 0.92,
                "quality": 0.68,
                "source": "tim_fletcher",
                "file": "childhood_trauma_interview.txt"
            }
        ]
        
        sample_file = segments_dir / "sample_segments.json"
        segments_dir.mkdir(exist_ok=True)
        with open(sample_file, 'w') as f:
            json.dump(sample_segments, f, indent=2)
        segment_files = [sample_file]
    
    for segment_file in segment_files:
        if segment_file.name.startswith("enhanced_summary"):
            continue
            
        output_file = lora_dir / f"enhanced_{segment_file.name}"
        file_stats = pipeline.process_segments_file(segment_file, output_file)
        all_file_stats.append(file_stats)
        
        # Load and combine training pairs
        with open(output_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        all_training_pairs.extend(pairs)
    
    # Create Lightning.ai dataset
    if all_training_pairs:
        config = pipeline.create_lightning_dataset(all_training_pairs, lightning_dir)
        logger.info(f"Created Lightning.ai dataset with {len(all_training_pairs)} conversations")
        
        # Generate quality report
        report = pipeline.generate_quality_report(all_file_stats, lightning_dir)
        
        # Log final summary
        logger.info("🎯 Enhanced Conversion Pipeline Complete!")
        logger.info(f"✅ Total Conversations: {len(all_training_pairs)}")
        logger.info(f"✅ Success Rate: {report['conversion_summary']['overall_success_rate']:.1%}")
        logger.info(f"✅ Quality Distribution: High={report['quality_analysis']['high_quality']}, Medium={report['quality_analysis']['medium_quality']}, Low={report['quality_analysis']['low_quality']}")
        logger.info(f"✅ Extracted Questions: {report['generation_methods']['extracted_questions']}")
        logger.info(f"✅ Contextual Questions: {report['generation_methods']['contextual_questions']}")
        logger.info(f"🚀 Ready for Lightning.ai H100 Training: {lightning_dir}")
        
    else:
        logger.error("No training pairs generated!")

if __name__ == "__main__":
    main()