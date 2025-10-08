#!/usr/bin/env python3
"""
Conversation Length and Complexity Analysis System
Analyzes conversation length patterns, complexity metrics, and provides insights
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ConversationMetrics:
    """Individual conversation metrics"""
    conversation_id: str
    turn_count: int
    word_count: int
    character_count: int
    avg_words_per_turn: float
    vocabulary_diversity: float
    sentence_complexity: float
    dialogue_depth: float
    topic_coherence: float
    overall_complexity_score: float

@dataclass
class ComplexityAnalysis:
    """Complexity analysis results"""
    dataset_name: str
    total_conversations: int
    length_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]
    average_metrics: Dict[str, float]
    correlation_analysis: Dict[str, float]
    patterns: List[str]
    recommendations: List[str]

class ConversationComplexityAnalyzer:
    """Enterprise-grade conversation complexity analysis system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/complexity_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis configuration
        self.analysis_config = {
            'max_conversations_per_dataset': 1000,
            'length_categories': {
                'very_short': (1, 2),      # 1-2 turns
                'short': (3, 5),           # 3-5 turns
                'medium': (6, 10),         # 6-10 turns
                'long': (11, 20),          # 11-20 turns
                'very_long': (21, 100)     # 21+ turns
            },
            'complexity_categories': {
                'simple': (0.0, 0.3),
                'moderate': (0.3, 0.6),
                'complex': (0.6, 0.8),
                'very_complex': (0.8, 1.0)
            },
            'min_word_count': 10
        }
        
    def analyze_conversation_complexity(self, dataset_name: Optional[str] = None) -> Dict[str, ComplexityAnalysis]:
        """Analyze conversation length and complexity patterns"""
        print("📏 Analyzing conversation length and complexity patterns...")
        
        try:
            # Get datasets to analyze
            if dataset_name:
                datasets = [dataset_name]
            else:
                datasets = self._get_dataset_list()
            
            complexity_analyses = {}
            
            for dataset in datasets:
                print(f"   Analyzing complexity for {dataset}...")
                analysis = self._analyze_dataset_complexity(dataset)
                if analysis:
                    complexity_analyses[dataset] = analysis
            
            print(f"✅ Analyzed complexity for {len(complexity_analyses)} datasets")
            return complexity_analyses
            
        except Exception as e:
            print(f"❌ Error analyzing conversation complexity: {e}")
            return {}
    
    def _get_dataset_list(self) -> List[str]:
        """Get list of datasets"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT DISTINCT dataset_source 
            FROM conversations 
            WHERE dataset_source IS NOT NULL 
            AND dataset_source != ''
            ORDER BY dataset_source
            """
            
            cursor = conn.execute(query)
            datasets = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return datasets
            
        except Exception as e:
            print(f"❌ Error getting dataset list: {e}")
            return []
    
    def _analyze_dataset_complexity(self, dataset_name: str) -> Optional[ComplexityAnalysis]:
        """Analyze complexity for specific dataset"""
        try:
            # Get conversations from dataset
            conversations = self._get_dataset_conversations(dataset_name)
            
            if not conversations:
                return None
            
            # Analyze individual conversations
            conversation_metrics = []
            for conv in conversations:
                metrics = self._analyze_individual_conversation(conv)
                if metrics:
                    conversation_metrics.append(metrics)
            
            if not conversation_metrics:
                return None
            
            # Generate complexity analysis
            analysis = self._generate_complexity_analysis(dataset_name, conversation_metrics)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing dataset complexity for {dataset_name}: {e}")
            return None
    
    def _get_dataset_conversations(self, dataset_name: str) -> List[Dict]:
        """Get conversations for dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                conversation_id,
                conversations_json,
                word_count,
                turn_count,
                character_count
            FROM conversations 
            WHERE dataset_source = ?
            AND conversations_json IS NOT NULL
            AND word_count >= ?
            ORDER BY RANDOM()
            LIMIT ?
            """
            
            cursor = conn.execute(query, (
                dataset_name, 
                self.analysis_config['min_word_count'],
                self.analysis_config['max_conversations_per_dataset']
            ))
            
            columns = [desc[0] for desc in cursor.description]
            conversations = []
            
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                try:
                    record['conversations'] = json.loads(record['conversations_json'])
                    conversations.append(record)
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            return conversations
            
        except Exception as e:
            print(f"❌ Error getting conversations for {dataset_name}: {e}")
            return []
    
    def _analyze_individual_conversation(self, conversation_data: Dict) -> Optional[ConversationMetrics]:
        """Analyze individual conversation metrics"""
        try:
            conversation_id = conversation_data['conversation_id']
            conversations = conversation_data.get('conversations', [])
            
            if not conversations:
                return None
            
            # Basic metrics
            turn_count = len(conversations)
            word_count = conversation_data.get('word_count', 0)
            character_count = conversation_data.get('character_count', 0)
            
            # Calculate derived metrics
            avg_words_per_turn = word_count / turn_count if turn_count > 0 else 0
            
            # Extract all text for analysis
            all_text = self._extract_conversation_text(conversations)
            
            # Vocabulary diversity
            vocabulary_diversity = self._calculate_vocabulary_diversity(all_text)
            
            # Sentence complexity
            sentence_complexity = self._calculate_sentence_complexity(all_text)
            
            # Dialogue depth
            dialogue_depth = self._calculate_dialogue_depth(conversations)
            
            # Topic coherence
            topic_coherence = self._calculate_topic_coherence(all_text)
            
            # Overall complexity score
            overall_complexity_score = self._calculate_overall_complexity(
                vocabulary_diversity, sentence_complexity, dialogue_depth, topic_coherence
            )
            
            return ConversationMetrics(
                conversation_id=conversation_id,
                turn_count=turn_count,
                word_count=word_count,
                character_count=character_count,
                avg_words_per_turn=avg_words_per_turn,
                vocabulary_diversity=vocabulary_diversity,
                sentence_complexity=sentence_complexity,
                dialogue_depth=dialogue_depth,
                topic_coherence=topic_coherence,
                overall_complexity_score=overall_complexity_score
            )
            
        except Exception as e:
            print(f"❌ Error analyzing individual conversation: {e}")
            return None
    
    def _extract_conversation_text(self, conversations: List) -> str:
        """Extract text from conversation structure"""
        try:
            all_text = ""
            
            for turn in conversations:
                if isinstance(turn, dict):
                    text = ""
                    text += turn.get('human', '') + ' '
                    text += turn.get('assistant', '') + ' '
                    text += turn.get('user', '') + ' '
                    text += turn.get('bot', '') + ' '
                    text += turn.get('input', '') + ' '
                    text += turn.get('output', '') + ' '
                    all_text += text
                elif isinstance(turn, str):
                    all_text += turn + ' '
            
            return all_text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting conversation text: {e}")
            return ""
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (unique words / total words)"""
        try:
            if not text:
                return 0.0
            
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            if not words:
                return 0.0
            
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            
            return min(1.0, diversity)
            
        except Exception as e:
            print(f"❌ Error calculating vocabulary diversity: {e}")
            return 0.0
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity based on length and structure"""
        try:
            if not text:
                return 0.0
            
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            # Calculate average sentence length
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            # Normalize to 0-1 scale (assuming max 30 words per sentence)
            complexity = min(1.0, avg_sentence_length / 30)
            
            return complexity
            
        except Exception as e:
            print(f"❌ Error calculating sentence complexity: {e}")
            return 0.0
    
    def _calculate_dialogue_depth(self, conversations: List) -> float:
        """Calculate dialogue depth based on turn patterns"""
        try:
            if not conversations:
                return 0.0
            
            # Simple depth calculation based on turn count
            turn_count = len(conversations)
            
            # Normalize to 0-1 scale (assuming max 20 turns for high depth)
            depth = min(1.0, turn_count / 20)
            
            return depth
            
        except Exception as e:
            print(f"❌ Error calculating dialogue depth: {e}")
            return 0.0
    
    def _calculate_topic_coherence(self, text: str) -> float:
        """Calculate topic coherence based on word repetition and flow"""
        try:
            if not text:
                return 0.0
            
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            if len(words) < 10:
                return 0.0
            
            # Calculate word frequency
            word_counts = Counter(words)
            
            # Calculate coherence based on repeated important words
            # (excluding common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            content_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            if not content_words:
                return 0.0
            
            content_word_counts = Counter(content_words)
            
            # Calculate coherence as ratio of repeated content words
            repeated_words = sum(1 for count in content_word_counts.values() if count > 1)
            coherence = repeated_words / len(content_word_counts) if content_word_counts else 0
            
            return min(1.0, coherence)
            
        except Exception as e:
            print(f"❌ Error calculating topic coherence: {e}")
            return 0.0
    
    def _calculate_overall_complexity(self, vocabulary_diversity: float, sentence_complexity: float,
                                    dialogue_depth: float, topic_coherence: float) -> float:
        """Calculate overall complexity score"""
        try:
            # Weighted average of complexity components
            weights = {
                'vocabulary': 0.3,
                'sentence': 0.25,
                'dialogue': 0.25,
                'coherence': 0.2
            }
            
            overall_score = (
                vocabulary_diversity * weights['vocabulary'] +
                sentence_complexity * weights['sentence'] +
                dialogue_depth * weights['dialogue'] +
                topic_coherence * weights['coherence']
            )
            
            return min(1.0, overall_score)
            
        except Exception as e:
            print(f"❌ Error calculating overall complexity: {e}")
            return 0.0
    
    def _generate_complexity_analysis(self, dataset_name: str, 
                                    conversation_metrics: List[ConversationMetrics]) -> ComplexityAnalysis:
        """Generate complexity analysis from conversation metrics"""
        try:
            total_conversations = len(conversation_metrics)
            
            # Length distribution
            length_distribution = {}
            for category, (min_turns, max_turns) in self.analysis_config['length_categories'].items():
                count = len([m for m in conversation_metrics 
                           if min_turns <= m.turn_count <= max_turns])
                length_distribution[category] = count
            
            # Complexity distribution
            complexity_distribution = {}
            for category, (min_score, max_score) in self.analysis_config['complexity_categories'].items():
                count = len([m for m in conversation_metrics 
                           if min_score <= m.overall_complexity_score < max_score])
                complexity_distribution[category] = count
            
            # Average metrics
            average_metrics = {
                'avg_turn_count': np.mean([m.turn_count for m in conversation_metrics]),
                'avg_word_count': np.mean([m.word_count for m in conversation_metrics]),
                'avg_words_per_turn': np.mean([m.avg_words_per_turn for m in conversation_metrics]),
                'avg_vocabulary_diversity': np.mean([m.vocabulary_diversity for m in conversation_metrics]),
                'avg_sentence_complexity': np.mean([m.sentence_complexity for m in conversation_metrics]),
                'avg_dialogue_depth': np.mean([m.dialogue_depth for m in conversation_metrics]),
                'avg_topic_coherence': np.mean([m.topic_coherence for m in conversation_metrics]),
                'avg_overall_complexity': np.mean([m.overall_complexity_score for m in conversation_metrics])
            }
            
            # Correlation analysis
            correlation_analysis = self._calculate_correlations(conversation_metrics)
            
            # Identify patterns
            patterns = self._identify_complexity_patterns(conversation_metrics, average_metrics)
            
            # Generate recommendations
            recommendations = self._generate_complexity_recommendations(
                average_metrics, length_distribution, complexity_distribution
            )
            
            return ComplexityAnalysis(
                dataset_name=dataset_name,
                total_conversations=total_conversations,
                length_distribution=length_distribution,
                complexity_distribution=complexity_distribution,
                average_metrics=average_metrics,
                correlation_analysis=correlation_analysis,
                patterns=patterns,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"❌ Error generating complexity analysis: {e}")
            return ComplexityAnalysis(
                dataset_name=dataset_name,
                total_conversations=0,
                length_distribution={},
                complexity_distribution={},
                average_metrics={},
                correlation_analysis={},
                patterns=[],
                recommendations=[]
            )
    
    def _calculate_correlations(self, conversation_metrics: List[ConversationMetrics]) -> Dict[str, float]:
        """Calculate correlations between different metrics"""
        try:
            if len(conversation_metrics) < 10:  # Need sufficient data
                return {}
            
            # Extract metric arrays
            turn_counts = [m.turn_count for m in conversation_metrics]
            word_counts = [m.word_count for m in conversation_metrics]
            complexity_scores = [m.overall_complexity_score for m in conversation_metrics]
            vocabulary_diversity = [m.vocabulary_diversity for m in conversation_metrics]
            
            correlations = {}
            
            # Calculate correlations
            correlations['length_complexity'] = np.corrcoef(turn_counts, complexity_scores)[0, 1]
            correlations['words_complexity'] = np.corrcoef(word_counts, complexity_scores)[0, 1]
            correlations['vocabulary_complexity'] = np.corrcoef(vocabulary_diversity, complexity_scores)[0, 1]
            correlations['length_words'] = np.corrcoef(turn_counts, word_counts)[0, 1]
            
            # Handle NaN values
            for key, value in correlations.items():
                if np.isnan(value):
                    correlations[key] = 0.0
                else:
                    correlations[key] = float(value)
            
            return correlations
            
        except Exception as e:
            print(f"❌ Error calculating correlations: {e}")
            return {}
    
    def _identify_complexity_patterns(self, conversation_metrics: List[ConversationMetrics],
                                    average_metrics: Dict[str, float]) -> List[str]:
        """Identify patterns in complexity data"""
        patterns = []
        
        try:
            # Length patterns
            avg_turns = average_metrics.get('avg_turn_count', 0)
            if avg_turns > 15:
                patterns.append("Dataset contains predominantly long conversations")
            elif avg_turns < 5:
                patterns.append("Dataset contains predominantly short conversations")
            else:
                patterns.append("Dataset has balanced conversation lengths")
            
            # Complexity patterns
            avg_complexity = average_metrics.get('avg_overall_complexity', 0)
            if avg_complexity > 0.7:
                patterns.append("High complexity conversations with rich vocabulary and deep dialogue")
            elif avg_complexity < 0.3:
                patterns.append("Simple conversations with basic vocabulary and structure")
            else:
                patterns.append("Moderate complexity conversations with balanced characteristics")
            
            # Vocabulary patterns
            avg_vocab_diversity = average_metrics.get('avg_vocabulary_diversity', 0)
            if avg_vocab_diversity > 0.6:
                patterns.append("Rich vocabulary diversity across conversations")
            elif avg_vocab_diversity < 0.3:
                patterns.append("Limited vocabulary diversity - repetitive language patterns")
            
            # Coherence patterns
            avg_coherence = average_metrics.get('avg_topic_coherence', 0)
            if avg_coherence > 0.6:
                patterns.append("Strong topic coherence - conversations stay on topic")
            elif avg_coherence < 0.3:
                patterns.append("Weak topic coherence - conversations may lack focus")
            
            return patterns
            
        except Exception as e:
            print(f"❌ Error identifying patterns: {e}")
            return []
    
    def _generate_complexity_recommendations(self, average_metrics: Dict[str, float],
                                           length_distribution: Dict[str, int],
                                           complexity_distribution: Dict[str, int]) -> List[str]:
        """Generate recommendations based on complexity analysis"""
        recommendations = []
        
        try:
            # Length recommendations
            total_conversations = sum(length_distribution.values())
            if total_conversations > 0:
                very_short_pct = (length_distribution.get('very_short', 0) / total_conversations) * 100
                if very_short_pct > 50:
                    recommendations.append("High percentage of very short conversations - consider enhancing dialogue depth")
                
                very_long_pct = (length_distribution.get('very_long', 0) / total_conversations) * 100
                if very_long_pct > 30:
                    recommendations.append("Many very long conversations - monitor for potential verbosity issues")
            
            # Complexity recommendations
            avg_complexity = average_metrics.get('avg_overall_complexity', 0)
            if avg_complexity < 0.4:
                recommendations.append("Low overall complexity - consider enriching conversation content and structure")
            elif avg_complexity > 0.8:
                recommendations.append("Very high complexity - ensure conversations remain accessible to users")
            
            # Vocabulary recommendations
            avg_vocab_diversity = average_metrics.get('avg_vocabulary_diversity', 0)
            if avg_vocab_diversity < 0.3:
                recommendations.append("Limited vocabulary diversity - expand language variety in conversations")
            
            # Coherence recommendations
            avg_coherence = average_metrics.get('avg_topic_coherence', 0)
            if avg_coherence < 0.4:
                recommendations.append("Improve topic coherence - ensure conversations maintain thematic consistency")
            
            if not recommendations:
                recommendations.append("Conversation complexity appears well-balanced across all metrics")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return ["Error generating complexity recommendations"]
    
    def export_complexity_report(self, complexity_analyses: Dict[str, ComplexityAnalysis]) -> str:
        """Export comprehensive complexity analysis report"""
        print("📄 Exporting complexity analysis report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"complexity_analysis_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'datasets_analyzed': len(complexity_analyses),
                    'analysis_type': 'conversation_length_and_complexity'
                },
                'complexity_analyses': {
                    dataset_name: {
                        'total_conversations': analysis.total_conversations,
                        'length_distribution': analysis.length_distribution,
                        'complexity_distribution': analysis.complexity_distribution,
                        'average_metrics': analysis.average_metrics,
                        'correlation_analysis': analysis.correlation_analysis,
                        'patterns': analysis.patterns,
                        'recommendations': analysis.recommendations
                    }
                    for dataset_name, analysis in complexity_analyses.items()
                },
                'summary_statistics': self._create_complexity_summary(complexity_analyses)
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported complexity analysis report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting complexity report: {e}")
            return ""
    
    def _create_complexity_summary(self, complexity_analyses: Dict[str, ComplexityAnalysis]) -> Dict[str, Any]:
        """Create summary statistics across all datasets"""
        try:
            if not complexity_analyses:
                return {}
            
            total_conversations = sum(analysis.total_conversations for analysis in complexity_analyses.values())
            
            # Average complexity across all datasets
            all_complexity_scores = []
            for analysis in complexity_analyses.values():
                if 'avg_overall_complexity' in analysis.average_metrics:
                    all_complexity_scores.append(analysis.average_metrics['avg_overall_complexity'])
            
            avg_complexity = np.mean(all_complexity_scores) if all_complexity_scores else 0
            
            # Most complex dataset
            most_complex_dataset = None
            if all_complexity_scores:
                dataset_complexities = [(name, analysis.average_metrics.get('avg_overall_complexity', 0)) 
                                      for name, analysis in complexity_analyses.items()]
                most_complex_dataset = max(dataset_complexities, key=lambda x: x[1])[0]
            
            return {
                'total_conversations_analyzed': total_conversations,
                'datasets_analyzed': len(complexity_analyses),
                'average_complexity_score': float(avg_complexity),
                'most_complex_dataset': most_complex_dataset,
                'analysis_completion_rate': 100.0
            }
            
        except Exception as e:
            print(f"❌ Error creating complexity summary: {e}")
            return {}

def main():
    """Main execution function"""
    print("📏 Conversation Length and Complexity Analysis System")
    print("=" * 65)
    
    # Initialize analyzer
    analyzer = ConversationComplexityAnalyzer()
    
    # Analyze conversation complexity
    complexity_analyses = analyzer.analyze_conversation_complexity()
    
    if not complexity_analyses:
        print("❌ No complexity analyses generated")
        return
    
    # Export report
    report_file = analyzer.export_complexity_report(complexity_analyses)
    
    # Display summary
    total_conversations = sum(analysis.total_conversations for analysis in complexity_analyses.values())
    
    print(f"\n✅ Complexity Analysis Complete")
    print(f"   - Datasets analyzed: {len(complexity_analyses)}")
    print(f"   - Total conversations analyzed: {total_conversations}")
    print(f"   - Report saved: {report_file}")
    
    # Show sample analysis
    if complexity_analyses:
        sample_dataset = list(complexity_analyses.keys())[0]
        sample_analysis = complexity_analyses[sample_dataset]
        
        print(f"\n📊 Sample Analysis for {sample_dataset}:")
        print(f"   - Conversations: {sample_analysis.total_conversations}")
        print(f"   - Avg complexity: {sample_analysis.average_metrics.get('avg_overall_complexity', 0):.3f}")
        print(f"   - Avg turns: {sample_analysis.average_metrics.get('avg_turn_count', 0):.1f}")
        print(f"   - Avg words: {sample_analysis.average_metrics.get('avg_word_count', 0):.1f}")
        
        if sample_analysis.patterns:
            print(f"\n🔍 Key Patterns:")
            for pattern in sample_analysis.patterns[:3]:
                print(f"   • {pattern}")

if __name__ == "__main__":
    main()
