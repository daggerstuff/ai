#!/usr/bin/env python3
"""
Conversation Quality Pattern Analyzer
Task 5.6.3.6: Create conversation quality pattern analysis

Analyzes patterns in conversation quality across datasets, identifying:
- Quality indicators and metrics
- Conversation flow patterns
- Response quality patterns
- Engagement quality patterns
- Quality degradation/improvement patterns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ConversationQualityPatternAnalyzer:
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        self.db_path = db_path
        self.quality_metrics = {}
        self.patterns = {}
        self.insights = []
        
    def connect_db(self):
        """Connect to the conversations database"""
        return sqlite3.connect(self.db_path)
    
    def analyze_quality_patterns(self) -> Dict[str, Any]:
        """Main analysis function for conversation quality patterns"""
        print("🔍 Starting Conversation Quality Pattern Analysis...")
        
        # Load conversation data
        conversations = self._load_conversation_data()
        print(f"📊 Loaded {len(conversations)} conversations for analysis")
        
        # Analyze different quality dimensions
        quality_results = {
            'response_quality': self._analyze_response_quality(conversations),
            'engagement_quality': self._analyze_engagement_quality(conversations),
            'flow_quality': self._analyze_conversation_flow(conversations),
            'content_quality': self._analyze_content_quality(conversations),
            'coherence_quality': self._analyze_coherence_quality(conversations),
            'quality_patterns': self._identify_quality_patterns(conversations),
            'quality_trends': self._analyze_quality_trends(conversations),
            'quality_correlations': self._analyze_quality_correlations(conversations)
        }
        
        # Generate insights and recommendations
        insights = self._generate_quality_insights(quality_results)
        
        # Create visualizations
        self._create_quality_visualizations(quality_results)
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'quality_analysis': quality_results,
            'insights': insights,
            'recommendations': self._generate_quality_recommendations(quality_results)
        }
    
    def _load_conversation_data(self) -> pd.DataFrame:
        """Load conversation data with quality-relevant fields"""
        with self.connect_db() as conn:
            query = """
            SELECT 
                conversation_id, dataset_source as dataset, tier, conversations_json, 
                character_count as text_length,
                word_count,
                turn_count as line_count,
                created_at
            FROM conversations 
            WHERE conversations_json IS NOT NULL 
            AND length(conversations_json) > 10
            """
            df = pd.read_sql_query(query, conn)
            
        # Extract conversation text from JSON
        df['conversation_text'] = df['conversations_json'].apply(self._extract_text_from_json)
            
        # Extract conversation text from JSON
        df['conversation_text'] = df['conversations_json'].apply(self._extract_text_from_json)
        
        # Recalculate text_length based on actual text
        df['text_length'] = df['conversation_text'].str.len()
        df['word_count'] = df['conversation_text'].apply(lambda x: len(x.split()) if x else 0)
        df['line_count'] = df['conversation_text'].apply(lambda x: len(x.split('\n')) if x else 1)
        
        # Add derived quality metrics
        df['avg_words_per_line'] = df['word_count'] / df['line_count']
        df['chars_per_word'] = df['text_length'] / df['word_count'].replace(0, 1)  # Avoid division by zero
        
        # Filter out empty conversations
        df = df[df['conversation_text'].str.len() > 10]
        
        return df
    
    def _extract_text_from_json(self, json_str: str) -> str:
        """Extract readable text from conversations JSON"""
        try:
            import json
            conversations = json.loads(json_str)
            
            if isinstance(conversations, list):
                text_parts = []
                for turn in conversations:
                    if isinstance(turn, dict):
                        for role, content in turn.items():
                            text_parts.append(f"{role}: {content}")
                    else:
                        text_parts.append(str(turn))
                return '\n'.join(text_parts)
            else:
                return str(conversations)
        except:
            return json_str  # Return original if parsing fails
    
    def _analyze_response_quality(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze response quality patterns"""
        print("📝 Analyzing response quality patterns...")
        
        quality_scores = []
        response_patterns = defaultdict(list)
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Calculate readability scores
            try:
                flesch_score = flesch_reading_ease(text)
                fk_grade = flesch_kincaid_grade(text)
            except:
                flesch_score = 50  # Default moderate score
                fk_grade = 8
            
            # Analyze response characteristics
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Question patterns
            question_count = text.count('?')
            question_ratio = question_count / len(sentences) if sentences else 0
            
            # Emotional indicators
            positive_words = len(re.findall(r'\b(good|great|excellent|wonderful|amazing|helpful|thank|appreciate)\b', text.lower()))
            negative_words = len(re.findall(r'\b(bad|terrible|awful|horrible|hate|angry|frustrated|disappointed)\b', text.lower()))
            
            # Empathy indicators
            empathy_words = len(re.findall(r'\b(understand|feel|sorry|empathize|support|care|concern)\b', text.lower()))
            
            quality_score = {
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'readability_score': flesch_score,
                'grade_level': fk_grade,
                'avg_sentence_length': avg_sentence_length,
                'question_ratio': question_ratio,
                'positive_sentiment': positive_words / conv['word_count'] * 100,
                'negative_sentiment': negative_words / conv['word_count'] * 100,
                'empathy_score': empathy_words / conv['word_count'] * 100,
                'overall_quality': self._calculate_overall_quality_score(
                    flesch_score, avg_sentence_length, question_ratio, 
                    positive_words, negative_words, empathy_words, conv['word_count']
                )
            }
            
            quality_scores.append(quality_score)
            response_patterns[conv['dataset']].append(quality_score)
        
        quality_df = pd.DataFrame(quality_scores)
        
        return {
            'overall_stats': {
                'mean_quality_score': quality_df['overall_quality'].mean(),
                'median_quality_score': quality_df['overall_quality'].median(),
                'quality_std': quality_df['overall_quality'].std(),
                'high_quality_conversations': len(quality_df[quality_df['overall_quality'] > 75]),
                'low_quality_conversations': len(quality_df[quality_df['overall_quality'] < 25])
            },
            'by_dataset': {
                dataset: {
                    'mean_quality': np.mean([q['overall_quality'] for q in scores]),
                    'mean_readability': np.mean([q['readability_score'] for q in scores]),
                    'mean_empathy': np.mean([q['empathy_score'] for q in scores]),
                    'conversation_count': len(scores)
                }
                for dataset, scores in response_patterns.items()
            },
            'quality_distribution': quality_df['overall_quality'].describe().to_dict(),
            'detailed_scores': quality_df.to_dict('records')[:100]  # Sample for detailed analysis
        }
    
    def _analyze_engagement_quality(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement quality patterns"""
        print("🎯 Analyzing engagement quality patterns...")
        
        engagement_metrics = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Dialogue turn analysis
            dialogue_turns = len(re.findall(r'\n\s*[A-Z][^:]*:', text))
            
            # Interactive elements
            questions = text.count('?')
            exclamations = text.count('!')
            
            # Conversational markers
            conversational_markers = len(re.findall(
                r'\b(well|so|now|then|actually|really|you know|I mean|let me|how about)\b', 
                text.lower()
            ))
            
            # Personal pronouns (engagement indicators)
            personal_pronouns = len(re.findall(r'\b(I|you|we|us|your|my|our)\b', text.lower()))
            
            # Engagement score calculation
            engagement_score = min(100, (
                (dialogue_turns * 10) +
                (questions * 5) +
                (exclamations * 3) +
                (conversational_markers * 2) +
                (personal_pronouns / conv['word_count'] * 100)
            ))
            
            engagement_metrics.append({
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'dialogue_turns': dialogue_turns,
                'questions': questions,
                'exclamations': exclamations,
                'conversational_markers': conversational_markers,
                'personal_pronouns': personal_pronouns,
                'engagement_score': engagement_score
            })
        
        engagement_df = pd.DataFrame(engagement_metrics)
        
        return {
            'overall_engagement': {
                'mean_engagement_score': engagement_df['engagement_score'].mean(),
                'median_engagement_score': engagement_df['engagement_score'].median(),
                'high_engagement_conversations': len(engagement_df[engagement_df['engagement_score'] > 70]),
                'low_engagement_conversations': len(engagement_df[engagement_df['engagement_score'] < 30])
            },
            'engagement_by_dataset': engagement_df.groupby('dataset')['engagement_score'].agg(['mean', 'median', 'std', 'count']).to_dict('index'),
            'engagement_by_tier': engagement_df.groupby('tier')['engagement_score'].agg(['mean', 'median', 'std', 'count']).to_dict('index'),
            'engagement_patterns': {
                'avg_dialogue_turns': engagement_df['dialogue_turns'].mean(),
                'avg_questions_per_conversation': engagement_df['questions'].mean(),
                'avg_conversational_markers': engagement_df['conversational_markers'].mean()
            }
        }
    
    def _analyze_conversation_flow(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conversation flow quality patterns"""
        print("🌊 Analyzing conversation flow patterns...")
        
        flow_metrics = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Flow continuity indicators
            transition_words = len(re.findall(
                r'\b(however|therefore|moreover|furthermore|additionally|meanwhile|consequently|thus|hence)\b',
                text.lower()
            ))
            
            # Coherence indicators
            repetitive_phrases = self._detect_repetitive_phrases(text)
            
            # Topic consistency (simplified)
            topic_shifts = self._detect_topic_shifts(lines)
            
            # Response appropriateness
            response_relevance = self._calculate_response_relevance(lines)
            
            flow_score = min(100, (
                (transition_words * 5) +
                (50 - repetitive_phrases * 10) +  # Penalize repetition
                (50 - topic_shifts * 5) +  # Penalize abrupt topic shifts
                response_relevance
            ))
            
            flow_metrics.append({
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'transition_words': transition_words,
                'repetitive_phrases': repetitive_phrases,
                'topic_shifts': topic_shifts,
                'response_relevance': response_relevance,
                'flow_score': max(0, flow_score)
            })
        
        flow_df = pd.DataFrame(flow_metrics)
        
        return {
            'overall_flow': {
                'mean_flow_score': flow_df['flow_score'].mean(),
                'median_flow_score': flow_df['flow_score'].median(),
                'good_flow_conversations': len(flow_df[flow_df['flow_score'] > 60]),
                'poor_flow_conversations': len(flow_df[flow_df['flow_score'] < 30])
            },
            'flow_by_dataset': flow_df.groupby('dataset')['flow_score'].agg(['mean', 'median', 'std', 'count']).to_dict('index'),
            'flow_patterns': {
                'avg_transition_words': flow_df['transition_words'].mean(),
                'avg_topic_shifts': flow_df['topic_shifts'].mean(),
                'avg_response_relevance': flow_df['response_relevance'].mean()
            }
        }
    
    def _analyze_content_quality(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content quality patterns"""
        print("📚 Analyzing content quality patterns...")
        
        content_metrics = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Information density
            unique_words = len(set(text.lower().split()))
            vocabulary_richness = unique_words / conv['word_count'] if conv['word_count'] > 0 else 0
            
            # Specificity indicators
            specific_terms = len(re.findall(r'\b\d+\b', text))  # Numbers
            proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
            
            # Depth indicators
            explanatory_phrases = len(re.findall(
                r'\b(because|since|due to|as a result|for example|such as|in other words)\b',
                text.lower()
            ))
            
            # Professional language
            professional_terms = len(re.findall(
                r'\b(analysis|evaluation|assessment|methodology|implementation|optimization)\b',
                text.lower()
            ))
            
            content_score = min(100, (
                (vocabulary_richness * 50) +
                (specific_terms * 2) +
                (explanatory_phrases * 3) +
                (professional_terms * 2)
            ))
            
            content_metrics.append({
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'vocabulary_richness': vocabulary_richness,
                'specific_terms': specific_terms,
                'explanatory_phrases': explanatory_phrases,
                'professional_terms': professional_terms,
                'content_score': content_score
            })
        
        content_df = pd.DataFrame(content_metrics)
        
        return {
            'overall_content': {
                'mean_content_score': content_df['content_score'].mean(),
                'median_content_score': content_df['content_score'].median(),
                'high_content_quality': len(content_df[content_df['content_score'] > 70]),
                'low_content_quality': len(content_df[content_df['content_score'] < 30])
            },
            'content_by_dataset': content_df.groupby('dataset')['content_score'].agg(['mean', 'median', 'std', 'count']).to_dict('index'),
            'vocabulary_patterns': {
                'avg_vocabulary_richness': content_df['vocabulary_richness'].mean(),
                'avg_specific_terms': content_df['specific_terms'].mean(),
                'avg_explanatory_phrases': content_df['explanatory_phrases'].mean()
            }
        }
    
    def _analyze_coherence_quality(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze coherence quality patterns"""
        print("🧩 Analyzing coherence quality patterns...")
        
        coherence_metrics = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Logical connectors
            logical_connectors = len(re.findall(
                r'\b(first|second|third|finally|in conclusion|to summarize|on the other hand)\b',
                text.lower()
            ))
            
            # Pronoun reference consistency
            pronoun_consistency = self._check_pronoun_consistency(text)
            
            # Sentence structure variety
            sentence_variety = self._calculate_sentence_variety(sentences)
            
            # Temporal consistency
            temporal_consistency = self._check_temporal_consistency(text)
            
            coherence_score = min(100, (
                (logical_connectors * 8) +
                (pronoun_consistency * 20) +
                (sentence_variety * 15) +
                (temporal_consistency * 20)
            ))
            
            coherence_metrics.append({
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'logical_connectors': logical_connectors,
                'pronoun_consistency': pronoun_consistency,
                'sentence_variety': sentence_variety,
                'temporal_consistency': temporal_consistency,
                'coherence_score': coherence_score
            })
        
        coherence_df = pd.DataFrame(coherence_metrics)
        
        return {
            'overall_coherence': {
                'mean_coherence_score': coherence_df['coherence_score'].mean(),
                'median_coherence_score': coherence_df['coherence_score'].median(),
                'high_coherence_conversations': len(coherence_df[coherence_df['coherence_score'] > 70]),
                'low_coherence_conversations': len(coherence_df[coherence_df['coherence_score'] < 30])
            },
            'coherence_by_dataset': coherence_df.groupby('dataset')['coherence_score'].agg(['mean', 'median', 'std', 'count']).to_dict('index'),
            'coherence_patterns': {
                'avg_logical_connectors': coherence_df['logical_connectors'].mean(),
                'avg_sentence_variety': coherence_df['sentence_variety'].mean(),
                'avg_temporal_consistency': coherence_df['temporal_consistency'].mean()
            }
        }
    
    def _identify_quality_patterns(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Identify recurring quality patterns across conversations"""
        print("🔍 Identifying quality patterns...")
        
        patterns = {
            'high_quality_characteristics': [],
            'low_quality_characteristics': [],
            'dataset_specific_patterns': {},
            'tier_specific_patterns': {}
        }
        
        # Analyze patterns by dataset
        for dataset in conversations['dataset'].unique():
            dataset_convs = conversations[conversations['dataset'] == dataset]
            
            # Calculate average metrics for this dataset
            avg_length = dataset_convs['text_length'].mean()
            avg_words = dataset_convs['word_count'].mean()
            avg_lines = dataset_convs['line_count'].mean()
            
            patterns['dataset_specific_patterns'][dataset] = {
                'avg_conversation_length': avg_length,
                'avg_word_count': avg_words,
                'avg_line_count': avg_lines,
                'conversation_count': len(dataset_convs),
                'quality_indicators': self._extract_quality_indicators(dataset_convs)
            }
        
        # Analyze patterns by tier
        for tier in conversations['tier'].unique():
            tier_convs = conversations[conversations['tier'] == tier]
            
            patterns['tier_specific_patterns'][tier] = {
                'avg_conversation_length': tier_convs['text_length'].mean(),
                'avg_word_count': tier_convs['word_count'].mean(),
                'conversation_count': len(tier_convs),
                'quality_characteristics': self._extract_tier_characteristics(tier_convs)
            }
        
        return patterns
    
    def _analyze_quality_trends(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality trends over time and across datasets"""
        print("📈 Analyzing quality trends...")
        
        # Convert created_at to datetime if it exists
        if 'created_at' in conversations.columns:
            conversations['created_at'] = pd.to_datetime(conversations['created_at'], errors='coerce')
            conversations = conversations.dropna(subset=['created_at'])
            
            # Group by month to see trends
            conversations['month'] = conversations['created_at'].dt.to_period('M')
            monthly_trends = conversations.groupby('month').agg({
                'text_length': 'mean',
                'word_count': 'mean',
                'line_count': 'mean'
            }).to_dict('index')
        else:
            monthly_trends = {}
        
        # Quality trends by dataset size
        dataset_sizes = conversations.groupby('dataset').size().sort_values(ascending=False)
        
        return {
            'monthly_trends': {str(k): v for k, v in monthly_trends.items()},
            'dataset_size_ranking': dataset_sizes.to_dict(),
            'quality_correlation_with_size': self._analyze_size_quality_correlation(conversations)
        }
    
    def _analyze_quality_correlations(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different quality metrics"""
        print("🔗 Analyzing quality correlations...")
        
        # Create correlation matrix for numerical columns
        numerical_cols = ['text_length', 'word_count', 'line_count', 'avg_words_per_line', 'chars_per_word']
        correlation_matrix = conversations[numerical_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'quality_insights': self._interpret_correlations(strong_correlations)
        }
    
    def _generate_quality_insights(self, quality_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from quality analysis"""
        insights = []
        
        # Response quality insights
        response_quality = quality_results['response_quality']
        if response_quality['overall_stats']['mean_quality_score'] < 50:
            insights.append("⚠️ Overall response quality is below average - focus on improving readability and empathy")
        
        # Engagement insights
        engagement_quality = quality_results['engagement_quality']
        if engagement_quality['overall_engagement']['mean_engagement_score'] < 40:
            insights.append("📉 Low engagement detected - increase interactive elements and personal pronouns")
        
        # Flow insights
        flow_quality = quality_results['flow_quality']
        if flow_quality['overall_flow']['mean_flow_score'] < 45:
            insights.append("🌊 Conversation flow needs improvement - add more transition words and reduce topic shifts")
        
        # Content insights
        content_quality = quality_results['content_quality']
        if content_quality['overall_content']['mean_content_score'] < 35:
            insights.append("📚 Content quality is low - increase vocabulary richness and explanatory phrases")
        
        # Dataset-specific insights
        best_dataset = max(response_quality['by_dataset'].items(), key=lambda x: x[1]['mean_quality'])
        worst_dataset = min(response_quality['by_dataset'].items(), key=lambda x: x[1]['mean_quality'])
        
        insights.append(f"🏆 Best performing dataset: {best_dataset[0]} (Quality: {best_dataset[1]['mean_quality']:.1f})")
        insights.append(f"⚡ Needs improvement: {worst_dataset[0]} (Quality: {worst_dataset[1]['mean_quality']:.1f})")
        
        return insights
    
    def _generate_quality_recommendations(self, quality_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for quality improvement"""
        recommendations = []
        
        # Response quality recommendations
        response_stats = quality_results['response_quality']['overall_stats']
        if response_stats['low_quality_conversations'] > response_stats['high_quality_conversations']:
            recommendations.append("🎯 Prioritize improving low-quality conversations through better empathy training")
            recommendations.append("📖 Implement readability guidelines to improve comprehension")
        
        # Engagement recommendations
        engagement_stats = quality_results['engagement_quality']['overall_engagement']
        if engagement_stats['low_engagement_conversations'] > 100:
            recommendations.append("💬 Increase dialogue turns and interactive elements in conversations")
            recommendations.append("❓ Train models to ask more engaging questions")
        
        # Flow recommendations
        flow_stats = quality_results['flow_quality']['overall_flow']
        if flow_stats['poor_flow_conversations'] > 50:
            recommendations.append("🔗 Improve conversation coherence with better transition phrases")
            recommendations.append("🎭 Reduce repetitive patterns and topic inconsistencies")
        
        # Content recommendations
        content_stats = quality_results['content_quality']['overall_content']
        if content_stats['low_content_quality'] > 75:
            recommendations.append("📝 Enhance vocabulary diversity and technical depth")
            recommendations.append("🔍 Include more specific examples and explanatory content")
        
        return recommendations
    
    def _create_quality_visualizations(self, quality_results: Dict[str, Any]):
        """Create visualizations for quality analysis"""
        print("📊 Creating quality visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Conversation Quality Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall Quality Distribution
        response_quality = quality_results['response_quality']
        quality_scores = [item['overall_quality'] for item in response_quality['detailed_scores']]
        
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'Mean: {np.mean(quality_scores):.1f}')
        axes[0, 0].legend()
        
        # 2. Quality by Dataset
        dataset_quality = response_quality['by_dataset']
        datasets = list(dataset_quality.keys())
        quality_means = [dataset_quality[d]['mean_quality'] for d in datasets]
        
        axes[0, 1].bar(range(len(datasets)), quality_means, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Quality by Dataset')
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Average Quality Score')
        axes[0, 1].set_xticks(range(len(datasets)))
        axes[0, 1].set_xticklabels([d[:10] + '...' if len(d) > 10 else d for d in datasets], rotation=45)
        
        # 3. Engagement vs Quality Scatter
        engagement_data = quality_results['engagement_quality']
        if 'engagement_by_dataset' in engagement_data:
            eng_datasets = list(engagement_data['engagement_by_dataset'].keys())
            eng_scores = [engagement_data['engagement_by_dataset'][d]['mean'] for d in eng_datasets]
            qual_scores = [dataset_quality.get(d, {}).get('mean_quality', 0) for d in eng_datasets]
            
            axes[0, 2].scatter(eng_scores, qual_scores, alpha=0.7, s=100, color='green')
            axes[0, 2].set_title('Engagement vs Quality Correlation')
            axes[0, 2].set_xlabel('Engagement Score')
            axes[0, 2].set_ylabel('Quality Score')
            
            # Add trend line
            if len(eng_scores) > 1:
                z = np.polyfit(eng_scores, qual_scores, 1)
                p = np.poly1d(z)
                axes[0, 2].plot(eng_scores, p(eng_scores), "r--", alpha=0.8)
        
        # 4. Quality Metrics Comparison
        metrics = ['Response', 'Engagement', 'Flow', 'Content', 'Coherence']
        metric_scores = [
            response_quality['overall_stats']['mean_quality_score'],
            quality_results['engagement_quality']['overall_engagement']['mean_engagement_score'],
            quality_results['flow_quality']['overall_flow']['mean_flow_score'],
            quality_results['content_quality']['overall_content']['mean_content_score'],
            quality_results['coherence_quality']['overall_coherence']['mean_coherence_score']
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = axes[1, 0].bar(metrics, metric_scores, color=colors, alpha=0.8)
        axes[1, 0].set_title('Quality Metrics Comparison')
        axes[1, 0].set_ylabel('Average Score')
        axes[1, 0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, metric_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{score:.1f}', ha='center', va='bottom')
        
        # 5. Quality Trends (if available)
        trends = quality_results['quality_trends']
        if trends['monthly_trends']:
            months = list(trends['monthly_trends'].keys())
            avg_lengths = [trends['monthly_trends'][m]['text_length'] for m in months]
            
            axes[1, 1].plot(range(len(months)), avg_lengths, marker='o', linewidth=2, markersize=6)
            axes[1, 1].set_title('Quality Trends Over Time')
            axes[1, 1].set_xlabel('Time Period')
            axes[1, 1].set_ylabel('Average Text Length')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels([m[:7] for m in months], rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No temporal data available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Quality Trends Over Time')
        
        # 6. Quality Distribution by Tier
        if 'engagement_by_tier' in quality_results['engagement_quality']:
            tier_data = quality_results['engagement_quality']['engagement_by_tier']
            tiers = list(tier_data.keys())
            tier_scores = [tier_data[t]['mean'] for t in tiers]
            
            axes[1, 2].pie(tier_scores, labels=tiers, autopct='%1.1f%%', startangle=90)
            axes[1, 2].set_title('Quality Distribution by Tier')
        else:
            axes[1, 2].text(0.5, 0.5, 'No tier data available', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Quality Distribution by Tier')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'/home/vivi/pixelated/ai/monitoring/quality_pattern_analysis_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Quality analysis visualizations saved as quality_pattern_analysis_{timestamp}.png")
    
    # Helper methods
    def _calculate_overall_quality_score(self, flesch_score, avg_sentence_length, question_ratio, 
                                       positive_words, negative_words, empathy_words, word_count):
        """Calculate overall quality score from individual metrics"""
        # Normalize flesch score (0-100 scale)
        readability_component = max(0, min(100, flesch_score))
        
        # Sentence length component (optimal around 15-20 words)
        sentence_length_component = max(0, 100 - abs(avg_sentence_length - 17.5) * 3)
        
        # Question engagement component
        question_component = min(100, question_ratio * 200)
        
        # Sentiment balance component
        sentiment_balance = max(0, 50 + (positive_words - negative_words) * 2)
        
        # Empathy component
        empathy_component = min(100, empathy_words / word_count * 1000)
        
        # Weighted average
        overall_score = (
            readability_component * 0.25 +
            sentence_length_component * 0.20 +
            question_component * 0.15 +
            sentiment_balance * 0.25 +
            empathy_component * 0.15
        )
        
        return min(100, max(0, overall_score))
    
    def _detect_repetitive_phrases(self, text: str) -> int:
        """Detect repetitive phrases in text"""
        words = text.lower().split()
        phrase_counts = Counter()
        
        # Check for 3-word phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] += 1
        
        # Count phrases that appear more than once
        repetitive_count = sum(1 for count in phrase_counts.values() if count > 1)
        return repetitive_count
    
    def _detect_topic_shifts(self, lines: List[str]) -> int:
        """Detect abrupt topic shifts in conversation"""
        if len(lines) < 2:
            return 0
        
        # Simple topic shift detection based on vocabulary overlap
        topic_shifts = 0
        for i in range(1, len(lines)):
            prev_words = set(lines[i-1].lower().split())
            curr_words = set(lines[i].lower().split())
            
            # Calculate overlap
            overlap = len(prev_words.intersection(curr_words))
            total_words = len(prev_words.union(curr_words))
            
            if total_words > 0:
                overlap_ratio = overlap / total_words
                if overlap_ratio < 0.1:  # Very low overlap indicates topic shift
                    topic_shifts += 1
        
        return topic_shifts
    
    def _calculate_response_relevance(self, lines: List[str]) -> float:
        """Calculate response relevance score"""
        if len(lines) < 2:
            return 50  # Default score for single-line conversations
        
        relevance_scores = []
        for i in range(1, len(lines)):
            prev_line = lines[i-1].lower()
            curr_line = lines[i].lower()
            
            # Check for question-answer patterns
            if '?' in prev_line and any(word in curr_line for word in ['yes', 'no', 'because', 'since']):
                relevance_scores.append(80)
            # Check for continuation words
            elif any(word in curr_line for word in ['and', 'also', 'furthermore', 'additionally']):
                relevance_scores.append(70)
            # Check for contrast words
            elif any(word in curr_line for word in ['but', 'however', 'although', 'despite']):
                relevance_scores.append(75)
            else:
                relevance_scores.append(50)  # Default relevance
        
        return np.mean(relevance_scores) if relevance_scores else 50
    
    def _check_pronoun_consistency(self, text: str) -> float:
        """Check pronoun reference consistency"""
        pronouns = re.findall(r'\b(he|she|it|they|him|her|them|his|hers|its|their)\b', text.lower())
        if not pronouns:
            return 1.0  # No pronouns to check
        
        # Simple consistency check - count pronoun types
        pronoun_types = set(pronouns)
        consistency_score = 1.0 - (len(pronoun_types) - 1) * 0.1  # Penalize mixed pronouns
        return max(0, consistency_score)
    
    def _calculate_sentence_variety(self, sentences: List[str]) -> float:
        """Calculate sentence structure variety"""
        if not sentences:
            return 0
        
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0
        
        # Calculate coefficient of variation as variety measure
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        if mean_length == 0:
            return 0
        
        variety_score = min(1.0, std_length / mean_length)
        return variety_score
    
    def _check_temporal_consistency(self, text: str) -> float:
        """Check temporal consistency in text"""
        # Look for temporal markers
        past_markers = len(re.findall(r'\b(was|were|had|did|yesterday|ago|before|earlier)\b', text.lower()))
        present_markers = len(re.findall(r'\b(is|are|am|do|does|now|today|currently)\b', text.lower()))
        future_markers = len(re.findall(r'\b(will|shall|going to|tomorrow|later|soon)\b', text.lower()))
        
        total_markers = past_markers + present_markers + future_markers
        if total_markers == 0:
            return 0.5  # Neutral score if no temporal markers
        
        # Calculate consistency (dominant tense should be > 60% of markers)
        max_markers = max(past_markers, present_markers, future_markers)
        consistency = max_markers / total_markers
        
        return consistency
    
    def _extract_quality_indicators(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Extract quality indicators for a dataset"""
        return {
            'avg_text_length': conversations['text_length'].mean(),
            'avg_word_count': conversations['word_count'].mean(),
            'length_std': conversations['text_length'].std(),
            'word_count_std': conversations['word_count'].std(),
            'conversation_count': len(conversations)
        }
    
    def _extract_tier_characteristics(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Extract characteristics for a specific tier"""
        return {
            'avg_text_length': conversations['text_length'].mean(),
            'avg_word_count': conversations['word_count'].mean(),
            'avg_words_per_line': conversations['avg_words_per_line'].mean(),
            'conversation_count': len(conversations)
        }
    
    def _analyze_size_quality_correlation(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between dataset size and quality"""
        dataset_stats = conversations.groupby('dataset').agg({
            'text_length': ['mean', 'count'],
            'word_count': 'mean'
        }).round(2)
        
        # Flatten column names
        dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns]
        
        return dataset_stats.to_dict('index')
    
    def _interpret_correlations(self, correlations: List[Dict]) -> List[str]:
        """Interpret correlation findings"""
        interpretations = []
        
        for corr in correlations:
            if corr['correlation'] > 0.8:
                interpretations.append(f"Strong positive correlation between {corr['metric1']} and {corr['metric2']}")
            elif corr['correlation'] < -0.8:
                interpretations.append(f"Strong negative correlation between {corr['metric1']} and {corr['metric2']}")
        
        return interpretations

def main():
    """Main execution function"""
    print("🚀 Starting Conversation Quality Pattern Analysis System")
    print("=" * 60)
    
    analyzer = ConversationQualityPatternAnalyzer()
    
    try:
        # Run the complete analysis
        results = analyzer.analyze_quality_patterns()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/vivi/pixelated/ai/monitoring/quality_pattern_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        print(f"📊 Total conversations analyzed: {results['total_conversations']}")
        print(f"🎯 Generated {len(results['insights'])} insights")
        print(f"💡 Generated {len(results['recommendations'])} recommendations")
        
        # Display key insights
        print("\n🔍 Key Quality Insights:")
        for insight in results['insights'][:5]:
            print(f"  • {insight}")
        
        print("\n💡 Top Recommendations:")
        for rec in results['recommendations'][:3]:
            print(f"  • {rec}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
