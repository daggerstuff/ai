#!/usr/bin/env python3
"""
Conversation Diversity and Coverage Analyzer
Task 5.6.3.7: Build conversation diversity and coverage analysis

Analyzes diversity and coverage patterns across conversations:
- Topic diversity and distribution
- Vocabulary diversity and richness
- Response pattern diversity
- Dataset coverage analysis
- Conversation style diversity
- Content gap identification
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ConversationDiversityCoverageAnalyzer:
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        self.db_path = db_path
        self.diversity_metrics = {}
        self.coverage_analysis = {}
        self.insights = []
        
    def connect_db(self):
        """Connect to the conversations database"""
        return sqlite3.connect(self.db_path)
    
    def analyze_diversity_coverage(self) -> Dict[str, Any]:
        """Main analysis function for conversation diversity and coverage"""
        print("🌈 Starting Conversation Diversity and Coverage Analysis...")
        
        # Load conversation data
        conversations = self._load_conversation_data()
        print(f"📊 Loaded {len(conversations)} conversations for analysis")
        
        # Analyze different diversity dimensions
        diversity_results = {
            'vocabulary_diversity': self._analyze_vocabulary_diversity(conversations),
            'topic_diversity': self._analyze_topic_diversity(conversations),
            'style_diversity': self._analyze_style_diversity(conversations),
            'response_pattern_diversity': self._analyze_response_pattern_diversity(conversations),
            'dataset_coverage': self._analyze_dataset_coverage(conversations),
            'content_gaps': self._identify_content_gaps(conversations),
            'diversity_trends': self._analyze_diversity_trends(conversations),
            'coverage_completeness': self._assess_coverage_completeness(conversations)
        }
        
        # Generate insights and recommendations
        insights = self._generate_diversity_insights(diversity_results)
        
        # Create visualizations
        self._create_diversity_visualizations(diversity_results)
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'diversity_analysis': diversity_results,
            'insights': insights,
            'recommendations': self._generate_diversity_recommendations(diversity_results)
        }
    
    def _load_conversation_data(self) -> pd.DataFrame:
        """Load conversation data for diversity analysis"""
        with self.connect_db() as conn:
            query = """
            SELECT 
                conversation_id, dataset_source as dataset, tier, conversations_json, 
                character_count as text_length,
                word_count,
                turn_count,
                created_at
            FROM conversations 
            WHERE conversations_json IS NOT NULL 
            AND length(conversations_json) > 10
            """
            df = pd.read_sql_query(query, conn)
            
        # Extract conversation text from JSON
        df['conversation_text'] = df['conversations_json'].apply(self._extract_text_from_json)
        
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
            return json_str
    
    def _analyze_vocabulary_diversity(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze vocabulary diversity across conversations"""
        print("📚 Analyzing vocabulary diversity...")
        
        vocabulary_stats = {}
        dataset_vocabularies = defaultdict(set)
        tier_vocabularies = defaultdict(set)
        
        # Global vocabulary analysis
        all_words = set()
        word_frequencies = Counter()
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text'].lower()
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            # Update global vocabulary
            all_words.update(words)
            word_frequencies.update(words)
            
            # Update dataset vocabularies
            dataset_vocabularies[conv['dataset']].update(words)
            tier_vocabularies[conv['tier']].update(words)
        
        # Calculate vocabulary diversity metrics
        vocabulary_stats = {
            'total_unique_words': len(all_words),
            'total_word_instances': sum(word_frequencies.values()),
            'vocabulary_richness': len(all_words) / sum(word_frequencies.values()) if sum(word_frequencies.values()) > 0 else 0,
            'most_common_words': word_frequencies.most_common(20),
            'rare_words_count': sum(1 for count in word_frequencies.values() if count == 1),
            'rare_words_percentage': sum(1 for count in word_frequencies.values() if count == 1) / len(all_words) * 100 if len(all_words) > 0 else 0
        }
        
        # Dataset vocabulary diversity
        dataset_vocab_stats = {}
        for dataset, vocab in dataset_vocabularies.items():
            dataset_vocab_stats[dataset] = {
                'unique_words': len(vocab),
                'vocabulary_overlap_with_global': len(vocab.intersection(all_words)) / len(all_words) * 100 if len(all_words) > 0 else 0,
                'unique_to_dataset': len(vocab - (all_words - vocab)),
                'conversation_count': len(conversations[conversations['dataset'] == dataset])
            }
        
        # Tier vocabulary diversity
        tier_vocab_stats = {}
        for tier, vocab in tier_vocabularies.items():
            tier_vocab_stats[tier] = {
                'unique_words': len(vocab),
                'vocabulary_overlap_with_global': len(vocab.intersection(all_words)) / len(all_words) * 100 if len(all_words) > 0 else 0,
                'conversation_count': len(conversations[conversations['tier'] == tier])
            }
        
        # Calculate vocabulary diversity index (Simpson's diversity)
        total_words = sum(word_frequencies.values())
        simpson_diversity = 1 - sum((count/total_words)**2 for count in word_frequencies.values())
        
        return {
            'global_vocabulary': vocabulary_stats,
            'dataset_vocabulary': dataset_vocab_stats,
            'tier_vocabulary': tier_vocab_stats,
            'diversity_indices': {
                'simpson_diversity': simpson_diversity,
                'shannon_entropy': self._calculate_shannon_entropy(word_frequencies)
            }
        }
    
    def _analyze_topic_diversity(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze topic diversity using TF-IDF and clustering"""
        print("🎯 Analyzing topic diversity...")
        
        # Prepare text data
        texts = conversations['conversation_text'].tolist()
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Topic clustering
            n_clusters = min(10, len(texts) // 100)  # Adaptive cluster count
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Analyze clusters
                cluster_analysis = {}
                for i in range(n_clusters):
                    cluster_docs = [j for j, label in enumerate(cluster_labels) if label == i]
                    cluster_center = kmeans.cluster_centers_[i]
                    
                    # Get top terms for this cluster
                    top_indices = cluster_center.argsort()[-10:][::-1]
                    top_terms = [feature_names[idx] for idx in top_indices]
                    
                    cluster_analysis[f'cluster_{i}'] = {
                        'document_count': len(cluster_docs),
                        'percentage': len(cluster_docs) / len(texts) * 100,
                        'top_terms': top_terms,
                        'datasets_in_cluster': list(conversations.iloc[cluster_docs]['dataset'].value_counts().to_dict().keys())[:5]
                    }
            else:
                cluster_analysis = {'single_cluster': {'document_count': len(texts), 'percentage': 100}}
            
            # Calculate topic diversity metrics
            topic_diversity = {
                'cluster_count': n_clusters,
                'cluster_analysis': cluster_analysis,
                'topic_distribution_entropy': self._calculate_cluster_entropy(cluster_labels) if n_clusters > 1 else 0,
                'average_cluster_size': len(texts) / n_clusters if n_clusters > 0 else len(texts)
            }
            
        except Exception as e:
            print(f"Warning: Topic analysis failed: {e}")
            topic_diversity = {
                'cluster_count': 0,
                'cluster_analysis': {},
                'topic_distribution_entropy': 0,
                'error': str(e)
            }
        
        return topic_diversity
    
    def _analyze_style_diversity(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conversation style diversity"""
        print("🎨 Analyzing style diversity...")
        
        style_metrics = []
        
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Style indicators
            question_density = text.count('?') / len(text) * 1000 if len(text) > 0 else 0
            exclamation_density = text.count('!') / len(text) * 1000 if len(text) > 0 else 0
            
            # Sentence length variation
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_lengths = [len(s.split()) for s in sentences]
            sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
            
            # Formality indicators
            formal_words = len(re.findall(r'\b(therefore|however|furthermore|consequently|nevertheless)\b', text.lower()))
            informal_words = len(re.findall(r'\b(yeah|okay|cool|awesome|wow|hey)\b', text.lower()))
            
            # Personal vs impersonal style
            personal_pronouns = len(re.findall(r'\b(I|you|we|my|your|our)\b', text.lower()))
            impersonal_indicators = len(re.findall(r'\b(one|it|there|this|that)\b', text.lower()))
            
            style_metrics.append({
                'conversation_id': conv['conversation_id'],
                'dataset': conv['dataset'],
                'tier': conv['tier'],
                'question_density': question_density,
                'exclamation_density': exclamation_density,
                'sentence_length_variation': sentence_length_std,
                'formality_score': formal_words - informal_words,
                'personal_style_score': personal_pronouns - impersonal_indicators,
                'text_length': len(text)
            })
        
        style_df = pd.DataFrame(style_metrics)
        
        # Calculate style diversity
        style_diversity = {
            'question_density_range': {
                'min': style_df['question_density'].min(),
                'max': style_df['question_density'].max(),
                'std': style_df['question_density'].std()
            },
            'formality_distribution': {
                'formal_conversations': len(style_df[style_df['formality_score'] > 0]),
                'informal_conversations': len(style_df[style_df['formality_score'] < 0]),
                'neutral_conversations': len(style_df[style_df['formality_score'] == 0])
            },
            'personal_style_distribution': {
                'personal_style': len(style_df[style_df['personal_style_score'] > 0]),
                'impersonal_style': len(style_df[style_df['personal_style_score'] < 0]),
                'neutral_style': len(style_df[style_df['personal_style_score'] == 0])
            },
            'style_by_dataset': style_df.groupby('dataset').agg({
                'question_density': 'mean',
                'formality_score': 'mean',
                'personal_style_score': 'mean'
            }).to_dict('index'),
            'style_diversity_index': self._calculate_style_diversity_index(style_df)
        }
        
        return style_diversity
    
    def _analyze_response_pattern_diversity(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze diversity in response patterns"""
        print("🔄 Analyzing response pattern diversity...")
        
        pattern_analysis = {
            'response_length_patterns': {},
            'dialogue_turn_patterns': {},
            'response_structure_patterns': {},
            'interaction_patterns': {}
        }
        
        # Response length patterns
        length_categories = []
        for _, conv in conversations.iterrows():
            text_length = len(conv['conversation_text'])
            if text_length < 100:
                length_categories.append('short')
            elif text_length < 500:
                length_categories.append('medium')
            elif text_length < 1500:
                length_categories.append('long')
            else:
                length_categories.append('very_long')
        
        length_distribution = Counter(length_categories)
        pattern_analysis['response_length_patterns'] = {
            'distribution': dict(length_distribution),
            'diversity_score': len(length_distribution) / 4 * 100  # Max 4 categories
        }
        
        # Dialogue turn patterns
        turn_counts = conversations['turn_count'].tolist()
        turn_distribution = Counter(turn_counts)
        pattern_analysis['dialogue_turn_patterns'] = {
            'distribution': dict(list(turn_distribution.most_common(10))),
            'average_turns': np.mean(turn_counts),
            'turn_diversity': len(turn_distribution)
        }
        
        # Response structure patterns
        structure_patterns = []
        for _, conv in conversations.iterrows():
            text = conv['conversation_text']
            
            # Identify structure patterns
            has_questions = '?' in text
            has_lists = bool(re.search(r'\n\s*[-*•]\s+', text))
            has_numbered_items = bool(re.search(r'\n\s*\d+\.\s+', text))
            has_dialogue = ':' in text
            
            pattern = []
            if has_questions:
                pattern.append('Q')
            if has_lists:
                pattern.append('L')
            if has_numbered_items:
                pattern.append('N')
            if has_dialogue:
                pattern.append('D')
            
            structure_patterns.append(''.join(pattern) if pattern else 'P')  # P for plain text
        
        structure_distribution = Counter(structure_patterns)
        pattern_analysis['response_structure_patterns'] = {
            'distribution': dict(structure_distribution.most_common(10)),
            'pattern_diversity': len(structure_distribution)
        }
        
        return pattern_analysis
    
    def _analyze_dataset_coverage(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze coverage across different datasets"""
        print("📋 Analyzing dataset coverage...")
        
        coverage_analysis = {}
        
        # Dataset distribution
        dataset_counts = conversations['dataset'].value_counts()
        total_conversations = len(conversations)
        
        coverage_analysis['dataset_distribution'] = {
            'counts': dataset_counts.to_dict(),
            'percentages': (dataset_counts / total_conversations * 100).to_dict(),
            'dataset_count': len(dataset_counts),
            'largest_dataset': dataset_counts.index[0],
            'smallest_dataset': dataset_counts.index[-1],
            'size_ratio': dataset_counts.iloc[0] / dataset_counts.iloc[-1] if len(dataset_counts) > 1 else 1
        }
        
        # Tier coverage
        tier_counts = conversations['tier'].value_counts()
        coverage_analysis['tier_coverage'] = {
            'counts': tier_counts.to_dict(),
            'percentages': (tier_counts / total_conversations * 100).to_dict(),
            'tier_count': len(tier_counts),
            'most_represented_tier': tier_counts.index[0],
            'least_represented_tier': tier_counts.index[-1]
        }
        
        # Cross-coverage analysis (dataset x tier)
        cross_coverage = conversations.groupby(['dataset', 'tier']).size().unstack(fill_value=0)
        coverage_analysis['cross_coverage'] = {
            'matrix': cross_coverage.to_dict(),
            'coverage_completeness': (cross_coverage > 0).sum().sum() / (len(dataset_counts) * len(tier_counts)) * 100,
            'empty_combinations': ((cross_coverage == 0).sum().sum()),
            'well_covered_combinations': ((cross_coverage >= 100).sum().sum())
        }
        
        # Content length coverage by dataset
        length_coverage = conversations.groupby('dataset')['text_length'].agg(['mean', 'std', 'min', 'max', 'count'])
        coverage_analysis['length_coverage_by_dataset'] = length_coverage.to_dict('index')
        
        return coverage_analysis
    
    def _identify_content_gaps(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Identify gaps in content coverage"""
        print("🔍 Identifying content gaps...")
        
        gaps_analysis = {
            'vocabulary_gaps': {},
            'topic_gaps': {},
            'style_gaps': {},
            'coverage_gaps': {}
        }
        
        # Vocabulary gaps - identify underrepresented word categories
        all_text = ' '.join(conversations['conversation_text'].tolist()).lower()
        
        # Check for specific domain vocabularies
        domain_vocabularies = {
            'emotional': ['feel', 'emotion', 'happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate'],
            'technical': ['system', 'process', 'method', 'algorithm', 'data', 'analysis', 'implementation'],
            'social': ['friend', 'family', 'relationship', 'community', 'social', 'group', 'team'],
            'health': ['health', 'medical', 'doctor', 'treatment', 'therapy', 'wellness', 'mental'],
            'educational': ['learn', 'study', 'education', 'school', 'knowledge', 'skill', 'training']
        }
        
        vocabulary_coverage = {}
        for domain, words in domain_vocabularies.items():
            found_words = sum(1 for word in words if word in all_text)
            vocabulary_coverage[domain] = {
                'coverage_percentage': found_words / len(words) * 100,
                'missing_words': [word for word in words if word not in all_text],
                'found_words': [word for word in words if word in all_text]
            }
        
        gaps_analysis['vocabulary_gaps'] = vocabulary_coverage
        
        # Length distribution gaps
        length_ranges = {
            'very_short': (0, 50),
            'short': (51, 200),
            'medium': (201, 500),
            'long': (501, 1000),
            'very_long': (1001, float('inf'))
        }
        
        length_distribution = {}
        for range_name, (min_len, max_len) in length_ranges.items():
            count = len(conversations[
                (conversations['text_length'] >= min_len) & 
                (conversations['text_length'] <= max_len)
            ])
            length_distribution[range_name] = {
                'count': count,
                'percentage': count / len(conversations) * 100
            }
        
        gaps_analysis['coverage_gaps']['length_distribution'] = length_distribution
        
        # Dataset-tier combination gaps
        dataset_tier_matrix = conversations.pivot_table(
            index='dataset', 
            columns='tier', 
            values='conversation_id', 
            aggfunc='count', 
            fill_value=0
        )
        
        empty_combinations = []
        sparse_combinations = []
        
        for dataset in dataset_tier_matrix.index:
            for tier in dataset_tier_matrix.columns:
                count = dataset_tier_matrix.loc[dataset, tier]
                if count == 0:
                    empty_combinations.append((dataset, tier))
                elif count < 10:  # Threshold for sparse coverage
                    sparse_combinations.append((dataset, tier, count))
        
        gaps_analysis['coverage_gaps']['empty_combinations'] = empty_combinations
        gaps_analysis['coverage_gaps']['sparse_combinations'] = sparse_combinations
        
        return gaps_analysis
    
    def _analyze_diversity_trends(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Analyze diversity trends over time"""
        print("📈 Analyzing diversity trends...")
        
        trends_analysis = {}
        
        # Convert created_at to datetime if available
        if 'created_at' in conversations.columns:
            conversations['created_at'] = pd.to_datetime(conversations['created_at'], errors='coerce')
            conversations = conversations.dropna(subset=['created_at'])
            
            if len(conversations) > 0:
                # Group by month
                conversations['month'] = conversations['created_at'].dt.to_period('M')
                monthly_stats = conversations.groupby('month').agg({
                    'dataset': 'nunique',
                    'tier': 'nunique',
                    'text_length': ['mean', 'std'],
                    'word_count': 'mean',
                    'conversation_id': 'count'
                })
                
                # Flatten column names
                monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
                
                trends_analysis['monthly_trends'] = {
                    str(month): stats for month, stats in monthly_stats.to_dict('index').items()
                }
                
                # Calculate diversity trend
                monthly_diversity = conversations.groupby('month').apply(
                    lambda x: len(x['dataset'].unique()) * len(x['tier'].unique())
                )
                
                trends_analysis['diversity_trend'] = {
                    'monthly_diversity_scores': {str(k): v for k, v in monthly_diversity.to_dict().items()},
                    'trend_direction': 'increasing' if monthly_diversity.iloc[-1] > monthly_diversity.iloc[0] else 'decreasing' if len(monthly_diversity) > 1 else 'stable'
                }
            else:
                trends_analysis['monthly_trends'] = {}
                trends_analysis['diversity_trend'] = {}
        else:
            trends_analysis = {'no_temporal_data': True}
        
        return trends_analysis
    
    def _assess_coverage_completeness(self, conversations: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall coverage completeness"""
        print("✅ Assessing coverage completeness...")
        
        completeness_analysis = {}
        
        # Dataset coverage completeness
        total_datasets = len(conversations['dataset'].unique())
        total_tiers = len(conversations['tier'].unique())
        total_possible_combinations = total_datasets * total_tiers
        
        actual_combinations = len(conversations.groupby(['dataset', 'tier']).size())
        
        completeness_analysis['combination_coverage'] = {
            'total_possible': total_possible_combinations,
            'actual_combinations': actual_combinations,
            'coverage_percentage': actual_combinations / total_possible_combinations * 100 if total_possible_combinations > 0 else 0
        }
        
        # Content coverage assessment
        total_conversations = len(conversations)
        
        # Size distribution assessment
        size_categories = {
            'small': len(conversations[conversations['text_length'] < 200]),
            'medium': len(conversations[(conversations['text_length'] >= 200) & (conversations['text_length'] < 1000)]),
            'large': len(conversations[conversations['text_length'] >= 1000])
        }
        
        completeness_analysis['size_distribution'] = {
            category: {
                'count': count,
                'percentage': count / total_conversations * 100,
                'adequacy': 'good' if count > total_conversations * 0.2 else 'needs_improvement'
            }
            for category, count in size_categories.items()
        }
        
        # Overall completeness score
        combination_score = actual_combinations / total_possible_combinations if total_possible_combinations > 0 else 0
        size_balance_score = 1 - abs(0.33 - min(size_categories.values()) / total_conversations) * 3  # Penalty for imbalance
        
        completeness_analysis['overall_completeness_score'] = (combination_score + max(0, size_balance_score)) / 2 * 100
        
        return completeness_analysis
    
    def _generate_diversity_insights(self, diversity_results: Dict[str, Any]) -> List[str]:
        """Generate insights from diversity analysis"""
        insights = []
        
        # Vocabulary diversity insights
        vocab_diversity = diversity_results['vocabulary_diversity']
        if vocab_diversity['diversity_indices']['simpson_diversity'] < 0.8:
            insights.append("📚 Vocabulary diversity is low - consider expanding word variety across conversations")
        
        # Topic diversity insights
        topic_diversity = diversity_results['topic_diversity']
        if topic_diversity.get('cluster_count', 0) < 5:
            insights.append("🎯 Limited topic diversity detected - expand conversation themes and subjects")
        
        # Coverage insights
        coverage = diversity_results['dataset_coverage']
        if coverage['dataset_distribution']['size_ratio'] > 10:
            insights.append(f"⚖️ Significant dataset imbalance - largest dataset is {coverage['dataset_distribution']['size_ratio']:.1f}x larger than smallest")
        
        # Content gaps insights
        gaps = diversity_results['content_gaps']
        low_coverage_domains = [domain for domain, data in gaps['vocabulary_gaps'].items() if data['coverage_percentage'] < 50]
        if low_coverage_domains:
            insights.append(f"🔍 Low coverage in domains: {', '.join(low_coverage_domains)}")
        
        # Completeness insights
        completeness = diversity_results['coverage_completeness']
        if completeness['overall_completeness_score'] < 70:
            insights.append(f"📊 Overall coverage completeness is {completeness['overall_completeness_score']:.1f}% - needs improvement")
        
        return insights
    
    def _generate_diversity_recommendations(self, diversity_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving diversity and coverage"""
        recommendations = []
        
        # Vocabulary recommendations
        vocab_diversity = diversity_results['vocabulary_diversity']
        if vocab_diversity['global_vocabulary']['rare_words_percentage'] > 50:
            recommendations.append("📝 High percentage of rare words - balance vocabulary with more common terms")
        
        # Topic recommendations
        topic_diversity = diversity_results['topic_diversity']
        if topic_diversity.get('cluster_count', 0) < 8:
            recommendations.append("🎯 Increase topic diversity by adding conversations on underrepresented themes")
        
        # Style recommendations
        style_diversity = diversity_results['style_diversity']
        formal_ratio = style_diversity['formality_distribution']['formal_conversations'] / (
            style_diversity['formality_distribution']['formal_conversations'] + 
            style_diversity['formality_distribution']['informal_conversations'] + 1
        )
        if formal_ratio > 0.8 or formal_ratio < 0.2:
            recommendations.append("🎨 Balance formal and informal conversation styles for better diversity")
        
        # Coverage recommendations
        gaps = diversity_results['content_gaps']
        if gaps['coverage_gaps']['empty_combinations']:
            recommendations.append(f"📋 Fill {len(gaps['coverage_gaps']['empty_combinations'])} empty dataset-tier combinations")
        
        # Completeness recommendations
        completeness = diversity_results['coverage_completeness']
        if completeness['combination_coverage']['coverage_percentage'] < 80:
            recommendations.append("✅ Improve dataset-tier combination coverage to reach 80%+ completeness")
        
        return recommendations
    
    def _create_diversity_visualizations(self, diversity_results: Dict[str, Any]):
        """Create visualizations for diversity analysis"""
        print("📊 Creating diversity visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Conversation Diversity and Coverage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dataset Distribution
        coverage = diversity_results['dataset_coverage']
        dataset_counts = coverage['dataset_distribution']['counts']
        
        if dataset_counts:
            datasets = list(dataset_counts.keys())[:10]  # Top 10 datasets
            counts = [dataset_counts[d] for d in datasets]
            
            axes[0, 0].bar(range(len(datasets)), counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Dataset Distribution (Top 10)')
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('Conversation Count')
            axes[0, 0].set_xticks(range(len(datasets)))
            axes[0, 0].set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45)
        
        # 2. Vocabulary Diversity by Dataset
        vocab_diversity = diversity_results['vocabulary_diversity']
        dataset_vocab = vocab_diversity['dataset_vocabulary']
        
        if dataset_vocab:
            datasets = list(dataset_vocab.keys())[:8]  # Top 8 for readability
            unique_words = [dataset_vocab[d]['unique_words'] for d in datasets]
            
            axes[0, 1].bar(range(len(datasets)), unique_words, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Vocabulary Diversity by Dataset')
            axes[0, 1].set_xlabel('Dataset')
            axes[0, 1].set_ylabel('Unique Words')
            axes[0, 1].set_xticks(range(len(datasets)))
            axes[0, 1].set_xticklabels([d[:10] + '...' if len(d) > 10 else d for d in datasets], rotation=45)
        
        # 3. Topic Diversity
        topic_diversity = diversity_results['topic_diversity']
        if 'cluster_analysis' in topic_diversity and topic_diversity['cluster_analysis']:
            clusters = list(topic_diversity['cluster_analysis'].keys())
            percentages = [topic_diversity['cluster_analysis'][c]['percentage'] for c in clusters]
            
            axes[0, 2].pie(percentages, labels=[f'Topic {i+1}' for i in range(len(clusters))], 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Topic Distribution')
        else:
            axes[0, 2].text(0.5, 0.5, 'Topic analysis unavailable', ha='center', va='center', 
                           transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Topic Distribution')
        
        # 4. Style Diversity
        style_diversity = diversity_results['style_diversity']
        formality_dist = style_diversity['formality_distribution']
        
        categories = ['Formal', 'Informal', 'Neutral']
        values = [
            formality_dist['formal_conversations'],
            formality_dist['informal_conversations'],
            formality_dist['neutral_conversations']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        axes[1, 0].bar(categories, values, color=colors, alpha=0.8)
        axes[1, 0].set_title('Style Distribution')
        axes[1, 0].set_ylabel('Conversation Count')
        
        # Add value labels
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
        
        # 5. Coverage Completeness
        completeness = diversity_results['coverage_completeness']
        
        # Coverage metrics
        metrics = ['Combination\nCoverage', 'Size\nBalance', 'Overall\nCompleteness']
        scores = [
            completeness['combination_coverage']['coverage_percentage'],
            np.mean([cat['percentage'] for cat in completeness['size_distribution'].values()]),
            completeness['overall_completeness_score']
        ]
        
        bars = axes[1, 1].bar(metrics, scores, color=['#96CEB4', '#FFEAA7', '#DDA0DD'], alpha=0.8)
        axes[1, 1].set_title('Coverage Completeness Metrics')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].set_ylim(0, 100)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{score:.1f}%', ha='center', va='bottom')
        
        # 6. Content Gaps Heatmap
        gaps = diversity_results['content_gaps']
        vocab_gaps = gaps['vocabulary_gaps']
        
        if vocab_gaps:
            domains = list(vocab_gaps.keys())
            coverage_scores = [vocab_gaps[d]['coverage_percentage'] for d in domains]
            
            # Create a simple heatmap-style visualization
            y_pos = np.arange(len(domains))
            bars = axes[1, 2].barh(y_pos, coverage_scores, color='orange', alpha=0.7)
            
            axes[1, 2].set_yticks(y_pos)
            axes[1, 2].set_yticklabels(domains)
            axes[1, 2].set_xlabel('Coverage Percentage')
            axes[1, 2].set_title('Domain Vocabulary Coverage')
            axes[1, 2].set_xlim(0, 100)
            
            # Add percentage labels
            for i, (bar, score) in enumerate(zip(bars, coverage_scores)):
                width = bar.get_width()
                axes[1, 2].text(width + 2, bar.get_y() + bar.get_height()/2,
                               f'{score:.1f}%', ha='left', va='center')
        else:
            axes[1, 2].text(0.5, 0.5, 'No vocabulary gap data', ha='center', va='center',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Domain Vocabulary Coverage')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'/home/vivi/pixelated/ai/monitoring/diversity_coverage_analysis_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Diversity analysis visualizations saved as diversity_coverage_analysis_{timestamp}.png")
    
    # Helper methods
    def _calculate_shannon_entropy(self, word_frequencies: Counter) -> float:
        """Calculate Shannon entropy for vocabulary diversity"""
        total_words = sum(word_frequencies.values())
        if total_words == 0:
            return 0
        
        entropy = 0
        for count in word_frequencies.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_cluster_entropy(self, cluster_labels: List[int]) -> float:
        """Calculate entropy of cluster distribution"""
        cluster_counts = Counter(cluster_labels)
        total_docs = len(cluster_labels)
        
        entropy = 0
        for count in cluster_counts.values():
            probability = count / total_docs
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_style_diversity_index(self, style_df: pd.DataFrame) -> float:
        """Calculate overall style diversity index"""
        # Normalize metrics to 0-1 range
        metrics = ['question_density', 'exclamation_density', 'sentence_length_variation', 
                  'formality_score', 'personal_style_score']
        
        diversity_scores = []
        for metric in metrics:
            if metric in style_df.columns:
                values = style_df[metric].values
                if len(values) > 1 and np.std(values) > 0:
                    # Use coefficient of variation as diversity measure
                    cv = np.std(values) / (np.abs(np.mean(values)) + 1e-6)
                    diversity_scores.append(min(1.0, cv))
                else:
                    diversity_scores.append(0)
        
        return np.mean(diversity_scores) * 100 if diversity_scores else 0

def main():
    """Main execution function"""
    print("🚀 Starting Conversation Diversity and Coverage Analysis System")
    print("=" * 70)
    
    analyzer = ConversationDiversityCoverageAnalyzer()
    
    try:
        # Run the complete analysis
        results = analyzer.analyze_diversity_coverage()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/vivi/pixelated/ai/monitoring/diversity_coverage_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        print(f"📊 Total conversations analyzed: {results['total_conversations']}")
        print(f"🎯 Generated {len(results['insights'])} insights")
        print(f"💡 Generated {len(results['recommendations'])} recommendations")
        
        # Display key insights
        print("\n🌈 Key Diversity Insights:")
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
