#!/usr/bin/env python3
"""
Dataset Performance and Impact Analytics
Task 5.6.3.10: Build dataset performance and impact analytics

Comprehensive analysis of dataset performance and business impact:
- Dataset ROI and value assessment
- Performance benchmarking across datasets
- Impact measurement and attribution
- Resource utilization analysis
- Strategic recommendations for dataset optimization
- Business value quantification
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DatasetPerformanceImpactAnalyzer:
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        self.db_path = db_path
        self.performance_metrics = {}
        self.impact_analysis = {}
        self.roi_calculations = {}
        
    def connect_db(self):
        """Connect to the conversations database"""
        return sqlite3.connect(self.db_path)
    
    def analyze_performance_impact(self) -> Dict[str, Any]:
        """Main function for dataset performance and impact analysis"""
        print("📊 Starting Dataset Performance and Impact Analysis...")
        
        # Load comprehensive dataset information
        datasets_info = self._load_dataset_information()
        print(f"📈 Loaded information for {len(datasets_info)} datasets")
        
        # Analyze performance metrics
        performance_analysis = {
            'quality_performance': self._analyze_quality_performance(datasets_info),
            'efficiency_metrics': self._analyze_efficiency_metrics(datasets_info),
            'utilization_analysis': self._analyze_utilization_patterns(datasets_info),
            'comparative_performance': self._perform_comparative_analysis(datasets_info),
            'trend_analysis': self._analyze_performance_trends(datasets_info)
        }
        
        # Measure business impact
        impact_analysis = {
            'value_contribution': self._measure_value_contribution(datasets_info),
            'roi_assessment': self._calculate_dataset_roi(datasets_info),
            'strategic_impact': self._assess_strategic_impact(datasets_info),
            'resource_efficiency': self._analyze_resource_efficiency(datasets_info),
            'scalability_analysis': self._analyze_scalability_potential(datasets_info)
        }
        
        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            performance_analysis, impact_analysis, datasets_info
        )
        
        # Create comprehensive visualizations
        self._create_performance_impact_visualizations(
            performance_analysis, impact_analysis, datasets_info
        )
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'datasets_analyzed': len(datasets_info),
            'performance_analysis': performance_analysis,
            'impact_analysis': impact_analysis,
            'strategic_recommendations': strategic_recommendations,
            'executive_summary': self._generate_executive_summary(performance_analysis, impact_analysis),
            'action_priorities': self._define_action_priorities(performance_analysis, impact_analysis),
            'success_metrics': self._define_success_metrics()
        }
    
    def _load_dataset_information(self) -> pd.DataFrame:
        """Load comprehensive dataset information with performance indicators"""
        with self.connect_db() as conn:
            query = """
            SELECT 
                dataset_source as dataset,
                tier,
                conversation_id,
                conversations_json,
                character_count as text_length,
                word_count,
                turn_count,
                created_at,
                processing_status,
                processed_at
            FROM conversations 
            WHERE conversations_json IS NOT NULL 
            AND length(conversations_json) > 10
            """
            df = pd.read_sql_query(query, conn)
        
        # Extract conversation text and calculate additional metrics
        df['conversation_text'] = df['conversations_json'].apply(self._extract_text_from_json)
        df['quality_score'] = df.apply(self._calculate_quality_score, axis=1)
        df['complexity_score'] = df.apply(self._calculate_complexity_score, axis=1)
        df['engagement_score'] = df.apply(self._calculate_engagement_score, axis=1)
        
        # Add temporal features
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce', utc=True)
        df['processing_time'] = df['processed_at'] - df['created_at']
        df['processing_time_hours'] = df['processing_time'].dt.total_seconds() / 3600
        
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
    
    def _calculate_quality_score(self, row: pd.Series) -> float:
        """Calculate quality score for a conversation"""
        text = row['conversation_text']
        word_count = row['word_count']
        
        if not text or word_count == 0:
            return 0
        
        # Quality components
        try:
            from textstat import flesch_reading_ease
            readability = max(0, min(100, flesch_reading_ease(text))) / 100
        except:
            readability = 0.5
        
        # Engagement indicators
        questions = text.count('?')
        engagement = min(1.0, questions / word_count * 50)
        
        # Empathy indicators
        empathy_words = len(re.findall(r'\b(understand|feel|sorry|empathize|support|care)\b', text.lower()))
        empathy = min(1.0, empathy_words / word_count * 100)
        
        # Structure indicators
        structure = 0.8 if (bool(re.search(r'\n\s*[-*•]\s+', text)) or 
                           bool(re.search(r'\n\s*\d+\.\s+', text))) else 0.4
        
        # Weighted quality score
        quality_score = (readability * 0.3 + engagement * 0.25 + empathy * 0.25 + structure * 0.2) * 100
        return quality_score
    
    def _calculate_complexity_score(self, row: pd.Series) -> float:
        """Calculate complexity score for a conversation"""
        text = row['conversation_text']
        word_count = row['word_count']
        turn_count = row['turn_count']
        
        if not text or word_count == 0:
            return 0
        
        # Vocabulary complexity
        unique_words = len(set(text.lower().split()))
        vocab_complexity = unique_words / word_count if word_count > 0 else 0
        
        # Sentence complexity
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / sentences if sentences > 0 else 0
        sentence_complexity = min(1.0, avg_sentence_length / 20)
        
        # Dialogue complexity
        dialogue_complexity = min(1.0, turn_count / 10)
        
        # Technical terms
        technical_terms = len(re.findall(r'\b(analysis|evaluation|methodology|implementation|optimization)\b', text.lower()))
        technical_complexity = min(1.0, technical_terms / word_count * 100)
        
        complexity_score = (vocab_complexity * 0.3 + sentence_complexity * 0.25 + 
                          dialogue_complexity * 0.25 + technical_complexity * 0.2) * 100
        return complexity_score
    
    def _calculate_engagement_score(self, row: pd.Series) -> float:
        """Calculate engagement score for a conversation"""
        text = row['conversation_text']
        word_count = row['word_count']
        
        if not text or word_count == 0:
            return 0
        
        # Interactive elements
        questions = text.count('?')
        exclamations = text.count('!')
        
        # Personal engagement
        personal_pronouns = len(re.findall(r'\b(I|you|we|us|your|my|our)\b', text.lower()))
        
        # Conversational markers
        markers = len(re.findall(r'\b(well|so|now|then|actually|really|you know)\b', text.lower()))
        
        engagement_score = min(100, (questions * 10) + (exclamations * 5) + 
                             (personal_pronouns / word_count * 200) + (markers * 3))
        return engagement_score
    
    def _analyze_quality_performance(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality performance across datasets"""
        print("🏆 Analyzing quality performance...")
        
        quality_analysis = {}
        
        # Dataset-level quality metrics
        dataset_quality = datasets_info.groupby('dataset').agg({
            'quality_score': ['mean', 'median', 'std', 'min', 'max'],
            'complexity_score': 'mean',
            'engagement_score': 'mean',
            'conversation_id': 'count'
        }).round(2)
        
        # Flatten column names
        dataset_quality.columns = ['_'.join(col).strip() for col in dataset_quality.columns]
        
        # Rank datasets by quality
        dataset_quality['overall_performance'] = (
            dataset_quality['quality_score_mean'] * 0.5 +
            dataset_quality['complexity_score_mean'] * 0.25 +
            dataset_quality['engagement_score_mean'] * 0.25
        )
        
        dataset_quality_ranked = dataset_quality.sort_values('overall_performance', ascending=False)
        
        quality_analysis['dataset_rankings'] = dataset_quality_ranked.to_dict('index')
        
        # Quality distribution analysis
        quality_analysis['quality_distribution'] = {
            'high_quality_datasets': len(dataset_quality[dataset_quality['quality_score_mean'] >= 70]),
            'medium_quality_datasets': len(dataset_quality[(dataset_quality['quality_score_mean'] >= 50) & 
                                                          (dataset_quality['quality_score_mean'] < 70)]),
            'low_quality_datasets': len(dataset_quality[dataset_quality['quality_score_mean'] < 50]),
            'total_datasets': len(dataset_quality)
        }
        
        # Quality consistency analysis
        quality_analysis['consistency_metrics'] = {
            'most_consistent': dataset_quality.loc[dataset_quality['quality_score_std'].idxmin()].name,
            'least_consistent': dataset_quality.loc[dataset_quality['quality_score_std'].idxmax()].name,
            'avg_quality_variance': dataset_quality['quality_score_std'].mean()
        }
        
        return quality_analysis
    
    def _analyze_efficiency_metrics(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze efficiency metrics for datasets"""
        print("⚡ Analyzing efficiency metrics...")
        
        efficiency_analysis = {}
        
        # Processing efficiency
        processing_efficiency = datasets_info.groupby('dataset').agg({
            'processing_time_hours': ['mean', 'median', 'std'],
            'word_count': 'mean',
            'text_length': 'mean',
            'conversation_id': 'count'
        }).round(3)
        
        processing_efficiency.columns = ['_'.join(col).strip() for col in processing_efficiency.columns]
        
        # Calculate efficiency ratios
        processing_efficiency['words_per_hour'] = (
            processing_efficiency['word_count_mean'] / 
            processing_efficiency['processing_time_hours_mean'].replace(0, 1)
        )
        
        processing_efficiency['conversations_per_day'] = (
            processing_efficiency['conversation_id_count'] / 
            (processing_efficiency['processing_time_hours_mean'] / 24).replace(0, 1)
        )
        
        efficiency_analysis['processing_efficiency'] = processing_efficiency.to_dict('index')
        
        # Resource utilization efficiency
        datasets_info['quality_per_word'] = datasets_info['quality_score'] / datasets_info['word_count']
        datasets_info['engagement_per_turn'] = datasets_info['engagement_score'] / datasets_info['turn_count']
        
        resource_efficiency = datasets_info.groupby('dataset').agg({
            'quality_per_word': 'mean',
            'engagement_per_turn': 'mean'
        }).round(4)
        
        efficiency_analysis['resource_efficiency'] = resource_efficiency.to_dict('index')
        
        # Efficiency rankings
        efficiency_scores = {}
        for dataset in processing_efficiency.index:
            proc_score = 1 / (processing_efficiency.loc[dataset, 'processing_time_hours_mean'] + 0.1)
            quality_score = resource_efficiency.loc[dataset, 'quality_per_word']
            efficiency_scores[dataset] = proc_score * quality_score
        
        efficiency_analysis['efficiency_rankings'] = dict(
            sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return efficiency_analysis
    
    def _analyze_utilization_patterns(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset utilization patterns"""
        print("📈 Analyzing utilization patterns...")
        
        utilization_analysis = {}
        
        # Volume utilization
        dataset_volumes = datasets_info['dataset'].value_counts()
        total_conversations = len(datasets_info)
        
        utilization_analysis['volume_distribution'] = {
            'dataset_volumes': dataset_volumes.to_dict(),
            'volume_percentages': (dataset_volumes / total_conversations * 100).round(2).to_dict(),
            'concentration_ratio': (dataset_volumes.head(3).sum() / total_conversations * 100).round(2)
        }
        
        # Tier utilization by dataset
        tier_utilization = datasets_info.groupby(['dataset', 'tier']).size().unstack(fill_value=0)
        tier_percentages = tier_utilization.div(tier_utilization.sum(axis=1), axis=0) * 100
        
        utilization_analysis['tier_distribution'] = {
            'tier_counts': tier_utilization.to_dict('index'),
            'tier_percentages': tier_percentages.round(2).to_dict('index')
        }
        
        # Temporal utilization patterns
        if 'created_at' in datasets_info.columns:
            datasets_info['month'] = datasets_info['created_at'].dt.to_period('M').astype(str)
            temporal_utilization = datasets_info.groupby(['dataset', 'month']).size().unstack(fill_value=0)
            
            utilization_analysis['temporal_patterns'] = {
                'monthly_activity': temporal_utilization.to_dict('index'),
                'growth_trends': self._calculate_growth_trends(temporal_utilization)
            }
        
        return utilization_analysis
    
    def _perform_comparative_analysis(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Perform comparative analysis across datasets"""
        print("🔍 Performing comparative analysis...")
        
        comparative_analysis = {}
        
        # Statistical comparisons
        datasets = datasets_info['dataset'].unique()
        
        # Quality comparisons
        quality_comparisons = {}
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                data1 = datasets_info[datasets_info['dataset'] == dataset1]['quality_score']
                data2 = datasets_info[datasets_info['dataset'] == dataset2]['quality_score']
                
                # Perform t-test
                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    quality_comparisons[f"{dataset1}_vs_{dataset2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_difference': data1.mean() - data2.mean()
                    }
        
        comparative_analysis['quality_comparisons'] = quality_comparisons
        
        # Performance clustering
        dataset_features = datasets_info.groupby('dataset').agg({
            'quality_score': 'mean',
            'complexity_score': 'mean',
            'engagement_score': 'mean',
            'word_count': 'mean',
            'turn_count': 'mean'
        })
        
        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(dataset_features)
        
        # Perform clustering
        n_clusters = min(5, len(dataset_features))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            # Assign clusters to datasets
            dataset_clusters = {}
            for i, dataset in enumerate(dataset_features.index):
                dataset_clusters[dataset] = int(cluster_labels[i])
            
            comparative_analysis['performance_clusters'] = {
                'cluster_assignments': dataset_clusters,
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
        
        return comparative_analysis
    
    def _analyze_performance_trends(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        print("📊 Analyzing performance trends...")
        
        trends_analysis = {}
        
        if 'created_at' in datasets_info.columns:
            # Monthly performance trends
            datasets_info['month'] = datasets_info['created_at'].dt.to_period('M').astype(str)
            
            monthly_performance = datasets_info.groupby(['dataset', 'month']).agg({
                'quality_score': 'mean',
                'engagement_score': 'mean',
                'complexity_score': 'mean',
                'conversation_id': 'count'
            }).reset_index()
            
            # Calculate trend slopes for each dataset
            trend_slopes = {}
            for dataset in datasets_info['dataset'].unique():
                dataset_data = monthly_performance[monthly_performance['dataset'] == dataset]
                
                if len(dataset_data) > 2:
                    # Calculate trend slopes
                    x = np.arange(len(dataset_data))
                    
                    quality_slope, _ = np.polyfit(x, dataset_data['quality_score'], 1)
                    engagement_slope, _ = np.polyfit(x, dataset_data['engagement_score'], 1)
                    volume_slope, _ = np.polyfit(x, dataset_data['conversation_id'], 1)
                    
                    trend_slopes[dataset] = {
                        'quality_trend': quality_slope,
                        'engagement_trend': engagement_slope,
                        'volume_trend': volume_slope,
                        'overall_trend': (quality_slope + engagement_slope) / 2
                    }
            
            trends_analysis['trend_slopes'] = trend_slopes
            
            # Identify improving and declining datasets
            improving_datasets = [d for d, trends in trend_slopes.items() 
                                if trends['overall_trend'] > 0.1]
            declining_datasets = [d for d, trends in trend_slopes.items() 
                                if trends['overall_trend'] < -0.1]
            
            trends_analysis['performance_trajectory'] = {
                'improving_datasets': improving_datasets,
                'declining_datasets': declining_datasets,
                'stable_datasets': [d for d in trend_slopes.keys() 
                                  if d not in improving_datasets and d not in declining_datasets]
            }
        
        return trends_analysis
    
    def _measure_value_contribution(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Measure value contribution of each dataset"""
        print("💰 Measuring value contribution...")
        
        value_analysis = {}
        
        # Calculate value metrics for each dataset
        dataset_value = datasets_info.groupby('dataset').agg({
            'quality_score': 'mean',
            'engagement_score': 'mean',
            'complexity_score': 'mean',
            'conversation_id': 'count',
            'word_count': 'sum',
            'turn_count': 'sum'
        })
        
        # Calculate value scores
        dataset_value['content_value'] = (
            dataset_value['word_count'] * dataset_value['quality_score'] / 100
        )
        
        dataset_value['engagement_value'] = (
            dataset_value['conversation_id'] * dataset_value['engagement_score'] / 100
        )
        
        dataset_value['complexity_value'] = (
            dataset_value['turn_count'] * dataset_value['complexity_score'] / 100
        )
        
        # Overall value score
        dataset_value['overall_value'] = (
            dataset_value['content_value'] * 0.4 +
            dataset_value['engagement_value'] * 0.35 +
            dataset_value['complexity_value'] * 0.25
        )
        
        # Normalize to percentage of total value
        total_value = dataset_value['overall_value'].sum()
        dataset_value['value_percentage'] = (dataset_value['overall_value'] / total_value * 100).round(2)
        
        value_analysis['dataset_value_scores'] = dataset_value.to_dict('index')
        
        # Value concentration analysis
        top_3_value = dataset_value.nlargest(3, 'overall_value')['value_percentage'].sum()
        value_analysis['value_concentration'] = {
            'top_3_datasets_value_share': top_3_value,
            'value_distribution_balance': 'concentrated' if top_3_value > 70 else 'balanced'
        }
        
        return value_analysis
    
    def _calculate_dataset_roi(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ROI for each dataset"""
        print("📊 Calculating dataset ROI...")
        
        roi_analysis = {}
        
        # Estimate costs (simplified model)
        dataset_costs = datasets_info.groupby('dataset').agg({
            'conversation_id': 'count',
            'word_count': 'sum',
            'processing_time_hours': 'sum'
        })
        
        # Estimate processing costs (hypothetical rates)
        COST_PER_CONVERSATION = 0.10  # $0.10 per conversation
        COST_PER_WORD = 0.001  # $0.001 per word
        COST_PER_HOUR = 5.0  # $5 per processing hour
        
        dataset_costs['estimated_cost'] = (
            dataset_costs['conversation_id'] * COST_PER_CONVERSATION +
            dataset_costs['word_count'] * COST_PER_WORD +
            dataset_costs['processing_time_hours'].fillna(1) * COST_PER_HOUR
        )
        
        # Calculate value (from previous analysis)
        dataset_quality = datasets_info.groupby('dataset')['quality_score'].mean()
        dataset_engagement = datasets_info.groupby('dataset')['engagement_score'].mean()
        
        # Estimate value in monetary terms (hypothetical)
        VALUE_PER_QUALITY_POINT = 0.50  # $0.50 per quality point
        VALUE_PER_ENGAGEMENT_POINT = 0.30  # $0.30 per engagement point
        
        dataset_costs['estimated_value'] = (
            dataset_quality * VALUE_PER_QUALITY_POINT * dataset_costs['conversation_id'] +
            dataset_engagement * VALUE_PER_ENGAGEMENT_POINT * dataset_costs['conversation_id']
        )
        
        # Calculate ROI
        dataset_costs['roi_percentage'] = (
            (dataset_costs['estimated_value'] - dataset_costs['estimated_cost']) / 
            dataset_costs['estimated_cost'] * 100
        ).round(2)
        
        roi_analysis['dataset_roi'] = dataset_costs.to_dict('index')
        
        # ROI categories
        high_roi_datasets = dataset_costs[dataset_costs['roi_percentage'] > 100].index.tolist()
        medium_roi_datasets = dataset_costs[
            (dataset_costs['roi_percentage'] >= 50) & (dataset_costs['roi_percentage'] <= 100)
        ].index.tolist()
        low_roi_datasets = dataset_costs[dataset_costs['roi_percentage'] < 50].index.tolist()
        
        roi_analysis['roi_categories'] = {
            'high_roi_datasets': high_roi_datasets,
            'medium_roi_datasets': medium_roi_datasets,
            'low_roi_datasets': low_roi_datasets
        }
        
        return roi_analysis
    
    def _assess_strategic_impact(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Assess strategic impact of datasets"""
        print("🎯 Assessing strategic impact...")
        
        strategic_analysis = {}
        
        # Strategic importance factors
        dataset_strategic_scores = {}
        
        for dataset in datasets_info['dataset'].unique():
            dataset_data = datasets_info[datasets_info['dataset'] == dataset]
            
            # Volume impact (market reach)
            volume_score = min(100, len(dataset_data) / 1000 * 10)
            
            # Quality impact (brand value)
            quality_score = dataset_data['quality_score'].mean()
            
            # Complexity impact (technical advancement)
            complexity_score = dataset_data['complexity_score'].mean()
            
            # Engagement impact (user satisfaction)
            engagement_score = dataset_data['engagement_score'].mean()
            
            # Tier diversity impact (market coverage)
            tier_diversity = len(dataset_data['tier'].unique())
            diversity_score = min(100, tier_diversity * 20)
            
            # Overall strategic score
            strategic_score = (
                volume_score * 0.25 +
                quality_score * 0.30 +
                complexity_score * 0.20 +
                engagement_score * 0.15 +
                diversity_score * 0.10
            )
            
            dataset_strategic_scores[dataset] = {
                'volume_impact': volume_score,
                'quality_impact': quality_score,
                'complexity_impact': complexity_score,
                'engagement_impact': engagement_score,
                'diversity_impact': diversity_score,
                'overall_strategic_score': strategic_score
            }
        
        strategic_analysis['strategic_scores'] = dataset_strategic_scores
        
        # Strategic priorities
        strategic_priorities = sorted(
            dataset_strategic_scores.items(),
            key=lambda x: x[1]['overall_strategic_score'],
            reverse=True
        )
        
        strategic_analysis['strategic_priorities'] = {
            'tier_1_strategic': [item[0] for item in strategic_priorities[:3]],
            'tier_2_strategic': [item[0] for item in strategic_priorities[3:6]],
            'tier_3_strategic': [item[0] for item in strategic_priorities[6:]]
        }
        
        return strategic_analysis
    
    def _analyze_resource_efficiency(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze resource efficiency across datasets"""
        print("⚙️ Analyzing resource efficiency...")
        
        efficiency_analysis = {}
        
        # Resource utilization metrics
        resource_metrics = datasets_info.groupby('dataset').agg({
            'word_count': ['sum', 'mean'],
            'turn_count': ['sum', 'mean'],
            'processing_time_hours': ['sum', 'mean'],
            'quality_score': 'mean',
            'conversation_id': 'count'
        })
        
        resource_metrics.columns = ['_'.join(col).strip() for col in resource_metrics.columns]
        
        # Calculate efficiency ratios
        resource_metrics['quality_per_resource_unit'] = (
            resource_metrics['quality_score_mean'] / 
            (resource_metrics['processing_time_hours_sum'].fillna(1) + 
             resource_metrics['word_count_sum'] / 1000)
        )
        
        resource_metrics['conversations_per_hour'] = (
            resource_metrics['conversation_id_count'] / 
            resource_metrics['processing_time_hours_sum'].fillna(1)
        )
        
        resource_metrics['quality_words_ratio'] = (
            resource_metrics['quality_score_mean'] / 
            resource_metrics['word_count_mean']
        )
        
        efficiency_analysis['resource_metrics'] = resource_metrics.to_dict('index')
        
        # Efficiency rankings
        efficiency_rankings = resource_metrics.sort_values(
            'quality_per_resource_unit', ascending=False
        ).index.tolist()
        
        efficiency_analysis['efficiency_rankings'] = efficiency_rankings
        
        return efficiency_analysis
    
    def _analyze_scalability_potential(self, datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scalability potential of datasets"""
        print("📈 Analyzing scalability potential...")
        
        scalability_analysis = {}
        
        # Scalability factors
        dataset_scalability = {}
        
        for dataset in datasets_info['dataset'].unique():
            dataset_data = datasets_info[datasets_info['dataset'] == dataset]
            
            # Current scale
            current_volume = len(dataset_data)
            
            # Quality consistency (lower variance = better scalability)
            quality_variance = dataset_data['quality_score'].var()
            consistency_score = max(0, 100 - quality_variance)
            
            # Processing efficiency
            avg_processing_time = dataset_data['processing_time_hours'].mean()
            efficiency_score = max(0, 100 - avg_processing_time * 10)
            
            # Resource requirements
            avg_word_count = dataset_data['word_count'].mean()
            resource_score = max(0, 100 - (avg_word_count - 100) / 10)
            
            # Growth potential (based on current performance)
            performance_score = dataset_data['quality_score'].mean()
            growth_potential = min(100, performance_score * 1.2)
            
            # Overall scalability score
            scalability_score = (
                consistency_score * 0.30 +
                efficiency_score * 0.25 +
                resource_score * 0.20 +
                growth_potential * 0.25
            )
            
            dataset_scalability[dataset] = {
                'current_volume': current_volume,
                'consistency_score': consistency_score,
                'efficiency_score': efficiency_score,
                'resource_score': resource_score,
                'growth_potential': growth_potential,
                'scalability_score': scalability_score
            }
        
        scalability_analysis['scalability_scores'] = dataset_scalability
        
        # Scalability recommendations
        high_scalability = [d for d, scores in dataset_scalability.items() 
                           if scores['scalability_score'] > 70]
        medium_scalability = [d for d, scores in dataset_scalability.items() 
                             if 50 <= scores['scalability_score'] <= 70]
        low_scalability = [d for d, scores in dataset_scalability.items() 
                          if scores['scalability_score'] < 50]
        
        scalability_analysis['scalability_categories'] = {
            'high_scalability': high_scalability,
            'medium_scalability': medium_scalability,
            'low_scalability': low_scalability
        }
        
        return scalability_analysis
    
    def _generate_strategic_recommendations(self, performance_analysis: Dict[str, Any], 
                                          impact_analysis: Dict[str, Any], 
                                          datasets_info: pd.DataFrame) -> Dict[str, Any]:
        """Generate strategic recommendations based on analysis"""
        print("💡 Generating strategic recommendations...")
        
        recommendations = {
            'investment_priorities': [],
            'optimization_opportunities': [],
            'resource_reallocation': [],
            'strategic_initiatives': [],
            'risk_mitigation': []
        }
        
        # Investment priorities based on ROI and strategic impact
        roi_data = impact_analysis['roi_assessment']['dataset_roi']
        strategic_data = impact_analysis['strategic_impact']['strategic_scores']
        
        high_value_datasets = []
        for dataset in roi_data.keys():
            roi_score = roi_data[dataset]['roi_percentage']
            strategic_score = strategic_data[dataset]['overall_strategic_score']
            
            if roi_score > 100 and strategic_score > 60:
                high_value_datasets.append(dataset)
        
        if high_value_datasets:
            recommendations['investment_priorities'].extend([
                f"Prioritize investment in high-ROI datasets: {', '.join(high_value_datasets[:3])}",
                "Allocate additional resources to scale top-performing datasets",
                "Develop advanced capabilities for strategic datasets"
            ])
        
        # Optimization opportunities
        quality_rankings = performance_analysis['quality_performance']['dataset_rankings']
        low_quality_datasets = [d for d, metrics in quality_rankings.items() 
                               if metrics['quality_score_mean'] < 50]
        
        if low_quality_datasets:
            recommendations['optimization_opportunities'].extend([
                f"Implement quality improvement programs for: {', '.join(low_quality_datasets[:3])}",
                "Establish quality benchmarks and monitoring systems",
                "Develop automated quality enhancement tools"
            ])
        
        # Resource reallocation
        efficiency_data = performance_analysis['efficiency_metrics']['efficiency_rankings']
        low_efficiency_datasets = list(efficiency_data.keys())[-3:]  # Bottom 3
        
        recommendations['resource_reallocation'].extend([
            f"Review resource allocation for low-efficiency datasets: {', '.join(low_efficiency_datasets)}",
            "Redistribute resources from underperforming to high-potential datasets",
            "Implement resource optimization strategies"
        ])
        
        # Strategic initiatives
        scalability_data = impact_analysis['scalability_analysis']['scalability_categories']
        high_scalability_datasets = scalability_data['high_scalability']
        
        if high_scalability_datasets:
            recommendations['strategic_initiatives'].extend([
                f"Launch scaling initiatives for high-potential datasets: {', '.join(high_scalability_datasets[:2])}",
                "Develop standardized scaling frameworks",
                "Create performance monitoring dashboards"
            ])
        
        return recommendations
    
    def _generate_executive_summary(self, performance_analysis: Dict[str, Any], 
                                   impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of findings"""
        
        # Key performance indicators
        quality_dist = performance_analysis['quality_performance']['quality_distribution']
        roi_categories = impact_analysis['roi_assessment']['roi_categories']
        
        summary = {
            'key_findings': [
                f"Analyzed {quality_dist['total_datasets']} datasets with varying performance levels",
                f"{quality_dist['high_quality_datasets']} datasets meet high-quality standards",
                f"{len(roi_categories['high_roi_datasets'])} datasets show high ROI potential",
                f"Strategic value concentrated in top-performing datasets"
            ],
            'performance_highlights': {
                'top_quality_datasets': quality_dist['high_quality_datasets'],
                'high_roi_datasets': len(roi_categories['high_roi_datasets']),
                'improvement_opportunities': quality_dist['low_quality_datasets']
            },
            'strategic_implications': [
                "Focus investment on high-performing, high-ROI datasets",
                "Implement quality improvement programs for underperforming datasets",
                "Develop scalability frameworks for growth-ready datasets",
                "Establish continuous monitoring and optimization processes"
            ]
        }
        
        return summary
    
    def _define_action_priorities(self, performance_analysis: Dict[str, Any], 
                                 impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Define action priorities based on analysis"""
        
        priorities = {
            'immediate_actions': [
                "Conduct detailed review of low-performing datasets",
                "Implement quality monitoring systems",
                "Establish performance benchmarks"
            ],
            'short_term_goals': [
                "Improve quality scores for bottom 25% of datasets",
                "Optimize resource allocation based on ROI analysis",
                "Develop scaling plans for high-potential datasets"
            ],
            'long_term_objectives': [
                "Achieve consistent high-quality performance across all datasets",
                "Establish market leadership in strategic dataset categories",
                "Build sustainable competitive advantages through data excellence"
            ]
        }
        
        return priorities
    
    def _define_success_metrics(self) -> Dict[str, Any]:
        """Define success metrics for performance improvement"""
        
        return {
            'quality_metrics': {
                'target_quality_score': 70,
                'quality_consistency_target': 'std < 15',
                'high_quality_dataset_percentage': '60%'
            },
            'efficiency_metrics': {
                'processing_time_reduction': '25%',
                'resource_utilization_improvement': '30%',
                'cost_per_conversation_reduction': '20%'
            },
            'impact_metrics': {
                'roi_improvement': '40%',
                'strategic_value_increase': '50%',
                'scalability_readiness': '80%'
            }
        }
    
    def _create_performance_impact_visualizations(self, performance_analysis: Dict[str, Any], 
                                                 impact_analysis: Dict[str, Any], 
                                                 datasets_info: pd.DataFrame):
        """Create comprehensive visualizations"""
        print("📊 Creating performance impact visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Dataset Performance and Impact Analysis', fontsize=18, fontweight='bold')
        
        # 1. Quality Performance Rankings
        quality_rankings = performance_analysis['quality_performance']['dataset_rankings']
        datasets = list(quality_rankings.keys())[:10]  # Top 10
        quality_scores = [quality_rankings[d]['quality_score_mean'] for d in datasets]
        
        bars = axes[0, 0].bar(range(len(datasets)), quality_scores, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Dataset Quality Performance Rankings')
        axes[0, 0].set_ylabel('Average Quality Score')
        axes[0, 0].set_xticks(range(len(datasets)))
        axes[0, 0].set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{score:.1f}', ha='center', va='bottom')
        
        # 2. ROI Analysis
        roi_data = impact_analysis['roi_assessment']['dataset_roi']
        datasets_roi = list(roi_data.keys())[:8]  # Top 8
        roi_values = [roi_data[d]['roi_percentage'] for d in datasets_roi]
        
        colors = ['green' if roi > 0 else 'red' for roi in roi_values]
        bars = axes[0, 1].bar(range(len(datasets_roi)), roi_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Dataset ROI Analysis')
        axes[0, 1].set_ylabel('ROI Percentage')
        axes[0, 1].set_xticks(range(len(datasets_roi)))
        axes[0, 1].set_xticklabels([d[:12] + '...' if len(d) > 12 else d for d in datasets_roi], rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Strategic Impact Scatter
        strategic_data = impact_analysis['strategic_impact']['strategic_scores']
        datasets_strategic = list(strategic_data.keys())
        
        strategic_scores = [strategic_data[d]['overall_strategic_score'] for d in datasets_strategic]
        quality_scores_strategic = [quality_rankings.get(d, {}).get('quality_score_mean', 0) for d in datasets_strategic]
        
        scatter = axes[0, 2].scatter(strategic_scores, quality_scores_strategic, 
                                   alpha=0.7, s=60, c='purple')
        axes[0, 2].set_title('Strategic Impact vs Quality')
        axes[0, 2].set_xlabel('Strategic Impact Score')
        axes[0, 2].set_ylabel('Quality Score')
        
        # 4. Volume Distribution
        volume_data = datasets_info['dataset'].value_counts().head(8)
        
        axes[1, 0].pie(volume_data.values, labels=[l[:15] + '...' if len(l) > 15 else l for l in volume_data.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Dataset Volume Distribution')
        
        # 5. Efficiency Metrics
        efficiency_data = performance_analysis['efficiency_metrics']['efficiency_rankings']
        top_efficient = list(efficiency_data.keys())[:6]
        efficiency_scores = list(efficiency_data.values())[:6]
        
        bars = axes[1, 1].barh(range(len(top_efficient)), efficiency_scores, color='lightgreen', alpha=0.8)
        axes[1, 1].set_title('Dataset Efficiency Rankings')
        axes[1, 1].set_xlabel('Efficiency Score')
        axes[1, 1].set_yticks(range(len(top_efficient)))
        axes[1, 1].set_yticklabels([d[:20] + '...' if len(d) > 20 else d for d in top_efficient])
        
        # 6. Scalability Analysis
        scalability_data = impact_analysis['scalability_analysis']['scalability_scores']
        datasets_scalability = list(scalability_data.keys())[:8]
        scalability_scores = [scalability_data[d]['scalability_score'] for d in datasets_scalability]
        
        bars = axes[1, 2].bar(range(len(datasets_scalability)), scalability_scores, 
                             color='orange', alpha=0.8)
        axes[1, 2].set_title('Dataset Scalability Potential')
        axes[1, 2].set_ylabel('Scalability Score')
        axes[1, 2].set_xticks(range(len(datasets_scalability)))
        axes[1, 2].set_xticklabels([d[:12] + '...' if len(d) > 12 else d for d in datasets_scalability], rotation=45)
        
        # 7. Performance Distribution
        quality_dist = performance_analysis['quality_performance']['quality_distribution']
        categories = ['High Quality', 'Medium Quality', 'Low Quality']
        values = [quality_dist['high_quality_datasets'], 
                 quality_dist['medium_quality_datasets'], 
                 quality_dist['low_quality_datasets']]
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        bars = axes[2, 0].bar(categories, values, color=colors, alpha=0.8)
        axes[2, 0].set_title('Quality Distribution Across Datasets')
        axes[2, 0].set_ylabel('Number of Datasets')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           str(value), ha='center', va='bottom')
        
        # 8. Value Contribution
        value_data = impact_analysis['value_contribution']['dataset_value_scores']
        top_value_datasets = sorted(value_data.items(), 
                                   key=lambda x: x[1]['overall_value'], reverse=True)[:6]
        
        datasets_value = [item[0] for item in top_value_datasets]
        value_percentages = [item[1]['value_percentage'] for item in top_value_datasets]
        
        bars = axes[2, 1].bar(range(len(datasets_value)), value_percentages, 
                             color='gold', alpha=0.8)
        axes[2, 1].set_title('Top Value Contributing Datasets')
        axes[2, 1].set_ylabel('Value Contribution (%)')
        axes[2, 1].set_xticks(range(len(datasets_value)))
        axes[2, 1].set_xticklabels([d[:12] + '...' if len(d) > 12 else d for d in datasets_value], rotation=45)
        
        # 9. Performance Trends (if available)
        if 'trend_slopes' in performance_analysis.get('trend_analysis', {}):
            trend_data = performance_analysis['trend_analysis']['trend_slopes']
            datasets_trend = list(trend_data.keys())[:8]
            trend_values = [trend_data[d]['overall_trend'] for d in datasets_trend]
            
            colors = ['green' if trend > 0 else 'red' for trend in trend_values]
            bars = axes[2, 2].bar(range(len(datasets_trend)), trend_values, 
                                 color=colors, alpha=0.7)
            axes[2, 2].set_title('Performance Trend Analysis')
            axes[2, 2].set_ylabel('Trend Slope')
            axes[2, 2].set_xticks(range(len(datasets_trend)))
            axes[2, 2].set_xticklabels([d[:10] + '...' if len(d) > 10 else d for d in datasets_trend], rotation=45)
            axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, 'Trend analysis\nnot available', 
                           ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Performance Trend Analysis')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'/home/vivi/pixelated/ai/monitoring/dataset_performance_impact_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance impact visualizations saved as dataset_performance_impact_{timestamp}.png")
    
    # Helper methods
    def _calculate_growth_trends(self, temporal_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate growth trends for datasets"""
        growth_trends = {}
        
        for dataset in temporal_data.index:
            values = temporal_data.loc[dataset].values
            if len(values) > 2:
                # Calculate linear trend
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                growth_trends[dataset] = slope
            else:
                growth_trends[dataset] = 0
        
        return growth_trends

def main():
    """Main execution function"""
    print("🚀 Starting Dataset Performance and Impact Analysis System")
    print("=" * 75)
    
    analyzer = DatasetPerformanceImpactAnalyzer()
    
    try:
        # Run the complete performance and impact analysis
        results = analyzer.analyze_performance_impact()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/vivi/pixelated/ai/monitoring/dataset_performance_impact_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        print(f"📊 Datasets analyzed: {results['datasets_analyzed']}")
        
        # Display executive summary
        exec_summary = results['executive_summary']
        print(f"\n📋 Executive Summary:")
        for finding in exec_summary['key_findings']:
            print(f"  • {finding}")
        
        # Display strategic implications
        print(f"\n🎯 Strategic Implications:")
        for implication in exec_summary['strategic_implications']:
            print(f"  • {implication}")
        
        # Display top recommendations
        recommendations = results['strategic_recommendations']
        print(f"\n💡 Top Strategic Recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"  {category.replace('_', ' ').title()}:")
                for rec in recs[:2]:  # Top 2 per category
                    print(f"    - {rec}")
        
        # Display performance highlights
        highlights = exec_summary['performance_highlights']
        print(f"\n📈 Performance Highlights:")
        print(f"  • High-quality datasets: {highlights['top_quality_datasets']}")
        print(f"  • High-ROI datasets: {highlights['high_roi_datasets']}")
        print(f"  • Improvement opportunities: {highlights['improvement_opportunities']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
