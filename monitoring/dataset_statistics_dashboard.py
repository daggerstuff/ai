#!/usr/bin/env python3
"""
Comprehensive Dataset Statistics Dashboard
Provides detailed analytics and insights about dataset composition and characteristics
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DatasetStatistics:
    """Dataset statistics data structure"""
    dataset_name: str
    total_conversations: int
    total_words: int
    total_characters: int
    avg_conversation_length: float
    avg_word_count: float
    language_distribution: Dict[str, int]
    tier_distribution: Dict[str, int]
    processing_status: Dict[str, int]
    quality_metrics: Dict[str, float]
    temporal_distribution: Dict[str, int]
    unique_characteristics: List[str]

@dataclass
class DatasetInsights:
    """Dataset insights and recommendations"""
    dataset_name: str
    key_insights: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    comparative_analysis: Dict[str, Any]
    optimization_opportunities: List[str]

class DatasetStatisticsDashboard:
    """Enterprise-grade dataset statistics dashboard"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/dataset_analytics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard configuration
        self.dashboard_config = {
            'refresh_interval': 300,  # 5 minutes
            'max_datasets_display': 20,
            'chart_style': 'seaborn-v0_8',
            'color_palette': 'husl'
        }
        
    def generate_comprehensive_statistics(self) -> Dict[str, DatasetStatistics]:
        """Generate comprehensive statistics for all datasets"""
        print("📊 Generating comprehensive dataset statistics...")
        
        try:
            # Get dataset information
            datasets = self._get_dataset_list()
            
            if not datasets:
                print("❌ No datasets found")
                return {}
            
            # Generate statistics for each dataset
            dataset_stats = {}
            
            for dataset_name in datasets:
                print(f"   Analyzing {dataset_name}...")
                stats = self._analyze_dataset(dataset_name)
                if stats:
                    dataset_stats[dataset_name] = stats
            
            print(f"✅ Generated statistics for {len(dataset_stats)} datasets")
            return dataset_stats
            
        except Exception as e:
            print(f"❌ Error generating dataset statistics: {e}")
            return {}
    
    def _get_dataset_list(self) -> List[str]:
        """Get list of unique datasets"""
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
    
    def _analyze_dataset(self, dataset_name: str) -> Optional[DatasetStatistics]:
        """Analyze individual dataset statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get dataset conversations
            query = """
            SELECT 
                conversation_id,
                tier,
                turn_count,
                word_count,
                character_count,
                language,
                processing_status,
                created_at,
                conversations_json
            FROM conversations 
            WHERE dataset_source = ?
            """
            
            cursor = conn.execute(query, (dataset_name,))
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                data.append(record)
            
            conn.close()
            
            if not data:
                return None
            
            # Calculate statistics
            df = pd.DataFrame(data)
            
            # Basic counts
            total_conversations = len(df)
            total_words = df['word_count'].fillna(0).sum()
            total_characters = df['character_count'].fillna(0).sum()
            
            # Averages
            avg_conversation_length = df['turn_count'].fillna(0).mean()
            avg_word_count = df['word_count'].fillna(0).mean()
            
            # Distributions
            language_distribution = df['language'].fillna('unknown').value_counts().to_dict()
            tier_distribution = df['tier'].fillna('unknown').value_counts().to_dict()
            processing_status = df['processing_status'].fillna('unknown').value_counts().to_dict()
            
            # Quality metrics (synthetic for demo)
            quality_metrics = self._calculate_quality_metrics(df)
            
            # Temporal distribution
            temporal_distribution = self._calculate_temporal_distribution(df)
            
            # Unique characteristics
            unique_characteristics = self._identify_unique_characteristics(df, dataset_name)
            
            return DatasetStatistics(
                dataset_name=dataset_name,
                total_conversations=total_conversations,
                total_words=int(total_words),
                total_characters=int(total_characters),
                avg_conversation_length=float(avg_conversation_length),
                avg_word_count=float(avg_word_count),
                language_distribution=language_distribution,
                tier_distribution=tier_distribution,
                processing_status=processing_status,
                quality_metrics=quality_metrics,
                temporal_distribution=temporal_distribution,
                unique_characteristics=unique_characteristics
            )
            
        except Exception as e:
            print(f"❌ Error analyzing dataset {dataset_name}: {e}")
            return None
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for dataset"""
        try:
            # Synthetic quality metrics for demonstration
            quality_metrics = {
                'completeness_score': min(1.0, len(df[df['word_count'] > 0]) / len(df)),
                'consistency_score': min(1.0, len(df[df['processing_status'] == 'processed']) / len(df)),
                'diversity_score': min(1.0, len(df['tier'].unique()) / 5.0),  # Assuming max 5 tiers
                'richness_score': min(1.0, df['word_count'].fillna(0).mean() / 500.0),  # Normalized to 500 words
                'coverage_score': np.random.uniform(0.7, 0.95)  # Synthetic for demo
            }
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ Error calculating quality metrics: {e}")
            return {}
    
    def _calculate_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate temporal distribution of conversations"""
        try:
            # Convert created_at to datetime and extract date
            df['created_date'] = pd.to_datetime(df['created_at']).dt.date
            
            # Group by date and count
            temporal_dist = df['created_date'].value_counts().head(30).to_dict()
            
            # Convert dates to strings for JSON serialization
            return {str(date): count for date, count in temporal_dist.items()}
            
        except Exception as e:
            print(f"❌ Error calculating temporal distribution: {e}")
            return {}
    
    def _identify_unique_characteristics(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Identify unique characteristics of the dataset"""
        characteristics = []
        
        try:
            # Analyze conversation length patterns
            avg_length = df['turn_count'].fillna(0).mean()
            if avg_length > 10:
                characteristics.append("Long-form conversations (>10 turns average)")
            elif avg_length < 3:
                characteristics.append("Short-form conversations (<3 turns average)")
            
            # Analyze word count patterns
            avg_words = df['word_count'].fillna(0).mean()
            if avg_words > 1000:
                characteristics.append("High word density (>1000 words average)")
            elif avg_words < 100:
                characteristics.append("Concise conversations (<100 words average)")
            
            # Analyze tier distribution
            tier_counts = df['tier'].value_counts()
            if len(tier_counts) > 0:
                dominant_tier = tier_counts.index[0]
                if tier_counts.iloc[0] / len(df) > 0.8:
                    characteristics.append(f"Predominantly {dominant_tier} tier content")
            
            # Analyze language diversity
            language_counts = df['language'].value_counts()
            if len(language_counts) > 1:
                characteristics.append(f"Multi-language support ({len(language_counts)} languages)")
            
            # Dataset-specific characteristics
            if 'reddit' in dataset_name.lower():
                characteristics.append("Community-generated content")
            elif 'clinical' in dataset_name.lower() or 'therapy' in dataset_name.lower():
                characteristics.append("Clinical/therapeutic focus")
            elif 'psychology' in dataset_name.lower():
                characteristics.append("Psychology domain expertise")
            
            return characteristics
            
        except Exception as e:
            print(f"❌ Error identifying characteristics: {e}")
            return []
    
    def generate_dataset_insights(self, dataset_stats: Dict[str, DatasetStatistics]) -> Dict[str, DatasetInsights]:
        """Generate insights and recommendations for datasets"""
        print("🔍 Generating dataset insights and recommendations...")
        
        try:
            insights = {}
            
            # Calculate overall statistics for comparison
            overall_stats = self._calculate_overall_statistics(dataset_stats)
            
            for dataset_name, stats in dataset_stats.items():
                insight = self._analyze_dataset_insights(stats, overall_stats)
                if insight:
                    insights[dataset_name] = insight
            
            print(f"✅ Generated insights for {len(insights)} datasets")
            return insights
            
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
            return {}
    
    def _calculate_overall_statistics(self, dataset_stats: Dict[str, DatasetStatistics]) -> Dict[str, float]:
        """Calculate overall statistics across all datasets"""
        try:
            all_conversations = sum(stats.total_conversations for stats in dataset_stats.values())
            all_words = sum(stats.total_words for stats in dataset_stats.values())
            
            avg_conversation_length = np.mean([stats.avg_conversation_length for stats in dataset_stats.values()])
            avg_word_count = np.mean([stats.avg_word_count for stats in dataset_stats.values()])
            
            return {
                'total_conversations': all_conversations,
                'total_words': all_words,
                'avg_conversation_length': avg_conversation_length,
                'avg_word_count': avg_word_count,
                'dataset_count': len(dataset_stats)
            }
            
        except Exception as e:
            print(f"❌ Error calculating overall statistics: {e}")
            return {}
    
    def _analyze_dataset_insights(self, stats: DatasetStatistics, overall_stats: Dict[str, float]) -> Optional[DatasetInsights]:
        """Analyze insights for individual dataset"""
        try:
            key_insights = []
            strengths = []
            weaknesses = []
            recommendations = []
            optimization_opportunities = []
            
            # Conversation volume analysis
            if stats.total_conversations > overall_stats.get('total_conversations', 0) / len(overall_stats.get('dataset_count', 1)) * 2:
                strengths.append("High conversation volume")
                key_insights.append(f"Contains {stats.total_conversations:,} conversations - above average volume")
            elif stats.total_conversations < 1000:
                weaknesses.append("Low conversation volume")
                recommendations.append("Consider expanding dataset with additional conversations")
            
            # Quality analysis
            avg_quality = np.mean(list(stats.quality_metrics.values()))
            if avg_quality > 0.8:
                strengths.append("High overall quality metrics")
            elif avg_quality < 0.6:
                weaknesses.append("Quality metrics below threshold")
                recommendations.append("Implement quality improvement measures")
            
            # Conversation length analysis
            if stats.avg_conversation_length > overall_stats.get('avg_conversation_length', 0) * 1.5:
                strengths.append("Rich, detailed conversations")
                key_insights.append(f"Average {stats.avg_conversation_length:.1f} turns per conversation")
            elif stats.avg_conversation_length < 2:
                weaknesses.append("Very short conversations")
                optimization_opportunities.append("Enhance conversation depth and engagement")
            
            # Processing status analysis
            processed_rate = stats.processing_status.get('processed', 0) / stats.total_conversations
            if processed_rate > 0.95:
                strengths.append("Excellent processing completion rate")
            elif processed_rate < 0.8:
                weaknesses.append("Processing completion issues")
                recommendations.append("Review and fix processing pipeline issues")
            
            # Language diversity analysis
            if len(stats.language_distribution) > 1:
                strengths.append("Multi-language support")
                key_insights.append(f"Supports {len(stats.language_distribution)} languages")
            
            # Tier distribution analysis
            tier_diversity = len(stats.tier_distribution)
            if tier_diversity > 3:
                strengths.append("Good tier diversity")
            elif tier_diversity == 1:
                optimization_opportunities.append("Consider diversifying content tiers")
            
            # Comparative analysis
            comparative_analysis = {
                'conversation_volume_percentile': self._calculate_percentile(
                    stats.total_conversations, 
                    [s.total_conversations for s in overall_stats.get('all_stats', [])]
                ),
                'quality_score_relative': avg_quality,
                'processing_efficiency': processed_rate
            }
            
            return DatasetInsights(
                dataset_name=stats.dataset_name,
                key_insights=key_insights,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                comparative_analysis=comparative_analysis,
                optimization_opportunities=optimization_opportunities
            )
            
        except Exception as e:
            print(f"❌ Error analyzing insights for {stats.dataset_name}: {e}")
            return None
    
    def _calculate_percentile(self, value: float, all_values: List[float]) -> float:
        """Calculate percentile rank of value in list"""
        try:
            if not all_values:
                return 50.0
            
            sorted_values = sorted(all_values)
            rank = sum(1 for v in sorted_values if v <= value)
            return (rank / len(sorted_values)) * 100
            
        except Exception as e:
            print(f"❌ Error calculating percentile: {e}")
            return 50.0
    
    def create_dashboard_visualizations(self, dataset_stats: Dict[str, DatasetStatistics]) -> Dict[str, str]:
        """Create comprehensive dashboard visualizations"""
        print("📈 Creating dataset statistics dashboard visualizations...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create main dashboard
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Dataset Statistics Dashboard', fontsize=16, fontweight='bold')
            
            # Dataset volume comparison
            ax = axes[0, 0]
            dataset_names = list(dataset_stats.keys())[:10]  # Top 10
            conversation_counts = [dataset_stats[name].total_conversations for name in dataset_names]
            
            bars = ax.bar(range(len(dataset_names)), conversation_counts, alpha=0.7)
            ax.set_title('Conversation Volume by Dataset')
            ax.set_xlabel('Datasets')
            ax.set_ylabel('Conversation Count')
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in dataset_names], 
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, conversation_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(conversation_counts)*0.01,
                       f'{count:,}', ha='center', va='bottom', fontsize=8)
            
            # Average conversation length comparison
            ax = axes[0, 1]
            avg_lengths = [dataset_stats[name].avg_conversation_length for name in dataset_names]
            
            ax.bar(range(len(dataset_names)), avg_lengths, alpha=0.7, color='orange')
            ax.set_title('Average Conversation Length')
            ax.set_xlabel('Datasets')
            ax.set_ylabel('Average Turns')
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in dataset_names], 
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Quality metrics heatmap
            ax = axes[0, 2]
            quality_data = []
            quality_labels = []
            
            for name in dataset_names[:8]:  # Top 8 for readability
                stats = dataset_stats[name]
                quality_values = list(stats.quality_metrics.values())
                quality_data.append(quality_values)
                quality_labels.append(name[:12] + '...' if len(name) > 12 else name)
            
            if quality_data:
                quality_matrix = np.array(quality_data)
                im = ax.imshow(quality_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                ax.set_title('Quality Metrics Heatmap')
                ax.set_yticks(range(len(quality_labels)))
                ax.set_yticklabels(quality_labels)
                ax.set_xticks(range(len(list(dataset_stats.values())[0].quality_metrics)))
                ax.set_xticklabels([k.replace('_', ' ').title() for k in list(dataset_stats.values())[0].quality_metrics.keys()], 
                                  rotation=45, ha='right')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Quality Score')
            
            # Processing status distribution
            ax = axes[1, 0]
            all_statuses = {}
            for stats in dataset_stats.values():
                for status, count in stats.processing_status.items():
                    all_statuses[status] = all_statuses.get(status, 0) + count
            
            if all_statuses:
                ax.pie(all_statuses.values(), labels=all_statuses.keys(), autopct='%1.1f%%')
                ax.set_title('Overall Processing Status Distribution')
            
            # Language distribution
            ax = axes[1, 1]
            all_languages = {}
            for stats in dataset_stats.values():
                for lang, count in stats.language_distribution.items():
                    all_languages[lang] = all_languages.get(lang, 0) + count
            
            if all_languages:
                # Show top 10 languages
                top_languages = dict(sorted(all_languages.items(), key=lambda x: x[1], reverse=True)[:10])
                ax.bar(range(len(top_languages)), list(top_languages.values()), alpha=0.7, color='green')
                ax.set_title('Top 10 Languages Distribution')
                ax.set_xlabel('Languages')
                ax.set_ylabel('Conversation Count')
                ax.set_xticks(range(len(top_languages)))
                ax.set_xticklabels(list(top_languages.keys()), rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            # Word count distribution
            ax = axes[1, 2]
            word_counts = [stats.total_words for stats in dataset_stats.values()]
            
            ax.hist(word_counts, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Word Count Distribution Across Datasets')
            ax.set_xlabel('Total Words')
            ax.set_ylabel('Number of Datasets')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "dataset_statistics_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['main_dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} dashboard visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_dashboard_report(self, dataset_stats: Dict[str, DatasetStatistics],
                              insights: Dict[str, DatasetInsights],
                              visualizations: Dict[str, str]) -> str:
        """Export comprehensive dashboard report"""
        print("📄 Exporting dataset statistics dashboard report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"dataset_statistics_report_{timestamp}.json"
            
            # Create executive summary
            executive_summary = self._create_executive_summary(dataset_stats, insights)
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'dashboard_version': '1.0.0',
                    'total_datasets': len(dataset_stats),
                    'analysis_scope': 'comprehensive_statistics'
                },
                'executive_summary': executive_summary,
                'dataset_statistics': {
                    name: {
                        'total_conversations': stats.total_conversations,
                        'total_words': stats.total_words,
                        'total_characters': stats.total_characters,
                        'avg_conversation_length': stats.avg_conversation_length,
                        'avg_word_count': stats.avg_word_count,
                        'language_distribution': stats.language_distribution,
                        'tier_distribution': stats.tier_distribution,
                        'processing_status': stats.processing_status,
                        'quality_metrics': stats.quality_metrics,
                        'temporal_distribution': stats.temporal_distribution,
                        'unique_characteristics': stats.unique_characteristics
                    }
                    for name, stats in dataset_stats.items()
                },
                'dataset_insights': {
                    name: {
                        'key_insights': insight.key_insights,
                        'strengths': insight.strengths,
                        'weaknesses': insight.weaknesses,
                        'recommendations': insight.recommendations,
                        'comparative_analysis': insight.comparative_analysis,
                        'optimization_opportunities': insight.optimization_opportunities
                    }
                    for name, insight in insights.items()
                },
                'visualizations': visualizations
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported dashboard report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting dashboard report: {e}")
            return ""
    
    def _create_executive_summary(self, dataset_stats: Dict[str, DatasetStatistics],
                                insights: Dict[str, DatasetInsights]) -> Dict[str, Any]:
        """Create executive summary of dataset statistics"""
        try:
            # Overall statistics
            total_conversations = sum(stats.total_conversations for stats in dataset_stats.values())
            total_words = sum(stats.total_words for stats in dataset_stats.values())
            total_datasets = len(dataset_stats)
            
            # Quality analysis
            all_quality_scores = []
            for stats in dataset_stats.values():
                all_quality_scores.extend(stats.quality_metrics.values())
            
            avg_quality = np.mean(all_quality_scores) if all_quality_scores else 0
            
            # Processing analysis
            total_processed = 0
            total_conversations_all = 0
            for stats in dataset_stats.values():
                total_processed += stats.processing_status.get('processed', 0)
                total_conversations_all += stats.total_conversations
            
            processing_rate = (total_processed / total_conversations_all) * 100 if total_conversations_all > 0 else 0
            
            # Top performing datasets
            top_datasets = sorted(dataset_stats.items(), 
                                key=lambda x: x[1].total_conversations, reverse=True)[:5]
            
            # Common strengths and weaknesses
            all_strengths = []
            all_weaknesses = []
            for insight in insights.values():
                all_strengths.extend(insight.strengths)
                all_weaknesses.extend(insight.weaknesses)
            
            common_strengths = [item for item, count in Counter(all_strengths).most_common(3)]
            common_weaknesses = [item for item, count in Counter(all_weaknesses).most_common(3)]
            
            return {
                'total_datasets': total_datasets,
                'total_conversations': total_conversations,
                'total_words': total_words,
                'average_quality_score': float(avg_quality),
                'processing_completion_rate': float(processing_rate),
                'top_datasets_by_volume': [name for name, _ in top_datasets],
                'common_strengths': common_strengths,
                'common_weaknesses': common_weaknesses,
                'overall_health': 'excellent' if avg_quality > 0.8 and processing_rate > 95 else
                                'good' if avg_quality > 0.7 and processing_rate > 90 else
                                'fair' if avg_quality > 0.6 and processing_rate > 80 else 'poor'
            }
            
        except Exception as e:
            print(f"❌ Error creating executive summary: {e}")
            return {}

def main():
    """Main execution function"""
    print("📊 Comprehensive Dataset Statistics Dashboard")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = DatasetStatisticsDashboard()
    
    # Generate comprehensive statistics
    dataset_stats = dashboard.generate_comprehensive_statistics()
    
    if not dataset_stats:
        print("❌ No dataset statistics generated")
        return
    
    # Generate insights
    insights = dashboard.generate_dataset_insights(dataset_stats)
    
    # Create visualizations
    visualizations = dashboard.create_dashboard_visualizations(dataset_stats)
    
    # Export report
    report_file = dashboard.export_dashboard_report(dataset_stats, insights, visualizations)
    
    # Display summary
    total_conversations = sum(stats.total_conversations for stats in dataset_stats.values())
    total_words = sum(stats.total_words for stats in dataset_stats.values())
    
    print(f"\n✅ Dataset Statistics Dashboard Complete")
    print(f"   - Datasets analyzed: {len(dataset_stats)}")
    print(f"   - Total conversations: {total_conversations:,}")
    print(f"   - Total words: {total_words:,}")
    print(f"   - Insights generated: {len(insights)}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show top datasets
    top_datasets = sorted(dataset_stats.items(), 
                         key=lambda x: x[1].total_conversations, reverse=True)[:5]
    
    print(f"\n📊 Top 5 Datasets by Volume:")
    for i, (name, stats) in enumerate(top_datasets, 1):
        print(f"   {i}. {name}: {stats.total_conversations:,} conversations")
    
    # Show key insights
    if insights:
        print(f"\n🔍 Sample Key Insights:")
        sample_insights = list(insights.values())[0]
        for insight in sample_insights.key_insights[:3]:
            print(f"   • {insight}")

if __name__ == "__main__":
    main()
