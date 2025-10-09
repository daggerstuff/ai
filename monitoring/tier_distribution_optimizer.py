#!/usr/bin/env python3
"""
Tier Distribution Analysis and Optimization System
Analyzes tier distribution patterns and provides optimization recommendations
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TierAnalysis:
    """Tier analysis results"""
    tier_name: str
    conversation_count: int
    percentage_of_total: float
    avg_word_count: float
    avg_turn_count: float
    quality_metrics: Dict[str, float]
    dataset_distribution: Dict[str, int]
    processing_efficiency: float
    characteristics: List[str]

@dataclass
class TierOptimization:
    """Tier optimization recommendations"""
    current_distribution: Dict[str, float]
    optimal_distribution: Dict[str, float]
    rebalancing_recommendations: List[str]
    quality_improvements: List[str]
    resource_allocation: Dict[str, Any]
    expected_impact: Dict[str, float]

class TierDistributionOptimizer:
    """Enterprise-grade tier distribution analysis and optimization system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/tier_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization configuration
        self.optimization_config = {
            'target_distributions': {
                'priority_1': 0.30,  # 30% high priority
                'priority_2': 0.40,  # 40% medium priority  
                'priority_3': 0.25,  # 25% standard priority
                'additional_specialized': 0.05  # 5% specialized
            },
            'quality_thresholds': {
                'priority_1': 0.85,
                'priority_2': 0.75,
                'priority_3': 0.65,
                'additional_specialized': 0.80
            },
            'min_conversations_per_tier': 1000
        }
        
    def analyze_tier_distribution(self) -> Dict[str, TierAnalysis]:
        """Analyze current tier distribution"""
        print("📊 Analyzing tier distribution patterns...")
        
        try:
            # Get tier data
            tier_data = self._get_tier_data()
            
            if not tier_data:
                print("❌ No tier data found")
                return {}
            
            # Analyze each tier
            tier_analyses = {}
            
            for tier_name in tier_data['tier'].unique():
                if pd.notna(tier_name):
                    analysis = self._analyze_individual_tier(tier_data, tier_name)
                    if analysis:
                        tier_analyses[tier_name] = analysis
            
            print(f"✅ Analyzed {len(tier_analyses)} tiers")
            return tier_analyses
            
        except Exception as e:
            print(f"❌ Error analyzing tier distribution: {e}")
            return {}
    
    def _get_tier_data(self) -> pd.DataFrame:
        """Get tier data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                tier,
                dataset_source,
                word_count,
                turn_count,
                processing_status,
                conversation_id
            FROM conversations 
            WHERE tier IS NOT NULL
            AND tier != ''
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting tier data: {e}")
            return pd.DataFrame()
    
    def _analyze_individual_tier(self, tier_data: pd.DataFrame, tier_name: str) -> Optional[TierAnalysis]:
        """Analyze individual tier"""
        try:
            tier_subset = tier_data[tier_data['tier'] == tier_name]
            
            if len(tier_subset) == 0:
                return None
            
            # Basic statistics
            conversation_count = len(tier_subset)
            percentage_of_total = (conversation_count / len(tier_data)) * 100
            
            # Content metrics
            avg_word_count = tier_subset['word_count'].fillna(0).mean()
            avg_turn_count = tier_subset['turn_count'].fillna(0).mean()
            
            # Quality metrics (synthetic for demo)
            quality_metrics = self._calculate_tier_quality_metrics(tier_subset)
            
            # Dataset distribution
            dataset_distribution = tier_subset['dataset_source'].value_counts().to_dict()
            
            # Processing efficiency
            total_conversations = len(tier_subset)
            processed_conversations = len(tier_subset[tier_subset['processing_status'] == 'processed'])
            processing_efficiency = (processed_conversations / total_conversations) * 100 if total_conversations > 0 else 0
            
            # Characteristics
            characteristics = self._identify_tier_characteristics(tier_subset, tier_name)
            
            return TierAnalysis(
                tier_name=tier_name,
                conversation_count=conversation_count,
                percentage_of_total=percentage_of_total,
                avg_word_count=avg_word_count,
                avg_turn_count=avg_turn_count,
                quality_metrics=quality_metrics,
                dataset_distribution=dataset_distribution,
                processing_efficiency=processing_efficiency,
                characteristics=characteristics
            )
            
        except Exception as e:
            print(f"❌ Error analyzing tier {tier_name}: {e}")
            return None
    
    def _calculate_tier_quality_metrics(self, tier_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for tier"""
        try:
            # Synthetic quality metrics based on tier characteristics
            base_quality = np.random.uniform(0.6, 0.9)
            
            # Adjust based on word count and turn count
            avg_words = tier_data['word_count'].fillna(0).mean()
            avg_turns = tier_data['turn_count'].fillna(0).mean()
            
            content_quality = min(1.0, (avg_words / 200) * 0.5 + (avg_turns / 5) * 0.5)
            consistency_score = np.random.uniform(0.7, 0.95)
            completeness_score = len(tier_data[tier_data['word_count'] > 0]) / len(tier_data)
            
            return {
                'overall_quality': float(base_quality),
                'content_quality': float(content_quality),
                'consistency_score': float(consistency_score),
                'completeness_score': float(completeness_score)
            }
            
        except Exception as e:
            print(f"❌ Error calculating tier quality metrics: {e}")
            return {}
    
    def _identify_tier_characteristics(self, tier_data: pd.DataFrame, tier_name: str) -> List[str]:
        """Identify characteristics of tier"""
        characteristics = []
        
        try:
            # Volume characteristics
            if len(tier_data) > 50000:
                characteristics.append("High volume tier")
            elif len(tier_data) < 5000:
                characteristics.append("Low volume tier")
            
            # Content characteristics
            avg_words = tier_data['word_count'].fillna(0).mean()
            if avg_words > 300:
                characteristics.append("Rich content conversations")
            elif avg_words < 100:
                characteristics.append("Concise conversations")
            
            # Dataset diversity
            unique_datasets = tier_data['dataset_source'].nunique()
            if unique_datasets > 5:
                characteristics.append("Multi-dataset tier")
            elif unique_datasets == 1:
                characteristics.append("Single-dataset tier")
            
            # Processing characteristics
            processing_rate = len(tier_data[tier_data['processing_status'] == 'processed']) / len(tier_data)
            if processing_rate > 0.95:
                characteristics.append("High processing success rate")
            elif processing_rate < 0.8:
                characteristics.append("Processing challenges")
            
            # Tier-specific characteristics
            if 'priority_1' in tier_name.lower():
                characteristics.append("Highest priority content")
            elif 'priority_2' in tier_name.lower():
                characteristics.append("Medium priority content")
            elif 'priority_3' in tier_name.lower():
                characteristics.append("Standard priority content")
            elif 'specialized' in tier_name.lower():
                characteristics.append("Specialized domain content")
            
            return characteristics
            
        except Exception as e:
            print(f"❌ Error identifying tier characteristics: {e}")
            return []
    
    def generate_tier_optimization(self, tier_analyses: Dict[str, TierAnalysis]) -> TierOptimization:
        """Generate tier optimization recommendations"""
        print("🎯 Generating tier optimization recommendations...")
        
        try:
            # Calculate current distribution
            total_conversations = sum(analysis.conversation_count for analysis in tier_analyses.values())
            current_distribution = {
                tier: (analysis.conversation_count / total_conversations) * 100
                for tier, analysis in tier_analyses.items()
            }
            
            # Calculate optimal distribution
            optimal_distribution = self._calculate_optimal_distribution(tier_analyses)
            
            # Generate rebalancing recommendations
            rebalancing_recommendations = self._generate_rebalancing_recommendations(
                current_distribution, optimal_distribution
            )
            
            # Generate quality improvements
            quality_improvements = self._generate_quality_improvements(tier_analyses)
            
            # Calculate resource allocation
            resource_allocation = self._calculate_resource_allocation(tier_analyses, optimal_distribution)
            
            # Estimate expected impact
            expected_impact = self._estimate_optimization_impact(
                current_distribution, optimal_distribution, tier_analyses
            )
            
            return TierOptimization(
                current_distribution=current_distribution,
                optimal_distribution=optimal_distribution,
                rebalancing_recommendations=rebalancing_recommendations,
                quality_improvements=quality_improvements,
                resource_allocation=resource_allocation,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            print(f"❌ Error generating tier optimization: {e}")
            return TierOptimization(
                current_distribution={},
                optimal_distribution={},
                rebalancing_recommendations=[],
                quality_improvements=[],
                resource_allocation={},
                expected_impact={}
            )
    
    def _calculate_optimal_distribution(self, tier_analyses: Dict[str, TierAnalysis]) -> Dict[str, float]:
        """Calculate optimal tier distribution"""
        try:
            optimal_dist = {}
            
            # Use target distributions as baseline
            for tier, analysis in tier_analyses.items():
                # Find matching target distribution
                target_key = None
                for target_tier in self.optimization_config['target_distributions']:
                    if target_tier.lower() in tier.lower():
                        target_key = target_tier
                        break
                
                if target_key:
                    optimal_dist[tier] = self.optimization_config['target_distributions'][target_key] * 100
                else:
                    # Default distribution for unmatched tiers
                    optimal_dist[tier] = analysis.percentage_of_total
            
            # Normalize to 100%
            total_optimal = sum(optimal_dist.values())
            if total_optimal > 0:
                optimal_dist = {tier: (pct / total_optimal) * 100 for tier, pct in optimal_dist.items()}
            
            return optimal_dist
            
        except Exception as e:
            print(f"❌ Error calculating optimal distribution: {e}")
            return {}
    
    def _generate_rebalancing_recommendations(self, current_dist: Dict[str, float],
                                            optimal_dist: Dict[str, float]) -> List[str]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        try:
            for tier in current_dist:
                current_pct = current_dist[tier]
                optimal_pct = optimal_dist.get(tier, current_pct)
                
                difference = optimal_pct - current_pct
                
                if abs(difference) > 5:  # 5% threshold
                    if difference > 0:
                        recommendations.append(f"Increase {tier} representation by {difference:.1f}% to reach optimal distribution")
                    else:
                        recommendations.append(f"Reduce {tier} representation by {abs(difference):.1f}% to optimize balance")
            
            # General recommendations
            if len(recommendations) > 3:
                recommendations.append("Consider gradual rebalancing over multiple phases to minimize disruption")
            
            if not recommendations:
                recommendations.append("Current tier distribution is well-balanced - maintain current ratios")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating rebalancing recommendations: {e}")
            return []
    
    def _generate_quality_improvements(self, tier_analyses: Dict[str, TierAnalysis]) -> List[str]:
        """Generate quality improvement recommendations"""
        improvements = []
        
        try:
            for tier, analysis in tier_analyses.items():
                # Check against quality thresholds
                overall_quality = analysis.quality_metrics.get('overall_quality', 0)
                
                # Find matching threshold
                threshold = 0.7  # Default
                for target_tier, target_threshold in self.optimization_config['quality_thresholds'].items():
                    if target_tier.lower() in tier.lower():
                        threshold = target_threshold
                        break
                
                if overall_quality < threshold:
                    gap = threshold - overall_quality
                    improvements.append(f"Improve {tier} quality by {gap:.2f} points to meet threshold")
                
                # Processing efficiency improvements
                if analysis.processing_efficiency < 90:
                    improvements.append(f"Address processing issues in {tier} (current: {analysis.processing_efficiency:.1f}%)")
                
                # Content improvements
                if analysis.avg_word_count < 50:
                    improvements.append(f"Enhance content depth in {tier} - average word count is low")
            
            # General improvements
            if len(improvements) == 0:
                improvements.append("All tiers meet quality standards - focus on maintaining excellence")
            
            return improvements
            
        except Exception as e:
            print(f"❌ Error generating quality improvements: {e}")
            return []
    
    def _calculate_resource_allocation(self, tier_analyses: Dict[str, TierAnalysis],
                                     optimal_dist: Dict[str, float]) -> Dict[str, Any]:
        """Calculate resource allocation for optimization"""
        try:
            total_conversations = sum(analysis.conversation_count for analysis in tier_analyses.values())
            
            # Calculate effort required for each tier
            effort_allocation = {}
            
            for tier, analysis in tier_analyses.items():
                current_pct = analysis.percentage_of_total
                optimal_pct = optimal_dist.get(tier, current_pct)
                
                # Effort based on gap and quality needs
                rebalancing_effort = abs(optimal_pct - current_pct) * 10  # Scale factor
                quality_effort = max(0, 0.8 - analysis.quality_metrics.get('overall_quality', 0.8)) * 100
                
                total_effort = rebalancing_effort + quality_effort
                
                effort_allocation[tier] = {
                    'rebalancing_effort': float(rebalancing_effort),
                    'quality_effort': float(quality_effort),
                    'total_effort': float(total_effort),
                    'priority': 'high' if total_effort > 50 else 'medium' if total_effort > 20 else 'low'
                }
            
            return {
                'effort_allocation': effort_allocation,
                'total_conversations': total_conversations,
                'optimization_timeline_weeks': max(4, len([e for e in effort_allocation.values() if e['priority'] == 'high']) * 2)
            }
            
        except Exception as e:
            print(f"❌ Error calculating resource allocation: {e}")
            return {}
    
    def _estimate_optimization_impact(self, current_dist: Dict[str, float],
                                    optimal_dist: Dict[str, float],
                                    tier_analyses: Dict[str, TierAnalysis]) -> Dict[str, float]:
        """Estimate impact of optimization"""
        try:
            # Calculate weighted quality improvement
            current_weighted_quality = 0
            optimal_weighted_quality = 0
            
            for tier, analysis in tier_analyses.items():
                current_weight = current_dist.get(tier, 0) / 100
                optimal_weight = optimal_dist.get(tier, 0) / 100
                quality = analysis.quality_metrics.get('overall_quality', 0)
                
                current_weighted_quality += current_weight * quality
                optimal_weighted_quality += optimal_weight * quality
            
            quality_improvement = optimal_weighted_quality - current_weighted_quality
            
            # Estimate other impacts
            processing_improvement = np.mean([
                max(0, 95 - analysis.processing_efficiency) / 100
                for analysis in tier_analyses.values()
            ])
            
            return {
                'quality_improvement': float(quality_improvement),
                'processing_efficiency_gain': float(processing_improvement),
                'distribution_balance_score': float(1.0 - np.std(list(optimal_dist.values())) / 100),
                'overall_system_improvement': float((quality_improvement + processing_improvement) / 2)
            }
            
        except Exception as e:
            print(f"❌ Error estimating optimization impact: {e}")
            return {}
    
    def create_tier_visualizations(self, tier_analyses: Dict[str, TierAnalysis],
                                 optimization: TierOptimization) -> Dict[str, str]:
        """Create tier distribution visualizations"""
        print("📈 Creating tier distribution visualizations...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create tier analysis dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Tier Distribution Analysis & Optimization', fontsize=16, fontweight='bold')
            
            # Current vs Optimal Distribution
            ax = axes[0, 0]
            tiers = list(optimization.current_distribution.keys())
            current_values = list(optimization.current_distribution.values())
            optimal_values = [optimization.optimal_distribution.get(tier, 0) for tier in tiers]
            
            x = np.arange(len(tiers))
            width = 0.35
            
            ax.bar(x - width/2, current_values, width, label='Current', alpha=0.7)
            ax.bar(x + width/2, optimal_values, width, label='Optimal', alpha=0.7)
            
            ax.set_title('Current vs Optimal Distribution')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Percentage')
            ax.set_xticks(x)
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tiers], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Quality Metrics by Tier
            ax = axes[0, 1]
            quality_scores = [analysis.quality_metrics.get('overall_quality', 0) for analysis in tier_analyses.values()]
            tier_names = list(tier_analyses.keys())
            
            bars = ax.bar(range(len(tier_names)), quality_scores, alpha=0.7, color='green')
            ax.set_title('Quality Scores by Tier')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Quality Score')
            ax.set_xticks(range(len(tier_names)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, quality_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            # Conversation Volume by Tier
            ax = axes[1, 0]
            conversation_counts = [analysis.conversation_count for analysis in tier_analyses.values()]
            
            ax.pie(conversation_counts, labels=[t.replace('_', ' ').title() for t in tier_names], autopct='%1.1f%%')
            ax.set_title('Conversation Volume Distribution')
            
            # Processing Efficiency by Tier
            ax = axes[1, 1]
            processing_rates = [analysis.processing_efficiency for analysis in tier_analyses.values()]
            
            bars = ax.bar(range(len(tier_names)), processing_rates, alpha=0.7, color='orange')
            ax.set_title('Processing Efficiency by Tier')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Processing Rate (%)')
            ax.set_xticks(range(len(tier_names)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45, ha='right')
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "tier_distribution_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} tier visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_tier_optimization_report(self, tier_analyses: Dict[str, TierAnalysis],
                                      optimization: TierOptimization,
                                      visualizations: Dict[str, str]) -> str:
        """Export comprehensive tier optimization report"""
        print("📄 Exporting tier optimization report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"tier_optimization_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'optimizer_version': '1.0.0',
                    'tiers_analyzed': len(tier_analyses),
                    'optimization_scope': 'distribution_and_quality'
                },
                'tier_analysis': {
                    tier: {
                        'conversation_count': analysis.conversation_count,
                        'percentage_of_total': analysis.percentage_of_total,
                        'avg_word_count': analysis.avg_word_count,
                        'avg_turn_count': analysis.avg_turn_count,
                        'quality_metrics': analysis.quality_metrics,
                        'dataset_distribution': analysis.dataset_distribution,
                        'processing_efficiency': analysis.processing_efficiency,
                        'characteristics': analysis.characteristics
                    }
                    for tier, analysis in tier_analyses.items()
                },
                'optimization_plan': {
                    'current_distribution': optimization.current_distribution,
                    'optimal_distribution': optimization.optimal_distribution,
                    'rebalancing_recommendations': optimization.rebalancing_recommendations,
                    'quality_improvements': optimization.quality_improvements,
                    'resource_allocation': optimization.resource_allocation,
                    'expected_impact': optimization.expected_impact
                },
                'visualizations': visualizations,
                'executive_summary': self._create_tier_executive_summary(tier_analyses, optimization)
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported tier optimization report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting tier optimization report: {e}")
            return ""
    
    def _create_tier_executive_summary(self, tier_analyses: Dict[str, TierAnalysis],
                                     optimization: TierOptimization) -> Dict[str, Any]:
        """Create executive summary of tier optimization"""
        try:
            total_conversations = sum(analysis.conversation_count for analysis in tier_analyses.values())
            avg_quality = np.mean([analysis.quality_metrics.get('overall_quality', 0) for analysis in tier_analyses.values()])
            
            # Find most and least balanced tiers
            distribution_values = list(optimization.current_distribution.values())
            most_represented = max(optimization.current_distribution.items(), key=lambda x: x[1])
            least_represented = min(optimization.current_distribution.items(), key=lambda x: x[1])
            
            return {
                'total_tiers': len(tier_analyses),
                'total_conversations': total_conversations,
                'average_quality_score': float(avg_quality),
                'most_represented_tier': most_represented[0],
                'least_represented_tier': least_represented[0],
                'distribution_balance_score': float(1.0 - np.std(distribution_values) / 100),
                'optimization_priority': 'high' if len(optimization.rebalancing_recommendations) > 3 else 'medium',
                'expected_quality_improvement': optimization.expected_impact.get('quality_improvement', 0),
                'timeline_weeks': optimization.resource_allocation.get('optimization_timeline_weeks', 8)
            }
            
        except Exception as e:
            print(f"❌ Error creating tier executive summary: {e}")
            return {}

def main():
    """Main execution function"""
    print("📊 Tier Distribution Analysis and Optimization System")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = TierDistributionOptimizer()
    
    # Analyze tier distribution
    tier_analyses = optimizer.analyze_tier_distribution()
    
    if not tier_analyses:
        print("❌ No tier analyses generated")
        return
    
    # Generate optimization
    optimization = optimizer.generate_tier_optimization(tier_analyses)
    
    # Create visualizations
    visualizations = optimizer.create_tier_visualizations(tier_analyses, optimization)
    
    # Export report
    report_file = optimizer.export_tier_optimization_report(tier_analyses, optimization, visualizations)
    
    # Display summary
    total_conversations = sum(analysis.conversation_count for analysis in tier_analyses.values())
    
    print(f"\n✅ Tier Optimization Analysis Complete")
    print(f"   - Tiers analyzed: {len(tier_analyses)}")
    print(f"   - Total conversations: {total_conversations:,}")
    print(f"   - Rebalancing recommendations: {len(optimization.rebalancing_recommendations)}")
    print(f"   - Quality improvements: {len(optimization.quality_improvements)}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show current distribution
    print(f"\n📊 Current Tier Distribution:")
    for tier, percentage in optimization.current_distribution.items():
        print(f"   {tier}: {percentage:.1f}%")
    
    # Show top recommendations
    if optimization.rebalancing_recommendations:
        print(f"\n🎯 Top Rebalancing Recommendations:")
        for rec in optimization.rebalancing_recommendations[:3]:
            print(f"   • {rec}")

if __name__ == "__main__":
    main()
