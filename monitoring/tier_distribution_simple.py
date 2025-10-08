#!/usr/bin/env python3
"""
Simple Tier Distribution Analysis and Optimization System
Analyzes tier distribution patterns and provides optimization recommendations
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleTierDistributionOptimizer:
    """Simple tier distribution analysis and optimization system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/tier_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_tier_distribution(self) -> Dict[str, Any]:
        """Analyze tier distribution"""
        print("📊 Analyzing tier distribution patterns...")
        
        try:
            # Get tier data
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                tier,
                COUNT(*) as conversation_count,
                AVG(word_count) as avg_word_count,
                AVG(turn_count) as avg_turn_count,
                dataset_source
            FROM conversations 
            WHERE tier IS NOT NULL AND tier != ''
            GROUP BY tier, dataset_source
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("❌ No tier data found")
                return {}
            
            # Analyze tier distribution
            tier_summary = df.groupby('tier').agg({
                'conversation_count': 'sum',
                'avg_word_count': 'mean',
                'avg_turn_count': 'mean'
            }).reset_index()
            
            # Calculate percentages
            total_conversations = tier_summary['conversation_count'].sum()
            tier_summary['percentage'] = (tier_summary['conversation_count'] / total_conversations) * 100
            
            # Create analysis results
            analysis_results = {
                'total_conversations': int(total_conversations),
                'tier_distribution': {},
                'optimization_recommendations': [],
                'quality_insights': []
            }
            
            # Process each tier
            for _, row in tier_summary.iterrows():
                tier_name = row['tier']
                analysis_results['tier_distribution'][tier_name] = {
                    'conversation_count': int(row['conversation_count']),
                    'percentage': float(row['percentage']),
                    'avg_word_count': float(row['avg_word_count']),
                    'avg_turn_count': float(row['avg_turn_count'])
                }
            
            # Generate recommendations
            analysis_results['optimization_recommendations'] = self._generate_recommendations(tier_summary)
            analysis_results['quality_insights'] = self._generate_quality_insights(tier_summary)
            
            print(f"✅ Analyzed {len(tier_summary)} tiers with {total_conversations:,} total conversations")
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error analyzing tier distribution: {e}")
            return {}
    
    def _generate_recommendations(self, tier_summary: pd.DataFrame) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            # Check for imbalanced distribution
            percentages = tier_summary['percentage'].values
            if len(percentages) > 1:
                std_dev = np.std(percentages)
                if std_dev > 20:  # High variation
                    recommendations.append("High tier distribution imbalance detected - consider rebalancing")
                
                # Check for dominant tiers
                max_pct = percentages.max()
                if max_pct > 60:
                    dominant_tier = tier_summary.loc[tier_summary['percentage'].idxmax(), 'tier']
                    recommendations.append(f"Tier '{dominant_tier}' dominates with {max_pct:.1f}% - consider diversification")
                
                # Check for underrepresented tiers
                min_pct = percentages.min()
                if min_pct < 5:
                    underrep_tier = tier_summary.loc[tier_summary['percentage'].idxmin(), 'tier']
                    recommendations.append(f"Tier '{underrep_tier}' is underrepresented with {min_pct:.1f}% - consider expansion")
            
            # Content quality recommendations
            for _, row in tier_summary.iterrows():
                tier_name = row['tier']
                avg_words = row['avg_word_count']
                
                if avg_words < 50:
                    recommendations.append(f"Tier '{tier_name}' has low word count ({avg_words:.1f}) - enhance content depth")
                elif avg_words > 500:
                    recommendations.append(f"Tier '{tier_name}' has high word count ({avg_words:.1f}) - monitor for verbosity")
            
            if not recommendations:
                recommendations.append("Tier distribution appears well-balanced - maintain current structure")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _generate_quality_insights(self, tier_summary: pd.DataFrame) -> List[str]:
        """Generate quality insights"""
        insights = []
        
        try:
            # Overall insights
            total_tiers = len(tier_summary)
            insights.append(f"System contains {total_tiers} distinct tiers")
            
            # Content insights
            avg_words_overall = tier_summary['avg_word_count'].mean()
            avg_turns_overall = tier_summary['avg_turn_count'].mean()
            
            insights.append(f"Average conversation length: {avg_words_overall:.1f} words, {avg_turns_overall:.1f} turns")
            
            # Tier-specific insights
            best_content_tier = tier_summary.loc[tier_summary['avg_word_count'].idxmax()]
            insights.append(f"Richest content tier: '{best_content_tier['tier']}' with {best_content_tier['avg_word_count']:.1f} words average")
            
            largest_tier = tier_summary.loc[tier_summary['conversation_count'].idxmax()]
            insights.append(f"Largest tier: '{largest_tier['tier']}' with {largest_tier['conversation_count']:,} conversations ({largest_tier['percentage']:.1f}%)")
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating quality insights: {e}")
            return ["Error generating insights"]
    
    def create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Create tier distribution visualizations"""
        print("📈 Creating tier distribution visualizations...")
        
        viz_files = {}
        
        try:
            if not analysis_results.get('tier_distribution'):
                return viz_files
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Tier Distribution Analysis Dashboard', fontsize=16, fontweight='bold')
            
            tier_data = analysis_results['tier_distribution']
            tier_names = list(tier_data.keys())
            
            # Tier distribution pie chart
            ax = axes[0, 0]
            percentages = [tier_data[tier]['percentage'] for tier in tier_names]
            ax.pie(percentages, labels=[t.replace('_', ' ').title() for t in tier_names], autopct='%1.1f%%')
            ax.set_title('Tier Distribution by Percentage')
            
            # Conversation count by tier
            ax = axes[0, 1]
            counts = [tier_data[tier]['conversation_count'] for tier in tier_names]
            bars = ax.bar(range(len(tier_names)), counts, alpha=0.7)
            ax.set_title('Conversation Count by Tier')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Conversation Count')
            ax.set_xticks(range(len(tier_names)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count:,}', ha='center', va='bottom', fontsize=8)
            
            # Average word count by tier
            ax = axes[1, 0]
            word_counts = [tier_data[tier]['avg_word_count'] for tier in tier_names]
            ax.bar(range(len(tier_names)), word_counts, alpha=0.7, color='orange')
            ax.set_title('Average Word Count by Tier')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Average Words')
            ax.set_xticks(range(len(tier_names)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Average turn count by tier
            ax = axes[1, 1]
            turn_counts = [tier_data[tier]['avg_turn_count'] for tier in tier_names]
            ax.bar(range(len(tier_names)), turn_counts, alpha=0.7, color='green')
            ax.set_title('Average Turn Count by Tier')
            ax.set_xlabel('Tiers')
            ax.set_ylabel('Average Turns')
            ax.set_xticks(range(len(tier_names)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "tier_distribution_simple_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} tier visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_report(self, analysis_results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Export tier distribution report"""
        print("📄 Exporting tier distribution report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"tier_distribution_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'analysis_type': 'tier_distribution'
                },
                'analysis_results': analysis_results,
                'visualizations': visualizations
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported tier distribution report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return ""

def main():
    """Main execution function"""
    print("📊 Simple Tier Distribution Analysis and Optimization System")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = SimpleTierDistributionOptimizer()
    
    # Analyze tier distribution
    analysis_results = optimizer.analyze_tier_distribution()
    
    if not analysis_results:
        print("❌ No analysis results generated")
        return
    
    # Create visualizations
    visualizations = optimizer.create_visualizations(analysis_results)
    
    # Export report
    report_file = optimizer.export_report(analysis_results, visualizations)
    
    # Display summary
    print(f"\n✅ Tier Distribution Analysis Complete")
    print(f"   - Total conversations: {analysis_results.get('total_conversations', 0):,}")
    print(f"   - Tiers analyzed: {len(analysis_results.get('tier_distribution', {}))}")
    print(f"   - Recommendations: {len(analysis_results.get('optimization_recommendations', []))}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show tier distribution
    tier_dist = analysis_results.get('tier_distribution', {})
    if tier_dist:
        print(f"\n📊 Tier Distribution:")
        for tier, data in tier_dist.items():
            print(f"   {tier}: {data['conversation_count']:,} conversations ({data['percentage']:.1f}%)")
    
    # Show top recommendations
    recommendations = analysis_results.get('optimization_recommendations', [])
    if recommendations:
        print(f"\n🎯 Top Recommendations:")
        for rec in recommendations[:3]:
            print(f"   • {rec}")

if __name__ == "__main__":
    main()
