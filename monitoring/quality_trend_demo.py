#!/usr/bin/env python3
"""
Quality Trend Analysis Demo
Creates synthetic date distribution to demonstrate trend analysis capabilities
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
from scipy import stats
import random
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityTrend:
    """Quality trend data structure"""
    metric: str
    period: str
    values: List[float]
    timestamps: List[datetime]
    trend_direction: str
    trend_strength: float
    statistical_significance: float

class QualityTrendDemo:
    """Demo quality trend analysis with synthetic data"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_trends")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics we can derive from current data
        self.quality_metrics = [
            'conversation_length',  # Based on turn_count
            'content_richness',     # Based on word_count
            'processing_success',   # Based on processing_status
            'tier_distribution',    # Based on tier
            'language_consistency', # Based on language
        ]
        
        # Trend analysis parameters
        self.min_data_points = 5
        self.significance_threshold = 0.05
        
    def create_synthetic_trends(self, days_back: int = 30) -> Dict[str, List[Dict]]:
        """Create synthetic trend data for demonstration"""
        print(f"🎭 Creating synthetic trend data for {days_back} days...")
        
        try:
            # Get base data from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT turn_count, word_count, processing_status, tier, language
                FROM conversations 
                LIMIT 1000
            """)
            
            base_data = cursor.fetchall()
            conn.close()
            
            if not base_data:
                print("❌ No base data found")
                return {}
            
            # Create synthetic daily data
            synthetic_data = {}
            base_date = datetime.now() - timedelta(days=days_back)
            
            for day in range(days_back):
                current_date = base_date + timedelta(days=day)
                date_key = current_date.strftime('%Y-%m-%d')
                
                # Create synthetic records for this day
                daily_records = []
                num_records = random.randint(50, 200)  # Random number of conversations per day
                
                for _ in range(num_records):
                    # Pick random base record and add some variation
                    base_record = random.choice(base_data)
                    
                    # Add trends and noise
                    trend_factor = day / days_back  # Linear trend component
                    noise = random.uniform(-0.2, 0.2)  # Random noise
                    
                    record = {
                        'created_at': current_date,
                        'turn_count': max(1, int(base_record[0] * (1 + trend_factor * 0.3 + noise))),
                        'word_count': max(10, int(base_record[1] * (1 + trend_factor * 0.2 + noise))),
                        'processing_status': base_record[2],
                        'tier': base_record[3],
                        'language': base_record[4]
                    }
                    daily_records.append(record)
                
                synthetic_data[date_key] = daily_records
            
            print(f"✅ Created synthetic data for {len(synthetic_data)} days")
            return synthetic_data
            
        except Exception as e:
            print(f"❌ Error creating synthetic data: {e}")
            return {}
    
    def analyze_synthetic_trends(self, synthetic_data: Dict[str, List[Dict]], 
                               period: str = 'daily') -> Dict[str, QualityTrend]:
        """Analyze trends from synthetic data"""
        print(f"🔍 Analyzing synthetic trends for {period} period...")
        
        try:
            trends = {}
            
            for metric in self.quality_metrics:
                trend = self._analyze_synthetic_metric_trend(synthetic_data, metric, period)
                if trend:
                    trends[metric] = trend
            
            print(f"✅ Analyzed trends for {len(trends)} quality metrics")
            return trends
            
        except Exception as e:
            print(f"❌ Error analyzing synthetic trends: {e}")
            return {}
    
    def _analyze_synthetic_metric_trend(self, synthetic_data: Dict[str, List[Dict]], 
                                      metric: str, period: str) -> Optional[QualityTrend]:
        """Analyze trend for specific metric from synthetic data"""
        try:
            period_values = []
            timestamps = []
            
            for date_key, records in sorted(synthetic_data.items()):
                metric_value = self._calculate_metric_value(records, metric)
                if metric_value is not None:
                    period_values.append(metric_value)
                    timestamps.append(datetime.strptime(date_key, '%Y-%m-%d'))
            
            if len(period_values) < self.min_data_points:
                return None
            
            # Perform trend analysis
            trend_direction, trend_strength, p_value = self._calculate_trend_statistics(period_values)
            
            return QualityTrend(
                metric=metric,
                period=period,
                values=period_values,
                timestamps=timestamps,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                statistical_significance=p_value
            )
            
        except Exception as e:
            print(f"❌ Error analyzing trend for {metric}: {e}")
            return None
    
    def _calculate_metric_value(self, records: List[Dict], metric: str) -> Optional[float]:
        """Calculate metric value for a period"""
        try:
            if not records:
                return None
            
            if metric == 'conversation_length':
                # Average turn count
                turn_counts = [r['turn_count'] for r in records if r['turn_count']]
                return np.mean(turn_counts) if turn_counts else None
            
            elif metric == 'content_richness':
                # Average word count
                word_counts = [r['word_count'] for r in records if r['word_count']]
                return np.mean(word_counts) if word_counts else None
            
            elif metric == 'processing_success':
                # Percentage of successful processing
                total = len(records)
                successful = len([r for r in records if r['processing_status'] == 'processed'])
                return (successful / total) * 100 if total > 0 else None
            
            elif metric == 'tier_distribution':
                # Priority tier percentage (higher is better)
                total = len(records)
                priority = len([r for r in records if r['tier'] and 'priority' in str(r['tier'])])
                return (priority / total) * 100 if total > 0 else None
            
            elif metric == 'language_consistency':
                # Percentage of English conversations
                total = len(records)
                english = len([r for r in records if r['language'] == 'en'])
                return (english / total) * 100 if total > 0 else None
            
            return None
            
        except Exception as e:
            print(f"❌ Error calculating {metric}: {e}")
            return None
    
    def _calculate_trend_statistics(self, values: List[float]) -> Tuple[str, float, float]:
        """Calculate trend statistics using linear regression"""
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                direction = "stable"
            elif slope > 0:
                direction = "improving"
            else:
                direction = "declining"
            
            # Trend strength (R-squared)
            strength = r_value ** 2
            
            return direction, strength, p_value
            
        except Exception as e:
            print(f"❌ Error calculating trend statistics: {e}")
            return "unknown", 0.0, 1.0
    
    def create_trend_visualizations(self, trends: Dict[str, QualityTrend]) -> Dict[str, str]:
        """Create trend visualizations"""
        print(f"📈 Creating trend visualizations for {len(trends)} metrics...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create summary dashboard
            if len(trends) > 1:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('Quality Trends Dashboard - Demo Analysis', fontsize=16, fontweight='bold')
                
                # Individual trend plots
                for i, (name, trend) in enumerate(list(trends.items())[:6]):
                    row = i // 3
                    col = i % 3
                    
                    if row < 2 and col < 3:
                        ax = axes[row, col]
                        
                        # Plot trend
                        ax.plot(trend.timestamps, trend.values, marker='o', linewidth=2, markersize=4)
                        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Value', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        # Add trend line
                        x_numeric = np.arange(len(trend.values))
                        z = np.polyfit(x_numeric, trend.values, 1)
                        p = np.poly1d(z)
                        ax.plot(trend.timestamps, p(x_numeric), "--", alpha=0.8, color='red')
                        
                        # Add statistics text
                        stats_text = f"""
                        {trend.trend_direction.title()}
                        R² = {trend.trend_strength:.3f}
                        p = {trend.statistical_significance:.3f}
                        """
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                        
                        # Rotate x-axis labels
                        ax.tick_params(axis='x', rotation=45)
                
                # Remove empty subplots
                for i in range(len(trends), 6):
                    row = i // 3
                    col = i % 3
                    if row < 2 and col < 3:
                        fig.delaxes(axes[row, col])
                
                plt.tight_layout()
                
                # Save dashboard
                dashboard_file = self.output_dir / "trends_dashboard_demo.png"
                plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def generate_demo_report(self, trends: Dict[str, QualityTrend]) -> Dict[str, Any]:
        """Generate demo trend report"""
        print("📊 Generating demo trend report...")
        
        try:
            # Overall trend assessment
            improving_count = sum(1 for t in trends.values() if t.trend_direction == "improving")
            declining_count = sum(1 for t in trends.values() if t.trend_direction == "declining")
            stable_count = sum(1 for t in trends.values() if t.trend_direction == "stable")
            
            if improving_count > declining_count:
                overall_trend = "improving"
            elif declining_count > improving_count:
                overall_trend = "declining"
            else:
                overall_trend = "mixed"
            
            # Key insights
            insights = []
            significant_trends = [
                (name, trend) for name, trend in trends.items()
                if trend.statistical_significance < self.significance_threshold
            ]
            
            for name, trend in significant_trends:
                direction = trend.trend_direction
                strength = trend.trend_strength
                current_value = trend.values[-1] if trend.values else 0
                
                if direction == "improving":
                    insights.append(f"{name.replace('_', ' ').title()} shows significant improvement (R² = {strength:.3f}, Current: {current_value:.1f})")
                elif direction == "declining":
                    insights.append(f"{name.replace('_', ' ').title()} shows concerning decline (R² = {strength:.3f}, Current: {current_value:.1f})")
            
            # Recommendations
            recommendations = []
            declining_metrics = [name for name, trend in trends.items() if trend.trend_direction == "declining"]
            
            if declining_metrics:
                recommendations.append(f"Address declining metrics: {', '.join([m.replace('_', ' ') for m in declining_metrics])}")
            
            improving_metrics = [name for name, trend in trends.items() if trend.trend_direction == "improving"]
            if improving_metrics:
                recommendations.append(f"Continue successful practices for: {', '.join([m.replace('_', ' ') for m in improving_metrics])}")
            
            report = {
                'overall_trend': overall_trend,
                'key_insights': insights,
                'recommendations': recommendations,
                'metrics_summary': {
                    name: {
                        'current_value': trend.values[-1] if trend.values else 0,
                        'trend_direction': trend.trend_direction,
                        'trend_strength': trend.trend_strength,
                        'statistical_significance': trend.statistical_significance,
                        'is_significant': trend.statistical_significance < self.significance_threshold
                    }
                    for name, trend in trends.items()
                }
            }
            
            print(f"✅ Generated demo report with {len(insights)} insights")
            return report
            
        except Exception as e:
            print(f"❌ Error generating demo report: {e}")
            return {}

def main():
    """Main demo execution"""
    print("🎭 Quality Trend Analysis Demo")
    print("=" * 40)
    
    # Initialize demo
    demo = QualityTrendDemo()
    
    # Create synthetic trend data
    synthetic_data = demo.create_synthetic_trends(days_back=30)
    
    if not synthetic_data:
        print("❌ Failed to create synthetic data")
        return
    
    # Analyze trends
    trends = demo.analyze_synthetic_trends(synthetic_data, period='daily')
    
    if not trends:
        print("❌ No trends found")
        return
    
    # Generate report
    report = demo.generate_demo_report(trends)
    
    # Create visualizations
    visualizations = demo.create_trend_visualizations(trends)
    
    # Display results
    print(f"\n✅ Demo Analysis Complete")
    print(f"   - Trends analyzed: {len(trends)}")
    print(f"   - Overall trend: {report.get('overall_trend', 'unknown')}")
    print(f"   - Key insights: {len(report.get('key_insights', []))}")
    print(f"   - Recommendations: {len(report.get('recommendations', []))}")
    print(f"   - Visualizations: {len(visualizations)}")
    
    # Show key insights
    insights = report.get('key_insights', [])
    if insights:
        print("\n🔍 Key Insights:")
        for insight in insights[:3]:
            print(f"   • {insight}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations[:3]:
            print(f"   • {rec}")
    
    # Show metrics summary
    print("\n📊 Metrics Summary:")
    for name, summary in report.get('metrics_summary', {}).items():
        direction = summary['trend_direction']
        current = summary['current_value']
        significant = "✓" if summary['is_significant'] else "✗"
        print(f"   {name.replace('_', ' ').title()}: {current:.1f} ({direction}) [{significant}]")

if __name__ == "__main__":
    main()
