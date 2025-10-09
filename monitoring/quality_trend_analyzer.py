#!/usr/bin/env python3
"""
Quality Trend Analysis and Reporting System
Analyzes quality trends over time and generates comprehensive reports
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
    
@dataclass
class TrendReport:
    """Comprehensive trend report"""
    period: str
    overall_trend: str
    key_insights: List[str]
    recommendations: List[str]
    metrics_summary: Dict[str, Any]
    statistical_tests: Dict[str, Any]

class QualityTrendAnalyzer:
    """Enterprise-grade quality trend analysis system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_trends")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics to track
        self.quality_metrics = [
            'overall_quality',
            'therapeutic_accuracy',
            'conversation_coherence', 
            'emotional_authenticity',
            'clinical_compliance',
            'personality_consistency',
            'language_quality',
            'safety_score'
        ]
        
        # Trend analysis parameters
        self.trend_periods = ['daily', 'weekly', 'monthly', 'quarterly']
        self.min_data_points = 5
        self.significance_threshold = 0.05
        
    def analyze_quality_trends(self, period: str = 'weekly', 
                             days_back: int = 90) -> Dict[str, QualityTrend]:
        """Analyze quality trends over specified period"""
        print(f"🔍 Analyzing quality trends for {period} period over {days_back} days...")
        
        try:
            # Get quality data from database
            quality_data = self._get_quality_data(days_back)
            
            if not quality_data:
                print("⚠️ No quality data found for trend analysis")
                return {}
            
            # Group data by period
            grouped_data = self._group_by_period(quality_data, period)
            
            # Analyze trends for each metric
            trends = {}
            for metric in self.quality_metrics:
                trend = self._analyze_metric_trend(grouped_data, metric, period)
                if trend:
                    trends[metric] = trend
            
            print(f"✅ Analyzed trends for {len(trends)} quality metrics")
            return trends
            
        except Exception as e:
            print(f"❌ Error analyzing quality trends: {e}")
            return {}
    
    def _get_quality_data(self, days_back: int) -> List[Dict]:
        """Get quality data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT 
                created_at,
                overall_quality,
                therapeutic_accuracy,
                conversation_coherence,
                emotional_authenticity,
                clinical_compliance,
                personality_consistency,
                language_quality,
                safety_score,
                tier,
                dataset_name
            FROM conversations 
            WHERE created_at >= ? 
            AND overall_quality IS NOT NULL
            ORDER BY created_at
            """
            
            cursor = conn.execute(query, (date_threshold.isoformat(),))
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                record['created_at'] = datetime.fromisoformat(record['created_at'])
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Error getting quality data: {e}")
            return []
    
    def _group_by_period(self, data: List[Dict], period: str) -> Dict[str, List[Dict]]:
        """Group data by time period"""
        grouped = {}
        
        for record in data:
            timestamp = record['created_at']
            
            if period == 'daily':
                key = timestamp.strftime('%Y-%m-%d')
            elif period == 'weekly':
                # Get Monday of the week
                monday = timestamp - timedelta(days=timestamp.weekday())
                key = monday.strftime('%Y-%m-%d')
            elif period == 'monthly':
                key = timestamp.strftime('%Y-%m')
            elif period == 'quarterly':
                quarter = (timestamp.month - 1) // 3 + 1
                key = f"{timestamp.year}-Q{quarter}"
            else:
                key = timestamp.strftime('%Y-%m-%d')
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        
        return grouped
    
    def _analyze_metric_trend(self, grouped_data: Dict[str, List[Dict]], 
                            metric: str, period: str) -> Optional[QualityTrend]:
        """Analyze trend for specific metric"""
        try:
            # Extract metric values and timestamps
            period_values = []
            timestamps = []
            
            for period_key, records in sorted(grouped_data.items()):
                # Calculate average for this period
                values = [r[metric] for r in records if r[metric] is not None]
                if values:
                    period_values.append(np.mean(values))
                    # Convert period key back to datetime
                    if period == 'daily':
                        timestamps.append(datetime.strptime(period_key, '%Y-%m-%d'))
                    elif period == 'weekly':
                        timestamps.append(datetime.strptime(period_key, '%Y-%m-%d'))
                    elif period == 'monthly':
                        timestamps.append(datetime.strptime(f"{period_key}-01", '%Y-%m-%d'))
                    elif period == 'quarterly':
                        year, quarter = period_key.split('-Q')
                        month = (int(quarter) - 1) * 3 + 1
                        timestamps.append(datetime(int(year), month, 1))
            
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
    
    def generate_trend_report(self, trends: Dict[str, QualityTrend], 
                            period: str = 'weekly') -> TrendReport:
        """Generate comprehensive trend report"""
        print(f"📊 Generating trend report for {period} analysis...")
        
        try:
            # Overall trend assessment
            overall_trend = self._assess_overall_trend(trends)
            
            # Key insights
            insights = self._extract_key_insights(trends)
            
            # Recommendations
            recommendations = self._generate_recommendations(trends)
            
            # Metrics summary
            metrics_summary = self._create_metrics_summary(trends)
            
            # Statistical tests
            statistical_tests = self._perform_statistical_tests(trends)
            
            report = TrendReport(
                period=period,
                overall_trend=overall_trend,
                key_insights=insights,
                recommendations=recommendations,
                metrics_summary=metrics_summary,
                statistical_tests=statistical_tests
            )
            
            print(f"✅ Generated comprehensive trend report with {len(insights)} insights")
            return report
            
        except Exception as e:
            print(f"❌ Error generating trend report: {e}")
            return TrendReport(
                period=period,
                overall_trend="unknown",
                key_insights=[],
                recommendations=[],
                metrics_summary={},
                statistical_tests={}
            )
    
    def _assess_overall_trend(self, trends: Dict[str, QualityTrend]) -> str:
        """Assess overall quality trend"""
        if not trends:
            return "insufficient_data"
        
        improving_count = 0
        declining_count = 0
        stable_count = 0
        
        for trend in trends.values():
            if trend.statistical_significance < self.significance_threshold:
                if trend.trend_direction == "improving":
                    improving_count += 1
                elif trend.trend_direction == "declining":
                    declining_count += 1
                else:
                    stable_count += 1
        
        total_significant = improving_count + declining_count + stable_count
        
        if total_significant == 0:
            return "stable"
        elif improving_count > declining_count:
            return "improving"
        elif declining_count > improving_count:
            return "declining"
        else:
            return "mixed"
    
    def _extract_key_insights(self, trends: Dict[str, QualityTrend]) -> List[str]:
        """Extract key insights from trends"""
        insights = []
        
        # Find strongest trends
        significant_trends = [
            (name, trend) for name, trend in trends.items()
            if trend.statistical_significance < self.significance_threshold
        ]
        
        # Sort by trend strength
        significant_trends.sort(key=lambda x: x[1].trend_strength, reverse=True)
        
        for name, trend in significant_trends[:5]:  # Top 5 insights
            direction = trend.trend_direction
            strength = trend.trend_strength
            
            if direction == "improving":
                insights.append(f"{name.replace('_', ' ').title()} shows significant improvement (R² = {strength:.3f})")
            elif direction == "declining":
                insights.append(f"{name.replace('_', ' ').title()} shows concerning decline (R² = {strength:.3f})")
            else:
                insights.append(f"{name.replace('_', ' ').title()} remains stable with strong consistency (R² = {strength:.3f})")
        
        # Add volatility insights
        volatile_metrics = [
            name for name, trend in trends.items()
            if np.std(trend.values) > 0.1  # High standard deviation
        ]
        
        if volatile_metrics:
            insights.append(f"High volatility detected in: {', '.join(volatile_metrics)}")
        
        return insights
    
    def _generate_recommendations(self, trends: Dict[str, QualityTrend]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Find declining metrics
        declining_metrics = [
            name for name, trend in trends.items()
            if trend.trend_direction == "declining" and 
               trend.statistical_significance < self.significance_threshold
        ]
        
        for metric in declining_metrics:
            if metric == "therapeutic_accuracy":
                recommendations.append("Review and enhance clinical training data quality")
            elif metric == "conversation_coherence":
                recommendations.append("Implement conversation flow validation and improvement")
            elif metric == "emotional_authenticity":
                recommendations.append("Enhance emotional intelligence training datasets")
            elif metric == "clinical_compliance":
                recommendations.append("Strengthen clinical guideline adherence validation")
            elif metric == "safety_score":
                recommendations.append("URGENT: Review safety protocols and crisis detection systems")
        
        # Find improving metrics to reinforce
        improving_metrics = [
            name for name, trend in trends.items()
            if trend.trend_direction == "improving" and 
               trend.statistical_significance < self.significance_threshold
        ]
        
        if improving_metrics:
            recommendations.append(f"Continue successful practices that improved: {', '.join(improving_metrics)}")
        
        # General recommendations
        if len(declining_metrics) > len(improving_metrics):
            recommendations.append("Consider comprehensive quality assurance review")
        
        return recommendations
    
    def _create_metrics_summary(self, trends: Dict[str, QualityTrend]) -> Dict[str, Any]:
        """Create metrics summary"""
        summary = {}
        
        for name, trend in trends.items():
            summary[name] = {
                'current_value': trend.values[-1] if trend.values else 0,
                'trend_direction': trend.trend_direction,
                'trend_strength': trend.trend_strength,
                'statistical_significance': trend.statistical_significance,
                'is_significant': trend.statistical_significance < self.significance_threshold,
                'volatility': np.std(trend.values) if trend.values else 0,
                'min_value': min(trend.values) if trend.values else 0,
                'max_value': max(trend.values) if trend.values else 0,
                'mean_value': np.mean(trend.values) if trend.values else 0
            }
        
        return summary
    
    def _perform_statistical_tests(self, trends: Dict[str, QualityTrend]) -> Dict[str, Any]:
        """Perform statistical tests on trends"""
        tests = {}
        
        for name, trend in trends.items():
            if len(trend.values) < 3:
                continue
            
            # Normality test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(trend.values)
                tests[f"{name}_normality"] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except:
                pass
            
            # Trend test (Mann-Kendall)
            try:
                tau, p_value = stats.kendalltau(range(len(trend.values)), trend.values)
                tests[f"{name}_trend"] = {
                    'test': 'Mann-Kendall',
                    'tau': tau,
                    'p_value': p_value,
                    'has_trend': p_value < 0.05
                }
            except:
                pass
        
        return tests
    
    def create_trend_visualizations(self, trends: Dict[str, QualityTrend], 
                                  period: str = 'weekly') -> Dict[str, str]:
        """Create trend visualizations"""
        print(f"📈 Creating trend visualizations for {len(trends)} metrics...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create individual metric plots
            for name, trend in trends.items():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Time series plot
                ax1.plot(trend.timestamps, trend.values, marker='o', linewidth=2, markersize=6)
                ax1.set_title(f'{name.replace("_", " ").title()} Trend Over Time', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Quality Score', fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # Add trend line
                x_numeric = np.arange(len(trend.values))
                z = np.polyfit(x_numeric, trend.values, 1)
                p = np.poly1d(z)
                ax1.plot(trend.timestamps, p(x_numeric), "--", alpha=0.8, color='red')
                
                # Distribution plot
                ax2.hist(trend.values, bins=min(10, len(trend.values)), alpha=0.7, edgecolor='black')
                ax2.set_title('Value Distribution', fontsize=12)
                ax2.set_xlabel('Quality Score', fontsize=12)
                ax2.set_ylabel('Frequency', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f"""
                Trend: {trend.trend_direction}
                Strength: {trend.trend_strength:.3f}
                P-value: {trend.statistical_significance:.3f}
                Mean: {np.mean(trend.values):.3f}
                Std: {np.std(trend.values):.3f}
                """
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                
                # Save plot
                filename = f"trend_{name}_{period}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_files[name] = str(filepath)
            
            # Create summary dashboard
            if len(trends) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Quality Trends Dashboard - {period.title()} Analysis', fontsize=16, fontweight='bold')
                
                # Overall trends comparison
                ax = axes[0, 0]
                trend_directions = [t.trend_direction for t in trends.values()]
                direction_counts = pd.Series(trend_directions).value_counts()
                ax.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%')
                ax.set_title('Trend Directions Distribution')
                
                # Trend strengths
                ax = axes[0, 1]
                strengths = [t.trend_strength for t in trends.values()]
                names = list(trends.keys())
                ax.barh(names, strengths)
                ax.set_title('Trend Strengths (R²)')
                ax.set_xlabel('R² Value')
                
                # Current values
                ax = axes[1, 0]
                current_values = [t.values[-1] if t.values else 0 for t in trends.values()]
                ax.bar(names, current_values)
                ax.set_title('Current Quality Scores')
                ax.set_ylabel('Quality Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Statistical significance
                ax = axes[1, 1]
                p_values = [t.statistical_significance for t in trends.values()]
                colors = ['green' if p < 0.05 else 'red' for p in p_values]
                ax.bar(names, p_values, color=colors)
                ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
                ax.set_title('Statistical Significance (p-values)')
                ax.set_ylabel('P-value')
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                
                # Save dashboard
                dashboard_file = self.output_dir / f"trends_dashboard_{period}.png"
                plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_trend_report(self, report: TrendReport, trends: Dict[str, QualityTrend],
                          visualizations: Dict[str, str]) -> str:
        """Export comprehensive trend report"""
        print("📄 Exporting comprehensive trend report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_trend_report_{report.period}_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period': report.period,
                    'analyzer_version': '1.0.0'
                },
                'overall_assessment': {
                    'trend': report.overall_trend,
                    'key_insights': report.key_insights,
                    'recommendations': report.recommendations
                },
                'metrics_analysis': report.metrics_summary,
                'statistical_tests': report.statistical_tests,
                'trend_data': {
                    name: {
                        'metric': trend.metric,
                        'period': trend.period,
                        'values': trend.values,
                        'timestamps': [t.isoformat() for t in trend.timestamps],
                        'trend_direction': trend.trend_direction,
                        'trend_strength': trend.trend_strength,
                        'statistical_significance': trend.statistical_significance
                    }
                    for name, trend in trends.items()
                },
                'visualizations': visualizations
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported trend report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting trend report: {e}")
            return ""

def main():
    """Main execution function"""
    print("🔍 Quality Trend Analysis and Reporting System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = QualityTrendAnalyzer()
    
    # Analyze trends for different periods
    periods = ['daily', 'weekly', 'monthly']
    
    for period in periods:
        print(f"\n📊 Analyzing {period} trends...")
        
        # Analyze trends
        trends = analyzer.analyze_quality_trends(period=period, days_back=90)
        
        if not trends:
            print(f"⚠️ No trend data available for {period} analysis")
            continue
        
        # Generate report
        report = analyzer.generate_trend_report(trends, period)
        
        # Create visualizations
        visualizations = analyzer.create_trend_visualizations(trends, period)
        
        # Export report
        report_file = analyzer.export_trend_report(report, trends, visualizations)
        
        print(f"✅ {period.title()} trend analysis complete")
        print(f"   - Trends analyzed: {len(trends)}")
        print(f"   - Overall trend: {report.overall_trend}")
        print(f"   - Key insights: {len(report.key_insights)}")
        print(f"   - Recommendations: {len(report.recommendations)}")
        print(f"   - Report saved: {report_file}")

if __name__ == "__main__":
    main()
