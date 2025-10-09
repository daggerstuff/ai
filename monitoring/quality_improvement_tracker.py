#!/usr/bin/env python3
"""
Quality Improvement Tracking System
Tracks quality improvements over time and provides actionable recommendations
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
class QualityImprovement:
    """Quality improvement tracking data"""
    metric: str
    baseline_value: float
    current_value: float
    improvement_percentage: float
    improvement_direction: str
    confidence_level: float
    recommendation: str
    
@dataclass
class ImprovementPlan:
    """Quality improvement plan"""
    metric: str
    current_status: str
    target_value: float
    timeline_days: int
    action_items: List[str]
    success_criteria: List[str]
    risk_factors: List[str]

class QualityImprovementTracker:
    """Enterprise-grade quality improvement tracking system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_improvements")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics to track
        self.quality_metrics = [
            'conversation_length',
            'content_richness', 
            'processing_efficiency',
            'tier_quality',
            'dataset_diversity'
        ]
        
        # Improvement thresholds
        self.improvement_thresholds = {
            'significant': 0.10,  # 10% improvement
            'moderate': 0.05,     # 5% improvement
            'minimal': 0.02       # 2% improvement
        }
        
        # Confidence levels
        self.confidence_threshold = 0.80
        
    def track_quality_improvements(self, baseline_days: int = 30, 
                                 current_days: int = 7) -> Dict[str, QualityImprovement]:
        """Track quality improvements between baseline and current periods"""
        print(f"📈 Tracking quality improvements (baseline: {baseline_days} days, current: {current_days} days)...")
        
        try:
            # Get baseline and current data
            baseline_data = self._get_period_data(baseline_days + current_days, baseline_days)
            current_data = self._get_period_data(current_days, 0)
            
            if not baseline_data or not current_data:
                print("❌ Insufficient data for improvement tracking")
                return {}
            
            # Calculate improvements for each metric
            improvements = {}
            
            for metric in self.quality_metrics:
                improvement = self._calculate_improvement(
                    baseline_data, current_data, metric
                )
                if improvement:
                    improvements[metric] = improvement
            
            print(f"✅ Tracked improvements for {len(improvements)} quality metrics")
            return improvements
            
        except Exception as e:
            print(f"❌ Error tracking quality improvements: {e}")
            return {}
    
    def _get_period_data(self, days_back: int, days_offset: int = 0) -> List[Dict]:
        """Get data for a specific time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now() - timedelta(days=days_offset)
            start_date = end_date - timedelta(days=days_back)
            
            query = """
            SELECT 
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status,
                created_at
            FROM conversations 
            WHERE created_at >= ? AND created_at <= ?
            AND turn_count IS NOT NULL 
            AND word_count IS NOT NULL
            """
            
            cursor = conn.execute(query, (start_date.isoformat(), end_date.isoformat()))
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                if record['created_at']:
                    record['created_at'] = datetime.fromisoformat(record['created_at'])
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Error getting period data: {e}")
            return []
    
    def _calculate_improvement(self, baseline_data: List[Dict], current_data: List[Dict], 
                             metric: str) -> Optional[QualityImprovement]:
        """Calculate improvement for a specific metric"""
        try:
            # Calculate baseline and current values
            baseline_value = self._calculate_metric_value(baseline_data, metric)
            current_value = self._calculate_metric_value(current_data, metric)
            
            if baseline_value is None or current_value is None:
                return None
            
            # Calculate improvement percentage
            if baseline_value == 0:
                improvement_percentage = 0.0
            else:
                improvement_percentage = ((current_value - baseline_value) / baseline_value) * 100
            
            # Determine improvement direction
            if improvement_percentage > self.improvement_thresholds['minimal']:
                direction = "improving"
            elif improvement_percentage < -self.improvement_thresholds['minimal']:
                direction = "declining"
            else:
                direction = "stable"
            
            # Calculate confidence level (simplified)
            confidence_level = min(0.95, 0.5 + abs(improvement_percentage) / 100)
            
            # Generate recommendation
            recommendation = self._generate_improvement_recommendation(
                metric, improvement_percentage, direction
            )
            
            return QualityImprovement(
                metric=metric,
                baseline_value=baseline_value,
                current_value=current_value,
                improvement_percentage=improvement_percentage,
                improvement_direction=direction,
                confidence_level=confidence_level,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"❌ Error calculating improvement for {metric}: {e}")
            return None
    
    def _calculate_metric_value(self, data: List[Dict], metric: str) -> Optional[float]:
        """Calculate metric value for a dataset"""
        try:
            if not data:
                return None
            
            if metric == 'conversation_length':
                values = [r['turn_count'] for r in data if r['turn_count']]
                return np.mean(values) if values else None
            
            elif metric == 'content_richness':
                values = [r['word_count'] for r in data if r['word_count']]
                return np.mean(values) if values else None
            
            elif metric == 'processing_efficiency':
                total = len(data)
                successful = len([r for r in data if r['processing_status'] == 'processed'])
                return (successful / total) * 100 if total > 0 else None
            
            elif metric == 'tier_quality':
                total = len(data)
                priority = len([r for r in data if r['tier'] and 'priority' in str(r['tier'])])
                return (priority / total) * 100 if total > 0 else None
            
            elif metric == 'dataset_diversity':
                unique_datasets = len(set(r['dataset_source'] for r in data if r['dataset_source']))
                return float(unique_datasets)
            
            return None
            
        except Exception as e:
            print(f"❌ Error calculating {metric}: {e}")
            return None
    
    def _generate_improvement_recommendation(self, metric: str, improvement_percentage: float, 
                                           direction: str) -> str:
        """Generate improvement recommendation"""
        try:
            if direction == "improving":
                if improvement_percentage > self.improvement_thresholds['significant']:
                    return f"Excellent progress! Continue current practices for {metric.replace('_', ' ')}"
                elif improvement_percentage > self.improvement_thresholds['moderate']:
                    return f"Good improvement in {metric.replace('_', ' ')}. Consider scaling successful approaches"
                else:
                    return f"Slight improvement in {metric.replace('_', ' ')}. Monitor trends and maintain focus"
            
            elif direction == "declining":
                if abs(improvement_percentage) > self.improvement_thresholds['significant']:
                    return f"URGENT: Significant decline in {metric.replace('_', ' ')}. Immediate intervention required"
                elif abs(improvement_percentage) > self.improvement_thresholds['moderate']:
                    return f"Concerning decline in {metric.replace('_', ' ')}. Review and adjust processes"
                else:
                    return f"Minor decline in {metric.replace('_', ' ')}. Monitor closely and investigate causes"
            
            else:  # stable
                return f"{metric.replace('_', ' ').title()} remains stable. Consider optimization opportunities"
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
            return "Unable to generate recommendation"
    
    def create_improvement_plans(self, improvements: Dict[str, QualityImprovement]) -> Dict[str, ImprovementPlan]:
        """Create improvement plans for declining or stable metrics"""
        print("📋 Creating improvement plans...")
        
        try:
            plans = {}
            
            for metric, improvement in improvements.items():
                if improvement.improvement_direction in ["declining", "stable"]:
                    plan = self._create_metric_improvement_plan(metric, improvement)
                    if plan:
                        plans[metric] = plan
            
            print(f"✅ Created {len(plans)} improvement plans")
            return plans
            
        except Exception as e:
            print(f"❌ Error creating improvement plans: {e}")
            return {}
    
    def _create_metric_improvement_plan(self, metric: str, 
                                      improvement: QualityImprovement) -> Optional[ImprovementPlan]:
        """Create improvement plan for specific metric"""
        try:
            # Determine target value
            if improvement.improvement_direction == "declining":
                target_value = improvement.baseline_value * 1.1  # 10% above baseline
                timeline_days = 30
            else:  # stable
                target_value = improvement.current_value * 1.05  # 5% improvement
                timeline_days = 60
            
            # Generate action items based on metric
            action_items = self._generate_action_items(metric, improvement)
            
            # Generate success criteria
            success_criteria = self._generate_success_criteria(metric, target_value)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(metric, improvement)
            
            return ImprovementPlan(
                metric=metric,
                current_status=improvement.improvement_direction,
                target_value=target_value,
                timeline_days=timeline_days,
                action_items=action_items,
                success_criteria=success_criteria,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            print(f"❌ Error creating improvement plan for {metric}: {e}")
            return None
    
    def _generate_action_items(self, metric: str, improvement: QualityImprovement) -> List[str]:
        """Generate action items for metric improvement"""
        actions = []
        
        if metric == 'conversation_length':
            actions = [
                "Review conversation templates for depth and engagement",
                "Implement multi-turn dialogue training",
                "Analyze successful long conversations for patterns",
                "Provide conversation extension guidelines"
            ]
        elif metric == 'content_richness':
            actions = [
                "Enhance content quality guidelines",
                "Implement detailed response training",
                "Review and improve information density",
                "Add content richness validation checks"
            ]
        elif metric == 'processing_efficiency':
            actions = [
                "Review processing pipeline for bottlenecks",
                "Implement error handling improvements",
                "Optimize data validation processes",
                "Add processing monitoring and alerts"
            ]
        elif metric == 'tier_quality':
            actions = [
                "Review tier classification criteria",
                "Improve high-quality data source identification",
                "Implement tier promotion processes",
                "Add quality-based tier validation"
            ]
        elif metric == 'dataset_diversity':
            actions = [
                "Identify and integrate new data sources",
                "Review dataset balance and representation",
                "Implement diversity metrics tracking",
                "Add dataset variety validation"
            ]
        
        return actions
    
    def _generate_success_criteria(self, metric: str, target_value: float) -> List[str]:
        """Generate success criteria for improvement plan"""
        criteria = [
            f"Achieve target {metric.replace('_', ' ')} value of {target_value:.2f}",
            f"Maintain improvement for at least 2 weeks",
            f"Show consistent upward trend in {metric.replace('_', ' ')}",
            "No significant regression in other quality metrics"
        ]
        
        return criteria
    
    def _identify_risk_factors(self, metric: str, improvement: QualityImprovement) -> List[str]:
        """Identify risk factors for improvement plan"""
        risks = []
        
        if improvement.confidence_level < self.confidence_threshold:
            risks.append("Low confidence in current trend analysis")
        
        if abs(improvement.improvement_percentage) > 20:
            risks.append("High volatility in metric values")
        
        # Metric-specific risks
        if metric == 'processing_efficiency':
            risks.append("Processing changes may affect other system components")
        elif metric == 'dataset_diversity':
            risks.append("New data sources may introduce quality variations")
        
        risks.append("Resource constraints may limit implementation speed")
        risks.append("External factors may influence metric performance")
        
        return risks
    
    def create_improvement_visualizations(self, improvements: Dict[str, QualityImprovement]) -> Dict[str, str]:
        """Create improvement tracking visualizations"""
        print("📈 Creating improvement tracking visualizations...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create improvement dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Quality Improvement Tracking Dashboard', fontsize=16, fontweight='bold')
            
            # Improvement percentages
            ax = axes[0, 0]
            metrics = list(improvements.keys())
            percentages = [imp.improvement_percentage for imp in improvements.values()]
            colors = ['green' if p > 0 else 'red' if p < -2 else 'orange' for p in percentages]
            
            bars = ax.bar(range(len(metrics)), percentages, color=colors, alpha=0.7)
            ax.set_title('Improvement Percentages by Metric')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Improvement %')
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                       f'{percentage:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            # Current vs Baseline values
            ax = axes[0, 1]
            baseline_values = [imp.baseline_value for imp in improvements.values()]
            current_values = [imp.current_value for imp in improvements.values()]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7)
            ax.bar(x + width/2, current_values, width, label='Current', alpha=0.7)
            
            ax.set_title('Current vs Baseline Values')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Confidence levels
            ax = axes[1, 0]
            confidence_levels = [imp.confidence_level for imp in improvements.values()]
            bars = ax.bar(range(len(metrics)), confidence_levels, alpha=0.7)
            ax.set_title('Confidence Levels')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Confidence Level')
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.axhline(y=self.confidence_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Improvement directions
            ax = axes[1, 1]
            directions = [imp.improvement_direction for imp in improvements.values()]
            direction_counts = pd.Series(directions).value_counts()
            colors = {'improving': 'green', 'declining': 'red', 'stable': 'orange'}
            pie_colors = [colors.get(d, 'gray') for d in direction_counts.index]
            
            ax.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', colors=pie_colors)
            ax.set_title('Improvement Directions Distribution')
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "improvement_tracking_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} improvement visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_improvement_report(self, improvements: Dict[str, QualityImprovement],
                                plans: Dict[str, ImprovementPlan],
                                visualizations: Dict[str, str]) -> str:
        """Export comprehensive improvement tracking report"""
        print("📄 Exporting improvement tracking report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_improvement_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'tracker_version': '1.0.0',
                    'total_metrics_tracked': len(improvements),
                    'improvement_plans_created': len(plans)
                },
                'improvement_tracking': {
                    metric: {
                        'baseline_value': imp.baseline_value,
                        'current_value': imp.current_value,
                        'improvement_percentage': imp.improvement_percentage,
                        'improvement_direction': imp.improvement_direction,
                        'confidence_level': imp.confidence_level,
                        'recommendation': imp.recommendation
                    }
                    for metric, imp in improvements.items()
                },
                'improvement_plans': {
                    metric: {
                        'current_status': plan.current_status,
                        'target_value': plan.target_value,
                        'timeline_days': plan.timeline_days,
                        'action_items': plan.action_items,
                        'success_criteria': plan.success_criteria,
                        'risk_factors': plan.risk_factors
                    }
                    for metric, plan in plans.items()
                },
                'visualizations': visualizations,
                'executive_summary': self._create_executive_summary(improvements, plans)
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported improvement report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting improvement report: {e}")
            return ""
    
    def _create_executive_summary(self, improvements: Dict[str, QualityImprovement],
                                plans: Dict[str, ImprovementPlan]) -> Dict[str, Any]:
        """Create executive summary of improvements"""
        try:
            improving_metrics = [m for m, i in improvements.items() if i.improvement_direction == "improving"]
            declining_metrics = [m for m, i in improvements.items() if i.improvement_direction == "declining"]
            stable_metrics = [m for m, i in improvements.items() if i.improvement_direction == "stable"]
            
            avg_improvement = np.mean([i.improvement_percentage for i in improvements.values()])
            avg_confidence = np.mean([i.confidence_level for i in improvements.values()])
            
            return {
                'overall_trend': 'improving' if avg_improvement > 2 else 'declining' if avg_improvement < -2 else 'stable',
                'average_improvement_percentage': float(avg_improvement),
                'average_confidence_level': float(avg_confidence),
                'metrics_improving': len(improving_metrics),
                'metrics_declining': len(declining_metrics),
                'metrics_stable': len(stable_metrics),
                'improvement_plans_needed': len(plans),
                'key_recommendations': [
                    f"Focus on {len(declining_metrics)} declining metrics" if declining_metrics else "Maintain current performance",
                    f"Scale successful practices from {len(improving_metrics)} improving metrics" if improving_metrics else "Implement improvement initiatives",
                    f"Monitor {len(stable_metrics)} stable metrics for optimization opportunities" if stable_metrics else "Continue monitoring"
                ]
            }
            
        except Exception as e:
            print(f"❌ Error creating executive summary: {e}")
            return {}

def main():
    """Main execution function"""
    print("📈 Quality Improvement Tracking System")
    print("=" * 45)
    
    # Initialize tracker
    tracker = QualityImprovementTracker()
    
    # Track improvements
    improvements = tracker.track_quality_improvements(baseline_days=30, current_days=7)
    
    if not improvements:
        print("❌ No improvement data found")
        return
    
    # Create improvement plans
    plans = tracker.create_improvement_plans(improvements)
    
    # Create visualizations
    visualizations = tracker.create_improvement_visualizations(improvements)
    
    # Export report
    report_file = tracker.export_improvement_report(improvements, plans, visualizations)
    
    # Display summary
    print(f"\n✅ Improvement Tracking Complete")
    print(f"   - Metrics tracked: {len(improvements)}")
    print(f"   - Improvement plans: {len(plans)}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show key findings
    print("\n🔍 Key Findings:")
    for metric, improvement in improvements.items():
        direction_icon = "📈" if improvement.improvement_direction == "improving" else "📉" if improvement.improvement_direction == "declining" else "➡️"
        print(f"   {direction_icon} {metric.replace('_', ' ').title()}: {improvement.improvement_percentage:+.1f}% (confidence: {improvement.confidence_level:.2f})")
    
    # Show recommendations
    if improvements:
        print("\n💡 Top Recommendations:")
        for metric, improvement in list(improvements.items())[:3]:
            print(f"   • {improvement.recommendation}")

if __name__ == "__main__":
    main()
