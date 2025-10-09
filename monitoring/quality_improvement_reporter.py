#!/usr/bin/env python3
"""
Task 5.6.2.4: Quality Improvement Reporting System

Enterprise-grade reporting system for quality improvement tracking with
comprehensive analysis, visualizations, and executive summaries.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from jinja2 import Template
import warnings

# Import our improvement tracker
from quality_improvement_tracker import (
    QualityImprovementTracker,
    QualityIntervention,
    ImprovementAnalysis,
    ImprovementReport
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityImprovementReporter:
    """
    Enterprise-grade quality improvement reporting system.
    
    Generates comprehensive improvement reports with visualizations,
    impact analysis, and executive summaries.
    """
    
    def __init__(self, output_dir: str = "/home/vivi/pixelated/ai/monitoring/reports"):
        """Initialize the improvement reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = QualityImprovementTracker()
        
        # Report templates
        self.report_templates = {
            'executive': self._get_executive_template(),
            'detailed': self._get_detailed_template(),
            'technical': self._get_technical_template()
        }
        
        logger.info("📊 Quality Improvement Reporter initialized")
    
    def generate_comprehensive_report(self, 
                                    report_period_days: int = 90,
                                    include_visualizations: bool = True) -> ImprovementReport:
        """Generate comprehensive quality improvement report."""
        logger.info(f"📈 Generating comprehensive improvement report ({report_period_days} days)")
        
        try:
            # Get interventions from the specified period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=report_period_days)
            
            active_interventions = self._get_interventions_by_status('active')
            completed_interventions = self._get_interventions_by_period(start_date, end_date, 'completed')
            
            # Analyze each completed intervention
            improvement_analyses = []
            for intervention in completed_interventions:
                analysis = self.tracker.analyze_intervention_impact(intervention.intervention_id)
                if analysis:
                    improvement_analyses.append(analysis)
            
            # Calculate overall impact
            overall_impact = self._calculate_overall_impact(improvement_analyses)
            
            # Calculate success metrics
            success_metrics = self._calculate_success_metrics(improvement_analyses)
            
            # Generate summaries
            executive_summary = self._generate_executive_summary(
                active_interventions, completed_interventions, improvement_analyses, overall_impact
            )
            detailed_insights = self._generate_detailed_insights(
                improvement_analyses, overall_impact, success_metrics
            )
            action_items = self._generate_action_items(
                active_interventions, improvement_analyses, success_metrics
            )
            
            # Create comprehensive report
            report = ImprovementReport(
                generated_at=datetime.now().isoformat(),
                report_period=f"{report_period_days}_days",
                active_interventions=active_interventions,
                completed_interventions=completed_interventions,
                improvement_analyses=improvement_analyses,
                overall_impact=overall_impact,
                success_metrics=success_metrics,
                executive_summary=executive_summary,
                detailed_insights=detailed_insights,
                action_items=action_items
            )
            
            logger.info("✅ Comprehensive improvement report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating improvement report: {e}")
            return self._create_empty_report()
    
    def _get_interventions_by_status(self, status: str) -> List[QualityIntervention]:
        """Get interventions by status."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.tracker.interventions_db))
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM interventions WHERE status = ?", (status,))
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            interventions = []
            for row in rows:
                intervention_data = dict(zip(columns, row))
                intervention = QualityIntervention(**intervention_data)
                interventions.append(intervention)
            
            conn.close()
            return interventions
            
        except Exception as e:
            logger.error(f"❌ Error getting interventions by status: {e}")
            return []
    
    def _get_interventions_by_period(self, 
                                   start_date: datetime, 
                                   end_date: datetime, 
                                   status: str) -> List[QualityIntervention]:
        """Get interventions completed within a specific period."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.tracker.interventions_db))
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT * FROM interventions 
            WHERE status = ? 
            AND end_date >= ? 
            AND end_date <= ?
            """, (status, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            interventions = []
            for row in rows:
                intervention_data = dict(zip(columns, row))
                intervention = QualityIntervention(**intervention_data)
                interventions.append(intervention)
            
            conn.close()
            return interventions
            
        except Exception as e:
            logger.error(f"❌ Error getting interventions by period: {e}")
            return []
    
    def _calculate_overall_impact(self, analyses: List[ImprovementAnalysis]) -> Dict[str, Any]:
        """Calculate overall impact across all interventions."""
        if not analyses:
            return {
                'total_interventions': 0,
                'average_improvement': 0.0,
                'total_improvement': 0.0,
                'success_rate': 0.0,
                'components_improved': []
            }
        
        # Calculate aggregate metrics
        total_interventions = len(analyses)
        improvements = [analysis.improvement_metrics['absolute_improvement'] for analysis in analyses]
        average_improvement = np.mean(improvements)
        total_improvement = np.sum(improvements)
        
        # Success rate (interventions that met their targets)
        successful_interventions = sum(1 for analysis in analyses 
                                     if analysis.improvement_metrics['target_achievement'] >= 1.0)
        success_rate = successful_interventions / total_interventions if total_interventions > 0 else 0
        
        # Components that showed improvement
        component_improvements = {}
        for analysis in analyses:
            component = analysis.intervention_name.split()[-1].lower()  # Simplified component extraction
            if component not in component_improvements:
                component_improvements[component] = []
            component_improvements[component].append(analysis.improvement_metrics['absolute_improvement'])
        
        components_improved = [
            {
                'component': component,
                'average_improvement': np.mean(improvements),
                'intervention_count': len(improvements)
            }
            for component, improvements in component_improvements.items()
            if np.mean(improvements) > 0
        ]
        
        return {
            'total_interventions': total_interventions,
            'average_improvement': float(average_improvement),
            'total_improvement': float(total_improvement),
            'success_rate': float(success_rate),
            'successful_interventions': successful_interventions,
            'components_improved': components_improved
        }
    
    def _calculate_success_metrics(self, analyses: List[ImprovementAnalysis]) -> Dict[str, float]:
        """Calculate success metrics for interventions."""
        if not analyses:
            return {
                'target_achievement_rate': 0.0,
                'statistical_significance_rate': 0.0,
                'practical_significance_rate': 0.0,
                'average_effect_size': 0.0
            }
        
        # Target achievement rate
        target_achievements = [analysis.improvement_metrics['target_achievement'] for analysis in analyses]
        target_achievement_rate = sum(1 for ta in target_achievements if ta >= 1.0) / len(target_achievements)
        
        # Statistical significance rate
        significant_analyses = sum(1 for analysis in analyses 
                                 if any(test.get('significant', False) for test in analysis.statistical_tests))
        statistical_significance_rate = significant_analyses / len(analyses)
        
        # Practical significance rate
        practical_significant = sum(1 for analysis in analyses 
                                  if analysis.impact_assessment.get('practical_significance', False))
        practical_significance_rate = practical_significant / len(analyses)
        
        # Average effect size
        effect_sizes = [analysis.impact_assessment.get('effect_size', 0) for analysis in analyses]
        average_effect_size = np.mean([es for es in effect_sizes if es is not None])
        
        return {
            'target_achievement_rate': float(target_achievement_rate),
            'statistical_significance_rate': float(statistical_significance_rate),
            'practical_significance_rate': float(practical_significance_rate),
            'average_effect_size': float(average_effect_size) if not np.isnan(average_effect_size) else 0.0
        }
    
    def _generate_executive_summary(self, 
                                  active_interventions: List[QualityIntervention],
                                  completed_interventions: List[QualityIntervention],
                                  analyses: List[ImprovementAnalysis],
                                  overall_impact: Dict[str, Any]) -> List[str]:
        """Generate executive summary for the improvement report."""
        summary = []
        
        # Overall status
        total_active = len(active_interventions)
        total_completed = len(completed_interventions)
        
        summary.append(f"📊 Quality Improvement Status: {total_active} active interventions, {total_completed} completed")
        
        # Overall impact
        if overall_impact['total_interventions'] > 0:
            avg_improvement = overall_impact['average_improvement']
            success_rate = overall_impact['success_rate'] * 100
            
            summary.append(f"📈 Average improvement: {avg_improvement:.3f} points across {overall_impact['total_interventions']} interventions")
            summary.append(f"🎯 Success rate: {success_rate:.1f}% of interventions met their targets")
            
            if overall_impact['successful_interventions'] > 0:
                summary.append(f"✅ {overall_impact['successful_interventions']} interventions achieved their improvement targets")
        
        # Component improvements
        if overall_impact['components_improved']:
            best_component = max(overall_impact['components_improved'], 
                               key=lambda x: x['average_improvement'])
            summary.append(f"🏆 Best performing component: {best_component['component']} (+{best_component['average_improvement']:.3f})")
        
        # Active interventions status
        if active_interventions:
            intervention_types = {}
            for intervention in active_interventions:
                intervention_types[intervention.intervention_type] = intervention_types.get(intervention.intervention_type, 0) + 1
            
            most_common_type = max(intervention_types.items(), key=lambda x: x[1])
            summary.append(f"🔄 Most common active intervention type: {most_common_type[0]} ({most_common_type[1]} interventions)")
        
        return summary
    
    def _generate_detailed_insights(self, 
                                  analyses: List[ImprovementAnalysis],
                                  overall_impact: Dict[str, Any],
                                  success_metrics: Dict[str, float]) -> List[str]:
        """Generate detailed insights for the improvement report."""
        insights = []
        
        # Statistical insights
        if success_metrics['statistical_significance_rate'] > 0:
            sig_rate = success_metrics['statistical_significance_rate'] * 100
            insights.append(f"📊 {sig_rate:.1f}% of interventions showed statistically significant improvements")
        
        if success_metrics['practical_significance_rate'] > 0:
            prac_rate = success_metrics['practical_significance_rate'] * 100
            insights.append(f"💡 {prac_rate:.1f}% of interventions achieved practically significant improvements")
        
        # Effect size insights
        avg_effect_size = success_metrics['average_effect_size']
        if avg_effect_size > 0.8:
            insights.append(f"💪 Large average effect size ({avg_effect_size:.3f}) indicates strong intervention impact")
        elif avg_effect_size > 0.5:
            insights.append(f"📈 Medium average effect size ({avg_effect_size:.3f}) indicates moderate intervention impact")
        elif avg_effect_size > 0.2:
            insights.append(f"📊 Small average effect size ({avg_effect_size:.3f}) indicates limited intervention impact")
        
        # Individual intervention insights
        if analyses:
            # Best performing intervention
            best_intervention = max(analyses, key=lambda x: x.improvement_metrics['absolute_improvement'])
            insights.append(f"🏆 Best intervention: {best_intervention.intervention_name} (+{best_intervention.improvement_metrics['absolute_improvement']:.3f})")
            
            # Interventions with declining trends
            declining_interventions = [a for a in analyses 
                                     if a.trend_analysis.get('trend_direction') == 'declining']
            if declining_interventions:
                insights.append(f"⚠️ {len(declining_interventions)} interventions show declining trends requiring attention")
        
        # Target achievement insights
        target_rate = success_metrics['target_achievement_rate'] * 100
        if target_rate < 50:
            insights.append(f"🎯 Low target achievement rate ({target_rate:.1f}%) suggests need for more realistic targets or stronger interventions")
        elif target_rate > 80:
            insights.append(f"🎯 High target achievement rate ({target_rate:.1f}%) indicates effective intervention planning")
        
        return insights
    
    def _generate_action_items(self, 
                             active_interventions: List[QualityIntervention],
                             analyses: List[ImprovementAnalysis],
                             success_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable items based on improvement analysis."""
        actions = []
        
        # Active interventions actions
        if active_interventions:
            overdue_interventions = []
            for intervention in active_interventions:
                start_date = datetime.fromisoformat(intervention.start_date)
                if (datetime.now() - start_date).days > 60:  # 60 days threshold
                    overdue_interventions.append(intervention)
            
            if overdue_interventions:
                actions.append(f"⏰ Review {len(overdue_interventions)} long-running interventions for completion or adjustment")
        
        # Success rate actions
        if success_metrics['target_achievement_rate'] < 0.5:
            actions.append("🎯 Review intervention targets - less than 50% are being achieved")
            actions.append("📊 Analyze successful interventions to identify best practices")
        
        # Statistical significance actions
        if success_metrics['statistical_significance_rate'] < 0.3:
            actions.append("📈 Increase sample sizes or measurement frequency for better statistical power")
        
        # Component-specific actions
        if analyses:
            # Find components with consistently poor performance
            component_performance = {}
            for analysis in analyses:
                component = analysis.intervention_name.split()[-1].lower()
                if component not in component_performance:
                    component_performance[component] = []
                component_performance[component].append(analysis.improvement_metrics['absolute_improvement'])
            
            poor_components = [
                component for component, improvements in component_performance.items()
                if np.mean(improvements) < 0.02  # Less than 2% improvement
            ]
            
            if poor_components:
                actions.append(f"🔧 Focus additional resources on underperforming components: {', '.join(poor_components)}")
        
        # Intervention type actions
        if active_interventions:
            training_interventions = [i for i in active_interventions if i.intervention_type == 'training']
            if len(training_interventions) > len(active_interventions) * 0.7:
                actions.append("🎓 Consider diversifying intervention types beyond training")
        
        return actions
    
    def create_improvement_visualizations(self, report: ImprovementReport) -> Dict[str, go.Figure]:
        """Create comprehensive improvement visualizations."""
        visualizations = {}
        
        # Intervention timeline
        if report.completed_interventions or report.active_interventions:
            fig = go.Figure()
            
            all_interventions = report.completed_interventions + report.active_interventions
            
            for i, intervention in enumerate(all_interventions):
                start_date = datetime.fromisoformat(intervention.start_date)
                end_date = datetime.fromisoformat(intervention.end_date) if intervention.end_date else datetime.now()
                
                color = 'green' if intervention.status == 'completed' else 'blue'
                
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date],
                    y=[i, i],
                    mode='lines+markers',
                    name=intervention.name,
                    line=dict(color=color, width=6),
                    hovertemplate=f"<b>{intervention.name}</b><br>" +
                                f"Type: {intervention.intervention_type}<br>" +
                                f"Status: {intervention.status}<br>" +
                                f"Target: {intervention.target_component}<br>" +
                                "<extra></extra>"
                ))
            
            fig.update_layout(
                title='Intervention Timeline',
                xaxis_title='Date',
                yaxis_title='Interventions',
                yaxis=dict(tickmode='array', tickvals=list(range(len(all_interventions))),
                          ticktext=[i.name[:30] + '...' if len(i.name) > 30 else i.name for i in all_interventions]),
                height=max(400, len(all_interventions) * 40)
            )
            
            visualizations['intervention_timeline'] = fig
        
        # Improvement impact chart
        if report.improvement_analyses:
            fig = go.Figure()
            
            intervention_names = [analysis.intervention_name for analysis in report.improvement_analyses]
            improvements = [analysis.improvement_metrics['absolute_improvement'] for analysis in report.improvement_analyses]
            targets = [analysis.improvement_metrics['target_achievement'] for analysis in report.improvement_analyses]
            
            fig.add_trace(go.Bar(
                x=intervention_names,
                y=improvements,
                name='Actual Improvement',
                marker_color=['green' if imp > 0 else 'red' for imp in improvements]
            ))
            
            fig.add_trace(go.Scatter(
                x=intervention_names,
                y=[t * max(improvements) if improvements else 0 for t in targets],
                mode='markers',
                name='Target Achievement',
                marker=dict(size=10, symbol='diamond', color='orange')
            ))
            
            fig.update_layout(
                title='Intervention Impact Analysis',
                xaxis_title='Interventions',
                yaxis_title='Quality Improvement',
                xaxis_tickangle=-45
            )
            
            visualizations['improvement_impact'] = fig
        
        # Success metrics dashboard
        if report.success_metrics:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Target Achievement Rate', 'Statistical Significance Rate',
                              'Practical Significance Rate', 'Average Effect Size'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Target achievement rate
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=report.success_metrics['target_achievement_rate'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Target Achievement %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=1, col=1)
            
            # Statistical significance rate
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=report.success_metrics['statistical_significance_rate'] * 100,
                title={'text': "Statistical Significance %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"}}
            ), row=1, col=2)
            
            # Practical significance rate
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=report.success_metrics['practical_significance_rate'] * 100,
                title={'text': "Practical Significance %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkorange"}}
            ), row=2, col=1)
            
            # Average effect size
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=report.success_metrics['average_effect_size'],
                title={'text': "Average Effect Size"},
                gauge={'axis': {'range': [0, 2]},
                       'bar': {'color': "darkred"},
                       'steps': [{'range': [0, 0.2], 'color': "lightgray"},
                                {'range': [0.2, 0.5], 'color': "gray"},
                                {'range': [0.5, 0.8], 'color': "lightblue"}]}
            ), row=2, col=2)
            
            fig.update_layout(height=600, title_text="Success Metrics Dashboard")
            visualizations['success_metrics'] = fig
        
        return visualizations
    
    def save_report(self, report: ImprovementReport, format: str = 'json') -> str:
        """Save improvement report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"quality_improvement_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert dataclasses to dict for JSON serialization
            report_dict = asdict(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
        elif format == 'html':
            filename = f"quality_improvement_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # Generate HTML report using template
            html_content = self._generate_html_report(report)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"📄 Report saved: {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, report: ImprovementReport) -> str:
        """Generate HTML report from improvement analysis."""
        template = Template(self.report_templates['detailed'])
        
        return template.render(
            report=report,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _create_empty_report(self) -> ImprovementReport:
        """Create empty report when no data is available."""
        return ImprovementReport(
            generated_at=datetime.now().isoformat(),
            report_period="no_data",
            active_interventions=[],
            completed_interventions=[],
            improvement_analyses=[],
            overall_impact={},
            success_metrics={},
            executive_summary=["No improvement data available for analysis"],
            detailed_insights=["Please create and track quality improvement interventions"],
            action_items=["Set up quality improvement interventions to begin tracking"]
        )
    
    def _get_executive_template(self) -> str:
        """Get executive summary template."""
        return """
        # Quality Improvement Report - Executive Summary
        
        **Generated**: {{ generated_at }}
        **Period**: {{ report.report_period }}
        
        ## Key Findings
        {% for item in report.executive_summary %}
        - {{ item }}
        {% endfor %}
        
        ## Action Items
        {% for item in report.action_items %}
        - {{ item }}
        {% endfor %}
        """
    
    def _get_detailed_template(self) -> str:
        """Get detailed report template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Improvement Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Improvement Report</h1>
                <p><strong>Generated:</strong> {{ generated_at }}</p>
                <p><strong>Report Period:</strong> {{ report.report_period }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <ul>
                {% for item in report.executive_summary %}
                    <li>{{ item }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>Overall Impact</h2>
                <div class="metric">
                    <strong>Total Interventions:</strong> {{ report.overall_impact.get('total_interventions', 0) }}
                </div>
                <div class="metric">
                    <strong>Average Improvement:</strong> {{ "%.3f"|format(report.overall_impact.get('average_improvement', 0)) }}
                </div>
                <div class="metric">
                    <strong>Success Rate:</strong> {{ "%.1f"|format(report.overall_impact.get('success_rate', 0) * 100) }}%
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Insights</h2>
                <ul>
                {% for insight in report.detailed_insights %}
                    <li>{{ insight }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>Action Items</h2>
                <ul>
                {% for action in report.action_items %}
                    <li>{{ action }}</li>
                {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
    
    def _get_technical_template(self) -> str:
        """Get technical report template."""
        return """
        # Technical Quality Improvement Analysis
        
        ## Success Metrics
        - Target Achievement Rate: {{ report.success_metrics.target_achievement_rate }}
        - Statistical Significance Rate: {{ report.success_metrics.statistical_significance_rate }}
        - Practical Significance Rate: {{ report.success_metrics.practical_significance_rate }}
        - Average Effect Size: {{ report.success_metrics.average_effect_size }}
        
        ## Intervention Analysis
        {% for analysis in report.improvement_analyses %}
        ### {{ analysis.intervention_name }}
        - Improvement: {{ analysis.improvement_metrics.absolute_improvement }}
        - Target Achievement: {{ analysis.improvement_metrics.target_achievement }}
        - Statistical Tests: {{ analysis.statistical_tests|length }}
        {% endfor %}
        """

def main():
    """Main function for testing the improvement reporter."""
    reporter = QualityImprovementReporter()
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(report_period_days=30)
    
    # Save report
    json_path = reporter.save_report(report, format='json')
    html_path = reporter.save_report(report, format='html')
    
    print(f"📊 Improvement Report Generated:")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    
    # Display summary
    print(f"\n📈 Executive Summary:")
    for item in report.executive_summary:
        print(f"  {item}")

if __name__ == "__main__":
    main()
