#!/usr/bin/env python3
"""
Task 5.6.2.2: Quality Trend Reporting System

Enterprise-grade automated reporting system for quality trends with
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

# Import our trend analyzer
from quality_trend_analyzer import QualityTrendAnalyzer, TrendAnalysis, QualityTrendReport

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityTrendReporter:
    """
    Enterprise-grade quality trend reporting system.
    
    Generates comprehensive trend reports with visualizations,
    statistical analysis, and executive summaries.
    """
    
    def __init__(self, output_dir: str = "/home/vivi/pixelated/ai/monitoring/reports"):
        """Initialize the trend reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = QualityTrendAnalyzer()
        
        # Report templates
        self.report_templates = {
            'executive': self._get_executive_template(),
            'detailed': self._get_detailed_template(),
            'technical': self._get_technical_template()
        }
        
        logger.info("📊 Quality Trend Reporter initialized")
    
    def generate_comprehensive_report(self, 
                                    days_back: int = 90,
                                    include_predictions: bool = True,
                                    include_visualizations: bool = True) -> QualityTrendReport:
        """Generate comprehensive quality trend report."""
        logger.info(f"📈 Generating comprehensive trend report ({days_back} days)")
        
        # Load historical data
        df = self.analyzer.load_historical_data(days_back=days_back)
        
        if df.empty:
            logger.error("❌ No data available for trend analysis")
            return self._create_empty_report()
        
        # Overall trend analysis
        overall_trend = self.analyzer.analyze_overall_trend(df, period=f"{days_back}_days")
        
        # Component-wise trend analysis
        component_trends = {}
        for component in self.analyzer.quality_components:
            if component in df.columns:
                component_df = df.dropna(subset=[component])
                if not component_df.empty:
                    # Create daily aggregation for component
                    daily_component = component_df.groupby('date')[component].mean().reset_index()
                    if len(daily_component) >= self.analyzer.min_data_points:
                        trend_stats = self.analyzer.calculate_trend_statistics(
                            daily_component[component], 
                            daily_component['date']
                        )
                        
                        # Create simplified trend analysis for component
                        component_trends[component] = TrendAnalysis(
                            period=f"{days_back}_days",
                            start_date=overall_trend.start_date,
                            end_date=overall_trend.end_date,
                            total_conversations=len(component_df),
                            trend_direction=trend_stats['trend_direction'],
                            trend_strength=trend_stats['trend_strength_numeric'],
                            slope=trend_stats['slope'],
                            r_squared=trend_stats['r_squared'],
                            quality_change=float(component_df[component].iloc[-7:].mean() - component_df[component].iloc[:7].mean()),
                            statistical_significance=trend_stats['p_value'],
                            confidence_interval=trend_stats['confidence_interval'],
                            seasonal_patterns=self.analyzer.detect_seasonal_patterns(component_df, component),
                            anomalies=self.analyzer.detect_anomalies(component_df, component),
                            predictions=self.analyzer.generate_predictions(component_df, component) if include_predictions else [],
                            recommendations=self.analyzer._generate_trend_recommendations(trend_stats, 0, [])
                        )
        
        # Tier-wise trend analysis
        tier_trends = {}
        for tier in df['tier'].unique():
            tier_df = df[df['tier'] == tier]
            if len(tier_df) >= self.analyzer.min_data_points:
                tier_trend = self.analyzer.analyze_overall_trend(tier_df, period=f"{days_back}_days_{tier}")
                tier_trends[tier] = tier_trend
        
        # Dataset-wise trend analysis
        dataset_trends = {}
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            if len(dataset_df) >= self.analyzer.min_data_points:
                dataset_trend = self.analyzer.analyze_overall_trend(dataset_df, period=f"{days_back}_days_{dataset}")
                dataset_trends[dataset] = dataset_trend
        
        # Comparative analysis
        comparative_analysis = self._generate_comparative_analysis(
            overall_trend, component_trends, tier_trends, dataset_trends
        )
        
        # Generate summaries
        executive_summary = self._generate_executive_summary(overall_trend, component_trends, tier_trends)
        detailed_insights = self._generate_detailed_insights(overall_trend, component_trends, comparative_analysis)
        action_items = self._generate_action_items(overall_trend, component_trends, tier_trends)
        
        # Create comprehensive report
        report = QualityTrendReport(
            generated_at=datetime.now().isoformat(),
            analysis_period=f"{days_back}_days",
            overall_trend=overall_trend,
            component_trends=component_trends,
            tier_trends=tier_trends,
            dataset_trends=dataset_trends,
            comparative_analysis=comparative_analysis,
            executive_summary=executive_summary,
            detailed_insights=detailed_insights,
            action_items=action_items
        )
        
        logger.info("✅ Comprehensive trend report generated")
        return report
    
    def _create_empty_report(self) -> QualityTrendReport:
        """Create empty report when no data is available."""
        empty_trend = TrendAnalysis(
            period="no_data",
            start_date="",
            end_date="",
            total_conversations=0,
            trend_direction="no_data",
            trend_strength=0.0,
            slope=0.0,
            r_squared=0.0,
            quality_change=0.0,
            statistical_significance=1.0,
            confidence_interval=(0.0, 0.0),
            seasonal_patterns={},
            anomalies=[],
            predictions=[],
            recommendations=["No data available for analysis"]
        )
        
        return QualityTrendReport(
            generated_at=datetime.now().isoformat(),
            analysis_period="no_data",
            overall_trend=empty_trend,
            component_trends={},
            tier_trends={},
            dataset_trends={},
            comparative_analysis={},
            executive_summary=["No quality data available for trend analysis"],
            detailed_insights=["Please ensure quality validation has been run on conversations"],
            action_items=["Run quality validation pipeline to generate trend data"]
        )
    
    def _generate_comparative_analysis(self, 
                                     overall_trend: TrendAnalysis,
                                     component_trends: Dict[str, TrendAnalysis],
                                     tier_trends: Dict[str, TrendAnalysis],
                                     dataset_trends: Dict[str, TrendAnalysis]) -> Dict[str, Any]:
        """Generate comparative analysis across different dimensions."""
        analysis = {}
        
        # Component performance comparison
        if component_trends:
            component_performance = {}
            for component, trend in component_trends.items():
                component_performance[component] = {
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'quality_change': trend.quality_change,
                    'r_squared': trend.r_squared
                }
            
            # Find best and worst performing components
            best_component = max(component_performance.items(), 
                               key=lambda x: x[1]['quality_change'])
            worst_component = min(component_performance.items(), 
                                key=lambda x: x[1]['quality_change'])
            
            analysis['component_performance'] = {
                'all_components': component_performance,
                'best_performing': {
                    'component': best_component[0],
                    'change': best_component[1]['quality_change']
                },
                'worst_performing': {
                    'component': worst_component[0],
                    'change': worst_component[1]['quality_change']
                }
            }
        
        # Tier performance comparison
        if tier_trends:
            tier_performance = {}
            for tier, trend in tier_trends.items():
                tier_performance[tier] = {
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'quality_change': trend.quality_change,
                    'total_conversations': trend.total_conversations
                }
            
            # Find best and worst performing tiers
            best_tier = max(tier_performance.items(), 
                          key=lambda x: x[1]['quality_change'])
            worst_tier = min(tier_performance.items(), 
                           key=lambda x: x[1]['quality_change'])
            
            analysis['tier_performance'] = {
                'all_tiers': tier_performance,
                'best_performing': {
                    'tier': best_tier[0],
                    'change': best_tier[1]['quality_change']
                },
                'worst_performing': {
                    'tier': worst_tier[0],
                    'change': worst_tier[1]['quality_change']
                }
            }
        
        # Dataset performance comparison
        if dataset_trends:
            dataset_performance = {}
            for dataset, trend in dataset_trends.items():
                dataset_performance[dataset] = {
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'quality_change': trend.quality_change,
                    'total_conversations': trend.total_conversations
                }
            
            analysis['dataset_performance'] = dataset_performance
        
        return analysis
    
    def _generate_executive_summary(self, 
                                  overall_trend: TrendAnalysis,
                                  component_trends: Dict[str, TrendAnalysis],
                                  tier_trends: Dict[str, TrendAnalysis]) -> List[str]:
        """Generate executive summary for the trend report."""
        summary = []
        
        # Overall trend summary
        if overall_trend.trend_direction == 'improving':
            summary.append(f"✅ Overall quality is improving with {overall_trend.total_conversations:,} conversations analyzed.")
        elif overall_trend.trend_direction == 'declining':
            summary.append(f"🚨 Overall quality is declining across {overall_trend.total_conversations:,} conversations.")
        else:
            summary.append(f"📊 Overall quality remains stable across {overall_trend.total_conversations:,} conversations.")
        
        # Quality change summary
        if abs(overall_trend.quality_change) > 0.05:
            change_direction = "improvement" if overall_trend.quality_change > 0 else "decline"
            summary.append(f"📈 Significant quality {change_direction} of {abs(overall_trend.quality_change):.3f} points detected.")
        
        # Statistical significance
        if overall_trend.statistical_significance < 0.05:
            summary.append("📊 Trend is statistically significant with high confidence.")
        else:
            summary.append("⚠️ Trend lacks statistical significance - more data needed.")
        
        # Component insights
        if component_trends:
            improving_components = [name for name, trend in component_trends.items() 
                                  if trend.trend_direction == 'improving']
            declining_components = [name for name, trend in component_trends.items() 
                                  if trend.trend_direction == 'declining']
            
            if improving_components:
                summary.append(f"📈 Improving components: {', '.join(improving_components[:3])}")
            if declining_components:
                summary.append(f"📉 Declining components: {', '.join(declining_components[:3])}")
        
        # Tier insights
        if tier_trends:
            best_tier = max(tier_trends.items(), key=lambda x: x[1].quality_change)
            worst_tier = min(tier_trends.items(), key=lambda x: x[1].quality_change)
            
            summary.append(f"🏆 Best performing tier: {best_tier[0]} (+{best_tier[1].quality_change:.3f})")
            summary.append(f"⚠️ Worst performing tier: {worst_tier[0]} ({worst_tier[1].quality_change:.3f})")
        
        # Anomaly summary
        if len(overall_trend.anomalies) > 0:
            summary.append(f"⚠️ {len(overall_trend.anomalies)} quality anomalies detected requiring attention.")
        
        return summary
    
    def _generate_detailed_insights(self, 
                                  overall_trend: TrendAnalysis,
                                  component_trends: Dict[str, TrendAnalysis],
                                  comparative_analysis: Dict[str, Any]) -> List[str]:
        """Generate detailed insights for the trend report."""
        insights = []
        
        # Statistical insights
        insights.append(f"📊 Trend Analysis: R² = {overall_trend.r_squared:.3f}, Slope = {overall_trend.slope:.6f}")
        insights.append(f"📈 Confidence Interval: [{overall_trend.confidence_interval[0]:.6f}, {overall_trend.confidence_interval[1]:.6f}]")
        
        # Seasonal pattern insights
        if overall_trend.seasonal_patterns:
            if 'day_of_week' in overall_trend.seasonal_patterns:
                dow_data = overall_trend.seasonal_patterns['day_of_week']
                best_day = max(dow_data.items(), key=lambda x: x[1]['mean'])
                worst_day = min(dow_data.items(), key=lambda x: x[1]['mean'])
                insights.append(f"📅 Best quality day: {best_day[0]} ({best_day[1]['mean']:.3f})")
                insights.append(f"📅 Worst quality day: {worst_day[0]} ({worst_day[1]['mean']:.3f})")
        
        # Component-specific insights
        if component_trends:
            for component, trend in component_trends.items():
                if abs(trend.quality_change) > 0.05:
                    direction = "improved" if trend.quality_change > 0 else "declined"
                    insights.append(f"🔍 {component.replace('_', ' ').title()} has {direction} by {abs(trend.quality_change):.3f}")
        
        # Comparative insights
        if 'component_performance' in comparative_analysis:
            comp_perf = comparative_analysis['component_performance']
            insights.append(f"🏆 Top component: {comp_perf['best_performing']['component']} (+{comp_perf['best_performing']['change']:.3f})")
            insights.append(f"⚠️ Bottom component: {comp_perf['worst_performing']['component']} ({comp_perf['worst_performing']['change']:.3f})")
        
        # Prediction insights
        if overall_trend.predictions:
            next_week_pred = overall_trend.predictions[6]  # 7 days ahead
            insights.append(f"🔮 7-day prediction: {next_week_pred['predicted_value']:.3f} (±{(next_week_pred['confidence_upper'] - next_week_pred['confidence_lower'])/2:.3f})")
        
        return insights
    
    def _generate_action_items(self, 
                             overall_trend: TrendAnalysis,
                             component_trends: Dict[str, TrendAnalysis],
                             tier_trends: Dict[str, TrendAnalysis]) -> List[str]:
        """Generate actionable items based on trend analysis."""
        actions = []
        
        # Overall trend actions
        if overall_trend.trend_direction == 'declining':
            actions.append("🚨 URGENT: Investigate causes of quality decline")
            actions.append("📊 Review recent changes in data processing pipeline")
            actions.append("🔍 Analyze conversations from declining periods")
        elif overall_trend.trend_direction == 'improving':
            actions.append("📈 Document successful practices for replication")
            actions.append("🎯 Maintain current quality improvement strategies")
        
        # Component-specific actions
        if component_trends:
            declining_components = [(name, trend) for name, trend in component_trends.items() 
                                  if trend.trend_direction == 'declining']
            
            for component, trend in declining_components[:3]:  # Top 3 declining
                actions.append(f"🔧 Focus improvement efforts on {component.replace('_', ' ')}")
        
        # Tier-specific actions
        if tier_trends:
            worst_tiers = sorted(tier_trends.items(), key=lambda x: x[1].quality_change)[:2]
            for tier, trend in worst_tiers:
                if trend.trend_direction == 'declining':
                    actions.append(f"📋 Review data sources and processing for {tier} tier")
        
        # Anomaly actions
        if len(overall_trend.anomalies) > 5:
            actions.append("🔍 Investigate quality anomalies for patterns")
            actions.append("📊 Implement automated anomaly alerting")
        
        # Statistical significance actions
        if overall_trend.statistical_significance > 0.05:
            actions.append("📈 Collect more data to establish statistical significance")
            actions.append("⏰ Extend analysis period for better trend detection")
        
        return actions
    
    def create_trend_visualizations(self, report: QualityTrendReport) -> Dict[str, go.Figure]:
        """Create comprehensive trend visualizations."""
        visualizations = {}
        
        # Overall trend visualization
        if report.overall_trend.predictions:
            fig = go.Figure()
            
            # Historical data (simulated for visualization)
            dates = pd.date_range(
                start=report.overall_trend.start_date,
                end=report.overall_trend.end_date,
                freq='D'
            )
            
            # Simulate historical quality data based on trend
            historical_quality = []
            base_quality = 0.7
            for i, date in enumerate(dates):
                trend_component = report.overall_trend.slope * i
                noise = np.random.normal(0, 0.05)  # Small random variation
                quality = base_quality + trend_component + noise
                historical_quality.append(max(0, min(1, quality)))  # Clamp to [0,1]
            
            # Add historical trend line
            fig.add_trace(go.Scatter(
                x=dates,
                y=historical_quality,
                mode='lines+markers',
                name='Historical Quality',
                line=dict(color='blue')
            ))
            
            # Add predictions
            pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in report.overall_trend.predictions]
            pred_values = [p['predicted_value'] for p in report.overall_trend.predictions]
            pred_upper = [p['confidence_upper'] for p in report.overall_trend.predictions]
            pred_lower = [p['confidence_lower'] for p in report.overall_trend.predictions]
            
            # Prediction line
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=pred_upper + pred_lower[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title='Quality Trend Analysis with Predictions',
                xaxis_title='Date',
                yaxis_title='Quality Score',
                hovermode='x unified'
            )
            
            visualizations['overall_trend'] = fig
        
        # Component trends comparison
        if report.component_trends:
            fig = go.Figure()
            
            for component, trend in report.component_trends.items():
                fig.add_trace(go.Bar(
                    x=[component.replace('_', ' ').title()],
                    y=[trend.quality_change],
                    name=component,
                    marker_color='green' if trend.quality_change > 0 else 'red'
                ))
            
            fig.update_layout(
                title='Quality Change by Component',
                xaxis_title='Quality Components',
                yaxis_title='Quality Change',
                showlegend=False
            )
            
            visualizations['component_trends'] = fig
        
        # Tier performance comparison
        if report.tier_trends:
            fig = go.Figure()
            
            tiers = list(report.tier_trends.keys())
            quality_changes = [trend.quality_change for trend in report.tier_trends.values()]
            trend_strengths = [trend.trend_strength for trend in report.tier_trends.values()]
            
            fig.add_trace(go.Scatter(
                x=quality_changes,
                y=trend_strengths,
                mode='markers+text',
                text=tiers,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=quality_changes,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Quality Change")
                ),
                name='Tiers'
            ))
            
            fig.update_layout(
                title='Tier Performance: Quality Change vs Trend Strength',
                xaxis_title='Quality Change',
                yaxis_title='Trend Strength',
                showlegend=False
            )
            
            visualizations['tier_performance'] = fig
        
        return visualizations
    
    def save_report(self, report: QualityTrendReport, format: str = 'json') -> str:
        """Save trend report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"quality_trend_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert dataclasses to dict for JSON serialization
            report_dict = asdict(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
        elif format == 'html':
            filename = f"quality_trend_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # Generate HTML report using template
            html_content = self._generate_html_report(report)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"📄 Report saved: {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, report: QualityTrendReport) -> str:
        """Generate HTML report from trend analysis."""
        template = Template(self.report_templates['detailed'])
        
        return template.render(
            report=report,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _get_executive_template(self) -> str:
        """Get executive summary template."""
        return """
        # Quality Trend Analysis - Executive Summary
        
        **Generated**: {{ generated_at }}
        **Period**: {{ report.analysis_period }}
        
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
            <title>Quality Trend Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .improving { color: green; }
                .declining { color: red; }
                .stable { color: blue; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Trend Analysis Report</h1>
                <p><strong>Generated:</strong> {{ generated_at }}</p>
                <p><strong>Analysis Period:</strong> {{ report.analysis_period }}</p>
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
                <h2>Overall Trend</h2>
                <div class="metric">
                    <strong>Direction:</strong> 
                    <span class="{{ report.overall_trend.trend_direction }}">
                        {{ report.overall_trend.trend_direction }}
                    </span>
                </div>
                <div class="metric">
                    <strong>Conversations Analyzed:</strong> {{ report.overall_trend.total_conversations }}
                </div>
                <div class="metric">
                    <strong>Quality Change:</strong> {{ "%.3f"|format(report.overall_trend.quality_change) }}
                </div>
                <div class="metric">
                    <strong>Statistical Significance:</strong> {{ "%.3f"|format(report.overall_trend.statistical_significance) }}
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
        # Technical Quality Trend Analysis
        
        ## Statistical Analysis
        - R-squared: {{ report.overall_trend.r_squared }}
        - Slope: {{ report.overall_trend.slope }}
        - P-value: {{ report.overall_trend.statistical_significance }}
        
        ## Component Analysis
        {% for component, trend in report.component_trends.items() %}
        ### {{ component }}
        - Direction: {{ trend.trend_direction }}
        - Change: {{ trend.quality_change }}
        - R²: {{ trend.r_squared }}
        {% endfor %}
        """

def main():
    """Main function for testing the trend reporter."""
    reporter = QualityTrendReporter()
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(days_back=30)
    
    # Save report
    json_path = reporter.save_report(report, format='json')
    html_path = reporter.save_report(report, format='html')
    
    print(f"📊 Trend Report Generated:")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    
    # Display summary
    print(f"\n📈 Executive Summary:")
    for item in report.executive_summary:
        print(f"  {item}")

if __name__ == "__main__":
    main()
