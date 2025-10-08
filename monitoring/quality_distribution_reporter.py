#!/usr/bin/env python3
"""
Task 5.6.2.3: Quality Distribution Reporting System

Enterprise-grade reporting system for quality distribution analysis with
comprehensive visualizations, statistical summaries, and executive reports.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from jinja2 import Template
import warnings

# Import our distribution components
from quality_distribution_analyzer import (
    QualityDistributionAnalyzer, 
    QualityDistributionAnalysis,
    QualityDistributionReport
)
from quality_distribution_comparator import QualityDistributionComparator

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityDistributionReporter:
    """
    Enterprise-grade quality distribution reporting system.
    
    Generates comprehensive distribution reports with visualizations,
    statistical analysis, and executive summaries.
    """
    
    def __init__(self, output_dir: str = "/home/vivi/pixelated/ai/monitoring/reports"):
        """Initialize the distribution reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = QualityDistributionAnalyzer()
        self.comparator = QualityDistributionComparator(self.analyzer)
        
        # Report templates
        self.report_templates = {
            'executive': self._get_executive_template(),
            'detailed': self._get_detailed_template(),
            'technical': self._get_technical_template()
        }
        
        logger.info("📊 Quality Distribution Reporter initialized")
    
    def generate_comprehensive_report(self, 
                                    days_back: int = 90,
                                    include_visualizations: bool = True,
                                    include_comparisons: bool = True) -> QualityDistributionReport:
        """Generate comprehensive quality distribution report."""
        logger.info(f"📈 Generating comprehensive distribution report ({days_back} days)")
        
        # Load quality data
        df = self.analyzer.load_quality_data(days_back=days_back)
        
        if df.empty:
            logger.error("❌ No data available for distribution analysis")
            return self._create_empty_report()
        
        # Overall distribution analysis
        overall_distribution = self.analyzer.analyze_quality_distribution(
            df['overall_quality'], 'overall_quality'
        )
        
        # Component-wise distribution analysis
        component_distributions = {}
        for component in self.analyzer.quality_components:
            if component in df.columns:
                component_series = df[component].dropna()
                if len(component_series) >= self.analyzer.min_sample_size:
                    component_analysis = self.analyzer.analyze_quality_distribution(
                        component_series, component
                    )
                    component_distributions[component] = component_analysis
        
        # Comparative analyses
        comparative_analyses = {}
        if include_comparisons:
            # Tier comparison
            tier_comparison = self.comparator.compare_across_tiers(df, 'overall_quality')
            if tier_comparison.groups:
                comparative_analyses['tier'] = tier_comparison
            
            # Dataset comparison
            dataset_comparison = self.comparator.compare_across_datasets(df, 'overall_quality')
            if dataset_comparison.groups:
                comparative_analyses['dataset'] = dataset_comparison
            
            # Component comparison
            component_comparison = self.comparator.compare_across_components(df)
            if component_comparison.groups:
                comparative_analyses['component'] = component_comparison
            
            # Time period comparison
            time_comparison = self.comparator.compare_across_time_periods(df, 'overall_quality', 'month')
            if time_comparison.groups:
                comparative_analyses['time_period'] = time_comparison
        
        # Correlation analysis
        correlation_analysis = self.comparator.calculate_correlation_analysis(df)
        
        # Generate summaries
        executive_summary = self._generate_executive_summary(
            overall_distribution, component_distributions, comparative_analyses
        )
        detailed_insights = self._generate_detailed_insights(
            overall_distribution, component_distributions, comparative_analyses, correlation_analysis
        )
        action_items = self._generate_action_items(
            overall_distribution, component_distributions, comparative_analyses
        )
        
        # Create comprehensive report
        report = QualityDistributionReport(
            generated_at=datetime.now().isoformat(),
            analysis_period=f"{days_back}_days",
            overall_distribution=overall_distribution,
            component_distributions=component_distributions,
            comparative_analyses=comparative_analyses,
            correlation_analysis=correlation_analysis,
            executive_summary=executive_summary,
            detailed_insights=detailed_insights,
            action_items=action_items
        )
        
        logger.info("✅ Comprehensive distribution report generated")
        return report
    
    def _create_empty_report(self) -> QualityDistributionReport:
        """Create empty report when no data is available."""
        from quality_distribution_analyzer import DistributionStatistics, QualityDistributionAnalysis
        
        empty_stats = DistributionStatistics(
            mean=0.0, median=0.0, mode=0.0, std_dev=0.0, variance=0.0,
            skewness=0.0, kurtosis=0.0, min_value=0.0, max_value=0.0,
            range_value=0.0, q1=0.0, q3=0.0, iqr=0.0,
            percentiles={}, coefficient_of_variation=0.0, mad=0.0
        )
        
        empty_distribution = QualityDistributionAnalysis(
            metric_name="overall_quality",
            sample_size=0,
            statistics=empty_stats,
            normality_tests=[],
            distribution_type='no_data',
            outliers=[],
            bins=[],
            frequencies=[],
            density=[],
            recommendations=["No data available for analysis"]
        )
        
        return QualityDistributionReport(
            generated_at=datetime.now().isoformat(),
            analysis_period="no_data",
            overall_distribution=empty_distribution,
            component_distributions={},
            comparative_analyses={},
            correlation_analysis={},
            executive_summary=["No quality data available for distribution analysis"],
            detailed_insights=["Please ensure quality validation has been run on conversations"],
            action_items=["Run quality validation pipeline to generate distribution data"]
        )
    
    def _generate_executive_summary(self, 
                                  overall_distribution: QualityDistributionAnalysis,
                                  component_distributions: Dict[str, QualityDistributionAnalysis],
                                  comparative_analyses: Dict[str, Any]) -> List[str]:
        """Generate executive summary for the distribution report."""
        summary = []
        
        # Overall distribution summary
        if overall_distribution.sample_size > 0:
            summary.append(f"📊 Analyzed {overall_distribution.sample_size:,} conversations for quality distribution")
            summary.append(f"📈 Overall quality: Mean={overall_distribution.statistics.mean:.3f}, Median={overall_distribution.statistics.median:.3f}")
            summary.append(f"📊 Distribution type: {overall_distribution.distribution_type.replace('_', ' ').title()}")
            
            # Variability assessment
            cv = overall_distribution.statistics.coefficient_of_variation
            if cv > 0.3:
                summary.append(f"⚠️ High quality variability detected (CV={cv:.3f})")
            elif cv < 0.1:
                summary.append(f"✅ Low quality variability - consistent performance (CV={cv:.3f})")
            
            # Outlier summary
            if len(overall_distribution.outliers) > 10:
                summary.append(f"🔍 {len(overall_distribution.outliers)} quality outliers detected requiring investigation")
        
        # Normality assessment
        if overall_distribution.normality_tests:
            normal_tests_passed = sum(1 for test in overall_distribution.normality_tests if test.is_normal)
            total_tests = len(overall_distribution.normality_tests)
            if normal_tests_passed / total_tests >= 0.5:
                summary.append("✅ Quality distribution is approximately normal")
            else:
                summary.append("⚠️ Quality distribution deviates from normality - consider non-parametric methods")
        
        # Component insights
        if component_distributions:
            best_component = max(component_distributions.items(), 
                               key=lambda x: x[1].statistics.mean)
            worst_component = min(component_distributions.items(), 
                                key=lambda x: x[1].statistics.mean)
            
            summary.append(f"🏆 Best performing component: {best_component[0].replace('_', ' ')} (mean: {best_component[1].statistics.mean:.3f})")
            summary.append(f"⚠️ Lowest performing component: {worst_component[0].replace('_', ' ')} (mean: {worst_component[1].statistics.mean:.3f})")
        
        # Comparative insights
        if 'tier' in comparative_analyses:
            tier_analysis = comparative_analyses['tier']
            if tier_analysis.statistical_tests:
                significant_tests = [test for test in tier_analysis.statistical_tests if test.get('significant', False)]
                if significant_tests:
                    summary.append("📊 Significant quality differences detected between tiers")
                else:
                    summary.append("✅ No significant quality differences between tiers")
        
        return summary
    
    def _generate_detailed_insights(self, 
                                  overall_distribution: QualityDistributionAnalysis,
                                  component_distributions: Dict[str, QualityDistributionAnalysis],
                                  comparative_analyses: Dict[str, Any],
                                  correlation_analysis: Dict[str, Any]) -> List[str]:
        """Generate detailed insights for the distribution report."""
        insights = []
        
        # Statistical insights
        if overall_distribution.sample_size > 0:
            stats = overall_distribution.statistics
            insights.append(f"📊 Distribution Statistics: Skewness={stats.skewness:.3f}, Kurtosis={stats.kurtosis:.3f}")
            insights.append(f"📈 Quality Range: {stats.min_value:.3f} to {stats.max_value:.3f} (range: {stats.range_value:.3f})")
            insights.append(f"📊 Quartiles: Q1={stats.q1:.3f}, Q3={stats.q3:.3f}, IQR={stats.iqr:.3f}")
        
        # Normality test insights
        if overall_distribution.normality_tests:
            for test in overall_distribution.normality_tests:
                insights.append(f"🧪 {test.test_name}: {test.interpretation}")
        
        # Component-specific insights
        if component_distributions:
            # Most variable component
            most_variable = max(component_distributions.items(), 
                              key=lambda x: x[1].statistics.coefficient_of_variation)
            insights.append(f"📈 Most variable component: {most_variable[0].replace('_', ' ')} (CV={most_variable[1].statistics.coefficient_of_variation:.3f})")
            
            # Most consistent component
            least_variable = min(component_distributions.items(), 
                               key=lambda x: x[1].statistics.coefficient_of_variation)
            insights.append(f"📊 Most consistent component: {least_variable[0].replace('_', ' ')} (CV={least_variable[1].statistics.coefficient_of_variation:.3f})")
        
        # Comparative insights
        for dimension, analysis in comparative_analyses.items():
            if hasattr(analysis, 'statistical_tests') and analysis.statistical_tests:
                significant_tests = [test for test in analysis.statistical_tests if test.get('significant', False)]
                if significant_tests:
                    insights.append(f"🔍 {dimension.title()} comparison: {significant_tests[0]['interpretation']}")
        
        # Correlation insights
        if correlation_analysis and 'strongest_correlations' in correlation_analysis:
            strongest_corr = correlation_analysis['strongest_correlations'][0] if correlation_analysis['strongest_correlations'] else None
            if strongest_corr:
                insights.append(f"🔗 Strongest correlation: {strongest_corr['component1']} ↔ {strongest_corr['component2']} (r={strongest_corr['correlation']:.3f})")
        
        return insights
    
    def _generate_action_items(self, 
                             overall_distribution: QualityDistributionAnalysis,
                             component_distributions: Dict[str, QualityDistributionAnalysis],
                             comparative_analyses: Dict[str, Any]) -> List[str]:
        """Generate actionable items based on distribution analysis."""
        actions = []
        
        # Overall distribution actions
        if overall_distribution.distribution_type in ['skewed_left', 'skewed_right', 'highly_skewed']:
            actions.append("📊 Consider data transformation to normalize skewed distribution")
        
        if len(overall_distribution.outliers) > 10:
            actions.append("🔍 Investigate quality outliers for data quality issues or process improvements")
        
        # Component-specific actions
        if component_distributions:
            # Focus on worst performing component
            worst_component = min(component_distributions.items(), 
                                key=lambda x: x[1].statistics.mean)
            if worst_component[1].statistics.mean < 0.6:  # Assuming 0-1 scale
                actions.append(f"🎯 Focus improvement efforts on {worst_component[0].replace('_', ' ')} component")
            
            # Address high variability
            high_variability_components = [
                name for name, analysis in component_distributions.items()
                if analysis.statistics.coefficient_of_variation > 0.3
            ]
            if high_variability_components:
                actions.append(f"📈 Reduce variability in: {', '.join(high_variability_components[:3])}")
        
        # Comparative analysis actions
        if 'tier' in comparative_analyses:
            tier_analysis = comparative_analyses['tier']
            if tier_analysis.groups:
                # Find worst performing tier
                tier_means = {name: analysis.statistics.mean for name, analysis in tier_analysis.groups.items()}
                worst_tier = min(tier_means.items(), key=lambda x: x[1])
                if worst_tier[1] < 0.6:
                    actions.append(f"📋 Review and improve quality processes for {worst_tier[0]} tier")
        
        # Statistical method recommendations
        if overall_distribution.normality_tests:
            normal_tests_passed = sum(1 for test in overall_distribution.normality_tests if test.is_normal)
            total_tests = len(overall_distribution.normality_tests)
            if normal_tests_passed / total_tests < 0.5:
                actions.append("📊 Use non-parametric statistical methods for quality analysis")
        
        return actions
    
    def create_distribution_visualizations(self, report: QualityDistributionReport) -> Dict[str, go.Figure]:
        """Create comprehensive distribution visualizations."""
        visualizations = {}
        
        # Overall distribution histogram
        if report.overall_distribution.bins and report.overall_distribution.frequencies:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=report.overall_distribution.bins,
                y=report.overall_distribution.frequencies,
                histnorm='probability density',
                name='Quality Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add mean and median lines
            stats = report.overall_distribution.statistics
            fig.add_vline(x=stats.mean, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {stats.mean:.3f}")
            fig.add_vline(x=stats.median, line_dash="dash", line_color="green", 
                         annotation_text=f"Median: {stats.median:.3f}")
            
            fig.update_layout(
                title='Overall Quality Distribution',
                xaxis_title='Quality Score',
                yaxis_title='Density',
                showlegend=True
            )
            
            visualizations['overall_histogram'] = fig
        
        # Component comparison box plot
        if report.component_distributions:
            fig = go.Figure()
            
            for component, analysis in report.component_distributions.items():
                # Create synthetic data for box plot (since we don't store raw data)
                stats = analysis.statistics
                synthetic_data = np.random.normal(stats.mean, stats.std_dev, 100)
                synthetic_data = np.clip(synthetic_data, stats.min_value, stats.max_value)
                
                fig.add_trace(go.Box(
                    y=synthetic_data,
                    name=component.replace('_', ' ').title(),
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title='Quality Components Distribution Comparison',
                yaxis_title='Quality Score',
                showlegend=False
            )
            
            visualizations['component_boxplot'] = fig
        
        # Comparative analysis visualization
        if 'tier' in report.comparative_analyses:
            tier_analysis = report.comparative_analyses['tier']
            if tier_analysis.groups:
                fig = go.Figure()
                
                tiers = list(tier_analysis.groups.keys())
                means = [analysis.statistics.mean for analysis in tier_analysis.groups.values()]
                stds = [analysis.statistics.std_dev for analysis in tier_analysis.groups.values()]
                
                fig.add_trace(go.Bar(
                    x=tiers,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    marker_color='lightcoral',
                    name='Mean Quality by Tier'
                ))
                
                fig.update_layout(
                    title='Quality Distribution by Tier',
                    xaxis_title='Tier',
                    yaxis_title='Mean Quality Score',
                    showlegend=False
                )
                
                visualizations['tier_comparison'] = fig
        
        # Correlation heatmap
        if report.correlation_analysis and 'correlation_matrix' in report.correlation_analysis:
            corr_matrix = report.correlation_analysis['correlation_matrix']
            
            # Convert to DataFrame for easier handling
            corr_df = pd.DataFrame(corr_matrix)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=[col.replace('_', ' ').title() for col in corr_df.columns],
                y=[col.replace('_', ' ').title() for col in corr_df.columns],
                colorscale='RdBu',
                zmid=0,
                text=corr_df.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Quality Components Correlation Matrix',
                xaxis_title='Quality Components',
                yaxis_title='Quality Components'
            )
            
            visualizations['correlation_heatmap'] = fig
        
        return visualizations
    
    def save_report(self, report: QualityDistributionReport, format: str = 'json') -> str:
        """Save distribution report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"quality_distribution_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert dataclasses to dict for JSON serialization
            report_dict = asdict(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
        elif format == 'html':
            filename = f"quality_distribution_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # Generate HTML report using template
            html_content = self._generate_html_report(report)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"📄 Report saved: {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, report: QualityDistributionReport) -> str:
        """Generate HTML report from distribution analysis."""
        template = Template(self.report_templates['detailed'])
        
        return template.render(
            report=report,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _get_executive_template(self) -> str:
        """Get executive summary template."""
        return """
        # Quality Distribution Analysis - Executive Summary
        
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
            <title>Quality Distribution Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .normal { color: green; }
                .skewed { color: orange; }
                .outliers { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Distribution Analysis Report</h1>
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
                <h2>Overall Distribution</h2>
                <div class="metric">
                    <strong>Sample Size:</strong> {{ report.overall_distribution.sample_size }}
                </div>
                <div class="metric">
                    <strong>Distribution Type:</strong> 
                    <span class="{{ report.overall_distribution.distribution_type }}">
                        {{ report.overall_distribution.distribution_type.replace('_', ' ').title() }}
                    </span>
                </div>
                <div class="metric">
                    <strong>Mean:</strong> {{ "%.3f"|format(report.overall_distribution.statistics.mean) }}
                </div>
                <div class="metric">
                    <strong>Median:</strong> {{ "%.3f"|format(report.overall_distribution.statistics.median) }}
                </div>
                <div class="metric">
                    <strong>Standard Deviation:</strong> {{ "%.3f"|format(report.overall_distribution.statistics.std_dev) }}
                </div>
                <div class="metric">
                    <strong>Outliers:</strong> 
                    <span class="outliers">{{ report.overall_distribution.outliers|length }}</span>
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
        # Technical Quality Distribution Analysis
        
        ## Statistical Summary
        - Sample Size: {{ report.overall_distribution.sample_size }}
        - Mean: {{ report.overall_distribution.statistics.mean }}
        - Median: {{ report.overall_distribution.statistics.median }}
        - Standard Deviation: {{ report.overall_distribution.statistics.std_dev }}
        - Skewness: {{ report.overall_distribution.statistics.skewness }}
        - Kurtosis: {{ report.overall_distribution.statistics.kurtosis }}
        
        ## Normality Tests
        {% for test in report.overall_distribution.normality_tests %}
        ### {{ test.test_name }}
        - Statistic: {{ test.statistic }}
        - P-value: {{ test.p_value }}
        - Result: {{ test.interpretation }}
        {% endfor %}
        """

def main():
    """Main function for testing the distribution reporter."""
    reporter = QualityDistributionReporter()
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(days_back=30)
    
    # Save report
    json_path = reporter.save_report(report, format='json')
    html_path = reporter.save_report(report, format='html')
    
    print(f"📊 Distribution Report Generated:")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    
    # Display summary
    print(f"\n📈 Executive Summary:")
    for item in report.executive_summary:
        print(f"  {item}")

if __name__ == "__main__":
    main()
