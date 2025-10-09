#!/usr/bin/env python3
"""
Task 5.6.2.5: Quality Comparison Reporting System

Enterprise-grade reporting system for quality comparisons with
comprehensive analysis, visualizations, and executive summaries.
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

# Import our quality comparator
from quality_comparator import (
    QualityComparator,
    QualityComparison,
    BenchmarkAnalysis,
    ComparisonReport
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityComparisonReporter:
    """
    Enterprise-grade quality comparison reporting system.
    
    Generates comprehensive comparison reports with visualizations,
    statistical analysis, and executive summaries.
    """
    
    def __init__(self, output_dir: str = "/home/vivi/pixelated/ai/monitoring/reports"):
        """Initialize the comparison reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparator = QualityComparator()
        
        # Report templates
        self.report_templates = {
            'executive': self._get_executive_template(),
            'detailed': self._get_detailed_template(),
            'technical': self._get_technical_template()
        }
        
        logger.info("📊 Quality Comparison Reporter initialized")
    
    def generate_comprehensive_report(self, 
                                    days_back: int = 90,
                                    include_visualizations: bool = True) -> ComparisonReport:
        """Generate comprehensive quality comparison report."""
        logger.info(f"📈 Generating comprehensive comparison report ({days_back} days)")
        
        try:
            # Load comparison data
            df = self.comparator.load_comparison_data(days_back=days_back)
            
            if df.empty:
                logger.error("❌ No data available for comparison analysis")
                return self._create_empty_report()
            
            # Perform tier comparisons
            logger.info("🔍 Performing tier comparisons...")
            tier_comparisons = self.comparator.compare_tiers(df)
            
            # Perform dataset comparisons
            logger.info("📁 Performing dataset comparisons...")
            dataset_comparisons = self.comparator.compare_datasets(df)
            
            # Perform component comparisons
            logger.info("🧩 Performing component comparisons...")
            component_comparisons = self.comparator.compare_components(df)
            
            # Perform benchmark analysis
            logger.info("📊 Performing benchmark analysis...")
            benchmark_analyses = self.comparator.perform_benchmark_analysis(df)
            
            # Calculate performance rankings
            logger.info("🏆 Calculating performance rankings...")
            performance_rankings = self.comparator.calculate_performance_rankings(df)
            
            # Generate summaries
            executive_summary = self._generate_executive_summary(
                tier_comparisons, dataset_comparisons, component_comparisons, 
                benchmark_analyses, performance_rankings
            )
            detailed_insights = self._generate_detailed_insights(
                tier_comparisons, dataset_comparisons, component_comparisons, benchmark_analyses
            )
            action_items = self._generate_action_items(
                tier_comparisons, dataset_comparisons, benchmark_analyses, performance_rankings
            )
            
            # Create comprehensive report
            report = ComparisonReport(
                generated_at=datetime.now().isoformat(),
                analysis_period=f"{days_back}_days",
                tier_comparisons=tier_comparisons,
                dataset_comparisons=dataset_comparisons,
                component_comparisons=component_comparisons,
                benchmark_analyses=benchmark_analyses,
                performance_rankings=performance_rankings,
                executive_summary=executive_summary,
                detailed_insights=detailed_insights,
                action_items=action_items
            )
            
            logger.info("✅ Comprehensive comparison report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating comparison report: {e}")
            return self._create_empty_report()
    
    def _create_empty_report(self) -> ComparisonReport:
        """Create empty report when no data is available."""
        return ComparisonReport(
            generated_at=datetime.now().isoformat(),
            analysis_period="no_data",
            tier_comparisons=[],
            dataset_comparisons=[],
            component_comparisons=[],
            benchmark_analyses=[],
            performance_rankings={},
            executive_summary=["No quality data available for comparison analysis"],
            detailed_insights=["Please ensure quality validation has been run on conversations"],
            action_items=["Run quality validation pipeline to generate comparison data"]
        )
    
    def _generate_executive_summary(self,
                                  tier_comparisons: List[QualityComparison],
                                  dataset_comparisons: List[QualityComparison],
                                  component_comparisons: List[QualityComparison],
                                  benchmark_analyses: List[BenchmarkAnalysis],
                                  performance_rankings: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate executive summary for the comparison report."""
        summary = []
        
        # Overall comparison summary
        total_comparisons = len(tier_comparisons) + len(dataset_comparisons) + len(component_comparisons)
        summary.append(f"📊 Quality Comparison Analysis: {total_comparisons} comparisons performed")
        
        # Tier comparison insights
        if tier_comparisons:
            significant_tier_comparisons = [c for c in tier_comparisons 
                                          if any(test.get('significant', False) for test in c.statistical_tests)]
            if significant_tier_comparisons:
                summary.append(f"🎯 {len(significant_tier_comparisons)} significant differences detected between tiers")
                
                # Find largest tier difference
                largest_diff = max(tier_comparisons, 
                                 key=lambda x: abs(x.group1_stats['mean'] - x.group2_stats['mean']))
                diff = abs(largest_diff.group1_stats['mean'] - largest_diff.group2_stats['mean'])
                better_tier = largest_diff.group1_name if largest_diff.group1_stats['mean'] > largest_diff.group2_stats['mean'] else largest_diff.group2_name
                summary.append(f"🏆 Largest tier gap: {better_tier} outperforms by {diff:.3f} points")
        
        # Dataset comparison insights
        if dataset_comparisons:
            significant_dataset_comparisons = [c for c in dataset_comparisons 
                                             if any(test.get('significant', False) for test in c.statistical_tests)]
            if significant_dataset_comparisons:
                summary.append(f"📁 {len(significant_dataset_comparisons)} significant differences detected between datasets")
        
        # Performance rankings insights
        if 'tiers' in performance_rankings and performance_rankings['tiers']:
            best_tier = performance_rankings['tiers'][0]
            worst_tier = performance_rankings['tiers'][-1]
            summary.append(f"🥇 Best performing tier: {best_tier['name']} (quality: {best_tier['mean_quality']:.3f})")
            summary.append(f"⚠️ Lowest performing tier: {worst_tier['name']} (quality: {worst_tier['mean_quality']:.3f})")
        
        if 'datasets' in performance_rankings and performance_rankings['datasets']:
            best_dataset = performance_rankings['datasets'][0]
            summary.append(f"📊 Best performing dataset: {best_dataset['name']} (quality: {best_dataset['mean_quality']:.3f})")
        
        # Benchmark analysis insights
        if benchmark_analyses:
            above_benchmark = [b for b in benchmark_analyses if b.performance_gap > 0.05]
            below_benchmark = [b for b in benchmark_analyses if b.performance_gap < -0.05]
            
            if above_benchmark:
                summary.append(f"✅ {len(above_benchmark)} groups exceed industry benchmarks")
            if below_benchmark:
                summary.append(f"⚠️ {len(below_benchmark)} groups below industry benchmarks requiring attention")
        
        return summary
    
    def _generate_detailed_insights(self,
                                  tier_comparisons: List[QualityComparison],
                                  dataset_comparisons: List[QualityComparison],
                                  component_comparisons: List[QualityComparison],
                                  benchmark_analyses: List[BenchmarkAnalysis]) -> List[str]:
        """Generate detailed insights for the comparison report."""
        insights = []
        
        # Effect size insights
        all_comparisons = tier_comparisons + dataset_comparisons + component_comparisons
        if all_comparisons:
            large_effects = [c for c in all_comparisons if c.effect_size >= 0.8]
            medium_effects = [c for c in all_comparisons if 0.5 <= c.effect_size < 0.8]
            
            if large_effects:
                insights.append(f"💪 {len(large_effects)} comparisons show large effect sizes (>0.8)")
            if medium_effects:
                insights.append(f"📊 {len(medium_effects)} comparisons show medium effect sizes (0.5-0.8)")
        
        # Statistical significance insights
        if all_comparisons:
            significant_comparisons = [c for c in all_comparisons 
                                     if any(test.get('significant', False) for test in c.statistical_tests)]
            significance_rate = len(significant_comparisons) / len(all_comparisons) * 100
            insights.append(f"📈 {significance_rate:.1f}% of comparisons are statistically significant")
        
        # Tier-specific insights
        if tier_comparisons:
            practical_tier_differences = [c for c in tier_comparisons if c.practical_significance]
            if practical_tier_differences:
                insights.append(f"🎯 {len(practical_tier_differences)} tier comparisons show practical significance")
        
        # Component insights
        if component_comparisons:
            # Find most variable component comparison
            most_variable = max(component_comparisons, 
                              key=lambda x: max(x.group1_stats['std'], x.group2_stats['std']))
            insights.append(f"📊 Most variable component comparison: {most_variable.group1_name} vs {most_variable.group2_name}")
        
        # Benchmark insights
        if benchmark_analyses:
            avg_gap = np.mean([b.performance_gap for b in benchmark_analyses])
            if avg_gap > 0:
                insights.append(f"📈 Average performance gap above benchmark: +{avg_gap:.3f} points")
            else:
                insights.append(f"📉 Average performance gap below benchmark: {avg_gap:.3f} points")
            
            # Improvement potential
            avg_potential = np.mean([b.improvement_potential for b in benchmark_analyses])
            insights.append(f"🎯 Average improvement potential: {avg_potential:.3f} points")
        
        return insights
    
    def _generate_action_items(self,
                             tier_comparisons: List[QualityComparison],
                             dataset_comparisons: List[QualityComparison],
                             benchmark_analyses: List[BenchmarkAnalysis],
                             performance_rankings: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate actionable items based on comparison analysis."""
        actions = []
        
        # Tier-based actions
        if tier_comparisons:
            significant_tier_gaps = [c for c in tier_comparisons 
                                   if c.practical_significance and 
                                   abs(c.group1_stats['mean'] - c.group2_stats['mean']) > 0.1]
            
            if significant_tier_gaps:
                actions.append(f"🎯 Address {len(significant_tier_gaps)} significant tier performance gaps")
                
                # Specific tier improvement actions
                for comparison in significant_tier_gaps[:3]:  # Top 3
                    better_tier = comparison.group1_name if comparison.group1_stats['mean'] > comparison.group2_stats['mean'] else comparison.group2_name
                    worse_tier = comparison.group2_name if comparison.group1_stats['mean'] > comparison.group2_stats['mean'] else comparison.group1_name
                    actions.append(f"📈 Apply {better_tier} best practices to improve {worse_tier}")
        
        # Dataset-based actions
        if dataset_comparisons:
            poor_performing_datasets = []
            for comparison in dataset_comparisons:
                if comparison.practical_significance:
                    worse_dataset = comparison.group1_name if comparison.group1_stats['mean'] < comparison.group2_stats['mean'] else comparison.group2_name
                    poor_performing_datasets.append(worse_dataset)
            
            if poor_performing_datasets:
                unique_datasets = list(set(poor_performing_datasets))
                actions.append(f"📁 Review data quality and processing for: {', '.join(unique_datasets[:3])}")
        
        # Benchmark-based actions
        if benchmark_analyses:
            below_benchmark = [b for b in benchmark_analyses if b.performance_gap < -0.05]
            if below_benchmark:
                actions.append(f"📊 Implement improvement initiatives for {len(below_benchmark)} groups below benchmark")
            
            # High improvement potential
            high_potential = [b for b in benchmark_analyses if b.improvement_potential > 0.2]
            if high_potential:
                actions.append(f"🎯 Focus on {len(high_potential)} groups with high improvement potential (>0.2 points)")
        
        # Performance ranking actions
        if 'tiers' in performance_rankings and len(performance_rankings['tiers']) > 1:
            worst_tier = performance_rankings['tiers'][-1]
            if worst_tier['mean_quality'] < 0.6:  # Below 60% quality
                actions.append(f"🚨 Urgent attention needed for {worst_tier['name']} tier (quality: {worst_tier['mean_quality']:.3f})")
        
        # Statistical significance actions
        all_comparisons = tier_comparisons + dataset_comparisons
        if all_comparisons:
            non_significant = [c for c in all_comparisons 
                             if not any(test.get('significant', False) for test in c.statistical_tests)]
            if len(non_significant) / len(all_comparisons) > 0.5:
                actions.append("📈 Increase sample sizes for more reliable statistical comparisons")
        
        return actions
    
    def create_comparison_visualizations(self, report: ComparisonReport) -> Dict[str, go.Figure]:
        """Create comprehensive comparison visualizations."""
        visualizations = {}
        
        # Performance rankings visualization
        if report.performance_rankings:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Tier Rankings', 'Dataset Rankings', 'Component Rankings'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            # Tier rankings
            if 'tiers' in report.performance_rankings:
                tiers = report.performance_rankings['tiers']
                tier_names = [t['name'] for t in tiers]
                tier_qualities = [t['mean_quality'] for t in tiers]
                
                fig.add_trace(
                    go.Bar(x=tier_names, y=tier_qualities, name='Tiers',
                          marker_color='lightblue'),
                    row=1, col=1
                )
            
            # Dataset rankings
            if 'datasets' in report.performance_rankings:
                datasets = report.performance_rankings['datasets'][:5]  # Top 5
                dataset_names = [d['name'][:15] + '...' if len(d['name']) > 15 else d['name'] for d in datasets]
                dataset_qualities = [d['mean_quality'] for d in datasets]
                
                fig.add_trace(
                    go.Bar(x=dataset_names, y=dataset_qualities, name='Datasets',
                          marker_color='lightgreen'),
                    row=1, col=2
                )
            
            # Component rankings
            if 'components' in report.performance_rankings:
                components = report.performance_rankings['components']
                component_names = [c['name'].replace('_', ' ').title() for c in components]
                component_qualities = [c['mean_quality'] for c in components]
                
                fig.add_trace(
                    go.Bar(x=component_names, y=component_qualities, name='Components',
                          marker_color='lightcoral'),
                    row=1, col=3
                )
            
            fig.update_layout(
                title='Quality Performance Rankings',
                showlegend=False,
                height=500
            )
            fig.update_xaxes(tickangle=45)
            
            visualizations['performance_rankings'] = fig
        
        # Comparison effect sizes
        if report.tier_comparisons or report.dataset_comparisons:
            fig = go.Figure()
            
            # Tier comparison effect sizes
            if report.tier_comparisons:
                tier_names = [f"{c.group1_name} vs {c.group2_name}" for c in report.tier_comparisons]
                tier_effects = [c.effect_size for c in report.tier_comparisons]
                
                fig.add_trace(go.Bar(
                    x=tier_names,
                    y=tier_effects,
                    name='Tier Comparisons',
                    marker_color='lightblue'
                ))
            
            # Dataset comparison effect sizes
            if report.dataset_comparisons:
                dataset_names = [f"{c.group1_name} vs {c.group2_name}" for c in report.dataset_comparisons[:5]]
                dataset_effects = [c.effect_size for c in report.dataset_comparisons[:5]]
                
                fig.add_trace(go.Bar(
                    x=dataset_names,
                    y=dataset_effects,
                    name='Dataset Comparisons',
                    marker_color='lightgreen'
                ))
            
            # Add effect size threshold lines
            fig.add_hline(y=0.2, line_dash="dash", line_color="orange", 
                         annotation_text="Small Effect")
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Medium Effect")
            fig.add_hline(y=0.8, line_dash="dash", line_color="darkred", 
                         annotation_text="Large Effect")
            
            fig.update_layout(
                title='Comparison Effect Sizes',
                xaxis_title='Comparisons',
                yaxis_title='Effect Size (Cohen\'s d)',
                xaxis_tickangle=45
            )
            
            visualizations['effect_sizes'] = fig
        
        # Benchmark analysis
        if report.benchmark_analyses:
            fig = go.Figure()
            
            benchmark_names = [b.target_group for b in report.benchmark_analyses]
            performance_gaps = [b.performance_gap for b in report.benchmark_analyses]
            colors = ['green' if gap > 0 else 'red' for gap in performance_gaps]
            
            fig.add_trace(go.Bar(
                x=benchmark_names,
                y=performance_gaps,
                marker_color=colors,
                name='Performance Gap'
            ))
            
            fig.add_hline(y=0, line_color="black", line_width=2)
            
            fig.update_layout(
                title='Benchmark Analysis - Performance Gaps',
                xaxis_title='Groups',
                yaxis_title='Performance Gap vs Benchmark',
                xaxis_tickangle=45
            )
            
            visualizations['benchmark_analysis'] = fig
        
        return visualizations
    
    def save_report(self, report: ComparisonReport, format: str = 'json') -> str:
        """Save comparison report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"quality_comparison_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert dataclasses to dict for JSON serialization
            report_dict = asdict(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
        elif format == 'html':
            filename = f"quality_comparison_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # Generate HTML report using template
            html_content = self._generate_html_report(report)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"📄 Report saved: {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, report: ComparisonReport) -> str:
        """Generate HTML report from comparison analysis."""
        template = Template(self.report_templates['detailed'])
        
        return template.render(
            report=report,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _get_executive_template(self) -> str:
        """Get executive summary template."""
        return """
        # Quality Comparison Report - Executive Summary
        
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
            <title>Quality Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .significant { color: green; }
                .non-significant { color: orange; }
                .below-benchmark { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Comparison Report</h1>
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
                <h2>Comparison Results</h2>
                <div class="metric">
                    <strong>Tier Comparisons:</strong> {{ report.tier_comparisons|length }}
                </div>
                <div class="metric">
                    <strong>Dataset Comparisons:</strong> {{ report.dataset_comparisons|length }}
                </div>
                <div class="metric">
                    <strong>Component Comparisons:</strong> {{ report.component_comparisons|length }}
                </div>
                <div class="metric">
                    <strong>Benchmark Analyses:</strong> {{ report.benchmark_analyses|length }}
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
        # Technical Quality Comparison Analysis
        
        ## Statistical Summary
        - Tier Comparisons: {{ report.tier_comparisons|length }}
        - Dataset Comparisons: {{ report.dataset_comparisons|length }}
        - Component Comparisons: {{ report.component_comparisons|length }}
        - Benchmark Analyses: {{ report.benchmark_analyses|length }}
        
        ## Comparison Details
        {% for comparison in report.tier_comparisons %}
        ### {{ comparison.group1_name }} vs {{ comparison.group2_name }}
        - Effect Size: {{ comparison.effect_size }}
        - Practical Significance: {{ comparison.practical_significance }}
        - Statistical Tests: {{ comparison.statistical_tests|length }}
        {% endfor %}
        """

def main():
    """Main function for testing the comparison reporter."""
    reporter = QualityComparisonReporter()
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(days_back=30)
    
    # Save report
    json_path = reporter.save_report(report, format='json')
    html_path = reporter.save_report(report, format='html')
    
    print(f"📊 Comparison Report Generated:")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    
    # Display summary
    print(f"\n📈 Executive Summary:")
    for item in report.executive_summary:
        print(f"  {item}")

if __name__ == "__main__":
    main()
