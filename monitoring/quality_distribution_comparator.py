#!/usr/bin/env python3
"""
Task 5.6.2.3: Quality Distribution Comparative Analysis

Enterprise-grade comparative analysis system for quality distributions
across different dimensions (tiers, datasets, components, time periods).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import warnings

# Import our distribution analyzer
from quality_distribution_analyzer import (
    QualityDistributionAnalyzer, 
    QualityDistributionAnalysis,
    ComparativeDistributionAnalysis,
    QualityDistributionReport
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityDistributionComparator:
    """
    Enterprise-grade comparative analysis for quality distributions.
    
    Provides statistical comparisons across tiers, datasets, components,
    and time periods with comprehensive hypothesis testing.
    """
    
    def __init__(self, analyzer: QualityDistributionAnalyzer):
        """Initialize the distribution comparator."""
        self.analyzer = analyzer
        self.significance_level = 0.05
        
        # Effect size thresholds (Cohen's conventions)
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
        
        logger.info("🔍 Quality Distribution Comparator initialized")
    
    def compare_across_tiers(self, df: pd.DataFrame, metric: str) -> ComparativeDistributionAnalysis:
        """Compare quality distributions across different tiers."""
        if df.empty or metric not in df.columns:
            return self._create_empty_comparative_analysis('tier')
        
        # Group data by tier
        tier_groups = {}
        tier_data = {}
        
        for tier in df['tier'].unique():
            tier_df = df[df['tier'] == tier]
            if len(tier_df) >= self.analyzer.min_sample_size:
                tier_analysis = self.analyzer.analyze_quality_distribution(
                    tier_df[metric], f"{metric}_{tier}"
                )
                tier_groups[tier] = tier_analysis
                tier_data[tier] = tier_df[metric].dropna()
        
        if len(tier_groups) < 2:
            return ComparativeDistributionAnalysis(
                dimension='tier',
                groups=tier_groups,
                statistical_tests=[],
                effect_sizes={},
                pairwise_comparisons=[],
                recommendations=["Insufficient tiers with adequate sample sizes for comparison"]
            )
        
        # Perform statistical tests
        statistical_tests = self._perform_group_comparisons(tier_data, 'tier')
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(tier_data)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(tier_data, 'tier')
        
        # Generate recommendations
        recommendations = self._generate_comparative_recommendations(
            tier_groups, statistical_tests, effect_sizes, 'tier'
        )
        
        return ComparativeDistributionAnalysis(
            dimension='tier',
            groups=tier_groups,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            pairwise_comparisons=pairwise_comparisons,
            recommendations=recommendations
        )
    
    def compare_across_datasets(self, df: pd.DataFrame, metric: str) -> ComparativeDistributionAnalysis:
        """Compare quality distributions across different datasets."""
        if df.empty or metric not in df.columns:
            return self._create_empty_comparative_analysis('dataset')
        
        # Group data by dataset
        dataset_groups = {}
        dataset_data = {}
        
        for dataset in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset]
            if len(dataset_df) >= self.analyzer.min_sample_size:
                dataset_analysis = self.analyzer.analyze_quality_distribution(
                    dataset_df[metric], f"{metric}_{dataset}"
                )
                dataset_groups[dataset] = dataset_analysis
                dataset_data[dataset] = dataset_df[metric].dropna()
        
        if len(dataset_groups) < 2:
            return ComparativeDistributionAnalysis(
                dimension='dataset',
                groups=dataset_groups,
                statistical_tests=[],
                effect_sizes={},
                pairwise_comparisons=[],
                recommendations=["Insufficient datasets with adequate sample sizes for comparison"]
            )
        
        # Perform statistical tests
        statistical_tests = self._perform_group_comparisons(dataset_data, 'dataset')
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(dataset_data)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(dataset_data, 'dataset')
        
        # Generate recommendations
        recommendations = self._generate_comparative_recommendations(
            dataset_groups, statistical_tests, effect_sizes, 'dataset'
        )
        
        return ComparativeDistributionAnalysis(
            dimension='dataset',
            groups=dataset_groups,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            pairwise_comparisons=pairwise_comparisons,
            recommendations=recommendations
        )
    
    def compare_across_components(self, df: pd.DataFrame) -> ComparativeDistributionAnalysis:
        """Compare distributions across different quality components."""
        if df.empty:
            return self._create_empty_comparative_analysis('component')
        
        # Analyze each component
        component_groups = {}
        component_data = {}
        
        for component in self.analyzer.quality_components:
            if component in df.columns:
                component_series = df[component].dropna()
                if len(component_series) >= self.analyzer.min_sample_size:
                    component_analysis = self.analyzer.analyze_quality_distribution(
                        component_series, component
                    )
                    component_groups[component] = component_analysis
                    component_data[component] = component_series
        
        if len(component_groups) < 2:
            return ComparativeDistributionAnalysis(
                dimension='component',
                groups=component_groups,
                statistical_tests=[],
                effect_sizes={},
                pairwise_comparisons=[],
                recommendations=["Insufficient components with adequate data for comparison"]
            )
        
        # Perform statistical tests
        statistical_tests = self._perform_group_comparisons(component_data, 'component')
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(component_data)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(component_data, 'component')
        
        # Generate recommendations
        recommendations = self._generate_comparative_recommendations(
            component_groups, statistical_tests, effect_sizes, 'component'
        )
        
        return ComparativeDistributionAnalysis(
            dimension='component',
            groups=component_groups,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            pairwise_comparisons=pairwise_comparisons,
            recommendations=recommendations
        )
    
    def compare_across_time_periods(self, df: pd.DataFrame, metric: str, period_type: str = 'month') -> ComparativeDistributionAnalysis:
        """Compare quality distributions across different time periods."""
        if df.empty or metric not in df.columns:
            return self._create_empty_comparative_analysis('time_period')
        
        # Group data by time period
        period_groups = {}
        period_data = {}
        
        period_column = period_type  # 'month' or 'week'
        if period_column not in df.columns:
            return self._create_empty_comparative_analysis('time_period')
        
        for period in df[period_column].unique():
            period_df = df[df[period_column] == period]
            if len(period_df) >= self.analyzer.min_sample_size:
                period_analysis = self.analyzer.analyze_quality_distribution(
                    period_df[metric], f"{metric}_{period}"
                )
                period_groups[period] = period_analysis
                period_data[period] = period_df[metric].dropna()
        
        if len(period_groups) < 2:
            return ComparativeDistributionAnalysis(
                dimension='time_period',
                groups=period_groups,
                statistical_tests=[],
                effect_sizes={},
                pairwise_comparisons=[],
                recommendations=["Insufficient time periods with adequate sample sizes for comparison"]
            )
        
        # Perform statistical tests
        statistical_tests = self._perform_group_comparisons(period_data, 'time_period')
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(period_data)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(period_data, 'time_period')
        
        # Generate recommendations
        recommendations = self._generate_comparative_recommendations(
            period_groups, statistical_tests, effect_sizes, 'time_period'
        )
        
        return ComparativeDistributionAnalysis(
            dimension='time_period',
            groups=period_groups,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            pairwise_comparisons=pairwise_comparisons,
            recommendations=recommendations
        )
    
    def _perform_group_comparisons(self, group_data: Dict[str, pd.Series], dimension: str) -> List[Dict[str, Any]]:
        """Perform statistical tests for group comparisons."""
        tests = []
        
        if len(group_data) < 2:
            return tests
        
        # Prepare data for tests
        groups = list(group_data.values())
        group_names = list(group_data.keys())
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        try:
            if len(groups) > 2:
                statistic, p_value = kruskal(*groups)
                tests.append({
                    'test_name': 'Kruskal-Wallis',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.significance_level,
                    'interpretation': f"{'Significant' if p_value < self.significance_level else 'Non-significant'} difference between {dimension}s (p={p_value:.4f})",
                    'groups_compared': group_names
                })
        except Exception as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")
        
        # ANOVA (if assumptions are met)
        try:
            if len(groups) > 2:
                # Check if data is approximately normal (simplified check)
                all_normal = True
                for group in groups:
                    if len(group) > 8:
                        _, p_norm = stats.normaltest(group)
                        if p_norm < 0.05:
                            all_normal = False
                            break
                
                if all_normal:
                    statistic, p_value = stats.f_oneway(*groups)
                    tests.append({
                        'test_name': 'One-way ANOVA',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_level,
                        'interpretation': f"{'Significant' if p_value < self.significance_level else 'Non-significant'} difference between {dimension}s (p={p_value:.4f})",
                        'groups_compared': group_names
                    })
        except Exception as e:
            logger.warning(f"ANOVA test failed: {e}")
        
        return tests
    
    def _calculate_effect_sizes(self, group_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate effect sizes for group comparisons."""
        effect_sizes = {}
        
        if len(group_data) < 2:
            return effect_sizes
        
        group_names = list(group_data.keys())
        
        # Calculate Cohen's d for pairwise comparisons
        for i, group1_name in enumerate(group_names):
            for j, group2_name in enumerate(group_names[i+1:], i+1):
                group1 = group_data[group1_name]
                group2 = group_data[group2_name]
                
                # Cohen's d
                pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var()) / 
                                   (len(group1) + len(group2) - 2))
                
                if pooled_std > 0:
                    cohens_d = (group1.mean() - group2.mean()) / pooled_std
                    effect_sizes[f"{group1_name}_vs_{group2_name}"] = float(abs(cohens_d))
        
        return effect_sizes
    
    def _perform_pairwise_comparisons(self, group_data: Dict[str, pd.Series], dimension: str) -> List[Dict[str, Any]]:
        """Perform pairwise comparisons between groups."""
        comparisons = []
        
        if len(group_data) < 2:
            return comparisons
        
        group_names = list(group_data.keys())
        
        # Pairwise Mann-Whitney U tests
        for i, group1_name in enumerate(group_names):
            for j, group2_name in enumerate(group_names[i+1:], i+1):
                group1 = group_data[group1_name]
                group2 = group_data[group2_name]
                
                try:
                    statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    
                    # Calculate effect size (r = Z / sqrt(N))
                    n1, n2 = len(group1), len(group2)
                    z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
                    effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
                    
                    comparisons.append({
                        'group1': group1_name,
                        'group2': group2_name,
                        'test': 'Mann-Whitney U',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_level,
                        'effect_size': float(effect_size_r),
                        'mean_difference': float(group1.mean() - group2.mean()),
                        'interpretation': f"{'Significant' if p_value < self.significance_level else 'Non-significant'} difference between {group1_name} and {group2_name}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Mann-Whitney U test failed for {group1_name} vs {group2_name}: {e}")
        
        return comparisons
    
    def _generate_comparative_recommendations(self, 
                                            groups: Dict[str, QualityDistributionAnalysis],
                                            tests: List[Dict[str, Any]],
                                            effect_sizes: Dict[str, float],
                                            dimension: str) -> List[str]:
        """Generate recommendations based on comparative analysis."""
        recommendations = []
        
        # Overall comparison results
        significant_tests = [test for test in tests if test.get('significant', False)]
        if significant_tests:
            recommendations.append(f"📊 Significant differences detected between {dimension}s. Further investigation recommended.")
        else:
            recommendations.append(f"✅ No significant differences detected between {dimension}s.")
        
        # Effect size recommendations
        large_effects = [name for name, size in effect_sizes.items() if size > self.effect_size_thresholds['large']]
        if large_effects:
            recommendations.append(f"🔍 Large effect sizes detected: {', '.join(large_effects[:3])}. Investigate practical significance.")
        
        # Group-specific recommendations
        if groups:
            # Find best and worst performing groups
            group_means = {name: analysis.statistics.mean for name, analysis in groups.items()}
            best_group = max(group_means.items(), key=lambda x: x[1])
            worst_group = min(group_means.items(), key=lambda x: x[1])
            
            if len(groups) > 1:
                recommendations.append(f"🏆 Best performing {dimension}: {best_group[0]} (mean: {best_group[1]:.3f})")
                recommendations.append(f"⚠️ Worst performing {dimension}: {worst_group[0]} (mean: {worst_group[1]:.3f})")
        
        # Variability recommendations
        if groups:
            cvs = {name: analysis.statistics.coefficient_of_variation for name, analysis in groups.items()}
            most_variable = max(cvs.items(), key=lambda x: x[1])
            least_variable = min(cvs.items(), key=lambda x: x[1])
            
            if most_variable[1] > 0.3:
                recommendations.append(f"📈 High variability in {most_variable[0]} (CV: {most_variable[1]:.3f})")
            if least_variable[1] < 0.1:
                recommendations.append(f"📊 Low variability in {least_variable[0]} (CV: {least_variable[1]:.3f})")
        
        return recommendations
    
    def _create_empty_comparative_analysis(self, dimension: str) -> ComparativeDistributionAnalysis:
        """Create empty comparative analysis for insufficient data."""
        return ComparativeDistributionAnalysis(
            dimension=dimension,
            groups={},
            statistical_tests=[],
            effect_sizes={},
            pairwise_comparisons=[],
            recommendations=[f"Insufficient data for {dimension} comparison analysis"]
        )
    
    def calculate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation analysis between quality components."""
        if df.empty:
            return {}
        
        # Select quality components that exist in the data
        available_components = [comp for comp in self.analyzer.quality_components if comp in df.columns]
        
        if len(available_components) < 2:
            return {'error': 'Insufficient components for correlation analysis'}
        
        # Calculate correlation matrix
        correlation_data = df[available_components].corr()
        
        # Find strongest correlations
        correlations = []
        for i, comp1 in enumerate(available_components):
            for j, comp2 in enumerate(available_components[i+1:], i+1):
                corr_value = correlation_data.loc[comp1, comp2]
                if not np.isnan(corr_value):
                    correlations.append({
                        'component1': comp1,
                        'component2': comp2,
                        'correlation': float(corr_value),
                        'strength': self._interpret_correlation_strength(abs(corr_value))
                    })
        
        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_data.to_dict(),
            'strongest_correlations': correlations[:10],  # Top 10
            'component_count': len(available_components),
            'analysis_summary': self._generate_correlation_summary(correlations)
        }
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'negligible'
    
    def _generate_correlation_summary(self, correlations: List[Dict[str, Any]]) -> List[str]:
        """Generate correlation analysis summary."""
        summary = []
        
        if not correlations:
            return ["No significant correlations found"]
        
        # Strongest positive correlation
        strongest_pos = max([c for c in correlations if c['correlation'] > 0], 
                           key=lambda x: x['correlation'], default=None)
        if strongest_pos:
            summary.append(f"🔗 Strongest positive correlation: {strongest_pos['component1']} ↔ {strongest_pos['component2']} (r={strongest_pos['correlation']:.3f})")
        
        # Strongest negative correlation
        strongest_neg = min([c for c in correlations if c['correlation'] < 0], 
                           key=lambda x: x['correlation'], default=None)
        if strongest_neg:
            summary.append(f"🔗 Strongest negative correlation: {strongest_neg['component1']} ↔ {strongest_neg['component2']} (r={strongest_neg['correlation']:.3f})")
        
        # Count strong correlations
        strong_correlations = [c for c in correlations if c['strength'] == 'strong']
        if strong_correlations:
            summary.append(f"💪 {len(strong_correlations)} strong correlations detected")
        
        return summary

def main():
    """Main function for testing the distribution comparator."""
    from quality_distribution_analyzer import QualityDistributionAnalyzer
    
    analyzer = QualityDistributionAnalyzer()
    comparator = QualityDistributionComparator(analyzer)
    
    # Load quality data
    df = analyzer.load_quality_data(days_back=30)
    
    if df.empty:
        print("❌ No quality data available for comparative analysis")
        return
    
    # Compare across tiers
    tier_comparison = comparator.compare_across_tiers(df, 'overall_quality')
    
    print(f"📊 Tier Comparison Results:")
    print(f"Groups: {len(tier_comparison.groups)}")
    print(f"Statistical Tests: {len(tier_comparison.statistical_tests)}")
    print(f"Effect Sizes: {len(tier_comparison.effect_sizes)}")
    print(f"Pairwise Comparisons: {len(tier_comparison.pairwise_comparisons)}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(tier_comparison.recommendations, 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
