#!/usr/bin/env python3
"""
Task 5.6.2.5: Quality Comparison System

Enterprise-grade quality comparison system providing comprehensive
cross-tier analysis, dataset benchmarking, and comparative reporting.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, f_oneway
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityComparison:
    """Quality comparison results between groups."""
    comparison_type: str  # 'tier', 'dataset', 'component', 'time_period'
    group1_name: str
    group2_name: str
    group1_stats: Dict[str, float]
    group2_stats: Dict[str, float]
    statistical_tests: List[Dict[str, Any]]
    effect_size: float
    practical_significance: bool
    confidence_interval: Tuple[float, float]
    recommendations: List[str]

@dataclass
class BenchmarkAnalysis:
    """Benchmark analysis results."""
    benchmark_type: str  # 'tier_benchmark', 'dataset_benchmark', 'industry_standard'
    target_group: str
    benchmark_group: str
    performance_gap: float
    percentile_ranking: float
    improvement_potential: float
    benchmark_metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class ComparisonReport:
    """Complete quality comparison report."""
    generated_at: str
    analysis_period: str
    tier_comparisons: List[QualityComparison]
    dataset_comparisons: List[QualityComparison]
    component_comparisons: List[QualityComparison]
    benchmark_analyses: List[BenchmarkAnalysis]
    performance_rankings: Dict[str, List[Dict[str, Any]]]
    executive_summary: List[str]
    detailed_insights: List[str]
    action_items: List[str]

class QualityComparator:
    """
    Enterprise-grade quality comparison system.
    
    Provides comprehensive cross-tier analysis, dataset benchmarking,
    and comparative reporting with statistical validation.
    """
    
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        """Initialize the quality comparator."""
        self.db_path = db_path
        self.cache_duration = 300  # 5 minutes cache
        self._cached_data = {}
        self._cache_timestamps = {}
        
        # Analysis parameters
        self.min_sample_size = 30  # Minimum sample size for meaningful comparison
        self.significance_threshold = 0.05  # Statistical significance threshold
        
        # Effect size thresholds (Cohen's conventions)
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
        
        # Quality components for comparison
        self.quality_components = [
            'therapeutic_accuracy',
            'conversation_coherence',
            'emotional_authenticity',
            'clinical_compliance',
            'personality_consistency',
            'language_quality',
            'safety_score',
            'overall_quality'
        ]
        
        # Industry benchmarks (placeholder values - would be updated with real data)
        self.industry_benchmarks = {
            'therapeutic_accuracy': 0.75,
            'conversation_coherence': 0.78,
            'emotional_authenticity': 0.72,
            'clinical_compliance': 0.80,
            'personality_consistency': 0.70,
            'language_quality': 0.76,
            'safety_score': 0.85,
            'overall_quality': 0.75
        }
        
        logger.info("🎯 Quality Comparator initialized")
    
    def load_comparison_data(self, 
                           days_back: int = 90,
                           force_refresh: bool = False) -> pd.DataFrame:
        """Load quality data for comparison analysis."""
        cache_key = f"comparison_data_{days_back}"
        current_time = datetime.now()
        
        # Check cache validity
        if (not force_refresh and 
            cache_key in self._cached_data and 
            cache_key in self._cache_timestamps and
            (current_time - self._cache_timestamps[cache_key]).seconds < self.cache_duration):
            return self._cached_data[cache_key]
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Load comprehensive quality data
            query = """
            SELECT 
                c.id,
                c.tier,
                c.dataset_name,
                c.created_at,
                c.conversation_length,
                q.therapeutic_accuracy,
                q.conversation_coherence,
                q.emotional_authenticity,
                q.clinical_compliance,
                q.personality_consistency,
                q.language_quality,
                q.safety_score,
                q.overall_quality,
                q.validated_at,
                DATE(c.created_at) as date,
                strftime('%Y-%m', c.created_at) as month,
                strftime('%Y-%W', c.created_at) as week
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            WHERE q.overall_quality IS NOT NULL
            AND c.created_at >= ?
            AND c.created_at <= ?
            ORDER BY c.created_at
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=[start_date.isoformat(), end_date.isoformat()]
            )
            conn.close()
            
            if df.empty:
                logger.warning("⚠️ No quality data found for comparison analysis")
                return df
            
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['validated_at'] = pd.to_datetime(df['validated_at'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Cache the data
            self._cached_data[cache_key] = df
            self._cache_timestamps[cache_key] = current_time
            
            logger.info(f"✅ Loaded {len(df)} quality records for comparison analysis")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading comparison data: {e}")
            return pd.DataFrame()
    
    def compare_tiers(self, df: pd.DataFrame, metric: str = 'overall_quality') -> List[QualityComparison]:
        """Compare quality across different tiers."""
        if df.empty or metric not in df.columns:
            return []
        
        comparisons = []
        tiers = df['tier'].unique()
        
        # Pairwise tier comparisons
        for i, tier1 in enumerate(tiers):
            for tier2 in tiers[i+1:]:
                tier1_data = df[df['tier'] == tier1][metric].dropna()
                tier2_data = df[df['tier'] == tier2][metric].dropna()
                
                if len(tier1_data) >= self.min_sample_size and len(tier2_data) >= self.min_sample_size:
                    comparison = self._perform_comparison(
                        tier1_data, tier2_data, tier1, tier2, 'tier'
                    )
                    if comparison:
                        comparisons.append(comparison)
        
        return comparisons
    
    def compare_datasets(self, df: pd.DataFrame, metric: str = 'overall_quality') -> List[QualityComparison]:
        """Compare quality across different datasets."""
        if df.empty or metric not in df.columns:
            return []
        
        comparisons = []
        datasets = df['dataset_name'].unique()
        
        # Pairwise dataset comparisons
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                dataset1_data = df[df['dataset_name'] == dataset1][metric].dropna()
                dataset2_data = df[df['dataset_name'] == dataset2][metric].dropna()
                
                if len(dataset1_data) >= self.min_sample_size and len(dataset2_data) >= self.min_sample_size:
                    comparison = self._perform_comparison(
                        dataset1_data, dataset2_data, dataset1, dataset2, 'dataset'
                    )
                    if comparison:
                        comparisons.append(comparison)
        
        return comparisons
    
    def compare_components(self, df: pd.DataFrame) -> List[QualityComparison]:
        """Compare quality across different components."""
        if df.empty:
            return []
        
        comparisons = []
        available_components = [comp for comp in self.quality_components if comp in df.columns]
        
        # Pairwise component comparisons
        for i, comp1 in enumerate(available_components):
            for comp2 in available_components[i+1:]:
                comp1_data = df[comp1].dropna()
                comp2_data = df[comp2].dropna()
                
                if len(comp1_data) >= self.min_sample_size and len(comp2_data) >= self.min_sample_size:
                    comparison = self._perform_comparison(
                        comp1_data, comp2_data, comp1, comp2, 'component'
                    )
                    if comparison:
                        comparisons.append(comparison)
        
        return comparisons
    
    def _perform_comparison(self, 
                          data1: pd.Series, 
                          data2: pd.Series,
                          name1: str,
                          name2: str,
                          comparison_type: str) -> Optional[QualityComparison]:
        """Perform statistical comparison between two groups."""
        try:
            # Calculate basic statistics
            group1_stats = {
                'mean': float(data1.mean()),
                'median': float(data1.median()),
                'std': float(data1.std()),
                'count': len(data1),
                'min': float(data1.min()),
                'max': float(data1.max())
            }
            
            group2_stats = {
                'mean': float(data2.mean()),
                'median': float(data2.median()),
                'std': float(data2.std()),
                'count': len(data2),
                'min': float(data2.min()),
                'max': float(data2.max())
            }
            
            # Perform statistical tests
            statistical_tests = []
            
            # Mann-Whitney U test (non-parametric)
            try:
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                statistical_tests.append({
                    'test_name': 'Mann-Whitney U',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.significance_threshold,
                    'interpretation': f"{'Significant' if p_value < self.significance_threshold else 'Non-significant'} difference (p={p_value:.4f})"
                })
            except Exception as e:
                logger.warning(f"Mann-Whitney U test failed: {e}")
            
            # Independent t-test (parametric)
            try:
                statistic, p_value = stats.ttest_ind(data1, data2)
                statistical_tests.append({
                    'test_name': 'Independent t-test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.significance_threshold,
                    'interpretation': f"{'Significant' if p_value < self.significance_threshold else 'Non-significant'} difference (p={p_value:.4f})"
                })
            except Exception as e:
                logger.warning(f"Independent t-test failed: {e}")
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) / 
                               (len(data1) + len(data2) - 2))
            
            if pooled_std > 0:
                effect_size = abs(data1.mean() - data2.mean()) / pooled_std
            else:
                effect_size = 0.0
            
            # Determine practical significance
            practical_significance = effect_size >= self.effect_size_thresholds['small']
            
            # Calculate confidence interval for difference in means
            diff_mean = data1.mean() - data2.mean()
            se_diff = np.sqrt(data1.var()/len(data1) + data2.var()/len(data2))
            t_critical = stats.t.ppf(0.975, len(data1) + len(data2) - 2)
            ci_lower = diff_mean - t_critical * se_diff
            ci_upper = diff_mean + t_critical * se_diff
            
            # Generate recommendations
            recommendations = self._generate_comparison_recommendations(
                group1_stats, group2_stats, statistical_tests, effect_size, comparison_type
            )
            
            return QualityComparison(
                comparison_type=comparison_type,
                group1_name=name1,
                group2_name=name2,
                group1_stats=group1_stats,
                group2_stats=group2_stats,
                statistical_tests=statistical_tests,
                effect_size=float(effect_size),
                practical_significance=practical_significance,
                confidence_interval=(float(ci_lower), float(ci_upper)),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Error performing comparison: {e}")
            return None
    
    def perform_benchmark_analysis(self, df: pd.DataFrame) -> List[BenchmarkAnalysis]:
        """Perform benchmark analysis against industry standards."""
        if df.empty:
            return []
        
        benchmark_analyses = []
        
        # Tier benchmarking
        for tier in df['tier'].unique():
            tier_data = df[df['tier'] == tier]
            if len(tier_data) >= self.min_sample_size:
                for component in self.quality_components:
                    if component in tier_data.columns:
                        analysis = self._perform_benchmark_analysis(
                            tier_data[component].dropna(),
                            tier,
                            f"industry_standard_{component}",
                            self.industry_benchmarks.get(component, 0.75),
                            'tier_benchmark'
                        )
                        if analysis:
                            benchmark_analyses.append(analysis)
        
        # Dataset benchmarking
        for dataset in df['dataset_name'].unique():
            dataset_data = df[df['dataset_name'] == dataset]
            if len(dataset_data) >= self.min_sample_size:
                overall_quality = dataset_data['overall_quality'].dropna()
                if len(overall_quality) > 0:
                    analysis = self._perform_benchmark_analysis(
                        overall_quality,
                        dataset,
                        "industry_standard_overall",
                        self.industry_benchmarks['overall_quality'],
                        'dataset_benchmark'
                    )
                    if analysis:
                        benchmark_analyses.append(analysis)
        
        return benchmark_analyses
    
    def _perform_benchmark_analysis(self,
                                  data: pd.Series,
                                  target_name: str,
                                  benchmark_name: str,
                                  benchmark_value: float,
                                  benchmark_type: str) -> Optional[BenchmarkAnalysis]:
        """Perform benchmark analysis for a specific group."""
        try:
            if data.empty:
                return None
            
            current_performance = data.mean()
            performance_gap = current_performance - benchmark_value
            
            # Calculate percentile ranking (simplified)
            percentile_ranking = (data > benchmark_value).mean() * 100
            
            # Calculate improvement potential
            max_possible = 1.0  # Assuming quality scores are 0-1
            improvement_potential = max_possible - current_performance
            
            # Benchmark metrics
            benchmark_metrics = {
                'current_performance': float(current_performance),
                'benchmark_value': float(benchmark_value),
                'performance_gap': float(performance_gap),
                'percentile_ranking': float(percentile_ranking),
                'improvement_potential': float(improvement_potential),
                'sample_size': len(data)
            }
            
            # Generate recommendations
            recommendations = self._generate_benchmark_recommendations(
                current_performance, benchmark_value, performance_gap, benchmark_type
            )
            
            return BenchmarkAnalysis(
                benchmark_type=benchmark_type,
                target_group=target_name,
                benchmark_group=benchmark_name,
                performance_gap=float(performance_gap),
                percentile_ranking=float(percentile_ranking),
                improvement_potential=float(improvement_potential),
                benchmark_metrics=benchmark_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Error performing benchmark analysis: {e}")
            return None
    
    def calculate_performance_rankings(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate performance rankings across different dimensions."""
        rankings = {}
        
        if df.empty:
            return rankings
        
        # Tier rankings
        tier_performance = []
        for tier in df['tier'].unique():
            tier_data = df[df['tier'] == tier]['overall_quality'].dropna()
            if len(tier_data) >= self.min_sample_size:
                tier_performance.append({
                    'name': tier,
                    'mean_quality': float(tier_data.mean()),
                    'median_quality': float(tier_data.median()),
                    'std_quality': float(tier_data.std()),
                    'sample_size': len(tier_data)
                })
        
        tier_performance.sort(key=lambda x: x['mean_quality'], reverse=True)
        rankings['tiers'] = tier_performance
        
        # Dataset rankings
        dataset_performance = []
        for dataset in df['dataset_name'].unique():
            dataset_data = df[df['dataset_name'] == dataset]['overall_quality'].dropna()
            if len(dataset_data) >= self.min_sample_size:
                dataset_performance.append({
                    'name': dataset,
                    'mean_quality': float(dataset_data.mean()),
                    'median_quality': float(dataset_data.median()),
                    'std_quality': float(dataset_data.std()),
                    'sample_size': len(dataset_data)
                })
        
        dataset_performance.sort(key=lambda x: x['mean_quality'], reverse=True)
        rankings['datasets'] = dataset_performance
        
        # Component rankings
        component_performance = []
        for component in self.quality_components:
            if component in df.columns:
                component_data = df[component].dropna()
                if len(component_data) >= self.min_sample_size:
                    component_performance.append({
                        'name': component,
                        'mean_quality': float(component_data.mean()),
                        'median_quality': float(component_data.median()),
                        'std_quality': float(component_data.std()),
                        'sample_size': len(component_data)
                    })
        
        component_performance.sort(key=lambda x: x['mean_quality'], reverse=True)
        rankings['components'] = component_performance
        
        return rankings
    
    def _generate_comparison_recommendations(self,
                                           group1_stats: Dict[str, float],
                                           group2_stats: Dict[str, float],
                                           statistical_tests: List[Dict[str, Any]],
                                           effect_size: float,
                                           comparison_type: str) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Performance difference
        mean_diff = group1_stats['mean'] - group2_stats['mean']
        if abs(mean_diff) > 0.05:  # 5% threshold
            better_group = group1_stats if mean_diff > 0 else group2_stats
            worse_group = group2_stats if mean_diff > 0 else group1_stats
            better_name = "Group 1" if mean_diff > 0 else "Group 2"
            worse_name = "Group 2" if mean_diff > 0 else "Group 1"
            
            recommendations.append(f"📊 {better_name} outperforms {worse_name} by {abs(mean_diff):.3f} points")
        
        # Statistical significance
        significant_tests = [test for test in statistical_tests if test.get('significant', False)]
        if significant_tests:
            recommendations.append("📈 Difference is statistically significant - results are reliable")
        else:
            recommendations.append("⚠️ Difference lacks statistical significance - collect more data")
        
        # Effect size interpretation
        if effect_size >= self.effect_size_thresholds['large']:
            recommendations.append(f"💪 Large effect size ({effect_size:.3f}) indicates substantial practical difference")
        elif effect_size >= self.effect_size_thresholds['medium']:
            recommendations.append(f"📊 Medium effect size ({effect_size:.3f}) indicates moderate practical difference")
        elif effect_size >= self.effect_size_thresholds['small']:
            recommendations.append(f"📈 Small effect size ({effect_size:.3f}) indicates limited practical difference")
        else:
            recommendations.append(f"📊 Negligible effect size ({effect_size:.3f}) - difference may not be meaningful")
        
        # Comparison-specific recommendations
        if comparison_type == 'tier':
            if mean_diff > 0.1:
                recommendations.append("🎯 Consider applying best practices from higher-performing tier")
        elif comparison_type == 'dataset':
            if mean_diff > 0.1:
                recommendations.append("📁 Review data quality and processing methods for lower-performing dataset")
        
        return recommendations
    
    def _generate_benchmark_recommendations(self,
                                          current_performance: float,
                                          benchmark_value: float,
                                          performance_gap: float,
                                          benchmark_type: str) -> List[str]:
        """Generate recommendations based on benchmark analysis."""
        recommendations = []
        
        if performance_gap > 0.05:  # Above benchmark
            recommendations.append(f"✅ Performance exceeds benchmark by {performance_gap:.3f} points")
            recommendations.append("🏆 Consider sharing best practices with underperforming groups")
        elif performance_gap < -0.05:  # Below benchmark
            recommendations.append(f"⚠️ Performance below benchmark by {abs(performance_gap):.3f} points")
            recommendations.append("📈 Implement improvement initiatives to reach industry standards")
        else:
            recommendations.append("📊 Performance aligns with industry benchmark")
        
        # Improvement potential
        improvement_potential = 1.0 - current_performance
        if improvement_potential > 0.2:
            recommendations.append(f"🎯 Significant improvement potential: {improvement_potential:.3f} points available")
        
        return recommendations

def main():
    """Main function for testing the quality comparator."""
    comparator = QualityComparator()
    
    # Load comparison data
    df = comparator.load_comparison_data(days_back=30)
    
    if df.empty:
        print("❌ No quality data available for comparison analysis")
        return
    
    # Perform tier comparisons
    tier_comparisons = comparator.compare_tiers(df)
    
    print(f"📊 Quality Comparison Results:")
    print(f"Tier Comparisons: {len(tier_comparisons)}")
    
    if tier_comparisons:
        comparison = tier_comparisons[0]
        print(f"\nSample Comparison: {comparison.group1_name} vs {comparison.group2_name}")
        print(f"Effect Size: {comparison.effect_size:.3f}")
        print(f"Practical Significance: {comparison.practical_significance}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(comparison.recommendations, 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
