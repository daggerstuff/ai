#!/usr/bin/env python3
"""
Quality Distribution Analysis System
Analyzes quality score distributions across datasets, tiers, and time periods
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
class DistributionAnalysis:
    """Quality distribution analysis results"""
    metric: str
    category: str
    values: List[float]
    statistics: Dict[str, float]
    distribution_type: str
    outliers: List[float]
    percentiles: Dict[str, float]
    
@dataclass
class ComparisonResult:
    """Distribution comparison results"""
    metric: str
    categories: List[str]
    test_statistic: float
    p_value: float
    effect_size: float
    interpretation: str

class QualityDistributionAnalyzer:
    """Enterprise-grade quality distribution analysis system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_distributions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics we can derive from current data
        self.quality_metrics = [
            'conversation_length',  # Based on turn_count
            'content_richness',     # Based on word_count
            'character_density',    # Based on character_count
            'processing_efficiency' # Based on processing metrics
        ]
        
        # Analysis categories
        self.analysis_categories = [
            'tier',
            'dataset_source',
            'language',
            'processing_status'
        ]
        
        # Statistical parameters
        self.outlier_threshold = 2.5  # Standard deviations
        self.significance_level = 0.05
        
    def analyze_quality_distributions(self) -> Dict[str, Dict[str, DistributionAnalysis]]:
        """Analyze quality distributions across all metrics and categories"""
        print("🔍 Analyzing quality distributions across all metrics and categories...")
        
        try:
            # Get conversation data
            data = self._get_conversation_data()
            
            if not data:
                print("❌ No conversation data found")
                return {}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(data)
            
            # Analyze distributions for each metric and category
            results = {}
            
            for metric in self.quality_metrics:
                results[metric] = {}
                
                # Calculate metric values
                metric_values = self._calculate_metric_values(df, metric)
                df[f'{metric}_value'] = metric_values
                
                for category in self.analysis_categories:
                    if category in df.columns:
                        analysis = self._analyze_distribution_by_category(
                            df, f'{metric}_value', category
                        )
                        if analysis:
                            results[metric][category] = analysis
            
            print(f"✅ Analyzed distributions for {len(results)} metrics across {len(self.analysis_categories)} categories")
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing quality distributions: {e}")
            return {}
    
    def _get_conversation_data(self) -> List[Dict]:
        """Get conversation data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                dataset_source,
                tier,
                turn_count,
                word_count,
                character_count,
                language,
                processing_status,
                created_at
            FROM conversations 
            WHERE turn_count IS NOT NULL 
            AND word_count IS NOT NULL
            LIMIT 10000
            """
            
            cursor = conn.execute(query)
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
            print(f"❌ Error getting conversation data: {e}")
            return []
    
    def _calculate_metric_values(self, df: pd.DataFrame, metric: str) -> List[float]:
        """Calculate metric values for all records"""
        try:
            if metric == 'conversation_length':
                return df['turn_count'].fillna(0).tolist()
            
            elif metric == 'content_richness':
                return df['word_count'].fillna(0).tolist()
            
            elif metric == 'character_density':
                # Characters per word
                word_counts = df['word_count'].fillna(1)
                char_counts = df['character_count'].fillna(0)
                return (char_counts / word_counts).fillna(0).tolist()
            
            elif metric == 'processing_efficiency':
                # Simple efficiency score based on processing status
                return [1.0 if status == 'processed' else 0.0 
                       for status in df['processing_status'].fillna('unknown')]
            
            return [0.0] * len(df)
            
        except Exception as e:
            print(f"❌ Error calculating {metric}: {e}")
            return [0.0] * len(df)
    
    def _analyze_distribution_by_category(self, df: pd.DataFrame, 
                                        metric_col: str, category_col: str) -> Dict[str, DistributionAnalysis]:
        """Analyze distribution for each category value"""
        try:
            results = {}
            
            # Group by category
            grouped = df.groupby(category_col)
            
            for category_value, group in grouped:
                if len(group) < 10:  # Skip small groups
                    continue
                
                values = group[metric_col].dropna().tolist()
                if not values:
                    continue
                
                # Calculate statistics
                statistics = self._calculate_distribution_statistics(values)
                
                # Detect distribution type
                distribution_type = self._detect_distribution_type(values)
                
                # Find outliers
                outliers = self._find_outliers(values)
                
                # Calculate percentiles
                percentiles = self._calculate_percentiles(values)
                
                analysis = DistributionAnalysis(
                    metric=metric_col,
                    category=str(category_value),
                    values=values,
                    statistics=statistics,
                    distribution_type=distribution_type,
                    outliers=outliers,
                    percentiles=percentiles
                )
                
                results[str(category_value)] = analysis
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing distribution for {category_col}: {e}")
            return {}
    
    def _calculate_distribution_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive distribution statistics"""
        try:
            values_array = np.array(values)
            
            return {
                'count': len(values),
                'mean': np.mean(values_array),
                'median': np.median(values_array),
                'std': np.std(values_array),
                'variance': np.var(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'range': np.max(values_array) - np.min(values_array),
                'skewness': stats.skew(values_array),
                'kurtosis': stats.kurtosis(values_array),
                'coefficient_of_variation': np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            return {}
    
    def _detect_distribution_type(self, values: List[float]) -> str:
        """Detect the type of distribution"""
        try:
            values_array = np.array(values)
            
            # Test for normality
            shapiro_stat, shapiro_p = stats.shapiro(values_array[:5000])  # Limit for shapiro test
            
            if shapiro_p > 0.05:
                return "normal"
            
            # Test for exponential
            try:
                ks_stat, ks_p = stats.kstest(values_array, 'expon')
                if ks_p > 0.05:
                    return "exponential"
            except:
                pass
            
            # Test for uniform
            try:
                ks_stat, ks_p = stats.kstest(values_array, 'uniform')
                if ks_p > 0.05:
                    return "uniform"
            except:
                pass
            
            # Check skewness for distribution type
            skewness = stats.skew(values_array)
            if abs(skewness) < 0.5:
                return "approximately_normal"
            elif skewness > 1:
                return "right_skewed"
            elif skewness < -1:
                return "left_skewed"
            else:
                return "moderately_skewed"
            
        except Exception as e:
            print(f"❌ Error detecting distribution type: {e}")
            return "unknown"
    
    def _find_outliers(self, values: List[float]) -> List[float]:
        """Find outliers using IQR method and Z-score"""
        try:
            values_array = np.array(values)
            
            # IQR method
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = values_array[(values_array < lower_bound) | (values_array > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values_array))
            z_outliers = values_array[z_scores > self.outlier_threshold]
            
            # Combine both methods
            all_outliers = np.unique(np.concatenate([iqr_outliers, z_outliers]))
            
            return all_outliers.tolist()
            
        except Exception as e:
            print(f"❌ Error finding outliers: {e}")
            return []
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate key percentiles"""
        try:
            values_array = np.array(values)
            
            return {
                'p5': np.percentile(values_array, 5),
                'p10': np.percentile(values_array, 10),
                'p25': np.percentile(values_array, 25),
                'p50': np.percentile(values_array, 50),
                'p75': np.percentile(values_array, 75),
                'p90': np.percentile(values_array, 90),
                'p95': np.percentile(values_array, 95),
                'p99': np.percentile(values_array, 99)
            }
            
        except Exception as e:
            print(f"❌ Error calculating percentiles: {e}")
            return {}
    
    def compare_distributions(self, distributions: Dict[str, Dict[str, DistributionAnalysis]]) -> Dict[str, List[ComparisonResult]]:
        """Compare distributions between categories"""
        print("📊 Comparing distributions between categories...")
        
        try:
            comparisons = {}
            
            for metric, metric_distributions in distributions.items():
                comparisons[metric] = []
                
                # Get all category values for this metric
                categories = list(metric_distributions.keys())
                
                if len(categories) < 2:
                    continue
                
                # Compare each pair of categories
                for i in range(len(categories)):
                    for j in range(i + 1, len(categories)):
                        cat1, cat2 = categories[i], categories[j]
                        
                        values1 = metric_distributions[cat1].values
                        values2 = metric_distributions[cat2].values
                        
                        comparison = self._compare_two_distributions(
                            values1, values2, [cat1, cat2], metric
                        )
                        
                        if comparison:
                            comparisons[metric].append(comparison)
            
            print(f"✅ Completed distribution comparisons for {len(comparisons)} metrics")
            return comparisons
            
        except Exception as e:
            print(f"❌ Error comparing distributions: {e}")
            return {}
    
    def _compare_two_distributions(self, values1: List[float], values2: List[float], 
                                 categories: List[str], metric: str) -> Optional[ComparisonResult]:
        """Compare two distributions using statistical tests"""
        try:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            
            # Calculate effect size (Cohen's d)
            mean1, mean2 = np.mean(values1), np.mean(values2)
            std1, std2 = np.std(values1), np.std(values2)
            pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2) / 
                               (len(values1) + len(values2) - 2))
            
            effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Interpret results
            if p_value < 0.001:
                significance = "highly significant"
            elif p_value < 0.01:
                significance = "very significant"
            elif p_value < 0.05:
                significance = "significant"
            else:
                significance = "not significant"
            
            if effect_size < 0.2:
                effect_interpretation = "negligible"
            elif effect_size < 0.5:
                effect_interpretation = "small"
            elif effect_size < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            interpretation = f"{significance} difference with {effect_interpretation} effect size"
            
            return ComparisonResult(
                metric=metric,
                categories=categories,
                test_statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                interpretation=interpretation
            )
            
        except Exception as e:
            print(f"❌ Error comparing distributions: {e}")
            return None
    
    def create_distribution_visualizations(self, distributions: Dict[str, Dict[str, DistributionAnalysis]]) -> Dict[str, str]:
        """Create comprehensive distribution visualizations"""
        print(f"📈 Creating distribution visualizations...")
        
        viz_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            for metric, metric_distributions in distributions.items():
                if not metric_distributions:
                    continue
                
                # Create figure with subplots
                n_categories = len(metric_distributions)
                n_cols = min(3, n_categories)
                n_rows = (n_categories + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                fig.suptitle(f'{metric.replace("_", " ").title()} Distribution Analysis', 
                           fontsize=16, fontweight='bold')
                
                for i, (category, analysis) in enumerate(metric_distributions.items()):
                    if i >= len(axes):
                        break
                    
                    ax = axes[i]
                    
                    # Create histogram with KDE
                    ax.hist(analysis.values, bins=30, alpha=0.7, density=True, edgecolor='black')
                    
                    # Add KDE curve
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(analysis.values)
                        x_range = np.linspace(min(analysis.values), max(analysis.values), 100)
                        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    except:
                        pass
                    
                    # Add statistics
                    mean_val = analysis.statistics.get('mean', 0)
                    median_val = analysis.statistics.get('median', 0)
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='blue', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                    
                    ax.set_title(f'{category} ({analysis.distribution_type})', fontsize=12)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics text
                    stats_text = f"""
                    N: {analysis.statistics.get('count', 0)}
                    Std: {analysis.statistics.get('std', 0):.2f}
                    Skew: {analysis.statistics.get('skewness', 0):.2f}
                    Outliers: {len(analysis.outliers)}
                    """
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
                # Remove empty subplots
                for i in range(len(metric_distributions), len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                
                # Save plot
                filename = f"distribution_{metric}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_files[metric] = str(filepath)
            
            print(f"✅ Created {len(viz_files)} distribution visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_distribution_report(self, distributions: Dict[str, Dict[str, DistributionAnalysis]], 
                                 comparisons: Dict[str, List[ComparisonResult]],
                                 visualizations: Dict[str, str]) -> str:
        """Export comprehensive distribution analysis report"""
        print("📄 Exporting distribution analysis report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_distribution_report_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'total_metrics': len(distributions),
                    'total_categories': sum(len(cats) for cats in distributions.values())
                },
                'distribution_analysis': {
                    metric: {
                        category: {
                            'statistics': analysis.statistics,
                            'distribution_type': analysis.distribution_type,
                            'outlier_count': len(analysis.outliers),
                            'percentiles': analysis.percentiles,
                            'sample_size': len(analysis.values)
                        }
                        for category, analysis in metric_distributions.items()
                    }
                    for metric, metric_distributions in distributions.items()
                },
                'distribution_comparisons': {
                    metric: [
                        {
                            'categories': comp.categories,
                            'test_statistic': comp.test_statistic,
                            'p_value': comp.p_value,
                            'effect_size': comp.effect_size,
                            'interpretation': comp.interpretation
                        }
                        for comp in metric_comparisons
                    ]
                    for metric, metric_comparisons in comparisons.items()
                },
                'visualizations': visualizations
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported distribution report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting distribution report: {e}")
            return ""

def main():
    """Main execution function"""
    print("📊 Quality Distribution Analysis System")
    print("=" * 45)
    
    # Initialize analyzer
    analyzer = QualityDistributionAnalyzer()
    
    # Analyze distributions
    distributions = analyzer.analyze_quality_distributions()
    
    if not distributions:
        print("❌ No distribution data found")
        return
    
    # Compare distributions
    comparisons = analyzer.compare_distributions(distributions)
    
    # Create visualizations
    visualizations = analyzer.create_distribution_visualizations(distributions)
    
    # Export report
    report_file = analyzer.export_distribution_report(distributions, comparisons, visualizations)
    
    # Display summary
    print(f"\n✅ Distribution Analysis Complete")
    print(f"   - Metrics analyzed: {len(distributions)}")
    print(f"   - Total distributions: {sum(len(cats) for cats in distributions.values())}")
    print(f"   - Comparisons performed: {sum(len(comps) for comps in comparisons.values())}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show key findings
    print("\n🔍 Key Findings:")
    for metric, metric_distributions in distributions.items():
        print(f"\n   {metric.replace('_', ' ').title()}:")
        for category, analysis in metric_distributions.items():
            mean_val = analysis.statistics.get('mean', 0)
            std_val = analysis.statistics.get('std', 0)
            outliers = len(analysis.outliers)
            print(f"     {category}: μ={mean_val:.2f}, σ={std_val:.2f}, outliers={outliers}")

if __name__ == "__main__":
    main()
