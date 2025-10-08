#!/usr/bin/env python3
"""
Quality Distribution Analysis Launcher (Task 5.6.2.3)

Enterprise-grade launcher for the quality distribution analysis system
with comprehensive setup, validation, and execution capabilities.
"""

import sys
import os
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityDistributionAnalysisLauncher:
    """
    Enterprise-grade launcher for quality distribution analysis system.
    
    Handles setup, validation, analysis execution, and report generation
    with comprehensive error handling and monitoring.
    """
    
    def __init__(self):
        """Initialize the distribution analysis launcher."""
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = Path(__file__).parent
        self.db_path = self.project_root / "database" / "conversations.db"
        self.reports_dir = self.monitoring_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Required dependencies
        self.required_packages = [
            'pandas',
            'numpy',
            'sqlite3',
            'scipy',
            'plotly',
            'seaborn',
            'matplotlib',
            'jinja2'
        ]
        
        logger.info("🚀 Quality Distribution Analysis Launcher initialized")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("🔍 Checking dependencies...")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                elif package == 'pandas':
                    import pandas
                elif package == 'numpy':
                    import numpy
                elif package == 'scipy':
                    import scipy
                elif package == 'plotly':
                    import plotly
                elif package == 'seaborn':
                    import seaborn
                elif package == 'matplotlib':
                    import matplotlib
                elif package == 'jinja2':
                    import jinja2
                
                logger.info(f"  ✅ {package}: Available")
                
            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: Missing")
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def validate_database(self) -> bool:
        """Validate that the database exists and has sufficient quality data."""
        logger.info("🔍 Validating database...")
        
        if not self.db_path.exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.info("💡 Run the dataset processing pipeline to create the database")
            return False
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['conversations', 'quality_metrics']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.error(f"❌ Missing tables: {', '.join(missing_tables)}")
                conn.close()
                return False
            
            # Check if sufficient quality data exists for distribution analysis
            cursor.execute("""
                SELECT COUNT(*) as total_conversations,
                       COUNT(DISTINCT c.tier) as tier_count,
                       COUNT(DISTINCT c.dataset_name) as dataset_count,
                       MIN(c.created_at) as earliest_date,
                       MAX(c.created_at) as latest_date
                FROM conversations c
                LEFT JOIN quality_metrics q ON c.id = q.conversation_id
                WHERE q.overall_quality IS NOT NULL
            """)
            
            result = cursor.fetchone()
            total_conversations, tier_count, dataset_count, earliest_date, latest_date = result
            
            if total_conversations < 30:  # Minimum for meaningful distribution analysis
                logger.error(f"❌ Insufficient data for distribution analysis: only {total_conversations} conversations")
                logger.info("💡 Need at least 30 conversations with quality data for distribution analysis")
                conn.close()
                return False
            
            logger.info(f"✅ Database validated: {total_conversations} conversations, {tier_count} tiers, {dataset_count} datasets")
            logger.info(f"📅 Data range: {earliest_date} to {latest_date}")
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False
    
    def validate_distribution_analysis_files(self) -> bool:
        """Validate that distribution analysis files exist and are accessible."""
        logger.info("🔍 Validating distribution analysis files...")
        
        required_files = [
            'quality_distribution_analyzer.py',
            'quality_distribution_comparator.py',
            'quality_distribution_reporter.py'
        ]
        
        for filename in required_files:
            filepath = self.monitoring_dir / filename
            
            if not filepath.exists():
                logger.error(f"❌ Required file not found: {filepath}")
                return False
            
            # Check if file has main classes
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                if filename == 'quality_distribution_analyzer.py':
                    if 'QualityDistributionAnalyzer' not in content:
                        logger.error(f"❌ QualityDistributionAnalyzer class not found in {filename}")
                        return False
                elif filename == 'quality_distribution_comparator.py':
                    if 'QualityDistributionComparator' not in content:
                        logger.error(f"❌ QualityDistributionComparator class not found in {filename}")
                        return False
                elif filename == 'quality_distribution_reporter.py':
                    if 'QualityDistributionReporter' not in content:
                        logger.error(f"❌ QualityDistributionReporter class not found in {filename}")
                        return False
                
            except Exception as e:
                logger.error(f"❌ Error reading {filename}: {e}")
                return False
        
        logger.info("✅ Distribution analysis files validated")
        return True
    
    def run_distribution_analysis(self, 
                                 days_back: int = 90,
                                 include_visualizations: bool = True,
                                 include_comparisons: bool = True,
                                 output_format: str = 'both',
                                 tier_filter: list = None,
                                 dataset_filter: list = None) -> dict:
        """Run comprehensive distribution analysis."""
        logger.info(f"📊 Running distribution analysis ({days_back} days back)...")
        
        try:
            # Import distribution analysis components
            sys.path.append(str(self.monitoring_dir))
            from quality_distribution_analyzer import QualityDistributionAnalyzer
            from quality_distribution_comparator import QualityDistributionComparator
            from quality_distribution_reporter import QualityDistributionReporter
            
            # Initialize components
            analyzer = QualityDistributionAnalyzer(db_path=str(self.db_path))
            comparator = QualityDistributionComparator(analyzer)
            reporter = QualityDistributionReporter(output_dir=str(self.reports_dir))
            
            # Load quality data
            logger.info("📊 Loading quality data...")
            df = analyzer.load_quality_data(
                days_back=days_back,
                tier_filter=tier_filter,
                dataset_filter=dataset_filter
            )
            
            if df.empty:
                logger.error("❌ No quality data available for analysis")
                return {'success': False, 'error': 'No quality data available'}
            
            logger.info(f"✅ Loaded {len(df)} conversations for analysis")
            
            # Analyze overall distribution
            logger.info("📈 Analyzing overall quality distribution...")
            overall_analysis = analyzer.analyze_quality_distribution(
                df['overall_quality'], 'overall_quality'
            )
            
            # Generate comprehensive report
            logger.info("📋 Generating comprehensive distribution report...")
            report = reporter.generate_comprehensive_report(
                days_back=days_back,
                include_visualizations=include_visualizations,
                include_comparisons=include_comparisons
            )
            
            # Save reports
            saved_files = []
            
            if output_format in ['json', 'both']:
                json_path = reporter.save_report(report, format='json')
                saved_files.append(json_path)
                logger.info(f"📄 JSON report saved: {json_path}")
            
            if output_format in ['html', 'both']:
                html_path = reporter.save_report(report, format='html')
                saved_files.append(html_path)
                logger.info(f"📄 HTML report saved: {html_path}")
            
            # Generate visualizations if requested
            visualization_files = []
            if include_visualizations:
                logger.info("📊 Generating distribution visualizations...")
                try:
                    visualizations = reporter.create_distribution_visualizations(report)
                    
                    # Save visualizations as HTML files
                    for viz_name, fig in visualizations.items():
                        viz_filename = f"distribution_visualization_{viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        viz_path = self.reports_dir / viz_filename
                        fig.write_html(str(viz_path))
                        visualization_files.append(str(viz_path))
                        logger.info(f"📊 Visualization saved: {viz_path}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not generate visualizations: {e}")
            
            # Create summary
            summary = {
                'success': True,
                'analysis_period': f"{days_back}_days",
                'total_conversations': overall_analysis.sample_size,
                'distribution_type': overall_analysis.distribution_type,
                'mean_quality': overall_analysis.statistics.mean,
                'median_quality': overall_analysis.statistics.median,
                'std_dev': overall_analysis.statistics.std_dev,
                'skewness': overall_analysis.statistics.skewness,
                'kurtosis': overall_analysis.statistics.kurtosis,
                'outliers_detected': len(overall_analysis.outliers),
                'normality_tests': len(overall_analysis.normality_tests),
                'report_files': saved_files,
                'visualization_files': visualization_files,
                'executive_summary': report.executive_summary,
                'action_items': report.action_items,
                'comparative_analyses': {
                    name: {
                        'groups': len(analysis.groups),
                        'statistical_tests': len(analysis.statistical_tests),
                        'significant_differences': any(test.get('significant', False) for test in analysis.statistical_tests)
                    }
                    for name, analysis in report.comparative_analyses.items()
                }
            }
            
            logger.info("✅ Distribution analysis completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error during distribution analysis: {e}")
            return {'success': False, 'error': str(e)}
    
    def display_analysis_summary(self, summary: dict):
        """Display analysis summary to console."""
        if not summary.get('success', False):
            print(f"❌ Analysis failed: {summary.get('error', 'Unknown error')}")
            return
        
        print(f"\n📊 Quality Distribution Analysis Summary")
        print(f"{'='*50}")
        print(f"Analysis Period: {summary['analysis_period']}")
        print(f"Total Conversations: {summary['total_conversations']:,}")
        print(f"Distribution Type: {summary['distribution_type'].replace('_', ' ').title()}")
        print(f"Mean Quality: {summary['mean_quality']:.3f}")
        print(f"Median Quality: {summary['median_quality']:.3f}")
        print(f"Standard Deviation: {summary['std_dev']:.3f}")
        print(f"Skewness: {summary['skewness']:.3f}")
        print(f"Kurtosis: {summary['kurtosis']:.3f}")
        print(f"Outliers Detected: {summary['outliers_detected']}")
        print(f"Normality Tests: {summary['normality_tests']}")
        
        print(f"\n📄 Generated Files:")
        for file_path in summary['report_files']:
            print(f"  📋 {file_path}")
        
        if summary['visualization_files']:
            print(f"\n📊 Visualization Files:")
            for file_path in summary['visualization_files']:
                print(f"  📈 {file_path}")
        
        print(f"\n🔍 Comparative Analyses:")
        for name, comp_summary in summary['comparative_analyses'].items():
            status = "✅ Significant differences" if comp_summary['significant_differences'] else "📊 No significant differences"
            print(f"  {name.title()}: {comp_summary['groups']} groups, {comp_summary['statistical_tests']} tests - {status}")
        
        print(f"\n💡 Executive Summary:")
        for i, item in enumerate(summary['executive_summary'][:5], 1):
            print(f"  {i}. {item}")
        
        print(f"\n🎯 Action Items:")
        for i, item in enumerate(summary['action_items'][:5], 1):
            print(f"  {i}. {item}")
    
    def launch(self, 
              days_back: int = 90,
              include_visualizations: bool = True,
              include_comparisons: bool = True,
              output_format: str = 'both',
              tier_filter: list = None,
              dataset_filter: list = None,
              skip_validation: bool = False) -> bool:
        """
        Complete launch sequence for quality distribution analysis.
        
        Args:
            days_back: Number of days to analyze
            include_visualizations: Generate distribution visualizations
            include_comparisons: Include comparative analysis
            output_format: Output format ('json', 'html', 'both')
            tier_filter: List of tiers to include
            dataset_filter: List of datasets to include
            skip_validation: Skip pre-analysis validation
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        logger.info("🎯 Starting Quality Distribution Analysis Launch Sequence...")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Launch aborted: Missing dependencies")
            return False
        
        # Step 2: Validate database (unless skipped)
        if not skip_validation:
            if not self.validate_database():
                logger.error("❌ Launch aborted: Database validation failed")
                return False
        
        # Step 3: Validate distribution analysis files
        if not self.validate_distribution_analysis_files():
            logger.error("❌ Launch aborted: Distribution analysis files validation failed")
            return False
        
        # Step 4: Run distribution analysis
        summary = self.run_distribution_analysis(
            days_back=days_back,
            include_visualizations=include_visualizations,
            include_comparisons=include_comparisons,
            output_format=output_format,
            tier_filter=tier_filter,
            dataset_filter=dataset_filter
        )
        
        # Step 5: Display results
        self.display_analysis_summary(summary)
        
        if summary.get('success', False):
            logger.info("🎉 Quality Distribution Analysis completed successfully!")
            return True
        else:
            logger.error("❌ Quality Distribution Analysis failed")
            return False

def main():
    """Main function to launch quality distribution analysis."""
    parser = argparse.ArgumentParser(description='Launch Quality Distribution Analysis System')
    parser.add_argument('--days-back', type=int, default=90, help='Number of days to analyze (default: 90)')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip distribution visualizations')
    parser.add_argument('--no-comparisons', action='store_true', help='Skip comparative analysis')
    parser.add_argument('--format', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--tiers', nargs='+', help='Filter by specific tiers')
    parser.add_argument('--datasets', nargs='+', help='Filter by specific datasets')
    parser.add_argument('--skip-validation', action='store_true', help='Skip pre-analysis validation')
    
    args = parser.parse_args()
    
    launcher = QualityDistributionAnalysisLauncher()
    success = launcher.launch(
        days_back=args.days_back,
        include_visualizations=not args.no_visualizations,
        include_comparisons=not args.no_comparisons,
        output_format=args.format,
        tier_filter=args.tiers,
        dataset_filter=args.datasets,
        skip_validation=args.skip_validation
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
